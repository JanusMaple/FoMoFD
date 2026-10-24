#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.npz_to_le_v21 import (
    _resolve_task_dirs,
    _load_frozen_vfmf_vae,
    _state_quat_to_rot6d,
)

from models.flow_matching_wm.score import (
    load_merged_config,
    load_model_from_checkpoint,
    get_device,
    get_amp_dtype,
    _compute_reactive_scores_from_eps_hat,
    _predict_endpoint_latent_center,
    _compute_predictive_scores_from_prediction,
)


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timer_start(device):
    cuda_sync(device)
    return time.perf_counter()


def timer_end(device, t0):
    cuda_sync(device)
    return time.perf_counter() - t0


def get_attr_any(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


@torch.inference_mode()
def encode_fix_wrist_one_frame(
    image_fix: np.ndarray,
    image_wrist: np.ndarray,
    dinov2_model: torch.nn.Module,
    vae_fix: torch.nn.Module,
    vae_wrist: torch.nn.Module,
    device: torch.device,
    deterministic: bool = True,
    layers=(2, 5, 8, 11),
):
    """
    Returns:
      z_fix:   [H, W, C]
      z_wrist: [H, W, C]
    """
    imgs = np.stack([image_fix, image_wrist], axis=0)  # [2,H,W,3]

    x = torch.from_numpy(imgs).float().to(device) / 255.0
    x = x.permute(0, 3, 1, 2)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std

    patch_list = dinov2_model.get_intermediate_layers(
        x,
        n=list(layers),
        reshape=False,
    )

    feat = torch.cat(patch_list, dim=-1)      # [2,256,3072]
    feat = F.normalize(feat, p=2, dim=2)

    hp = vae_fix.config.data.patch_grid_height
    wp = vae_fix.config.data.patch_grid_width

    feat = feat.view(2, hp, wp, -1)

    z_fix = vae_fix.encode(feat[0:1], deterministic=deterministic)
    z_wrist = vae_wrist.encode(feat[1:2], deterministic=deterministic)

    z_fix = z_fix.view(hp, wp, -1).float()
    z_wrist = z_wrist.view(hp, wp, -1).float()

    return z_fix, z_wrist


@torch.no_grad()
def score_one_window(
    *,
    mode: str,
    model,
    fix_hist,
    wrist_hist,
    state_hist,
    action_chunk,
    future_fix,
    future_wrist,
    future_state,
    device,
    use_amp: bool,
    amp_dtype: torch.dtype,
):
    autocast_enabled = device.type == "cuda" and use_amp

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast_enabled):
        if mode == "reactive":
            out = model.compute_reactive_nonconformity_scores(
                fix_hist=fix_hist,
                wrist_hist=wrist_hist,
                state_hist=state_hist,
                action_chunk=action_chunk,
                future_fix=future_fix,
                future_wrist=future_wrist,
                future_state=future_state,
            )

            score_parts = _compute_reactive_scores_from_eps_hat(
                out["eps_hat"],
                score_kind=model.config.fm_head.score_kind,
                base_std=model.config.fm_head.base_std,
            )

            score = score_parts["score_union"]

        elif mode == "predictive":
            pred_future = _predict_endpoint_latent_center(
                model,
                fix_hist=fix_hist,
                wrist_hist=wrist_hist,
                state_hist=state_hist,
                action_chunk=action_chunk,
            )

            x_future = torch.stack([future_fix, future_wrist], dim=1)
            score_parts = _compute_predictive_scores_from_prediction(
                pred_future,
                x_future,
            )

            score = score_parts["score_union"]

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    return score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae_ckpt_fix", required=True)
    parser.add_argument("--vae_ckpt_wrist", required=True)

    parser.add_argument("--mode", choices=["reactive", "predictive"], default="reactive")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_json", default=None)

    parser.add_argument("--history_len", type=int, default=None)
    parser.add_argument("--future_offset", type=int, default=None)
    parser.add_argument("--action_chunk_len", type=int, default=None)

    parser.add_argument("--state_key", default="states")
    parser.add_argument("--including_gripper", action="store_true")
    parser.add_argument("--quat_order", default="wxyz", choices=["wxyz", "xyzw"])

    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=500)
    parser.add_argument("--vae_deterministic", action="store_true")
    parser.add_argument("--no_shift_actions_to_next_obs", action="store_true")

    args = parser.parse_args()

    shift_actions_to_next_obs = not args.no_shift_actions_to_next_obs

    # --- Load config/model ---
    config, train_config_path, _, _, _ = load_merged_config(
        score_config_path=args.config,
        checkpoint_path=args.checkpoint,
        score_config_key=None,
        train_config_key=None,
    )

    device = get_device(args.device or config.runtime.device)
    amp_dtype = get_amp_dtype(config.runtime.amp_dtype)

    model = load_model_from_checkpoint(config, args.checkpoint, device=device)
    model.eval()

    # --- Load DINOv2 + VAEs ---
    print(f"[INFO] Loading DINOv2 to {device}")
    dinov2_model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14_reg",
    ).to(device)
    dinov2_model.eval()

    print(f"[INFO] Loading FIX VAE: {args.vae_ckpt_fix}")
    vae_fix, _ = _load_frozen_vfmf_vae(args.vae_ckpt_fix, str(device))

    print(f"[INFO] Loading WRIST VAE: {args.vae_ckpt_wrist}")
    vae_wrist, _ = _load_frozen_vfmf_vae(args.vae_ckpt_wrist, str(device))

    # --- Resolve raw PRL episodes ---
    raw_dir, _ = _resolve_task_dirs(
        task=args.task,
        mode="latents_vfmf_vae_fix_wrist",
        prl=True,
    )

    episode_files = sorted(raw_dir.glob("episode_*.npz"))
    if args.max_episodes is not None:
        episode_files = episode_files[: args.max_episodes]

    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz found in {raw_dir}")

    print(f"[INFO] Raw PRL dir: {raw_dir}")
    print(f"[INFO] Episodes: {len(episode_files)}")

    # --- Infer window parameters ---
    first = np.load(episode_files[0], allow_pickle=True)
    if "policy_action_chunks" not in first.files:
        raise KeyError(
            "PRL raw file must contain policy_action_chunks for deployment-style runtime benchmarking."
        )

    policy_chunk_len = int(first["policy_action_chunks"].shape[1])

    history_len = args.history_len
    if history_len is None:
        history_len = get_attr_any(
            config.data,
            ["history_len", "num_history_frames", "obs_history_len", "context_len"],
            default=None,
        )
    if history_len is None:
        raise ValueError("Could not infer history_len. Pass --history_len explicitly.")

    action_chunk_len = args.action_chunk_len
    if action_chunk_len is None:
        action_chunk_len = get_attr_any(
            config.data,
            ["action_chunk_len", "action_horizon", "chunk_len"],
            default=policy_chunk_len,
        )

    future_offset = args.future_offset
    if future_offset is None:
        future_offset = get_attr_any(
            config.data,
            ["future_offset", "endpoint_offset", "prediction_horizon", "action_horizon"],
            default=action_chunk_len,
        )

    history_len = int(history_len)
    action_chunk_len = int(action_chunk_len)
    future_offset = int(future_offset)

    print(f"[INFO] history_len={history_len}")
    print(f"[INFO] action_chunk_len={action_chunk_len}")
    print(f"[INFO] future_offset={future_offset}")
    print(f"[INFO] shift_actions_to_next_obs={shift_actions_to_next_obs}")

    use_state = bool(get_attr_any(config.data, ["use_state"], default=True))
    use_state = use_state and args.state_key in first.files

    # --- Benchmark accumulators ---
    feature_time_s = 0.0
    score_time_s = 0.0
    total_feature_frames = 0
    total_score_windows = 0

    score_values = []

    cuda_sync(device)

    for ep_file in episode_files:
        data = np.load(ep_file, allow_pickle=True)

        images_fix = data["images_fix"]
        images_wrist = data["images_wrist"]
        policy_action_chunks = np.asarray(data["policy_action_chunks"], dtype=np.float32)

        states = None
        if use_state:
            states = np.asarray(data[args.state_key], dtype=np.float32)

        T = int(images_fix.shape[0])

        fix_cache = []
        wrist_cache = []
        state_cache = []

        for t in range(T):
            # 1) Online feature extraction for the newly arrived frame.
            t0 = timer_start(device)
            z_fix, z_wrist = encode_fix_wrist_one_frame(
                image_fix=images_fix[t],
                image_wrist=images_wrist[t],
                dinov2_model=dinov2_model,
                vae_fix=vae_fix,
                vae_wrist=vae_wrist,
                device=device,
                deterministic=args.vae_deterministic,
            )
            feature_time_s += timer_end(device, t0)

            fix_cache.append(z_fix)
            wrist_cache.append(z_wrist)
            total_feature_frames += 1

            if use_state:
                s = _state_quat_to_rot6d(
                    states[t],
                    quat_order=args.quat_order,
                    including_gripper=args.including_gripper,
                )
                state_cache.append(torch.from_numpy(s).float().to(device))

            # If current frame t is the future endpoint, score anchor = t - future_offset.
            anchor = t - future_offset
            if anchor < history_len - 1:
                continue

            action_idx = anchor + 1 if shift_actions_to_next_obs else anchor
            if action_idx < 0 or action_idx >= policy_action_chunks.shape[0]:
                continue

            hist_start = anchor - history_len + 1
            hist_end = anchor + 1

            chunk_np = policy_action_chunks[action_idx, :action_chunk_len]
            action_chunk = torch.from_numpy(chunk_np).float().unsqueeze(0).to(device)

            fix_hist = torch.stack(fix_cache[hist_start:hist_end], dim=0).unsqueeze(0)
            wrist_hist = torch.stack(wrist_cache[hist_start:hist_end], dim=0).unsqueeze(0)

            future_fix = fix_cache[t].unsqueeze(0)
            future_wrist = wrist_cache[t].unsqueeze(0)

            if use_state:
                state_hist = torch.stack(state_cache[hist_start:hist_end], dim=0).unsqueeze(0)
                future_state = state_cache[t].unsqueeze(0)
            else:
                state_hist = None
                future_state = None

            # 2) Detector scoring.
            t0 = timer_start(device)
            score = score_one_window(
                mode=args.mode,
                model=model,
                fix_hist=fix_hist,
                wrist_hist=wrist_hist,
                state_hist=state_hist,
                action_chunk=action_chunk,
                future_fix=future_fix,
                future_wrist=future_wrist,
                future_state=future_state,
                device=device,
                use_amp=config.runtime.use_amp,
                amp_dtype=amp_dtype,
            )
            score_time_s += timer_end(device, t0)

            score_values.append(float(score.detach().cpu().item()))
            total_score_windows += 1

            if args.max_windows is not None and total_score_windows >= args.max_windows:
                break

        if args.max_windows is not None and total_score_windows >= args.max_windows:
            break

    feature_ms = 1000.0 * feature_time_s / max(total_feature_frames, 1)
    score_ms = 1000.0 * score_time_s / max(total_score_windows, 1)

    # Steady-state online detector cost:
    # one new frame feature extraction + one score.
    e2e_ms = feature_ms + score_ms
    e2e_hz = 1000.0 / e2e_ms if e2e_ms > 0 else float("inf")

    summary = {
        "task": args.task,
        "mode": args.mode,
        "device": str(device),
        "checkpoint": args.checkpoint,
        "vae_ckpt_fix": args.vae_ckpt_fix,
        "vae_ckpt_wrist": args.vae_ckpt_wrist,
        "num_feature_frames": total_feature_frames,
        "num_score_windows": total_score_windows,
        "feature_time_s_total": feature_time_s,
        "score_time_s_total": score_time_s,
        "feature_ms_per_frame": feature_ms,
        "score_ms_per_window": score_ms,
        "e2e_ms_per_window": e2e_ms,
        "feature_hz_only": 1000.0 / feature_ms if feature_ms > 0 else float("inf"),
        "score_hz_only": 1000.0 / score_ms if score_ms > 0 else float("inf"),
        "e2e_hz": e2e_hz,
        "history_len": history_len,
        "action_chunk_len": action_chunk_len,
        "future_offset": future_offset,
        "shift_actions_to_next_obs": shift_actions_to_next_obs,
        "use_state": use_state,
        "vae_deterministic": bool(args.vae_deterministic),
        "score_mean": float(np.mean(score_values)) if score_values else None,
        "score_std": float(np.std(score_values)) if score_values else None,
    }

    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[DONE] Saved summary to {out_path}")


if __name__ == "__main__":
    main()