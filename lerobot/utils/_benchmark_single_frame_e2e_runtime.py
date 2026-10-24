#!/usr/bin/env python3
"""Shared deployment-style runtime benchmark for single-frame detectors."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.npz_to_le_v21 import (  # noqa: E402
    _load_frozen_vfmf_vae,
    _resolve_task_dirs,
    _state_quat_to_rot6d,
)


ScoreFrameFn = Callable[
    [torch.nn.Module, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    torch.Tensor,
]


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timer_start(device: torch.device) -> float:
    cuda_sync(device)
    return time.perf_counter()


def timer_end(device: torch.device, t0: float) -> float:
    cuda_sync(device)
    return time.perf_counter() - t0


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--task", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_key", default=None)

    # Only checkpoints for cameras enabled by the detector config are required.
    parser.add_argument("--vae_ckpt_fix", default=None)
    parser.add_argument("--vae_ckpt_wrist", default=None)

    parser.add_argument("--device", default=None)
    parser.add_argument("--output_json", default=None)

    parser.add_argument("--state_key", default="states")
    parser.add_argument("--including_gripper", action="store_true")
    parser.add_argument("--quat_order", default="wxyz", choices=["wxyz", "xyzw"])

    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--vae_deterministic", action="store_true")
    return parser


def _require_vae_checkpoint(
    *,
    view: str,
    enabled: bool,
    checkpoint: str | None,
) -> None:
    if enabled and checkpoint is None:
        raise ValueError(
            f"The detector config enables the {view} camera; pass --vae_ckpt_{view}."
        )


@torch.inference_mode()
def encode_camera_views_one_frame(
    *,
    image_fix: np.ndarray | None,
    image_wrist: np.ndarray | None,
    dinov2_model: torch.nn.Module,
    vae_fix: torch.nn.Module | None,
    vae_wrist: torch.nn.Module | None,
    device: torch.device,
    deterministic: bool,
    layers: tuple[int, ...] = (2, 5, 8, 11),
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Encode only the camera views consumed by the detector."""
    view_inputs: list[tuple[str, np.ndarray, torch.nn.Module]] = []
    if vae_fix is not None:
        if image_fix is None:
            raise ValueError("image_fix is required when the fix-camera VAE is loaded")
        view_inputs.append(("fix", image_fix, vae_fix))
    if vae_wrist is not None:
        if image_wrist is None:
            raise ValueError("image_wrist is required when the wrist-camera VAE is loaded")
        view_inputs.append(("wrist", image_wrist, vae_wrist))
    if not view_inputs:
        raise ValueError("At least one camera view must be enabled")

    images = np.stack([image for _, image, _ in view_inputs], axis=0)
    x = torch.from_numpy(images).float().to(device) / 255.0
    x = x.permute(0, 3, 1, 2)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std

    patch_list = dinov2_model.get_intermediate_layers(
        x,
        n=list(layers),
        reshape=False,
    )
    features = F.normalize(torch.cat(patch_list, dim=-1), p=2, dim=2)

    encoded: dict[str, torch.Tensor] = {}
    for index, (view, _, vae) in enumerate(view_inputs):
        hp = int(vae.config.data.patch_grid_height)
        wp = int(vae.config.data.patch_grid_width)
        if features.shape[1] != hp * wp:
            raise ValueError(
                f"DINOv2 produced {features.shape[1]} patch tokens for {view}, "
                f"but its VAE expects a {hp}x{wp} grid."
            )

        feature_map = features[index : index + 1].view(1, hp, wp, -1)
        latent = vae.encode(feature_map, deterministic=deterministic)
        encoded[view] = latent.view(hp, wp, -1).float()

    return encoded.get("fix"), encoded.get("wrist")


def _resolve_existing_config_path(path_value: str | Path, label: str) -> Path:
    path = Path(path_value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path]

    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique_candidates:
            unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in unique_candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{tried}")


def _load_latent_normalizers(
    config: Any,
    device: torch.device,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], Path | None]:
    if not bool(config.data.normalize_vfmf_vae_latents):
        return {}, None

    configured_stats_path = config.data.vfmf_vae_latent_stats_path
    if configured_stats_path is None:
        raise ValueError(
            "normalize_vfmf_vae_latents=True requires "
            "data.vfmf_vae_latent_stats_path"
        )
    stats_path = _resolve_existing_config_path(
        configured_stats_path,
        "data.vfmf_vae_latent_stats_path",
    )
    print(f"[INFO] Latent normalization stats: {stats_path}")

    normalizers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    std_floor = float(config.data.vfmf_vae_latent_std_floor)
    with np.load(stats_path) as stats:
        for view, enabled in (
            ("fix", bool(config.data.use_fix_camera)),
            ("wrist", bool(config.data.use_wrist_camera)),
        ):
            if not enabled:
                continue

            mean_key = f"{view}_z_per_channel_mean"
            std_key = f"{view}_z_per_channel_std"
            if mean_key not in stats.files or std_key not in stats.files:
                raise KeyError(
                    f"Latent stats file {stats_path} must contain "
                    f"'{mean_key}' and '{std_key}'."
                )

            mean = torch.from_numpy(stats[mean_key].astype(np.float32)).view(1, 1, -1)
            std_np = np.maximum(stats[std_key].astype(np.float32), std_floor)
            std = torch.from_numpy(std_np).view(1, 1, -1)
            normalizers[view] = (mean.to(device), std.to(device))

    return normalizers, stats_path


def _normalize_latent(
    latent: torch.Tensor | None,
    normalizer: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor | None:
    if latent is None or normalizer is None:
        return latent
    mean, std = normalizer
    return (latent - mean) / std


def _validate_latent_shape(latent: torch.Tensor | None, view: str, config: Any) -> None:
    if latent is None:
        return
    expected = (
        int(config.data.latent_grid_height),
        int(config.data.latent_grid_width),
        int(config.data.latent_channels),
    )
    if tuple(latent.shape) != expected:
        raise ValueError(
            f"Expected {view} latent shape {expected}, got {tuple(latent.shape)}"
        )


@torch.no_grad()
def run_benchmark(
    *,
    args: argparse.Namespace,
    method: str,
    config: Any,
    model: torch.nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    score_frame: ScoreFrameFn,
    method_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay raw episodes one frame at a time and measure detector latency."""
    use_fix = bool(config.data.use_fix_camera)
    use_wrist = bool(config.data.use_wrist_camera)
    use_state = bool(config.data.use_state)
    if not (use_fix or use_wrist):
        raise ValueError("At least one camera must be enabled in the detector config.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max_frames must be at least 1 when provided.")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max_episodes must be at least 1 when provided.")

    _require_vae_checkpoint(
        view="fix", enabled=use_fix, checkpoint=args.vae_ckpt_fix
    )
    _require_vae_checkpoint(
        view="wrist", enabled=use_wrist, checkpoint=args.vae_ckpt_wrist
    )

    # DINOv2 otherwise selects xFormers when it is installed, even though its
    # attention kernels are CUDA-only in common installations.
    if device.type != "cuda":
        os.environ.setdefault("XFORMERS_DISABLED", "1")

    print(f"[INFO] Loading DINOv2 to {device}")
    dinov2_model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14_reg",
    ).to(device)
    dinov2_model.eval()

    vae_fix = None
    if use_fix:
        print(f"[INFO] Loading FIX VAE: {args.vae_ckpt_fix}")
        vae_fix, _ = _load_frozen_vfmf_vae(args.vae_ckpt_fix, str(device))

    vae_wrist = None
    if use_wrist:
        print(f"[INFO] Loading WRIST VAE: {args.vae_ckpt_wrist}")
        vae_wrist, _ = _load_frozen_vfmf_vae(args.vae_ckpt_wrist, str(device))

    normalizers, latent_stats_path = _load_latent_normalizers(config, device)

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
    print(
        f"[INFO] Inputs: fix={use_fix}, wrist={use_wrist}, state={use_state}, "
        f"normalize_latents={bool(normalizers)}"
    )

    feature_time_s = 0.0
    score_time_s = 0.0
    total_frames = 0
    score_values: list[float] = []
    autocast_enabled = device.type == "cuda" and bool(config.runtime.use_amp)

    cuda_sync(device)
    for ep_file in episode_files:
        with np.load(ep_file, allow_pickle=True) as data:
            required_keys: list[str] = []
            if use_fix:
                required_keys.append("images_fix")
            if use_wrist:
                required_keys.append("images_wrist")
            if use_state:
                required_keys.append(args.state_key)
            missing = [key for key in required_keys if key not in data.files]
            if missing:
                raise KeyError(f"{ep_file} is missing required arrays: {missing}")

            images_fix = np.asarray(data["images_fix"]) if use_fix else None
            images_wrist = np.asarray(data["images_wrist"]) if use_wrist else None
            states = np.asarray(data[args.state_key], dtype=np.float32) if use_state else None

        lengths = [
            array.shape[0]
            for array in (images_fix, images_wrist, states)
            if array is not None
        ]
        if len(set(lengths)) != 1:
            raise ValueError(f"Input arrays in {ep_file} have different lengths: {lengths}")

        for frame_index in range(lengths[0]):
            # Match deployment: encode only the newly arrived camera frame(s).
            t0 = timer_start(device)
            fix_obs, wrist_obs = encode_camera_views_one_frame(
                image_fix=None if images_fix is None else images_fix[frame_index],
                image_wrist=None if images_wrist is None else images_wrist[frame_index],
                dinov2_model=dinov2_model,
                vae_fix=vae_fix,
                vae_wrist=vae_wrist,
                device=device,
                deterministic=args.vae_deterministic,
            )
            fix_obs = _normalize_latent(fix_obs, normalizers.get("fix"))
            wrist_obs = _normalize_latent(wrist_obs, normalizers.get("wrist"))
            feature_time_s += timer_end(device, t0)

            _validate_latent_shape(fix_obs, "fix", config)
            _validate_latent_shape(wrist_obs, "wrist", config)
            fix_batch = None if fix_obs is None else fix_obs.unsqueeze(0)
            wrist_batch = None if wrist_obs is None else wrist_obs.unsqueeze(0)

            state_batch = None
            if states is not None:
                converted_state = _state_quat_to_rot6d(
                    states[frame_index],
                    quat_order=args.quat_order,
                    including_gripper=args.including_gripper,
                )
                expected_state_dim = int(config.data.state_dim)
                if converted_state.shape != (expected_state_dim,):
                    raise ValueError(
                        f"Converted state has shape {converted_state.shape}, but the "
                        f"detector expects ({expected_state_dim},). Check "
                        "--including_gripper and --quat_order."
                    )
                state_batch = torch.from_numpy(converted_state).float().unsqueeze(0).to(device)

            t0 = timer_start(device)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=autocast_enabled,
            ):
                score = score_frame(
                    model,
                    fix_batch,
                    wrist_batch,
                    state_batch,
                )
            score_time_s += timer_end(device, t0)

            if score.numel() != 1:
                raise ValueError(
                    f"Expected one scalar score per frame, got shape {tuple(score.shape)}"
                )
            score_values.append(float(score.detach().float().cpu().item()))
            total_frames += 1

            if args.max_frames is not None and total_frames >= args.max_frames:
                break

        if args.max_frames is not None and total_frames >= args.max_frames:
            break

    if total_frames == 0:
        raise RuntimeError("No frames were benchmarked.")

    feature_ms = 1000.0 * feature_time_s / total_frames
    score_ms = 1000.0 * score_time_s / total_frames
    e2e_ms = feature_ms + score_ms

    summary: dict[str, Any] = {
        "method": method,
        "task": args.task,
        "device": str(device),
        "checkpoint": args.checkpoint,
        "vae_ckpt_fix": args.vae_ckpt_fix if use_fix else None,
        "vae_ckpt_wrist": args.vae_ckpt_wrist if use_wrist else None,
        "num_frames": total_frames,
        "feature_time_s_total": feature_time_s,
        "score_time_s_total": score_time_s,
        "e2e_time_s_total": feature_time_s + score_time_s,
        "feature_ms_per_frame": feature_ms,
        "score_ms_per_frame": score_ms,
        "e2e_ms_per_frame": e2e_ms,
        "feature_hz_only": 1000.0 / feature_ms if feature_ms > 0 else float("inf"),
        "score_hz_only": 1000.0 / score_ms if score_ms > 0 else float("inf"),
        "e2e_hz": 1000.0 / e2e_ms if e2e_ms > 0 else float("inf"),
        "use_fix_camera": use_fix,
        "use_wrist_camera": use_wrist,
        "use_state": use_state,
        "normalize_vfmf_vae_latents": bool(normalizers),
        "latent_stats_path": None if latent_stats_path is None else str(latent_stats_path),
        "vae_deterministic": bool(args.vae_deterministic),
        "score_mean": float(np.mean(score_values)),
        "score_std": float(np.std(score_values)),
    }
    if method_summary is not None:
        summary.update(method_summary)

    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[DONE] Saved summary to {out_path}")

    return summary
