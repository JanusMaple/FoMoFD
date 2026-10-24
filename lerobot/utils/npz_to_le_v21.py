#!/usr/bin/env python3
"""
npz_to_le_v21.py

Convert fomofd raw NPZ episodes -> LeRobotDataset v2.1 (local).
Writes to .../data/processed_le_v21 OR .../data/processed_le_v21_latents

Example (Policy: Images + State):
  python utils/npz_to_le_v21.py \
    --task needle_pickup \
    --fps auto \
    --overwrite \
    --gripper_transform std_scale \
    --gripper_std_scale 10 \
    --including_gripper \
    --shift_actions_to_next_obs

Example (World Model: DINOv2 Latents + States):
  python utils/npz_to_le_v21.py \
    --task needle_pickup \
    --fps auto \
    --overwrite \
    --gripper_transform std_scale \
    --gripper_std_scale 10 \
    --including_gripper \
    --extract_latents \
    --shift_actions_to_next_obs

Example (World Model: Multi-layer DINOv2 Latents + States):
  python utils/npz_to_le_v21.py \
    --task needle_pickup \
    --fps auto \
    --overwrite \
    --gripper_transform std_scale \
    --gripper_std_scale 10 \
    --including_gripper \
    --extract_latents_L \
    --shift_actions_to_next_obs

Example (World Model: Multi-layer DINOv2 Fix-Camera Latents + States):
  python utils/npz_to_le_v21.py \
    --task needle_pickup \
    --fps auto \
    --overwrite \
    --gripper_transform std_scale \
    --gripper_std_scale 10 \
    --including_gripper \
    --extract_latents_L_fix \
    --shift_actions_to_next_obs

Example (World Model: Multi-layer DINOv2 Wrist-Camera Latents + States):
  python utils/npz_to_le_v21.py \
    --task needle_pickup \
    --fps auto \
    --overwrite \
    --gripper_transform std_scale \
    --gripper_std_scale 10 \
    --including_gripper \
    --extract_latents_L_wrist \
    --shift_actions_to_next_obs

Example (World Model: VFMF-VAE Fix-Camera Latents + States):
  python utils/npz_to_le_v21.py \
    --task needle_pickup \
    --fps auto \
    --overwrite \
    --gripper_transform std_scale \
    --gripper_std_scale 10 \
    --including_gripper \
    --extract_vfmf_vae_fix \
    --vae_ckpt ../fomofd/tasks/needle_pickup/model/vfmf_vae/dino_vfmf_vae_fix_L8_3072/checkpoints/best.pt \
    --vae_deterministic \
    --shift_actions_to_next_obs

Example (World Model: VFMF-VAE Fix+Wrist Latents + States):
  python utils/npz_to_le_v21.py \
    --task needle_pickup \
    --fps auto \
    --overwrite \
    --gripper_transform std_scale \
    --gripper_std_scale 10 \
    --including_gripper \
    --extract_vfmf_vae_fix_wrist \
    --vae_ckpt_fix ../fomofd/tasks/needle_pickup/model/vfmf_vae/dino_vfmf_vae_fix_L8_3072/checkpoints/best.pt \
    --vae_ckpt_wrist ../fomofd/tasks/needle_pickup/model/vfmf_vae/dino_vfmf_vae_wrist_L8_3072/checkpoints/best.pt \
    --vae_deterministic \
    --shift_actions_to_next_obs

Example (World Model: DINOv2 Camera Tokens + States):
  python utils/npz_to_le_v21.py \
    --task needle_pickup \
    --fps auto \
    --overwrite \
    --gripper_transform std_scale \
    --gripper_std_scale 10 \
    --including_gripper \
    --extract_camera_tokens \
    --act_ckpt_dir ../fomofd/tasks/needle_pickup/model/act_dinov2/checkpoints/last/pretrained_model \
    --shift_actions_to_next_obs

Example (World Model: Pretrained Camera Tokens + States):
    python utils/npz_to_le_v21.py \
        --task needle_pickup \
        --fps auto \
        --overwrite \
        --gripper_transform std_scale \
        --gripper_std_scale 10 \
        --including_gripper \
        --extract_pretrained_tokens \
        --tokenizer_ckpt ../fomofd/tasks/needle_pickup/model/dino_act_wm_tokenizer_only/best_tokenizer.pt \
        --shift_actions_to_next_obs
"""

from __future__ import annotations

import json
import argparse
import shutil
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from lerobot_policy_act_dinov2.configuration_act_dinov2 import ACTDINOv2Config
from lerobot_policy_act_dinov2.dino_visual_tokenizer import DINOVisualTokenizer

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.vfmf_vae.model import FeatureSpaceVAE
from models.vfmf_vae.config import (
    FeatureSpaceVAEConfig,
    DataConfig as VAEDataConfig,
    RuntimeConfig as VAERuntimeConfig,
    OptimConfig as VAEOptimConfig,
    FeatureConfig as VAEFeatureConfig,
    ConvNeXtIsotropicVAEConfig,
    LossConfig as VAELossConfig,
)

"""
utils: action representation conversion
"""
def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = (q / n).tolist()

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ], dtype=np.float32)

def _rotmat_to_rot6d_first_two_cols(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    c0 = R[:, 0]
    c1 = R[:, 1]
    return np.concatenate([c0, c1], axis=0).astype(np.float32, copy=False)

def _action_quat_to_rot6d(
        act8: np.ndarray,
        quat_order: str = "wxyz",
        *,
        gripper_transform: str = "centered_logit",
        gripper_min: float = 0.0436,
        gripper_max: float = 0.6545,
        gripper_eps: float = 1e-4,
        p_u_mean: float | None = None,
        ) -> np.ndarray:
    act8 = np.asarray(act8, dtype=np.float32).reshape(-1,)
    if act8.shape[0] != 8:
        raise ValueError(f"Expected 8D action [dp, quat, gripper], got shape {act8.shape}")

    dp = act8[0:3]
    quat = act8[3:7]
    g = act8[7:8]

    if quat_order.lower() == "xyzw":
        quat = quat[[3, 0, 1, 2]]  # -> wxyz
    elif quat_order.lower() != "wxyz":
        raise ValueError(f"Unsupported quat_order={quat_order} (expected 'wxyz' or 'xyzw').")
    
    if gripper_transform == "centered_logit":
        if p_u_mean is None:
            raise ValueError("p_u_mean is required for centered_logit")
        g = _centered_logit_from_angle(
            g,
            a_min=gripper_min,
            a_max=gripper_max,
            p_u_mean=p_u_mean,
            eps=gripper_eps,
        ).astype(np.float32, copy=False)

    rot6d = _rotmat_to_rot6d_first_two_cols(_quat_wxyz_to_rotmat(quat))
    return np.concatenate([dp, rot6d, g], axis=0).astype(np.float32, copy=False)

def _policy_chunk_to_feature(
    chunk10: np.ndarray,
    *,
    gripper_transform: str = "none",
    gripper_min: float = 0.0436,
    gripper_max: float = 0.6545,
    gripper_eps: float = 1e-4,
    p_u_mean: float | None = None,
) -> np.ndarray:
    """
    chunk10: [K, 10] already in policy space = [dp(3), rot6d(6), gripper(1)].
    Only apply optional gripper transform. Do NOT do quat->rot6d here.
    """
    out = np.asarray(chunk10, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] != 10:
        raise ValueError(f"Expected chunk shape [K,10], got {out.shape}")

    if gripper_transform == "centered_logit":
        if p_u_mean is None:
            raise ValueError("p_u_mean is required for centered_logit")
        out[:, 9] = _centered_logit_from_angle(
            out[:, 9],
            a_min=gripper_min,
            a_max=gripper_max,
            p_u_mean=p_u_mean,
            eps=gripper_eps,
        ).astype(np.float32, copy=False)

    return out

def _policy_chunk_batch_to_feature(
    chunk_batch10: np.ndarray,
    *,
    gripper_transform: str = "none",
    gripper_min: float = 0.0436,
    gripper_max: float = 0.6545,
    gripper_eps: float = 1e-4,
    p_u_mean: float | None = None,
) -> np.ndarray:
    """
    chunk_batch10: [B, K, 10] already in policy space = [dp(3), rot6d(6), gripper(1)].
    Only apply optional gripper transform. Do NOT do quat->rot6d here.
    """
    out = np.asarray(chunk_batch10, dtype=np.float32).copy()
    if out.ndim != 3 or out.shape[2] != 10:
        raise ValueError(f"Expected chunk batch shape [B,K,10], got {out.shape}")

    if gripper_transform == "centered_logit":
        if p_u_mean is None:
            raise ValueError("p_u_mean is required for centered_logit")
        out[:, :, 9] = _centered_logit_from_angle(
            out[:, :, 9],
            a_min=gripper_min,
            a_max=gripper_max,
            p_u_mean=p_u_mean,
            eps=gripper_eps,
        ).astype(np.float32, copy=False)

    return out

"""
utils: state representation conversion
"""
def _state_quat_to_rot6d(state8: np.ndarray, quat_order: str = "wxyz", including_gripper = True) -> np.ndarray:
    s = np.asarray(state8, dtype=np.float32).reshape(-1,)
    if s.shape[0] != 8:
        raise ValueError(f"Expected state8=[p(3),quat(4), gripper], got {s.shape}")

    p = s[0:3]
    q = s[3:7]
    if including_gripper:
        g = s[7:8]

    if quat_order.lower() == "xyzw":
        q = q[[3, 0, 1, 2]]  # -> wxyz
    elif quat_order.lower() != "wxyz":
        raise ValueError(f"Unsupported quat_order={quat_order}")

    R = _quat_wxyz_to_rotmat(q)
    rot6 = _rotmat_to_rot6d_first_two_cols(R)
    if including_gripper:
        return np.concatenate([p, rot6, g], axis=0).astype(np.float32, copy=False)
    else:
        return np.concatenate([p, rot6], axis=0).astype(np.float32, copy=False)

"""
utils: path solver
"""
def _resolve_task_dirs(task: str, mode: str, prl: bool = False) -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    base_name = "prl_data" if prl else "data"
    task_data_dir = repo_root / "fomofd" / "tasks" / task / base_name

    if mode == "images":
        out_dir_name = "processed_le_v21"
    elif mode == "latents":
        out_dir_name = "processed_le_v21_latents"
    elif mode == "latents_L":
        out_dir_name = "processed_le_v21_latents_L"
    elif mode == "latents_L_fix":
        out_dir_name = "processed_le_v21_latents_L_fix"
    elif mode == "latents_L_wrist":
        out_dir_name = "processed_le_v21_latents_L_wrist"
    elif mode == "latents_vfmf_vae_fix":
        out_dir_name = "processed_le_v21_vfmf_vae_fix"
    elif mode == "latents_vfmf_vae_fix_wrist":
        out_dir_name = "processed_le_v21_vfmf_vae_fix_wrist"
    elif mode == "tokens":
        out_dir_name = "processed_le_v21_tokens"
    elif mode == "pretrained_tokens":
        out_dir_name = "processed_le_v21_pretrained_tokens"
    else:
        raise ValueError(f"Unsupported mode={mode}")

    return task_data_dir / "raw", task_data_dir / out_dir_name

"""
utils: fps estimator
"""
def _episode_fps_from_timestamps(
    timestamps: np.ndarray,
    *,
    min_dt_s: float,
    max_gap_s: float,
) -> float | None:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size < 3:
        return None

    dts = np.diff(ts)
    dts = dts[np.isfinite(dts)]
    dts = dts[dts > float(min_dt_s)]  # drop duplicates / non-positive

    if max_gap_s is not None and max_gap_s > 0:
        dts = dts[dts <= float(max_gap_s)]  # drop pauses

    if dts.size == 0:
        return None

    dt = float(np.median(dts))
    if not np.isfinite(dt) or dt <= 0:
        return None
    return 1.0 / dt


def _infer_global_fps(
    episode_files: list[Path],
    *,
    min_dt_s: float,
    max_gap_s: float,
) -> float:
    fps_list: list[float] = []
    for ep in episode_files:
        data = np.load(ep, allow_pickle=True)
        fps = _episode_fps_from_timestamps(
            data["timestamps"],
            min_dt_s=min_dt_s,
            max_gap_s=max_gap_s,
        )
        if fps is not None and np.isfinite(fps) and fps > 0:
            fps_list.append(float(fps))

    if not fps_list:
        return 30.0
    return float(np.median(np.array(fps_list, dtype=np.float64)))

"""
utils: gripper action conversion
"""
def _logit(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    return np.log(u) - np.log(1.0 - u)

def _gripper_angle_to_u(a: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return (a - a_min) / max(1e-12, (a_max - a_min))

def _centered_logit_from_angle(
        a: np.ndarray,
        *,
        a_min: float,
        a_max: float,
        p_u_mean: float,
        eps: float,) -> np.ndarray:
    u = _gripper_angle_to_u(a, a_min, a_max)
    u = np.clip(u, eps, 1.0 - eps)
    p = float(np.clip(p_u_mean, eps, 1.0 - eps))
    return _logit(u) - _logit(p)

def _apply_gripper_std_scale(out_dir: Path, gripper_idx: int, scale: float) -> None:
    if scale is None or scale == 1.0:
        return
    if scale <= 0:
        raise ValueError(f"gripper_std_scale must be > 0, got {scale}")

    stats_path = out_dir / "meta" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found at {out_dir}/meta/stats.json")

    stats = json.loads(stats_path.read_text())
    if "action" not in stats or "std" not in stats["action"]:
        raise KeyError(f"stats.json missing action.std: {stats_path}")

    std = stats["action"]["std"]
    if gripper_idx >= len(std):
        raise IndexError(f"gripper_idx={gripper_idx} out of range for action.std len={len(std)}")

    old = float(std[gripper_idx])
    new = old / float(scale)
    std[gripper_idx] = new
    stats["action"]["std"] = std

    stats.setdefault("_fomofd_overrides", {})
    stats["_fomofd_overrides"]["gripper_std_scale"] = float(scale)
    stats["_fomofd_overrides"]["gripper_std_scaled_index"] = int(gripper_idx)
    stats["_fomofd_overrides"]["gripper_std_before"] = old
    stats["_fomofd_overrides"]["gripper_std_after"] = new

    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[INFO] Applied gripper std scale: idx={gripper_idx} std {old:.6g} -> {new:.6g} (scale={scale}) at {stats_path}")

""" utils: DINOv2 latent extraction (batched) """
def _load_dinov2_model(device: str, *, model_name: str = "dinov2_vitb14_reg", pretrained: bool = True):
    """Use cached Hub code without GitHub's default-branch discovery request."""
    cached_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if (cached_repo / "hubconf.py").is_file():
        print(f"[INFO] Using cached DINOv2 code: {cached_repo}")
        kwargs = {} if pretrained else {"pretrained": False}
        model = torch.hub.load(str(cached_repo), model_name, source="local", **kwargs)
    else:
        # An explicit ref also avoids the main/master discovery probe on first use.
        kwargs = {} if pretrained else {"pretrained": False}
        model = torch.hub.load("facebookresearch/dinov2:main", model_name, trust_repo=True, **kwargs)
    return model.to(device).eval()


@torch.inference_mode()
def _extract_latents_batch(images_np: np.ndarray, model: torch.nn.Module, device: str, batch_size: int = 64):
    T = images_np.shape[0]
    patch_latents = np.zeros((T, 256, 768), dtype=np.float32)
    cls_latents = np.zeros((T, 768), dtype=np.float32)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for i in range(0, T, batch_size):
        batch_np = images_np[i : i + batch_size]
        batch_t = torch.from_numpy(batch_np).float().to(device) / 255.0
        batch_t = batch_t.permute(0, 3, 1, 2)
        batch_t = (batch_t - mean) / std

        feat = model.forward_features(batch_t)
        patch_out = feat["x_norm_patchtokens"]
        cls_out = feat["x_norm_clstoken"]

        patch_out = F.normalize(patch_out, p=2, dim=2)
        cls_out = F.normalize(cls_out, p=2, dim=1)

        patch_latents[i : i + batch_size] = patch_out.cpu().numpy()
        cls_latents[i : i + batch_size] = cls_out.cpu().numpy()

    return {
        "x_norm_patchtokens": patch_latents,
        "x_norm_clstoken": cls_latents,
    }

@torch.inference_mode()
def _extract_latents_L_batch(
    images_np: np.ndarray,
    model: torch.nn.Module,
    device: str,
    batch_size: int = 64,
    layers: list[int] | tuple[int, ...] = (2, 5, 8, 11),
):
    T = images_np.shape[0]
    patch_latents = np.zeros((T, 256, 3072), dtype=np.float32)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for i in range(0, T, batch_size):
        batch_np = images_np[i : i + batch_size]
        batch_t = torch.from_numpy(batch_np).float().to(device) / 255.0
        batch_t = batch_t.permute(0, 3, 1, 2)
        batch_t = (batch_t - mean) / std

        patch_list = model.get_intermediate_layers(
            batch_t,
            n=list(layers),
            reshape=False,
        )  # list of [(B,256,768), ...]

        patch_out = torch.cat(patch_list, dim=-1)   # (B,256,3072)
        patch_out = F.normalize(patch_out, p=2, dim=2)

        patch_latents[i : i + batch_t.shape[0]] = patch_out.cpu().numpy()

    return {
        "x_multiscale_patchtokens": patch_latents,
    }

@torch.inference_mode()
def _extract_latents_L_fix_batch(
    images_np: np.ndarray,
    model: torch.nn.Module,
    device: str,
    batch_size: int = 64,
    layers: list[int] | tuple[int, ...] = (2, 5, 8, 11),
):
    T = images_np.shape[0]
    patch_latents = np.zeros((T, 256, 3072), dtype=np.float32)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for i in range(0, T, batch_size):
        batch_np = images_np[i : i + batch_size]
        batch_t = torch.from_numpy(batch_np).float().to(device) / 255.0
        batch_t = batch_t.permute(0, 3, 1, 2)
        batch_t = (batch_t - mean) / std

        patch_list = model.get_intermediate_layers(
            batch_t,
            n=list(layers),
            reshape=False,
        )  # list of (B,256,768)

        patch_out = torch.cat(patch_list, dim=-1)   # (B,256,3072)
        patch_out = F.normalize(patch_out, p=2, dim=2)

        patch_latents[i : i + batch_t.shape[0]] = patch_out.cpu().numpy()

    return {
        "x_multiscale_patchtokens_fix": patch_latents,
    }

""" utils: DINOv2 token extraction (batched) """
@torch.inference_mode()
def _load_frozen_visual_tokenizer(act_ckpt_dir: str, device: str):
    """Restore exactly the trained visual encoder, including its DINO weights."""
    from safetensors import safe_open
    from lerobot.configs.policies import PreTrainedConfig

    checkpoint = Path(act_ckpt_dir)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, ACTDINOv2Config):
        raise ValueError(f"Expected an ACT-DINOv2 checkpoint: {checkpoint}")
    config.device = "cpu"
    backbone = _load_dinov2_model("cpu", model_name=config.dinov2_model_name, pretrained=False)
    tokenizer = DINOVisualTokenizer(config, dinov2=backbone)
    prefix = "model.visual_tokenizer."
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        state = {key[len(prefix):]: weights.get_tensor(key) for key in weights.keys() if key.startswith(prefix)}
    if not state:
        raise ValueError(f"No ACT visual-tokenizer weights found in {checkpoint}")
    tokenizer.load_state_dict(state, strict=True)
    print(f"[INFO] Restored {len(state)} ACT visual tensors, including the policy's DINOv2 backbone.")
    del state
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False
    return tokenizer

@torch.inference_mode()
def _extract_camera_tokens_batch(
    images_fix_np: np.ndarray,
    images_wrist_np: np.ndarray,
    visual_tokenizer: torch.nn.Module,
    device: str,
    batch_size: int = 64,
):
    T = images_fix_np.shape[0]
    if T == 0 or images_wrist_np.shape != images_fix_np.shape:
        raise ValueError("ACT token extraction requires aligned, nonempty fixed/wrist RGB arrays")
    if images_fix_np.ndim != 4 or images_fix_np.shape[-1] != 3:
        raise ValueError("Expected ACT images as [T,H,W,3]")

    # infer token shape from one mini-batch
    first_fix = torch.from_numpy(images_fix_np[:1]).float().to(device) / 255.0
    first_wrist = torch.from_numpy(images_wrist_np[:1]).float().to(device) / 255.0
    first_fix = first_fix.permute(0, 3, 1, 2)
    first_wrist = first_wrist.permute(0, 3, 1, 2)

    first_out = visual_tokenizer({
        "observation.images.fix_camera": first_fix,
        "observation.images.wrist_camera": first_wrist,
    })

    q = first_out.compressed_tokens["observation.images.fix_camera"].shape[1]
    d = first_out.compressed_tokens["observation.images.fix_camera"].shape[2]

    fix_tokens = np.zeros((T, q, d), dtype=np.float32)
    wrist_tokens = np.zeros((T, q, d), dtype=np.float32)

    for i in range(0, T, batch_size):
        batch_fix = torch.from_numpy(images_fix_np[i:i+batch_size]).float().to(device) / 255.0
        batch_wrist = torch.from_numpy(images_wrist_np[i:i+batch_size]).float().to(device) / 255.0

        batch_fix = batch_fix.permute(0, 3, 1, 2)
        batch_wrist = batch_wrist.permute(0, 3, 1, 2)

        out = visual_tokenizer({
            "observation.images.fix_camera": batch_fix,
            "observation.images.wrist_camera": batch_wrist,
        })

        fix_out = out.compressed_tokens["observation.images.fix_camera"]      # (B,Q,D)
        wrist_out = out.compressed_tokens["observation.images.wrist_camera"]  # (B,Q,D)
        if not torch.isfinite(fix_out).all() or not torch.isfinite(wrist_out).all():
            raise ValueError("ACT camera tokens contain non-finite values")

        fix_tokens[i:i+batch_fix.shape[0]] = fix_out.cpu().numpy()
        wrist_tokens[i:i+batch_wrist.shape[0]] = wrist_out.cpu().numpy()

    return {
        "fix_camera": fix_tokens,
        "wrist_camera": wrist_tokens,
    }

""" utils: DINOv2 token extraction with pretrained tokenizer (batched) """
@torch.inference_mode()
def _load_pretrained_tokenizer(tokenizer_ckpt: str, device: str):
    raise NotImplementedError("Deprecated Feature....")
    return tokenizer

@torch.inference_mode()
def _extract_pretrained_tokens_batch(
    images_fix_np: np.ndarray,
    images_wrist_np: np.ndarray,
    dinov2_model: torch.nn.Module,
    pretrained_tokenizer: torch.nn.Module,
    device: str,
    batch_size: int = 64,
):
    T = images_fix_np.shape[0]
    q = int(pretrained_tokenizer.num_bottleneck_tokens)
    d = int(pretrained_tokenizer.latent_dim)

    fix_tokens = np.zeros((T, q, d), dtype=np.float32)
    wrist_tokens = np.zeros((T, q, d), dtype=np.float32)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for i in range(0, T, batch_size):
        batch_fix = torch.from_numpy(images_fix_np[i : i + batch_size]).float().to(device) / 255.0
        batch_wrist = torch.from_numpy(images_wrist_np[i : i + batch_size]).float().to(device) / 255.0

        batch_fix = batch_fix.permute(0, 3, 1, 2)
        batch_wrist = batch_wrist.permute(0, 3, 1, 2)

        batch_fix = (batch_fix - mean) / std
        batch_wrist = (batch_wrist - mean) / std

        feat_fix = dinov2_model.forward_features(batch_fix)["x_norm_patchtokens"]
        feat_wrist = dinov2_model.forward_features(batch_wrist)["x_norm_patchtokens"]

        feat_fix = F.normalize(feat_fix, p=2, dim=2)
        feat_wrist = F.normalize(feat_wrist, p=2, dim=2)

        out_fix = pretrained_tokenizer(
            patch_tokens=feat_fix,
            camera_id=0,
            sampled_patch_indices=None,
            return_attn=False,
        )
        out_wrist = pretrained_tokenizer(
            patch_tokens=feat_wrist,
            camera_id=1,
            sampled_patch_indices=None,
            return_attn=False,
        )

        fix_tokens[i : i + batch_fix.shape[0]] = out_fix.tokens.cpu().numpy()
        wrist_tokens[i : i + batch_wrist.shape[0]] = out_wrist.tokens.cpu().numpy()

    return {
        "fix_camera": fix_tokens,
        "wrist_camera": wrist_tokens,
    }

""" utils: VFMF-VAE latent extraction (batched) """
def _build_vfmf_vae_config_from_checkpoint_dict(raw: dict) -> FeatureSpaceVAEConfig:
    cfg = FeatureSpaceVAEConfig(
        data=VAEDataConfig(**raw["data"]),
        runtime=VAERuntimeConfig(**raw.get("runtime", {})),
        optim=VAEOptimConfig(**raw.get("optim", {})),
        feature=VAEFeatureConfig(**raw.get("feature", {})),
        vae=ConvNeXtIsotropicVAEConfig(**raw.get("vae", {})),
        loss=VAELossConfig(**raw.get("loss", {})),
        experiment_name=raw.get("experiment_name", "dino_feature_convnext_vae"),
        seed=int(raw.get("seed", 0)),
    )
    cfg.validate()
    return cfg


@torch.inference_mode()
def _load_frozen_vfmf_vae(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "config" not in ckpt:
        raise KeyError(f"Checkpoint missing 'config': {ckpt_path}")

    cfg = _build_vfmf_vae_config_from_checkpoint_dict(ckpt["config"])
    model = FeatureSpaceVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg

@torch.inference_mode()
def _extract_vfmf_vae_fix_batch(
    images_np: np.ndarray,
    dinov2_model: torch.nn.Module,
    vfmf_vae: FeatureSpaceVAE,
    device: str,
    batch_size: int = 64,
    layers: list[int] | tuple[int, ...] = (2, 5, 8, 11),
    deterministic: bool = True,
):
    T = images_np.shape[0]
    hp = vfmf_vae.config.data.patch_grid_height
    wp = vfmf_vae.config.data.patch_grid_width
    cz = vfmf_vae.config.vae.latent_channels

    latents = np.zeros((T, hp * wp, cz), dtype=np.float32)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for i in range(0, T, batch_size):
        batch_np = images_np[i : i + batch_size]
        batch_t = torch.from_numpy(batch_np).float().to(device) / 255.0
        batch_t = batch_t.permute(0, 3, 1, 2)
        batch_t = (batch_t - mean) / std

        patch_list = dinov2_model.get_intermediate_layers(
            batch_t,
            n=list(layers),
            reshape=False,
        )  # list of (B,256,768)

        feat = torch.cat(patch_list, dim=-1)         # (B,256,3072)
        feat = F.normalize(feat, p=2, dim=2)
        feat = feat.view(batch_t.shape[0], hp, wp, -1)  # (B,16,16,3072)

        z = vfmf_vae.encode(feat, deterministic=deterministic)  # (B,16,16,Cz)
        z = z.view(batch_t.shape[0], hp * wp, cz)

        latents[i : i + batch_t.shape[0]] = z.float().cpu().numpy()

    return {
        "fix_camera": latents,
    }

@torch.inference_mode()
def _extract_vfmf_vae_single_batch(
    images_np: np.ndarray,
    dinov2_model: torch.nn.Module,
    vfmf_vae: FeatureSpaceVAE,
    device: str,
    batch_size: int = 64,
    layers: list[int] | tuple[int, ...] = (2, 5, 8, 11),
    deterministic: bool = True,
):
    T = images_np.shape[0]
    hp = vfmf_vae.config.data.patch_grid_height
    wp = vfmf_vae.config.data.patch_grid_width
    cz = vfmf_vae.config.vae.latent_channels

    latents = np.zeros((T, hp * wp, cz), dtype=np.float32)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for i in range(0, T, batch_size):
        batch_np = images_np[i : i + batch_size]
        batch_t = torch.from_numpy(batch_np).float().to(device) / 255.0
        batch_t = batch_t.permute(0, 3, 1, 2)
        batch_t = (batch_t - mean) / std

        patch_list = dinov2_model.get_intermediate_layers(
            batch_t,
            n=list(layers),
            reshape=False,
        )

        feat = torch.cat(patch_list, dim=-1)         # (B,256,3072)
        feat = F.normalize(feat, p=2, dim=2)
        feat = feat.view(batch_t.shape[0], hp, wp, -1)

        z = vfmf_vae.encode(feat, deterministic=deterministic)  # (B,16,16,Cz)
        z = z.view(batch_t.shape[0], hp * wp, cz)

        latents[i : i + batch_t.shape[0]] = z.float().cpu().numpy()

    return latents


@torch.inference_mode()
def _extract_vfmf_vae_fix_wrist_batch(
    images_fix_np: np.ndarray,
    images_wrist_np: np.ndarray,
    dinov2_model: torch.nn.Module,
    vfmf_vae_fix: FeatureSpaceVAE,
    vfmf_vae_wrist: FeatureSpaceVAE,
    device: str,
    batch_size: int = 64,
    layers: list[int] | tuple[int, ...] = (2, 5, 8, 11),
    deterministic: bool = True,
):
    fix_latents = _extract_vfmf_vae_single_batch(
        images_np=images_fix_np,
        dinov2_model=dinov2_model,
        vfmf_vae=vfmf_vae_fix,
        device=device,
        batch_size=batch_size,
        layers=layers,
        deterministic=deterministic,
    )
    wrist_latents = _extract_vfmf_vae_single_batch(
        images_np=images_wrist_np,
        dinov2_model=dinov2_model,
        vfmf_vae=vfmf_vae_wrist,
        device=device,
        batch_size=batch_size,
        layers=layers,
        deterministic=deterministic,
    )
    return {
        "fix_camera": fix_latents,
        "wrist_camera": wrist_latents,
    }

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, type=str)
    p.add_argument("--fps", type=str, default="auto")
    p.add_argument("--repo_id", type=str, default=None)
    p.add_argument("--raw_dir", type=Path, default=None,
                   help="Override the raw episode directory.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Override the output directory (LeRobot version follows the installed package).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quat_order", type=str, default="wxyz", choices=("wxyz", "xyzw"))
    p.add_argument("--max_gap_s", type=float, default=0.5)
    p.add_argument("--min_dt_s", type=float, default=1e-4)
    p.add_argument("--state_key", type=str, default="states")
    p.add_argument("--including_gripper", action="store_true")
    p.add_argument("--gripper_transform", type=str, default="none", choices=("none", "centered_logit", "std_scale"))
    p.add_argument("--gripper_min", type=float, default=0.0436)
    p.add_argument("--gripper_max", type=float, default=0.6545)
    p.add_argument("--gripper_eps", type=float, default=1e-4)
    p.add_argument("--gripper_std_scale", type=float, default=5.0)
    p.add_argument("--shift_actions_to_next_obs", action="store_true",
                   help="Store frame t with action from t+1, so action causes the next observation.")
    p.add_argument("--extract_latents", action="store_true",
                   help="Extract DINOv2 latents, i.e., patch features instead of raw images.")
    p.add_argument("--extract_latents_L", action="store_true",
                   help="Extract multi-layer DINOv2 patch latents (layers 3,6,9,12), shape (256,3072).")
    p.add_argument("--extract_latents_L_fix", action="store_true",
                   help="Extract multi-layer DINOv2 patch latents for FIX camera only, shape (256,3072).")
    p.add_argument("--extract_latents_L_wrist", action="store_true",
                   help="Extract multi-layer DINOv2 patch latents for WRIST camera only, shape (256,3072).")
    p.add_argument("--extract_vfmf_vae_fix", action="store_true",
                   help="Extract fix-camera VAE latents from multi-layer DINO features.")
    p.add_argument("--vae_ckpt", type=str, default=None,
                   help="Path to trained VFMF-VAE checkpoint, required for --extract_vfmf_vae_fix.")
    p.add_argument("--vae_deterministic", action="store_true",
                   help="Use deterministic VAE encoding (posterior mode) when caching latents.")
    p.add_argument("--extract_vfmf_vae_fix_wrist", action="store_true",
                   help="Extract fix+wrist VAE latents from multi-layer DINO features.")
    p.add_argument("--vae_ckpt_fix", type=str, default=None,
                   help="Path to trained FIX-camera VFMF-VAE checkpoint, required for --extract_vfmf_vae_fix_wrist.")
    p.add_argument("--vae_ckpt_wrist", type=str, default=None,
                   help="Path to trained WRIST-camera VFMF-VAE checkpoint, required for --extract_vfmf_vae_fix_wrist.")
    p.add_argument("--extract_camera_tokens", action="store_true",
               help="Extract compressed camera tokens using a trained ACT visual tokenizer.")
    p.add_argument("--act_ckpt_dir", type=str, default=None,
               help="Path to ACT pretrained_model dir, required for --extract_camera_tokens.")
    p.add_argument("--extract_pretrained_tokens", action="store_true",
                   help="Extract compressed camera tokens using the pretrained LearnedBottleneckTokenizer on top of frozen DINOv2 patch features.")
    p.add_argument("--tokenizer_ckpt", type=str, default=None,
                   help="Path to best_tokenizer.pt, required for --extract_pretrained_tokens.")
    p.add_argument("--prl", action="store_true",
                   help="Use fomofd/tasks/<task>/prl_data instead of data.")
    args = p.parse_args()

    mode = "images"
    n_extract_modes = int(args.extract_latents) + int(args.extract_latents_L) + int(args.extract_latents_L_fix) + int(args.extract_latents_L_wrist) + int(args.extract_vfmf_vae_fix) + int(args.extract_vfmf_vae_fix_wrist) + int(args.extract_camera_tokens) + int(args.extract_pretrained_tokens)
    if n_extract_modes > 1:
        raise ValueError("Use only one of --extract_latents, --extract_latents_L, --extract_latents_L_fix, --extract_latents_L_wrist, --extract_vfmf_vae_fix, --extract_vfmf_vae_fix_wrist, --extract_camera_tokens, or --extract_pretrained_tokens.")
    if args.extract_latents:
        mode = "latents"
    elif args.extract_latents_L:
        mode = "latents_L"
    elif args.extract_latents_L_fix:
        mode = "latents_L_fix"
    elif args.extract_latents_L_wrist:
        mode = "latents_L_wrist"
    elif args.extract_vfmf_vae_fix:
        mode = "latents_vfmf_vae_fix"
    elif args.extract_vfmf_vae_fix_wrist:
        mode = "latents_vfmf_vae_fix_wrist"
    elif args.extract_camera_tokens:
        mode = "tokens"
    elif args.extract_pretrained_tokens:
        mode = "pretrained_tokens"
    in_dir, out_dir = _resolve_task_dirs(args.task, mode, args.prl)
    in_dir = args.raw_dir.resolve() if args.raw_dir else in_dir
    out_dir = args.output_dir.resolve() if args.output_dir else out_dir
    print(f"[INFO] Target output directory: {out_dir}")

    if not in_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found:\n  {in_dir}")

    episode_files = sorted(in_dir.glob("episode_*.npz"),
                           key=lambda path: int(path.stem.split("_")[-1]))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz found in {in_dir}")

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists:\n  {out_dir}\nUse --overwrite.")
        shutil.rmtree(out_dir)

    # Load DINOv2 if extracting latents
    dinov2_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.extract_latents or args.extract_latents_L or args.extract_latents_L_fix or args.extract_latents_L_wrist or args.extract_vfmf_vae_fix or args.extract_vfmf_vae_fix_wrist or args.extract_pretrained_tokens:
        print(f"[INFO] Loading DINOv2 model to {device} for latent extraction...")
        dinov2_model = _load_dinov2_model(device)

    # Load VFMF-VAE if extracting VAE latents
    vfmf_vae = None
    vfmf_vae_cfg = None
    vfmf_vae_fix = None
    vfmf_vae_fix_cfg = None
    vfmf_vae_wrist = None
    vfmf_vae_wrist_cfg = None
    if args.extract_vfmf_vae_fix:
        if args.vae_ckpt is None:
            raise ValueError("--vae_ckpt is required when using --extract_vfmf_vae_fix")
        print(f"[INFO] Loading VFMF-VAE from {args.vae_ckpt} to {device}...")
        vfmf_vae, vfmf_vae_cfg = _load_frozen_vfmf_vae(args.vae_ckpt, device)
    if args.extract_vfmf_vae_fix_wrist:
        if args.vae_ckpt_fix is None or args.vae_ckpt_wrist is None:
            raise ValueError("--vae_ckpt_fix and --vae_ckpt_wrist are required when using --extract_vfmf_vae_fix_wrist")
        print(f"[INFO] Loading FIX VFMF-VAE from {args.vae_ckpt_fix} to {device}...")
        vfmf_vae_fix, vfmf_vae_fix_cfg = _load_frozen_vfmf_vae(args.vae_ckpt_fix, device)
        print(f"[INFO] Loading WRIST VFMF-VAE from {args.vae_ckpt_wrist} to {device}...")
        vfmf_vae_wrist, vfmf_vae_wrist_cfg = _load_frozen_vfmf_vae(args.vae_ckpt_wrist, device)

    # Load ACT visual tokenizer if extracting camera tokens
    visual_tokenizer = None
    if args.extract_camera_tokens:
        if args.act_ckpt_dir is None:
            raise ValueError("--act_ckpt_dir is required when using --extract_camera_tokens")
        print(f"[INFO] Loading ACT visual tokenizer from {args.act_ckpt_dir} to {device}...")
        visual_tokenizer = _load_frozen_visual_tokenizer(args.act_ckpt_dir, device)

    # Load pretrained tokenizer if extracting pretrained tokens
    pretrained_tokenizer = None
    if args.extract_pretrained_tokens:
        if args.tokenizer_ckpt is None:
            raise ValueError("--tokenizer_ckpt is required when using --extract_pretrained_tokens")
        print(f"[INFO] Loading pretrained tokenizer from {args.tokenizer_ckpt} to {device}...")
        pretrained_tokenizer = _load_pretrained_tokenizer(args.tokenizer_ckpt, device)

    # fps
    if args.fps.lower() == "auto":
        fps = _infer_global_fps(episode_files, min_dt_s=args.min_dt_s, max_gap_s=args.max_gap_s)
    else:
        fps = float(args.fps)
    fps_int = int(round(fps))
    print(f"[INFO] Using fps={fps:.3f} (written as int {fps_int})")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset # type: ignore

    # Peek first episode for schema
    first = np.load(episode_files[0], allow_pickle=True)
    images_fix0 = first["images_fix"]
    images_wrist0 = first["images_wrist"]
    actions0 = first["actions"]
    timestamps0 = first["timestamps"]

    policy_chunk_len = None
    policy_chunk_batch_size = None

    has_policy_chunks = bool(args.prl and "policy_action_chunks" in first.files)
    if has_policy_chunks:
        chunks0 = np.asarray(first["policy_action_chunks"], dtype=np.float32)
        if chunks0.ndim != 3 or chunks0.shape[2] != 10:
            raise ValueError(
                f"{episode_files[0].name}: policy_action_chunks must be [T,K,10], got {chunks0.shape}"
            )
        policy_chunk_len = int(chunks0.shape[1])
    
    has_policy_chunk_batches = bool(args.prl and "policy_action_chunk_batches" in first.files)
    if has_policy_chunk_batches:
        batch_chunks0 = np.asarray(first["policy_action_chunk_batches"], dtype=np.float32)
        if batch_chunks0.ndim != 4 or batch_chunks0.shape[3] != 10:
            raise ValueError(
                f"{episode_files[0].name}: policy_action_chunk_batches must be [T,B,K,10], got {batch_chunks0.shape}"
            )
        policy_chunk_batch_size = int(batch_chunks0.shape[1])
        batch_chunk_len = int(batch_chunks0.shape[2])

        if policy_chunk_len is None:
            policy_chunk_len = batch_chunk_len
        elif policy_chunk_len != batch_chunk_len:
            raise ValueError(
                f"{episode_files[0].name}: chunk len mismatch between policy_action_chunks "
                f"({policy_chunk_len}) and policy_action_chunk_batches ({batch_chunk_len})"
            )

    if images_fix0.ndim != 4 or images_fix0.shape[-1] != 3:
        raise ValueError(f"{episode_files[0].name}: images_fix must be [T,H,W,3], got {images_fix0.shape}")
    if images_wrist0.shape != images_fix0.shape:
        raise ValueError(f"{episode_files[0].name}: fix/wrist mismatch: {images_fix0.shape} vs {images_wrist0.shape}")
    if actions0.ndim != 2:
        raise ValueError(f"{episode_files[0].name}: actions must be [T,Da], got {actions0.shape}")
    if timestamps0.shape[0] != images_fix0.shape[0]:
        raise ValueError(f"{episode_files[0].name}: timestamps length mismatch")

    _, H, W, _ = images_fix0.shape
    action_dim = 10

    # state schema
    if args.state_key in first.files:
        state0 = np.asarray(first[args.state_key])
        raw_state_dim = int(state0.shape[1])
        if raw_state_dim == 8:
            state_dim = 10 if args.including_gripper else 9
        else:
            raise ValueError(f"Unexpected state dim={raw_state_dim}. Expected 8.")
    else:
        state_dim = 0

    # Build LeRobot features
    features = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"a{i}" for i in range(action_dim)],
        },
        # LeRobot's built-in timestamp is a regular frame clock. Preserve the
        # recorder's wall clock separately, including for simulation episodes.
        "observation.wall_timestamp": {"dtype": "float64", "shape": (1,), "names": ["unix_seconds"]},
        "observation.raw_frame_index": {"dtype": "int64", "shape": (1,), "names": ["raw_frame"]},
        "observation.raw_episode_id": {"dtype": "int64", "shape": (1,), "names": ["raw_episode"]},
        "action.raw_frame_index": {"dtype": "int64", "shape": (1,), "names": ["raw_frame"]},
    }
    if has_policy_chunks:
        features["observation.policy_action_chunk"] = {
            "dtype": "float32",
            "shape": (policy_chunk_len, action_dim),
            "names": ["chunk_step", "action_dim"],
        }
    if has_policy_chunk_batches:
        features["observation.policy_action_chunk_batch"] = {
            "dtype": "float32",
            "shape": (policy_chunk_batch_size, policy_chunk_len, action_dim),
            "names": ["sample_idx", "chunk_step", "action_dim"],
        }
    
    # --- Add Image Features (raw, latent or camera tokens) ---
    if args.extract_latents:
        print("[INFO] Configuring LeRobot Dataset schema for DINO patch latents.")
        features["observation.latents.fix_camera"] = {
            "dtype": "float32",
            "shape": (256, 768),
            "names": ["patch_idx", "feature_dim"],
        }
        features["observation.latents.wrist_camera"] = {
            "dtype": "float32",
            "shape": (256, 768),
            "names": ["patch_idx", "feature_dim"],
        }
    elif args.extract_latents_L:
        print("[INFO] Configuring LeRobot Dataset schema for multi-layer DINO patch latents.")
        features["observation.latents.fix_camera"] = {
            "dtype": "float32",
            "shape": (256, 3072),
            "names": ["patch_idx", "feature_dim"],
        }
        features["observation.latents.wrist_camera"] = {
            "dtype": "float32",
            "shape": (256, 3072),
            "names": ["patch_idx", "feature_dim"],
        }
    elif args.extract_latents_L_fix:
        print("[INFO] Configuring LeRobot Dataset schema for multi-layer FIX-camera DINO patch latents.")
        features["observation.latents.fix_camera"] = {
            "dtype": "float32",
            "shape": (256, 3072),
            "names": ["patch_idx", "feature_dim"],
        }
    elif args.extract_latents_L_wrist:
        print("[INFO] Configuring LeRobot Dataset schema for multi-layer WRIST-camera DINO patch latents.")
        features["observation.latents.wrist_camera"] = {
            "dtype": "float32",
            "shape": (256, 3072),
            "names": ["patch_idx", "feature_dim"],
        }
    elif args.extract_vfmf_vae_fix:
        cz = int(vfmf_vae.config.vae.latent_channels)
        print("[INFO] Configuring LeRobot Dataset schema for fix-camera VAE latents.")
        features["observation.latents.fix_camera"] = {
            "dtype": "float32",
            "shape": (256, cz),
            "names": ["patch_idx", "latent_dim"],
        }
    elif args.extract_vfmf_vae_fix_wrist:
        cz_fix = int(vfmf_vae_fix.config.vae.latent_channels)
        cz_wrist = int(vfmf_vae_wrist.config.vae.latent_channels)
        if cz_fix != cz_wrist:
            raise ValueError(f"Fix/wrist latent channels must match, got {cz_fix} vs {cz_wrist}")
        print("[INFO] Configuring LeRobot Dataset schema for fix+wrist VAE latents.")
        features["observation.latents.fix_camera"] = {
            "dtype": "float32",
            "shape": (256, cz_fix),
            "names": ["patch_idx", "latent_dim"],
        }
        features["observation.latents.wrist_camera"] = {
            "dtype": "float32",
            "shape": (256, cz_wrist),
            "names": ["patch_idx", "latent_dim"],
        }
    elif args.extract_camera_tokens:
        # infer Q from tokenizer config if possible
        q = int(visual_tokenizer.config.per_camera_num_queries) + int(visual_tokenizer.config.use_cls_token)
        d = int(visual_tokenizer.config.dim_model)

        print("[INFO] Configuring LeRobot Dataset schema for compressed camera tokens.")
        features["observation.tokens.fix_camera"] = {
            "dtype": "float32",
            "shape": (q, d),
            "names": ["token_idx", "feature_dim"],
        }
        features["observation.tokens.wrist_camera"] = {
            "dtype": "float32",
            "shape": (q, d),
            "names": ["token_idx", "feature_dim"],
        }
    elif args.extract_pretrained_tokens:
        q = int(pretrained_tokenizer.num_bottleneck_tokens)
        d = int(pretrained_tokenizer.latent_dim)

        print("[INFO] Configuring LeRobot Dataset schema for pretrained tokenizer camera tokens.")
        features["observation.tokens.fix_camera"] = {
            "dtype": "float32",
            "shape": (q, d),
            "names": ["token_idx", "feature_dim"],
        }
        features["observation.tokens.wrist_camera"] = {
            "dtype": "float32",
            "shape": (q, d),
            "names": ["token_idx", "feature_dim"],
        }
    else:
        features["observation.images.fix_camera"] = {
            "dtype": "image",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        }
        features["observation.images.wrist_camera"] = {
            "dtype": "image",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        }

    if state_dim > 0:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"s{i}" for i in range(state_dim)],
        }

    repo_id = args.repo_id or f"local/{args.task}"
    if args.extract_latents:
        repo_id += "_latents"
    elif args.extract_latents_L:
        repo_id += "_latents_L"
    elif args.extract_latents_L_fix:
        repo_id += "_latents_L_fix"
    elif args.extract_latents_L_wrist:
        repo_id += "_latents_L_wrist"
    elif args.extract_vfmf_vae_fix:
        repo_id += "_vfmf_vae_fix"
    elif args.extract_vfmf_vae_fix_wrist:
        repo_id += "_vfmf_vae_fix_wrist"
    elif args.extract_camera_tokens:
        repo_id += "_tokens"
    elif args.extract_pretrained_tokens:
        repo_id += "_pretrained_tokens"

    # Fetch gripper transform parameters
    p_u_mean = None
    if args.gripper_transform == "centered_logit":
        us = []
        for ep in episode_files:
            d = np.load(ep, allow_pickle=True)
            a = np.asarray(d["actions"], dtype=np.float32)
            if args.shift_actions_to_next_obs:
                if a.shape[0] < 2:
                    continue
                g_ang = a[1:, 7]
            else:
                g_ang = a[:, 7]
            u = _gripper_angle_to_u(g_ang, args.gripper_min, args.gripper_max)
            u = np.clip(u, args.gripper_eps, 1.0 - args.gripper_eps)
            us.append(u)
        u_all = np.concatenate(us, axis=0)
        p_u_mean = float(np.mean(u_all))

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps_int,
        root=out_dir,
        robot_type="fomofd",
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    # Save extra transform metadata
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    extra = {
        "gripper_transform": args.gripper_transform,
        "gripper_min": float(args.gripper_min),
        "gripper_max": float(args.gripper_max),
        "gripper_eps": float(args.gripper_eps),
        "gripper_p_u_mean": (None if p_u_mean is None else float(p_u_mean)),
        "gripper_dim_index_in_action": 9,
        "gripper_raw_is_jaw_angle_rad": True,
        "gripper_std_scale": float(args.gripper_std_scale) if args.gripper_transform == "std_scale" else 1.0,
        "action_aligned_to_next_observation": bool(args.shift_actions_to_next_obs),
    }
    (meta_dir / "gripper_transform.json").write_text(json.dumps(extra, indent=2))

    total_frames = 0

    for ep_idx, ep_file in enumerate(episode_files):
        data = np.load(ep_file, allow_pickle=True)
        images_fix = data["images_fix"]
        images_wrist = data["images_wrist"]
        actions = data["actions"]
        timestamps = data["timestamps"]

        policy_action_chunks = None
        if has_policy_chunks:
            if "policy_action_chunks" not in data.files:
                raise ValueError(f"{ep_file.name}: missing policy_action_chunks")
            policy_action_chunks = np.asarray(data["policy_action_chunks"], dtype=np.float32)

        policy_action_chunk_batches = None
        if has_policy_chunk_batches:
            if "policy_action_chunk_batches" not in data.files:
                raise ValueError(f"{ep_file.name}: missing policy_action_chunk_batches")
            policy_action_chunk_batches = np.asarray(data["policy_action_chunk_batches"], dtype=np.float32)

            if policy_action_chunk_batches.ndim != 4 or policy_action_chunk_batches.shape[3] != 10:
                raise ValueError(
                    f"{ep_file.name}: policy_action_chunk_batches must be [T,B,K,10], got {policy_action_chunk_batches.shape}"
                )
            if int(policy_action_chunk_batches.shape[1]) != policy_chunk_batch_size:
                raise ValueError(
                    f"{ep_file.name}: policy_action_chunk_batches batch-size mismatch: "
                    f"{policy_action_chunk_batches.shape[1]} vs expected {policy_chunk_batch_size}"
                )
            if int(policy_action_chunk_batches.shape[2]) != policy_chunk_len:
                raise ValueError(
                    f"{ep_file.name}: policy_action_chunk_batches chunk-len mismatch: "
                    f"{policy_action_chunk_batches.shape[2]} vs expected {policy_chunk_len}"
                )

        T_raw = int(images_fix.shape[0])
        T_out = T_raw - 1 if args.shift_actions_to_next_obs else T_raw

        # --- Compute Bateched Latents ---
        latents_fix = None
        latents_wrist = None
        if args.extract_latents:
            latents_fix = _extract_latents_batch(images_fix, dinov2_model, device)
            latents_wrist = _extract_latents_batch(images_wrist, dinov2_model, device)

        # --- Compute Batached Long Latents
        latents_L_fix = None
        latents_L_wrist = None
        if args.extract_latents_L:
            latents_L_fix = _extract_latents_L_batch(images_fix, dinov2_model, device)
            latents_L_wrist = _extract_latents_L_batch(images_wrist, dinov2_model, device)

        # --- Compute Batached Long Latents with FIX camera only
        if args.extract_latents_L_fix:
            latents_L_fix = _extract_latents_L_fix_batch(images_fix, dinov2_model, device)

        # --- Compute Batched Long Latents with WRIST camera only
        if args.extract_latents_L_wrist:
            latents_L_wrist = _extract_latents_L_fix_batch(images_wrist, dinov2_model, device)

        # --- Compute Batched VFMF-VAE latents with FIX camera only
        vfmf_vae_fix_batch = None
        if args.extract_vfmf_vae_fix:
            vfmf_vae_fix_batch = _extract_vfmf_vae_fix_batch(
                images_np=images_fix,
                dinov2_model=dinov2_model,
                vfmf_vae=vfmf_vae,
                device=device,
                deterministic=args.vae_deterministic,
            )

        # --- Compute Batched VFMF-VAE latents with FIX+WRIST cameras ---
        vfmf_vae_fix_wrist = None
        if args.extract_vfmf_vae_fix_wrist:
            vfmf_vae_fix_wrist = _extract_vfmf_vae_fix_wrist_batch(
                images_fix_np=images_fix,
                images_wrist_np=images_wrist,
                dinov2_model=dinov2_model,
                vfmf_vae_fix=vfmf_vae_fix,
                vfmf_vae_wrist=vfmf_vae_wrist,
                device=device,
                deterministic=args.vae_deterministic,
            )

        # --- Compute Batched Camera Tokens ---
        camera_tokens = None
        if args.extract_camera_tokens:
            camera_tokens = _extract_camera_tokens_batch(
                images_fix_np=images_fix,
                images_wrist_np=images_wrist,
                visual_tokenizer=visual_tokenizer,
                device=device,
            )

        # --- Compute Batched Pretrained Tokens ---
        pretrained_tokens = None
        if args.extract_pretrained_tokens:
            pretrained_tokens = _extract_pretrained_tokens_batch(
                images_fix_np=images_fix,
                images_wrist_np=images_wrist,
                dinov2_model=dinov2_model,
                pretrained_tokenizer=pretrained_tokenizer,
                device=device,
            )

        if state_dim > 0:
            states = np.asarray(data[args.state_key], dtype=np.float32)
        else:
            states = None
        
        for t in range(T_out):
            obs_idx = t
            act_idx = t + 1 if args.shift_actions_to_next_obs else t

            act_raw = actions[act_idx].astype(np.float32, copy=False)
            act_out = _action_quat_to_rot6d(
                act_raw,
                quat_order=args.quat_order,
                gripper_transform=args.gripper_transform,
                gripper_min=args.gripper_min,
                gripper_max=args.gripper_max,
                gripper_eps=args.gripper_eps,
                p_u_mean=p_u_mean,
            )

            frame = {
                "task": args.task,
                "action": act_out,
                "observation.wall_timestamp": np.array([timestamps[obs_idx]], dtype=np.float64),
                "observation.raw_frame_index": np.array([obs_idx], dtype=np.int64),
                "observation.raw_episode_id": np.array([int(ep_file.stem.split("_")[-1])], dtype=np.int64),
                "action.raw_frame_index": np.array([act_idx], dtype=np.int64),
            }

            if has_policy_chunks:
                frame["observation.policy_action_chunk"] = _policy_chunk_to_feature(
                    policy_action_chunks[act_idx],   # use act_idx, not obs_idx
                    gripper_transform=args.gripper_transform,
                    gripper_min=args.gripper_min,
                    gripper_max=args.gripper_max,
                    gripper_eps=args.gripper_eps,
                    p_u_mean=p_u_mean,
                )

            if has_policy_chunk_batches:
                frame["observation.policy_action_chunk_batch"] = _policy_chunk_batch_to_feature(
                    policy_action_chunk_batches[act_idx],
                    gripper_transform=args.gripper_transform,
                    gripper_min=args.gripper_min,
                    gripper_max=args.gripper_max,
                    gripper_eps=args.gripper_eps,
                    p_u_mean=p_u_mean,
                )

            if args.extract_latents:
                frame["observation.latents.fix_camera"] = latents_fix["x_norm_patchtokens"][obs_idx]
                frame["observation.latents.wrist_camera"] = latents_wrist["x_norm_patchtokens"][obs_idx]
            elif args.extract_camera_tokens:
                frame["observation.tokens.fix_camera"] = camera_tokens["fix_camera"][obs_idx]
                frame["observation.tokens.wrist_camera"] = camera_tokens["wrist_camera"][obs_idx]
            elif args.extract_latents_L:
                frame["observation.latents.fix_camera"] = latents_L_fix["x_multiscale_patchtokens"][obs_idx]
                frame["observation.latents.wrist_camera"] = latents_L_wrist["x_multiscale_patchtokens"][obs_idx]
            elif args.extract_latents_L_fix:
                frame["observation.latents.fix_camera"] = latents_L_fix["x_multiscale_patchtokens_fix"][obs_idx]
            elif args.extract_latents_L_wrist:
                frame["observation.latents.wrist_camera"] = latents_L_wrist["x_multiscale_patchtokens_fix"][obs_idx]
            elif args.extract_vfmf_vae_fix:
                frame["observation.latents.fix_camera"] = vfmf_vae_fix_batch["fix_camera"][obs_idx]
            elif args.extract_vfmf_vae_fix_wrist:
                frame["observation.latents.fix_camera"] = vfmf_vae_fix_wrist["fix_camera"][obs_idx]
                frame["observation.latents.wrist_camera"] = vfmf_vae_fix_wrist["wrist_camera"][obs_idx]
            elif args.extract_pretrained_tokens:
                frame["observation.tokens.fix_camera"] = pretrained_tokens["fix_camera"][obs_idx]
                frame["observation.tokens.wrist_camera"] = pretrained_tokens["wrist_camera"][obs_idx]
            else:
                frame["observation.images.fix_camera"] = images_fix[obs_idx].astype(np.uint8, copy=False)
                frame["observation.images.wrist_camera"] = images_wrist[obs_idx].astype(np.uint8, copy=False)

            if state_dim > 0:
                state_raw = states[obs_idx].astype(np.float32, copy=False)
                state = _state_quat_to_rot6d(
                    state_raw,
                    quat_order=args.quat_order,
                    including_gripper=args.including_gripper,
                )
                frame["observation.state"] = state

            ds.add_frame(frame)

        ds.save_episode()
        total_frames += T_out
        print(f"[INFO] saved episode {ep_idx} ({T_out} frames): {ep_file.name}")

    # LeRobot 0.4.x buffers parquet writers and episode metadata until finalize.
    if hasattr(ds, "finalize"):
        ds.finalize()

    if args.gripper_transform == "std_scale":
        _apply_gripper_std_scale(out_dir, gripper_idx=9, scale=args.gripper_std_scale)

    print(f"[DONE] wrote LeRobot dataset to:\n  {out_dir}")
    print(f"       episodes={len(episode_files)} frames={total_frames}")

if __name__ == "__main__":
    main()
