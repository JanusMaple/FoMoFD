"""
lerobot/utils/replay_hf.py

Replay a processed HuggingFace dataset episode.

Example Run:
  python utils/replay_hf.py --task needle_pickup --episode_idx 0
  python utils/replay_hf.py --task needle_pickup --episode_idx 0 --show_dinov2_pca
  python utils/replay_hf.py \
    --task ring_over_post \
    --episode_idx 0 \
    --show_vfmf_vae_pca
  python utils/replay_hf.py \
    --task ring_over_post \
    --episode_idx 0 \
    --show_vfmf_vae_recon_pca \
    --vae_ckpt_fix ../fomofd/tasks/ring_over_post/model/vfmf_vae/dino_vfmf_vae_fix_L8_3072/checkpoints/best.pt \
    --vae_ckpt_wrist ../fomofd/tasks/ring_over_post/model/vfmf_vae/dino_vfmf_vae_wrist_L8_3072/checkpoints/best.pt
  python utils/replay_hf.py --task needle_pickup --episode_idx 0 --show_significance --policy_dir ../fomofd/tasks/needle_pickup/model/act_dinov2
  python utils/replay_hf.py \
    --task needle_pickup \
    --episode_idx 0 \
    --show_teacher_loss \
    --fix_policy_dir ../fomofd/tasks/needle_pickup/model/act_dinov2_no_state_fix/checkpoints/last/pretrained_model \
    --wrist_policy_dir ../fomofd/tasks/needle_pickup/model/act_dinov2_no_state_wrist/checkpoints/last/pretrained_model \
    --stats_json ../fomofd/tasks/needle_pickup/data/processed_le_v21/meta/stats.json
"""

from __future__ import annotations

import argparse
import time
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
import matplotlib.pyplot as plt

import sys
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
from npz_to_le_v21 import _load_frozen_vfmf_vae

# --- utils ---
def _resolve_processed_hf_dir(root_dir: Path, task: str, prl: bool = False) -> Path:
    base = "prl_data" if prl else "data"
    return root_dir / "fomofd" / "tasks" / task / base / "processed_hf"

# --- DINOv2 PCA & Significance Score ---
import torch
import torchvision.transforms as T

# --- Significance Score ---
from lerobot_policy_act_dinov2.modeling_act_dinov2 import ACTDINOv2Policy

FPS = 30

def _chw_u8_to_hwc_u8(x) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = x.astype(np.uint8, copy=False)

    # CHW
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[1] > 4 and x.shape[2] > 4:
        return np.transpose(x, (1, 2, 0))
    # HWC
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        return x
    raise ValueError(f"Unexpected image shape {x.shape} (expected CHW or HWC).")

def _resize_nn(img: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = img.shape[:2]
    if (h, w) == (H, W):
        return img
    ys = (np.linspace(0, h - 1, H)).astype(np.int64)
    xs = (np.linspace(0, w - 1, W)).astype(np.int64)
    return img[ys][:, xs]

def _stack_side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    Ha, Wa = a.shape[:2]
    Hb, Wb = b.shape[:2]
    if (Ha, Wa) != (Hb, Wb):
        b = _resize_nn(b, Ha, Wa)
    return np.concatenate([a, b], axis=1)

def _get_episode_indices(ds, episode_idx: int) -> np.ndarray:
    ep = np.asarray(ds["episode_index"], dtype=np.int64)
    idx = np.where(ep == int(episode_idx))[0]
    if idx.size == 0:
        raise ValueError(f"No rows found for episode_idx={episode_idx}")
    # sort by step_index (robust)
    steps = np.asarray([ds[int(i)]["step_index"] for i in idx], dtype=np.int64)
    return idx[np.argsort(steps)]

def _rot6d_to_rotmat(r6d: np.ndarray) -> np.ndarray:
    r6d = np.asarray(r6d, dtype=np.float32).reshape(6,)
    a1 = r6d[0:3]
    a2 = r6d[3:6]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=1)
    return R

def _rotmat_to_rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    roll  = np.arctan2(R[2, 1], R[2, 2])
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])

def _format_action(action, *, precision: int = 2) -> str:
    a = np.asarray(action).reshape(-1)
    if a.shape[0] != 10:
        raise ValueError(f"Expected 10D action [dp, rot6d, g], got {a.shape}")
    dp_mm = a[0:3] * 1000.0 
    r6d = a[3:9]
    R = _rot6d_to_rotmat(r6d)
    roll, pitch, yaw = _rotmat_to_rpy_deg(R)
    g = a[9]
    fmt = f"{{:.{precision}f}}"
    dp_str = ", ".join(fmt.format(x) for x in dp_mm)
    rpy_str = ", ".join(fmt.format(x) for x in (roll, pitch, yaw))
    return f"dp [mm]=[{dp_str}]   rpy [deg]=[{rpy_str}]   g={fmt.format(g)}"

def _format_state(state, *, precision: int = 2) -> str:
    s = np.asarray(state).reshape(-1)
    fmt = f"{{:.{precision}f}}"
    if s.shape[0] == 10:
        p_mm = s[0:3] * 1000.0
        r6d = s[3:9]
        gpr = s[9] / np.pi * 180.0
        R = _rot6d_to_rotmat(r6d)
        roll, pitch, yaw = _rotmat_to_rpy_deg(R)
        p_str = ", ".join(fmt.format(x) for x in p_mm)
        rpy_str = ", ".join(fmt.format(x) for x in (roll, pitch, yaw))
        gpr_str = fmt.format(gpr)
        return f"state p[mm]=[{p_str}]  rpy[deg]=[{rpy_str}  gpr[deg]={gpr_str}]"
    if s.shape[0] == 8:
        p_mm = s[0:3] * 1000.0
        q = s[3:7]
        gpr = s[7] / np.pi * 180.0
        p_str = ", ".join(fmt.format(x) for x in p_mm)
        q_str = ", ".join(fmt.format(x) for x in q)
        gpr_str = fmt.format(gpr)
        return f"state p[mm]=[{p_str}]  q[wxyz]=[{q_str}  gpr[deg]={gpr_str}]"
    return f"state dim={s.shape[0]} " + np.array2string(s, precision=precision)

# --- Added PCA Computation Helper ---
@torch.inference_mode()
def _compute_dinov2_pca(img_u8: np.ndarray, model: torch.nn.Module, device: str) -> np.ndarray:
    """Takes an RGB uint8 image, extracts DINOv2 patches, and returns an RGB PCA image."""
    H_orig, W_orig, _ = img_u8.shape
    
    img_t = torch.from_numpy(img_u8).permute(2, 0, 1).float() / 255.0
    
    # DINOv2 requires 224x224 (or multiples of 14)
    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img_t).unsqueeze(0).to(device)

    # Extract patches (256, 768)
    features = model.forward_features(img_tensor)["x_norm_patchtokens"].squeeze(0)
    
    # GPU PCA calculation
    _, _, V = torch.pca_lowrank(features, q=3)
    pca_features = torch.matmul(features, V)
    
    # Normalize to [0, 1] mapped to RGB
    p_min = pca_features.min(dim=0, keepdim=True)[0]
    p_max = pca_features.max(dim=0, keepdim=True)[0]
    pca_features = (pca_features - p_min) / (p_max - p_min + 1e-6)
    
    # Reshape to 16x16 grid and upscale to uint8
    pca_img = pca_features.reshape(16, 16, 3).cpu().numpy()
    pca_img_u8 = (pca_img * 255).astype(np.uint8)
    
    # Scale back to original HxW so it stacks perfectly
    return _resize_nn(pca_img_u8, H_orig, W_orig)

def _resolve_vfmf_vae_le_dir(root_dir: Path, task: str, prl: bool = False) -> Path:
    base = "prl_data" if prl else "data"
    return root_dir / "fomofd" / "tasks" / task / base / "processed_le_v21_vfmf_vae_fix_wrist"


def _get_lerobot_episode_indices(ds: LeRobotDataset, episode_idx: int) -> list[int]:
    pairs = []
    for i in range(len(ds)):
        item = ds[i]
        if int(item["episode_index"]) == int(episode_idx):
            pairs.append((int(item["frame_index"]), i))
    if not pairs:
        raise ValueError(f"No rows found for episode_idx={episode_idx} in VFMF-VAE dataset")
    pairs.sort(key=lambda x: x[0])
    return [i for _, i in pairs]


def _latent_row_to_hw_c(row: dict, key: str, h: int = 16, w: int = 16) -> torch.Tensor:
    z = torch.as_tensor(row[key]).float()
    if z.ndim == 2:
        n, c = z.shape
        if n != h * w:
            raise ValueError(f"Expected {h*w} latent slots, got {n}")
        z = z.view(h, w, c)
    elif z.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected latent shape for {key}: {tuple(z.shape)}")
    return z


@torch.inference_mode()
def _extract_dino_multiscale_feature_map(
    img_u8: np.ndarray,
    model: torch.nn.Module,
    device: str,
    layers: tuple[int, ...] = (2, 5, 8, 11),
) -> torch.Tensor:
    """
    Returns [1,16,16,3072], matching npz_to_le_v21.py VFMF-VAE extraction.
    """
    img_t = torch.from_numpy(img_u8).permute(2, 0, 1).float() / 255.0

    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img_t).unsqueeze(0).to(device)

    patch_list = model.get_intermediate_layers(
        img_tensor,
        n=list(layers),
        reshape=False,
    )
    feat = torch.cat(patch_list, dim=-1)      # [1,256,3072]
    feat = F.normalize(feat, p=2, dim=2)
    feat = feat.view(1, 16, 16, -1)
    return feat


def _pca_grid_to_rgb_u8(x: torch.Tensor, H_orig: int, W_orig: int) -> np.ndarray:
    """
    x: [H,W,C] or [N,C]
    Returns RGB uint8 image resized to original frame size.
    """
    if x.ndim == 3:
        h, w, c = x.shape
        flat = x.reshape(h * w, c)
    elif x.ndim == 2:
        flat = x
        h = w = int(np.sqrt(flat.shape[0]))
    else:
        raise ValueError(f"Expected [H,W,C] or [N,C], got {tuple(x.shape)}")

    flat = flat.float()
    q = min(3, flat.shape[-1], flat.shape[0])
    _, _, V = torch.pca_lowrank(flat, q=q)
    p = flat @ V

    if q < 3:
        pad = torch.zeros(p.shape[0], 3 - q, device=p.device, dtype=p.dtype)
        p = torch.cat([p, pad], dim=-1)

    p_min = p.min(dim=0, keepdim=True)[0]
    p_max = p.max(dim=0, keepdim=True)[0]
    p = (p - p_min) / (p_max - p_min + 1e-6)

    img = p[:, :3].reshape(h, w, 3).detach().cpu().numpy()
    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    return _resize_nn(img_u8, H_orig, W_orig)


def _paired_pca_grid_to_rgb_u8(
    a: torch.Tensor,
    b: torch.Tensor,
    H_orig: int,
    W_orig: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use one shared PCA basis/min/max for comparing original DINO feature and decoded DINO feature.
    a,b: [16,16,C]
    """
    h, w, c = a.shape
    af = a.reshape(h * w, c).float()
    bf = b.reshape(h * w, c).float()

    both = torch.cat([af, bf], dim=0)
    q = min(3, both.shape[-1], both.shape[0])
    _, _, V = torch.pca_lowrank(both, q=q)

    ap = af @ V
    bp = bf @ V

    if q < 3:
        ap = torch.cat([ap, torch.zeros(ap.shape[0], 3 - q, device=ap.device)], dim=-1)
        bp = torch.cat([bp, torch.zeros(bp.shape[0], 3 - q, device=bp.device)], dim=-1)

    both_p = torch.cat([ap, bp], dim=0)
    p_min = both_p.min(dim=0, keepdim=True)[0]
    p_max = both_p.max(dim=0, keepdim=True)[0]

    ap = (ap - p_min) / (p_max - p_min + 1e-6)
    bp = (bp - p_min) / (p_max - p_min + 1e-6)

    a_img = (ap[:, :3].reshape(h, w, 3).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    b_img = (bp[:, :3].reshape(h, w, 3).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    return _resize_nn(a_img, H_orig, W_orig), _resize_nn(b_img, H_orig, W_orig)


@torch.inference_mode()
def _compute_vfmf_vae_latent_pca_from_row(
    row_latent: dict,
    key: str,
    H_orig: int,
    W_orig: int,
    device: str,
) -> np.ndarray:
    z = _latent_row_to_hw_c(row_latent, key).to(device)
    return _pca_grid_to_rgb_u8(z, H_orig, W_orig)


@torch.inference_mode()
def _compute_dino_vs_vfmf_decoded_pca(
    img_u8: np.ndarray,
    row_latent: dict,
    latent_key: str,
    dinov2_model: torch.nn.Module,
    vfmf_vae,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      original DINO multiscale PCA,
      decoded-DINO-feature PCA from stored VAE latent.
    """
    H_orig, W_orig, _ = img_u8.shape

    feat = _extract_dino_multiscale_feature_map(
        img_u8=img_u8,
        model=dinov2_model,
        device=device,
    )  # [1,16,16,3072]

    z = _latent_row_to_hw_c(row_latent, latent_key).unsqueeze(0).to(device)  # [1,16,16,Cz]
    decoded_feat = vfmf_vae.decode(z)  # [1,16,16,3072]

    return _paired_pca_grid_to_rgb_u8(
        feat[0],
        decoded_feat[0],
        H_orig,
        W_orig,
    )

def _load_policy(policy_dir: str, device: str):
    policy_path = Path(policy_dir).expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"policy_dir not found: {policy_path}")

    policy = ACTDINOv2Policy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()

    # Important: keep config and processor steps on the same device too
    policy.config.device = device

    if hasattr(policy, "pre_processor") and hasattr(policy.pre_processor, "steps"):
        for step in policy.pre_processor.steps:
            if hasattr(step, "device"):
                step.device = device

    if hasattr(policy, "post_processor") and hasattr(policy.post_processor, "steps"):
        for step in policy.post_processor.steps:
            if hasattr(step, "device"):
                step.device = "cpu"

    if hasattr(policy, "reset"):
        policy.reset()

    return policy

def _load_stats_json(stats_json: str | Path) -> dict:
    stats_path = Path(stats_json).expanduser().resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found: {stats_path}")
    return json.loads(stats_path.read_text())


def _normalize_np(x: np.ndarray, mean, std) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std = np.clip(std, 1e-6, None)
    return (x - mean) / std

def _row_to_teacher_obs(row: dict, stats: dict | None = None) -> dict[str, torch.Tensor]:
    """
    Build teacher observation batch with batch dimension.
    Images stay in [0,1] as in current replay path.
    State is normalized if stats are provided.
    """
    obs = {}

    if "observation.images.fix_camera" in row:
        obs["observation.images.fix_camera"] = _to_image_tensor(row["observation.images.fix_camera"])

    if "observation.images.wrist_camera" in row:
        obs["observation.images.wrist_camera"] = _to_image_tensor(row["observation.images.wrist_camera"])

    if "observation.state" in row:
        s = np.asarray(row["observation.state"], dtype=np.float32)
        if stats is not None and "observation.state" in stats:
            s = _normalize_np(
                s,
                stats["observation.state"]["mean"],
                stats["observation.state"]["std"],
            )
        obs["observation.state"] = torch.from_numpy(s).unsqueeze(0)

    return obs

@torch.no_grad()
def _first_step_teacher_l1(
    policy,
    obs: dict[str, torch.Tensor],
    action_row,
    stats: dict | None,
    device: str,
) -> tuple[float, np.ndarray]:
    """
    Compute first-step L1 between teacher chunk prediction and normalized GT action.
    Returns:
      loss_scalar,
      first_step_pred (normalized action space)
    """
    obs = {k: v.to(device) for k, v in obs.items()}

    pred = policy.predict_action_chunk(obs)[:, 0, :]   # (1, Da)

    gt = np.asarray(action_row, dtype=np.float32).reshape(-1)
    if stats is not None and "action" in stats:
        gt = _normalize_np(
            gt,
            stats["action"]["mean"],
            stats["action"]["std"],
        )
    gt = torch.from_numpy(gt).unsqueeze(0).to(device)  # (1, Da)

    loss = (pred - gt).abs().mean().item()
    return loss, pred[0].detach().cpu().numpy()

def _to_image_tensor(x) -> torch.Tensor:
    x = np.asarray(x)

    # HWC -> CHW
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        x = np.transpose(x, (2, 0, 1))
    elif x.ndim == 3 and x.shape[0] in (1, 3):
        pass
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")

    # uint8 -> float32 in [0,1]
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)

    return torch.from_numpy(x).unsqueeze(0)   # (1, C, H, W)


def _to_vector_tensor(x) -> torch.Tensor:
    x = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(x).unsqueeze(0)   # (1, D)


def _row_to_policy_obs(row: dict) -> dict[str, torch.Tensor]:
    obs = {}

    if "observation.images.fix_camera" in row:
        obs["observation.images.fix_camera"] = _to_image_tensor(row["observation.images.fix_camera"])

    if "observation.images.wrist_camera" in row:
        obs["observation.images.wrist_camera"] = _to_image_tensor(row["observation.images.wrist_camera"])

    if "observation.state" in row:
        obs["observation.state"] = _to_vector_tensor(row["observation.state"])

    return obs

def _extract_significance(policy) -> np.ndarray | None:
    w = policy.model._last_significance_weights
    if w is None: return None
    w = w.float().cpu().numpy()
    w = np.asarray(w)
    if w.ndim == 2:
        w = w[0]   # (1, 2) -> (2,)
    return w

class PlayerState:
    def __init__(self):
        self.paused = False
        self.quit = False
        self.step_delta = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--episode_idx", type=int, required=True)
    parser.add_argument("--fps", type=float, default=float(FPS))
    parser.add_argument("--wall_time", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--show_dinov2_pca", action="store_true", help="Toggle DINOv2 PCA visualization below original frames.")
    parser.add_argument("--policy_dir", type=str, default=None, help="Path to saved ACT_DINOv2 policy directory")
    parser.add_argument("--show_significance", action="store_true", help="Run the policy and display learned significance weights.")
    parser.add_argument("--show_teacher_loss", action="store_true", help="Load two policies and display their first-step L1 losses.")
    parser.add_argument("--fix_policy_dir", type=str, default=None, help="Path to fix-view pretrained_model dir.")
    parser.add_argument("--wrist_policy_dir", type=str, default=None, help="Path to wrist-view pretrained_model dir.")
    parser.add_argument("--stats_json", type=str, default=None, help="Path to stats.json used to normalize state/action for teacher loss.")
    parser.add_argument("--prl", action="store_true", help="Replay from fomofd/tasks/<task>/prl_data/processed_hf instead of data/processed_hf.")
    parser.add_argument("--show_vfmf_vae_pca", action="store_true",
                        help="Show PCA of stored VFMF-VAE compressed latents.")
    parser.add_argument("--show_vfmf_vae_recon_pca", action="store_true",
                        help="Show PCA of original DINO multiscale features and VAE-decoded DINO features.")
    parser.add_argument("--vfmf_vae_root", type=str, default=None,
                        help="Path to processed_le_v21_vfmf_vae_fix_wrist.")
    parser.add_argument("--vfmf_vae_repo_id", type=str, default=None,
                        help="Repo id for VFMF-VAE LeRobot dataset.")
    parser.add_argument("--vae_ckpt_fix", type=str, default=None,
                        help="Fix-camera VFMF-VAE checkpoint, required for --show_vfmf_vae_recon_pca.")
    parser.add_argument("--vae_ckpt_wrist", type=str, default=None,
                        help="Wrist-camera VFMF-VAE checkpoint, required for --show_vfmf_vae_recon_pca.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    dpath = _resolve_processed_hf_dir(root_dir, args.task, args.prl)
    if not dpath.exists():
        raise FileNotFoundError(f"Processed dataset dir not found: {dpath}")

    ds = load_from_disk(str(dpath))
    if len(ds) == 0:
        raise RuntimeError(f"Dataset is empty: {dpath}")

    # Load DINOv2 if flag is passed
    dinov2_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.show_dinov2_pca or args.show_vfmf_vae_recon_pca:
        print("[INFO] Loading frozen DINOv2 model for live PCA visualization...")
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(device)
        dinov2_model.eval()

    # Load vfmf-vae if any VAE-related flag is passed
    vfmf_latent_ds = None
    vfmf_latent_idx = None
    vfmf_vae_fix = None
    vfmf_vae_wrist = None

    if args.show_vfmf_vae_pca or args.show_vfmf_vae_recon_pca:
        vfmf_root = (
            Path(args.vfmf_vae_root)
            if args.vfmf_vae_root is not None
            else _resolve_vfmf_vae_le_dir(root_dir, args.task, args.prl)
        )
        vfmf_repo_id = args.vfmf_vae_repo_id or f"local/{args.task}_vfmf_vae_fix_wrist"

        print(f"[INFO] Loading VFMF-VAE compressed latent dataset: {vfmf_root}")
        vfmf_latent_ds = LeRobotDataset(
            repo_id=vfmf_repo_id,
            root=vfmf_root,
        )
        vfmf_latent_idx = _get_lerobot_episode_indices(vfmf_latent_ds, args.episode_idx)

    if args.show_vfmf_vae_recon_pca:
        if args.vae_ckpt_fix is None or args.vae_ckpt_wrist is None:
            raise ValueError("--show_vfmf_vae_recon_pca requires --vae_ckpt_fix and --vae_ckpt_wrist")

        print(f"[INFO] Loading FIX VFMF-VAE decoder from: {args.vae_ckpt_fix}")
        vfmf_vae_fix, _ = _load_frozen_vfmf_vae(args.vae_ckpt_fix, device)

        print(f"[INFO] Loading WRIST VFMF-VAE decoder from: {args.vae_ckpt_wrist}")
        vfmf_vae_wrist, _ = _load_frozen_vfmf_vae(args.vae_ckpt_wrist, device)

    # Load policy if flag is passed
    policy = None
    policy_device =  "cuda" if torch.cuda.is_available() else "cpu"
    if args.show_significance:
        if args.policy_dir is None:
            raise ValueError("--show_significance requires --policy_dir")
        print(f"[INFO] Loading policy from: {args.policy_dir}")
        policy = _load_policy(args.policy_dir, policy_device)

    # Load teacher policies and stats if flag is passed
    teacher_fix = None
    teacher_wrist = None
    teacher_stats = None
    if args.show_teacher_loss:
        if args.fix_policy_dir is None or args.wrist_policy_dir is None:
            raise ValueError("--show_teacher_loss requires --fix_policy_dir and --wrist_policy_dir")
        if args.stats_json is None:
            teacher_stats_path = (
                root_dir / "fomofd" / "tasks" / args.task / "data" / "processed_le_v21" / "meta" / "stats.json"
            )
        else:
            teacher_stats_path = Path(args.stats_json)
        print(f"[INFO] Loading teacher stats from: {teacher_stats_path}")
        teacher_stats = _load_stats_json(teacher_stats_path)
        print(f"[INFO] Loading fix teacher from: {args.fix_policy_dir}")
        teacher_fix = _load_policy(args.fix_policy_dir, policy_device)
        print(f"[INFO] Loading wrist teacher from: {args.wrist_policy_dir}")
        teacher_wrist = _load_policy(args.wrist_policy_dir, policy_device)

    def build_vis(row: dict, k: int) -> np.ndarray:
        img_fix = _chw_u8_to_hwc_u8(row["observation.images.fix_camera"])
        img_wrist = _chw_u8_to_hwc_u8(row["observation.images.wrist_camera"])

        rows = [_stack_side_by_side(img_fix, img_wrist)]

        if args.show_dinov2_pca:
            pca_fix = _compute_dinov2_pca(img_fix, dinov2_model, device)
            pca_wrist = _compute_dinov2_pca(img_wrist, dinov2_model, device)
            rows.append(_stack_side_by_side(pca_fix, pca_wrist))

        if args.show_vfmf_vae_pca:
            if vfmf_latent_ds is None or vfmf_latent_idx is None:
                raise RuntimeError("VFMF latent dataset not loaded")

            kk = min(k, len(vfmf_latent_idx) - 1)
            row_lat = vfmf_latent_ds[int(vfmf_latent_idx[kk])]

            z_fix_pca = _compute_vfmf_vae_latent_pca_from_row(
                row_lat,
                "observation.latents.fix_camera",
                img_fix.shape[0],
                img_fix.shape[1],
                device,
            )
            z_wrist_pca = _compute_vfmf_vae_latent_pca_from_row(
                row_lat,
                "observation.latents.wrist_camera",
                img_wrist.shape[0],
                img_wrist.shape[1],
                device,
            )
            rows.append(_stack_side_by_side(z_fix_pca, z_wrist_pca))

        if args.show_vfmf_vae_recon_pca:
            if vfmf_latent_ds is None or vfmf_latent_idx is None:
                raise RuntimeError("VFMF latent dataset not loaded")
            if vfmf_vae_fix is None or vfmf_vae_wrist is None:
                raise RuntimeError("VFMF VAE decoders not loaded")

            kk = min(k, len(vfmf_latent_idx) - 1)
            row_lat = vfmf_latent_ds[int(vfmf_latent_idx[kk])]

            dino_fix_pca, dec_fix_pca = _compute_dino_vs_vfmf_decoded_pca(
                img_fix,
                row_lat,
                "observation.latents.fix_camera",
                dinov2_model,
                vfmf_vae_fix,
                device,
            )
            dino_wrist_pca, dec_wrist_pca = _compute_dino_vs_vfmf_decoded_pca(
                img_wrist,
                row_lat,
                "observation.latents.wrist_camera",
                dinov2_model,
                vfmf_vae_wrist,
                device,
            )

            rows.append(_stack_side_by_side(dino_fix_pca, dino_wrist_pca))
            rows.append(_stack_side_by_side(dec_fix_pca, dec_wrist_pca))

        return np.concatenate(rows, axis=0)

    # Start replay
    idx = _get_episode_indices(ds, args.episode_idx)
    print(f"[REPLAY] task={args.task} episode_idx={args.episode_idx} steps={len(idx)}")
    print("[REPLAY] Controls: space=pause/resume, q=quit, left/right=step when paused")

    plt.ion()
    num_rows = 1
    if args.show_dinov2_pca:
        num_rows += 1
    if args.show_vfmf_vae_pca:
        num_rows += 1
    if args.show_vfmf_vae_recon_pca:
        num_rows += 2

    figsize = (10, max(5, 2.6 * num_rows))
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.manager.set_window_title(f"Replay: {args.task} | episode {args.episode_idx}")
    ax.axis("off")

    state = PlayerState()

    def on_key(event):
        if event.key == " ":
            state.paused = not state.paused
        elif event.key in ("q", "escape"):
            state.quit = True
        elif event.key == "left":
            if state.paused:
                state.step_delta = -1
        elif event.key == "right":
            if state.paused:
                state.step_delta = +1

    fig.canvas.mpl_connect("key_press_event", on_key)

    def on_close(event):
        state.quit = True

    fig.canvas.mpl_connect("close_event", on_close)

    k = 0
    row0 = ds[int(idx[0])]
    vis = build_vis(row0, k=0)

    im = ax.imshow(vis)
    txt = ax.text(
        0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=8, bbox=dict(facecolor="black", alpha=0.55, pad=6)
    )

    dt = 1.0 / float(args.fps)
    next_t = time.time()
    t_init = None         
    ts_prev = None        

    while True:
        if not plt.fignum_exists(fig.number) or state.quit:
            break

        if state.paused:
            if state.step_delta != 0:
                k = int(np.clip(k + state.step_delta, -1, len(idx)))
                if k == -1: k = len(idx) - 1
                elif k == len(idx): k = 0
                state.step_delta = 0
            else:
                plt.pause(0.03)
                continue
        else:
            now = time.time()
            if now < next_t:
                plt.pause(min(0.01, next_t - now))
                continue

        row = ds[int(idx[k])]
        done = bool(row["done"])
        action = np.asarray(row["action"], dtype=np.float32).reshape(-1)
        obs_state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
        ts = float(row["timestamp"])
        if t_init is None:
            t_init = ts
            ts_prev = ts

        if args.wall_time:
            dt_data = max(0.0, ts - (ts_prev if ts_prev is not None else ts))
            ts_prev = ts
            next_t = time.time() + (dt_data / max(args.speed, 1e-6))
        else:
            next_t = time.time() + (dt / max(args.speed, 1e-6))

        vis = build_vis(row, k=k)

        # Compute significance scores live during playback
        significance_text = ""
        if args.show_significance and policy is not None:
            obs = _row_to_policy_obs(row)
            obs = {k: v.to(policy_device) for k, v in obs.items()}
            with torch.inference_mode():
                _ = policy.select_action(obs)
            sig = _extract_significance(policy)
            if sig is not None and sig.shape[0] >= 2:
                significance_text = f"\nsig fix={sig[0]:.3f}   wrist={sig[1]:.3f}"

        # Compute teacher losses live during playback
        teacher_text = ""
        if args.show_teacher_loss and teacher_fix is not None and teacher_wrist is not None:
            obs_teacher = _row_to_teacher_obs(row, teacher_stats)

            loss_fix, pred_fix = _first_step_teacher_l1(
                teacher_fix,
                obs_teacher,
                row["action"],
                teacher_stats,
                policy_device,
            )
            loss_wrist, pred_wrist = _first_step_teacher_l1(
                teacher_wrist,
                obs_teacher,
                row["action"],
                teacher_stats,
                policy_device,
            )
            if loss_fix < loss_wrist:
                hard_fix, hard_wrist = 1, 0
                winner = "fix"
            elif loss_wrist < loss_fix:
                hard_fix, hard_wrist = 0, 1
                winner = "wrist"
            else:
                hard_fix, hard_wrist = 0, 0
                winner = "tie"
            teacher_text = (
                f"\nL_fix={loss_fix:.4f}   L_wrist={loss_wrist:.4f}   "
                f"hard=[{hard_fix},{hard_wrist}]   winner={winner}"
            )

        im.set_data(vis)
        txt.set_text(f"t: {ts - t_init:.3f}   k: {k}/{len(idx)-1}   action: {_format_action(action, precision=2)}\n"
            f"{_format_state(obs_state, precision=2)}")
        txt.set_text(
            f"t: {ts - t_init:.3f}   k: {k}/{len(idx)-1}   action: {_format_action(action, precision=2)}\n"
            f"{_format_state(obs_state, precision=2)}"
            f"{significance_text}"
            f"{teacher_text}"
        )
        fig.canvas.draw_idle()
        plt.pause(0.001)

        if (not state.paused) and (k < len(idx) - 1):
            k += 1

        if not state.paused and done:
            k = 0
            next_t = time.time() + 0.5
            while (not state.quit) and (time.time() < next_t):
                plt.pause(0.03)

            if policy is not None and hasattr(policy, "reset"):
                policy.reset()

    plt.ioff()
    plt.close(fig)
    print("[REPLAY] Finished.")

if __name__ == "__main__":
    main()