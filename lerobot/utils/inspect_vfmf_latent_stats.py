# utils/inspect_vfmf_latent_stats.py

"""
Example Usage:
  python utils/inspect_vfmf_latent_stats.py \
    --repo_id local/ring_over_post_vfmf_vae_fix_wrist \
    --root ../fomofd/tasks/ring_over_post/data/processed_le_v21_vfmf_vae_fix_wrist \
    --out ../fomofd/tasks/ring_over_post/data/processed_le_v21_vfmf_vae_fix_wrist/meta/latent_stats.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def collect_with_meta(ds, key: str, max_frames: int | None):
    xs = []
    eps = []
    frames = []

    n = len(ds) if max_frames is None else min(len(ds), max_frames)

    for i in range(n):
        item = ds[i]

        x = torch.as_tensor(item[key]).float()
        if x.ndim == 2:
            x = x.view(16, 16, x.shape[-1])

        xs.append(x)
        eps.append(int(item["episode_index"]))
        frames.append(int(item["frame_index"]))

    return (
        torch.stack(xs, dim=0),
        np.asarray(eps, dtype=np.int64),
        np.asarray(frames, dtype=np.int64),
    )


def summarize(name: str, x: torch.Tensor):
    # x: [N,H,W,C]
    flat = x.reshape(-1, x.shape[-1])  # [N*H*W,C]
    per_c_mean = flat.mean(dim=0)
    per_c_std = flat.std(dim=0, unbiased=False)

    global_mean = flat.mean().item()
    global_std = flat.std(unbiased=False).item()
    global_min = flat.min().item()
    global_max = flat.max().item()

    norm = x.flatten(1).norm(dim=1)

    print(f"\n[{name}] shape={tuple(x.shape)}")
    print(f"  global mean/std/min/max: {global_mean:.6f} / {global_std:.6f} / {global_min:.6f} / {global_max:.6f}")
    print(f"  per-channel mean: {per_c_mean.cpu().numpy()}")
    print(f"  per-channel std : {per_c_std.cpu().numpy()}")
    print(f"  frame L2 norm mean/std/min/max: {norm.mean():.6f} / {norm.std(unbiased=False):.6f} / {norm.min():.6f} / {norm.max():.6f}")

    return {
        f"{name}_global_mean": global_mean,
        f"{name}_global_std": global_std,
        f"{name}_global_min": global_min,
        f"{name}_global_max": global_max,
        f"{name}_per_channel_mean": per_c_mean.cpu().numpy(),
        f"{name}_per_channel_std": per_c_std.cpu().numpy(),
        f"{name}_frame_l2_norm": norm.cpu().numpy(),
    }


def collect_actions_with_meta(ds, key: str, max_frames: int | None):
    xs = []
    eps = []
    frames = []

    n = len(ds) if max_frames is None else min(len(ds), max_frames)

    for i in range(n):
        item = ds[i]

        x = torch.as_tensor(item[key]).float()
        if x.ndim != 1:
            raise ValueError(f"Expected action shape [A], got {tuple(x.shape)} for key={key}")

        xs.append(x)
        eps.append(int(item["episode_index"]))
        frames.append(int(item["frame_index"]))

    return (
        torch.stack(xs, dim=0),  # [N,A]
        np.asarray(eps, dtype=np.int64),
        np.asarray(frames, dtype=np.int64),
    )


def summarize_action(name: str, x: torch.Tensor):
    # x: [N,A]
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    min_v = x.min(dim=0).values
    max_v = x.max(dim=0).values

    global_mean = x.mean().item()
    global_std = x.std(unbiased=False).item()
    global_min = x.min().item()
    global_max = x.max().item()

    norm = x.norm(dim=1)

    print(f"\n[{name}] shape={tuple(x.shape)}")
    print(f"  global mean/std/min/max: {global_mean:.6f} / {global_std:.6f} / {global_min:.6f} / {global_max:.6f}")
    print(f"  per-dim mean: {mean.cpu().numpy()}")
    print(f"  per-dim std : {std.cpu().numpy()}")
    print(f"  per-dim min : {min_v.cpu().numpy()}")
    print(f"  per-dim max : {max_v.cpu().numpy()}")
    print(f"  frame L2 norm mean/std/min/max: {norm.mean():.6f} / {norm.std(unbiased=False):.6f} / {norm.min():.6f} / {norm.max():.6f}")

    if x.shape[1] >= 3:
        dp = x[:, 0:3]
        dp_norm = dp.norm(dim=1)
        print(f"  dp mean: {dp.mean(dim=0).cpu().numpy()}")
        print(f"  dp std : {dp.std(dim=0, unbiased=False).cpu().numpy()}")
        print(f"  dp L2 norm mean/std/min/max: {dp_norm.mean():.6f} / {dp_norm.std(unbiased=False):.6f} / {dp_norm.min():.6f} / {dp_norm.max():.6f}")

    if x.shape[1] >= 9:
        rot6d = x[:, 3:9]
        rot_norm = rot6d.norm(dim=1)
        print(f"  rot6d mean: {rot6d.mean(dim=0).cpu().numpy()}")
        print(f"  rot6d std : {rot6d.std(dim=0, unbiased=False).cpu().numpy()}")
        print(f"  rot6d L2 norm mean/std/min/max: {rot_norm.mean():.6f} / {rot_norm.std(unbiased=False):.6f} / {rot_norm.min():.6f} / {rot_norm.max():.6f}")

    if x.shape[1] >= 10:
        g = x[:, 9]
        print(f"  gripper mean/std/min/max: {g.mean():.6f} / {g.std(unbiased=False):.6f} / {g.min():.6f} / {g.max():.6f}")

    out = {
        f"{name}_global_mean": global_mean,
        f"{name}_global_std": global_std,
        f"{name}_global_min": global_min,
        f"{name}_global_max": global_max,
        f"{name}_mean": mean.cpu().numpy(),
        f"{name}_std": std.cpu().numpy(),
        f"{name}_min": min_v.cpu().numpy(),
        f"{name}_max": max_v.cpu().numpy(),
        f"{name}_frame_l2_norm": norm.cpu().numpy(),
    }

    if name == "action":
        out["action_mean"] = mean.cpu().numpy()
        out["action_std"] = std.cpu().numpy()
        out["action_min"] = min_v.cpu().numpy()
        out["action_max"] = max_v.cpu().numpy()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--fix_key", type=str, default="observation.latents.fix_camera")
    ap.add_argument("--wrist_key", type=str, default="observation.latents.wrist_camera")
    ap.add_argument("--action_key", type=str, default="action")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    stats = {}

    for key_name, key in [("fix", args.fix_key), ("wrist", args.wrist_key)]:
        try:
            z, ep, frame = collect_with_meta(ds, key, args.max_frames)
        except Exception as e:
            print(f"[WARN] skipped {key}: {e}")
            continue

        stats.update(summarize(f"{key_name}_z", z))

        # Use collected length only.
        n = z.shape[0]
        ep = ep[:n]
        frame = frame[:n]

        dzs = []
        for i in range(n - 1):
            if ep[i] == ep[i + 1] and frame[i + 1] == frame[i] + 1:
                dzs.append(z[i + 1] - z[i])
        if dzs:
            dz = torch.stack(dzs, dim=0)
            stats.update(summarize(f"{key_name}_delta_z_t1", dz))

    try:
        actions, ep, frame = collect_actions_with_meta(ds, args.action_key, args.max_frames)
        stats.update(summarize_action("action", actions))

        das = []
        n = actions.shape[0]
        for i in range(n - 1):
            if ep[i] == ep[i + 1] and frame[i + 1] == frame[i] + 1:
                das.append(actions[i + 1] - actions[i])

        if das:
            da = torch.stack(das, dim=0)
            stats.update(summarize_action("delta_action_t1", da))

    except Exception as e:
        print(f"[WARN] skipped action stats for {args.action_key}: {e}")

    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **stats)
        print(f"\nSaved stats to {out}")


if __name__ == "__main__":
    main()