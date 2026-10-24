from __future__ import annotations

import argparse
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.flow_matching_wm.score import load_config
from models.flow_matching_wm.dataset import (
    build_episode_index,
    split_episode_ids,
    EndpointWindowDataset,
)


def summarize(name: str, xs: list[torch.Tensor]) -> None:
    x = torch.cat(xs, dim=0)  # [N,H,W,C]
    flat = x.reshape(-1, x.shape[-1])

    print(f"\n[{name}] shape={tuple(x.shape)}")
    print(f"  global mean/std/min/max: "
          f"{flat.mean().item():.6f} / "
          f"{flat.std(unbiased=False).item():.6f} / "
          f"{flat.min().item():.6f} / "
          f"{flat.max().item():.6f}")
    print(f"  per-channel mean: {flat.mean(dim=0).cpu().numpy()}")
    print(f"  per-channel std : {flat.std(dim=0, unbiased=False).cpu().numpy()}")

    norm = x.flatten(1).norm(dim=-1)
    print(f"  frame L2 norm mean/std/min/max: "
          f"{norm.mean().item():.6f} / "
          f"{norm.std(unbiased=False).item():.6f} / "
          f"{norm.min().item():.6f} / "
          f"{norm.max().item():.6f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="all", choices=["train", "val", "all"])
    p.add_argument("--max_items", type=int, default=2000)
    args = p.parse_args()

    cfg = load_config(args.config)

    ds = LeRobotDataset(
        repo_id=cfg.data.dataset_repo_id,
        root=cfg.data.dataset_root,
    )

    episode_to_indices = build_episode_index(ds)
    episode_ids = sorted(episode_to_indices.keys())

    train_eps, val_eps = split_episode_ids(
        episode_ids,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.split_seed,
    )

    if args.split == "train":
        chosen_eps = train_eps
    elif args.split == "val":
        chosen_eps = val_eps
    else:
        chosen_eps = episode_ids

    wds = EndpointWindowDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=chosen_eps,
        config=cfg,
        include_proactive=False,
    )

    fix_last, fix_future = [], []
    wrist_last, wrist_future = [], []

    n = min(len(wds), args.max_items)
    for i in range(n):
        item = wds[i]

        if "fix_hist" in item:
            fix_last.append(item["fix_hist"][-1].unsqueeze(0))
            fix_future.append(item["future_fix"].unsqueeze(0))

        if "wrist_hist" in item:
            wrist_last.append(item["wrist_hist"][-1].unsqueeze(0))
            wrist_future.append(item["future_wrist"].unsqueeze(0))

    print(f"Inspected {n} windows from split={args.split}")

    if fix_last:
        summarize("fix_last_hist", fix_last)
        summarize("fix_future", fix_future)

    if wrist_last:
        summarize("wrist_last_hist", wrist_last)
        summarize("wrist_future", wrist_future)


if __name__ == "__main__":
    main()