from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

from .config import FeatureSpaceVAEConfig


def build_episode_index(ds: LeRobotDataset) -> dict[int, list[int]]:
    ep_to_pairs: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for i in tqdm(range(len(ds)), desc="Indexing dataset frames", leave=False):
        item = ds[i]
        ep = int(item["episode_index"])
        frame_idx = int(item["frame_index"])
        ep_to_pairs[ep].append((frame_idx, i))

    ep_to_indices: dict[int, list[int]] = {}
    for ep, pairs in ep_to_pairs.items():
        pairs.sort(key=lambda x: x[0])
        ep_to_indices[ep] = [i for _, i in pairs]

    return ep_to_indices


def split_episode_ids(
    episode_ids: list[int],
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    episode_ids = sorted(episode_ids)
    if len(episode_ids) < 2:
        raise ValueError("Need at least 2 episodes for train/val split.")

    rng = random.Random(seed)
    shuffled = episode_ids[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    n_val = min(n_val, len(shuffled) - 1)

    val_episode_ids = sorted(shuffled[:n_val])
    train_episode_ids = sorted(shuffled[n_val:])
    return train_episode_ids, val_episode_ids


@dataclass(frozen=True)
class FrameSampleRef:
    episode_id: int
    dataset_index: int


def _to_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)


def _reshape_patch_features(x: Tensor, hp: int, wp: int, c: int) -> Tensor:
    """
    Accept:
      - [N, C] where N = hp * wp
      - [hp, wp, C]
    Return:
      - [hp, wp, C]
    """
    if x.ndim == 2:
        n, cc = x.shape
        if n != hp * wp:
            raise ValueError(f"Expected {hp*wp} patches, got {n}")
        if cc != c:
            raise ValueError(f"Expected feature dim {c}, got {cc}")
        return x.view(hp, wp, c)

    if x.ndim == 3:
        h, w, cc = x.shape
        if h != hp or w != wp:
            raise ValueError(f"Expected grid [{hp}, {wp}], got [{h}, {w}]")
        if cc != c:
            raise ValueError(f"Expected feature dim {c}, got {cc}")
        return x

    raise ValueError(f"Expected feature tensor rank 2 or 3, got shape {tuple(x.shape)}")


class SingleFrameFeatureDataset(Dataset):
    """
    Single-frame dataset for feature-space VAE training.

    Each item returns one or two precomputed DINO feature maps:
      - fix_features:   [H_p, W_p, C] if enabled
      - wrist_features: [H_p, W_p, C] if enabled
    """

    def __init__(
        self,
        ds: LeRobotDataset,
        episode_to_indices: dict[int, list[int]],
        episode_ids: list[int],
        config: FeatureSpaceVAEConfig,
    ):
        self.ds = ds
        self.episode_to_indices = episode_to_indices
        self.episode_ids = episode_ids
        self.config = config

        self.hp = config.data.patch_grid_height
        self.wp = config.data.patch_grid_width
        self.c = config.feature.input_dim

        self.samples: list[FrameSampleRef] = []
        for ep in tqdm(episode_ids, desc="Building single-frame samples", leave=False):
            indices = self.episode_to_indices[ep]
            for dataset_index in indices:
                self.samples.append(FrameSampleRef(episode_id=ep, dataset_index=dataset_index))

    def __len__(self) -> int:
        return len(self.samples)

    def _feature(self, frame: dict[str, Any], key: str) -> Tensor:
        x = _to_tensor(frame[key])
        return _reshape_patch_features(x, self.hp, self.wp, self.c)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.samples[idx]
        frame = self.ds[sample.dataset_index]

        out: dict[str, Tensor] = {}

        if self.config.data.use_fix_camera:
            out["fix_features"] = self._feature(frame, self.config.data.fix_key)

        if self.config.data.use_wrist_camera:
            out["wrist_features"] = self._feature(frame, self.config.data.wrist_key)

        out["episode_index"] = torch.tensor(sample.episode_id, dtype=torch.long)
        out["frame_index"] = torch.tensor(int(frame["frame_index"]), dtype=torch.long)

        return out


def make_datasets(
    config: FeatureSpaceVAEConfig,
) -> tuple[LeRobotDataset, SingleFrameFeatureDataset, SingleFrameFeatureDataset, dict[int, list[int]]]:
    config.validate()

    ds = LeRobotDataset(
        repo_id=config.data.dataset_repo_id,
        root=config.data.dataset_root,
    )

    episode_to_indices = build_episode_index(ds)
    episode_ids = sorted(episode_to_indices.keys())

    train_episode_ids, val_episode_ids = split_episode_ids(
        episode_ids,
        val_ratio=config.data.val_ratio,
        seed=config.data.split_seed,
    )

    train_ds = SingleFrameFeatureDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=train_episode_ids,
        config=config,
    )
    val_ds = SingleFrameFeatureDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=val_episode_ids,
        config=config,
    )

    return ds, train_ds, val_ds, episode_to_indices


def move_batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def default_model_inputs_from_batch(batch: dict[str, Tensor], config: FeatureSpaceVAEConfig) -> Tensor:
    """
    Returns the single feature tensor to train on.

    For now:
      - if only one camera is enabled, use that camera
      - if both are enabled, concatenate along batch dimension
        so one optimizer step sees both cameras as independent samples
    """
    use_fix = config.data.use_fix_camera
    use_wrist = config.data.use_wrist_camera

    if use_fix and not use_wrist:
        return batch["fix_features"]

    if use_wrist and not use_fix:
        return batch["wrist_features"]

    if use_fix and use_wrist:
        return torch.cat([batch["fix_features"], batch["wrist_features"]], dim=0)

    raise RuntimeError("At least one camera must be enabled.")