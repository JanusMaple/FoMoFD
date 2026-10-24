from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

from .config import LogpZOConfig


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


def _reshape_latent_map(
    x: Tensor,
    h: int,
    w: int,
    c: int,
) -> Tensor:
    """
    Accept:
      - [H*W, C]
      - [H, W, C]
    Return:
      - [H, W, C]
    """
    if x.ndim == 2:
        n, cc = x.shape
        if n != h * w:
            raise ValueError(f"Expected {h*w} latent slots, got {n}")
        if cc != c:
            raise ValueError(f"Expected latent dim {c}, got {cc}")
        return x.view(h, w, c)

    if x.ndim == 3:
        hh, ww, cc = x.shape
        if hh != h or ww != w:
            raise ValueError(f"Expected latent map [{h}, {w}, {c}], got [{hh}, {ww}, {cc}]")
        if cc != c:
            raise ValueError(f"Expected latent dim {c}, got {cc}")
        return x

    raise ValueError(f"Expected rank-2 or rank-3 latent tensor, got shape {tuple(x.shape)}")


class SingleStepObservationDataset(Dataset):
    """
    One sample = one current timestep.
    Returns:
      - current fix latent map (optional)
      - current wrist latent map (optional)
      - current state (optional)
      - episode_index
      - frame_index
    """

    def __init__(
        self,
        ds: LeRobotDataset,
        episode_to_indices: dict[int, list[int]],
        episode_ids: list[int],
        config: LogpZOConfig,
    ):
        self.ds = ds
        self.episode_to_indices = episode_to_indices
        self.episode_ids = episode_ids
        self.config = config

        self.use_fix = config.data.use_fix_camera
        self.use_wrist = config.data.use_wrist_camera
        self.use_state = config.data.use_state

        if not (self.use_fix or self.use_wrist):
            raise ValueError("At least one of use_fix_camera / use_wrist_camera must be enabled.")

        self.hg = config.data.latent_grid_height
        self.wg = config.data.latent_grid_width
        self.cg = config.data.latent_channels

        self.normalize_vfmf_vae_latents = bool(config.data.normalize_vfmf_vae_latents)
        self.vfmf_vae_latent_std_floor = float(config.data.vfmf_vae_latent_std_floor)

        self.fix_latent_mean = None
        self.fix_latent_std = None
        self.wrist_latent_mean = None
        self.wrist_latent_std = None

        if self.normalize_vfmf_vae_latents:
            stats_path = config.data.vfmf_vae_latent_stats_path
            if stats_path is None:
                raise ValueError(
                    "normalize_vfmf_vae_latents=True requires vfmf_vae_latent_stats_path"
                )

            stats = np.load(stats_path)

            if self.use_fix:
                fix_mean = stats["fix_z_per_channel_mean"].astype(np.float32)
                fix_std = stats["fix_z_per_channel_std"].astype(np.float32)
                fix_std = np.maximum(fix_std, self.vfmf_vae_latent_std_floor)

                self.fix_latent_mean = torch.from_numpy(fix_mean).view(1, 1, -1)
                self.fix_latent_std = torch.from_numpy(fix_std).view(1, 1, -1)

            if self.use_wrist:
                wrist_mean = stats["wrist_z_per_channel_mean"].astype(np.float32)
                wrist_std = stats["wrist_z_per_channel_std"].astype(np.float32)
                wrist_std = np.maximum(wrist_std, self.vfmf_vae_latent_std_floor)

                self.wrist_latent_mean = torch.from_numpy(wrist_mean).view(1, 1, -1)
                self.wrist_latent_std = torch.from_numpy(wrist_std).view(1, 1, -1)

        self.samples: list[FrameSampleRef] = []
        for ep in tqdm(episode_ids, desc="Building single-step samples", leave=False):
            for ds_idx in episode_to_indices[ep]:
                self.samples.append(FrameSampleRef(episode_id=ep, dataset_index=ds_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _latent_map(self, frame: dict[str, Any], key: str) -> Tensor:
        x = _to_tensor(frame[key])
        return _reshape_latent_map(x, self.hg, self.wg, self.cg)

    def _fix_obs(self, frame: dict[str, Any]) -> Tensor:
        x = self._latent_map(frame, self.config.data.fix_key)
        return self._normalize_latent_if_needed(x, "fix")

    def _wrist_obs(self, frame: dict[str, Any]) -> Tensor:
        x = self._latent_map(frame, self.config.data.wrist_key)
        return self._normalize_latent_if_needed(x, "wrist")

    def _state_obs(self, frame: dict[str, Any]) -> Tensor:
        x = _to_tensor(frame[self.config.data.state_key])
        if x.ndim != 1:
            raise ValueError(f"Expected state vector [D], got shape {tuple(x.shape)}")
        if x.shape[0] != self.config.data.state_dim:
            raise ValueError(
                f"Expected state dim {self.config.data.state_dim}, got {x.shape[0]}"
            )
        return x

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.samples[idx]
        frame = self.ds[sample.dataset_index]

        out: dict[str, Tensor] = {}

        if self.use_fix:
            out["fix_obs"] = self._fix_obs(frame)         # [H,W,C]

        if self.use_wrist:
            out["wrist_obs"] = self._wrist_obs(frame)     # [H,W,C]

        if self.use_state:
            out["state_obs"] = self._state_obs(frame)     # [D]

        out["episode_index"] = torch.tensor(sample.episode_id, dtype=torch.long)
        out["frame_index"] = torch.tensor(int(frame["frame_index"]), dtype=torch.long)
        return out
    
    def _normalize_latent_if_needed(self, x: Tensor, view: str) -> Tensor:
        if not self.normalize_vfmf_vae_latents:
            return x

        if view == "fix":
            if self.fix_latent_mean is None or self.fix_latent_std is None:
                raise RuntimeError("Fix latent normalization stats are not loaded")
            return (x - self.fix_latent_mean.to(x.device)) / self.fix_latent_std.to(x.device)

        if view == "wrist":
            if self.wrist_latent_mean is None or self.wrist_latent_std is None:
                raise RuntimeError("Wrist latent normalization stats are not loaded")
            return (x - self.wrist_latent_mean.to(x.device)) / self.wrist_latent_std.to(x.device)

        raise ValueError(f"Unknown view={view}")


def _move_to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_move_to_device(v, device) for v in x)
    return x


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: _move_to_device(v, device) for k, v in batch.items()}


def default_model_inputs_from_batch(batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
    return {
        "fix_obs": batch.get("fix_obs"),
        "wrist_obs": batch.get("wrist_obs"),
        "state_obs": batch.get("state_obs"),
    }