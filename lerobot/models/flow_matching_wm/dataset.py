from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

from .config import FlowMatchingWorldModelConfig


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


def select_training_episode_ids(
    episode_ids: list[int], fraction: float = 1.0, seed: int = 0,
) -> list[int]:
    """Nested, deterministic subsets of expert training episodes.

    The validation split is made separately and never changes with fraction.
    Select floor(N*fraction), with at least one training episode. Smaller
    fractions are prefixes of the same shuffled ordering for a fixed seed.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Training episode fraction must be in (0, 1]")
    if not episode_ids:
        raise ValueError("No training episodes available")
    ids = sorted(episode_ids)
    if fraction == 1.0:
        return ids
    random.Random(seed).shuffle(ids)
    return sorted(ids[: max(1, int(len(ids) * fraction))])


def _load_view_global_mean_std(
    stats: np.lib.npyio.NpzFile,
    view: str,
    std_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns scalar mean/std as float32 arrays with shape [1].

    Prefer explicit global stats if inspect_vfmf_latent_stats.py saved them.
    Otherwise derive global mean/std from per-channel stats. This is exact if
    all channels have the same number of samples, which they do here.
    """
    explicit_key_options = [
        (f"{view}_z_global_mean", f"{view}_z_global_std"),
        (f"{view}_global_mean", f"{view}_global_std"),
        (f"{view}_z_mean", f"{view}_z_std"),
    ]

    for mean_key, std_key in explicit_key_options:
        if mean_key in stats and std_key in stats:
            mean = np.asarray(stats[mean_key], dtype=np.float32).reshape(-1)
            std = np.asarray(stats[std_key], dtype=np.float32).reshape(-1)
            mean = np.array([float(mean[0])], dtype=np.float32)
            std = np.array([max(float(std[0]), std_floor)], dtype=np.float32)
            return mean, std

    # Fallback: derive from per-channel mean/std.
    mean_c = stats[f"{view}_z_per_channel_mean"].astype(np.float32)
    std_c = stats[f"{view}_z_per_channel_std"].astype(np.float32)

    global_mean = float(mean_c.mean())
    second_moment = float(np.mean(std_c ** 2 + mean_c ** 2))
    global_var = max(second_moment - global_mean ** 2, 0.0)
    global_std = max(global_var ** 0.5, std_floor)

    return (
        np.array([global_mean], dtype=np.float32),
        np.array([global_std], dtype=np.float32),
    )


@dataclass(frozen=True)
class WindowSampleRef:
    episode_id: int
    anchor_dataset_index: int
    anchor_pos_in_episode: int


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


def _load_action_normalization_stats(
    stats_path: str,
    std_floor: float,
    mode: str,
) -> dict[str, Tensor]:
    stats = np.load(stats_path)

    if "action_mean" not in stats or "action_std" not in stats:
        raise ValueError(
            f"Action stats file {stats_path} must contain 'action_mean' and 'action_std'."
        )

    mean = np.asarray(stats["action_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(stats["action_std"], dtype=np.float32).reshape(-1)

    if mean.shape[0] < 10 or std.shape[0] < 10:
        raise ValueError(
            f"Action normalization expects action_dim >= 10, got "
            f"mean shape {mean.shape}, std shape {std.shape}"
        )

    std = np.maximum(std, float(std_floor)).astype(np.float32)

    if mode == "dp_gripper_zscore_rot_identity":
        rot_center = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    elif mode == "dp_gripper_zscore_rot_mean":
        rot_center = mean[3:9].astype(np.float32)
    else:
        raise ValueError(f"Unsupported action_normalization_mode={mode!r}")

    return {
        "dp_mean": torch.from_numpy(mean[0:3].astype(np.float32)),
        "dp_std": torch.from_numpy(std[0:3].astype(np.float32)),
        "rot_center": torch.from_numpy(rot_center),
        "gripper_mean": torch.from_numpy(mean[9:10].astype(np.float32)),
        "gripper_std": torch.from_numpy(std[9:10].astype(np.float32)),
    }


class EndpointWindowDataset(Dataset):
    def __init__(
        self,
        ds: LeRobotDataset,
        episode_to_indices: dict[int, list[int]],
        episode_ids: list[int],
        config: FlowMatchingWorldModelConfig,
        include_proactive: bool = False,
    ):
        self.ds = ds
        self.episode_to_indices = episode_to_indices
        self.episode_ids = episode_ids
        self.config = config
        self.include_proactive = bool(include_proactive)

        self.use_fix = config.data.use_fix_camera
        self.use_wrist = config.data.use_wrist_camera

        self.proactive_action_key = (
            getattr(config.data, "proactive_action_key", None)
            if self.include_proactive
            else None
        )
        self.proactive_action_batch_key = (
            getattr(config.data, "proactive_action_batch_key", None)
            if self.include_proactive
            else None
        )

        if not (self.use_fix or self.use_wrist):
            raise ValueError("At least one of use_fix_camera / use_wrist_camera must be enabled.")

        self.h = config.data.history_len
        self.k = config.data.pred_horizon
        self.chunk_len = config.data.action_chunk_len

        self.hg = config.data.latent_grid_height
        self.wg = config.data.latent_grid_width
        self.cg = config.data.latent_channels

        self.normalize_vfmf_vae_latents = bool(
            getattr(config.data, "normalize_vfmf_vae_latents", False)
        )
        self.latent_normalization_mode = getattr(
            config.data,
            "latent_normalization_mode",
            "channel",
        )
        if self.latent_normalization_mode not in {"channel", "view_global"}:
            raise ValueError(
                f"Unsupported latent_normalization_mode={self.latent_normalization_mode!r}. "
                "Use 'channel' or 'view_global'."
            )
        self.vfmf_vae_latent_std_floor = float(
            getattr(config.data, "vfmf_vae_latent_std_floor", 0.02)
        )

        self.fix_latent_mean = None
        self.fix_latent_std = None
        self.wrist_latent_mean = None
        self.wrist_latent_std = None

        if self.normalize_vfmf_vae_latents:
            stats_path = getattr(config.data, "vfmf_vae_latent_stats_path", None)
            if stats_path is None:
                raise ValueError(
                    "normalize_vfmf_vae_latents=True requires vfmf_vae_latent_stats_path"
                )

            stats = np.load(stats_path)

            if self.use_fix:
                if self.latent_normalization_mode == "channel":
                    fix_mean = stats["fix_z_per_channel_mean"].astype(np.float32)
                    fix_std = stats["fix_z_per_channel_std"].astype(np.float32)
                    fix_std = np.maximum(fix_std, self.vfmf_vae_latent_std_floor)
                else:
                    fix_mean, fix_std = _load_view_global_mean_std(
                        stats,
                        view="fix",
                        std_floor=self.vfmf_vae_latent_std_floor,
                    )

                self.fix_latent_mean = torch.from_numpy(fix_mean).view(1, 1, -1)
                self.fix_latent_std = torch.from_numpy(fix_std).view(1, 1, -1)

            if self.use_wrist:
                if self.latent_normalization_mode == "channel":
                    wrist_mean = stats["wrist_z_per_channel_mean"].astype(np.float32)
                    wrist_std = stats["wrist_z_per_channel_std"].astype(np.float32)
                    wrist_std = np.maximum(wrist_std, self.vfmf_vae_latent_std_floor)
                else:
                    wrist_mean, wrist_std = _load_view_global_mean_std(
                        stats,
                        view="wrist",
                        std_floor=self.vfmf_vae_latent_std_floor,
                    )

                self.wrist_latent_mean = torch.from_numpy(wrist_mean).view(1, 1, -1)
                self.wrist_latent_std = torch.from_numpy(wrist_std).view(1, 1, -1)
        

        self.normalize_actions = bool(getattr(config.data, "normalize_actions", False))
        self.action_normalization_mode = getattr(
            config.data,
            "action_normalization_mode",
            "dp_gripper_zscore_rot_identity",
        )
        self.action_stats = None

        if self.normalize_actions:
            action_stats_path = getattr(config.data, "action_stats_path", None)
            if action_stats_path is None:
                raise ValueError("normalize_actions=True requires action_stats_path")

            self.action_stats = _load_action_normalization_stats(
                stats_path=action_stats_path,
                std_floor=float(getattr(config.data, "action_std_floor", 1.0e-6)),
                mode=self.action_normalization_mode,
            )

        self.samples: list[WindowSampleRef] = []
        for ep in tqdm(episode_ids, desc="Building endpoint-window samples", leave=False):
            indices = self.episode_to_indices[ep]
            n = len(indices)

            start_pos = self.h - 1
            end_pos_exclusive = n - self.k

            for pos in range(start_pos, end_pos_exclusive):
                self.samples.append(
                    WindowSampleRef(
                        episode_id=ep,
                        anchor_dataset_index=indices[pos],
                        anchor_pos_in_episode=pos,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _latent_map(self, frame: dict[str, Any], key: str) -> Tensor:
        x = _to_tensor(frame[key])
        return _reshape_latent_map(x, self.hg, self.wg, self.cg)

    def _fix_latent_map(self, frame: dict[str, Any]) -> Tensor:
        x = self._latent_map(frame, self.config.data.fix_key)
        return self._normalize_latent_if_needed(x, "fix")

    def _wrist_latent_map(self, frame: dict[str, Any]) -> Tensor:
        x = self._latent_map(frame, self.config.data.wrist_key)
        return self._normalize_latent_if_needed(x, "wrist")

    def _state(self, frame: dict[str, Any]) -> Tensor:
        return _to_tensor(frame[self.config.data.state_key])

    def _action(self, frame: dict[str, Any]) -> Tensor:
        x = _to_tensor(frame[self.config.data.action_key])
        return self._normalize_action_if_needed(x)

    def _proactive_action_chunk(self, frame: dict[str, Any]) -> Tensor:
        if self.proactive_action_key is None:
            raise RuntimeError("proactive_action_key is not configured")

        x = _to_tensor(frame[self.proactive_action_key])

        if x.ndim != 2:
            raise ValueError(
                f"Expected proactive action chunk [K,A], got shape {tuple(x.shape)} "
                f"for key={self.proactive_action_key}"
            )

        if x.shape[0] < self.chunk_len:
            raise ValueError(
                f"Expected proactive action chunk length at least {self.chunk_len}, got {x.shape[0]} "
                f"for key={self.proactive_action_key}"
            )

        return self._normalize_action_if_needed(x[: self.chunk_len])
    
    def _proactive_action_chunk_batch(self, frame: dict[str, Any]) -> Tensor:
        key = self.proactive_action_batch_key
        if key is None:
            raise RuntimeError("proactive_action_batch_key is not configured")

        x = _to_tensor(frame[key])

        if x.ndim != 3:
            raise ValueError(
                f"Expected proactive action chunk batch [B,K,A], got {tuple(x.shape)}"
            )

        if x.shape[1] < self.chunk_len:
            raise ValueError(
                f"Expected proactive action chunk length at least {self.chunk_len}, got {x.shape[1]}"
            )

        return self._normalize_action_if_needed(x[:, : self.chunk_len, :])
    
    def _normalize_action_if_needed(self, x: Tensor) -> Tensor:
        if not self.normalize_actions:
            return x

        if self.action_stats is None:
            raise RuntimeError("Action normalization stats are not loaded")

        if x.shape[-1] < 10:
            raise ValueError(
                f"Action normalization expects last dim >= 10 for "
                f"dp(3)+rot6d(6)+gripper(1), got {tuple(x.shape)}"
            )

        y = x.clone()
        device = y.device
        dtype = y.dtype

        dp_mean = self.action_stats["dp_mean"].to(device=device, dtype=dtype)
        dp_std = self.action_stats["dp_std"].to(device=device, dtype=dtype)
        rot_center = self.action_stats["rot_center"].to(device=device, dtype=dtype)
        gripper_mean = self.action_stats["gripper_mean"].to(device=device, dtype=dtype)
        gripper_std = self.action_stats["gripper_std"].to(device=device, dtype=dtype)

        # dp(3): z-score.
        y[..., 0:3] = (y[..., 0:3] - dp_mean) / dp_std

        # rot6d(6): center only, no std scaling.
        y[..., 3:9] = y[..., 3:9] - rot_center

        # gripper(1): z-score.
        y[..., 9:10] = (y[..., 9:10] - gripper_mean) / gripper_std

        return y

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.samples[idx]
        ep_indices = self.episode_to_indices[sample.episode_id]
        t = sample.anchor_pos_in_episode

        hist_positions = list(range(t - self.h + 1, t + 1))
        chunk_positions = list(range(t, t + self.chunk_len))
        next_pos = t + 1
        future_pos = t + self.k

        hist_frames = [self.ds[ep_indices[p]] for p in hist_positions]
        chunk_frames = [self.ds[ep_indices[p]] for p in chunk_positions]
        next_frame = self.ds[ep_indices[next_pos]]
        future_frame = self.ds[ep_indices[future_pos]]
        anchor_frame = self.ds[sample.anchor_dataset_index]

        out: dict[str, Tensor] = {}

        if self.use_fix:
            out["fix_hist"] = torch.stack([self._fix_latent_map(f) for f in hist_frames], dim=0)
            out["future_fix"] = self._fix_latent_map(future_frame)
            out["next_fix"] = self._fix_latent_map(next_frame)

        if self.use_wrist:
            out["wrist_hist"] = torch.stack([self._wrist_latent_map(f) for f in hist_frames], dim=0)
            out["future_wrist"] = self._wrist_latent_map(future_frame)
            out["next_wrist"] = self._wrist_latent_map(next_frame)

        if self.config.data.use_state:
            out["state_hist"] = torch.stack([self._state(f) for f in hist_frames], dim=0)
            out["future_state"] = self._state(future_frame)
            out["next_state"] = self._state(next_frame)

        # executed chunk used by reactive mode
        out["action_chunk"] = torch.stack([self._action(f) for f in chunk_frames], dim=0)

        # intended policy chunk at current anchor step, used by proactive mode
        if self.include_proactive:
            if self.proactive_action_key is not None and self.proactive_action_key in anchor_frame:
                out["proactive_action_chunk"] = self._proactive_action_chunk(anchor_frame)

            if self.proactive_action_batch_key is not None and self.proactive_action_batch_key in anchor_frame:
                out["proactive_action_chunk_batch"] = self._proactive_action_chunk_batch(anchor_frame)

        out["episode_index"] = torch.tensor(sample.episode_id, dtype=torch.long)
        out["anchor_frame_index"] = torch.tensor(
            int(self.ds[sample.anchor_dataset_index]["frame_index"]),
            dtype=torch.long,
        )
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


def default_model_inputs_from_batch(batch: dict[str, Tensor], *, include_proactive: bool = False) -> dict[str, Tensor | None]:
    out = {
        "fix_hist": batch.get("fix_hist"),
        "wrist_hist": batch.get("wrist_hist"),
        "action_chunk": batch["action_chunk"],
        "future_fix": batch.get("future_fix"),
        "future_wrist": batch.get("future_wrist"),
        "state_hist": batch.get("state_hist"),
        "future_state": batch.get("future_state"),
        "next_fix": batch.get("next_fix"),
        "next_wrist": batch.get("next_wrist"),
        "next_state": batch.get("next_state"),
    }
    if include_proactive:
        out["proactive_action_chunk"] = batch.get("proactive_action_chunk")
        out["proactive_action_chunk_batch"] = batch.get("proactive_action_chunk_batch")
    return out
