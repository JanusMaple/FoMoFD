# ----------------------------
# Recorder
# ----------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import re

@dataclass
class EpisodeRecorder:
    out_dir: Path
    episode_idx: int = 0

    images_fix: list = field(default_factory=list)
    images_wrist: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    timestamps: list = field(default_factory=list)     # [T] float64

    states: list = field(default_factory=list)

    is_recording: bool = False

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx = self._infer_next_episode_idx()

    def _infer_next_episode_idx(self) -> int:
        max_idx = -1
        _EPISODE_RE = re.compile(r"episode_(\d+)\.npz$")
        for p in self.out_dir.iterdir():
            if not p.is_file():
                continue
            m = _EPISODE_RE.match(p.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        return max_idx + 1

    def reset(self) -> None:                            # Stop and clear current episode
        self.is_recording = False
        self.images_fix.clear()
        self.images_wrist.clear()
        self.actions.clear()
        self.timestamps.clear()
        self.states.clear()

    def pause(self) -> None:
        self.is_recording = False

    def start(self) -> None:
        self.is_recording = True

    def push(self,
            rgb_fix: torch.Tensor,
            rgb_wrist: torch.Tensor,
            action: torch.Tensor,
            state: torch.Tensor,
            t_wall: float,) -> None:
        self.images_fix.append(rgb_fix)
        self.images_wrist.append(rgb_wrist)
        self.actions.append(action.detach().clone())
        self.states.append(state.detach().clone())
        self.timestamps.append(float(t_wall))

    def save(self, meta: Optional[dict] = None) -> None:    # Save and stop current episode
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"episode_{self.episode_idx:06d}.npz"

        T = len(self.actions)
        if T == 0:
            self.reset()
            return

        if not (len(self.images_fix) == T and len(self.images_wrist) == T):
            raise RuntimeError(
                f"Length mismatch: fix={len(self.images_fix)} wrist={len(self.images_wrist)} act={T}"
            )

        # Stack on GPU first; then one-time transfer to CPU
        images_fix_t = torch.stack(self.images_fix, dim=0).contiguous()
        images_wrist_t = torch.stack(self.images_wrist, dim=0).contiguous()
        actions_t = torch.stack(self.actions, dim=0).contiguous()
        states_t = torch.stack(self.states, dim=0).contiguous()

        # Optional sanity checks (safe, can remove later)
        if images_fix_t.dtype != torch.uint8 or images_wrist_t.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 images, got {images_fix_t.dtype}, {images_wrist_t.dtype}")
        if actions_t.dtype != torch.float32:
            actions_t = actions_t.to(torch.float32)

        payload = dict(
            images_fix=images_fix_t.cpu().numpy(),              # [T,H,W,3] uint8
            images_wrist=images_wrist_t.cpu().numpy(),          # [T,H,W,3] uint8
            actions=actions_t.cpu().numpy(),                    # [T,Da] float32
            states=states_t.to(torch.float32).cpu().numpy(),    # [T,Ds] float32
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
        )
        if meta is not None:
            payload["meta"] = np.array([meta], dtype=object)

        np.savez_compressed(path, **payload)
        self.episode_idx += 1
        self.reset()