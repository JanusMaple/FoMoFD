from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .config import FiperRNDConfig


class IdentityBase(nn.Module):
    """
    Placeholder 'base network' so the module structure stays similar to FIPER's
    Predictor(Target(base_network(...)) pattern, while our observation embedding
    is already precomputed as a flat vector.
    """
    def forward(self, x: Tensor) -> Tensor:
        return x


class PredictorNetwork(nn.Module):
    """
    FIPER-style predictor:
      input -> 2048 -> 1024 -> 256  (configurable)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        output_dim: int,
        use_layernorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers: list[nn.Module] = []

        def maybe_ln(dim: int) -> nn.Module:
            return nn.LayerNorm(dim) if use_layernorm else nn.Identity()

        layers.extend([
            nn.Linear(input_dim, hidden_dim_1),
            maybe_ln(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            maybe_ln(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim_2, output_dim),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TargetNetwork(nn.Module):
    """
    FIPER-style target:
      input -> 256  (single linear head)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class RNDLossOutput:
    total: Tensor


class FiperRNDModel(nn.Module):
    """
    FIPER-style observation-embedding RND baseline.

    Default intended use:
      - wrist-only VFMF-VAE latent
      - no state

    Supported variants:
      - fix only
      - wrist only
      - fix + wrist
      - optional + state concatenation
    """

    def __init__(self, config: FiperRNDConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.use_fix = config.data.use_fix_camera
        self.use_wrist = config.data.use_wrist_camera
        self.use_state = config.data.use_state

        if not (self.use_fix or self.use_wrist):
            raise ValueError("At least one of use_fix_camera / use_wrist_camera must be enabled.")

        h = config.data.latent_grid_height
        w = config.data.latent_grid_width
        c = config.data.latent_channels
        map_dim = h * w * c

        obs_dim = 0
        if self.use_fix:
            obs_dim += map_dim
        if self.use_wrist:
            obs_dim += map_dim
        if self.use_state:
            obs_dim += config.data.state_dim

        self.obs_dim = obs_dim
        self.output_dim = config.model.output_dim

        self.base_network = IdentityBase()

        self.predictor_network = PredictorNetwork(
            input_dim=self.obs_dim,
            hidden_dim_1=config.model.predictor_hidden_dim_1,
            hidden_dim_2=config.model.predictor_hidden_dim_2,
            output_dim=config.model.output_dim,
            use_layernorm=config.model.use_layernorm,
            dropout=config.model.dropout,
        )
        self.target_network = TargetNetwork(
            input_dim=self.obs_dim,
            output_dim=config.model.output_dim,
        )

        # Freeze target network
        for p in self.target_network.parameters():
            p.requires_grad = False

        if config.model.rnd_loss == "mse":
            self._rnd_loss = self._rnd_mse
        elif config.model.rnd_loss == "l2":
            self._rnd_loss = self._rnd_l2
        else:
            raise ValueError(f"Unsupported rnd_loss: {config.model.rnd_loss}")

    def _flatten_map(self, x: Tensor, name: str) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected {name} as [B,H,W,C], got {tuple(x.shape)}")
        b, h, w, c = x.shape
        if h != self.config.data.latent_grid_height or w != self.config.data.latent_grid_width:
            raise ValueError(
                f"Expected {name} grid "
                f"{(self.config.data.latent_grid_height, self.config.data.latent_grid_width)}, "
                f"got {(h, w)}"
            )
        if c != self.config.data.latent_channels:
            raise ValueError(f"Expected {name} latent channels {self.config.data.latent_channels}, got {c}")
        return x.reshape(b, -1)

    def build_obs_vector(
        self,
        *,
        fix_obs: Tensor | None,
        wrist_obs: Tensor | None,
        state_obs: Tensor | None,
    ) -> Tensor:
        xs = []

        if self.use_fix:
            if fix_obs is None:
                raise ValueError("fix_obs is required when use_fix_camera=True")
            xs.append(self._flatten_map(fix_obs, "fix_obs"))

        if self.use_wrist:
            if wrist_obs is None:
                raise ValueError("wrist_obs is required when use_wrist_camera=True")
            xs.append(self._flatten_map(wrist_obs, "wrist_obs"))

        if self.use_state:
            if state_obs is None:
                raise ValueError("state_obs is required when use_state=True")
            if state_obs.ndim != 2 or state_obs.shape[1] != self.config.data.state_dim:
                raise ValueError(
                    f"Expected state_obs [B,{self.config.data.state_dim}], got {tuple(state_obs.shape)}"
                )
            xs.append(state_obs)

        return torch.cat(xs, dim=-1)

    @staticmethod
    def _rnd_mse(pred: Tensor, target: Tensor) -> Tensor:
        # per-sample scalar score
        return ((pred - target) ** 2).mean(dim=-1)

    @staticmethod
    def _rnd_l2(pred: Tensor, target: Tensor) -> Tensor:
        # per-sample scalar score
        return torch.norm(pred - target, dim=-1)

    def forward_rnd(
        self,
        *,
        fix_obs: Tensor | None,
        wrist_obs: Tensor | None,
        state_obs: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
          obs_vec: [B, D]
          pred:    [B, O]
          target:  [B, O]
        """
        obs_vec = self.build_obs_vector(
            fix_obs=fix_obs,
            wrist_obs=wrist_obs,
            state_obs=state_obs,
        )

        base = self.base_network(obs_vec)
        pred = self.predictor_network(base)
        with torch.no_grad():
            target = self.target_network(base)

        return obs_vec, pred, target

    def compute_loss(
        self,
        *,
        fix_obs: Tensor | None,
        wrist_obs: Tensor | None,
        state_obs: Tensor | None,
    ) -> tuple[RNDLossOutput, dict[str, Tensor]]:
        obs_vec, pred, target = self.forward_rnd(
            fix_obs=fix_obs,
            wrist_obs=wrist_obs,
            state_obs=state_obs,
        )

        per_sample = self._rnd_loss(pred, target)   # [B]
        total = per_sample.mean()

        losses = RNDLossOutput(total=total)
        stats = {
            "loss_total": total.detach(),
            "obs_norm_mean": torch.norm(obs_vec, dim=-1).mean().detach(),
            "pred_norm_mean": torch.norm(pred, dim=-1).mean().detach(),
            "target_norm_mean": torch.norm(target, dim=-1).mean().detach(),
            "rnd_score_mean": per_sample.mean().detach(),
            "rnd_score_std": per_sample.std(unbiased=False).detach(),
        }
        return losses, stats

    @torch.no_grad()
    def compute_scores(
        self,
        *,
        fix_obs: Tensor | None,
        wrist_obs: Tensor | None,
        state_obs: Tensor | None,
    ) -> dict[str, Tensor]:
        self.eval()

        obs_vec, pred, target = self.forward_rnd(
            fix_obs=fix_obs,
            wrist_obs=wrist_obs,
            state_obs=state_obs,
        )

        scores = self._rnd_loss(pred, target)   # [B]

        return {
            "x_obs": obs_vec.detach(),
            "pred": pred.detach(),
            "target": target.detach(),
            "scores": scores.detach(),
        }

    def forward(
        self,
        *,
        fix_obs: Tensor | None,
        wrist_obs: Tensor | None,
        state_obs: Tensor | None,
    ) -> dict[str, Tensor | RNDLossOutput | dict[str, Tensor]]:
        losses, stats = self.compute_loss(
            fix_obs=fix_obs,
            wrist_obs=wrist_obs,
            state_obs=state_obs,
        )
        return {
            "loss": losses.total,
            "losses": losses,
            "stats": stats,
        }