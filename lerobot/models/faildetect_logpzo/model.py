from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchdiffeq import odeint

from .config import LogpZOConfig


def standard_normal_logprob(x: Tensor, std: float = 1.0) -> Tensor:
    var = std * std
    log_z = 0.5 * math.log(2.0 * math.pi * var)
    return -(x * x) / (2.0 * var) - log_z


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, out_dim: int | None = None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = FeedForward(
            frequency_embedding_size,
            4 * hidden_size,
            out_dim=hidden_size,
        )
        nn.init.normal_(self.mlp.net[0].weight, std=0.02)
        nn.init.normal_(self.mlp.net[3].weight, std=0.02)

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class AdaptivePreNormMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim, hidden_dim, dropout=dropout, out_dim=dim)

    def forward(self, x: Tensor, shift: Tensor, scale: Tensor, gate: Tensor) -> Tensor:
        h = self.ff(modulate(self.norm(x), shift, scale))
        return x + gate * h


class VectorFlowBackbone(nn.Module):
    def __init__(self, hidden_dim: int, depth: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AdaptivePreNormMLP(hidden_dim, mlp_dim, dropout=dropout) for _ in range(depth)]
        )
        self.modulators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(hidden_dim, 3 * hidden_dim, bias=True),
                )
                for _ in range(depth)
            ]
        )

        for mod in self.modulators:
            nn.init.constant_(mod[-1].weight, 0)
            nn.init.constant_(mod[-1].bias, 0)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        for block, mod in zip(self.blocks, self.modulators):
            shift, scale, gate = mod(t_emb).chunk(3, dim=-1)
            x = block(x, shift=shift, scale=scale, gate=gate)
        return x


@dataclass
class LogpZOLossOutput:
    total: Tensor
    flow: Tensor


class VectorTransportODEFunc:
    def __init__(self, model: "LogpZOModel"):
        self.model = model

    def __call__(self, t: Tensor, x_t: Tensor) -> Tensor:
        tau = t.expand(x_t.shape[0])
        return self.model.predict_velocity(x_t=x_t, tau=tau)


class LogpZOModel(nn.Module):
    """
    FAIL-Detect-style logpZO baseline:
      - models density of current observation O_t
      - O_t = concat(current visual latent(s), current state)
      - no action conditioning
      - no temporal history
    """

    def __init__(self, config: LogpZOConfig):
        super().__init__()
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
        d = config.model.hidden_dim

        self.time_embd = TimestepEmbedder(d)
        self.in_proj = nn.Linear(obs_dim, d, bias=True)
        self.backbone = VectorFlowBackbone(
            hidden_dim=d,
            depth=config.model.depth,
            mlp_dim=config.model.mlp_dim,
            dropout=config.model.dropout,
        )
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.out_mod = nn.Sequential(
            nn.GELU(),
            nn.Linear(d, 2 * d, bias=True),
        )
        self.out_proj = nn.Linear(d, obs_dim, bias=True)

        nn.init.constant_(self.out_mod[-1].weight, 0)
        nn.init.constant_(self.out_mod[-1].bias, 0)
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

        self.odefunc = VectorTransportODEFunc(self)

    def _flatten_map(self, x: Tensor, name: str) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected {name} as [B,H,W,C], got {tuple(x.shape)}")
        b, h, w, c = x.shape
        if h != self.config.data.latent_grid_height or w != self.config.data.latent_grid_width:
            raise ValueError(
                f"Expected {name} grid {(self.config.data.latent_grid_height, self.config.data.latent_grid_width)}, got {(h, w)}"
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

    def _sample_times(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        cfg = self.config.fm_head
        t = torch.rand(size=[batch_size], device=device, dtype=dtype)
        return cfg.min_time + (cfg.max_time - cfg.min_time) * t

    def _sample_base_noise(self, x1: Tensor) -> Tensor:
        return self.config.fm_head.base_std * torch.randn_like(x1)

    def predict_velocity(
        self,
        *,
        x_t: Tensor,
        tau: Tensor,
    ) -> Tensor:
        t_emb = self.time_embd(tau)
        h = self.in_proj(x_t)
        h = self.backbone(h, t_emb)

        shift, scale = self.out_mod(t_emb).chunk(2, dim=-1)
        h = modulate(self.out_norm(h), shift, scale)
        return self.out_proj(h)

    def compute_loss(
        self,
        *,
        fix_obs: Tensor | None,
        wrist_obs: Tensor | None,
        state_obs: Tensor | None,
    ) -> tuple[LogpZOLossOutput, dict[str, Tensor]]:
        x1 = self.build_obs_vector(
            fix_obs=fix_obs,
            wrist_obs=wrist_obs,
            state_obs=state_obs,
        )

        b = x1.shape[0]
        tau = self._sample_times(b, x1.device, x1.dtype)
        x0 = self._sample_base_noise(x1)
        x_t = (1.0 - tau[:, None]) * x0 + tau[:, None] * x1
        target_v = x1 - x0

        pred_v = self.predict_velocity(x_t=x_t, tau=tau)
        flow_loss = torch.nn.functional.mse_loss(pred_v, target_v)
        total = self.config.loss.flow_weight * flow_loss

        losses = LogpZOLossOutput(total=total, flow=flow_loss)
        stats = {
            "loss_total": total.detach(),
            "flow_loss": flow_loss.detach(),
            "x1_norm_mean": torch.norm(x1, dim=-1).mean().detach(),
            "x0_norm_mean": torch.norm(x0, dim=-1).mean().detach(),
            "xt_norm_mean": torch.norm(x_t, dim=-1).mean().detach(),
            "pred_v_norm_mean": torch.norm(pred_v, dim=-1).mean().detach(),
            "target_v_norm_mean": torch.norm(target_v, dim=-1).mean().detach(),
        }
        return losses, stats

    @torch.no_grad()
    def transport_observed_to_base(self, x_obs: Tensor) -> Tensor:
        self.eval()
        t = torch.tensor([1.0, 0.0], device=x_obs.device, dtype=x_obs.dtype)
        traj = odeint(
            self.odefunc,
            x_obs,
            t,
            rtol=self.config.fm_head.ode_rtol,
            atol=self.config.fm_head.ode_atol,
            method=self.config.fm_head.ode_solver,
        )
        return traj[-1]

    @torch.no_grad()
    def transport_observed_one_step_t1(self, x_obs: Tensor) -> Tensor:
        self.eval()
        tau1 = torch.ones(x_obs.shape[0], device=x_obs.device, dtype=x_obs.dtype)
        v1 = self.predict_velocity(x_t=x_obs, tau=tau1)
        return x_obs - v1

    @torch.no_grad()
    def transport_observed_for_scoring(self, x_obs: Tensor) -> Tensor:
        mode = self.config.fm_head.score_transport_mode
        if mode == "ode":
            return self.transport_observed_to_base(x_obs)
        if mode == "one_step_t0":
            return self.transport_observed_one_step_t1(x_obs)
        raise ValueError(f"Unsupported score_transport_mode: {mode}")

    def score_from_base(self, eps_hat: Tensor) -> tuple[Tensor, Tensor]:
        log_prob = standard_normal_logprob(eps_hat, std=self.config.fm_head.base_std).sum(dim=-1)

        if self.config.fm_head.score_kind == "base_logp":
            scores = -log_prob
        elif self.config.fm_head.score_kind == "squared_norm":
            scores = ((eps_hat / self.config.fm_head.base_std) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported score_kind: {self.config.fm_head.score_kind}")

        return log_prob, scores

    @torch.no_grad()
    def compute_nonconformity_scores(
        self,
        *,
        fix_obs: Tensor | None,
        wrist_obs: Tensor | None,
        state_obs: Tensor | None,
    ) -> dict[str, Tensor]:
        self.eval()
        x_obs = self.build_obs_vector(
            fix_obs=fix_obs,
            wrist_obs=wrist_obs,
            state_obs=state_obs,
        )
        eps_hat = self.transport_observed_for_scoring(x_obs)
        log_prob, scores = self.score_from_base(eps_hat)

        return {
            "x_obs": x_obs.detach(),
            "eps_hat": eps_hat.detach(),
            "log_prob": log_prob.detach(),
            "scores": scores.detach(),
        }

    def forward(
        self,
        *,
        fix_obs: Tensor | None,
        wrist_obs: Tensor | None,
        state_obs: Tensor | None,
    ) -> dict[str, Tensor | LogpZOLossOutput | dict[str, Tensor]]:
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