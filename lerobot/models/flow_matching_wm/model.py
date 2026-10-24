from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchdiffeq import odeint

from .config import FlowMatchingWorldModelConfig


def standard_normal_logprob(x: Tensor, std: float = 1.0) -> Tensor:
    var = std * std
    log_z = 0.5 * math.log(2.0 * math.pi * var)
    return -(x * x) / (2.0 * var) - log_z


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def rot6d_to_rotmat(x: Tensor) -> Tensor:
    """
    x: [..., 6]
    returns: [..., 3, 3], where columns are the orthonormalized first two
             input vectors plus their cross product.
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - a2_proj, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def rotmat_to_rot6d(R: Tensor) -> Tensor:
    """
    R: [..., 3, 3]
    returns: [..., 6]
    """
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    return torch.cat([c0, c1], dim=-1)


def compose_action_chunk_raw(action_chunk: Tensor) -> Tensor:
    """
    Compose an action chunk into one summary action.

    Assumes action format starts with:
      dp(3) + rot6d(6) + gripper(1) + optional extras

    Summary rule:
      - dp_total = sum(dp_t)
      - rot_total = compose relative rotations in temporal order
      - gripper   = last value
      - extras    = last value(s), if present

    action_chunk: [B, K, A]
    returns:      [B, A]
    """
    if action_chunk.ndim != 3:
        raise ValueError(f"Expected action_chunk [B,K,A], got {tuple(action_chunk.shape)}")

    b, k, a = action_chunk.shape
    if a < 10:
        raise ValueError(
            "compose_raw_chunk currently requires action_dim >= 10 "
            "(dp(3) + rot6d(6) + gripper(1))."
        )

    dp_total = action_chunk[:, :, 0:3].sum(dim=1)  # [B,3]

    rot6_seq = action_chunk[:, :, 3:9]             # [B,K,6]
    R_seq = rot6d_to_rotmat(rot6_seq)              # [B,K,3,3]

    R_total = torch.eye(3, device=action_chunk.device, dtype=action_chunk.dtype).unsqueeze(0).repeat(b, 1, 1)
    for i in range(k):
        R_total = R_total @ R_seq[:, i]

    rot6_total = rotmat_to_rot6d(R_total)          # [B,6]
    g_last = action_chunk[:, -1, 9:10]             # [B,1]

    if a > 10:
        extra_last = action_chunk[:, -1, 10:]      # [B,A-10]
        return torch.cat([dp_total, rot6_total, g_last, extra_last], dim=-1)

    return torch.cat([dp_total, rot6_total, g_last], dim=-1)


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


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal

        self.qkv_attn = nn.Linear(embed_dim, embed_dim * 3)
        self.project = nn.Linear(embed_dim, embed_dim)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.size()
        qkv = self.qkv_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        q = q.view(b, n, self.num_heads, self.head_dim)
        k = k.view(b, n, self.num_heads, self.head_dim)
        v = v.view(b, n, self.num_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, n, c)
        y = self.project(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.q_dim = q_dim
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.out_proj = nn.Linear(q_dim, q_dim)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, q_in: Tensor, kv_in: Tensor) -> Tensor:
        b, nq, _ = q_in.shape
        nk = kv_in.shape[1]

        q = self.q_proj(q_in).view(b, nq, self.num_heads, self.head_dim)
        k = self.k_proj(kv_in).view(b, nk, self.num_heads, self.head_dim)
        v = self.v_proj(kv_in).view(b, nk, self.num_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(b, nq, self.q_dim)
        return self.out_proj(y)


class AdaptivePreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fn = fn

    def forward(self, x: Tensor, scale: Tensor, shift: Tensor, gate: Tensor, **kwargs) -> Tensor:
        h = self.fn(modulate(self.norm(x), shift, scale), **kwargs)
        return gate * h


class CrossPreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, kv: Tensor) -> Tensor:
        return self.fn(self.q_norm(x), kv)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = FeedForward(
            frequency_embedding_size,
            4 * hidden_size,
            out_dim=hidden_size,
        )
        nn.init.normal_(self.mlp.net[0].weight, std=0.02)
        nn.init.normal_(self.mlp.net[3].weight, std=0.02)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class AddBroadcastPosEmbed(nn.Module):
    def __init__(self, shape: tuple[int, int, int], embd_dim: int):
        super().__init__()
        self.shape = shape
        self.n_dim = 3
        self.embd_dim = embd_dim
        assert embd_dim % self.n_dim == 0
        self.emb = nn.ParameterDict(
            {
                f"d_{i}": nn.Parameter(torch.randn(self.shape[i], embd_dim // self.n_dim) * 0.02)
                for i in range(self.n_dim)
            }
        )

    def forward(self, x: Tensor) -> Tensor:
        b, t, h, w, c = x.shape
        shape = (t, h, w)
        embs = []
        for i in range(self.n_dim):
            e = self.emb[f"d_{i}"]
            if i == 0:
                e = e[:t, ...]
            e = e.view(1, *((1,) * i), shape[i], *((1,) * (self.n_dim - i - 1)), -1)
            e = e.expand(1, *shape, -1)
            embs.append(e)
        embs = torch.cat(embs, dim=-1)
        return x + embs


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.linear(x)


class SeparableActionConditionedBackbone(nn.Module):
    def __init__(self, config: FlowMatchingWorldModelConfig):
        super().__init__()
        d = config.backbone.hidden_dim
        self.depth = config.backbone.depth
        self.use_cross_attention = config.action_conditioner.use_cross_attention

        self.scale_shift = nn.Parameter(torch.randn(9, d) / (d**0.5))
        self.layers = nn.ModuleList([])
        for _ in range(config.backbone.depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "temp_attn": AdaptivePreNorm(
                            d,
                            Attention(
                                d,
                                config.backbone.heads,
                                dropout=config.backbone.dropout,
                                causal=False,
                            ),
                        ),
                        "spatial_attn": AdaptivePreNorm(
                            d,
                            Attention(
                                d,
                                config.backbone.heads,
                                dropout=config.backbone.dropout,
                                causal=False,
                            ),
                        ),
                        "action_cross": CrossPreNorm(
                            d,
                            CrossAttention(
                                d,
                                d,
                                num_heads=config.backbone.heads,
                                dropout=config.backbone.dropout,
                            ),
                        ),
                        "ff": AdaptivePreNorm(
                            d,
                            FeedForward(d, config.backbone.mlp_dim, dropout=config.backbone.dropout),
                        ),
                    }
                )
            )

    def forward(self, x: Tensor, mod_cond: Tensor, action_tokens: Tensor | None) -> Tensor:
        # x: [B, T, H, W, D]
        b, t, h, w, d = x.shape
        time_mod = mod_cond.reshape(b, 9, -1)
        (
            temp_shift_msa,
            temp_scale_msa,
            temp_gate_msa,
            spatial_shift_msa,
            spatial_scale_msa,
            spatial_gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (self.scale_shift[None] + time_mod).chunk(9, dim=1)

        x = einops.rearrange(x, "b t h w c -> b (t h w) c")
        for layer in self.layers:
            xt = einops.rearrange(x, "b (t h w) c -> (b h w) t c", b=b, t=t, h=h, w=w)
            reps = h * w
            xt = xt + layer["temp_attn"](
                xt,
                scale=einops.repeat(temp_scale_msa, "b 1 d -> (b r) 1 d", r=reps),
                shift=einops.repeat(temp_shift_msa, "b 1 d -> (b r) 1 d", r=reps),
                gate=einops.repeat(temp_gate_msa, "b 1 d -> (b r) 1 d", r=reps),
            )

            xs = einops.rearrange(xt, "(b h w) t c -> (b t) (h w) c", b=b, t=t, h=h, w=w)
            xs = xs + layer["spatial_attn"](
                xs,
                scale=einops.repeat(spatial_scale_msa, "b 1 d -> (b t) 1 d", t=t),
                shift=einops.repeat(spatial_shift_msa, "b 1 d -> (b t) 1 d", t=t),
                gate=einops.repeat(spatial_gate_msa, "b 1 d -> (b t) 1 d", t=t),
            )
            x = einops.rearrange(xs, "(b t) (h w) c -> b (t h w) c", b=b, t=t, h=h, w=w)

            if self.use_cross_attention and action_tokens is not None:
                x = x + layer["action_cross"](x, action_tokens)

            x = x + layer["ff"](x, scale=scale_mlp, shift=shift_mlp, gate=gate_mlp)

        return einops.rearrange(x, "b (t h w) c -> b t h w c", b=b, t=t, h=h, w=w)


class ActionChunkEncoder(nn.Module):
    """
    Action tokenizer / action conditioner token path.
    Produces one token per action step in the chunk.
    """
    def __init__(self, config: FlowMatchingWorldModelConfig):
        super().__init__()
        d = config.action_conditioner.token_dim
        self.in_proj = nn.Linear(config.action_conditioner.action_dim, d)
        self.pos = nn.Embedding(config.data.action_chunk_len, d)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=config.action_conditioner.num_heads,
                    dim_feedforward=config.action_conditioner.hidden_dim,
                    dropout=config.action_conditioner.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.action_conditioner.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, action_chunk: Tensor) -> Tensor:
        _, k, _ = action_chunk.shape
        pos = self.pos(torch.arange(k, device=action_chunk.device)).unsqueeze(0)
        x = self.in_proj(action_chunk) + pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class ActionGlobalConditioner(nn.Module):
    """
    Global action embedding for adaLN-style modulation.
    Supports:
      - mean_pool_tokens
      - compose_raw_chunk
    """
    def __init__(self, config: FlowMatchingWorldModelConfig):
        super().__init__()
        self.config = config
        self.use_action_adaln = config.action_conditioner.use_action_adaln
        self.mode = config.action_conditioner.global_pool_mode
        d = config.action_conditioner.global_cond_dim
        h = config.action_conditioner.global_hidden_dim

        self.token_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, d),
        )
        self.raw_proj = nn.Sequential(
            nn.LayerNorm(config.action_conditioner.action_dim),
            nn.Linear(config.action_conditioner.action_dim, h),
            nn.GELU(),
            nn.Linear(h, d),
        )

    def forward(self, action_chunk: Tensor, action_tokens: Tensor | None) -> Tensor | None:
        if not self.use_action_adaln:
            return None

        if self.mode == "mean_pool_tokens":
            if action_tokens is None:
                raise ValueError("action_tokens required for global_pool_mode='mean_pool_tokens'")
            pooled = action_tokens.mean(dim=1)
            return self.token_proj(pooled)

        if self.mode == "compose_raw_chunk":
            summary = compose_action_chunk_raw(action_chunk)
            return self.raw_proj(summary)

        raise ValueError(f"Unsupported global_pool_mode: {self.mode}")


class StateConditioner(nn.Module):
    def __init__(self, config: FlowMatchingWorldModelConfig):
        super().__init__()
        d = config.state_conditioner.token_dim
        self.use_state = config.data.use_state and config.state_conditioner.use_state_tokens
        if self.use_state:
            self.proj = nn.Sequential(
                nn.LayerNorm(config.data.state_dim),
                nn.Linear(config.data.state_dim, config.state_conditioner.hidden_dim),
                nn.GELU(),
                nn.Linear(config.state_conditioner.hidden_dim, d),
            )
        else:
            self.proj = None

    def forward(self, state_hist: Tensor | None, target_slots: int) -> Tensor | None:
        if not self.use_state:
            return None
        if state_hist is None:
            raise ValueError("state_hist required when state conditioning is enabled")
        hist = self.proj(state_hist)  # [B,T,D]
        last = hist[:, -1:, :].expand(-1, target_slots, -1)
        return torch.cat([hist, last], dim=1)  # [B,T+target_slots,D]


class SingleViewFMBranch(nn.Module):
    """
    FM-WM visual branch, accepting one view or a channel-concatenated pair.
    This is the old single-view visual path, with optional action adaLN coming
    in through the shared modulation condition.
    """
    def __init__(self, config: FlowMatchingWorldModelConfig, *, latent_channels: int | None = None):
        super().__init__()
        c_in = config.data.latent_channels if latent_channels is None else latent_channels
        self.latent_channels = c_in
        d = config.backbone.hidden_dim
        self.config = config

        self.fc_in = nn.Linear(c_in, d, bias=False)
        self.pos_embd = AddBroadcastPosEmbed(
            shape=(config.data.history_len + 1, config.data.latent_grid_height, config.data.latent_grid_width),
            embd_dim=d,
        )
        self.modulation = FeedForward(d, 4 * d, out_dim=9 * d)
        self.backbone = SeparableActionConditionedBackbone(config)
        self.fc_out = FinalLayer(d, c_in)

        nn.init.constant_(self.modulation.net[3].weight, 0)
        nn.init.constant_(self.modulation.net[3].bias, 0)
        nn.init.constant_(self.fc_out.linear.weight, 0)
        nn.init.constant_(self.fc_out.linear.bias, 0)
        nn.init.constant_(self.fc_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.fc_out.adaLN_modulation[-1].bias, 0)

        self.aux_head = None
        if config.aux_head.use_one_step_aux_head:
            self.aux_head = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, c_in),
            )

    def _check_latent_map(self, x: Tensor, name: str = "tensor") -> None:
        if x.ndim != 5:
            raise ValueError(f"Expected {name} as [B,T,H,W,C], got {tuple(x.shape)}")
        _, _, h, w, c = x.shape
        if h != self.config.data.latent_grid_height or w != self.config.data.latent_grid_width:
            raise ValueError(
                f"Expected {name} grid {(self.config.data.latent_grid_height, self.config.data.latent_grid_width)}, got {(h, w)}"
            )
        if c != self.latent_channels:
            raise ValueError(f"Expected {name} latent channels {self.latent_channels}, got {c}")

    def _check_single_map(self, x: Tensor, name: str = "tensor") -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected {name} as [B,H,W,C], got {tuple(x.shape)}")
        _, h, w, c = x.shape
        if h != self.config.data.latent_grid_height or w != self.config.data.latent_grid_width:
            raise ValueError(
                f"Expected {name} grid {(self.config.data.latent_grid_height, self.config.data.latent_grid_width)}, got {(h, w)}"
            )
        if c != self.latent_channels:
            raise ValueError(f"Expected {name} latent channels {self.latent_channels}, got {c}")

    def _build_input_sequence(
        self,
        view_hist: Tensor,
        state_seq: Tensor | None,
        x_t: Tensor,
    ) -> Tensor:
        self._check_latent_map(view_hist, "view_hist")

        if x_t.ndim == 5:
            if x_t.shape[1] != 1:
                raise ValueError(f"Expected x_t shape [B,1,H,W,C], got {tuple(x_t.shape)}")
            x_t = x_t[:, 0]
        self._check_single_map(x_t, "x_t")

        b, th, h, w, _ = view_hist.shape

        x = torch.cat([view_hist, x_t.unsqueeze(1)], dim=1)  # [B,T+1,H,W,C]
        x = self.fc_in(x)

        if state_seq is not None:
            state_b = state_seq[:, :, None, None, :].expand(-1, -1, h, w, -1)
            x = x + state_b

        x = self.pos_embd(x)
        return x

    def predict_velocity(
        self,
        *,
        view_hist: Tensor,
        state_seq: Tensor | None,
        action_tokens: Tensor | None,
        mod_cond: Tensor,
        x_t: Tensor,
    ) -> Tensor:
        x = self._build_input_sequence(view_hist, state_seq, x_t)
        mod = self.modulation(mod_cond)

        h = self.backbone(x, mod, action_tokens)
        h = einops.rearrange(h, "b t h w c -> b (t h w) c")
        v = self.fc_out(h, mod_cond)
        v = einops.rearrange(
            v,
            "b (t h w) c -> b t h w c",
            t=view_hist.shape[1] + 1,
            h=self.config.data.latent_grid_height,
            w=self.config.data.latent_grid_width,
        )
        return v[:, -1, ...]

    def predict_next_map(
        self,
        *,
        view_hist: Tensor,
        state_seq: Tensor | None,
        action_tokens: Tensor | None,
        mod_cond: Tensor,
    ) -> Tensor:
        if self.aux_head is None:
            raise RuntimeError("Auxiliary head is disabled")

        zero_xt = torch.zeros(
            view_hist.shape[0],
            self.config.data.latent_grid_height,
            self.config.data.latent_grid_width,
            self.latent_channels,
            device=view_hist.device,
            dtype=view_hist.dtype,
        )
        x = self._build_input_sequence(view_hist, state_seq, zero_xt)
        mod = self.modulation(mod_cond)

        h = self.backbone(x, mod, action_tokens)
        h_last_ctx = h[:, -2, ...]
        return self.aux_head(h_last_ctx)


@dataclass
class FMWMLossOutput:
    total: Tensor
    flow: Tensor
    aux_one_step: Tensor | None


class ConditionalTransportODEFunc:
    def __init__(self, model: "FlowMatchingWorldModel"):
        self.model = model
        self.fix_hist: Tensor | None = None
        self.wrist_hist: Tensor | None = None
        self.state_hist: Tensor | None = None
        self.action_chunk: Tensor | None = None

    def set_context(
        self,
        fix_hist: Tensor | None,
        wrist_hist: Tensor | None,
        state_hist: Tensor | None,
        action_chunk: Tensor,
    ) -> None:
        self.fix_hist = fix_hist
        self.wrist_hist = wrist_hist
        self.state_hist = state_hist
        self.action_chunk = action_chunk

    def __call__(self, t: Tensor, x_t: Tensor) -> Tensor:
        if self.action_chunk is None:
            raise RuntimeError("ODE context not set")
        tau = t.expand(x_t.shape[0])
        return self.model.predict_velocity(
            fix_hist=self.fix_hist,
            wrist_hist=self.wrist_hist,
            state_hist=self.state_hist,
            action_chunk=self.action_chunk,
            x_t=x_t,
            tau=tau,
        )


class FlowMatchingWorldModel(nn.Module):
    def __init__(self, config: FlowMatchingWorldModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.use_fix = config.data.use_fix_camera
        self.use_wrist = config.data.use_wrist_camera
        self.concat_views = config.backbone.view_fusion == "concat_channels"

        self.time_embd = TimestepEmbedder(hidden_size=config.backbone.hidden_dim)
        self.state_conditioner = StateConditioner(config)
        self.action_encoder = ActionChunkEncoder(config)
        self.action_global_conditioner = ActionGlobalConditioner(config)

        self.fix_branch = SingleViewFMBranch(config) if self.use_fix and not self.concat_views else None
        self.wrist_branch = SingleViewFMBranch(config) if self.use_wrist and not self.concat_views else None
        self.joint_branch = (
            SingleViewFMBranch(config, latent_channels=2 * config.data.latent_channels)
            if self.concat_views else None
        )

        self.odefunc = ConditionalTransportODEFunc(self)

    def _num_views(self) -> int:
        return int(self.use_fix) + int(self.use_wrist)

    def _concat_views(self, fix: Tensor | None, wrist: Tensor | None, *, history: bool = False) -> Tensor:
        """Fixed channels first, wrist channels second; dataset channels stay C."""
        if fix is None or wrist is None:
            raise ValueError("concat_channels requires both fixed and wrist tensors")
        shape = (self.config.data.latent_grid_height, self.config.data.latent_grid_width,
                 self.config.data.latent_channels)
        if fix.ndim != (5 if history else 4) or fix.shape != wrist.shape or tuple(fix.shape[-3:]) != shape:
            raise ValueError(f"Expected matching {'history' if history else 'endpoint'} maps ending in {shape}")
        if history and fix.shape[1] != self.config.data.history_len:
            raise ValueError("Concatenated history length differs from data.history_len")
        return torch.cat((fix, wrist), dim=-1)

    def _unpack_joint_map(self, joint: Tensor) -> Tensor:
        """Public ODE/scoring layout remains [B,2,H,W,C], fixed then wrist."""
        fix, wrist = joint.split(self.config.data.latent_channels, dim=-1)
        return torch.stack((fix, wrist), dim=1)

    def _sample_times(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        cfg = self.config.fm_head
        t = torch.rand(size=[batch_size], device=device, dtype=dtype)
        return cfg.min_time + (cfg.max_time - cfg.min_time) * t

    def _sample_base_noise(self, x1: Tensor) -> Tensor:
        return self.config.fm_head.base_std * torch.randn_like(x1)

    def _shared_condition(
        self,
        action_chunk: Tensor,
        tau: Tensor,
        state_hist: Tensor | None,
    ) -> tuple[Tensor | None, Tensor, Tensor | None]:
        action_tokens = self.action_encoder(action_chunk)
        time_cond = self.time_embd(tau)

        action_global = self.action_global_conditioner(action_chunk, action_tokens)
        if action_global is None:
            mod_cond = time_cond
        else:
            mod_cond = time_cond + action_global

        state_seq = self.state_conditioner(state_hist, target_slots=1)
        return action_tokens, mod_cond, state_seq

    def _split_xt(
        self,
        x_t: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.use_fix and self.use_wrist:
            if x_t.ndim != 5 or x_t.shape[1] != 2:
                raise ValueError(f"Expected x_t [B,2,H,W,C] for dual-view mode, got {tuple(x_t.shape)}")
            return x_t[:, 0], x_t[:, 1]
        if self.use_fix:
            if x_t.ndim == 5 and x_t.shape[1] == 1:
                return x_t[:, 0], None
            if x_t.ndim == 4:
                return x_t, None
            raise ValueError(f"Expected x_t [B,H,W,C] or [B,1,H,W,C] for fix-only mode, got {tuple(x_t.shape)}")
        if self.use_wrist:
            if x_t.ndim == 5 and x_t.shape[1] == 1:
                return None, x_t[:, 0]
            if x_t.ndim == 4:
                return None, x_t
            raise ValueError(f"Expected x_t [B,H,W,C] or [B,1,H,W,C] for wrist-only mode, got {tuple(x_t.shape)}")
        raise RuntimeError("No enabled views")

    def _stack_enabled_views(
        self,
        fix_tensor: Tensor | None,
        wrist_tensor: Tensor | None,
    ) -> Tensor:
        xs = []
        if self.use_fix:
            if fix_tensor is None:
                raise ValueError("fix_tensor is required when use_fix_camera=True")
            xs.append(fix_tensor)
        if self.use_wrist:
            if wrist_tensor is None:
                raise ValueError("wrist_tensor is required when use_wrist_camera=True")
            xs.append(wrist_tensor)

        if len(xs) == 1:
            return xs[0]
        return torch.stack(xs, dim=1)

    def predict_velocity(
        self,
        *,
        fix_hist: Tensor | None,
        wrist_hist: Tensor | None,
        state_hist: Tensor | None,
        action_chunk: Tensor,
        x_t: Tensor,
        tau: Tensor,
    ) -> Tensor:
        action_tokens, mod_cond, state_seq = self._shared_condition(action_chunk, tau, state_hist)
        x_fix, x_wrist = self._split_xt(x_t)

        if self.joint_branch is not None:
            joint = self.joint_branch.predict_velocity(
                view_hist=self._concat_views(fix_hist, wrist_hist, history=True),
                state_seq=state_seq, action_tokens=action_tokens, mod_cond=mod_cond,
                x_t=self._concat_views(x_fix, x_wrist),
            )
            return self._unpack_joint_map(joint)

        v_fix = None
        v_wrist = None

        if self.use_fix:
            if self.fix_branch is None or fix_hist is None or x_fix is None:
                raise ValueError("fix branch inputs missing")
            v_fix = self.fix_branch.predict_velocity(
                view_hist=fix_hist,
                state_seq=state_seq,
                action_tokens=action_tokens,
                mod_cond=mod_cond,
                x_t=x_fix,
            )

        if self.use_wrist:
            if self.wrist_branch is None or wrist_hist is None or x_wrist is None:
                raise ValueError("wrist branch inputs missing")
            v_wrist = self.wrist_branch.predict_velocity(
                view_hist=wrist_hist,
                state_seq=state_seq,
                action_tokens=action_tokens,
                mod_cond=mod_cond,
                x_t=x_wrist,
            )

        return self._stack_enabled_views(v_fix, v_wrist)

    def predict_next_map(
        self,
        *,
        fix_hist: Tensor | None,
        wrist_hist: Tensor | None,
        state_hist: Tensor | None,
        action_chunk: Tensor,
    ) -> Tensor:
        tau0 = torch.zeros(action_chunk.shape[0], device=action_chunk.device, dtype=action_chunk.dtype)
        action_tokens, mod_cond, state_seq = self._shared_condition(action_chunk, tau0, state_hist)

        if self.joint_branch is not None:
            joint = self.joint_branch.predict_next_map(
                view_hist=self._concat_views(fix_hist, wrist_hist, history=True),
                state_seq=state_seq, action_tokens=action_tokens, mod_cond=mod_cond,
            )
            return self._unpack_joint_map(joint)

        next_fix = None
        next_wrist = None

        if self.use_fix:
            if self.fix_branch is None or fix_hist is None:
                raise ValueError("fix branch inputs missing")
            next_fix = self.fix_branch.predict_next_map(
                view_hist=fix_hist,
                state_seq=state_seq,
                action_tokens=action_tokens,
                mod_cond=mod_cond,
            )

        if self.use_wrist:
            if self.wrist_branch is None or wrist_hist is None:
                raise ValueError("wrist branch inputs missing")
            next_wrist = self.wrist_branch.predict_next_map(
                view_hist=wrist_hist,
                state_seq=state_seq,
                action_tokens=action_tokens,
                mod_cond=mod_cond,
            )

        return self._stack_enabled_views(next_fix, next_wrist)

    def compute_loss(
        self,
        *,
        fix_hist: Tensor | None,
        action_chunk: Tensor,
        future_fix: Tensor | None,
        wrist_hist: Tensor | None = None,
        future_wrist: Tensor | None = None,
        state_hist: Tensor | None = None,
        future_state: Tensor | None = None,
        next_fix: Tensor | None = None,
        next_wrist: Tensor | None = None,
        next_state: Tensor | None = None,
    ) -> tuple[FMWMLossOutput, dict[str, Tensor]]:
        _ = future_state, next_state

        if self.joint_branch is not None:
            return self._compute_joint_loss(
                fix_hist=fix_hist, wrist_hist=wrist_hist, state_hist=state_hist,
                action_chunk=action_chunk, future_fix=future_fix, future_wrist=future_wrist,
                next_fix=next_fix, next_wrist=next_wrist,
            )

        b = action_chunk.shape[0]
        tau = self._sample_times(b, action_chunk.device, action_chunk.dtype)
        action_tokens, mod_cond, state_seq = self._shared_condition(action_chunk, tau, state_hist)

        flow_losses = []
        aux_losses = []

        x1_fix = x0_fix = xt_fix = pred_v_fix = target_v_fix = None
        x1_wrist = x0_wrist = xt_wrist = pred_v_wrist = target_v_wrist = None

        if self.use_fix:
            if self.fix_branch is None or fix_hist is None or future_fix is None:
                raise ValueError("fix branch training inputs missing")
            self.fix_branch._check_latent_map(fix_hist, "fix_hist")
            self.fix_branch._check_single_map(future_fix, "future_fix")

            x1_fix = future_fix
            x0_fix = self._sample_base_noise(x1_fix)
            xt_fix = (1.0 - tau[:, None, None, None]) * x0_fix + tau[:, None, None, None] * x1_fix
            target_v_fix = x1_fix - x0_fix

            pred_v_fix = self.fix_branch.predict_velocity(
                view_hist=fix_hist,
                state_seq=state_seq,
                action_tokens=action_tokens,
                mod_cond=mod_cond,
                x_t=xt_fix,
            )
            flow_fix = F.mse_loss(pred_v_fix, target_v_fix)
            flow_losses.append(flow_fix)

            if self.fix_branch.aux_head is not None and next_fix is not None:
                self.fix_branch._check_single_map(next_fix, "next_fix")
                pred_next_fix = self.fix_branch.predict_next_map(
                    view_hist=fix_hist,
                    state_seq=state_seq,
                    action_tokens=action_tokens,
                    mod_cond=mod_cond,
                )
                aux_fix = F.mse_loss(pred_next_fix, next_fix)
                aux_losses.append(aux_fix)
            else:
                aux_fix = None
        else:
            flow_fix = None
            aux_fix = None

        if self.use_wrist:
            if self.wrist_branch is None or wrist_hist is None or future_wrist is None:
                raise ValueError("wrist branch training inputs missing")
            self.wrist_branch._check_latent_map(wrist_hist, "wrist_hist")
            self.wrist_branch._check_single_map(future_wrist, "future_wrist")

            x1_wrist = future_wrist
            x0_wrist = self._sample_base_noise(x1_wrist)
            xt_wrist = (1.0 - tau[:, None, None, None]) * x0_wrist + tau[:, None, None, None] * x1_wrist
            target_v_wrist = x1_wrist - x0_wrist

            pred_v_wrist = self.wrist_branch.predict_velocity(
                view_hist=wrist_hist,
                state_seq=state_seq,
                action_tokens=action_tokens,
                mod_cond=mod_cond,
                x_t=xt_wrist,
            )
            flow_wrist = F.mse_loss(pred_v_wrist, target_v_wrist)
            flow_losses.append(flow_wrist)

            if self.wrist_branch.aux_head is not None and next_wrist is not None:
                self.wrist_branch._check_single_map(next_wrist, "next_wrist")
                pred_next_wrist = self.wrist_branch.predict_next_map(
                    view_hist=wrist_hist,
                    state_seq=state_seq,
                    action_tokens=action_tokens,
                    mod_cond=mod_cond,
                )
                aux_wrist = F.mse_loss(pred_next_wrist, next_wrist)
                aux_losses.append(aux_wrist)
            else:
                aux_wrist = None
        else:
            flow_wrist = None
            aux_wrist = None

        if not flow_losses:
            raise RuntimeError("No enabled view produced a flow loss")

        flow_loss = sum(flow_losses)
        total = self.config.loss.flow_weight * flow_loss

        aux_loss = None
        if aux_losses:
            aux_loss = sum(aux_losses)
            total = total + self.config.aux_head.one_step_loss_weight * aux_loss

        x1 = self._stack_enabled_views(x1_fix, x1_wrist)
        x0 = self._stack_enabled_views(x0_fix, x0_wrist)
        xt = self._stack_enabled_views(xt_fix, xt_wrist)
        pred_v = self._stack_enabled_views(pred_v_fix, pred_v_wrist)
        target_v = self._stack_enabled_views(target_v_fix, target_v_wrist)

        losses = FMWMLossOutput(total=total, flow=flow_loss, aux_one_step=aux_loss)
        stats = {
            "loss_total": total.detach(),
            "flow_loss": flow_loss.detach(),
            "x1_norm_mean": torch.norm(x1.flatten(1), dim=-1).mean().detach(),
            "x0_norm_mean": torch.norm(x0.flatten(1), dim=-1).mean().detach(),
            "xt_norm_mean": torch.norm(xt.flatten(1), dim=-1).mean().detach(),
            "pred_v_norm_mean": torch.norm(pred_v.flatten(1), dim=-1).mean().detach(),
            "target_v_norm_mean": torch.norm(target_v.flatten(1), dim=-1).mean().detach(),
        }
        if flow_fix is not None:
            stats["flow_fix_loss"] = flow_fix.detach()
        if flow_wrist is not None:
            stats["flow_wrist_loss"] = flow_wrist.detach()
        if aux_fix is not None:
            stats["aux_fix_loss"] = aux_fix.detach()
        if aux_wrist is not None:
            stats["aux_wrist_loss"] = aux_wrist.detach()
        if aux_loss is not None:
            stats["aux_one_step_loss"] = aux_loss.detach()

        return losses, stats

    def _compute_joint_loss(
        self, *, fix_hist: Tensor | None, wrist_hist: Tensor | None,
        state_hist: Tensor | None, action_chunk: Tensor,
        future_fix: Tensor | None, future_wrist: Tensor | None,
        next_fix: Tensor | None, next_wrist: Tensor | None,
    ) -> tuple[FMWMLossOutput, dict[str, Tensor]]:
        """One flow/auxiliary branch; retain SUM of the two per-view MSEs.

        Averaging over all 2C channels alone would halve the historical loss
        and change gradient/clipping scale. Per-view diagnostics remain useful,
        but both channel groups are jointly conditioned on both cameras.
        """
        branch = self.joint_branch
        assert branch is not None
        history = self._concat_views(fix_hist, wrist_hist, history=True)
        x1 = self._concat_views(future_fix, future_wrist)
        tau = self._sample_times(action_chunk.shape[0], action_chunk.device, action_chunk.dtype)
        action_tokens, mod_cond, state_seq = self._shared_condition(action_chunk, tau, state_hist)
        x0 = self._sample_base_noise(x1)
        xt = (1 - tau[:, None, None, None]) * x0 + tau[:, None, None, None] * x1
        target = x1 - x0
        prediction = branch.predict_velocity(
            view_hist=history, state_seq=state_seq, action_tokens=action_tokens,
            mod_cond=mod_cond, x_t=xt,
        )
        channels = self.config.data.latent_channels
        flow_fix = F.mse_loss(prediction[..., :channels], target[..., :channels])
        flow_wrist = F.mse_loss(prediction[..., channels:], target[..., channels:])
        flow = flow_fix + flow_wrist
        total = self.config.loss.flow_weight * flow
        aux = None
        aux_fix = aux_wrist = None
        if branch.aux_head is not None and (next_fix is not None or next_wrist is not None):
            next_target = self._concat_views(next_fix, next_wrist)
            next_prediction = branch.predict_next_map(
                view_hist=history, state_seq=state_seq, action_tokens=action_tokens, mod_cond=mod_cond,
            )
            aux_fix = F.mse_loss(next_prediction[..., :channels], next_target[..., :channels])
            aux_wrist = F.mse_loss(next_prediction[..., channels:], next_target[..., channels:])
            aux = aux_fix + aux_wrist
            total = total + self.config.aux_head.one_step_loss_weight * aux
        stats = {"loss_total": total.detach(), "flow_loss": flow.detach(),
                 "flow_fix_loss": flow_fix.detach(), "flow_wrist_loss": flow_wrist.detach()}
        for name, tensor in (("x1", x1), ("x0", x0), ("xt", xt), ("pred_v", prediction), ("target_v", target)):
            stats[name + "_norm_mean"] = torch.norm(tensor.flatten(1), dim=-1).mean().detach()
        if aux is not None:
            stats.update(aux_one_step_loss=aux.detach(), aux_fix_loss=aux_fix.detach(), aux_wrist_loss=aux_wrist.detach())
        return FMWMLossOutput(total=total, flow=flow, aux_one_step=aux), stats

    @torch.no_grad()
    def sample_endpoint_latent(
        self,
        *,
        fix_hist: Tensor | None,
        action_chunk: Tensor,
        wrist_hist: Tensor | None = None,
        state_hist: Tensor | None = None,
        num_samples: int = 1,
    ) -> Tensor:
        h = self.config.data.latent_grid_height
        w = self.config.data.latent_grid_width
        c = self.config.data.latent_channels
        v = self._num_views()

        def _sample_once(
            fix_hist_i: Tensor | None,
            wrist_hist_i: Tensor | None,
            state_hist_i: Tensor | None,
            action_chunk_i: Tensor,
        ) -> Tensor:
            b = action_chunk_i.shape[0]
            if v == 1:
                x0 = self.config.fm_head.base_std * torch.randn(
                    b, h, w, c, device=action_chunk_i.device, dtype=action_chunk_i.dtype
                )
            else:
                x0 = self.config.fm_head.base_std * torch.randn(
                    b, v, h, w, c, device=action_chunk_i.device, dtype=action_chunk_i.dtype
                )

            self.odefunc.set_context(fix_hist_i, wrist_hist_i, state_hist_i, action_chunk_i)
            t = torch.tensor([0.0, 1.0], device=x0.device, dtype=x0.dtype)
            traj = odeint(
                self.odefunc,
                x0,
                t,
                rtol=self.config.fm_head.ode_rtol,
                atol=self.config.fm_head.ode_atol,
                method=self.config.fm_head.ode_solver,
            )
            return traj[-1]

        if num_samples == 1:
            return _sample_once(fix_hist, wrist_hist, state_hist, action_chunk)

        outs = []
        for _ in range(num_samples):
            outs.append(_sample_once(fix_hist, wrist_hist, state_hist, action_chunk))
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def predictive_uncertainty_trace(
        self,
        *,
        fix_hist: Tensor | None,
        action_chunk: Tensor,
        wrist_hist: Tensor | None = None,
        state_hist: Tensor | None = None,
        num_samples: int = 8,
    ) -> Tensor:
        samples = self.sample_endpoint_latent(
            fix_hist=fix_hist,
            wrist_hist=wrist_hist,
            state_hist=state_hist,
            action_chunk=action_chunk,
            num_samples=num_samples,
        )
        var = samples.var(dim=1, unbiased=False)
        return var.flatten(1).sum(dim=-1)

    def transport_observed_endpoint_to_base(
        self,
        future_obs: Tensor,
        *,
        fix_hist: Tensor | None,
        wrist_hist: Tensor | None,
        state_hist: Tensor | None,
        action_chunk: Tensor,
    ) -> Tensor:
        self.odefunc.set_context(fix_hist, wrist_hist, state_hist, action_chunk)
        t = torch.tensor([1.0, 0.0], device=future_obs.device, dtype=future_obs.dtype)
        traj = odeint(
            self.odefunc,
            future_obs,
            t,
            rtol=self.config.fm_head.ode_rtol,
            atol=self.config.fm_head.ode_atol,
            method=self.config.fm_head.ode_solver,
        )
        return traj[-1]

    def transport_observed_endpoint_one_step_t1(
        self,
        future_obs: Tensor,
        *,
        fix_hist: Tensor | None,
        wrist_hist: Tensor | None,
        state_hist: Tensor | None,
        action_chunk: Tensor,
    ) -> Tensor:
        tau1 = torch.ones(future_obs.shape[0], device=future_obs.device, dtype=future_obs.dtype)
        v1 = self.predict_velocity(
            fix_hist=fix_hist,
            wrist_hist=wrist_hist,
            state_hist=state_hist,
            action_chunk=action_chunk,
            x_t=future_obs,
            tau=tau1,
        )
        return future_obs - v1

    def transport_observed_endpoint_for_scoring(
        self,
        future_obs: Tensor,
        *,
        fix_hist: Tensor | None,
        wrist_hist: Tensor | None,
        state_hist: Tensor | None,
        action_chunk: Tensor,
    ) -> Tensor:
        mode = self.config.fm_head.score_transport_mode
        if mode == "ode":
            return self.transport_observed_endpoint_to_base(
                future_obs,
                fix_hist=fix_hist,
                wrist_hist=wrist_hist,
                state_hist=state_hist,
                action_chunk=action_chunk,
            )
        if mode == "one_step_t0":
            return self.transport_observed_endpoint_one_step_t1(
                future_obs,
                fix_hist=fix_hist,
                wrist_hist=wrist_hist,
                state_hist=state_hist,
                action_chunk=action_chunk,
            )
        raise ValueError(f"Unsupported score_transport_mode: {mode}")

    def score_from_base(self, eps_hat: Tensor) -> tuple[Tensor, Tensor]:
        flat = eps_hat.flatten(1)
        log_prob = standard_normal_logprob(flat, std=self.config.fm_head.base_std).sum(dim=-1)

        if self.config.fm_head.score_kind == "base_logp":
            scores = -log_prob
        elif self.config.fm_head.score_kind == "squared_norm":
            scores = ((flat / self.config.fm_head.base_std) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported score_kind: {self.config.fm_head.score_kind}")
        return log_prob, scores

    @torch.no_grad()
    def compute_reactive_nonconformity_scores(
        self,
        *,
        fix_hist: Tensor | None,
        action_chunk: Tensor,
        future_fix: Tensor | None,
        wrist_hist: Tensor | None = None,
        future_wrist: Tensor | None = None,
        state_hist: Tensor | None = None,
        future_state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        _ = future_state
        future_obs = self._stack_enabled_views(future_fix, future_wrist)
        eps_hat = self.transport_observed_endpoint_for_scoring(
            future_obs,
            fix_hist=fix_hist,
            wrist_hist=wrist_hist,
            state_hist=state_hist,
            action_chunk=action_chunk,
        )
        log_prob, scores = self.score_from_base(eps_hat)
        return {
            "x_future": future_obs.detach(),
            "eps_hat": eps_hat.detach(),
            "log_prob": log_prob.detach(),
            "scores": scores.detach(),
        }

    def forward(
        self,
        *,
        fix_hist: Tensor | None,
        action_chunk: Tensor,
        future_fix: Tensor | None,
        wrist_hist: Tensor | None = None,
        future_wrist: Tensor | None = None,
        state_hist: Tensor | None = None,
        future_state: Tensor | None = None,
        next_fix: Tensor | None = None,
        next_wrist: Tensor | None = None,
        next_state: Tensor | None = None,
    ) -> dict[str, Any]:
        losses, stats = self.compute_loss(
            fix_hist=fix_hist,
            wrist_hist=wrist_hist,
            state_hist=state_hist,
            action_chunk=action_chunk,
            future_fix=future_fix,
            future_wrist=future_wrist,
            future_state=future_state,
            next_fix=next_fix,
            next_wrist=next_wrist,
            next_state=next_state,
        )
        return {
            "loss": losses.total,
            "losses": losses,
            "stats": stats,
        }
