from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import FeatureSpaceVAEConfig
from .convnext import ConvNeXtIsotropic
from .diagonal_gaussian import DiagonalGaussianDistribution


@dataclass
@dataclass
class VAEOutput:
    sample: Tensor
    latents: Tensor
    elbo: Tensor
    ll: Tensor
    kl: Tensor
    latent_dist: DiagonalGaussianDistribution
    channel_utilization_loss: Tensor | None = None
    channel_energy: Tensor | None = None


class FeatureNormalizer(nn.Module):
    """
    Optional per-channel normalization for precomputed DINO feature maps.
    Expects channels-last tensors: (..., C)
    """

    def __init__(self, feature_stats: str | None):
        super().__init__()
        self.eps = 1e-8

        if feature_stats is None:
            self.enabled = False
            self.register_buffer("mean", torch.empty(0), persistent=False)
            self.register_buffer("std", torch.empty(0), persistent=False)
            return

        stats_path = Path(feature_stats)
        if not stats_path.exists():
            raise FileNotFoundError(f"feature_stats not found: {stats_path}")

        stats = np.load(stats_path)
        mean = torch.from_numpy(stats["mean"]).float()
        std = torch.from_numpy(stats["std"]).float()

        self.enabled = True
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def normalize(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return x
        mean = self.mean.view(*([1] * (x.ndim - 1)), -1).to(x.device)
        std = self.std.view(*([1] * (x.ndim - 1)), -1).to(x.device)
        return (x - mean) / (std + self.eps)

    def denormalize(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return x
        mean = self.mean.view(*([1] * (x.ndim - 1)), -1).to(x.device)
        std = self.std.view(*([1] * (x.ndim - 1)), -1).to(x.device)
        return x * (std + self.eps) + mean


class ConvNeXtFeatureVAE(nn.Module):
    """
    Feature-space ConvNeXt isotropic β-VAE.

    Input:
      x: [B, H, W, C]  (channels-last DINO patch feature map)

    Latent:
      z: [B, H, W, Cz]

    Reconstruction:
      x_hat: [B, H, W, C]
    """

    def __init__(
        self,
        input_dim: int,
        depth: int,
        dim: int,
        drop_path_rate: float,
        layer_scale_init_value: float,
        latent_channels: int,
        beta: float,
        latent_channel_dropout_p: float = 0.0,
        latent_channel_utilization_weight: float = 0.0,
        latent_channel_utilization_min_frac: float = 0.5,
    ) -> None:
        super().__init__()

        self.beta = beta
        self.latent_channel_dropout_p = float(latent_channel_dropout_p)
        self.latent_channel_utilization_weight = float(latent_channel_utilization_weight)
        self.latent_channel_utilization_min_frac = float(latent_channel_utilization_min_frac)

        # Encoder backbone
        self.encoder_backbone = ConvNeXtIsotropic(
            in_chans=input_dim,
            depth=depth,
            dim=dim,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.to_latents_norm = nn.LayerNorm(dim)
        self.to_latents = nn.Linear(dim, 2 * latent_channels)
        self.to_latents.weight.data.normal_(std=0.02)

        # Decoder backbone
        self.from_latents = nn.Linear(latent_channels, dim)
        self.from_latents.weight.data.normal_(std=0.02)

        self.decoder_backbone = ConvNeXtIsotropic(
            in_chans=dim,
            depth=depth,
            dim=dim,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.out_proj = nn.Linear(dim, input_dim)
        self.out_proj.weight.data.normal_(std=0.02)

    def encode(self, x: Tensor) -> DiagonalGaussianDistribution:
        """
        x: [B, H, W, C]
        """
        h = einops.rearrange(x, "b h w c -> b c h w").contiguous()
        h = self.encoder_backbone(h)
        h = einops.rearrange(h, "b c h w -> b h w c").contiguous()
        h = self.to_latents_norm(h)
        params = self.to_latents(h)
        return DiagonalGaussianDistribution(params)

    def decode(self, z: Tensor) -> Tensor:
        """
        z: [B, H, W, Cz]
        """
        h = self.from_latents(z)
        h = einops.rearrange(h, "b h w c -> b c h w").contiguous()
        h = self.decoder_backbone(h)
        h = einops.rearrange(h, "b c h w -> b h w c").contiguous()
        x_hat = self.out_proj(h)
        return x_hat

    def forward(self, x: Tensor, deterministic: bool = False) -> VAEOutput:
        q_z_given_x = self.encode(x)

        if deterministic:
            z = q_z_given_x.mode()
        else:
            z = q_z_given_x.sample()

        z_for_decode = self._latent_channel_dropout(z)
        x_hat = self.decode(z_for_decode)

        kl = q_z_given_x.kl()  # [B]
        ll = -F.mse_loss(x_hat, x, reduction="none").sum(dim=[1, 2, 3])  # [B]

        channel_utilization_loss = x.new_zeros(())
        channel_energy = None

        if deterministic:
            elbo = ll
        else:
            elbo = ll - self.beta * kl

            if self.latent_channel_utilization_weight > 0.0:
                z_mean = q_z_given_x.mode()
                channel_utilization_loss, channel_energy = self._channel_utilization_loss(z_mean)

                elbo = elbo - self.latent_channel_utilization_weight * channel_utilization_loss

        return VAEOutput(
            sample=x_hat,
            latents=z,
            elbo=elbo,
            ll=ll,
            kl=kl,
            latent_dist=q_z_given_x,
            channel_utilization_loss=channel_utilization_loss,
            channel_energy=channel_energy,
        )

    def _latent_channel_dropout(self, z: Tensor) -> Tensor:
        """
        With probability p per sample, drop exactly one latent channel.
        z: [B, H, W, C]
        """
        p = float(self.latent_channel_dropout_p)
        if not self.training or p <= 0.0:
            return z

        b, _, _, c = z.shape
        if c <= 1:
            return z

        apply = torch.rand(b, device=z.device) < p
        if not torch.any(apply):
            return z

        mask = torch.ones((b, 1, 1, c), device=z.device, dtype=z.dtype)
        drop_idx = torch.randint(0, c, (b,), device=z.device)

        batch_idx = torch.arange(b, device=z.device)
        mask[batch_idx[apply], 0, 0, drop_idx[apply]] = 0.0

        # Preserve expected activation scale for samples where one channel is dropped.
        scale = torch.ones((b, 1, 1, 1), device=z.device, dtype=z.dtype)
        scale[apply] = float(c) / float(c - 1)

        return z * mask * scale

    def _channel_utilization_loss(self, z: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
        """
        Encourage latent information to spread across channels.

        z: [B, H, W, C]
        Returns:
        loss: scalar
        energy: [C], normalized per-channel variance
        """
        c = z.shape[-1]
        if c <= 1:
            energy = torch.ones(c, device=z.device, dtype=z.dtype)
            return z.new_zeros(()), energy

        x = z.float().reshape(-1, c)
        x = x - x.mean(dim=0, keepdim=True)

        var = x.pow(2).mean(dim=0) + eps
        energy = var / var.sum().clamp_min(eps)

        # Penalize energy concentration.
        entropy_balance = (energy * torch.log(energy * c + eps)).sum()

        # Ensure no channel falls too far below uniform usage.
        min_energy = float(self.latent_channel_utilization_min_frac) / float(c)
        floor_loss = F.relu(min_energy - energy).pow(2).sum() * c

        loss = entropy_balance + floor_loss
        return loss.to(dtype=z.dtype), energy.to(dtype=z.dtype)


class FeatureSpaceVAE(nn.Module):
    """
    Thin wrapper around ConvNeXtFeatureVAE for precomputed DINO patch features.

    Supports:
      - optional per-channel normalization / inverse normalization
      - either single-frame [B,H,W,C] or sequence [B,T,H,W,C] input
    """

    def __init__(self, config: FeatureSpaceVAEConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.normalizer = FeatureNormalizer(config.feature.feature_stats)

        self.vae = ConvNeXtFeatureVAE(
            input_dim=config.feature.input_dim,
            depth=config.vae.depth,
            dim=config.vae.dim,
            drop_path_rate=config.vae.drop_path_rate,
            layer_scale_init_value=config.vae.layer_scale_init_value,
            latent_channels=config.vae.latent_channels,
            beta=config.vae.beta,
            latent_channel_dropout_p=config.vae.latent_channel_dropout_p,
            latent_channel_utilization_weight=getattr(config.vae, "latent_channel_utilization_weight", 0.0),
            latent_channel_utilization_min_frac=getattr(config.vae, "latent_channel_utilization_min_frac", 0.5),
        )

    def _check_spatial_shape(self, x: Tensor) -> None:
        hp = self.config.data.patch_grid_height
        wp = self.config.data.patch_grid_width

        if x.shape[-3] != hp or x.shape[-2] != wp:
            raise ValueError(
                f"Expected patch grid [{hp}, {wp}], got [{x.shape[-3]}, {x.shape[-2]}]"
            )

        if x.shape[-1] != self.config.feature.input_dim:
            raise ValueError(
                f"Expected feature dim {self.config.feature.input_dim}, got {x.shape[-1]}"
            )

    def preprocess(self, x: Tensor) -> Tensor:
        """
        Normalize precomputed DINO features.

        Accepts:
          - [B, H, W, C]
          - [B, T, H, W, C]
        Returns same shape.
        """
        self._check_spatial_shape(x if x.ndim == 4 else x[:, 0])
        return self.normalizer.normalize(x)

    def postprocess(self, x: Tensor) -> Tensor:
        """
        Inverse normalization.
        """
        return self.normalizer.denormalize(x)

    def encode(self, x: Tensor, deterministic: bool = False) -> Tensor:
        """
        Returns latent z.

        Input:
          - [B, H, W, C]
          - [B, T, H, W, C]

        Output:
          - [B, H, W, Cz]
          - [B, T, H, W, Cz]
        """
        x = self.preprocess(x)

        if x.ndim == 4:
            out = self.vae(x, deterministic=deterministic)
            return out.latents

        if x.ndim == 5:
            b, t, h, w, c = x.shape
            flat = einops.rearrange(x, "b t h w c -> (b t) h w c")
            out = self.vae(flat, deterministic=deterministic)
            z = einops.rearrange(out.latents, "(b t) h w c -> b t h w c", b=b, t=t)
            return z

        raise ValueError(f"Expected input rank 4 or 5, got shape {tuple(x.shape)}")

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent z back to feature map space.

        Input:
          - [B, H, W, Cz]
          - [B, T, H, W, Cz]

        Output:
          - [B, H, W, C]
          - [B, T, H, W, C]
        """
        if z.ndim == 4:
            x_hat = self.vae.decode(z)
            return self.postprocess(x_hat)

        if z.ndim == 5:
            b, t, h, w, c = z.shape
            flat = einops.rearrange(z, "b t h w c -> (b t) h w c")
            x_hat = self.vae.decode(flat)
            x_hat = self.postprocess(x_hat)
            x_hat = einops.rearrange(x_hat, "(b t) h w c -> b t h w c", b=b, t=t)
            return x_hat

        raise ValueError(f"Expected latent rank 4 or 5, got shape {tuple(z.shape)}")

    def forward(self, x: Tensor, deterministic: bool = False) -> VAEOutput:
        """
        Train/eval path.

        Input:
          - [B, H, W, C]
          - [B, T, H, W, C]

        Output VAEOutput:
          sample : same shape as input
          latents: same leading dims, channel changed to latent_channels
        """
        x = self.preprocess(x)

        if x.ndim == 4:
            out = self.vae(x, deterministic=deterministic)
            out.sample = self.postprocess(out.sample)
            return out

        if x.ndim == 5:
            b, t, h, w, c = x.shape
            flat = einops.rearrange(x, "b t h w c -> (b t) h w c")
            out = self.vae(flat, deterministic=deterministic)

            sample = self.postprocess(out.sample)
            sample = einops.rearrange(sample, "(b t) h w c -> b t h w c", b=b, t=t)
            latents = einops.rearrange(out.latents, "(b t) h w c -> b t h w c", b=b, t=t)
            elbo = einops.rearrange(out.elbo, "(b t) -> b t", b=b, t=t).mean(dim=1)
            ll = einops.rearrange(out.ll, "(b t) -> b t", b=b, t=t).mean(dim=1)
            kl = einops.rearrange(out.kl, "(b t) -> b t", b=b, t=t).mean(dim=1)

            return VAEOutput(
                sample=sample,
                latents=latents,
                elbo=elbo,
                ll=ll,
                kl=kl,
                latent_dist=out.latent_dist,
                channel_utilization_loss=out.channel_utilization_loss,
                channel_energy=out.channel_energy,
            )

        raise ValueError(f"Expected input rank 4 or 5, got shape {tuple(x.shape)}")