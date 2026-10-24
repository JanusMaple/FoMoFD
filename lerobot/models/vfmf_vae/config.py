from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ObjectiveKind = Literal[
    "AE",
    "VAE",
]


@dataclass
class DataConfig:
    dataset_repo_id: str
    dataset_root: str
    output_dir: str

    val_ratio: float = 0.1
    split_seed: int = 0

    # Precomputed DINO feature keys
    fix_key: str = "observation.latents.fix_camera"
    wrist_key: str = "observation.latents.wrist_camera"

    use_fix_camera: bool = True
    use_wrist_camera: bool = False

    # Patch-grid shape of stored DINO features.
    # Example: 224x224 image with patch_size=14 -> 16x16 grid
    patch_grid_height: int = 16
    patch_grid_width: int = 16


@dataclass
class RuntimeConfig:
    batch_size: int = 32
    val_batch_size: int | None = None

    max_steps: int = 50000
    log_freq: int = 200
    val_freq: int = 2000
    save_freq: int = 2000

    num_workers: int = 8
    persistent_workers: bool = False
    pin_memory: bool | None = None

    device: str = "cuda"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    max_val_batches: int | None = None


@dataclass
class OptimConfig:
    lr: float = 1.0e-4
    min_lr: float = 1.0e-6
    weight_decay: float = 1.0e-5
    grad_clip_norm: float = 1.0

    scheduler: str = "cosine"  # none | cosine | plateau
    plateau_factor: float = 0.5
    plateau_patience: int = 3


@dataclass
class FeatureConfig:
    # Stored DINO patch feature dimension
    input_dim: int = 768

    # Optional per-channel normalization stats for stored DINO features
    feature_stats: str | None = None


@dataclass
class ConvNeXtIsotropicVAEConfig:
    depth: int = 18
    dim: int = 384
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-6

    # Channel compression at each spatial patch location
    latent_channels: int = 64

    # Must match dataset patch grid
    shape: list[int] = field(default_factory=lambda: [16, 16])

    beta: float = 0.01
    latent_channel_dropout_p: float = 0.0
    latent_channel_utilization_weight: float = 0.0
    latent_channel_utilization_min_frac: float = 0.0


@dataclass
class LossConfig:
    objective: ObjectiveKind = "VAE"


@dataclass
class FeatureSpaceVAEConfig:
    data: DataConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    vae: ConvNeXtIsotropicVAEConfig = field(default_factory=ConvNeXtIsotropicVAEConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    experiment_name: str = "dino_feature_convnext_vae"
    seed: int = 0

    def validate(self) -> None:
        if self.runtime.batch_size < 1:
            raise ValueError("runtime.batch_size must be >= 1")
        if self.runtime.val_batch_size is not None and self.runtime.val_batch_size < 1:
            raise ValueError("runtime.val_batch_size must be >= 1 when provided")
        if self.runtime.max_steps < 1:
            raise ValueError("runtime.max_steps must be >= 1")
        if self.runtime.log_freq < 1:
            raise ValueError("runtime.log_freq must be >= 1")
        if self.runtime.val_freq < 1:
            raise ValueError("runtime.val_freq must be >= 1")
        if self.runtime.save_freq < 1:
            raise ValueError("runtime.save_freq must be >= 1")

        if self.runtime.amp_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("runtime.amp_dtype must be one of: float16, bfloat16, float32")

        if self.optim.lr <= 0:
            raise ValueError("optim.lr must be > 0")
        if self.optim.min_lr <= 0:
            raise ValueError("optim.min_lr must be > 0")
        if self.optim.grad_clip_norm <= 0:
            raise ValueError("optim.grad_clip_norm must be > 0")
        if self.optim.scheduler not in {"none", "cosine", "plateau"}:
            raise ValueError("optim.scheduler must be one of: none, cosine, plateau")

        if not (self.data.use_fix_camera or self.data.use_wrist_camera):
            raise ValueError("At least one camera must be enabled")
        if self.data.patch_grid_height < 1 or self.data.patch_grid_width < 1:
            raise ValueError("Patch grid dimensions must be >= 1")

        if self.feature.input_dim < 1:
            raise ValueError("feature.input_dim must be >= 1")

        if self.vae.depth < 1:
            raise ValueError("vae.depth must be >= 1")
        if self.vae.dim < 1:
            raise ValueError("vae.dim must be >= 1")
        if self.vae.latent_channels < 1:
            raise ValueError("vae.latent_channels must be >= 1")
        if self.vae.beta < 0:
            raise ValueError("vae.beta must be >= 0")
        if len(self.vae.shape) != 2:
            raise ValueError("vae.shape must be [H_p, W_p]")

        expected_hp = self.data.patch_grid_height
        expected_wp = self.data.patch_grid_width
        if self.vae.shape[0] != expected_hp or self.vae.shape[1] != expected_wp:
            raise ValueError(
                f"vae.shape ({self.vae.shape}) must match "
                f"[patch_grid_height, patch_grid_width] = [{expected_hp}, {expected_wp}]"
            )

        if self.loss.objective not in {"AE", "VAE"}:
            raise ValueError("loss.objective must be one of: AE, VAE")