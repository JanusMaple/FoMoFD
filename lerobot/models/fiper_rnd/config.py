from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RNDLossKind = Literal[
    "mse",
    "l2",
]

PoolKind = Literal[
    "flatten",   # default for strict FIPER-style obs embedding use
]


@dataclass
class DataConfig:
    dataset_repo_id: str
    dataset_root: str
    output_dir: str

    val_ratio: float = 0.1
    split_seed: int = 0

    fix_key: str = "observation.latents.fix_camera"
    wrist_key: str = "observation.latents.wrist_camera"
    state_key: str = "observation.state"

    use_fix_camera: bool = False
    use_wrist_camera: bool = True
    use_state: bool = False

    latent_grid_height: int = 16
    latent_grid_width: int = 16
    latent_channels: int = 8

    normalize_vfmf_vae_latents: bool = False
    vfmf_vae_latent_stats_path: str | None = None
    vfmf_vae_latent_std_floor: float = 0.02

    state_dim: int = 10

    obs_pool: PoolKind = "flatten"


@dataclass
class RuntimeConfig:
    batch_size: int = 128
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

    scheduler: str = "cosine"
    plateau_factor: float = 0.5
    plateau_patience: int = 3


@dataclass
class ModelConfig:
    """
    Predictor / target widths for RND.
    For wrist-only VFMF-VAE latents:
      obs_dim = 16 * 16 * 8 = 2048
    """

    rnd_loss: RNDLossKind = "mse"

    predictor_hidden_dim_1: int = 2048
    predictor_hidden_dim_2: int = 1024
    output_dim: int = 256

    use_layernorm: bool = False
    dropout: float = 0.0


@dataclass
class ScoreConfig:
    """
    Runtime score reduction for frame-level RND error.
    """
    score_reduction: Literal["mean", "sum"] = "mean"


@dataclass
class FiperRNDConfig:
    data: DataConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)

    experiment_name: str = "fiper_rnd"
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
            raise ValueError("At least one of use_fix_camera or use_wrist_camera must be True")

        if self.data.latent_grid_height < 1 or self.data.latent_grid_width < 1:
            raise ValueError("latent grid dimensions must be >= 1")
        if self.data.latent_channels < 1:
            raise ValueError("latent_channels must be >= 1")
        if self.data.use_state and self.data.state_dim < 1:
            raise ValueError("state_dim must be >= 1 when use_state=True")

        if self.model.predictor_hidden_dim_1 < 1:
            raise ValueError("model.predictor_hidden_dim_1 must be >= 1")
        if self.model.predictor_hidden_dim_2 < 1:
            raise ValueError("model.predictor_hidden_dim_2 must be >= 1")
        if self.model.output_dim < 1:
            raise ValueError("model.output_dim must be >= 1")

        if self.score.score_reduction not in {"mean", "sum"}:
            raise ValueError("score.score_reduction must be 'mean' or 'sum'")