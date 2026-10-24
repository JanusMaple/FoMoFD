from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FMPathKind = Literal["linear"]
TimeSamplingKind = Literal["uniform"]
ODESolverKind = Literal["dopri5", "rk4"]
ScoreKind = Literal["base_logp", "squared_norm"]
ScoreTransportModeKind = Literal["ode", "one_step_t0",]


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
    use_state: bool = True

    latent_grid_height: int = 16
    latent_grid_width: int = 16
    latent_channels: int = 8
    normalize_vfmf_vae_latents: bool = False
    vfmf_vae_latent_stats_path: str | None = None
    vfmf_vae_latent_std_floor: float = 0.02
    state_dim: int = 10


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
    hidden_dim: int = 384
    depth: int = 8
    heads: int = 8
    mlp_dim: int = 1536
    dropout: float = 0.1

    state_hidden_dim: int = 256
    state_token_dim: int = 384


@dataclass
class FMHeadConfig:
    path_kind: FMPathKind = "linear"
    time_sampling: TimeSamplingKind = "uniform"

    min_time: float = 0.0
    max_time: float = 1.0
    base_std: float = 1.0

    ode_solver: ODESolverKind = "dopri5"
    ode_rtol: float = 1.0e-4
    ode_atol: float = 1.0e-5

    score_kind: ScoreKind = "base_logp"
    score_transport_mode: ScoreTransportModeKind = "ode"


@dataclass
class LossConfig:
    flow_weight: float = 1.0


@dataclass
class LogpZOConfig:
    data: DataConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fm_head: FMHeadConfig = field(default_factory=FMHeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    experiment_name: str = "faildetect_logpzo_baseline"
    seed: int = 0