from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FMPathKind = Literal[
    "linear",
]

TimeSamplingKind = Literal[
    "uniform",
]

ODESolverKind = Literal[
    "dopri5",
    "rk4",
]

ScoreKind = Literal[
    "base_logp",
    "squared_norm",
]

ScoreTransportModeKind = Literal[
    "ode",
    "one_step_t0",
]

ProactiveScoreModeKind = Literal[
    "policy_fm_wm_entropy",
    "fm_wm_entropy", 
    "ace"
]

ActionGlobalPoolKind = Literal[
    "mean_pool_tokens",
    "compose_raw_chunk",
]

LatentNormalizationModeKind = Literal[
    "channel",
    "view_global",
]

ActionNormalizationModeKind = Literal[
    "dp_gripper_zscore_rot_identity",
    "dp_gripper_zscore_rot_mean",
]


@dataclass
class DataConfig:
    dataset_repo_id: str
    dataset_root: str
    output_dir: str

    val_ratio: float = 0.1
    split_seed: int = 0
    # Subsample training episodes AFTER the fixed episode-level train/val split.
    train_episode_fraction: float = 1.0
    train_subset_seed: int = 0

    fix_key: str = "observation.latents.fix_camera"
    wrist_key: str = "observation.latents.wrist_camera"
    state_key: str = "observation.state"
    action_key: str = "action"
    proactive_action_key: str | None = None
    proactive_action_batch_key: str | None = None

    use_fix_camera: bool = True
    use_wrist_camera: bool = False
    num_cameras: int = 1

    history_len: int = 5
    action_chunk_len: int = 5
    pred_horizon: int = 5

    latent_grid_height: int = 16
    latent_grid_width: int = 16
    latent_channels: int = 16

    # Normalize latents output by pretrained VAE
    normalize_vfmf_vae_latents: bool = False
    latent_normalization_mode: LatentNormalizationModeKind = "channel"  # By defualt is channel
    vfmf_vae_latent_stats_path: str | None = None
    vfmf_vae_latent_std_floor: float = 0.02

    # Normalize actions for FM-WM conditioning.
    # Assumes action format: dp(3) + rot6d(6) + gripper(1).
    # - dp and gripper are z-scored.
    # - rot6d is centered only, either around identity or empirical mean.
    normalize_actions: bool = False
    action_normalization_mode: ActionNormalizationModeKind = "dp_gripper_zscore_rot_identity"
    action_stats_path: str | None = None
    action_std_floor: float = 1.0e-6

    # kept for interface compatibility, but default off for this version
    use_state: bool = False
    state_dim: int = 10


@dataclass
class RuntimeConfig:
    batch_size: int = 32
    val_batch_size: int | None = None

    max_steps: int = 50000
    log_freq: int = 200
    val_freq: int = 2000
    save_freq: int = 2000

    # None preserves fixed-budget training. Count optimizer steps without a
    # strictly lower validation flow loss; evaluate this after validation.
    early_stop_patience_steps: int | None = None

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
class StateConditionerConfig:
    use_state_tokens: bool = False
    token_dim: int = 384
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class ActionConditionerConfig:
    action_dim: int = 10

    # token path (existing action tokenizer / action conditioner)
    token_dim: int = 384
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1

    # keep token-level cross attention
    use_cross_attention: bool = True

    # new global action conditioning for adaLN-style modulation
    use_action_adaln: bool = True
    global_pool_mode: ActionGlobalPoolKind = "mean_pool_tokens"
    global_cond_dim: int = 384
    global_hidden_dim: int = 512


@dataclass
class BackboneConfig:
    # shared hyperparameter spec for each enabled view branch
    hidden_dim: int = 384
    depth: int = 8
    heads: int = 8
    mlp_dim: int = 1536
    dropout: float = 0.1

    separable_attention: bool = True
    separable_window_size: int = 1

    # Keep legacy branches by default. Concatenation happens inside the model,
    # AFTER the dataset normalizes each camera with its existing statistics.
    view_fusion: Literal["separate", "concat_channels"] = "separate"


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

    score_kind: ScoreKind = "squared_norm"
    score_transport_mode: ScoreTransportModeKind = "one_step_t0"

    proactive_score_mode: ProactiveScoreModeKind = "fm_wm_entropy"
    per_chunk_wm_samples: int = 1
    proactive_num_action_chunks: int | None = None
    ace_cov_eps: float = 1.0e-6


@dataclass
class AuxiliaryHeadConfig:
    use_one_step_aux_head: bool = True
    one_step_loss_weight: float = 1.0


@dataclass
class LossConfig:
    flow_weight: float = 1.0


@dataclass
class FlowMatchingWorldModelConfig:
    data: DataConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    state_conditioner: StateConditionerConfig = field(default_factory=StateConditionerConfig)
    action_conditioner: ActionConditionerConfig = field(default_factory=ActionConditionerConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fm_head: FMHeadConfig = field(default_factory=FMHeadConfig)
    aux_head: AuxiliaryHeadConfig = field(default_factory=AuxiliaryHeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    experiment_name: str = "flow_matching_world_model_map"
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
        patience = self.runtime.early_stop_patience_steps
        if patience is not None and (type(patience) is not int or patience < 1):
            raise ValueError("runtime.early_stop_patience_steps must be a positive integer or null")

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

        if self.data.history_len < 1:
            raise ValueError("data.history_len must be >= 1")
        if not 0.0 < self.data.train_episode_fraction <= 1.0:
            raise ValueError("data.train_episode_fraction must be in (0, 1]")
        if self.data.action_chunk_len < 1:
            raise ValueError("data.action_chunk_len must be >= 1")
        if self.data.pred_horizon < 1:
            raise ValueError("data.pred_horizon must be >= 1")
        if self.data.action_chunk_len != self.data.pred_horizon:
            raise ValueError("data.action_chunk_len must equal data.pred_horizon for endpoint prediction")
        if self.data.latent_grid_height < 1 or self.data.latent_grid_width < 1:
            raise ValueError("latent grid dimensions must be >= 1")
        if self.data.latent_channels < 1:
            raise ValueError("data.latent_channels must be >= 1")

        enabled_views = int(self.data.use_fix_camera) + int(self.data.use_wrist_camera)
        if enabled_views < 1:
            raise ValueError("At least one of data.use_fix_camera or data.use_wrist_camera must be True")
        if self.data.num_cameras != enabled_views:
            raise ValueError(
                f"data.num_cameras must equal the number of enabled views, got "
                f"num_cameras={self.data.num_cameras}, enabled_views={enabled_views}"
            )
        if self.backbone.view_fusion not in {"separate", "concat_channels"}:
            raise ValueError("backbone.view_fusion must be separate or concat_channels")
        if self.backbone.view_fusion == "concat_channels" and enabled_views != 2:
            raise ValueError("concat_channels requires both fixed and wrist cameras")

        if self.data.use_state and self.data.state_dim < 1:
            raise ValueError("data.state_dim must be >= 1 when use_state=True")
        
        if self.data.latent_normalization_mode not in {"channel", "view_global"}:
            raise ValueError("data.latent_normalization_mode must be one of: channel, view_global")
        
        if self.data.action_normalization_mode not in {
            "dp_gripper_zscore_rot_identity",
            "dp_gripper_zscore_rot_mean",
        }:
            raise ValueError(
                "data.action_normalization_mode must be one of: "
                "dp_gripper_zscore_rot_identity, dp_gripper_zscore_rot_mean"
            )

        if self.data.normalize_actions:
            if self.data.action_stats_path is None:
                raise ValueError("data.normalize_actions=True requires data.action_stats_path")
            if self.data.action_std_floor <= 0:
                raise ValueError("data.action_std_floor must be > 0")
            if self.action_conditioner.action_dim < 10:
                raise ValueError(
                    "data.normalize_actions=True assumes action_dim >= 10 "
                    "for dp(3)+rot6d(6)+gripper(1)"
                )
            if self.action_conditioner.global_pool_mode == "compose_raw_chunk":
                raise ValueError(
                    "Action normalization is incompatible with "
                    "action_conditioner.global_pool_mode='compose_raw_chunk', because "
                    "normalized/centered rot6d is no longer a valid rotation. Use "
                    "global_pool_mode='mean_pool_tokens' instead."
                )

        if self.state_conditioner.token_dim != self.backbone.hidden_dim:
            raise ValueError("state_conditioner.token_dim must match backbone.hidden_dim")
        if self.action_conditioner.token_dim != self.backbone.hidden_dim:
            raise ValueError("action_conditioner.token_dim must match backbone.hidden_dim")
        if self.action_conditioner.global_cond_dim != self.backbone.hidden_dim:
            raise ValueError("action_conditioner.global_cond_dim must match backbone.hidden_dim")

        if self.action_conditioner.action_dim < 1:
            raise ValueError("action_conditioner.action_dim must be >= 1")
        if self.action_conditioner.hidden_dim < 1:
            raise ValueError("action_conditioner.hidden_dim must be >= 1")
        if self.action_conditioner.num_layers < 1:
            raise ValueError("action_conditioner.num_layers must be >= 1")
        if self.action_conditioner.num_heads < 1:
            raise ValueError("action_conditioner.num_heads must be >= 1")
        if self.action_conditioner.global_hidden_dim < 1:
            raise ValueError("action_conditioner.global_hidden_dim must be >= 1")
        # if not (self.action_conditioner.use_cross_attention or self.action_conditioner.use_action_adaln):
        #     raise ValueError("At least one of use_cross_attention or use_action_adaln must be True")

        if self.backbone.hidden_dim < 1:
            raise ValueError("backbone.hidden_dim must be >= 1")
        if self.backbone.depth < 1:
            raise ValueError("backbone.depth must be >= 1")
        if self.backbone.heads < 1:
            raise ValueError("backbone.heads must be >= 1")
        if self.backbone.mlp_dim < 1:
            raise ValueError("backbone.mlp_dim must be >= 1")

        if not (0.0 <= self.fm_head.min_time < self.fm_head.max_time <= 1.0):
            raise ValueError("fm_head.min_time and max_time must satisfy 0 <= min < max <= 1")
        if self.fm_head.base_std <= 0:
            raise ValueError("fm_head.base_std must be > 0")
        if self.fm_head.per_chunk_wm_samples < 1:
            raise ValueError("fm_head.per_chunk_wm_samples must be >= 1")

        if self.aux_head.one_step_loss_weight <= 0:
            raise ValueError("aux_head.one_step_loss_weight must be > 0")
        if self.loss.flow_weight <= 0:
            raise ValueError("loss.flow_weight must be > 0")
