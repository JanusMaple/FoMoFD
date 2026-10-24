#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig   # type: ignore
from lerobot.configs.types import NormalizationMode     # type: ignore
from lerobot.optim.optimizers import AdamWConfig        # type: ignore


@PreTrainedConfig.register_subclass("act_dinov2")
@dataclass
class ACTDINOv2Config(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision / ACT_DINOv2 specifics
    dinov2_model_name: str = "dinov2_vitb14_reg"
    dinov2_feature_dim: int = 768
    robot_state_token_repetition: int = 1
    use_2d_pos_emb: bool = False
    per_camera_num_queries: int = 8
    camera_embed_init_std: float = 0.02
    use_cls_token: bool = False

    # Camera selection
    use_fix_camera: bool = True
    use_wrist_camera: bool = True

    # State usage
    use_robot_state_in_policy: bool = True
    use_robot_state_in_vae: bool = True

    # Weak-state controls
    state_scale: float = 1.0
    state_dropout_prob: float = 0.0
    state_noise_std: float = 0.0

    # Transformer layers
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1

    # VAE
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference
    temporal_ensemble_coeff: float | None = None

    # Training / loss
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. "
                "The policy must be queried every step to compute the ensembled action."
            )

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` must be <= `chunk_size`. "
                f"Got n_action_steps={self.n_action_steps}, chunk_size={self.chunk_size}."
            )

        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got n_obs_steps={self.n_obs_steps}"
            )

        if not self.use_fix_camera and not self.use_wrist_camera:
            raise ValueError("At least one camera must be enabled.")

        if not (0.0 <= self.state_dropout_prob < 1.0):
            raise ValueError(f"state_dropout_prob must be in [0,1), got {self.state_dropout_prob}")

        if self.state_scale < 0:
            raise ValueError(f"state_scale must be >= 0, got {self.state_scale}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def _camera_key_enabled(self, key: str) -> bool:
        if key.endswith(".fix_camera"):
            return self.use_fix_camera
        if key.endswith(".wrist_camera"):
            return self.use_wrist_camera
        return True

    @property
    def image_features(self) -> dict:
        return {
            k: v
            for k, v in self.input_features.items()
            if k.startswith("observation.images.") and self._camera_key_enabled(k)
        }

    @property
    def token_features(self) -> dict:
        return {
            k: v
            for k, v in self.input_features.items()
            if k.startswith("observation.tokens.") and self._camera_key_enabled(k)
        }

    @property
    def token_feature_dim(self) -> int | None:
        if not self.token_features:
            return None
        dims = {feat.shape[-1] for feat in self.token_features.values()}
        if len(dims) != 1:
            raise ValueError(f"All token features must share the same last dim, got {dims}")
        return next(iter(dims))

    @property
    def token_num_queries(self) -> int | None:
        if not self.token_features:
            return None
        qs = {feat.shape[0] for feat in self.token_features.values()}
        if len(qs) != 1:
            raise ValueError(f"All token features must share the same token count, got {qs}")
        return next(iter(qs))

    def validate_features(self) -> None:
        has_images = bool(self.image_features)
        has_tokens = bool(self.token_features)
        has_env = bool(self.env_state_feature)

        if has_images and has_tokens:
            raise ValueError("Use either image_features or token_features, not both.")

        if not has_images and not has_tokens and not has_env:
            raise ValueError(
                "You must provide at least one image, token feature, or environment state among the inputs."
            )

        if has_tokens:
            _ = self.token_feature_dim
            _ = self.token_num_queries

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None