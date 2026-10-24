import math
import einops
import torch
from torch import Tensor, nn


class DINOVisualTokenizerOutput:
    def __init__(
        self,
        compressed_tokens: dict[str, Tensor],
        patch_tokens: dict[str, Tensor] | None = None,
        cls_tokens: dict[str, Tensor | None] | None = None,
    ):
        self.compressed_tokens = compressed_tokens
        self.patch_tokens = patch_tokens or {}
        self.cls_tokens = cls_tokens or {}


class CameraTokenCompressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        num_queries: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.num_queries = num_queries

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, model_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(
        self,
        patch_tokens: Tensor,              # (B, N, C_in)
        camera_embed: Tensor | None = None,
        patch_pos: Tensor | None = None,   # (1, N, D)
    ) -> Tensor:
        x = self.input_proj(patch_tokens)

        if patch_pos is not None:
            x = x + patch_pos
        if camera_embed is not None:
            x = x + camera_embed

        q = self.query_tokens.expand(x.shape[0], -1, -1)
        if camera_embed is not None:
            q = q + camera_embed

        attn_out, _ = self.cross_attn(q, x, x, need_weights=False)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        return q


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


class DINOVisualTokenizer(nn.Module):
    """
    Multi-view DINOv2 front-end:
      raw image OR pre-extracted DINO patch features OR pre-compressed tokens
      -> compressed camera tokens
    """
    def __init__(self, config, *, dinov2: nn.Module | None = None):
        super().__init__()
        self.config = config

        # Dataset export can supply an uninitialized backbone and then strictly
        # restore this entire module from the ACT checkpoint, without downloading
        # unrelated pretrained weights. Policy construction keeps its usual path.
        self.dinov2 = dinov2 if dinov2 is not None else torch.hub.load(
            "facebookresearch/dinov2", config.dinov2_model_name
        )
        for p in self.dinov2.parameters():
            p.requires_grad = False

        self.camera_embeds = nn.ParameterDict()
        self.camera_tokenizers = nn.ModuleDict()

        for cam_key in config.image_features.keys():
            safe_key = cam_key.replace(".", "_")

            cam_embed = nn.Parameter(torch.zeros(1, 1, config.dim_model))
            nn.init.normal_(cam_embed, std=config.camera_embed_init_std)
            self.camera_embeds[safe_key] = cam_embed

            self.camera_tokenizers[safe_key] = CameraTokenCompressor(
                in_dim=config.dinov2_feature_dim,
                model_dim=config.dim_model,
                num_queries=config.per_camera_num_queries,
                n_heads=config.n_heads,
                dropout=config.dropout,
            )

        self.spatial_pos_embed = None
        if config.use_2d_pos_emb:
            self.spatial_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        if config.use_cls_token:
            self.dinov2_cls_proj = nn.Sequential(
                nn.LayerNorm(config.dinov2_feature_dim),
                nn.Linear(config.dinov2_feature_dim, config.dim_model),
            )

    def _extract_dino(self, img_tensor: Tensor) -> tuple[Tensor, Tensor | None]:
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        with torch.no_grad():
            feat = self.dinov2.forward_features(img_tensor)
            patch_tokens = feat["x_norm_patchtokens"]
            cls_token = feat.get("x_norm_clstoken", None)
        return patch_tokens, cls_token

    def _make_patch_pos(self, patch_tokens: Tensor) -> Tensor | None:
        if not self.config.use_2d_pos_emb:
            return None

        seq_len = patch_tokens.shape[1]
        grid_size = int(math.sqrt(seq_len))
        if grid_size * grid_size != seq_len:
            raise ValueError(f"Patch token count {seq_len} is not a square grid.")

        dummy_grid = torch.zeros((1, 1, grid_size, grid_size), device=patch_tokens.device, dtype=torch.float32)
        pos_2d = self.spatial_pos_embed(dummy_grid)
        pos_2d = einops.rearrange(pos_2d, "1 d h w -> 1 (h w) d")
        return pos_2d

    def compress_from_patch_features(
        self,
        patch_dict: dict[str, Tensor | dict[str, Tensor]],
    ) -> DINOVisualTokenizerOutput:
        compressed = {}
        patch_out = {}
        cls_out = {}

        for cam_key in self.config.image_features.keys():
            safe_key = cam_key.replace(".", "_")

            feat = patch_dict[cam_key]
            cls_token = None

            if isinstance(feat, dict):
                patch_tokens = feat["x_norm_patchtokens"]
                cls_token = feat.get("x_norm_clstoken", None)
            else:
                patch_tokens = feat

            patch_pos = self._make_patch_pos(patch_tokens)

            cam_tokens = self.camera_tokenizers[safe_key](
                patch_tokens=patch_tokens,
                camera_embed=self.camera_embeds[safe_key],
                patch_pos=patch_pos,
            )

            if self.config.use_cls_token:
                if cls_token is None:
                    raise ValueError(f"use_cls_token=True but no CLS token for {cam_key}")
                cls_proj = self.dinov2_cls_proj(cls_token).unsqueeze(1)
                cls_proj = cls_proj + self.camera_embeds[safe_key]
                cam_tokens = torch.cat([cls_proj, cam_tokens], dim=1)

            compressed[cam_key] = cam_tokens
            patch_out[cam_key] = patch_tokens
            cls_out[cam_key] = cls_token

        return DINOVisualTokenizerOutput(
            compressed_tokens=compressed,
            patch_tokens=patch_out,
            cls_tokens=cls_out,
        )

    def forward(self, batch: dict[str, Tensor]) -> DINOVisualTokenizerOutput:
        # Highest priority: already compressed tokens
        if "compressed_visual_tokens" in batch:
            return DINOVisualTokenizerOutput(
                compressed_tokens=batch["compressed_visual_tokens"],
                patch_tokens={},
                cls_tokens={},
            )

        # Next: pre-extracted DINO patch features
        if "pre_extracted_dinov2" in batch:
            return self.compress_from_patch_features(batch["pre_extracted_dinov2"])

        # Fallback: raw images
        patch_dict = {}
        for cam_key in self.config.image_features.keys():
            patch_tokens, cls_token = self._extract_dino(batch[cam_key])
            patch_dict[cam_key] = {
                "x_norm_patchtokens": patch_tokens,
                "x_norm_clstoken": cls_token,
            }

        return self.compress_from_patch_features(patch_dict)
