from __future__ import annotations

"""
Example usage:
  python -m models.flow_matching_wm.score \
    --config configs/needle_pickup/flow_matching_wm_score.yaml \
    --checkpoint ../fomofd/tasks/needle_pickup/model/fm_wm/needle_pickup_flow_matching_wm/checkpoints/best.pt \
    --output_dir ../fomofd/tasks/needle_pickup/prl_data/score_result \
    --split all \
    --mode reactive

  python -m models.flow_matching_wm.score \
    --config configs/needle_pickup/flow_matching_wm_score.yaml \
    --checkpoint ../fomofd/tasks/needle_pickup/model/fm_wm/needle_pickup_flow_matching_wm/checkpoints/best.pt \
    --output_dir ../fomofd/tasks/needle_pickup/prl_data/score_result \
    --split all \
    --mode proactive

  python -m models.flow_matching_wm.score \
    --config ../fomofd/tasks/needle_pickup/model/fm_wm/needle_pickup_flow_matching_wm/flow_matching_wm.yaml \
    --checkpoint ../fomofd/tasks/needle_pickup/model/fm_wm/needle_pickup_flow_matching_wm/checkpoints/best.pt \
    --output_dir ../fomofd/tasks/needle_pickup/model/fm_wm/needle_pickup_flow_matching_wm/copy_last_val \
    --split val \
    --mode copy_last_baseline

  python -m models.flow_matching_wm.score \
    --config ../fomofd/tasks/needle_pickup/model/fm_wm/needle_pickup_flow_matching_wm/flow_matching_wm.yaml \
    --checkpoint ../fomofd/tasks/needle_pickup/model/fm_wm/needle_pickup_flow_matching_wm/checkpoints/best.pt \
    --output_dir ../fomofd/tasks/needle_pickup/model/fm_wm/needle_pickup_flow_matching_wm/prediction_eval_val \
    --split val \
    --mode prediction_eval
"""

import argparse
import json
from tqdm.auto import tqdm
from pathlib import Path
from typing import Any
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdiffeq import odeint

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Please `pip install pyyaml`.") from exc

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

from .config import (
    ActionConditionerConfig,
    AuxiliaryHeadConfig,
    BackboneConfig,
    DataConfig,
    FlowMatchingWorldModelConfig,
    FMHeadConfig,
    LossConfig,
    OptimConfig,
    RuntimeConfig,
    StateConditionerConfig,
)
from .dataset import (
    EndpointWindowDataset,
    build_episode_index,
    default_model_inputs_from_batch,
    move_batch_to_device,
    split_episode_ids,
    select_training_episode_ids,
)
from .model import FlowMatchingWorldModel, standard_normal_logprob


# ----------------------------
# Config loading
# ----------------------------

def infer_train_config_path_from_checkpoint(checkpoint_path: str | Path) -> Path:
    """
    Example:
      .../needle_pickup_flow_matching_wm_crc/checkpoints/best.pt
    ->
      .../needle_pickup_flow_matching_wm_crc/config.json
    Legacy fallback: flow_matching_wm.yaml in the same run directory.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent.parent
    # The trainer always saves config.json with the effective configuration,
    # including CLI overrides. Source filenames vary between experiment groups.
    for name in ("config.json", "flow_matching_wm.yaml"):
        train_cfg = run_dir / name
        if train_cfg.exists():
            return train_cfg
    raise FileNotFoundError(f"No config.json or flow_matching_wm.yaml next to checkpoint: {run_dir}")


def deep_update_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursive dict merge:
      - nested dicts are merged
      - scalars/lists are replaced by override
    """
    out = deepcopy(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = deep_update_dict(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_merged_config(
    score_config_path: str | Path,
    checkpoint_path: str | Path,
    score_config_key: str | None = None,
    train_config_key: str | None = None,
) -> tuple[FlowMatchingWorldModelConfig, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Precedence:
      training config from checkpoint run dir
      < overwritten by >
      score config provided on CLI
    """
    train_config_path = infer_train_config_path_from_checkpoint(checkpoint_path)

    raw_train = _load_text_config(train_config_path)
    node_train = _extract_node(raw_train, config_key=train_config_key)

    raw_score = _load_text_config(score_config_path)
    node_score = _extract_node(raw_score, config_key=score_config_key)

    merged_node = deep_update_dict(node_train, node_score)
    config = build_config(merged_node)

    return config, train_config_path, node_train, node_score, merged_node


def _load_text_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a dict.")
    return data


def _extract_node(raw: dict[str, Any], config_key: str | None = None) -> dict[str, Any]:
    if config_key is not None:
        if config_key not in raw:
            raise KeyError(f"config_key='{config_key}' not found in config file.")
        node = raw[config_key]
    elif "fm_wm" in raw:
        node = raw["fm_wm"]
    else:
        node = raw

    if not isinstance(node, dict):
        raise ValueError("FM-WM config node must be a dict.")
    return dict(node)


def build_config(raw: dict[str, Any]) -> FlowMatchingWorldModelConfig:
    cfg = FlowMatchingWorldModelConfig(
        data=DataConfig(**raw["data"]),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        optim=OptimConfig(**raw.get("optim", {})),
        state_conditioner=StateConditionerConfig(**raw.get("state_conditioner", {})),
        action_conditioner=ActionConditionerConfig(**raw.get("action_conditioner", {})),
        backbone=BackboneConfig(**raw.get("backbone", {})),
        fm_head=FMHeadConfig(**raw.get("fm_head", {})),
        aux_head=AuxiliaryHeadConfig(**raw.get("aux_head", {})),
        loss=LossConfig(**raw.get("loss", {})),
        experiment_name=raw.get("experiment_name", "flow_matching_world_model_map"),
        seed=int(raw.get("seed", 0)),
    )
    cfg.validate()
    return cfg


def load_config(config_path: str | Path, config_key: str | None = None) -> FlowMatchingWorldModelConfig:
    raw = _load_text_config(config_path)
    node = _extract_node(raw, config_key=config_key)
    return build_config(node)


# ----------------------------
# Utils
# ----------------------------

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def get_amp_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported amp_dtype: {name}")


def _loader_progress(
    loader: DataLoader,
    *,
    desc: str,
    max_batches: int | None = None,
):
    total = len(loader)
    if max_batches is not None:
        total = min(total, max_batches)
    return tqdm(loader, total=total, desc=desc, leave=True)


def _truncate_action_chunk_batch(
    proactive_chunk_batch: torch.Tensor,
    proactive_num_action_chunks: int | None,
) -> torch.Tensor:
    """
    proactive_chunk_batch: [B,S,K,A]
    If proactive_num_action_chunks is provided, keep only the first x chunks.
    """
    if proactive_chunk_batch.ndim != 4:
        raise ValueError(
            f"Expected proactive_action_chunk_batch [B,S,K,A], got {tuple(proactive_chunk_batch.shape)}"
        )

    if proactive_num_action_chunks is None:
        return proactive_chunk_batch

    proactive_num_action_chunks = int(proactive_num_action_chunks)
    if proactive_num_action_chunks < 1:
        raise ValueError(
            f"proactive_num_action_chunks must be >= 1 when provided, got {proactive_num_action_chunks}"
        )

    return proactive_chunk_batch[:, :proactive_num_action_chunks]


def _tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()
    if x.is_floating_point():
        x = x.float()
    return x.numpy()


def _ensure_view_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Input:
      single-view: [B,H,W,C]
      multi-view : [B,V,H,W,C]
    Output:
      always     : [B,V,H,W,C]
    """
    if x.ndim == 4:
        return x.unsqueeze(1)
    if x.ndim == 5:
        return x
    raise ValueError(f"Expected rank-4 or rank-5 tensor, got shape {tuple(x.shape)}")


def _repeat_first_dim(x: torch.Tensor | None, repeats: int) -> torch.Tensor | None:
    if x is None:
        return None
    return x.repeat_interleave(repeats, dim=0)


def _normalize_endpoint_samples(
    samples: torch.Tensor,
    per_chunk_wm_samples: int,
) -> torch.Tensor:
    """
    Normalize sample_endpoint_latent(...) output so it always has an explicit WM-sample axis.

    Input possibilities:
      per_chunk_wm_samples == 1:
        single-view: [N,H,W,C]
        multi-view : [N,V,H,W,C]

      per_chunk_wm_samples > 1:
        single-view: [N,S,H,W,C]
        multi-view : [N,S,V,H,W,C]

    Output:
      single-view: [N,S,H,W,C]
      multi-view : [N,S,V,H,W,C]
    """
    if per_chunk_wm_samples < 1:
        raise ValueError(f"per_chunk_wm_samples must be >= 1, got {per_chunk_wm_samples}")

    if per_chunk_wm_samples == 1:
        if samples.ndim == 4:
            return samples.unsqueeze(1)   # [N,1,H,W,C]
        if samples.ndim == 5:
            return samples.unsqueeze(1)   # [N,1,V,H,W,C]
        raise ValueError(
            f"Unexpected sample rank for per_chunk_wm_samples=1: {tuple(samples.shape)}"
        )

    if samples.ndim == 5:
        return samples                    # [N,S,H,W,C]
    if samples.ndim == 6:
        return samples                    # [N,S,V,H,W,C]

    raise ValueError(
        f"Unexpected sample rank for per_chunk_wm_samples={per_chunk_wm_samples}: {tuple(samples.shape)}"
    )


def _compute_reactive_scores_from_eps_hat(
    eps_hat: torch.Tensor,
    score_kind: str,
    base_std: float,
) -> dict[str, torch.Tensor]:
    """
    eps_hat: [B,V,H,W,C] or [B,H,W,C]
    """
    eps_hat = _ensure_view_dim(eps_hat)  # [B,V,H,W,C]
    flat_view = eps_hat.flatten(2)       # [B,V,D]
    flat_union = eps_hat.flatten(1)      # [B,V*D]

    log_prob_view = standard_normal_logprob(flat_view, std=base_std).sum(dim=-1)   # [B,V]
    log_prob_union = standard_normal_logprob(flat_union, std=base_std).sum(dim=-1) # [B]

    if score_kind == "base_logp":
        score_view = -log_prob_view
        score_union = -log_prob_union
    elif score_kind == "squared_norm":
        score_view = ((flat_view / base_std) ** 2).sum(dim=-1)
        score_union = ((flat_union / base_std) ** 2).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported score_kind: {score_kind}")

    return {
        "log_prob_view": log_prob_view,
        "log_prob_union": log_prob_union,
        "score_view": score_view,
        "score_union": score_union,
    }


def _compute_proactive_scores_from_samples(
    samples: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    samples:
      [B,S,V,H,W,C] or [B,S,H,W,C]
    Returns variance-trace style uncertainty scores, per-view and union.
    """
    if samples.ndim == 5:
        samples = samples.unsqueeze(2)  # [B,S,1,H,W,C]
    if samples.ndim != 6:
        raise ValueError(f"Expected sample tensor rank 5 or 6, got {tuple(samples.shape)}")

    var = samples.var(dim=1, unbiased=False)       # [B,V,H,W,C]
    var_view = var.flatten(2).sum(dim=-1)          # [B,V]
    var_union = var.flatten(1).sum(dim=-1)         # [B]
    return {
        "var_map": var,
        "score_view": var_view,
        "score_union": var_union,
    }


def _predict_endpoint_latent_center(
    model: FlowMatchingWorldModel,
    *,
    fix_hist: torch.Tensor | None,
    wrist_hist: torch.Tensor | None,
    state_hist: torch.Tensor | None,
    action_chunk: torch.Tensor,
) -> torch.Tensor:
    """
    Deterministic endpoint prediction from the center of the FM base normal.

    sample_endpoint_latent(...) draws x0 ~ N(0, base_std^2 I). For predictive
    scoring, use x0 = 0 and integrate the same ODE from t=0 to t=1.
    """
    h = model.config.data.latent_grid_height
    w = model.config.data.latent_grid_width
    c = model.config.data.latent_channels
    v = model._num_views() if hasattr(model, "_num_views") else int(model.config.data.num_cameras)

    b = action_chunk.shape[0]
    if v == 1:
        x0 = torch.zeros(b, h, w, c, device=action_chunk.device, dtype=action_chunk.dtype)
    else:
        x0 = torch.zeros(b, v, h, w, c, device=action_chunk.device, dtype=action_chunk.dtype)

    model.odefunc.set_context(fix_hist, wrist_hist, state_hist, action_chunk)
    t = torch.tensor([0.0, 1.0], device=x0.device, dtype=x0.dtype)
    traj = odeint(
        model.odefunc,
        x0,
        t,
        rtol=model.config.fm_head.ode_rtol,
        atol=model.config.fm_head.ode_atol,
        method=model.config.fm_head.ode_solver,
    )
    return traj[-1]


def _stack_future_views_from_inputs(inputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
    views: list[torch.Tensor] = []
    if inputs.get("future_fix") is not None:
        views.append(inputs["future_fix"])
    if inputs.get("future_wrist") is not None:
        views.append(inputs["future_wrist"])

    if len(views) == 0:
        raise RuntimeError("Predictive scoring requires at least one future observation view.")
    if len(views) == 1:
        return views[0].unsqueeze(1)
    return torch.stack(views, dim=1)


def _compute_predictive_scores_from_prediction(
    pred_future: torch.Tensor,
    x_future: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Difference-based predictive nonconformity.

    Main score = per-view squared L2 error summed over latent dimensions.
    Union score = sum over views.
    """
    pred_future = _ensure_view_dim(pred_future)
    x_future = _ensure_view_dim(x_future)

    if pred_future.shape != x_future.shape:
        raise ValueError(
            f"pred_future shape {tuple(pred_future.shape)} does not match "
            f"x_future shape {tuple(x_future.shape)}"
        )

    err = pred_future - x_future
    err_flat_view = err.flatten(2)

    score_view = err_flat_view.square().sum(dim=-1)     # [B,V]
    mse_view = err_flat_view.square().mean(dim=-1)      # [B,V]
    l2_view = torch.norm(err_flat_view, dim=-1)         # [B,V]

    score_union = score_view.sum(dim=-1)                # [B]
    mse_union = err.flatten(1).square().mean(dim=-1)    # [B]
    l2_union = torch.norm(err.flatten(1), dim=-1)       # [B]

    return {
        "pred_err": err,
        "score_view": score_view,
        "score_union": score_union,
        "mse_view": mse_view,
        "mse_union": mse_union,
        "l2_view": l2_view,
        "l2_union": l2_union,
    }


def _compute_ace_scores_from_action_chunk_batch(
    action_chunk_batch: torch.Tensor,
    cov_eps: float,
) -> dict[str, torch.Tensor]:
    """
    action_chunk_batch: [B, S, K, A]
    Returns Gaussian-entropy style ACE score over sampled action chunks.
    """
    if action_chunk_batch.ndim != 4:
        raise ValueError(
            f"Expected proactive action chunk batch [B,S,K,A], got {tuple(action_chunk_batch.shape)}"
        )

    b, s, k, a = action_chunk_batch.shape
    d = k * a

    x = action_chunk_batch.reshape(b, s, d)              # [B,S,D]
    x = x - x.mean(dim=1, keepdim=True)

    if s == 1:
        score_union = torch.zeros((b,), device=x.device, dtype=x.dtype)
        score_view = score_union.unsqueeze(1)
        return {
            "score_union": score_union,
            "score_view": score_view,
        }

    cov = torch.matmul(x.transpose(1, 2), x) / float(s - 1)  # [B,D,D]

    eye = torch.eye(d, device=x.device, dtype=x.dtype).unsqueeze(0)
    cov = cov + cov_eps * eye

    sign, logabsdet = torch.linalg.slogdet(cov)
    if not torch.all(sign > 0):
        raise RuntimeError("ACE covariance is not positive definite even after diagonal jitter.")

    const = d * (1.0 + np.log(2.0 * np.pi))
    score_union = 0.5 * (const + logabsdet)   # [B]

    # keep shape compatible with existing proactive plotting/summaries
    score_view = score_union.unsqueeze(1)      # [B,1]

    return {
        "score_union": score_union,
        "score_view": score_view,
    }


def _split_view_arrays(x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    x:
      [N,V,...] or [N,...] if already single-view collapsed
    Returns:
      fix, wrist_or_none
    """
    if x.ndim >= 2 and x.shape[1] == 1:
        return x[:, 0], None
    if x.ndim >= 2 and x.shape[1] == 2:
        return x[:, 0], x[:, 1]
    raise ValueError(f"Expected view dimension of size 1 or 2, got shape {x.shape}")


# ----------------------------
# Dataset / model loading
# ----------------------------

def build_inference_loader(
    config: FlowMatchingWorldModelConfig,
    split: str,
    batch_size: int | None = None,
    include_proactive: bool = False,
) -> DataLoader:
    ds = LeRobotDataset(
        repo_id=config.data.dataset_repo_id,
        root=config.data.dataset_root,
    )
    episode_to_indices = build_episode_index(ds)
    episode_ids = sorted(episode_to_indices.keys())
    train_episode_ids, val_episode_ids = split_episode_ids(
        episode_ids,
        val_ratio=config.data.val_ratio,
        seed=config.data.split_seed,
    )

    if split == "train":
        chosen = select_training_episode_ids(
            train_episode_ids,
            fraction=config.data.train_episode_fraction,
            seed=config.data.train_subset_seed,
        )
    elif split == "val":
        chosen = val_episode_ids
    elif split == "all":
        chosen = episode_ids
    else:
        raise ValueError(f"Unsupported split: {split}")

    window_ds = EndpointWindowDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=chosen,
        config=config,
        include_proactive=include_proactive,
    )

    bs = (
        batch_size
        if batch_size is not None
        else (
            config.runtime.val_batch_size
            if config.runtime.val_batch_size is not None
            else config.runtime.batch_size
        )
    )

    loader = DataLoader(
        window_ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=config.runtime.num_workers,
        pin_memory=(config.runtime.pin_memory if config.runtime.pin_memory is not None else True),
        persistent_workers=(config.runtime.persistent_workers if config.runtime.num_workers > 0 else False),
    )
    return loader


def load_model_from_checkpoint(
    config: FlowMatchingWorldModelConfig,
    checkpoint_path: str | Path,
    device: torch.device,
) -> FlowMatchingWorldModel:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model = FlowMatchingWorldModel(config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


# ----------------------------
# Summaries
# ----------------------------

def _summary_stats(name: str, x: np.ndarray) -> dict[str, Any]:
    return {
        f"{name}_mean": float(x.mean()),
        f"{name}_std": float(x.std()),
        f"{name}_min": float(x.min()),
        f"{name}_max": float(x.max()),
    }


def summarize_reactive(results: dict[str, np.ndarray]) -> dict[str, Any]:
    x_future = results["x_future"]  # [N,V,H,W,C]
    eps_hat = results["eps_hat"]    # [N,V,H,W,C]
    n, v, h, w, c = x_future.shape

    summary: dict[str, Any] = {
        "num_windows": int(n),
        "num_views": int(v),
        "latent_grid_height": int(h),
        "latent_grid_width": int(w),
        "latent_channels": int(c),
    }

    summary.update(_summary_stats("score", results["scores_union"]))       # legacy alias
    summary.update(_summary_stats("score_union", results["scores_union"]))
    summary.update(_summary_stats("log_prob", results["log_prob_union"]))   # legacy alias
    summary.update(_summary_stats("log_prob_union", results["log_prob_union"]))

    eps_hat_norm_union = np.linalg.norm(eps_hat.reshape(n, -1), axis=1)
    x_future_norm_union = np.linalg.norm(x_future.reshape(n, -1), axis=1)
    summary.update(_summary_stats("eps_hat_norm", eps_hat_norm_union))      # legacy alias
    summary.update(_summary_stats("eps_hat_norm_union", eps_hat_norm_union))
    summary.update(_summary_stats("x_future_norm", x_future_norm_union))    # legacy alias
    summary.update(_summary_stats("x_future_norm_union", x_future_norm_union))

    eps_fix, eps_wrist = _split_view_arrays(eps_hat)
    fut_fix, fut_wrist = _split_view_arrays(x_future)
    lp_fix, lp_wrist = _split_view_arrays(results["log_prob_view"])
    sc_fix, sc_wrist = _split_view_arrays(results["scores_view"])

    summary.update(_summary_stats("score_fix", sc_fix))
    summary.update(_summary_stats("log_prob_fix", lp_fix))
    summary.update(_summary_stats("eps_hat_norm_fix", np.linalg.norm(eps_fix.reshape(n, -1), axis=1)))
    summary.update(_summary_stats("x_future_norm_fix", np.linalg.norm(fut_fix.reshape(n, -1), axis=1)))

    if sc_wrist is not None:
        summary.update(_summary_stats("score_wrist", sc_wrist))
        summary.update(_summary_stats("log_prob_wrist", lp_wrist))
        summary.update(_summary_stats("eps_hat_norm_wrist", np.linalg.norm(eps_wrist.reshape(n, -1), axis=1)))
        summary.update(_summary_stats("x_future_norm_wrist", np.linalg.norm(fut_wrist.reshape(n, -1), axis=1)))

    return summary


def summarize_proactive(results: dict[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_windows": int(results["scores_union"].shape[0]),
        "num_views": int(results["scores_view"].shape[1]),
    }
    summary.update(_summary_stats("score", results["scores_union"]))       # legacy alias
    summary.update(_summary_stats("score_union", results["scores_union"]))

    sc_fix, sc_wrist = _split_view_arrays(results["scores_view"])
    summary.update(_summary_stats("score_fix", sc_fix))
    if sc_wrist is not None:
        summary.update(_summary_stats("score_wrist", sc_wrist))
    return summary


def summarize_predictive(results: dict[str, np.ndarray]) -> dict[str, Any]:
    x_future = results["x_future"]
    n, v, h, w, c = x_future.shape

    summary: dict[str, Any] = {
        "num_windows": int(n),
        "num_views": int(v),
        "latent_grid_height": int(h),
        "latent_grid_width": int(w),
        "latent_channels": int(c),
        "score_definition": "sum_squared_error(predicted_endpoint_latent_from_x0_zero, actual_future_latent)",
    }

    summary.update(_summary_stats("score", results["scores_union"]))
    summary.update(_summary_stats("score_union", results["scores_union"]))
    summary.update(_summary_stats("mse_union", results["mse_union"]))
    summary.update(_summary_stats("l2_union", results["l2_union"]))

    sc_fix, sc_wrist = _split_view_arrays(results["scores_view"])
    mse_fix, mse_wrist = _split_view_arrays(results["mse_view"])
    l2_fix, l2_wrist = _split_view_arrays(results["l2_view"])

    summary.update(_summary_stats("score_fix", sc_fix))
    summary.update(_summary_stats("mse_fix", mse_fix))
    summary.update(_summary_stats("l2_fix", l2_fix))

    if sc_wrist is not None:
        summary.update(_summary_stats("score_wrist", sc_wrist))
        summary.update(_summary_stats("mse_wrist", mse_wrist))
        summary.update(_summary_stats("l2_wrist", l2_wrist))

    return summary


# ----------------------------
# Scoring
# ----------------------------

@torch.no_grad()
def run_reactive_scoring(
    model: FlowMatchingWorldModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    all_episode_index: list[np.ndarray] = []
    all_anchor_frame_index: list[np.ndarray] = []

    all_x_future: list[np.ndarray] = []
    all_eps_hat: list[np.ndarray] = []
    all_log_prob_union: list[np.ndarray] = []
    all_log_prob_view: list[np.ndarray] = []
    all_scores_union: list[np.ndarray] = []
    all_scores_view: list[np.ndarray] = []

    autocast_enabled = (device.type == "cuda" and use_amp)

    for batch_idx, batch in enumerate(_loader_progress(loader, desc="Reactive scoring", max_batches=max_batches)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        meta_episode_index = batch["episode_index"].cpu().numpy()
        meta_anchor_frame_index = batch["anchor_frame_index"].cpu().numpy()

        batch = move_batch_to_device(batch, device)
        inputs = default_model_inputs_from_batch(batch, include_proactive=False)

        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model.compute_reactive_nonconformity_scores(
                    fix_hist=inputs["fix_hist"],
                    wrist_hist=inputs["wrist_hist"],
                    state_hist=inputs["state_hist"],
                    action_chunk=inputs["action_chunk"],
                    future_fix=inputs["future_fix"],
                    future_wrist=inputs["future_wrist"],
                    future_state=inputs["future_state"],
                )
                score_parts = _compute_reactive_scores_from_eps_hat(
                    out["eps_hat"],
                    score_kind=model.config.fm_head.score_kind,
                    base_std=model.config.fm_head.base_std,
                )
        else:
            out = model.compute_reactive_nonconformity_scores(
                fix_hist=inputs["fix_hist"],
                wrist_hist=inputs["wrist_hist"],
                state_hist=inputs["state_hist"],
                action_chunk=inputs["action_chunk"],
                future_fix=inputs["future_fix"],
                future_wrist=inputs["future_wrist"],
                future_state=inputs["future_state"],
            )
            score_parts = _compute_reactive_scores_from_eps_hat(
                out["eps_hat"],
                score_kind=model.config.fm_head.score_kind,
                base_std=model.config.fm_head.base_std,
            )

        x_future = _ensure_view_dim(out["x_future"])
        eps_hat = _ensure_view_dim(out["eps_hat"])

        all_episode_index.append(meta_episode_index)
        all_anchor_frame_index.append(meta_anchor_frame_index)
        all_x_future.append(_tensor_to_numpy(x_future))
        all_eps_hat.append(_tensor_to_numpy(eps_hat))
        all_log_prob_union.append(_tensor_to_numpy(score_parts["log_prob_union"]))
        all_log_prob_view.append(_tensor_to_numpy(score_parts["log_prob_view"]))
        all_scores_union.append(_tensor_to_numpy(score_parts["score_union"]))
        all_scores_view.append(_tensor_to_numpy(score_parts["score_view"]))

    scores_union = np.concatenate(all_scores_union, axis=0)
    log_prob_union = np.concatenate(all_log_prob_union, axis=0)

    return {
        "episode_index": np.concatenate(all_episode_index, axis=0),
        "anchor_frame_index": np.concatenate(all_anchor_frame_index, axis=0),
        "x_future": np.concatenate(all_x_future, axis=0),
        "eps_hat": np.concatenate(all_eps_hat, axis=0),
        "log_prob_union": log_prob_union,
        "log_prob_view": np.concatenate(all_log_prob_view, axis=0),
        "scores_union": scores_union,
        "scores_view": np.concatenate(all_scores_view, axis=0),
        "log_prob": log_prob_union,
        "scores": scores_union,
    }


@torch.no_grad()
def run_proactive_scoring(
    model: FlowMatchingWorldModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    all_episode_index: list[np.ndarray] = []
    all_anchor_frame_index: list[np.ndarray] = []
    all_scores_union: list[np.ndarray] = []
    all_scores_view: list[np.ndarray] = []

    proactive_mode = getattr(model.config.fm_head, "proactive_score_mode", "fm_wm_entropy")
    per_chunk_wm_samples = int(getattr(model.config.fm_head, "per_chunk_wm_samples", 1))
    proactive_num_action_chunks = getattr(model.config.fm_head, "proactive_num_action_chunks", None)

    if per_chunk_wm_samples < 1:
        raise ValueError(f"per_chunk_wm_samples must be >= 1, got {per_chunk_wm_samples}")

    if proactive_num_action_chunks is not None and int(proactive_num_action_chunks) < 1:
        raise ValueError(
            f"proactive_num_action_chunks must be >= 1 when provided, got {proactive_num_action_chunks}"
        )

    autocast_enabled = (device.type == "cuda" and use_amp)

    for batch_idx, batch in enumerate(_loader_progress(loader, desc="Proactive scoring", max_batches=max_batches)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        meta_episode_index = batch["episode_index"].cpu().numpy()
        meta_anchor_frame_index = batch["anchor_frame_index"].cpu().numpy()

        batch = move_batch_to_device(batch, device)
        inputs = default_model_inputs_from_batch(batch, include_proactive=True)

        if proactive_mode == "fm_wm_entropy":
            proactive_chunk = inputs.get("proactive_action_chunk")
            if proactive_chunk is None:
                proactive_chunk = inputs["action_chunk"]

            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    samples = model.sample_endpoint_latent(
                        fix_hist=inputs["fix_hist"],
                        wrist_hist=inputs["wrist_hist"],
                        state_hist=inputs["state_hist"],
                        action_chunk=proactive_chunk,
                        num_samples=per_chunk_wm_samples,
                    )
            else:
                samples = model.sample_endpoint_latent(
                    fix_hist=inputs["fix_hist"],
                    wrist_hist=inputs["wrist_hist"],
                    state_hist=inputs["state_hist"],
                    action_chunk=proactive_chunk,
                    num_samples=per_chunk_wm_samples,
                )

            samples = _normalize_endpoint_samples(samples, per_chunk_wm_samples)
            score_parts = _compute_proactive_scores_from_samples(samples)

        elif proactive_mode == "ace":
            proactive_chunk_batch = inputs.get("proactive_action_chunk_batch")
            if proactive_chunk_batch is None:
                raise RuntimeError(
                    "ACE mode requires proactive_action_chunk_batch in the scoring dataset."
                )

            proactive_chunk_batch = _truncate_action_chunk_batch(
                proactive_chunk_batch,
                proactive_num_action_chunks,
            )

            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    score_parts = _compute_ace_scores_from_action_chunk_batch(
                        proactive_chunk_batch,
                        cov_eps=float(getattr(model.config.fm_head, "ace_cov_eps", 1.0e-6)),
                    )
            else:
                score_parts = _compute_ace_scores_from_action_chunk_batch(
                    proactive_chunk_batch,
                    cov_eps=float(getattr(model.config.fm_head, "ace_cov_eps", 1.0e-6)),
                )

        elif proactive_mode == "policy_fm_wm_entropy":
            proactive_chunk_batch = inputs.get("proactive_action_chunk_batch")
            if proactive_chunk_batch is None:
                raise RuntimeError(
                    "policy_fm_wm_entropy mode requires proactive_action_chunk_batch in the scoring dataset."
                )

            proactive_chunk_batch = _truncate_action_chunk_batch(
                proactive_chunk_batch,
                proactive_num_action_chunks,
            )

            if proactive_chunk_batch.ndim != 4:
                raise ValueError(
                    f"Expected proactive_action_chunk_batch [B,S,K,A], got {tuple(proactive_chunk_batch.shape)}"
                )

            bsz, num_action_chunks, chunk_len, action_dim = proactive_chunk_batch.shape

            flat_chunk_batch = proactive_chunk_batch.reshape(
                bsz * num_action_chunks,
                chunk_len,
                action_dim,
            )  # [B*S_action, K, A]

            fix_hist_rep = _repeat_first_dim(inputs["fix_hist"], num_action_chunks)
            wrist_hist_rep = _repeat_first_dim(inputs["wrist_hist"], num_action_chunks)
            state_hist_rep = _repeat_first_dim(inputs["state_hist"], num_action_chunks)

            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    future_samples = model.sample_endpoint_latent(
                        fix_hist=fix_hist_rep,
                        wrist_hist=wrist_hist_rep,
                        state_hist=state_hist_rep,
                        action_chunk=flat_chunk_batch,
                        num_samples=per_chunk_wm_samples,
                    )
            else:
                future_samples = model.sample_endpoint_latent(
                    fix_hist=fix_hist_rep,
                    wrist_hist=wrist_hist_rep,
                    state_hist=state_hist_rep,
                    action_chunk=flat_chunk_batch,
                    num_samples=per_chunk_wm_samples,
                )

            future_samples = _normalize_endpoint_samples(
                future_samples,
                per_chunk_wm_samples,
            )

            if future_samples.ndim == 5:
                # [B*S_action, S_wm, H, W, C]
                h, w, c = future_samples.shape[-3:]
                future_samples = future_samples.reshape(
                    bsz,
                    num_action_chunks,
                    per_chunk_wm_samples,
                    h,
                    w,
                    c,
                )
                future_samples = future_samples.reshape(
                    bsz,
                    num_action_chunks * per_chunk_wm_samples,
                    h,
                    w,
                    c,
                )
            elif future_samples.ndim == 6:
                # [B*S_action, S_wm, V, H, W, C]
                v, h, w, c = future_samples.shape[-4:]
                future_samples = future_samples.reshape(
                    bsz,
                    num_action_chunks,
                    per_chunk_wm_samples,
                    v,
                    h,
                    w,
                    c,
                )
                future_samples = future_samples.reshape(
                    bsz,
                    num_action_chunks * per_chunk_wm_samples,
                    v,
                    h,
                    w,
                    c,
                )
            else:
                raise ValueError(
                    f"Unexpected future_samples rank for policy_fm_wm_entropy: {tuple(future_samples.shape)}"
                )

            score_parts = _compute_proactive_scores_from_samples(future_samples)

        else:
            raise ValueError(f"Unsupported proactive_score_mode: {proactive_mode}")

        all_episode_index.append(meta_episode_index)
        all_anchor_frame_index.append(meta_anchor_frame_index)
        all_scores_union.append(_tensor_to_numpy(score_parts["score_union"]))
        all_scores_view.append(_tensor_to_numpy(score_parts["score_view"]))

    scores_union = np.concatenate(all_scores_union, axis=0)

    return {
        "episode_index": np.concatenate(all_episode_index, axis=0),
        "anchor_frame_index": np.concatenate(all_anchor_frame_index, axis=0),
        "scores_union": scores_union,
        "scores_view": np.concatenate(all_scores_view, axis=0),
        "scores": scores_union,
    }


@torch.no_grad()
def run_predictive_scoring(
    model: FlowMatchingWorldModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    all_episode_index: list[np.ndarray] = []
    all_anchor_frame_index: list[np.ndarray] = []

    all_x_future: list[np.ndarray] = []
    all_pred_future: list[np.ndarray] = []
    all_pred_err: list[np.ndarray] = []
    all_scores_union: list[np.ndarray] = []
    all_scores_view: list[np.ndarray] = []
    all_mse_union: list[np.ndarray] = []
    all_mse_view: list[np.ndarray] = []
    all_l2_union: list[np.ndarray] = []
    all_l2_view: list[np.ndarray] = []

    autocast_enabled = (device.type == "cuda" and use_amp)

    for batch_idx, batch in enumerate(_loader_progress(loader, desc="Predictive scoring", max_batches=max_batches)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        meta_episode_index = batch["episode_index"].cpu().numpy()
        meta_anchor_frame_index = batch["anchor_frame_index"].cpu().numpy()

        batch = move_batch_to_device(batch, device)
        inputs = default_model_inputs_from_batch(batch, include_proactive=False)

        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                pred_future = _predict_endpoint_latent_center(
                    model,
                    fix_hist=inputs["fix_hist"],
                    wrist_hist=inputs["wrist_hist"],
                    state_hist=inputs["state_hist"],
                    action_chunk=inputs["action_chunk"],
                )
                x_future = _stack_future_views_from_inputs(inputs)
                score_parts = _compute_predictive_scores_from_prediction(pred_future, x_future)
        else:
            pred_future = _predict_endpoint_latent_center(
                model,
                fix_hist=inputs["fix_hist"],
                wrist_hist=inputs["wrist_hist"],
                state_hist=inputs["state_hist"],
                action_chunk=inputs["action_chunk"],
            )
            x_future = _stack_future_views_from_inputs(inputs)
            score_parts = _compute_predictive_scores_from_prediction(pred_future, x_future)

        pred_future = _ensure_view_dim(pred_future)
        x_future = _ensure_view_dim(x_future)

        all_episode_index.append(meta_episode_index)
        all_anchor_frame_index.append(meta_anchor_frame_index)
        all_x_future.append(_tensor_to_numpy(x_future))
        all_pred_future.append(_tensor_to_numpy(pred_future))
        all_pred_err.append(_tensor_to_numpy(score_parts["pred_err"]))
        all_scores_union.append(_tensor_to_numpy(score_parts["score_union"]))
        all_scores_view.append(_tensor_to_numpy(score_parts["score_view"]))
        all_mse_union.append(_tensor_to_numpy(score_parts["mse_union"]))
        all_mse_view.append(_tensor_to_numpy(score_parts["mse_view"]))
        all_l2_union.append(_tensor_to_numpy(score_parts["l2_union"]))
        all_l2_view.append(_tensor_to_numpy(score_parts["l2_view"]))

    scores_union = np.concatenate(all_scores_union, axis=0)

    return {
        "episode_index": np.concatenate(all_episode_index, axis=0),
        "anchor_frame_index": np.concatenate(all_anchor_frame_index, axis=0),
        "x_future": np.concatenate(all_x_future, axis=0),
        "pred_future": np.concatenate(all_pred_future, axis=0),
        "pred_err": np.concatenate(all_pred_err, axis=0),
        "scores_union": scores_union,
        "scores_view": np.concatenate(all_scores_view, axis=0),
        "mse_union": np.concatenate(all_mse_union, axis=0),
        "mse_view": np.concatenate(all_mse_view, axis=0),
        "l2_union": np.concatenate(all_l2_union, axis=0),
        "l2_view": np.concatenate(all_l2_view, axis=0),
        "scores": scores_union,
    }


# ----------------------------
# Copy-last-baseline for evaluating world model
# ----------------------------

@torch.no_grad()
def run_copy_last_baseline(
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    all_episode_index: list[np.ndarray] = []
    all_anchor_frame_index: list[np.ndarray] = []

    all_next_mse_fix: list[np.ndarray] = []
    all_next_l2_fix: list[np.ndarray] = []
    all_endpoint_mse_fix: list[np.ndarray] = []
    all_endpoint_l2_fix: list[np.ndarray] = []

    all_next_mse_wrist: list[np.ndarray] = []
    all_next_l2_wrist: list[np.ndarray] = []
    all_endpoint_mse_wrist: list[np.ndarray] = []
    all_endpoint_l2_wrist: list[np.ndarray] = []

    use_fix = None
    use_wrist = None

    for batch_idx, batch in enumerate(_loader_progress(loader, desc="Copy-last baseline", max_batches=max_batches)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        meta_episode_index = batch["episode_index"].cpu().numpy()
        meta_anchor_frame_index = batch["anchor_frame_index"].cpu().numpy()

        batch = move_batch_to_device(batch, device)
        inputs = default_model_inputs_from_batch(batch, include_proactive=True)

        all_episode_index.append(meta_episode_index)
        all_anchor_frame_index.append(meta_anchor_frame_index)

        if inputs["fix_hist"] is not None:
            use_fix = True
            pred_fix = inputs["fix_hist"][:, -1]
            next_fix = inputs["next_fix"]
            future_fix = inputs["future_fix"]

            next_err = pred_fix - next_fix
            future_err = pred_fix - future_fix

            all_next_mse_fix.append(_tensor_to_numpy((next_err.square()).flatten(1).mean(dim=-1)))
            all_next_l2_fix.append(_tensor_to_numpy(torch.norm(next_err.flatten(1), dim=-1)))
            all_endpoint_mse_fix.append(_tensor_to_numpy((future_err.square()).flatten(1).mean(dim=-1)))
            all_endpoint_l2_fix.append(_tensor_to_numpy(torch.norm(future_err.flatten(1), dim=-1)))
        else:
            use_fix = False if use_fix is None else use_fix

        if inputs["wrist_hist"] is not None:
            use_wrist = True
            pred_wrist = inputs["wrist_hist"][:, -1]
            next_wrist = inputs["next_wrist"]
            future_wrist = inputs["future_wrist"]

            next_err = pred_wrist - next_wrist
            future_err = pred_wrist - future_wrist

            all_next_mse_wrist.append(_tensor_to_numpy((next_err.square()).flatten(1).mean(dim=-1)))
            all_next_l2_wrist.append(_tensor_to_numpy(torch.norm(next_err.flatten(1), dim=-1)))
            all_endpoint_mse_wrist.append(_tensor_to_numpy((future_err.square()).flatten(1).mean(dim=-1)))
            all_endpoint_l2_wrist.append(_tensor_to_numpy(torch.norm(future_err.flatten(1), dim=-1)))
        else:
            use_wrist = False if use_wrist is None else use_wrist

    out = {
        "episode_index": np.concatenate(all_episode_index, axis=0),
        "anchor_frame_index": np.concatenate(all_anchor_frame_index, axis=0),
    }

    if use_fix:
        out["next_mse_fix"] = np.concatenate(all_next_mse_fix, axis=0)
        out["next_l2_fix"] = np.concatenate(all_next_l2_fix, axis=0)
        out["endpoint_mse_fix"] = np.concatenate(all_endpoint_mse_fix, axis=0)
        out["endpoint_l2_fix"] = np.concatenate(all_endpoint_l2_fix, axis=0)

    if use_wrist:
        out["next_mse_wrist"] = np.concatenate(all_next_mse_wrist, axis=0)
        out["next_l2_wrist"] = np.concatenate(all_next_l2_wrist, axis=0)
        out["endpoint_mse_wrist"] = np.concatenate(all_endpoint_mse_wrist, axis=0)
        out["endpoint_l2_wrist"] = np.concatenate(all_endpoint_l2_wrist, axis=0)

    if use_fix and use_wrist:
        out["next_mse_union"] = out["next_mse_fix"] + out["next_mse_wrist"]
        out["next_l2_union"] = out["next_l2_fix"] + out["next_l2_wrist"]
        out["endpoint_mse_union"] = out["endpoint_mse_fix"] + out["endpoint_mse_wrist"]
        out["endpoint_l2_union"] = out["endpoint_l2_fix"] + out["endpoint_l2_wrist"]
    elif use_fix:
        out["next_mse_union"] = out["next_mse_fix"]
        out["next_l2_union"] = out["next_l2_fix"]
        out["endpoint_mse_union"] = out["endpoint_mse_fix"]
        out["endpoint_l2_union"] = out["endpoint_l2_fix"]
    elif use_wrist:
        out["next_mse_union"] = out["next_mse_wrist"]
        out["next_l2_union"] = out["next_l2_wrist"]
        out["endpoint_mse_union"] = out["endpoint_mse_wrist"]
        out["endpoint_l2_union"] = out["endpoint_l2_wrist"]

    return out

def summarize_copy_last(results: dict[str, np.ndarray]) -> dict[str, Any]:
    summary = {
        "num_windows": int(results["episode_index"].shape[0]),
    }

    for key, value in results.items():
        if key in {"episode_index", "anchor_frame_index"}:
            continue
        summary.update(_summary_stats(key, value))

    return summary


# ----------------------------
# Prediction evaluation mode (to be compared with copy-last baseline)
# ----------------------------
@torch.no_grad()
def run_prediction_eval(
    model: FlowMatchingWorldModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    endpoint_num_samples: int = 1,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    all_episode_index = []
    all_anchor_frame_index = []

    all_next_mse_fix, all_next_l2_fix = [], []
    all_endpoint_mse_fix, all_endpoint_l2_fix = [], []

    all_next_mse_wrist, all_next_l2_wrist = [], []
    all_endpoint_mse_wrist, all_endpoint_l2_wrist = [], []

    autocast_enabled = (device.type == "cuda" and use_amp)

    for batch_idx, batch in enumerate(_loader_progress(loader, desc="Prediction eval", max_batches=max_batches)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        meta_episode_index = batch["episode_index"].cpu().numpy()
        meta_anchor_frame_index = batch["anchor_frame_index"].cpu().numpy()

        batch = move_batch_to_device(batch, device)
        inputs = default_model_inputs_from_batch(batch, include_proactive=True)

        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                pred_next = model.predict_next_map(
                    fix_hist=inputs["fix_hist"],
                    wrist_hist=inputs["wrist_hist"],
                    state_hist=inputs["state_hist"],
                    action_chunk=inputs["action_chunk"],
                )

                pred_future = model.sample_endpoint_latent(
                    fix_hist=inputs["fix_hist"],
                    wrist_hist=inputs["wrist_hist"],
                    state_hist=inputs["state_hist"],
                    action_chunk=inputs["action_chunk"],
                    num_samples=endpoint_num_samples,
                )
        else:
            pred_next = model.predict_next_map(
                fix_hist=inputs["fix_hist"],
                wrist_hist=inputs["wrist_hist"],
                state_hist=inputs["state_hist"],
                action_chunk=inputs["action_chunk"],
            )

            pred_future = model.sample_endpoint_latent(
                fix_hist=inputs["fix_hist"],
                wrist_hist=inputs["wrist_hist"],
                state_hist=inputs["state_hist"],
                action_chunk=inputs["action_chunk"],
                num_samples=endpoint_num_samples,
            )

        if pred_next.ndim == 4:
            pred_next = pred_next.unsqueeze(1)

        pred_future = _normalize_endpoint_samples(pred_future, endpoint_num_samples)
        pred_future = pred_future.mean(dim=1)

        all_episode_index.append(meta_episode_index)
        all_anchor_frame_index.append(meta_anchor_frame_index)

        view_idx = 0
        if inputs["fix_hist"] is not None:
            next_fix = inputs["next_fix"]
            future_fix = inputs["future_fix"]

            pred_next_fix = pred_next[:, view_idx]
            pred_future_fix = pred_future[:, view_idx]

            next_err = pred_next_fix - next_fix
            future_err = pred_future_fix - future_fix

            all_next_mse_fix.append(_tensor_to_numpy(next_err.square().flatten(1).mean(dim=-1)))
            all_next_l2_fix.append(_tensor_to_numpy(torch.norm(next_err.flatten(1), dim=-1)))
            all_endpoint_mse_fix.append(_tensor_to_numpy(future_err.square().flatten(1).mean(dim=-1)))
            all_endpoint_l2_fix.append(_tensor_to_numpy(torch.norm(future_err.flatten(1), dim=-1)))
            view_idx += 1

        if inputs["wrist_hist"] is not None:
            next_wrist = inputs["next_wrist"]
            future_wrist = inputs["future_wrist"]

            pred_next_wrist = pred_next[:, view_idx]
            pred_future_wrist = pred_future[:, view_idx]

            next_err = pred_next_wrist - next_wrist
            future_err = pred_future_wrist - future_wrist

            all_next_mse_wrist.append(_tensor_to_numpy(next_err.square().flatten(1).mean(dim=-1)))
            all_next_l2_wrist.append(_tensor_to_numpy(torch.norm(next_err.flatten(1), dim=-1)))
            all_endpoint_mse_wrist.append(_tensor_to_numpy(future_err.square().flatten(1).mean(dim=-1)))
            all_endpoint_l2_wrist.append(_tensor_to_numpy(torch.norm(future_err.flatten(1), dim=-1)))

    out = {
        "episode_index": np.concatenate(all_episode_index, axis=0),
        "anchor_frame_index": np.concatenate(all_anchor_frame_index, axis=0),
    }

    if all_next_mse_fix:
        out["next_mse_fix"] = np.concatenate(all_next_mse_fix, axis=0)
        out["next_l2_fix"] = np.concatenate(all_next_l2_fix, axis=0)
        out["endpoint_mse_fix"] = np.concatenate(all_endpoint_mse_fix, axis=0)
        out["endpoint_l2_fix"] = np.concatenate(all_endpoint_l2_fix, axis=0)

    if all_next_mse_wrist:
        out["next_mse_wrist"] = np.concatenate(all_next_mse_wrist, axis=0)
        out["next_l2_wrist"] = np.concatenate(all_next_l2_wrist, axis=0)
        out["endpoint_mse_wrist"] = np.concatenate(all_endpoint_mse_wrist, axis=0)
        out["endpoint_l2_wrist"] = np.concatenate(all_endpoint_l2_wrist, axis=0)

    if "next_mse_fix" in out and "next_mse_wrist" in out:
        out["next_mse_union"] = out["next_mse_fix"] + out["next_mse_wrist"]
        out["next_l2_union"] = out["next_l2_fix"] + out["next_l2_wrist"]
        out["endpoint_mse_union"] = out["endpoint_mse_fix"] + out["endpoint_mse_wrist"]
        out["endpoint_l2_union"] = out["endpoint_l2_fix"] + out["endpoint_l2_wrist"]
    elif "next_mse_fix" in out:
        out["next_mse_union"] = out["next_mse_fix"]
        out["next_l2_union"] = out["next_l2_fix"]
        out["endpoint_mse_union"] = out["endpoint_mse_fix"]
        out["endpoint_l2_union"] = out["endpoint_l2_fix"]
    else:
        out["next_mse_union"] = out["next_mse_wrist"]
        out["next_l2_union"] = out["next_l2_wrist"]
        out["endpoint_mse_union"] = out["endpoint_mse_wrist"]
        out["endpoint_l2_union"] = out["endpoint_l2_wrist"]

    return out

def summarize_prediction_eval(results: dict[str, np.ndarray]) -> dict[str, Any]:
    summary = {"num_windows": int(results["episode_index"].shape[0])}
    for key, value in results.items():
        if key in {"episode_index", "anchor_frame_index"}:
            continue
        summary.update(_summary_stats(key, value))
    return summary


# ----------------------------
# Main
# ----------------------------

def main(args: argparse.Namespace) -> None:
    config, train_config_path, train_node, score_node, merged_node = load_merged_config(
        score_config_path=args.config,
        checkpoint_path=args.checkpoint,
        score_config_key=args.config_key,
        train_config_key=None,
    )
    device = get_device(args.device or config.runtime.device)
    amp_dtype = get_amp_dtype(config.runtime.amp_dtype)

    model = load_model_from_checkpoint(config, args.checkpoint, device=device)
    loader = build_inference_loader(
        config=config,
        split=args.split,
        batch_size=args.batch_size,
        include_proactive=(args.mode == "proactive"),
    )

    if args.mode in {"reactive", "proactive", "predictive"}:
        out_dir = ensure_dir(Path(args.output_dir) / f"score_val_{args.mode}")
    else:
        out_dir = ensure_dir(Path(args.output_dir))
    save_json(
        out_dir / "score_run_config.json",
        {
            "score_config_path": str(args.config),
            "train_config_path": str(train_config_path),
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "mode": args.mode,
            "batch_size": args.batch_size,
            "device": str(device),
            "use_amp": bool(config.runtime.use_amp and device.type == "cuda"),
            "amp_dtype": config.runtime.amp_dtype,
            "max_batches": args.max_batches,
            "score_kind": getattr(config.fm_head, "score_kind", None),
            "score_transport_mode": getattr(config.fm_head, "score_transport_mode", None),
            "use_wrist_camera": bool(config.data.use_wrist_camera),
            "num_cameras": int(config.data.num_cameras),
            "per_chunk_wm_samples": int(getattr(config.fm_head, "per_chunk_wm_samples", 1)),
            "proactive_num_action_chunks": getattr(config.fm_head, "proactive_num_action_chunks", None),
            "proactive_score_mode": getattr(config.fm_head, "proactive_score_mode", "fm_wm_entropy"),
            "proactive_action_key": getattr(config.data, "proactive_action_key", None),
            "proactive_action_batch_key": getattr(config.data, "proactive_action_batch_key", None),
        },
    )
    save_json(out_dir / "merged_flow_matching_wm_config.json", merged_node)

    if args.mode == "reactive":
        results = run_reactive_scoring(
            model=model,
            loader=loader,
            device=device,
            use_amp=config.runtime.use_amp,
            amp_dtype=amp_dtype,
            max_batches=args.max_batches,
        )

        np.savez_compressed(
            out_dir / "window_scores_reactive.npz",
            episode_index=results["episode_index"],
            anchor_frame_index=results["anchor_frame_index"],
            x_future=results["x_future"],
            eps_hat=results["eps_hat"],
            log_prob_union=results["log_prob_union"],
            log_prob_view=results["log_prob_view"],
            scores_union=results["scores_union"],
            scores_view=results["scores_view"],
            # legacy aliases
            log_prob=results["log_prob"],
            scores=results["scores"],
        )

        summary = summarize_reactive(results)
        save_json(out_dir / "summary_reactive.json", summary)

        print("Saved:")
        print(f"  {out_dir / 'window_scores_reactive.npz'}")
        print(f"  {out_dir / 'summary_reactive.json'}")
        print("Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    elif args.mode == "proactive":
        results = run_proactive_scoring(
            model=model,
            loader=loader,
            device=device,
            use_amp=config.runtime.use_amp,
            amp_dtype=amp_dtype,
            max_batches=args.max_batches,
        )

        np.savez_compressed(
            out_dir / "window_scores_proactive.npz",
            episode_index=results["episode_index"],
            anchor_frame_index=results["anchor_frame_index"],
            scores_union=results["scores_union"],
            scores_view=results["scores_view"],
            # legacy alias
            scores=results["scores"],
        )

        summary = summarize_proactive(results)
        save_json(out_dir / "summary_proactive.json", summary)

        print("Saved:")
        print(f"  {out_dir / 'window_scores_proactive.npz'}")
        print(f"  {out_dir / 'summary_proactive.json'}")
        print("Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    elif args.mode == "predictive":
        results = run_predictive_scoring(
            model=model,
            loader=loader,
            device=device,
            use_amp=config.runtime.use_amp,
            amp_dtype=amp_dtype,
            max_batches=args.max_batches,
        )

        np.savez_compressed(
            out_dir / "window_scores_predictive.npz",
            episode_index=results["episode_index"],
            anchor_frame_index=results["anchor_frame_index"],
            x_future=results["x_future"],
            pred_future=results["pred_future"],
            pred_err=results["pred_err"],
            scores_union=results["scores_union"],
            scores_view=results["scores_view"],
            mse_union=results["mse_union"],
            mse_view=results["mse_view"],
            l2_union=results["l2_union"],
            l2_view=results["l2_view"],
            scores=results["scores"],
        )

        summary = summarize_predictive(results)
        save_json(out_dir / "summary_predictive.json", summary)

        print("Saved:")
        print(f"  {out_dir / 'window_scores_predictive.npz'}")
        print(f"  {out_dir / 'summary_predictive.json'}")
        print("Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    elif args.mode == "copy_last_baseline":
        results = run_copy_last_baseline(
            loader=loader,
            device=device,
            max_batches=args.max_batches,
        )

        np.savez_compressed(
            out_dir / "copy_last_baseline.npz",
            **results,
        )

        summary = summarize_copy_last(results)
        save_json(out_dir / "summary_copy_last_baseline.json", summary)

        print("Saved:")
        print(f"  {out_dir / 'copy_last_baseline.npz'}")
        print(f"  {out_dir / 'summary_copy_last_baseline.json'}")
        print("Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    elif args.mode == "prediction_eval":
        results = run_prediction_eval(
            model=model,
            loader=loader,
            device=device,
            use_amp=config.runtime.use_amp,
            amp_dtype=amp_dtype,
            endpoint_num_samples=8,
            max_batches=args.max_batches,
        )

        np.savez_compressed(out_dir / "prediction_eval.npz", **results)
        summary = summarize_prediction_eval(results)
        save_json(out_dir / "summary_prediction_eval.json", summary)

        print("Saved:")
        print(f"  {out_dir / 'prediction_eval.npz'}")
        print(f"  {out_dir / 'summary_prediction_eval.json'}")
        print("Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_key", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--mode", type=str, default="reactive", choices=["reactive", "proactive", "predictive", "copy_last_baseline", "prediction_eval"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
