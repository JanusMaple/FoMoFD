from __future__ import annotations

"""
Example:
  python -m models.faildetect_logpzo.score \
    --config configs/needle_pickup/faildetect_logpzo_score.yaml \
    --checkpoint ../fomofd/tasks/needle_pickup/model/faildetect_logpzo/faildetect_logpzo_wrist_state_vfmfvae/checkpoints/best.pt \
    --output_dir ../fomofd/tasks/needle_pickup/prl_data/score_result/faildetect_logpzo \
    --split all
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Please `pip install pyyaml`.") from exc

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

from .config import (
    DataConfig,
    RuntimeConfig,
    OptimConfig,
    ModelConfig,
    FMHeadConfig,
    LossConfig,
    LogpZOConfig,
)
from .dataset import (
    SingleStepObservationDataset,
    build_episode_index,
    split_episode_ids,
    move_batch_to_device,
    default_model_inputs_from_batch,
)
from .model import LogpZOModel


# ----------------------------
# Config loading
# ----------------------------

def infer_train_config_path_from_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent.parent
    train_cfg = run_dir / "faildetect_logpzo.yaml"
    if not train_cfg.exists():
        # fallback: maybe copied original filename as config.yaml
        fallback = run_dir / "config.json"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f"Could not find training config next to checkpoint: {train_cfg}"
        )
    return train_cfg


def deep_update_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update_dict(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


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
    elif "faildetect_logpzo" in raw:
        node = raw["faildetect_logpzo"]
    else:
        node = raw

    if not isinstance(node, dict):
        raise ValueError("FAIL-Detect logpZO config node must be a dict.")
    return dict(node)


def build_config(raw: dict[str, Any]) -> LogpZOConfig:
    return LogpZOConfig(
        data=DataConfig(**raw["data"]),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        optim=OptimConfig(**raw.get("optim", {})),
        model=ModelConfig(**raw.get("model", {})),
        fm_head=FMHeadConfig(**raw.get("fm_head", {})),
        loss=LossConfig(**raw.get("loss", {})),
        experiment_name=raw.get("experiment_name", "faildetect_logpzo"),
        seed=int(raw.get("seed", 0)),
    )


def load_merged_config(
    score_config_path: str | Path,
    checkpoint_path: str | Path,
    score_config_key: str | None = None,
    train_config_key: str | None = None,
) -> tuple[LogpZOConfig, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_config_path = infer_train_config_path_from_checkpoint(checkpoint_path)

    raw_train = _load_text_config(train_config_path)
    node_train = _extract_node(raw_train, config_key=train_config_key)

    raw_score = _load_text_config(score_config_path)
    node_score = _extract_node(raw_score, config_key=score_config_key)

    merged_node = deep_update_dict(node_train, node_score)
    config = build_config(merged_node)

    return config, train_config_path, node_train, node_score, merged_node


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


def _tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()
    if x.is_floating_point():
        x = x.float()
    return x.numpy()


# ----------------------------
# Dataset / model loading
# ----------------------------

def build_inference_loader(
    config: LogpZOConfig,
    split: str,
    batch_size: int | None = None,
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
        chosen = train_episode_ids
    elif split == "val":
        chosen = val_episode_ids
    elif split == "all":
        chosen = episode_ids
    else:
        raise ValueError(f"Unsupported split: {split}")

    frame_ds = SingleStepObservationDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=chosen,
        config=config,
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
        frame_ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=config.runtime.num_workers,
        pin_memory=(config.runtime.pin_memory if config.runtime.pin_memory is not None else True),
        persistent_workers=(config.runtime.persistent_workers if config.runtime.num_workers > 0 else False),
    )
    return loader


def load_model_from_checkpoint(
    config: LogpZOConfig,
    checkpoint_path: str | Path,
    device: torch.device,
) -> LogpZOModel:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model = LogpZOModel(config).to(device)
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


def summarize_scores(results: dict[str, np.ndarray]) -> dict[str, Any]:
    x_obs = results["x_obs"]      # [N,D]
    eps_hat = results["eps_hat"]  # [N,D]
    n, d = x_obs.shape

    summary: dict[str, Any] = {
        "num_frames": int(n),
        "obs_dim": int(d),
    }

    summary.update(_summary_stats("score", results["scores"]))
    summary.update(_summary_stats("log_prob", results["log_prob"]))
    summary.update(_summary_stats("x_obs_norm", np.linalg.norm(x_obs, axis=1)))
    summary.update(_summary_stats("eps_hat_norm", np.linalg.norm(eps_hat, axis=1)))

    return summary


# ----------------------------
# Scoring
# ----------------------------

@torch.no_grad()
def run_scoring(
    model: LogpZOModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    all_episode_index: list[np.ndarray] = []
    all_frame_index: list[np.ndarray] = []

    all_x_obs: list[np.ndarray] = []
    all_eps_hat: list[np.ndarray] = []
    all_log_prob: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    autocast_enabled = (device.type == "cuda" and use_amp)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        meta_episode_index = batch["episode_index"].cpu().numpy()
        meta_frame_index = batch["frame_index"].cpu().numpy()

        batch = move_batch_to_device(batch, device)
        inputs = default_model_inputs_from_batch(batch)

        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model.compute_nonconformity_scores(**inputs)
        else:
            out = model.compute_nonconformity_scores(**inputs)

        all_episode_index.append(meta_episode_index)
        all_frame_index.append(meta_frame_index)

        all_x_obs.append(_tensor_to_numpy(out["x_obs"]))
        all_eps_hat.append(_tensor_to_numpy(out["eps_hat"]))
        all_log_prob.append(_tensor_to_numpy(out["log_prob"]))
        all_scores.append(_tensor_to_numpy(out["scores"]))

    return {
        "episode_index": np.concatenate(all_episode_index, axis=0),
        "frame_index": np.concatenate(all_frame_index, axis=0),
        "x_obs": np.concatenate(all_x_obs, axis=0),
        "eps_hat": np.concatenate(all_eps_hat, axis=0),
        "log_prob": np.concatenate(all_log_prob, axis=0),
        "scores": np.concatenate(all_scores, axis=0),
    }


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
    )

    out_dir = ensure_dir(args.output_dir)
    save_json(
        out_dir / "score_run_config.json",
        {
            "score_config_path": str(args.config),
            "train_config_path": str(train_config_path),
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "batch_size": args.batch_size,
            "device": str(device),
            "use_amp": bool(config.runtime.use_amp and device.type == "cuda"),
            "amp_dtype": config.runtime.amp_dtype,
            "max_batches": args.max_batches,
            "score_kind": getattr(config.fm_head, "score_kind", None),
            "score_transport_mode": getattr(config.fm_head, "score_transport_mode", None),
            "use_fix_camera": bool(config.data.use_fix_camera),
            "use_wrist_camera": bool(config.data.use_wrist_camera),
            "use_state": bool(config.data.use_state),
        },
    )
    save_json(out_dir / "merged_faildetect_logpzo_config.json", merged_node)

    results = run_scoring(
        model=model,
        loader=loader,
        device=device,
        use_amp=config.runtime.use_amp,
        amp_dtype=amp_dtype,
        max_batches=args.max_batches,
    )

    np.savez_compressed(
        out_dir / "frame_scores.npz",
        episode_index=results["episode_index"],
        frame_index=results["frame_index"],
        x_obs=results["x_obs"],
        eps_hat=results["eps_hat"],
        log_prob=results["log_prob"],
        scores=results["scores"],
    )

    summary = summarize_scores(results)
    save_json(out_dir / "summary_scores.json", summary)

    print("Saved:")
    print(f"  {out_dir / 'frame_scores.npz'}")
    print(f"  {out_dir / 'summary_scores.json'}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_key", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())