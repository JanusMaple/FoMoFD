from __future__ import annotations

"""
Train FAIL-Detect-style logpZO baseline on current observation O_t only.

Example:
    python -m models.faildetect_logpzo.train --config configs/faildetect_logpzo.yaml

Resume:
    python -m models.faildetect_logpzo.train \
        --config configs/faildetect_logpzo.yaml \
        --resume ../fomofd/tasks/needle_pickup/model/faildetect_logpzo/faildetect_logpzo/checkpoints/last.pt
"""

import argparse
import json
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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
# Config IO
# ----------------------------

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
    cfg = LogpZOConfig(
        data=DataConfig(**raw["data"]),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        optim=OptimConfig(**raw.get("optim", {})),
        model=ModelConfig(**raw.get("model", {})),
        fm_head=FMHeadConfig(**raw.get("fm_head", {})),
        loss=LossConfig(**raw.get("loss", {})),
        experiment_name=raw.get("experiment_name", "faildetect_logpzo"),
        seed=int(raw.get("seed", 0)),
    )
    return cfg


def load_config(config_path: str | Path, config_key: str | None = None) -> LogpZOConfig:
    raw = _load_text_config(config_path)
    node = _extract_node(raw, config_key=config_key)
    return build_config(node)


# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


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


def autocast_context(config: LogpZOConfig, device: torch.device):
    if device.type != "cuda" or not config.runtime.use_amp:
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=get_amp_dtype(config.runtime.amp_dtype),
    )


def make_grad_scaler(config: LogpZOConfig, device: torch.device):
    use_scaler = (
        device.type == "cuda"
        and config.runtime.use_amp
        and config.runtime.amp_dtype.lower() == "float16"
    )
    return torch.amp.GradScaler("cuda", enabled=use_scaler)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def warmup_cuda_blas(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.init()
    x = torch.randn(8, 8, device=device)
    _ = x @ x
    torch.cuda.synchronize()


# ----------------------------
# Dataset construction
# ----------------------------

def _episode_index_cache_path(run_dir: Path, config: LogpZOConfig) -> Path:
    repo_tag = config.data.dataset_repo_id.replace("/", "__")
    return run_dir / f"episode_index_cache__{repo_tag}.pt"


def build_datasets_with_cache(
    config: LogpZOConfig,
    run_dir: Path,
) -> tuple[
    LeRobotDataset,
    SingleStepObservationDataset,
    SingleStepObservationDataset,
    dict[int, list[int]],
    list[int],
    list[int],
]:
    ds = LeRobotDataset(
        repo_id=config.data.dataset_repo_id,
        root=config.data.dataset_root,
    )

    cache_path = _episode_index_cache_path(run_dir, config)
    if cache_path.exists():
        print(f"Loading cached episode index: {cache_path}")
        episode_to_indices = torch.load(cache_path, map_location="cpu")
    else:
        print("Building episode index from dataset...")
        episode_to_indices = build_episode_index(ds)
        torch.save(episode_to_indices, cache_path)
        print(f"Saved episode index cache to: {cache_path}")

    episode_ids = sorted(episode_to_indices.keys())
    train_episode_ids, val_episode_ids = split_episode_ids(
        episode_ids,
        val_ratio=config.data.val_ratio,
        seed=config.data.split_seed,
    )

    train_ds = SingleStepObservationDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=train_episode_ids,
        config=config,
    )
    val_ds = SingleStepObservationDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=val_episode_ids,
        config=config,
    )

    return ds, train_ds, val_ds, episode_to_indices, train_episode_ids, val_episode_ids


def make_dataloaders(
    config: LogpZOConfig,
    train_ds: SingleStepObservationDataset,
    val_ds: SingleStepObservationDataset,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    pin_memory = config.runtime.pin_memory
    if pin_memory is None:
        pin_memory = (device.type == "cuda")

    num_workers = int(config.runtime.num_workers)
    persistent_workers = bool(config.runtime.persistent_workers) if num_workers > 0 else False

    train_batch_size = int(config.runtime.batch_size)
    val_batch_size = int(
        config.runtime.val_batch_size
        if config.runtime.val_batch_size is not None
        else config.runtime.batch_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


# ----------------------------
# Optim / sched
# ----------------------------

def build_optimizer(config: LogpZOConfig, model: nn.Module) -> torch.optim.Optimizer:
    return AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )


def build_scheduler(config: LogpZOConfig, optimizer: torch.optim.Optimizer):
    name = config.optim.scheduler.lower()
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.runtime.max_steps,
            eta_min=config.optim.min_lr,
        )
    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.optim.plateau_factor,
            patience=config.optim.plateau_patience,
        )
    raise ValueError(f"Unknown scheduler: {config.optim.scheduler}")


# ----------------------------
# Checkpointing
# ----------------------------

def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: LogpZOConfig,
    step: int,
    epoch: int,
    best_val_flow: float,
) -> None:
    path = Path(path)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "config": asdict(config),
        "step": step,
        "epoch": epoch,
        "best_val_flow": best_val_flow,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt


# ----------------------------
# Validation
# ----------------------------

def run_validation(
    model: LogpZOModel,
    loader: DataLoader,
    config: LogpZOConfig,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    sums: dict[str, float] = {}
    total_items = 0
    total_batches = 0
    max_val_batches = config.runtime.max_val_batches

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_val_batches is not None and batch_idx >= max_val_batches:
                break

            batch = move_batch_to_device(batch, device)
            inputs = default_model_inputs_from_batch(batch)
            bs = int(batch["episode_index"].shape[0])

            with autocast_context(config, device):
                losses, stats = model.compute_loss(**inputs)

            metrics = {
                "loss_total": float(losses.total.detach().item()),
                "flow_loss": float(losses.flow.detach().item()),
                "x1_norm_mean": float(stats["x1_norm_mean"].item()),
                "x0_norm_mean": float(stats["x0_norm_mean"].item()),
                "xt_norm_mean": float(stats["xt_norm_mean"].item()),
                "pred_v_norm_mean": float(stats["pred_v_norm_mean"].item()),
                "target_v_norm_mean": float(stats["target_v_norm_mean"].item()),
            }

            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v * bs

            total_items += bs
            total_batches += 1

    if total_items == 0:
        raise RuntimeError("Validation loader produced zero items.")

    out = {k: v / total_items for k, v in sums.items()}
    out["num_batches"] = float(total_batches)
    out["num_items"] = float(total_items)

    if was_training:
        model.train()

    return out


# ----------------------------
# Main train
# ----------------------------

def train(args: argparse.Namespace) -> None:
    config = load_config(args.config, config_key=args.config_key)

    if args.output_dir is not None:
        config.data.output_dir = args.output_dir
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.seed is not None:
        config.seed = int(args.seed)

    set_seed(config.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = get_device(config.runtime.device)
    run_dir = ensure_dir(Path(config.data.output_dir) / config.experiment_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    save_json(run_dir / "config.json", asdict(config))
    shutil.copy2(args.config, run_dir / Path(args.config).name)

    ds, train_ds, val_ds, _, train_episode_ids, val_episode_ids = build_datasets_with_cache(config, run_dir)
    save_json(
        run_dir / "splits.json",
        {
            "train_episode_ids": train_episode_ids,
            "val_episode_ids": val_episode_ids,
            "num_train_episodes": len(train_episode_ids),
            "num_val_episodes": len(val_episode_ids),
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
            "dataset_len": len(ds),
        },
    )

    train_loader, val_loader = make_dataloaders(config, train_ds, val_ds, device)

    model = LogpZOModel(config).to(device)
    warmup_cuda_blas(device)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    scaler = make_grad_scaler(config, device)

    total_params, trainable_params = count_parameters(model)
    print(f"Run dir:           {run_dir}")
    print(f"Device:            {device}")
    print(f"Train samples:     {len(train_ds)}")
    print(f"Val samples:       {len(val_ds)}")
    print(f"Total params:      {total_params:,}")
    print(f"Trainable params:  {trainable_params:,}")

    start_step = 0
    start_epoch = 0
    best_val_flow = float("inf")

    if args.resume is not None:
        ckpt = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
        )
        start_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_flow = float(ckpt.get("best_val_flow", float("inf")))
        print(f"Resumed from:      {args.resume}")
        print(f"Start step:        {start_step}")
        print(f"Best val flow:     {best_val_flow:.6f}")

    history_path = run_dir / "history.jsonl"

    step = start_step
    epoch = start_epoch
    interval_sums: dict[str, float] = {}
    interval_items = 0
    interval_start_time = time.time()

    while step < config.runtime.max_steps:
        epoch += 1
        model.train()

        for batch in train_loader:
            if step >= config.runtime.max_steps:
                break

            step += 1

            batch = move_batch_to_device(batch, device)
            inputs = default_model_inputs_from_batch(batch)
            bs = int(batch["episode_index"].shape[0])

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(config, device):
                losses, stats = model.compute_loss(**inputs)
                loss = losses.total

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.optim.grad_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.optim.grad_clip_norm,
                )
                optimizer.step()

            if scheduler is not None and isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()

            train_metrics = {
                "loss_total": float(losses.total.detach().item()),
                "flow_loss": float(losses.flow.detach().item()),
                "x1_norm_mean": float(stats["x1_norm_mean"].item()),
                "x0_norm_mean": float(stats["x0_norm_mean"].item()),
                "xt_norm_mean": float(stats["xt_norm_mean"].item()),
                "pred_v_norm_mean": float(stats["pred_v_norm_mean"].item()),
                "target_v_norm_mean": float(stats["target_v_norm_mean"].item()),
                "grad_norm": float(grad_norm.item()) if isinstance(grad_norm, Tensor) else float(grad_norm),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }

            for k, v in train_metrics.items():
                interval_sums[k] = interval_sums.get(k, 0.0) + v * bs
            interval_items += bs

            should_log = (step == 1) or (step % config.runtime.log_freq == 0) or (step == config.runtime.max_steps)
            should_validate = (step % config.runtime.val_freq == 0) or (step == config.runtime.max_steps)
            should_save = (step % config.runtime.save_freq == 0) or (step == config.runtime.max_steps)

            if should_log:
                elapsed = time.time() - interval_start_time
                avg = {k: (v / max(interval_items, 1)) for k, v in interval_sums.items()}
                log_row = {
                    "step": step,
                    "epoch": epoch,
                    "split": "train",
                    "samples": interval_items,
                    "elapsed_sec": elapsed,
                    **avg,
                }
                append_jsonl(history_path, log_row)

                print(
                    f"[train] step={step:7d} "
                    f"loss={avg['loss_total']:.6f} "
                    f"flow={avg['flow_loss']:.6f} "
                    f"grad={avg['grad_norm']:.4f} "
                    f"lr={avg['lr']:.3e} "
                    f"samples={interval_items}"
                )

                interval_sums = {}
                interval_items = 0
                interval_start_time = time.time()

            if should_validate:
                val_metrics = run_validation(
                    model=model,
                    loader=val_loader,
                    config=config,
                    device=device,
                )

                if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics["flow_loss"])

                val_row = {
                    "step": step,
                    "epoch": epoch,
                    "split": "val",
                    **val_metrics,
                }
                append_jsonl(history_path, val_row)

                print(
                    f"[val]   step={step:7d} "
                    f"loss={val_metrics['loss_total']:.6f} "
                    f"flow={val_metrics['flow_loss']:.6f} "
                    f"items={int(val_metrics['num_items'])}"
                )

                if val_metrics["flow_loss"] < best_val_flow:
                    prev_best = best_val_flow
                    best_val_flow = val_metrics["flow_loss"]

                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        step=step,
                        epoch=epoch,
                        best_val_flow=best_val_flow,
                    )

                    banner = "-" * 78
                    print()
                    print(banner)
                    print(
                        f"----- NEW BEST VAL FLOW -----  "
                        f"{prev_best:.6f}  ->  {best_val_flow:.6f}   @ step {step}"
                    )
                    print(f"----- Saved best checkpoint to: {ckpt_dir / 'best.pt'}")
                    print(banner)
                    print()

            if should_save:
                save_checkpoint(
                    ckpt_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    step=step,
                    epoch=epoch,
                    best_val_flow=best_val_flow,
                )

    print("Training complete.")
    print(f"Best val flow: {best_val_flow:.6f}")
    print(f"Run dir: {run_dir}")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_key", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())