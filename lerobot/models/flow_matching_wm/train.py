from __future__ import annotations

"""
Train map-level endpoint-conditioned Flow Matching World Model.

Example:
    python -m models.flow_matching_wm.train --config configs/flow_matching_wm.yaml

Resume:
    python -m models.flow_matching_wm.train \
        --config configs/flow_matching_wm.yaml \
        --resume ../fomofd/tasks/needle_pickup/model/fm_wm/needle_pickup_flow_matching_wm_map/checkpoints/last.pt
"""

import argparse
import json
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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
from .model import FlowMatchingWorldModel


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


def autocast_context(config: FlowMatchingWorldModelConfig, device: torch.device):
    if device.type != "cuda" or not config.runtime.use_amp:
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=get_amp_dtype(config.runtime.amp_dtype),
    )


def make_grad_scaler(config: FlowMatchingWorldModelConfig, device: torch.device):
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

def _episode_index_cache_path(run_dir: Path, config: FlowMatchingWorldModelConfig) -> Path:
    repo_tag = config.data.dataset_repo_id.replace("/", "__")
    return run_dir / f"episode_index_cache__{repo_tag}.pt"


def build_datasets_with_cache(
    config: FlowMatchingWorldModelConfig,
    run_dir: Path,
) -> tuple[
    LeRobotDataset,
    EndpointWindowDataset,
    EndpointWindowDataset,
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
    train_episode_ids = select_training_episode_ids(
        train_episode_ids,
        fraction=config.data.train_episode_fraction,
        seed=config.data.train_subset_seed,
    )

    train_ds = EndpointWindowDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=train_episode_ids,
        config=config,
    )
    val_ds = EndpointWindowDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=val_episode_ids,
        config=config,
    )

    return ds, train_ds, val_ds, episode_to_indices, train_episode_ids, val_episode_ids


def make_dataloaders(
    config: FlowMatchingWorldModelConfig,
    train_ds: EndpointWindowDataset,
    val_ds: EndpointWindowDataset,
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

def build_optimizer(config: FlowMatchingWorldModelConfig, model: nn.Module) -> torch.optim.Optimizer:
    return AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )


def build_scheduler(config: FlowMatchingWorldModelConfig, optimizer: torch.optim.Optimizer):
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

@dataclass
class ValidationEarlyStopping:
    """Track the existing strict flow-loss best; patience is in optimizer steps."""

    patience_steps: int | None
    best_step: int | None = None
    start_step: int | None = None  # Conservative origin if a legacy best step is unknown.

    def update(self, step: int, *, is_best: bool) -> bool:
        # A new best always resets patience before considering a stop.
        if is_best:
            self.best_step = step
            self.start_step = None
        reference_step = self.best_step if self.best_step is not None else self.start_step
        return (
            self.patience_steps is not None
            and reference_step is not None
            and step - reference_step >= self.patience_steps
        )


def restore_best_val_step(checkpoint: dict[str, Any], history_path: Path) -> int | None:
    """Recover patience state from a new checkpoint or a legacy run's history."""
    best_step = checkpoint.get("best_val_step")
    resume_step = int(checkpoint.get("step", 0))
    if best_step is not None:
        best_step = int(best_step)
        if not 0 <= best_step <= resume_step:
            raise ValueError("Checkpoint best_val_step must be between zero and its saved step")
        return best_step
    if not history_path.is_file():
        return None
    best_flow = float("inf")
    for line in history_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A previous interrupted write may have a partial row.
        if row.get("kind") != "val" or int(row["step"]) > resume_step:
            continue
        if float(row["flow_loss"]) < best_flow:
            best_flow = float(row["flow_loss"])
            best_step = int(row["step"])
    if best_flow != float(checkpoint.get("best_val_flow", float("inf"))):
        return None
    return best_step


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: FlowMatchingWorldModelConfig,
    step: int,
    epoch: int,
    best_val_flow: float,
    best_val_step: int | None = None,
    early_stop_start_step: int | None = None,
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
        "best_val_step": best_val_step,
        "early_stop_start_step": early_stop_start_step,
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
    model: FlowMatchingWorldModel,
    loader: DataLoader,
    config: FlowMatchingWorldModelConfig,
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
            bs = int(batch["action_chunk"].shape[0])

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
            if "flow_fix_loss" in stats:
                metrics["flow_fix_loss"] = float(stats["flow_fix_loss"].item())
            if "flow_wrist_loss" in stats:
                metrics["flow_wrist_loss"] = float(stats["flow_wrist_loss"].item())
            if "aux_fix_loss" in stats:
                metrics["aux_fix_loss"] = float(stats["aux_fix_loss"].item())
            if "aux_wrist_loss" in stats:
                metrics["aux_wrist_loss"] = float(stats["aux_wrist_loss"].item())
            if losses.aux_one_step is not None:
                metrics["aux_one_step_loss"] = float(losses.aux_one_step.detach().item())

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

    config.validate()
    set_seed(config.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = get_device(config.runtime.device)
    run_dir = ensure_dir(Path(config.data.output_dir) / config.experiment_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    save_json(run_dir / "config.json", asdict(config))
    shutil.copy2(args.config, run_dir / Path(args.config).name)

    ds, train_ds, val_ds, episode_index, train_episode_ids, val_episode_ids = build_datasets_with_cache(config, run_dir)
    unused_train_episode_ids = sorted(set(episode_index) - set(train_episode_ids) - set(val_episode_ids))
    save_json(
        run_dir / "splits.json",
        {
            "train_episode_ids": train_episode_ids,
            "val_episode_ids": val_episode_ids,
            "unused_train_episode_ids": unused_train_episode_ids,
            "train_episode_fraction": config.data.train_episode_fraction,
            "train_subset_seed": config.data.train_subset_seed,
            "num_train_episodes": len(train_episode_ids),
            "num_val_episodes": len(val_episode_ids),
            "num_train_windows": len(train_ds),
            "num_val_windows": len(val_ds),
            "dataset_len": len(ds),
        },
    )

    train_loader, val_loader = make_dataloaders(config, train_ds, val_ds, device)

    model = FlowMatchingWorldModel(config).to(device)
    warmup_cuda_blas(device)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    scaler = make_grad_scaler(config, device)

    total_params, trainable_params = count_parameters(model)
    print(f"Run dir:           {run_dir}")
    print(f"Device:            {device}")
    print(f"Train windows:     {len(train_ds)}")
    print(f"Val windows:       {len(val_ds)}")
    print(f"Total params:      {total_params:,}")
    print(f"Trainable params:  {trainable_params:,}")

    start_step = 0
    start_epoch = 0
    best_val_flow = float("inf")
    history_path = run_dir / "history.jsonl"
    early_stopping = ValidationEarlyStopping(config.runtime.early_stop_patience_steps)

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
        resume_history = history_path
        if not resume_history.is_file():
            resume_history = Path(args.resume).resolve().parent.parent / "history.jsonl"
        early_stopping.best_step = restore_best_val_step(ckpt, resume_history)
        if early_stopping.patience_steps is not None and early_stopping.best_step is None:
            # Missing legacy history cannot justify an immediate early stop.
            saved_origin = ckpt.get("early_stop_start_step")
            early_stopping.start_step = start_step if saved_origin is None else int(saved_origin)
            if not 0 <= early_stopping.start_step <= start_step:
                raise ValueError("Checkpoint early_stop_start_step is outside its saved training steps")
            print(f"[early stop] Original best step is unknown; patience starts at step {early_stopping.start_step}.")
        print(f"Resumed from: {args.resume}")
        print(f"Start step:   {start_step}")
        print(f"Best val flow:{best_val_flow:.6f}")

    if early_stopping.patience_steps is not None:
        print(f"Early stopping: {early_stopping.patience_steps} optimizer steps without a lower val flow loss")

    step = start_step
    epoch = start_epoch
    stopped_early = False
    interval_sums: dict[str, float] = {}
    interval_items = 0
    interval_start_time = time.time()

    while step < config.runtime.max_steps and not stopped_early:
        epoch += 1
        model.train()

        for batch in train_loader:
            if step >= config.runtime.max_steps:
                break

            step += 1

            batch = move_batch_to_device(batch, device)
            inputs = default_model_inputs_from_batch(batch)
            bs = int(batch["action_chunk"].shape[0])

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
            if "flow_fix_loss" in stats:
                train_metrics["flow_fix_loss"] = float(stats["flow_fix_loss"].item())
            if "flow_wrist_loss" in stats:
                train_metrics["flow_wrist_loss"] = float(stats["flow_wrist_loss"].item())
            if "aux_fix_loss" in stats:
                train_metrics["aux_fix_loss"] = float(stats["aux_fix_loss"].item())
            if "aux_wrist_loss" in stats:
                train_metrics["aux_wrist_loss"] = float(stats["aux_wrist_loss"].item())
            if losses.aux_one_step is not None:
                train_metrics["aux_one_step_loss"] = float(losses.aux_one_step.detach().item())

            for k, v in train_metrics.items():
                interval_sums[k] = interval_sums.get(k, 0.0) + v * bs
            interval_items += bs

            should_log = (step == 1) or (step % config.runtime.log_freq == 0) or (step == config.runtime.max_steps)
            should_validate = (step % config.runtime.val_freq == 0) or (step == config.runtime.max_steps)
            should_save = (step % config.runtime.save_freq == 0) or (step == config.runtime.max_steps)

            if should_log:
                elapsed = time.time() - interval_start_time
                averaged = {k: v / max(interval_items, 1) for k, v in interval_sums.items()}
                log_row = {
                    "kind": "train",
                    "step": step,
                    "epoch": epoch,
                    "time_sec": elapsed,
                    **averaged,
                }
                append_jsonl(history_path, log_row)

                msg = (
                    f"[train] step={step:06d} "
                    f"loss={averaged['loss_total']:.6f} "
                    f"flow={averaged['flow_loss']:.6f} "
                )
                if "aux_one_step_loss" in averaged:
                    msg += f"aux={averaged['aux_one_step_loss']:.6f} "
                if "flow_fix_loss" in averaged:
                    msg += f"flow_fix={averaged['flow_fix_loss']:.6f} "
                if "flow_wrist_loss" in averaged:
                    msg += f"flow_wrist={averaged['flow_wrist_loss']:.6f} "
                if "aux_fix_loss" in averaged:
                    msg += f"aux_fix={averaged['aux_fix_loss']:.6f} "
                if "aux_wrist_loss" in averaged:
                    msg += f"aux_wrist={averaged['aux_wrist_loss']:.6f} "
                msg += (
                    f"x1={averaged['x1_norm_mean']:.6f} "
                    f"x0={averaged['x0_norm_mean']:.6f} "
                    f"xt={averaged['xt_norm_mean']:.6f} "
                    f"pred_v={averaged['pred_v_norm_mean']:.6f} "
                    f"target_v={averaged['target_v_norm_mean']:.6f} "
                    f"grad={averaged['grad_norm']:.6f} "
                    f"lr={averaged['lr']:.6e} "
                    f"time={elapsed:.1f}s"
                )
                print(msg)

                interval_sums.clear()
                interval_items = 0
                interval_start_time = time.time()

            if should_validate:
                val_metrics = run_validation(model, val_loader, config, device)
                val_row = {
                    "kind": "val",
                    "step": step,
                    "epoch": epoch,
                    **val_metrics,
                }
                append_jsonl(history_path, val_row)

                msg = (
                    f"[val]   step={step:06d} "
                    f"loss={val_metrics['loss_total']:.6f} "
                    f"flow={val_metrics['flow_loss']:.6f} "
                )
                if "aux_one_step_loss" in val_metrics:
                    msg += f"aux={val_metrics['aux_one_step_loss']:.6f} "
                if "flow_fix_loss" in val_metrics:
                    msg += f"flow_fix={val_metrics['flow_fix_loss']:.6f} "
                if "flow_wrist_loss" in val_metrics:
                    msg += f"flow_wrist={val_metrics['flow_wrist_loss']:.6f} "
                if "aux_fix_loss" in val_metrics:
                    msg += f"aux_fix={val_metrics['aux_fix_loss']:.6f} "
                if "aux_wrist_loss" in val_metrics:
                    msg += f"aux_wrist={val_metrics['aux_wrist_loss']:.6f} "
                msg += (
                    f"x1={val_metrics['x1_norm_mean']:.6f} "
                    f"x0={val_metrics['x0_norm_mean']:.6f} "
                    f"xt={val_metrics['xt_norm_mean']:.6f} "
                    f"pred_v={val_metrics['pred_v_norm_mean']:.6f} "
                    f"target_v={val_metrics['target_v_norm_mean']:.6f}"
                )
                print(msg)

                if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics["flow_loss"])

                is_best = val_metrics["flow_loss"] < best_val_flow
                patience_exhausted = early_stopping.update(step, is_best=is_best)
                stopped_early = patience_exhausted and step < config.runtime.max_steps
                if is_best:
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
                        best_val_step=early_stopping.best_step,
                        early_stop_start_step=early_stopping.start_step,
                    )
                    print(f"Saved new best checkpoint with val_flow={best_val_flow:.6f}")

            if should_save or stopped_early:
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
                    best_val_step=early_stopping.best_step,
                    early_stop_start_step=early_stopping.start_step,
                )

            if stopped_early:
                stop_row = {
                    "kind": "early_stop",
                    "step": step,
                    "epoch": epoch,
                    "metric": "flow_loss",
                    "patience_steps": early_stopping.patience_steps,
                    "best_val_step": early_stopping.best_step,
                    "early_stop_start_step": early_stopping.start_step,
                    "best_val_flow": best_val_flow,
                }
                append_jsonl(history_path, stop_row)
                print(
                    f"[early stop] step={step:06d} best_step={early_stopping.best_step} "
                    f"best_val_flow={best_val_flow:.6f} patience_steps={early_stopping.patience_steps}"
                )
                break

    print("Training finished.")
    print(f"Best val flow: {best_val_flow:.6f}")
    print(f"Last checkpoint: {ckpt_dir / 'last.pt'}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


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
