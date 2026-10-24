from __future__ import annotations

"""
Train feature-space ConvNeXt β-VAE on precomputed DINO patch features.

Example:
    python -m models.vfmf_vae.train --config configs/vfmf_vae.yaml

Resume:
    python -m models.vfmf_vae.train \
        --config configs/vfmf_vae.yaml \
        --resume ../fomofd/tasks/needle_pickup/model/vfmf_vae/dino_vfmf_vae_fix/checkpoints/last.pt
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
    raise ImportError("PyYAML is required for train.py. Please `pip install pyyaml`.") from exc

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

from .config import (
    DataConfig,
    FeatureConfig,
    FeatureSpaceVAEConfig,
    ConvNeXtIsotropicVAEConfig,
    LossConfig,
    OptimConfig,
    RuntimeConfig,
)
from .dataset import (
    SingleFrameFeatureDataset,
    build_episode_index,
    default_model_inputs_from_batch,
    move_batch_to_device,
    split_episode_ids,
)
from .model import FeatureSpaceVAE


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
    elif "vae" in raw:
        node = raw["vae"]
    else:
        node = raw

    if not isinstance(node, dict):
        raise ValueError("Config node must be a dict.")
    return dict(node)


def build_config(raw: dict[str, Any]) -> FeatureSpaceVAEConfig:
    if "data" not in raw:
        raise ValueError("VAE config must contain a top-level 'data' section.")

    cfg = FeatureSpaceVAEConfig(
        data=DataConfig(**raw["data"]),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        optim=OptimConfig(**raw.get("optim", {})),
        feature=FeatureConfig(**raw.get("feature", {})),
        vae=ConvNeXtIsotropicVAEConfig(**raw.get("vae_model", raw.get("vae", {})) if "vae_model" in raw else raw.get("vae_model", {}) or raw.get("model", {}) or raw.get("inner_vae", {}) or raw.get("convnext_vae", {}) or raw.get("vae_backbone", {}) or raw.get("vae_cfg", {}) or raw.get("feature_vae", {}) or raw.get("feature_space_vae", {}) or raw.get("vae_inner", {}) or raw.get("vae_arch", {}) or raw.get("architecture", {}) or raw.get("convnext_isotropic_vae", {})),
        loss=LossConfig(**raw.get("loss", {})),
        experiment_name=raw.get("experiment_name", "dino_feature_convnext_vae"),
        seed=int(raw.get("seed", 0)),
    )
    # Allow the plain key "vae_model" but also fallback if user nests actual model config under "vae"
    if "vae_model" not in raw and "model" in raw and isinstance(raw["model"], dict):
        cfg.vae = ConvNeXtIsotropicVAEConfig(**raw["model"])
    elif "vae_model" in raw and isinstance(raw["vae_model"], dict):
        cfg.vae = ConvNeXtIsotropicVAEConfig(**raw["vae_model"])
    elif "convnext_vae" in raw and isinstance(raw["convnext_vae"], dict):
        cfg.vae = ConvNeXtIsotropicVAEConfig(**raw["convnext_vae"])

    cfg.validate()
    return cfg


def load_config(config_path: str | Path, config_key: str | None = None) -> FeatureSpaceVAEConfig:
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


def autocast_context(config: FeatureSpaceVAEConfig, device: torch.device):
    if device.type != "cuda" or not config.runtime.use_amp:
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=get_amp_dtype(config.runtime.amp_dtype),
    )


def make_grad_scaler(config: FeatureSpaceVAEConfig, device: torch.device):
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

def _episode_index_cache_path(run_dir: Path, config: FeatureSpaceVAEConfig) -> Path:
    repo_tag = config.data.dataset_repo_id.replace("/", "__")
    return run_dir / f"episode_index_cache__{repo_tag}.pt"


def build_datasets_with_cache(
    config: FeatureSpaceVAEConfig,
    run_dir: Path,
):
    print(
        f"[vfmf_vae] loading LeRobotDataset...\n"
        f"  repo_id={config.data.dataset_repo_id}\n"
        f"  root={config.data.dataset_root}",
        flush=True,
    )

    t0 = time.time()
    ds = LeRobotDataset(
        repo_id=config.data.dataset_repo_id,
        root=config.data.dataset_root,
    )
    print(
        f"[vfmf_vae] LeRobotDataset loaded: len={len(ds)} "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )

    cache_path = _episode_index_cache_path(run_dir, config)
    if cache_path.exists():
        print(f"Loading cached episode index: {cache_path}", flush=True)
        episode_to_indices = torch.load(cache_path, map_location="cpu")
    else:
        print("Building episode index from dataset...", flush=True)
        episode_to_indices = build_episode_index(ds)
        torch.save(episode_to_indices, cache_path)
        print(f"Saved episode index cache to: {cache_path}", flush=True)

    episode_ids = sorted(episode_to_indices.keys())
    train_episode_ids, val_episode_ids = split_episode_ids(
        episode_ids,
        val_ratio=config.data.val_ratio,
        seed=config.data.split_seed,
    )

    train_ds = SingleFrameFeatureDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=train_episode_ids,
        config=config,
    )
    val_ds = SingleFrameFeatureDataset(
        ds=ds,
        episode_to_indices=episode_to_indices,
        episode_ids=val_episode_ids,
        config=config,
    )

    return ds, train_ds, val_ds, episode_to_indices, train_episode_ids, val_episode_ids


def make_dataloaders(
    config: FeatureSpaceVAEConfig,
    train_ds: SingleFrameFeatureDataset,
    val_ds: SingleFrameFeatureDataset,
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

def build_optimizer(config: FeatureSpaceVAEConfig, model: nn.Module) -> torch.optim.Optimizer:
    return AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )


def build_scheduler(config: FeatureSpaceVAEConfig, optimizer: torch.optim.Optimizer):
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
    config: FeatureSpaceVAEConfig,
    step: int,
    epoch: int,
    best_val_l2: float,
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
        "best_val_l2": best_val_l2,
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
# Train / val helpers
# ----------------------------

def _deterministic_flag(config: FeatureSpaceVAEConfig) -> bool:
    return config.loss.objective == "AE"


def _batch_l2_per_sample(ll: Tensor, x: Tensor) -> Tensor:
    # ll is negative SSE summed over [H, W, C], so normalized l2 proxy:
    denom = x.shape[-1] * x.shape[-2] * x.shape[-3]
    return (-ll) / float(denom)


def run_validation(
    model: FeatureSpaceVAE,
    loader: DataLoader,
    config: FeatureSpaceVAEConfig,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    sums: dict[str, float] = {}
    total_items = 0
    total_batches = 0
    max_val_batches = config.runtime.max_val_batches
    deterministic = _deterministic_flag(config)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_val_batches is not None and batch_idx >= max_val_batches:
                break

            batch = move_batch_to_device(batch, device)
            x = default_model_inputs_from_batch(batch, config)
            bs = int(x.shape[0])

            with autocast_context(config, device):
                out = model(x, deterministic=deterministic)

            l2 = _batch_l2_per_sample(out.ll, x).mean()

            metrics = {
                "loss_total": float((-out.elbo.mean()).detach().item()),
                "recon_nll": float((-out.ll.mean()).detach().item()),
                "kl": float(out.kl.mean().detach().item()),
                "l2": float(l2.detach().item()),
                "latent_norm_mean": float(torch.norm(out.latents, dim=-1).mean().detach().item()),
            }

            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v * bs

            total_items += bs
            total_batches += 1

    if total_items == 0:
        raise RuntimeError("Validation loader produced zero items.")

    out_metrics = {k: v / total_items for k, v in sums.items()}
    out_metrics["num_batches"] = float(total_batches)
    out_metrics["num_items"] = float(total_items)

    if was_training:
        model.train()

    return out_metrics


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

    ds, train_ds, val_ds, _, train_episode_ids, val_episode_ids = build_datasets_with_cache(config, run_dir)
    save_json(
        run_dir / "splits.json",
        {
            "train_episode_ids": train_episode_ids,
            "val_episode_ids": val_episode_ids,
            "num_train_episodes": len(train_episode_ids),
            "num_val_episodes": len(val_episode_ids),
            "num_train_frames": len(train_ds),
            "num_val_frames": len(val_ds),
            "dataset_len": len(ds),
        },
    )

    train_loader, val_loader = make_dataloaders(config, train_ds, val_ds, device)

    model = FeatureSpaceVAE(config).to(device)
    warmup_cuda_blas(device)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    scaler = make_grad_scaler(config, device)
    deterministic = _deterministic_flag(config)

    total_params, trainable_params = count_parameters(model)
    print(f"Run dir:           {run_dir}")
    print(f"Device:            {device}")
    print(f"Objective:         {config.loss.objective}")
    print(f"Train frames:      {len(train_ds)}")
    print(f"Val frames:        {len(val_ds)}")
    print(f"Total params:      {total_params:,}")
    print(f"Trainable params:  {trainable_params:,}")

    start_step = 0
    start_epoch = 0
    best_val_l2 = float("inf")

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
        best_val_l2 = float(ckpt.get("best_val_l2", float("inf")))
        print(f"Resumed from: {args.resume}")
        print(f"Start step:   {start_step}")
        print(f"Best val l2:  {best_val_l2:.6f}")

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
            x = default_model_inputs_from_batch(batch, config)
            bs = int(x.shape[0])

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(config, device):
                out = model(x, deterministic=deterministic)
                loss = -out.elbo.mean()

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

            l2 = _batch_l2_per_sample(out.ll.detach(), x).mean()

            train_metrics = {
                "loss_total": float(loss.detach().item()),
                "recon_nll": float((-out.ll.mean()).detach().item()),
                "kl": float(out.kl.mean().detach().item()),
                "l2": float(l2.detach().item()),
                "latent_norm_mean": float(torch.norm(out.latents, dim=-1).mean().detach().item()),
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
                averaged = {k: v / max(interval_items, 1) for k, v in interval_sums.items()}
                log_row = {
                    "kind": "train",
                    "step": step,
                    "epoch": epoch,
                    "time_sec": elapsed,
                    **averaged,
                }
                append_jsonl(history_path, log_row)

                print(
                    f"[train] step={step:06d} "
                    f"loss={averaged['loss_total']:.6f} "
                    f"l2={averaged['l2']:.6f} "
                    f"nll={averaged['recon_nll']:.6f} "
                    f"kl={averaged['kl']:.6f} "
                    f"latent_norm={averaged['latent_norm_mean']:.6f} "
                    f"grad={averaged['grad_norm']:.6f} "
                    f"lr={averaged['lr']:.6e} "
                    f"time={elapsed:.1f}s"
                )

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

                print(
                    f"[val]   step={step:06d} "
                    f"loss={val_metrics['loss_total']:.6f} "
                    f"l2={val_metrics['l2']:.6f} "
                    f"nll={val_metrics['recon_nll']:.6f} "
                    f"kl={val_metrics['kl']:.6f} "
                    f"latent_norm={val_metrics['latent_norm_mean']:.6f}"
                )

                if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics["l2"])

                is_best = val_metrics["l2"] < best_val_l2
                if is_best:
                    best_val_l2 = val_metrics["l2"]
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        step=step,
                        epoch=epoch,
                        best_val_l2=best_val_l2,
                    )
                    print(f"Saved new best checkpoint with val_l2={best_val_l2:.6f}")

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
                    best_val_l2=best_val_l2,
                )

    print("Training finished.")
    print(f"Best val l2: {best_val_l2:.6f}")
    print(f"Last checkpoint: {ckpt_dir / 'last.pt'}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML/JSON config path.")
    parser.add_argument("--config_key", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())