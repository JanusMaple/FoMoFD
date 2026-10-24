#!/usr/bin/env python3
"""Single-GPU ACT training/smoke runner with bounded best/last retention.

Uses LeRobot's training loop, processors, optimizer and checkpoint format.
No dataset split or architecture changes. See configs/policy_ensemble.md.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+'.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def replace_directory(staged, target):
    """Preserve the previous complete directory until the new one is installed."""
    previous = target.with_name('.'+target.name+'.previous')
    if previous.exists():
        if not target.exists():
            previous.rename(target)
        else:
            shutil.rmtree(previous)
    if target.exists():
        target.rename(previous)
    try:
        staged.rename(target)
    except BaseException:
        if previous.exists() and not target.exists():
            previous.rename(target)
        raise
    if previous.exists():
        shutil.rmtree(previous)


class BestLast:
    """Only rotate this run's best/last directories; never touch other runs."""
    def __init__(self, root):
        self.root = Path(root)
        self.best_loss = math.inf
        self.best_step = None
        self.save_seconds = []

    def save(self, step, loss, writer):
        if not math.isfinite(loss):
            raise ValueError('Refusing a checkpoint with nonfinite training loss')
        self.root.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        better = loss < self.best_loss
        with tempfile.TemporaryDirectory(prefix='.last-staging-', dir=self.root) as temp:
            staged = Path(temp)/'checkpoint'
            writer(staged)
            write_json(staged/'selection.json', {
                'step': step, 'training_loss_mean': loss,
                'criterion': 'mean of recent training losses; not validation loss',
                'best_loss': loss if better else self.best_loss,
                'best_step': step if better else self.best_step})
            replace_directory(staged, self.root/'last')
        if better:
            # Same-filesystem hard links avoid duplicating weights/optimizer
            # while best and last refer to the same training step. Later saves
            # replace complete directories, never modify these files in place.
            with tempfile.TemporaryDirectory(prefix='.best-staging-', dir=self.root) as temp:
                staged = Path(temp)/'checkpoint'
                shutil.copytree(self.root/'last', staged, copy_function=os.link)
                replace_directory(staged, self.root/'best')
            self.best_loss, self.best_step = loss, step
        elapsed = time.perf_counter()-started
        self.save_seconds.append(elapsed)
        print(f'[checkpoint] last={step} best={self.best_step} training_loss_mean={loss:.6f} save_s={elapsed:.2f}', flush=True)


def validate_dataset(root):
    info_path = root/'meta/info.json'
    if not info_path.is_file():
        raise FileNotFoundError(f'Prepare the RGB expert dataset first: {info_path}')
    info = json.loads(info_path.read_text())
    for key in ('observation.images.fix_camera', 'observation.images.wrist_camera'):
        if info['features'].get(key, {}).get('dtype') != 'image':
            raise ValueError(f'Require RGB image feature {key}; cached compressed tokens/VAE latents cannot substitute')
    return info


@contextmanager
def local_dino(torch, repo):
    """Avoid GitHub branch checks when the DINO source is already cached."""
    original = torch.hub.load
    repo = Path(repo) if repo else Path(torch.hub.get_dir())/'facebookresearch_dinov2_main'
    def load(source, model, *args, **kwargs):
        if source == 'facebookresearch/dinov2' and repo.is_dir():
            kwargs['source'] = 'local'
            return original(str(repo), model, *args, **kwargs)
        return original(source, model, *args, **kwargs)
    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original


def run(args):
    # Import this checkout's plugin, not an editable install of a sibling repo.
    sys.path.insert(0, str(ROOT/'lerobot/policies/act_dinov2/src'))
    import torch
    import lerobot_policy_act_dinov2
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train as trainer

    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one GPU per model; submit independent seeds as separate jobs')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training/timing runner')
    config_path = args.config.resolve()
    os.chdir(ROOT/'lerobot')
    cfg = TrainPipelineConfig.from_pretrained(str(config_path), local_files_only=True)
    if cfg.policy.type != 'act_dinov2' or cfg.resume or cfg.policy.pretrained_path:
        raise ValueError('Require a fresh ACT-DINOv2 policy (no resume/pretrained initialization)')
    if cfg.env is not None or cfg.wandb.enable or cfg.policy.push_to_hub:
        raise ValueError('This runner expects offline training, no environment eval or publishing')
    if args.dataset_root:
        cfg.dataset.root = str(args.dataset_root)
    cfg.dataset.root = str(Path(cfg.dataset.root).resolve())
    if args.output:
        cfg.output_dir = args.output
    cfg.output_dir = Path(cfg.output_dir).resolve()
    if cfg.output_dir.exists():
        raise FileExistsError(f'Choose a fresh output directory: {cfg.output_dir}')
    info = validate_dataset(Path(cfg.dataset.root))
    planned_steps, planned_save_freq = cfg.steps, cfg.save_freq
    if not cfg.save_checkpoint or cfg.save_freq <= 0:
        raise ValueError('Enable checkpointing with a positive save_freq')
    if args.mode == 'smoke':
        if not args.output or not 10 <= args.smoke_steps <= 100:
            raise ValueError('Smoke requires a separate --output and 10..100 updates')
        if not 0 <= args.warmup_steps < args.smoke_steps:
            raise ValueError('Warmup must be shorter than smoke test')
        # Two save events exercise rotation using the same full checkpoint writer.
        cfg.steps = args.smoke_steps
        cfg.save_freq = max(1, args.smoke_steps//2)
        cfg.log_freq = 5
        cfg.eval_freq = 0
    cfg.validate()
    loss_window = min(args.loss_window, cfg.save_freq)
    recent = deque(maxlen=loss_window)
    store = BestLast(cfg.output_dir/'checkpoints')
    elapsed_steps = []
    saved_inference = {}
    original_update, original_save = trainer.update_policy, trainer.save_checkpoint
    original_last = trainer.update_last_checkpoint
    step = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def update(*pos, **kw):
        nonlocal step
        start = time.perf_counter()
        metrics, out = original_update(*pos, **kw)
        if args.mode == 'smoke':
            torch.cuda.synchronize()
        duration = time.perf_counter()-start + metrics.dataloading_s.val
        step += 1
        loss = float(metrics.loss.val)
        batch = kw.get('batch', pos[2] if len(pos) > 2 else None)
        actual_batch_size = int(batch['action'].shape[0])
        if not math.isfinite(loss) or not math.isfinite(float(metrics.grad_norm.val)):
            raise FloatingPointError(f'Nonfinite loss/gradient at step {step}')
        recent.append(loss)
        if step > args.warmup_steps and actual_batch_size == cfg.batch_size:
            elapsed_steps.append(duration)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        with (cfg.output_dir/'history.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': step, 'loss': loss, 'grad_norm': float(metrics.grad_norm.val),
                                     'lr': float(metrics.lr.val), 'batch_size': actual_batch_size,
                                     'update_and_data_s': duration})+'\n')
        return metrics, out

    def save(**kw):
        policy = kw['policy']
        if args.mode == 'smoke':
            # Saved checkpoint reload is checked below. Here exercise the
            # deterministic action path on a real normalized expert sample.
            from lerobot.datasets.factory import make_dataset
            dataset = make_dataset(cfg)
            batch = kw['preprocessor'](dataset[0])
            policy.eval()
            with torch.inference_mode():
                action = policy.predict_action_chunk(batch)
            if action.shape != (1, cfg.policy.chunk_size, 10) or not torch.isfinite(action).all():
                raise ValueError(f'Invalid predicted action chunk: {action.shape}')
            saved_inference.update(batch=batch, action=action.detach().cpu(), step=kw['step'])
            policy.train()
        def writer(target):
            original_save(**{**kw, 'checkpoint_dir': target})
        store.save(kw['step'], statistics.mean(recent), writer)

    # Limit changes to retention and timing; preserve the installed training loop.
    trainer.update_policy, trainer.save_checkpoint = update, save
    trainer.update_last_checkpoint = lambda directory: cfg.output_dir/'checkpoints/last'
    try:
        with local_dino(torch, args.dino_repo):
            trainer.train(cfg)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        final = cfg.output_dir/'checkpoints/last/pretrained_model'
        best = cfg.output_dir/'checkpoints/best/pretrained_model'
        for path in (final, best):
            for name in ('model.safetensors', 'train_config.json', 'policy_preprocessor.json', 'policy_postprocessor.json'):
                if not (path/name).is_file():
                    raise FileNotFoundError(path/name)
        if args.mode == 'smoke':
            with local_dino(torch, args.dino_repo):
                restored = lerobot_policy_act_dinov2.ACTDINOv2Policy.from_pretrained(final).to('cuda').eval()
            with torch.inference_mode():
                reloaded_action = restored.predict_action_chunk(saved_inference['batch']).cpu()
            torch.testing.assert_close(reloaded_action, saved_inference['action'], rtol=1e-5, atol=1e-6)
            del restored
        if not elapsed_steps:
            raise ValueError('Need full-size batches after warmup for a timing estimate')
        timings = {'measured_full_batch_updates': len(elapsed_steps), 'warmup_updates': args.warmup_steps,
                   'mean_update_and_data_s': statistics.mean(elapsed_steps),
                   'median_update_and_data_s': statistics.median(elapsed_steps),
                   'checkpoint_save_mean_s': statistics.mean(store.save_seconds)}
        projected = timings['mean_update_and_data_s']*planned_steps
        projected += math.ceil(planned_steps/planned_save_freq)*timings['checkpoint_save_mean_s']
        report = {'mode': args.mode, 'status': 'complete', 'steps': step, 'config': str(config_path),
                  'input_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  'trainer_source_sha256': hashlib.sha256(Path(trainer.__file__).read_bytes()).hexdigest(),
                  'runner_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                  'dataset_episodes': info['total_episodes'], 'dataset_frames': info['total_frames'],
                  'seed': cfg.seed, 'batch_size': cfg.batch_size, 'use_amp': cfg.policy.use_amp,
                  'best_step': store.best_step, 'best_training_loss': store.best_loss,
                  'loss_window_updates': loss_window, 'criterion': 'recent training loss; not validation',
                  'timings': timings, 'planned_training_steps': planned_steps,
                  'estimated_training_hours': projected/3600,
                  'elapsed_s': time.perf_counter()-started,
                  'estimate_limitations': 'Short warm-cache sample; full-dataset I/O, initialization and other GPU hardware may differ.',
                  'smoke_inference_shape': list(saved_inference['action'].shape) if saved_inference else None}
        report['checkpoint_reload_verified'] = args.mode == 'smoke'
        write_json(cfg.output_dir/'run_report.json', report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        write_json(cfg.output_dir/'run_report.json', {'mode': args.mode, 'status': 'failed', 'steps': step,
                                                    'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        trainer.update_policy, trainer.save_checkpoint = original_update, original_save
        trainer.update_last_checkpoint = original_last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'train'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dino-repo', type=Path)
    parser.add_argument('--smoke-steps', type=int, default=30)
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--loss-window', type=int, default=1000)
    args = parser.parse_args()
    if args.loss_window <= 0 or args.warmup_steps < 0:
        parser.error('loss-window must be positive; warmup-steps must be nonnegative')
    for field in ('config', 'dataset_root', 'output', 'dino_repo'):
        if getattr(args, field):
            setattr(args, field, getattr(args, field).expanduser().resolve())
    run(args)


if __name__ == '__main__':
    main()
