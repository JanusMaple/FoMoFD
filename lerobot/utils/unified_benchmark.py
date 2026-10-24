"""Unified failure benchmark: cached scores, calibration comparisons, and reports.

Dry-run/cache-only never import Torch. Inference is explicitly opt-in, runs
one model at a time with configurable OFFLINE batches, and never edits labels.
score_batch_sizes selects a batch per method; task score_batch_sizes can override
it, and --batch-size overrides both. These are throughput settings, not an
online multi-window protocol. Sampled methods key random draws by task, episode,
and frame, so changing batch boundaries preserves each frame's random inputs.

    python lerobot/utils/unified_benchmark.py --dry-run
    python lerobot/utils/unified_benchmark.py --cache-only
    python lerobot/utils/unified_benchmark.py --allow-inference --tasks needle_pickup
    python lerobot/utils/unified_benchmark.py --allow-inference --mode override
    python lerobot/utils/unified_benchmark.py --methods wm_nc logpzo rnd wm_pe --dry-run

Scorers: WM-NC, WM-PE, WM-PU (predictive uncertainty), logpZO/RND on VAE features
and ACT-native tokens, and ACE from ACT prior action samples. WM-PU estimates
mean latent coordinate SD using streaming endpoint samples. It adapts Ward's
uncertainty idea, not their architecture. Executed future actions make its safe
availability time anchor+K, as for our other WM scores; no early timestamp claim.
Validated revisions use frozen expert-normal bin widths and explicit WM-PU RK4
steps. Historical settings remain the default when no sampling manifest is set.
ACE sums joint XYZ histogram entropies across a sampled action chunk, adapting
FIPER to ACT. Sample the CVAE PRIOR z~N(0,I), never a posterior conditioned on
recorded future actions. Ordinary ACT deployment uses z=0. Cached camera tokens
are checked against the exact ACT weights; saved state/action normalization is
restored. Accumulate fixed-frame delta XYZ actions into relative positions.
Fixed widths come from expert workspace ranges, not test data. This differs from
FIPER's calibration-derived ranges and is not a reproduction of full FIPER.
References: https://arxiv.org/html/2510.09459v1 (ACE Appendix A.1);
https://arxiv.org/html/2603.06987v1 (WM uncertainty Section IV-B).

Score caches: PRL/score_result/unified_benchmark/METHOD/RUN/<fingerprint>.npz.
normal reuses exact matching caches; override recomputes. Atomic replacement
preserves previous scores on errors. Input content hashes identify scores.
Annotations/calibration never invalidate raw scores. Native baselines must reach
their configured training cap before scoring; --dry-run lists pending models.

Reports: paper_table[ _time_varying].csv/.md, summary_metrics.csv (FAR/FDR and
EPISODE AUROC), episode_results.csv, score_distributions.csv, calibration.csv,
latency_episodes.csv, latency_summary.csv, and report.json. Latency uses the
matching camera onset; combined detectors report both onset views separately.
Missing camera labels remain unannotated. Labels are reread
once scoring ends; per-annotator onset coverage stays explicit. Successful
rollouts have no onset labels. Latency uses recorded wall time, including sim;
pre-onset alarms and misses are separate from nonnegative detected delays.

Calibration comparison:
* episode: success-episode maxima, rank ceil((n+1)*(1-alpha)), strict > threshold.
* wall_time_bin (legacy alias time_varying): maxima within elapsed-wall-time bins, one calibration
  value per available success episode/bin. This is binwise calibration, NOT
  simultaneous episode-FAR control. Late/unsupported bins yield +inf; we never
  pad completed rollouts or align using the eventual episode duration. Coverage
  assumes exchangeability of supported episodes; censoring can break that.
* functional_band: a frozen expert-validation mean and modulation, followed by
  one maximum signed residual per original policy calibration trajectory.
  Use functional_benchmark.py for validated expert scoring and protocol-separated
  reports, or supply --functional-templates to this script. Functional ROC uses
  maximum template residuals, sweeping h. See functional_conformal.py for the
  documented adaptations and reference implementation quantile difference.

AND/OR: min/max of signed margins (score-threshold)/calibration_scale at matched
RAW frame indices, with matching wall times. AND requires simultaneous alarms,
not alarms at unrelated moments. Includes fixed/wrist WM-NC and same-view
WM-NC + logpZO/RND (both representations). OR retains component alpha and must
not be claimed to have alpha-level familywise coverage. No independence assumed.

AUROC: one maximum signed margin per TEST episode; positives are all failures,
negatives success-test only. Tie-aware; no onsets used. For scalar thresholds
this equals ranking by raw episode maxima. For time-varying thresholds/fusion
it ranks the declared standardized exceedance margin. Unsupported episodes have
-inf evidence and tie accordingly; coverage counts accompany AUROC. Overall
AUROC is the equal-task macro mean, avoiding pooled incompatible raw scores.
FAR/FDR overall are pooled episode counts. Threshold/scale fitting never sees
success-test or failure scores. No frame-level AUROC or label-based point adjustment.

"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import csv
import zipfile

import numpy as np
import yaml
import sampled_score_protocol as sampled_protocol
from model_paths import resolve_model_path, relocate_identity

REPO = Path(__file__).resolve().parents[2]
CACHE_VERSION = 1
METHODS = {
    "wm_nc": ("WM-NC", "flow_matching_wm", "fm_wm", "reactive"),
    "logpzo": ("logpZO", "faildetect_logpzo", "faildetect_logpzo", None),
    "rnd": ("RND", "fiper_rnd", "fiper_rnd", None),
    "wm_pe": ("WM-PE", "flow_matching_wm", "fm_wm", "predictive"),
    "wm_pu": ("WM-PU", "flow_matching_wm", "fm_wm", "sampled_std"),
    "logpzo_native": ("logpZO Native", "faildetect_logpzo", "faildetect_logpzo", None),
    "rnd_native": ("RND Native", "fiper_rnd", "fiper_rnd", None),
    "ace": ("ACE (ACT prior)", "act_dinov2", "act", None),
    "act_ensemble_pos": ("ACT ensemble · position slope", "act_dinov2", "act", None),
    "act_ensemble_rot": ("ACT ensemble · rotation slope", "act_dinov2", "act", None),
}
CAMERAS = ("fix", "wrist")
VIEWS = {"fix": "Fixed", "wrist": "Wrist", "both": "Fixed+Wrist"}
HASHES: dict[tuple, str] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    """Stream once per stable file; no checkpoint deserialization."""
    path = resolve_model_path(path)
    before = path.stat()
    key = (str(path), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if key not in HASHES:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != key[1:]:
            raise ValueError(f"Input changed while reading: {path}")
        HASHES[key] = h.hexdigest()
    return HASHES[key]


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rooted(value: str, base: Path) -> Path:
    return resolve_model_path(value, base)


def load_node(path: Path, key: str) -> dict:
    # PyYAML interprets JSON exponents such as 1e-06 as strings. Match the
    # scorers' JSON parser exactly, or sampled workers/merged-config checks fail.
    raw = json.loads(path.read_text()) if path.suffix.lower() == '.json' else yaml.safe_load(path.read_text())
    node = raw.get(key, raw) if isinstance(raw, dict) else None
    if not isinstance(node, dict):
        raise ValueError(f"Expected configuration mapping: {path}")
    return node


def merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


@dataclass
class Job:
    task: str
    method: str
    views: tuple[str, ...]
    checkpoint: Path
    train_config: Path
    score_config: Path
    override: dict
    merged: dict
    dataset: Path
    family: str
    config_key: str
    mode: str | None
    batch_size: int
    seed: int
    device: str
    # Optional explicit expert-validation selection; never changes PRL defaults.
    split: str = 'all'
    episode_ids: tuple[int, ...] | None = None

    @property
    def cache_dir(self) -> Path:
        return self.dataset.parent / "score_result/unified_benchmark" / self.method / self.checkpoint.parent.parent.name


def score_batch_size(config: dict, task: str, method: str, override=None) -> int:
    value = config.get('score_batch_size', 1)
    for node in (config, config['tasks'][task]):
        mapping = node.get('score_batch_sizes', {})
        if not isinstance(mapping, dict) or set(mapping) - set(METHODS):
            raise ValueError('score_batch_sizes must map known methods to positive integers')
        if any(type(n) is not int or n < 1 for n in mapping.values()):
            raise ValueError('score_batch_sizes must contain positive integers')
        value = mapping.get(method, value)
    if override is not None:
        value = override
    if type(value) is not int or value < 1:
        raise ValueError('Score batch size must be a positive integer')
    return value


def make_jobs(config: dict, repo: Path, tasks: list[str], methods: list[str], args) -> list[Job]:
    jobs = []
    for task in tasks:
        node = config["tasks"][task]
        for method in methods:
            if method.startswith('act_ensemble_'):
                jobs.append(make_ensemble_job(config, repo, task, method, args))
                continue
            if method == "ace":
                jobs.append(make_ace_job(config, repo, task, args))
                continue
            _, family, config_key, mode = METHODS[method]
            entry = node["wm" if method.startswith("wm_") else method]
            score_path = rooted(entry["score_config"], repo)
            checkpoints = {"both": entry["checkpoint"]} if mode else entry["checkpoints"]
            if not mode and set(checkpoints) != set(CAMERAS):
                raise ValueError(f"Both fixed and wrist checkpoints are required: {task}/{method}")
            for view, checkpoint in checkpoints.items():
                checkpoint = rooted(checkpoint, repo)
                if not checkpoint.is_file() and not args.dry_run:
                    raise FileNotFoundError(checkpoint)
                run = checkpoint.parent.parent
                # Match the existing scorers' train-config resolution exactly.
                names = ["config.local.json", "config.json", "flow_matching_wm.yaml"] if mode else ["config.local.json", f"{family}.yaml", "config.json"]
                train_path = next((run / name for name in names if (run / name).is_file()), None)
                if train_path is None and args.dry_run and method.endswith("_native"):
                    train_path = rooted(entry["train_configs"][view], repo)
                if train_path is None:
                    raise FileNotFoundError(f"Training config missing beside {checkpoint}")
                trained = load_node(train_path, config_key)
                expert = rooted(trained["data"]["dataset_root"], repo / "lerobot")
                expected_expert = repo / "fomofd/tasks" / task / "data"
                if not expert.is_relative_to(expected_expert):
                    raise ValueError(f"Checkpoint must be trained on this task's expert data: {checkpoint}")
                override = load_node(score_path, config_key)
                if method == "wm_pu":
                    # A new score implementation, not the historical proactive
                    # variance trace or policy-action entropy mode.
                    override = merge(override, {"fm_head": {"per_chunk_wm_samples": config["wm_pu"]["num_samples"]}})
                override = merge(override, {"runtime": {"num_workers": 0, "persistent_workers": False}})
                merged = merge(trained, override)
                if method == 'wm_pu':
                    validate_sampling(config['wm_pu'])
                    merged['sampling'] = merge({'sample_batch_size': 1, 'precision': 'float32',
                                                'matmul_precision': 'highest'}, config['wm_pu'])
                    merged['sampling']['seed_scheme'] = 'episode_frame_v1'
                    sampling = merged['sampling']
                    if sampling.get('sampling_manifest'):
                        frozen = sampled_protocol.load_frozen_sampling(sampling['sampling_manifest'], repo,
                            task, method, digest(checkpoint))
                        if sampling['num_samples'] != frozen['num_samples']:
                            raise ValueError('WM sample count differs from normal-data validation')
                        sampling['ode_steps'] = frozen['ode_steps']
                        sampling['matmul_precision'] = frozen['matmul_precision']
                data = merged["data"]
                dataset = rooted(data["dataset_root"], repo / "lerobot")
                if not dataset.is_relative_to(repo / "fomofd/tasks" / task / "prl_data"):
                    raise ValueError(f"Scoring config must point to this task's PRL data: {score_path}")
                if not (dataset / "meta/preparation.json").is_file():
                    raise FileNotFoundError(f"Prepared dataset provenance missing: {dataset}")
                views = tuple(v for v, key in [("fix", "use_fix_camera"), ("wrist", "use_wrist_camera")] if data.get(key, False))
                if views != (("fix", "wrist") if mode else (view,)):
                    raise ValueError(f"Checkpoint camera selection differs from benchmark entry: {checkpoint}")
                for key in ["vfmf_vae_latent_stats_path", "action_stats_path", "state_stats_path"]:
                    if data.get(key):
                        stats = rooted(data[key], repo / "lerobot")
                        if not stats.is_file() or not stats.is_relative_to(expected_expert):
                            raise ValueError(f"Normalization must use saved expert statistics: {stats}")
                jobs.append(Job(task, method, views, checkpoint, train_path, score_path,
                                override, merged, dataset, family, config_key, mode,
                                score_batch_size(config, task, method, args.batch_size),
                                config["score_seed"], args.device))
    return jobs


def fingerprint(job: Job, repo: Path) -> dict:
    """Metrics/annotations intentionally excluded: reuse scores when labels change."""
    files = {job.checkpoint, job.train_config}
    if job.method != "ace":
        files.add(job.score_config)
    if job.method.startswith('act_ensemble_'):
        files.update(Path(p) for p in job.merged['ensemble']['checkpoints'])
        for checkpoint in job.merged['ensemble']['checkpoints']:
            files.update(Path(checkpoint).parent.glob('*.json'))
            files.update(Path(checkpoint).parent.glob('*processor.safetensors'))
        files.update((repo / 'lerobot/policies/act_dinov2/src/lerobot_policy_act_dinov2').glob('*.py'))
        files.update(repo / 'lerobot/utils' / name for name in
                     ['act_ensemble_score.py', 'npz_to_le_v21.py', 'online_benchmark.py', 'unified_benchmark.py'])
        preparation = json.loads((job.dataset/'meta/preparation.json').read_text())
        files.update(job.dataset.parent/'raw'/r['raw_file'] for r in preparation['episode_mapping'])
    elif job.method == "ace":
        files.update(job.checkpoint.parent.glob("*.json"))
        files.update(job.checkpoint.parent.glob("*processor.safetensors"))
        files.update((repo / "lerobot/policies/act_dinov2/src/lerobot_policy_act_dinov2").glob("*.py"))
        files.add(rooted(job.merged["sampling"]["expert_stats"], repo))
    else:
        files.update(repo / "lerobot/models" / job.family / name for name in
                     ["__init__.py", "score.py", "config.py", "dataset.py", "model.py"])
    if job.method in {"wm_pu", "ace"}:
        files.add(Path(__file__).resolve())
        files.add(Path(sampled_protocol.__file__).resolve())
        if job.merged.get('sampling', {}).get('sampling_manifest'):
            files.add(rooted(job.merged['sampling']['sampling_manifest'], repo))
    files.update(p for p in (job.dataset / "meta").rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".npz", ".parquet"})
    files.update((job.dataset / "data").rglob("*.parquet"))
    for key in ["vfmf_vae_latent_stats_path", "action_stats_path", "state_stats_path"]:
        if job.merged["data"].get(key):
            files.add(rooted(job.merged["data"][key], repo / "lerobot"))
    if job.episode_ids is not None:
        if job.method == 'ace' or job.method.startswith('act_ensemble_'):
            files.add(job.dataset / 'meta/preparation_splits.json')
        else:
            files.add(job.checkpoint.parent.parent / 'splits.json')
    return {"schema_version": CACHE_VERSION, "task": job.task, "method": job.method,
            "views": job.views, "dataset": str(job.dataset), "checkpoint": str(job.checkpoint),
            "merged_config": job.merged, "split": job.split, "seed": job.seed,
            "batch_size": job.batch_size, "device": job.device,
            **({'episode_ids': list(job.episode_ids)} if job.episode_ids is not None else {}),
            "sha256": {str(path): digest(path) for path in sorted(files)}}


def load_timeline(dataset: Path) -> dict[int, dict[str, np.ndarray]]:
    # Read only these small columns; never import Torch/LeRobot or read features.
    import pyarrow.parquet as pq
    columns = ["episode_index", "frame_index", "observation.raw_frame_index", "observation.wall_timestamp"]
    parts = {key: [] for key in columns}
    paths = sorted((dataset / "data").rglob("*.parquet"))
    if not paths:
        raise ValueError(f"Dataset has no parquet files: {dataset}")
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, use_threads=False):
            for key in columns:
                index = batch.schema.get_field_index(key)
                if index < 0:
                    raise ValueError(f"Required prepared observation column is missing: {path}/{key}")
                parts[key].append(np.asarray(batch.column(index).to_pylist()).reshape(-1))
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    result = {}
    for episode in np.unique(arrays["episode_index"]):
        keep = arrays["episode_index"] == episode
        order = np.argsort(arrays["frame_index"][keep])
        rows = {key: value[keep][order] for key, value in arrays.items()}
        frames, raw, wall = (rows[key] for key in columns[1:])
        if (not np.issubdtype(frames.dtype, np.integer) or not np.array_equal(frames, np.arange(len(frames)))
                or not np.issubdtype(raw.dtype, np.integer) or raw[0] < 0
                or (np.diff(raw) <= 0).any() or not np.isfinite(wall).all() or (np.diff(wall) < 0).any()):
            raise ValueError(f"Invalid frame IDs or recorded wall timestamps: {dataset}, episode {episode}")
        result[int(episode)] = {"frame": frames.astype(np.int64), "raw": raw.astype(np.int64), "wall": wall.astype(np.float64)}
    return result


def validate_scores(arrays: dict, job: Job, timeline: dict) -> None:
    episodes, frames, scores = arrays["episode_index"], arrays["frame_index"], arrays["scores"]
    if (episodes.ndim != 1 or frames.shape != episodes.shape or scores.shape != (len(episodes), len(job.views))
            or not np.issubdtype(episodes.dtype, np.integer) or not np.issubdtype(frames.dtype, np.integer)
            or not np.isfinite(scores).all()):
        raise ValueError(f"Malformed/nonfinite score cache: {job.task}/{job.method}")
    if set(episodes.tolist()) != set(timeline):
        raise ValueError(f"Scores do not cover every PRL episode: {job.task}/{job.method}")
    start = (job.merged["data"]["history_len"] - 1 + job.merged["data"]["pred_horizon"] if job.mode
             else 1 if job.method.startswith('act_ensemble_') else 0)
    for episode, clock in timeline.items():
        keep = episodes == episode
        expected = np.arange(start, len(clock["frame"]))
        if not len(expected) or not np.array_equal(frames[keep], expected):
            raise ValueError(f"Missing/duplicate/unordered scored frames: {job.task}/{job.method}/{episode}")
        if "raw_frame_index" in arrays:
            if (not np.array_equal(arrays["raw_frame_index"][keep], clock["raw"][expected])
                    or not np.array_equal(arrays["wall_timestamp"][keep], clock["wall"][expected])):
                raise ValueError("Cached frame/timestamp mapping differs from prepared observations")


# Only this subprocess imports Torch and loads a model. Normal/override invokes
# it on demand; --cache-only and --dry-run cannot reach it.
SCORER_BOOTSTRAP = r'''
import json, random, runpy, sys, importlib.util
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
source, module, seed, device, metadata = sys.argv[1:6]
sys.path.insert(0, source)
sys.path.insert(0, str(Path(source) / 'utils'))
seed = int(seed)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if device.startswith('cuda'):
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; refusing automatic CPU fallback')
    torch.cuda.manual_seed_all(seed)
with open(metadata, 'w') as stream:
    json.dump({'python':sys.version, 'torch':torch.__version__, 'numpy':np.__version__,
               'packages':{name:version(name) for name in ['lerobot','datasets','pyarrow','torchdiffeq']},
               'cuda_runtime':torch.version.cuda,
               'device':device, 'gpu':torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else None},stream)
spec = importlib.util.spec_from_file_location('benchmark_ode_meter', Path(source) / 'utils/unified_benchmark.py')
benchmark = importlib.util.module_from_spec(spec); sys.modules[spec.name] = benchmark; spec.loader.exec_module(benchmark)
benchmark.install_ode_meter(Path(metadata))
sys.argv = [module] + sys.argv[6:]
runpy.run_module(module, run_name='__main__')
'''


def score_job(job: Job, repo: Path, timeline: dict, identity: dict, target: Path) -> dict:
    log_path = target.with_suffix(f".{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log")
    with tempfile.TemporaryDirectory(prefix=".scoring-", dir=target.parent) as temporary:
        work = Path(temporary)
        override = work / "score_config.yaml"
        # Imported runs preserve their original remote config. Pass the complete
        # local merge so the legacy scorer's config.json fallback cannot restore
        # obsolete machine paths.
        override.write_text(yaml.safe_dump({job.config_key: job.merged if job.train_config.name == 'config.local.json' else job.override}, sort_keys=False))
        command = [sys.executable, "-u", "-c", SCORER_BOOTSTRAP, str(repo / "lerobot"),
                   f"models.{job.family}.score", str(job.seed), job.device, str(work / "environment.json"),
                   "--config", str(override), "--checkpoint", str(job.checkpoint),
                   "--output_dir", str(work), "--split", job.split, "--device", job.device,
                   "--batch_size", str(job.batch_size)]
        if job.mode:
            command += ["--mode", job.mode]
        sampled = job.method in {"wm_pu", "ace"} or job.method.startswith('act_ensemble_')
        if sampled:
            spec_path = work / "worker_spec.json"
            write_json(spec_path, {"repo": str(repo), "method": job.method,
                "task": job.task,
                "checkpoint": str(job.checkpoint), "dataset": str(job.dataset),
                "merged": job.merged, "seed": job.seed, "device": job.device,
                "batch_size": job.batch_size, "output": str(work), "split": job.split,
                "episode_ids": list(job.episode_ids) if job.episode_ids is not None else None,
                "ensemble_cache_key": hashlib.sha256(canonical({k: v for k, v in identity.items()
                     if k != 'method'}).encode()).hexdigest() if job.method.startswith('act_ensemble_') else None})
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-spec", str(spec_path)]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        if job.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        print(f"Scoring {job.task}/{job.method}/{','.join(job.views)}; log: {log_path}", flush=True)
        with log_path.open("w") as stream:
            completed = subprocess.run(command, cwd=repo / "lerobot", env=env, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Scoring failed ({completed.returncode}); previous cache retained. Read {log_path}")
        native = work / f"score_val_{job.mode}" / f"window_scores_{job.mode}.npz" if job.mode else work / "frame_scores.npz"
        if sampled:
            native = work / "sampled_scores.npz"
        else:
            effective = json.loads((native.parent / f"merged_{job.family}_config.json").read_text())
            if canonical(effective) != canonical(job.merged):
                raise ValueError("Scorer used a different merged configuration; score cache was not replaced")
        with np.load(native, allow_pickle=False) as data:
            episodes = np.asarray(data["episode_index"])
            frames = np.asarray(data["frame_index"] if sampled else
                (data["anchor_frame_index"] + job.merged["data"]["pred_horizon"] if job.mode else data["frame_index"]))
            scores = np.asarray(data["scores"] if sampled else
                (data["scores_view"] if job.mode else data["scores"][:, None]), dtype=np.float64)
        order = np.lexsort((frames, episodes))
        arrays = {"episode_index": episodes[order], "frame_index": frames[order], "scores": scores[order]}
        validate_scores(arrays, job, timeline)
        arrays["raw_frame_index"] = np.empty(len(episodes), dtype=np.int64)
        arrays["wall_timestamp"] = np.empty(len(episodes), dtype=np.float64)
        for episode, clock in timeline.items():
            keep = arrays["episode_index"] == episode
            positions = arrays["frame_index"][keep]
            arrays["raw_frame_index"][keep] = clock["raw"][positions]
            arrays["wall_timestamp"][keep] = clock["wall"][positions]
        # Refuse to attach provenance to scores if any source changed mid-run.
        if canonical(fingerprint(job, repo)) != canonical(identity):
            raise ValueError("Scoring input changed during inference; score cache was not replaced")
        provenance = {"identity": identity, "created_utc": utc_now(), "environment": json.loads((work / "environment.json").read_text()), "log": str(log_path)}
        with (work / "cache.npz").open("wb") as stream:
            np.savez_compressed(stream, **arrays, views=np.asarray(job.views), provenance_json=np.asarray(canonical(provenance)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(work / "cache.npz", target)
    return arrays


def acquire_scores(job: Job, repo: Path, timeline: dict, identity: dict, target: Path, args) -> dict:
    def read_cache():
        with np.load(target, allow_pickle=False) as data:
            provenance = json.loads(str(data["provenance_json"].item()))
            if canonical(relocate_identity(provenance["identity"])) != canonical(relocate_identity(identity)) or tuple(data["views"].tolist()) != job.views:
                raise ValueError(f"Cache provenance mismatch: {target}")
            arrays = {key: data[key] for key in ["episode_index", "frame_index", "raw_frame_index", "wall_timestamp", "scores"]}
        validate_scores(arrays, job, timeline)
        return arrays
    if args.cache_only:
        return read_cache()  # Missing files fail; this branch never creates dirs or launches a scorer.
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.mode == "normal" and target.exists():
            print(f"Reusing {target}", flush=True)
            return read_cache()
        if not args.allow_inference:
            raise ValueError("Scoring requires --allow-inference. Training may still be active; use --dry-run or --cache-only first.")
        return score_job(job, repo, timeline, identity, target)


def validate_protocol(protocol: dict, *, causal: bool = False) -> None:
    if protocol["smoothing"] not in {"centered", "trailing"}:
        raise ValueError("Smoothing must be centered or trailing; window=1 means no smoothing")
    window = protocol["window"]
    if type(window) is not int or window < 1 or window % 2 != 1:
        raise ValueError("Smoothing window must be a positive odd integer")
    if causal and protocol["smoothing"] == "centered" and window != 1:
        raise ValueError("Latency requires causal smoothing")
    if protocol["wm_frame_policy"] not in {"replay_legacy", "all_endpoints"}:
        raise ValueError("Unknown WM frame policy")
    seen = set()
    for limit in protocol.get('frame_limits', []):
        key = (limit.get('task'), limit.get('episode_index'))
        if (not isinstance(key[0], str) or not key[0] or type(key[1]) is not int or key[1] < 0
                or type(limit.get('raw_frame_stop')) is not int or limit['raw_frame_stop'] <= 0
                or key in seen):
            raise ValueError('Frame limits require unique task/episode IDs and a positive exclusive raw-frame stop')
        seen.add(key)


def smooth_scores(values: np.ndarray, protocol: dict) -> np.ndarray:
    window = protocol["window"]
    if window == 1:
        return values.copy()
    if protocol["smoothing"] == "centered":
        if len(values) < window:
            raise ValueError("Episode has fewer scores than the centered smoothing window")
        weights = np.ones(window)
        return np.convolve(values, weights, "same") / np.convolve(np.ones(len(values)), weights, "same")
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (prefix[ends] - prefix[starts]) / (ends - starts)


def curves(arrays: dict, job: Job, view_index: int, protocol: dict) -> dict[int, dict]:
    result = {}
    for episode in np.unique(arrays["episode_index"]):
        keep = arrays["episode_index"] == episode
        for limit in protocol.get('frame_limits', []):
            if limit['task'] == job.task and limit['episode_index'] == episode:
                keep &= arrays['raw_frame_index'] < limit['raw_frame_stop']
        if not keep.any():
            raise ValueError(f'Frame limit removes every score: {job.task}/{episode}')
        frames = arrays["frame_index"][keep]
        if job.mode and protocol["wm_frame_policy"] == "replay_legacy":
            # Exactly the additional valid-target restriction in the replay scripts.
            valid_hi = frames[-1] - (job.merged["data"]["pred_horizon"] - 1)
            keep &= (arrays["frame_index"] >= job.merged["data"]["history_len"]) & (arrays["frame_index"] <= valid_hi)
        if not keep.any():
            raise ValueError(f"No evaluable scores: {job.task}/{episode}")
        result[int(episode)] = {"frame": arrays["frame_index"][keep], "raw": arrays["raw_frame_index"][keep],
                                "wall": arrays["wall_timestamp"][keep],
                                "score": smooth_scores(arrays["scores"][keep, view_index], protocol)}
    return result


def threshold_for(episode_curves: dict, calibration: list[int], alpha: float) -> float:
    maxima = sorted(float(episode_curves[episode]["score"].max()) for episode in calibration)
    rank = math.ceil((len(maxima) + 1) * (1 - alpha))
    return maxima[rank - 1] if rank <= len(maxima) else math.inf


def load_labels(path: Path, tasks: list[str]) -> tuple[dict, str]:
    text = path.read_text()  # Inspector publishes changes with atomic replacement.
    labels = yaml.safe_load(text)
    if (labels.get("episode_id_space") != "prepared_prl_episode_index"
            or labels.get("onset_frame_id_space") != "raw_recording_frame_index"
            or labels.get("onset_frame_index_base") != 0):
        raise ValueError("Unexpected episode/onset index convention in labels")
    for task in tasks:
        node = labels["tasks"][task]
        groups, ids = node["groups"], []
        for group in groups:
            lo, hi = group["episode_range"]
            if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi:
                raise ValueError(f"Invalid episode range: {task}")
            if (group["split"] not in {"calibration", "test"} or group["outcome"] not in {"success", "failure"}
                    or (group["split"] == "calibration" and group["outcome"] != "success")):
                raise ValueError(f"Invalid calibration/test label group: {task}")
            ids.extend(range(lo, hi + 1))
        if len(ids) != len(set(ids)) or set(ids) != set(range(node["expected_episode_count"])):
            raise ValueError(f"Labels overlap or do not cover every expected episode: {task}")
        if not any(group["split"] == "calibration" for group in groups):
            raise ValueError(f"No success calibration episodes: {task}")
        lookup = episode_labels(node)
        for episode, annotations in node.get("failure_onsets", {}).items():
            if type(episode) is not int or episode not in lookup or lookup[episode]["outcome"] != "failure":
                raise ValueError(f"Onsets may only label failure episodes: {task}/{episode}")
            for annotator, annotation in annotations.items():
                if annotator not in labels.get("annotators", {}) or not isinstance(annotation, dict):
                    raise ValueError(f"Invalid onset annotator: {task}/{episode}/{annotator}")
                views = annotation_views(annotation)
                if not views or not set(views) <= set(CAMERAS):
                    raise ValueError("Onset views must contain fix and/or wrist")
                for value in views.values():
                    index = value.get("frame_index") if isinstance(value, dict) else None
                    if type(index) is not int or index < 0:
                        raise ValueError(f"Invalid onset annotation: {task}/{episode}/{annotator}")
                if "views" in annotation:
                    if annotation.get("mode") not in {"shared", "separate"}:
                        raise ValueError("Invalid camera onset mode")
                    if annotation["mode"] == "shared" and (set(views) != set(CAMERAS)
                            or views['fix']['frame_index'] != views['wrist']['frame_index']):
                        raise ValueError("Shared onsets must agree across views")
    return labels, text


def annotation_views(entry):
    """Legacy single onsets apply to both cameras; absent new views stay missing."""
    if not entry:
        return {}
    views = entry.get('views', {view: entry for view in CAMERAS})
    if not isinstance(views, dict):
        raise ValueError('Onset views must be a mapping')
    return views


def episode_labels(task_labels: dict) -> dict[int, dict]:
    return {episode: group for group in task_labels["groups"] for episode in range(group["episode_range"][0], group["episode_range"][1] + 1)}


def first_alarm(curve: dict, threshold: float) -> dict | None:
    positions = np.flatnonzero(curve["score"] > threshold)
    if not len(positions):
        return None
    index = positions[0]
    return {"frame": int(curve["frame"][index]), "raw": int(curve["raw"][index]), "wall": float(curve["wall"][index])}


def onset_time(dataset: Path, episode: int, raw_index: int, timeline: dict) -> float | None:
    clock = timeline[episode]
    positions = np.flatnonzero(clock["raw"] == raw_index)
    if len(positions):
        return float(clock["wall"][positions[0]])
    # Simulation preparation can omit the terminal raw observation. Load only
    # its timestamps member if an onset lands there; never load images/pickle.
    preparation = json.loads((dataset / "meta/preparation.json").read_text())
    entry = next(item for item in preparation["episode_mapping"] if item["dataset_episode_index"] == episode)
    if raw_index >= entry["raw_frames"]:
        raise ValueError(f"Onset exceeds raw episode: {episode}/{raw_index}")
    filename = entry["raw_file"]
    if Path(filename).name != filename:
        raise ValueError("Invalid raw filename in preparation metadata")
    raw = dataset.parent / "raw" / filename
    if not raw.is_file():
        return None
    with np.load(raw, allow_pickle=False) as data:
        times = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
    if len(times) != entry["raw_frames"] or not np.isfinite(times).all() or (np.diff(times) < 0).any():
        return None
    return float(times[raw_index])


def statistics(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: None for key in ["mean", "std", "min", "p05", "median", "p95", "max"]}
    return {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()),
            "p05": float(np.quantile(values, .05)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def make_ensemble_job(config, repo, task, method, args):
    entry = config['tasks'][task]['act_ensemble']
    checkpoints = [rooted(p, repo) for p in entry['checkpoints']]
    if len(checkpoints) != 3 or len(set(checkpoints)) != 3 or any(not p.is_file() for p in checkpoints):
        raise ValueError('ACT ensemble requires three distinct installed checkpoints')
    configs = [json.loads((p.parent/'config.json').read_text()) for p in checkpoints]
    keys = ['type', 'chunk_size', 'dim_model', 'latent_dim', 'per_camera_num_queries',
            'use_vae', 'use_cls_token', 'input_features', 'output_features']
    if any(any(c[k] != configs[0][k] for k in keys) for c in configs):
        raise ValueError('ACT ensemble architectures/input representations differ')
    dataset = rooted(entry['dataset'], repo)
    if not dataset.is_relative_to(repo/'fomofd/tasks'/task/'prl_data'):
        raise ValueError('ACT ensemble scoring requires this task PRL data')
    merged = {'data': {'dataset_root': str(dataset)},
              'ensemble': {'checkpoints': [str(p) for p in checkpoints], 'latent': 'zero',
                           'chunk_action_index': 0, 'difference': 'adjacent_prepared_observations',
                           'protocol': 'thompson_act_adaptation_v1', 'member_seeds': [1000, 1001, 1002]}}
    return Job(task, method, ('both',), checkpoints[0], checkpoints[0].parent/'config.json',
               args.config.resolve(), {}, merged, dataset, 'act_dinov2', 'act', None,
               score_batch_size(config, task, method, args.batch_size), config['score_seed'], args.device)


def make_ace_job(config, repo, task, args):
    entry = config['tasks'][task]['ace']
    checkpoint = rooted(entry['checkpoint'], repo)
    train_path = checkpoint.parent / 'config.json'
    trained = json.loads(train_path.read_text())
    if not trained.get('use_vae') or trained.get('type') != 'act_dinov2':
        raise ValueError('ACE requires an ACT-DINOv2 policy with its VAE prior enabled')
    image_keys = [f'observation.images.{v}_camera' for v in CAMERAS]
    if (trained.get('per_camera_num_queries') != 16 or trained.get('dim_model') != 768
            or trained.get('use_cls_token')
            or not set(image_keys) <= set(trained.get('input_features', {}))):
        raise ValueError('ACE adapter requires the current two-camera ACT 16x768 token representation')
    dataset = rooted(entry['dataset'], repo)
    if not dataset.is_relative_to(repo / 'fomofd/tasks' / task / 'prl_data'):
        raise ValueError('ACE must use the task PRL token dataset')
    sampling = merge(config['ace'], {'expert_stats': entry['expert_stats']})
    sampling['seed_scheme'] = 'episode_frame_v1'
    if sampling.get('sampling_manifest'):
        frozen = sampled_protocol.load_frozen_sampling(sampling['sampling_manifest'], repo,
            task, 'ace', digest(checkpoint))
        if sampling['num_samples'] != frozen['num_samples'] or trained['chunk_size'] != frozen['horizon']:
            raise ValueError('ACT sample count/horizon differs from normal-data validation')
        sampling['cell_widths_m'] = frozen['widths_m']
        sampling['resolution_method'] = 'expert_normal_conditional_iqr_fixed_widths'
    stats = rooted(sampling['expert_stats'], repo)
    if not stats.is_relative_to(repo / 'fomofd/tasks' / task / 'data') or not stats.is_file():
        raise ValueError('ACE cell widths require saved expert statistics')
    validate_sampling(sampling, ace=True)
    merged = {'data': {'dataset_root': str(dataset)}, 'sampling': sampling}
    return Job(task, 'ace', ('both',), checkpoint, train_path, args.config.resolve(), {},
               merged, dataset, 'act_dinov2', 'act', None,
               score_batch_size(config, task, 'ace', args.batch_size), config['score_seed'], args.device)


def validate_sampling(settings, *, ace=False):
    if type(settings['num_samples']) is not int or settings['num_samples'] < 2:
        raise ValueError('Uncertainty scoring needs at least two independent samples')
    if type(settings.get('sample_batch_size', 1)) is not int or settings.get('sample_batch_size', 1) < 1:
        raise ValueError('sample_batch_size must be positive')
    if not ace and settings.get('precision', 'float32') not in {'float32', 'bfloat16'}:
        raise ValueError('WM-PU precision must be float32 or bfloat16')
    if not ace and settings.get('matmul_precision', 'highest') not in {'highest', 'high'}:
        raise ValueError('WM-PU matmul_precision must be highest or high')
    if not ace:
        sampled_protocol.validate_ode_steps(settings.get('ode_steps', 1))
    if ace:
        if 'cell_widths_m' in settings:
            sampled_protocol.ace_cell_widths(settings, REPO)
        if not 0 < settings['cell_size_factor'] < 1 or settings['min_cell_width'] <= 0:
            raise ValueError('Invalid ACE cell width settings')
        if settings['action_coordinates'] != 'cumulative_delta_xyz':
            raise ValueError('Current ACT policies output fixed-frame delta XYZ; require cumulative_delta_xyz')


def job_readiness(job):
    if not job.checkpoint.is_file():
        return 'checkpoint missing'
    if job.method.endswith('_native'):
        # The current baseline trainers run to the configured cap. Do not score
        # best.pt while it is still being replaced by the active training queue.
        history = job.checkpoint.parent.parent / 'history.jsonl'
        if not history.is_file():
            return 'training completion not verified (history missing)'
        with history.open('rb') as stream:
            stream.seek(max(0, history.stat().st_size - 65536))
            lines = stream.read().decode().splitlines()
        steps = []
        for line in lines:
            try:
                row = json.loads(line)
                if row.get('split') == 'val':
                    steps.append(int(row.get('step', 0)))
            except (ValueError, TypeError):
                continue
        if max(steps, default=0) < job.merged['runtime']['max_steps']:
            return 'training has not reached its configured cap'
        last = job.checkpoint.parent / 'last.pt'
        # History is appended before the final checkpoint write. Check the
        # closing ZIP directory and ordering too, without deserializing tensors.
        if not last.is_file() or last.stat().st_mtime_ns < history.stat().st_mtime_ns:
            return 'final checkpoint has not finished after final validation'
        try:
            with zipfile.ZipFile(last) as archive:
                if not any(name.endswith('/data.pkl') for name in archive.namelist()):
                    return 'final checkpoint archive is incomplete'
        except (OSError, zipfile.BadZipFile):
            return 'final checkpoint archive is incomplete'
    return None


def ace_histogram_entropy(positions, cell_width):
    """Joint XYZ histogram per chunk step, summed Shannon entropy in bits.

    Positions: [samples, horizon, 3]. Each step's grid starts at its sample
    minimum; fixed widths come only from expert workspace ranges. Sparse bins
    avoid allocating a potentially enormous dense Cartesian histogram.
    """
    positions = np.asarray(positions, dtype=np.float64)
    width = np.asarray(cell_width, dtype=np.float64)
    if (positions.ndim != 3 or positions.shape[0] < 2 or positions.shape[2] != 3
            or width.shape != (3,) or not np.isfinite(positions).all()
            or not np.isfinite(width).all() or (width <= 0).any()):
        raise ValueError('Invalid ACE sample positions/cell widths')
    total = 0.0
    for step in positions.transpose(1, 0, 2):
        span = (step.max(0) - step.min(0)) / width
        if not np.isfinite(span).all() or (span > 1e15).any():
            raise ValueError('ACE grid overflow; check coordinate units/cell widths')
        n_bins = np.maximum(1, np.ceil(span)).astype(np.int64)
        cells = np.floor((step - step.min(0)) / width).astype(np.int64)
        cells = np.minimum(cells, n_bins - 1)  # rightmost edge belongs to final bin
        counts = np.unique(cells, axis=0, return_counts=True)[1]
        probabilities = counts / positions.shape[0]
        total -= float(np.sum(probabilities * np.log2(probabilities)))
    return total


def run_sampled_worker(spec_path):
    """Explicit subprocess only. Never entered by dry-run/cache-only/reporting."""
    import random
    import torch
    spec = json.loads(Path(spec_path).read_text())
    repo = Path(spec['repo'])
    sys.path.insert(0, str(repo / 'lerobot'))
    sys.path.insert(0, str(repo / 'lerobot/policies/act_dinov2/src'))
    random.seed(spec['seed'])
    np.random.seed(spec['seed'])
    torch.manual_seed(spec['seed'])
    if spec['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    torch.set_num_threads(2)
    device = torch.device(spec['device'])
    output = Path(spec['output'])
    install_ode_meter(output / 'environment.json')
    if spec['method'].startswith('act_ensemble_'):
        from act_ensemble_score import run_worker
        episodes, frames, scores = run_worker(spec, device)
    elif spec['method'] == 'wm_pu':
        torch.set_float32_matmul_precision(spec['merged']['sampling']['matmul_precision'])
        from models.flow_matching_wm.score import (build_config, build_inference_loader,
            load_model_from_checkpoint, move_batch_to_device, default_model_inputs_from_batch)
        cfg = build_config(spec['merged'])
        model = load_model_from_checkpoint(cfg, spec['checkpoint'], device)
        loader = build_inference_loader(cfg, spec.get('split', 'all'), spec['batch_size'], include_proactive=False)
        episodes, frames, scores = [], [], []
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if spec.get('max_batches') is not None and index >= spec['max_batches']:
                    break  # Internal adapter check only; normal jobs never set this.
                episodes.extend(batch['episode_index'].tolist())
                frames.extend((batch['anchor_frame_index'] + cfg.data.pred_horizon).tolist())
                inputs = default_model_inputs_from_batch(move_batch_to_device(batch, device))
                # Executed actions are available at the endpoint, never earlier.
                sampling = spec['merged']['sampling']
                noise = frame_noise(spec, batch['episode_index'].tolist(),
                    (batch['anchor_frame_index'] + cfg.data.pred_horizon).tolist(),
                    sampling['num_samples'], (model._num_views(), cfg.data.latent_grid_height,
                                              cfg.data.latent_grid_width, cfg.data.latent_channels))
                uncertainty = score_wm_pu_batch(model, inputs, noise,
                    sampling['sample_batch_size'], sampling['precision'], sampling.get('ode_steps', 1))
                scores.extend(uncertainty.cpu().numpy())
                if index % 100 == 0:
                    print(f'WM-PU batch {index}/{len(loader)}', flush=True)
    else:
        episodes, frames, scores = run_ace_worker(spec, device)
    np.savez_compressed(output / 'sampled_scores.npz', episode_index=np.asarray(episodes, dtype=np.int64),
                        frame_index=np.asarray(frames, dtype=np.int64), scores=np.asarray(scores, dtype=np.float64))
    write_json(output / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
        'numpy': np.__version__, 'device': str(device), 'cuda_runtime': torch.version.cuda,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None})


def load_act_decoder(checkpoint, device):
    """Strictly restore ACT's decoder, using verified cached visual tokens.

    A local constructor patch avoids constructing/loading DINOv2 at all. Every
    non-visual tensor is still checked with strict=True; no learned projection
    is added between the saved tokens and the ACT encoder.
    """
    import torch
    from safetensors import safe_open
    from unittest.mock import patch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot_policy_act_dinov2 import modeling_act_dinov2 as act
    from lerobot_policy_act_dinov2.dino_visual_tokenizer import DINOVisualTokenizerOutput
    config = PreTrainedConfig.from_pretrained(checkpoint.parent)
    config.device = str(device)

    class CachedTokens(torch.nn.Module):
        def forward(self, batch):
            return DINOVisualTokenizerOutput(compressed_tokens=batch['compressed_visual_tokens'])

    with patch.object(act, 'DINOVisualTokenizer', lambda cfg: CachedTokens()):
        policy = act.ACTDINOv2Policy(config)
    with safe_open(checkpoint, framework='pt', device='cpu') as weights:
        state = {key: weights.get_tensor(key) for key in weights.keys()
                 if not key.startswith('model.visual_tokenizer.')}
    policy.load_state_dict(state, strict=True)
    return policy.to(device).eval()


def processor_stats(checkpoint, filename, registry):
    from safetensors import safe_open
    recipe = json.loads((checkpoint.parent / filename).read_text())
    entries = [entry for entry in recipe['steps'] if entry['registry_name'] == registry]
    if len(entries) != 1:
        raise ValueError(f'Expected one {registry} in {filename}')
    entry = entries[0]
    norm = entry['config']['norm_map']
    if norm.get('STATE', 'MEAN_STD') != 'MEAN_STD' or norm.get('ACTION') != 'MEAN_STD':
        raise ValueError('ACE exporter currently supports saved MEAN_STD policy normalization')
    with safe_open(checkpoint.parent / entry['state_file'], framework='np') as weights:
        stats = {key: weights.get_tensor(key) for key in weights.keys()}
    return stats, entry['config'].get('eps', 1e-8)


def validate_native_provenance(job):
    if job.method != 'ace':
        return
    preparation = json.loads((job.dataset / 'meta/preparation.json').read_text())
    hashes = preparation.get('checkpoint_sha256', {})
    for path in [job.checkpoint, job.train_config]:
        # Transfers may change the root path, but never the bytes or basename.
        matches = [value for name, value in hashes.items() if Path(name).name == path.name]
        if matches != [digest(path)]:
            raise ValueError(f'ACE tokens were not exported from this ACT checkpoint: {path}')


def frame_noise(spec, episodes, frames, samples, shape):
    """CPU standard-normal draws keyed by recording identity, independent of batching.

    A new explicit RNG scheme replaces the old iteration-order GPU RNG. It is
    fingerprinted in sampling settings; old stochastic caches remain distinct.
    """
    if len(episodes) != len(frames) or not episodes:
        raise ValueError('Need matching nonempty episode/frame IDs')
    rows = []
    for episode, frame in zip(episodes, frames):
        key = canonical(['episode_frame_v1', spec['seed'], spec['task'], spec['method'],
                         int(episode), int(frame)])
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'little')
        rng = np.random.Generator(np.random.PCG64(seed))
        rows.append(rng.standard_normal((samples, *shape), dtype=np.float32))
    return np.stack(rows)


def repeat_batch(value, repeats):
    """Repeat each available input consecutively for its Monte Carlo samples."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, dict):
        return {k: repeat_batch(v, repeats) for k, v in value.items()}
    if isinstance(value, list):
        return [repeat_batch(v, repeats) for v in value]
    if value is None:
        return None
    raise TypeError(f'Unsupported batch input: {type(value)}')


def score_wm_pu_batch(model, inputs, noise, sample_batch_size, precision, ode_steps=1):
    """Ward-inspired SD on FM-WM; explicit validated RK4 count, legacy default 1."""
    return sampled_protocol.score_wm_pu_batch(model, inputs, noise,
        sample_batch_size, precision, ode_steps)


def ace_inputs(rows, prep, eps, device):
    import torch
    state = np.asarray([row['observation.state'] for row in rows], dtype=np.float32).reshape(len(rows), -1)
    normalized = (state - prep['observation.state.mean']) / (prep['observation.state.std'] + eps)
    tokens = {f'observation.images.{view}_camera': torch.as_tensor(
        np.asarray([row[f'observation.tokens.{view}_camera'] for row in rows],
                   dtype=np.float32).reshape(len(rows), 16, 768), device=device) for view in CAMERAS}
    return {'observation.state': torch.as_tensor(normalized, device=device),
            'compressed_visual_tokens': tokens, 'observation.images': list(tokens.values())}


def sample_ace_batch(policy, batch, noise, sample_batch_size, horizon):
    """Use ACT's existing explicit-prior decoder for multiple independent frames."""
    import torch
    b, samples, latent_dim = noise.shape
    device = batch['observation.state'].device
    parts = []
    with torch.inference_mode():
        for start in range(0, samples, sample_batch_size):
            n = min(sample_batch_size, samples-start)
            latent = torch.as_tensor(noise[:, start:start+n].copy(), device=device).reshape(b*n, latent_dim)
            out = policy.model.decode_with_given_latent(repeat_batch(batch, n), latent)
            parts.append(out[:, :horizon].reshape(b, n, horizon, -1).cpu().numpy())
    return np.concatenate(parts, axis=1)


def run_ace_worker(spec, device):
    import pyarrow.parquet as pq
    checkpoint = Path(spec['checkpoint'])
    policy = load_act_decoder(checkpoint, device)
    prep, eps = processor_stats(checkpoint, 'policy_preprocessor.json', 'normalizer_processor')
    post, _ = processor_stats(checkpoint, 'policy_postprocessor.json', 'unnormalizer_processor')
    sampling = spec['merged']['sampling']
    width = sampled_protocol.ace_cell_widths(sampling, Path(spec['repo']))
    cols = ['episode_index', 'frame_index', 'observation.state',
            'observation.tokens.fix_camera', 'observation.tokens.wrist_camera']
    horizon = sampling.get('horizon') or policy.config.chunk_size
    if type(horizon) is not int or not 1 <= horizon <= policy.config.chunk_size:
        raise ValueError('ACE horizon exceeds the ACT chunk size')
    episodes, frames, scores = [], [], []
    for path in sorted((Path(spec['dataset']) / 'data').rglob('*.parquet')):
        for chunk in pq.ParquetFile(path).iter_batches(columns=cols, batch_size=spec['batch_size'], use_threads=False):
            rows = chunk.to_pylist()
            if spec.get('episode_ids') is not None:
                rows = [r for r in rows if int(r['episode_index']) in spec['episode_ids']]
                if not rows:
                    continue
            if spec.get('max_frames') is not None:
                rows = rows[:spec['max_frames']-len(frames)]
            ep = [int(row['episode_index']) for row in rows]
            fr = [int(row['frame_index']) for row in rows]
            batch = ace_inputs(rows, prep, eps, device)
            noise = frame_noise(spec, ep, fr, sampling['num_samples'], (policy.config.latent_dim,))
            actions = sample_ace_batch(policy, batch, noise, sampling['sample_batch_size'], horizon)
            actions = actions * post['action.std'] + post['action.mean']
            positions = actions[..., :3].cumsum(axis=2)
            scores.extend([[ace_histogram_entropy(p, width)] for p in positions])
            episodes.extend(ep); frames.extend(fr)
            if spec.get('max_frames') is not None and len(frames) >= spec['max_frames']:
                return episodes, frames, scores
        print(f'ACE completed {path.name}; frames={len(frames)}', flush=True)
    return episodes, frames, scores


def conformal_threshold(values, alpha):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Calibration values must be finite')
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else math.inf


def robust_scale(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 1.0
    spread = np.quantile(values, .75) - np.quantile(values, .25)
    return float(max(spread, np.std(values), 1e-8))


def calibrate_curves(source, calibration_ids, timeline, alpha, scheme, options):
    """Calibrate once on success episodes; use no test values or onset labels.

    wall_time_bin (legacy time_varying) uses maxima within elapsed-wall-time bins. This
    is binwise CP, not simultaneous whole-episode error control. Unsupported
    bins return +inf, without last-value padding or hindsight time alignment.
    """
    from functional_conformal import canonical_scheme, calibrate_band, apply_band
    scheme = canonical_scheme(scheme)
    if scheme == 'functional_band':
        template = options.get('template')
        if template is None:
            raise ValueError('functional_band requires a frozen expert-validation template; use functional_benchmark.py')
        band = calibrate_band(source, timeline, calibration_ids, template, alpha)
        return apply_band(source, timeline, template, band), [band]
    width = float(options.get('bin_seconds', 1.0))
    if not math.isfinite(width) or width <= 0:
        raise ValueError('bin_seconds must be finite and positive')
    bins = {}
    for episode, curve in source.items():
        elapsed = curve['wall'] - timeline[episode]['wall'][0]
        if (elapsed < 0).any() or not np.isfinite(elapsed).all():
            raise ValueError('Scored frames must be at/after the recording start')
        bins[episode] = np.floor(elapsed / width).astype(np.int64)
    fitted = {}
    if scheme == 'episode':
        maxima = [source[e]['score'].max() for e in calibration_ids]
        fitted[None] = (conformal_threshold(maxima, alpha), robust_scale(maxima), len(maxima))
    elif scheme == 'wall_time_bin':
        for b in sorted(set(np.concatenate([bins[e] for e in calibration_ids]).tolist())):
            maxima = [source[e]['score'][bins[e] == b].max() for e in calibration_ids if (bins[e] == b).any()]
            fitted[b] = (conformal_threshold(maxima, alpha), robust_scale(maxima), len(maxima))
    else:
        raise ValueError(f'Unknown calibration scheme: {scheme}')
    transformed = {}
    for episode, curve in source.items():
        thresholds, scales, support = [], [], []
        for b in bins[episode]:
            threshold, scale, n = fitted.get(None if scheme == 'episode' else int(b), (math.inf, 1.0, 0))
            thresholds.append(threshold); scales.append(scale); support.append(n)
        thresholds = np.asarray(thresholds)
        transformed[episode] = {**curve, 'raw_score': curve['score'],
            'score': (curve['score'] - thresholds) / np.asarray(scales),
            'threshold': thresholds, 'support': np.asarray(support)}
    records = [{'bin_index': b, 'bin_start_s': b * width if b is not None else None,
                'bin_end_s': (b + 1) * width if b is not None else None,
                'threshold': t if math.isfinite(t) else '+inf', 'scale': scale,
                'n_calibration': n, 'rank': math.ceil((n + 1) * (1 - alpha)), 'alpha': alpha}
               for b, (t, scale, n) in fitted.items()]
    return transformed, records


def combine_curves(left, right, operation):
    """AND=min / OR=max of signed normalized threshold margins at the SAME frame.

    Join on raw recording indices, checking wall time; never combine alarms
    from two different points of an episode or correlate frames by row number.
    """
    if operation not in {'and', 'or'} or set(left) != set(right):
        raise ValueError('Invalid fusion operation/episode sets')
    result = {}
    for episode, lhs in left.items():
        rhs = right[episode]
        raw, a, b = np.intersect1d(lhs['raw'], rhs['raw'], return_indices=True)
        if not len(raw) or not np.allclose(lhs['wall'][a], rhs['wall'][b], rtol=0, atol=1e-6):
            raise ValueError(f'Fusion raw-frame/wall-time mismatch at episode {episode}')
        fn = np.minimum if operation == 'and' else np.maximum
        result[episode] = {'raw': raw, 'wall': lhs['wall'][a], 'frame': lhs['frame'][a],
                          'score': fn(lhs['score'][a], rhs['score'][b])}
    return result


def episode_auroc(outcomes, scores):
    """Mann-Whitney probability with half credit for ties; one value/episode."""
    outcomes = np.asarray(outcomes, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(outcomes) or np.isnan(scores).any():
        raise ValueError('Malformed episode AUROC input')
    pos, neg = scores[outcomes], scores[~outcomes]
    if not len(pos) or not len(neg):
        return None
    return float(np.mean((pos[:, None] > neg[None, :]) + .5 * (pos[:, None] == neg[None, :])))


def detector_definitions(methods, config):
    joint = config.get('wm_score_aggregation', 'per_view') == 'joint'
    if config.get('wm_score_aggregation', 'per_view') not in {'joint', 'per_view'}:
        raise ValueError('wm_score_aggregation must be joint or per_view')
    definitions = [(method, view, None) for method in methods
                   for view in (('both',) if method == 'ace' or method.startswith('act_ensemble_')
                                or (joint and method.startswith('wm_')) else CAMERAS)]
    fusion = config.get('fusion', {})
    if joint and (fusion.get('wm_views', True) or fusion.get('conditional_marginal', True)):
        raise ValueError('Joint WM scoring requires disabling per-view alarm fusion')
    operations = fusion.get('operations', ['and', 'or'])
    if not operations or not set(operations) <= {'and', 'or'} or len(operations) != len(set(operations)):
        raise ValueError('Choose unique fusion operations from and/or')
    if 'wm_nc' in methods and fusion.get('wm_views', True):
        for op in operations:
            definitions.append((f'wm_nc_{op}', 'both', (('wm_nc', 'fix'), ('wm_nc', 'wrist'), op)))
    if 'wm_nc' in methods and fusion.get('conditional_marginal', True):
        for marginal in ['logpzo', 'rnd', 'logpzo_native', 'rnd_native']:
            if marginal in methods:
                for op in operations:
                    for view in CAMERAS:
                        definitions.append((f'wm_nc_{marginal}_{op}', view, (('wm_nc', view), (marginal, view), op)))
    return definitions


def aggregate_wm_scores(method, scores, aggregation):
    """Reduce each endpoint before calibration; keep original caches unchanged.

    NC/PE are sums of coordinate squared norms/errors. PU is the mean of
    coordinate standard deviations, so two equally sized views are averaged.
    This is feature-space reduction, never OR/AND of calibrated alarms.
    """
    values = np.asarray(scores)
    if aggregation == 'per_view':
        return values
    if aggregation != 'joint' or method not in {'wm_nc', 'wm_pe', 'wm_pu'}:
        raise ValueError('Unknown joint WM score aggregation')
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError('Joint WM scores require fixed,wrist columns at identical endpoints')
    return values.mean(axis=1, keepdims=True) if method == 'wm_pu' else values.sum(axis=1, keepdims=True)


def report_detector(job, view, method, scheme, paper, causal, timeline, labels, annotators):
    """Consume calibrated margin curves (alarm iff >0); annotation is read-only."""
    lookup = episode_labels(labels['tasks'][job.task])
    result = {key: [] for key in ['episodes', 'latencies', 'distributions']}
    common = {'task': job.task, 'method': method, 'view': view, 'calibration': scheme}
    for episode, group in lookup.items():
        curve = paper[episode]
        alarm = first_alarm(curve, 0)
        fields = {key: group[key] for key in ['split', 'outcome', 'failure_mode', 'failure_origin']}
        base = {**common, 'episode_index': episode, **fields}
        # Functional ROC sweeps h on max((r-mu)/g), not raw score maxima.
        maximum = float(curve.get('roc_score', curve['score']).max())
        result['episodes'].append({**base, 'alarm': alarm is not None,
            'score_max': maximum, 'n_scores': len(curve['score']),
            'n_supported_scores': int(np.isfinite(curve['score']).sum()),
            'first_alarm_raw_frame': alarm['raw'] if alarm else None,
            'first_alarm_wall_timestamp': alarm['wall'] if alarm else None})
        if group['outcome'] != 'failure':
            continue
        causal_alarm = first_alarm(causal[episode], 0)
        for annotator in annotators:
            entry = labels['tasks'][job.task].get('failure_onsets', {}).get(episode, {}).get(annotator)
            # A combined detector is timed against each camera's onset separately.
            # Missing fixed labels never borrow a wrist onset.
            for onset_view in (CAMERAS if view == 'both' else (view,)):
                annotation = annotation_views(entry).get(onset_view)
                row = {**base, 'onset_view': onset_view, 'annotator_id': annotator, 'annotator': labels['annotators'][annotator]['name'],
                    'onset_raw_frame': annotation['frame_index'] if annotation else None,
                    'copied_from_annotator': annotation.get('copied_from_annotator') if annotation else None,
                    'onset_wall_timestamp': None, 'first_alarm_raw_frame': causal_alarm['raw'] if causal_alarm else None,
                    'first_alarm_wall_timestamp': causal_alarm['wall'] if causal_alarm else None,
                    'signed_first_alarm_delay_s': None, 'detection_delay_s': None,
                    'last_scored_wall_timestamp': float(causal[episode]['wall'][-1]), 'status': 'unannotated'}
                if annotation:
                    from benchmark_consensus_onsets import annotation_timestamp
                    timestamp = annotation_timestamp(annotation, lambda: onset_time(job.dataset, episode, annotation['frame_index'], timeline))
                    row['onset_wall_timestamp'] = timestamp
                    if timestamp is None:
                        row['status'] = 'timestamp_unavailable'
                    elif causal_alarm and causal_alarm['raw'] < annotation['frame_index']:
                        row['status'] = 'pre_onset_alarm'
                        row['signed_first_alarm_delay_s'] = causal_alarm['wall'] - timestamp
                    elif annotation['frame_index'] > int(causal[episode]['raw'][-1]):
                        row['status'] = 'onset_after_scored_end'
                    elif causal_alarm:
                        row['status'] = 'detected'
                        row['detection_delay_s'] = row['signed_first_alarm_delay_s'] = causal_alarm['wall'] - timestamp
                    else:
                        row['status'] = 'missed'
                result['latencies'].append(row)
    for group in labels['tasks'][job.task]['groups']:
        ids = range(group['episode_range'][0], group['episode_range'][1] + 1)
        for score_type in ['margin', 'raw']:
            key = 'score' if score_type == 'margin' else 'raw_score'
            if key not in paper[next(iter(ids))]:
                continue
            values = np.concatenate([paper[e][key] for e in ids])
            finite = values[np.isfinite(values)]
            result['distributions'].append({**common, **{k: group[k] for k in ['split', 'outcome', 'failure_mode', 'failure_origin']},
                'score_type': score_type, 'n_episodes': len(ids), 'n_scores': len(values),
                'n_unsupported_scores': len(values) - len(finite), **statistics(finite)})
    return result


def evaluate_suite(scored, labels, config, annotators, methods, schemes, expert_templates=None, curve_observer=None):
    reports = {key: [] for key in ['episodes', 'latencies', 'distributions', 'calibration']}
    frozen = None
    if config.get('evaluation_protocol'):
        from benchmark_frozen_protocol import validate_config, apply_offline
        frozen = validate_config(config)
        from benchmark_consensus_onsets import derive_labels
        labels = derive_labels(labels, frozen)
        if frozen['detection']['primary_annotator'] not in annotators:
            annotators = [frozen['detection']['primary_annotator'], *annotators]
    definitions = detector_definitions(methods, config)
    for task in dict.fromkeys(job.task for job, _, _ in scored):
        jobs = [(job, arrays, timeline) for job, arrays, timeline in scored if job.task == task]
        lookup = episode_labels(labels['tasks'][task])
        calibration_ids = [e for e, g in lookup.items() if g['split'] == 'calibration']
        for scheme in schemes:
            streams, contexts = {}, {}
            for job, arrays, timeline in jobs:
                aggregation = config.get('wm_score_aggregation', 'per_view') if job.method.startswith('wm_') else 'per_view'
                output_views = job.views
                if aggregation == 'joint':
                    if job.views != ('fix', 'wrist') or job.merged.get('backbone', {}).get('view_fusion') != 'concat_channels':
                        raise ValueError('Joint fused scoring requires a concat_channels model with both views')
                    arrays = {**arrays, 'scores': aggregate_wm_scores(job.method, arrays['scores'], aggregation)}
                    output_views = ('both',)
                for index, view in enumerate(output_views):
                    key = (job.method, view)
                    contexts[key] = (job, timeline)
                    streams[key] = {}
                    for name, protocol in [('paper', config['paper_protocol']), ('latency', config['latency_protocol'])]:
                        source = curves(arrays, job, index, protocol)
                        options = config.get('wall_time_bin', config.get('time_varying', {}))
                        if scheme == 'functional_band':
                            options = {'template': (expert_templates or {}).get((task, job.method, view, name))}
                        calibrated, records = calibrate_curves(source, calibration_ids, timeline, config['alpha'], scheme, options)
                        streams[key][name] = calibrated
                        reports['calibration'].extend({'task': task, 'method': job.method, 'view': view,
                            'calibration': scheme, 'protocol': name, **record} for record in records)
            for method, view, fusion in definitions:
                if scheme == 'functional_band' and fusion:
                    # Calibrated-component fusion has no independent raw template
                    # here; do not claim a joint conformal band for these detectors.
                    continue
                if fusion:
                    lhs, rhs, op = fusion
                    paper = combine_curves(streams[lhs]['paper'], streams[rhs]['paper'], op)
                    causal = combine_curves(streams[lhs]['latency'], streams[rhs]['latency'], op)
                    job, timeline = contexts[lhs]
                else:
                    paper, causal = streams[(method, view)]['paper'], streams[(method, view)]['latency']
                    job, timeline = contexts[(method, view)]
                evaluated = report_detector(job, view, method, scheme, paper, causal, timeline, labels, annotators)
                if frozen:
                    apply_offline(evaluated, job, view, paper, causal, timeline, labels, frozen)
                if curve_observer is not None:
                    curve_observer(job, view, method, scheme, paper, causal, evaluated)
                for key, rows in evaluated.items():
                    reports[key].extend(rows)
    return reports


def aggregate_metrics(episodes):
    rows = []
    keys = sorted({(r['task'], r['method'], r['view'], r['calibration']) for r in episodes})
    for task, method, view, scheme in keys:
        selected = [r for r in episodes if (r['task'], r['method'], r['view'], r['calibration']) == (task, method, view, scheme)
                    and r['split'] == 'test']
        negatives = [r for r in selected if r['outcome'] == 'success']
        positives = [r for r in selected if r['outcome'] == 'failure']
        far_n, fdr_n = sum(r['alarm'] for r in negatives), sum(r['alarm'] for r in positives)
        rows.append({'task': task, 'method': method, 'view': view, 'calibration': scheme,
            'false_alarms': far_n, 'success_episodes': len(negatives), 'FAR': far_n / len(negatives) if negatives else None,
            'detected_failures': fdr_n, 'failure_episodes': len(positives), 'FDR': fdr_n / len(positives) if positives else None,
            'episode_AUROC': episode_auroc([r['outcome'] == 'failure' for r in selected], [r['score_max'] for r in selected]),
            'unsupported_episodes': sum(r['n_supported_scores'] == 0 for r in selected)})
    for method, view, scheme in sorted({(r['method'], r['view'], r['calibration']) for r in rows}):
        selected = [r for r in rows if (r['method'], r['view'], r['calibration']) == (method, view, scheme)]
        neg, pos = sum(r['success_episodes'] for r in selected), sum(r['failure_episodes'] for r in selected)
        fn, tp = sum(r['false_alarms'] for r in selected), sum(r['detected_failures'] for r in selected)
        aucs = [r['episode_AUROC'] for r in selected if r['episode_AUROC'] is not None]
        rows.append({'task': 'Overall', 'method': method, 'view': view, 'calibration': scheme,
            'false_alarms': fn, 'success_episodes': neg, 'FAR': fn / neg if neg else None,
            'detected_failures': tp, 'failure_episodes': pos, 'FDR': tp / pos if pos else None,
            'episode_AUROC': float(np.mean(aucs)) if aucs else None,
            'unsupported_episodes': sum(r['unsupported_episodes'] for r in selected)})
    return rows


def paper_table(episodes, config, labels, tasks, methods):
    columns = [(method, view) for method, view, _ in detector_definitions(methods, config)]
    header = ['Task', 'Failure Mode'] + [f"{METHODS[m][0] if m in METHODS else m} {VIEWS[v]}" for m, v in columns]
    rows = []
    for task in tasks:
        for group in labels['tasks'][task]['groups']:
            if group['split'] != 'test':
                continue
            ids = set(range(group['episode_range'][0], group['episode_range'][1] + 1))
            row = [config['tasks'][task]['title'], group['failure_mode'] or 'No Failure']
            for method, view in columns:
                selected = [r for r in episodes if r['task'] == task and r['method'] == method
                            and r['view'] == view and r['episode_index'] in ids]
                if len(selected) != len(ids):
                    raise ValueError(f'Incomplete table cell: {task}/{method}/{view}')
                row.append(f"{sum(r['alarm'] for r in selected)}/{len(selected)}")
            rows.append(row)
    for outcome, metric in [('success', 'FAR'), ('failure', 'FDR')]:
        row = ['Overall', f'Overall {metric}']
        for method, view in columns:
            selected = [r for r in episodes if r['method'] == method and r['view'] == view
                        and r['split'] == 'test' and r['outcome'] == outcome]
            if selected:
                percent = Decimal(100 * sum(r['alarm'] for r in selected)) / Decimal(len(selected))
                row.append(f"{percent.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%")
            else:
                row.append('N/A')
        rows.append(row)
    return header, rows


def latency_summary(rows: list[dict]) -> list[dict]:
    result = []
    keys = sorted({(row["task"], row["method"], row["view"], row["annotator_id"], row["calibration"], row["onset_view"]) for row in rows})
    statuses = ["detected", "pre_onset_alarm", "missed", "onset_after_scored_end", "timestamp_unavailable", "unannotated"]
    for task, method, view, annotator, scheme, onset_view in keys:
        selected = [row for row in rows if (row["task"], row["method"], row["view"], row["annotator_id"], row["calibration"], row["onset_view"]) == (task, method, view, annotator, scheme, onset_view)]
        counts = Counter(row["status"] for row in selected)
        delays = [row["detection_delay_s"] for row in selected if row["status"] == "detected"]
        result.append({"task": task, "method": method, "view": view, "annotator_id": annotator,
            "calibration": scheme, "onset_view": onset_view, "annotator": selected[0]["annotator"], "n_failure_episodes": len(selected),
            "n_annotated": len(selected)-counts["unannotated"],
            "annotation_complete": counts["unannotated"] == 0,
            "n_copied": sum(bool(row.get('copied_from_annotator')) for row in selected),
            "timing_complete": counts["unannotated"] == 0 and counts["timestamp_unavailable"] == 0,
            "n_early_alarm": sum(bool(row.get('early_alarm')) for row in selected),
            "n_detected_before_human_label": sum(bool(row.get('detected_before_human_label')) for row in selected),
            **{f"n_{status}": counts[status] for status in statuses},
            **{f"delay_{key}_s": value for key, value in statistics(delays).items()}})
    return result


def write_csv(path: Path, rows: list[dict], fallback: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)) if rows else fallback)
        writer.writeheader()
        writer.writerows(rows)


def install_ode_meter(metadata: Path) -> None:
    """Measure actual fixed/adaptive ODE steps and evaluations without changing solves.

    Installed only inside inference subprocesses, before model imports. Counters
    are batch-size weighted so summaries describe the work seen by each example.
    No GPU synchronizations, tensor copies, or changes to integration options.
    """
    import atexit
    import torchdiffeq
    from torchdiffeq._impl.odeint import SOLVERS
    original = torchdiffeq.odeint
    counts = Counter()

    def measured(func, y0, t, *args, **kwargs):
        solver = kwargs.get('method') or 'dopri5'
        work = Counter()

        class Counted:
            def __call__(self, *a, **kw):
                work['evaluations'] += 1
                return func(*a, **kw)

        counted = Counted()
        callbacks = SOLVERS[solver].valid_callbacks()
        for name in callbacks:
            def callback(*a, _name=name, **kw):
                work[_name] += 1
                if hasattr(func, _name):
                    return getattr(func, _name)(*a, **kw)
            setattr(counted, name, callback)
        result = original(counted, y0, t, *args, **kwargs)
        steps = work['callback_accept_step'] if 'callback_accept_step' in callbacks else work['callback_step']
        batch_size = int((y0[0] if isinstance(y0, tuple) else y0).shape[0])
        counts[solver, steps, work['evaluations'], work['callback_step'], work['callback_reject_step']] += batch_size
        return result

    def save():
        if not metadata.is_file():
            return
        value = json.loads(metadata.read_text())
        total = sum(counts.values())
        stats = {}
        for name, index in [('steps', 1), ('evaluations', 2), ('attempted_steps', 3), ('rejected_steps', 4)]:
            values = [key[index] for key in counts]
            stats.update({f'{name}_min': min(values) if values else None,
                          f'{name}_max': max(values) if values else None,
                          f'{name}_mean': sum(key[index] * n for key, n in counts.items()) / total if total else None})
        value['ode_work'] = {'solves': total, 'basis': 'Measured per-example per-solve; accepted steps, with rejected attempts recorded separately',
                             'solvers': sorted({key[0] for key in counts}), **stats}
        write_json(metadata, value)

    torchdiffeq.odeint = measured
    atexit.register(save)


def scoring_settings(job: Job, measured: dict | None = None) -> dict:
    """Describe the actual integrator used by the existing scoring code.

    Legacy FM calls use endpoint-only time grids (one fixed-solver step).
    Validated WM-PU runs can set explicit RK4 steps; each step evaluates the
    vector field four times. Adaptive solver counts are not fixed.
    Both camera branches are evaluated together; do not double the solve count.
    """
    head = job.merged.get('fm_head', {})
    sampling = job.merged.get('sampling', {})
    runtime = job.merged.get('runtime', {})
    row = {'task': job.task, 'method': job.method, 'views': ','.join(job.views),
           'checkpoint': str(job.checkpoint), 'dataset': str(job.dataset),
           'K': job.merged['data'].get('pred_horizon'), 'L': job.merged['data'].get('history_len'),
           'batch_size': job.batch_size, 'seed': job.seed,
           'sample_batch_size': job.merged.get('sampling', {}).get('sample_batch_size'),
           'sampling_seed_scheme': job.merged.get('sampling', {}).get('seed_scheme'),
           'float32_matmul_precision': job.merged.get('sampling', {}).get('matmul_precision', 'highest'),
           'precision': runtime.get('amp_dtype', 'float32') if runtime.get('use_amp') and job.device.startswith('cuda') else 'float32',
           'ode_solver': None, 'ode_direction': None, 'ode_steps_per_solve': None,
           'vector_field_evals_per_solve': None, 'ode_solves_per_score': None,
           'ode_steps_per_score': None, 'vector_field_evals_per_score': None,
           'ode_rtol': None, 'ode_atol': None, 'samples_per_score': None, 'ode_count_basis': 'Not applicable'}
    if job.method.startswith('act_ensemble_'):
        from act_ensemble_score import BOUNDARY
        row.update(reproduction_boundary=BOUNDARY, ensemble=job.merged['ensemble'],
                   score_units='m^2 per observation' if job.method.endswith('_pos') else 'rad per observation')
    if job.method in sampled_protocol.BOUNDARIES:
        row.update(reproduction_boundary=sampled_protocol.BOUNDARIES[job.method],
                   sampling_manifest=sampling.get('sampling_manifest'),
                   ace_cell_widths_m=sampling.get('cell_widths_m'))
    measured = measured or {}
    row.update({f'measured_{name}': measured.get(name) for name in
                ['steps_min', 'steps_mean', 'steps_max', 'evaluations_min', 'evaluations_mean', 'evaluations_max',
                 'attempted_steps_mean', 'rejected_steps_mean', 'solves']})
    if job.method == 'ace':
        row.update(precision='float32', samples_per_score=job.merged.get('sampling', {}).get('num_samples'))
    if not job.method.startswith('wm_') and job.method not in {'logpzo', 'logpzo_native'}:
        return row
    if job.method == 'wm_pu':
        row['precision'] = job.merged.get('sampling', {}).get('precision', 'float32') if job.device.startswith('cuda') else 'float32'
    reverse = job.method in {'wm_nc', 'logpzo', 'logpzo_native'}
    transport = head.get('score_transport_mode', 'ode' if job.method.startswith('logpzo') else 'one_step_t0')
    if reverse and transport != 'ode':
        row.update(ode_solver='direct_endpoint', ode_steps_per_solve=0,
                   vector_field_evals_per_solve=1, ode_solves_per_score=0,
                   ode_steps_per_score=0, vector_field_evals_per_score=1,
                   ode_count_basis='Direct endpoint velocity approximation; no ODE solver')
        return row
    solver = head.get('ode_solver', 'dopri5')
    samples = head.get('per_chunk_wm_samples', 1) if job.method == 'wm_pu' else 1
    fixed_evals = {'euler': 1, 'midpoint': 2, 'rk4': 4}
    steps = (sampling.get('ode_steps', 1) if job.method == 'wm_pu' else 1) if solver in fixed_evals else None
    nfe = fixed_evals[solver]*steps if solver in fixed_evals else None
    row.update(ode_solver=solver, ode_direction='1 -> 0' if reverse else '0 -> 1',
               ode_steps_per_solve=steps, vector_field_evals_per_solve=nfe,
               ode_solves_per_score=samples, ode_steps_per_score=steps * samples if steps is not None else None,
               vector_field_evals_per_score=nfe * samples if nfe is not None else None,
               ode_rtol=head.get('ode_rtol', 1e-4), ode_atol=head.get('ode_atol', 1e-5),
               samples_per_score=samples if job.method == 'wm_pu' else None,
               ode_count_basis='Fixed solver: explicit validated grid' if steps is not None and steps > 1 else 'Fixed solver: endpoint-only grid' if steps is not None
               else 'Adaptive solver: actual step/evaluation counts vary and are not recorded')
    return row


def markdown_table(header, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(map(cell, header)) + ' |',
                      '| ' + ' | '.join(['---'] * len(header)) + ' |']
                     + ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]) + '\n'


def write_inspection_reports(directory: Path, report: dict, tables: dict, distributions: list[dict]) -> None:
    """Standalone Markdown/HTML from the same final metrics, without dependencies."""
    def display(value, *, percent=False):
        if value is None:
            return '—'
        if isinstance(value, float):
            return f'{100 * value:.1f}%' if percent else f'{value:.4g}'
        return str(value)

    sections = []
    notes = [f"Created {report['created_utc']}. Tasks: {', '.join(report['tasks'])}.",
             f"Calibration alpha: {report['alpha']}; smoothing windows: paper={report['protocols']['paper_protocol']['window']}, latency={report['protocols']['latency_protocol']['window']} (1 means no smoothing).",
             'FAR/FDR are episode counts; calibration episodes are excluded from test metrics. Overall FAR/FDR pool episode counts; overall AUROC averages tasks equally.',
             'ODE steps below are integration steps, not vector-field evaluations. Current RK4 uses 1 step and 4 evaluations per solve. Fixed/wrist scores share a solve; fusion reuses component scores.',
             report['latency_note'], report['time_varying_note'], report['fusion_note'],
             'Copied human labels retain their source in labels_snapshot.yaml; annotators are reported separately. A copied label is not an independent annotation.']
    settings_header = ['Task', 'Method', 'Views', 'K', 'L', 'Solver', 'Direction', 'Steps/solve', 'Evals/solve',
                       'Solves/score', 'Steps/score', 'Evals/score', 'Samples', 'Precision', 'Batch',
                       'Sample batch', 'Sampling RNG', 'Float32 matmul',
                       'Measured steps min', 'Measured steps mean', 'Measured steps max', 'Measured evals mean']
    settings_keys = ['task', 'method', 'views', 'K', 'L', 'ode_solver', 'ode_direction', 'ode_steps_per_solve',
                     'vector_field_evals_per_solve', 'ode_solves_per_score', 'ode_steps_per_score',
                     'vector_field_evals_per_score', 'samples_per_score', 'precision', 'batch_size', 'sample_batch_size', 'sampling_seed_scheme',
                     'float32_matmul_precision',
                     'measured_steps_min', 'measured_steps_mean', 'measured_steps_max', 'measured_evaluations_mean']
    sections.append(('scoring-settings', 'Scoring settings and ODE work', settings_header,
                     [[display(row.get(k)) for k in settings_keys] for row in report['scoring_settings']], True))
    for scheme in report['calibration_schemes']:
        metrics = [r for r in report['summary_metrics'] if r['calibration'] == scheme]
        header = ['Task', 'Method', 'View', 'False alarms', 'FAR', 'Detected failures', 'FDR', 'Episode AUROC', 'Unsupported episodes']
        rows = [[r['task'], r['method'], VIEWS[r['view']], f"{r['false_alarms']}/{r['success_episodes']}", display(r['FAR'], percent=True),
                 f"{r['detected_failures']}/{r['failure_episodes']}", display(r['FDR'], percent=True),
                 display(r['episode_AUROC']), r['unsupported_episodes']] for r in metrics]
        overall = [row for row in rows if row[0] == 'Overall']
        sections.append((f'{scheme}-overall', f'{scheme}: overall FAR, FDR and AUROC', header, overall, True))
        sections.append((f'{scheme}-metrics', f'{scheme}: metrics by task and detector', header, rows, False))
        paper_header, paper_rows = tables[scheme]
        sections.append((f'{scheme}-paper', f'{scheme}: paper-style failure-mode table', paper_header, paper_rows, False))
        latency = [r for r in report['latency_summary'] if r['calibration'] == scheme]
        latency_header = ['Task', 'Method', 'View', 'Onset view', 'Annotator', 'Labeled/total', 'Copied',
                          'Detected after onset', 'Pre-onset alarms', 'Missed', 'Unannotated', 'Timestamp unavailable',
                          'Onset after scored end', 'Mean delay (s)', 'Median delay (s)', 'P95 delay (s)']
        latency_rows = [[r['task'], r['method'], VIEWS[r['view']], VIEWS[r['onset_view']], r['annotator'],
                         f"{r['n_annotated']}/{r['n_failure_episodes']}", r.get('n_copied', 0), r['n_detected'],
                         r['n_pre_onset_alarm'], r['n_missed'], r['n_unannotated'], r['n_timestamp_unavailable'],
                         r['n_onset_after_scored_end'], display(r['delay_mean_s']), display(r['delay_median_s']),
                         display(r['delay_p95_s'])] for r in latency]
        sections.append((f'{scheme}-latency', f'{scheme}: latency and annotation coverage', latency_header, latency_rows, False))
    dist_header = ['Task', 'Method', 'View', 'Calibration', 'Split', 'Failure mode', 'Origin', 'Score type',
                   'Episodes', 'Scores', 'Unsupported', 'Mean', 'Std', 'Median', 'P95']
    dist_keys = ['task', 'method', 'view', 'calibration', 'split', 'failure_mode', 'failure_origin', 'score_type',
                 'n_episodes', 'n_scores', 'n_unsupported_scores', 'mean', 'std', 'median', 'p95']
    sections.append(('distributions', 'Score distribution summaries', dist_header,
                     [[display(row.get(k)) for k in dist_keys] for row in distributions], False))
    artifacts = ['report.json', 'summary_metrics.csv', 'scoring_settings.csv', 'episode_results.csv',
                 'latency_summary.csv', 'latency_episodes.csv', 'score_distributions.csv', 'calibration.csv',
                 'labels_snapshot.yaml', 'benchmark_config.json']
    artifacts += ['paper_table.csv' if s == 'episode' else f'paper_table_{s}.csv' for s in tables]
    md = ['# Unified failure benchmark', '', *[text + '\n' for text in notes],
          'Files: ' + ' · '.join(f'[{name}]({name})' for name in artifacts), '']
    html_sections = []
    for sid, title, header, rows, opened in sections:
        md += ['## ' + title, '', markdown_table(header, rows)]
        table = '<div class="table-scroll"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in header) + '</tr></thead><tbody>'
        table += ''.join('<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in row) + '</tr>' for row in rows)
        table += '</tbody></table></div>'
        html_sections.append(f'<details id="{sid}" {"open" if opened else ""}><summary>{html.escape(title)} <small>{len(rows)} rows</small></summary>{table}</details>')
    (directory / 'report.md').write_text('\n'.join(md), encoding='utf-8')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unified failure benchmark</title><style>
:root{font:15px/1.5 system-ui,sans-serif;color:#182638;background:#f3f6fa}body{margin:0}main{max-width:1600px;margin:auto;padding:28px}
h1{font-size:28px;margin:0 0 14px}p{margin:8px 0}.notes{padding:18px;background:white;border-radius:10px;border-left:4px solid #236ac8}
nav{position:sticky;top:0;z-index:3;background:#f3f6faf2;padding:12px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
input,button{font:inherit;padding:8px 12px;border:1px solid #bdc9d8;border-radius:6px;background:white}input{min-width:300px}
details{margin:16px 0;background:white;border:1px solid #d7e0eb;border-radius:8px;overflow:hidden}summary{padding:14px;cursor:pointer;font-weight:600}small{font-weight:400;color:#52647b;margin-left:8px}
.table-scroll{overflow:auto;max-height:72vh}table{border-collapse:collapse;min-width:100%;font-size:13px}th,td{padding:8px 12px;border-bottom:1px solid #e5eaf0;white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:#eaf0f8;z-index:1;text-align:left}tr:nth-child(even){background:#f8fafc}tr:hover{background:#eaf3ff}a{color:#175bb1}footer{margin-top:28px;color:#53647a}
@media print{nav{display:none}.table-scroll{max-height:none;overflow:visible}main{padding:0}details{break-inside:avoid}th{position:static}}
</style></head><body><main><h1>Unified failure benchmark</h1>'''
    page += '<div class="notes">' + ''.join(f'<p>{html.escape(text)}</p>' for text in notes) + '</div>'
    page += '<nav><label>Filter rows <input id="filter" type="search" placeholder="e.g. shunt_insertion wm_nc wrist"></label><button id="expand">Expand all</button><button id="collapse">Collapse all</button><a href="report.md">Markdown report</a></nav>'
    page += ''.join(html_sections) + '<footer>Download: ' + ' · '.join(f'<a href="{name}">{name}</a>' for name in artifacts) + '</footer>'
    page += '''</main><script>
document.getElementById('filter').addEventListener('input',e=>{const words=e.target.value.toLowerCase().trim().split(/\\s+/).filter(Boolean);for(const row of document.querySelectorAll('tbody tr'))row.hidden=!words.every(w=>row.textContent.toLowerCase().includes(w));});
document.getElementById('expand').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=true);
document.getElementById('collapse').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=false);
</script></body></html>'''
    (directory / 'report.html').write_text(page, encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "lerobot/configs/benchmark/unified_benchmark.yaml")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS))
    parser.add_argument("--calibrations", nargs="+", choices=["episode", "wall_time_bin", "time_varying", "functional_band"])
    parser.add_argument('--functional-templates', type=Path, help='Frozen template manifest from functional_benchmark.py')
    parser.add_argument("--allow-inference", action="store_true", help="Explicitly allow missing scores to load models and run inference")
    parser.add_argument("--mode", choices=["normal", "override"], default="normal")
    parser.add_argument("--cache-only", action="store_true", help="Require valid existing caches; never launch inference")
    parser.add_argument('--reuse-score-report', type=Path,
                        help='With --cache-only, recompute metrics from the exact SHA-256-verified caches of a saved report')
    parser.add_argument("--dry-run", action="store_true", help="Only read configs/metadata and list jobs; no models, hashes or writes")
    parser.add_argument("--device", default="cuda", help="Used only for inference/cache identity; cuda, cuda:N, or cpu")
    parser.add_argument("--batch-size", type=int, help="Override score batching (part of cache identity)")
    parser.add_argument("--annotator", action="append", help="Annotator ID or name; default reports each registered annotator separately")
    parser.add_argument("--report-dir", type=Path, help="New report directory; existing directories are rejected")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != 1 or not 0 < config["alpha"] < 1:
        raise ValueError("Unsupported benchmark schema/alpha")
    if type(config["score_seed"]) is not int or config["score_seed"] < 0:
        raise ValueError("score_seed must be a nonnegative integer")
    if type(args.batch_size if args.batch_size is not None else config.get('score_batch_size', 1)) is not int or (args.batch_size if args.batch_size is not None else config.get('score_batch_size', 1)) < 1:
        raise ValueError("Score batch size must be positive")
    if args.cache_only and args.mode == "override":
        raise ValueError("--cache-only cannot be combined with --mode override")
    if args.reuse_score_report and (not args.cache_only or args.allow_inference or args.mode != 'normal'):
        raise ValueError('--reuse-score-report requires --cache-only without inference or override')
    if args.mode == "override" and not args.allow_inference and not args.dry_run:
        raise ValueError("--mode override requires --allow-inference")
    if args.device != "cpu" and not (args.device == "cuda" or (args.device.startswith("cuda:") and args.device[5:].isdigit())):
        raise ValueError("Device must be cpu, cuda or cuda:N")
    validate_protocol(config["paper_protocol"])
    validate_protocol(config["latency_protocol"], causal=True)
    if config.get('evaluation_protocol'):
        from benchmark_frozen_protocol import validate_config
        validate_config(config)
    tasks = args.tasks or list(config["tasks"])
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in config["tasks"] for task in tasks):
        raise ValueError("Choose unique tasks listed in the benchmark configuration")
    args.methods = args.methods or config.get("methods", list(METHODS))
    from functional_conformal import canonical_scheme
    schemes = [canonical_scheme(s) for s in (args.calibrations or config.get("calibrations", ["episode", "wall_time_bin"]))]
    if not schemes or len(set(schemes)) != len(schemes) or not set(schemes) <= {"episode", "wall_time_bin", "functional_band"}:
        raise ValueError("Choose unique calibration schemes")
    if not set(args.methods) <= set(METHODS):
        raise ValueError("Unknown benchmark method")
    if "wm_pu" in args.methods:
        validate_sampling(config["wm_pu"])
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("Choose unique methods")
    labels_path = rooted(config["labels"], repo)
    labels, _ = load_labels(labels_path, tasks)
    jobs = make_jobs(config, repo, tasks, args.methods, args)
    expert_templates = None
    if 'functional_band' in schemes:
        if args.functional_templates is None:
            raise ValueError('Use functional_benchmark.py to score expert validation and save templates, then pass --functional-templates')
        from functional_conformal import load_template_manifest
        expert_templates = load_template_manifest(args.functional_templates)
        required = {(j.task, j.method, v, p) for j in jobs
                    for v in (('both',) if config.get('wm_score_aggregation') == 'joint' and j.method.startswith('wm_') else j.views)
                    for p in ['paper', 'latency']}
        if required-set(expert_templates):
            raise ValueError(f'Missing expert templates: {sorted(required-set(expert_templates))}')
    for dataset in {job.dataset for job in jobs}:
        preparation = json.loads((dataset / "meta/preparation.json").read_text())
        task = next(job.task for job in jobs if job.dataset == dataset)
        prepared_ids = [row["dataset_episode_index"] for row in preparation["episode_mapping"]]
        if len(prepared_ids) != len(set(prepared_ids)) or set(prepared_ids) != set(episode_labels(labels["tasks"][task])):
            raise ValueError(f"Preparation episode count/IDs differ from labels: {task}")
    if args.dry_run:
        for job in jobs:
            print(f"{job.task:18} {job.method:7} {','.join(job.views):9} K={job.merged['data'].get('pred_horizon','-')} L={job.merged['data'].get('history_len','-')} batch={job.batch_size}")
            print("  status:", job_readiness(job) or "ready")
            print(f"  config: {job.score_config}\n  checkpoint: {job.checkpoint}\n  PRL: {job.dataset}\n  cache: {job.cache_dir}/<fingerprint>.npz")
        print(f"{len(jobs)} scorer jobs; no models loaded or files written. Frozen sampling manifests, when configured, verify checkpoint hashes.")
        print("Calibration comparisons:", schemes, "Detector columns:", len(detector_definitions(args.methods, config)))
        print("Paper protocol:", config["paper_protocol"], "Latency protocol:", config["latency_protocol"])
        return
    report_dir = args.report_dir or repo / "logs/benchmarks" / datetime.now().strftime("%Y-%m-%d") / datetime.now(timezone.utc).strftime("%H%M%S_%fZ")
    report_dir = report_dir.resolve()
    if report_dir.exists():
        raise FileExistsError(f"Report directory already exists: {report_dir}")
    pending = [f"{job.task}/{job.method}/{job.views}: {job_readiness(job)}" for job in jobs if job_readiness(job)]
    if pending:
        raise ValueError("Scoring requires completed checkpoints; choose completed tasks/methods:\n" + "\n".join(pending))
    planned = []
    reused_report = json.loads((args.reuse_score_report / 'report.json').read_text()) if args.reuse_score_report else None
    for job in jobs:
        validate_native_provenance(job)
        if reused_report is not None:
            matches = [r for r in reused_report['score_caches']
                       if (r['task'], r['method'], tuple(r['views'])) == (job.task, job.method, job.views)]
            if len(matches) != 1:
                raise ValueError('Missing/ambiguous cache in reuse report')
            saved = matches[0]
            path = Path(saved['path'])
            if digest(path) != saved['sha256']:
                raise ValueError(f'Reused score cache failed SHA-256 verification: {path}')
            with np.load(path, allow_pickle=False) as z:
                identity = json.loads(str(z['provenance_json'].item()))['identity']
            expected = {'task':job.task,'method':job.method,'views':job.views,'checkpoint':str(job.checkpoint),
                        'dataset':str(job.dataset),'merged_config':job.merged,'seed':job.seed,
                        'batch_size':job.batch_size,'split':job.split,'device':job.device}
            if any(canonical(relocate_identity(identity.get(k))) != canonical(relocate_identity(v)) for k,v in expected.items()):
                raise ValueError('Reuse report does not match requested model, dataset, or scoring settings')
            planned.append((job, identity, path))
        else:
            identity = fingerprint(job, repo)
            key = hashlib.sha256(canonical(identity).encode()).hexdigest()
            target = job.cache_dir / f"{key}.npz"
            if not target.exists() and args.mode == 'normal':
                # Directory-only model moves do not require GPU rescoring.
                for old in sorted(job.cache_dir.parent.glob('*/*.npz')):
                    with np.load(old, allow_pickle=False) as z:
                        old_identity = json.loads(str(z['provenance_json'].item()))['identity']
                    if canonical(relocate_identity(old_identity)) == canonical(relocate_identity(identity)):
                        target = old
                        print(f'Reusing relocated model cache: {target}', flush=True)
                        break
            planned.append((job, identity, target))
    missing = [str(path) for _, _, path in planned if not path.is_file()]
    if (args.cache_only or not args.allow_inference) and missing:
        raise FileNotFoundError("Matching caches are missing; no models were loaded:\n" + "\n".join(missing))
    timelines, scored, inventories = {}, [], []
    for job, identity, path in planned:
        if reused_report is None and canonical(fingerprint(job, repo)) != canonical(identity):
            raise ValueError("Input changed after benchmark planning; rerun with the intended inputs")
        if job.dataset not in timelines:
            timelines[job.dataset] = load_timeline(job.dataset)
        timeline = timelines[job.dataset]
        if set(timeline) != set(episode_labels(labels["tasks"][job.task])):
            raise ValueError(f"Prepared dataset episode count/IDs differ from labels: {job.task}")
        arrays = acquire_scores(job, repo, timeline, identity, path, args)
        scored.append((job, arrays, timeline))
        with np.load(path, allow_pickle=False) as cached:
            provenance = json.loads(str(cached['provenance_json'].item()))
        measured = provenance.get('environment', {}).get('ode_work', {})
        inventories.append({"task": job.task, "method": job.method, "views": job.views, "path": str(path), "sha256": digest(path),
                            "scoring_settings": scoring_settings(job, measured)})
        if config.get('wm_score_aggregation') == 'joint' and job.method.startswith('wm_'):
            inventories[-1].update(output_views=['both'], score_aggregation='joint')
            inventories[-1]['scoring_settings'].update(output_views='both', score_aggregation='joint')
    labels, labels_text = load_labels(labels_path, tasks)
    for job, _, timeline in scored:
        if set(timeline) != set(episode_labels(labels["tasks"][job.task])):
            raise ValueError(f"Episode labels changed while scoring: {job.task}")
    annotators = list(labels.get("annotators", {}))
    if args.annotator:
        names = {entry["name"]: key for key, entry in labels["annotators"].items()}
        annotators = list(dict.fromkeys(name if name in annotators else names.get(name) for name in args.annotator))
        if None in annotators:
            raise ValueError("Unknown annotator ID/name")
    reports = evaluate_suite(scored, labels, config, annotators, args.methods, schemes, expert_templates)
    summary = latency_summary(reports['latencies'])
    metrics = aggregate_metrics(reports['episodes'])
    report_dir.mkdir(parents=True, exist_ok=False)
    tables = {}
    for scheme in schemes:
        selected = [r for r in reports['episodes'] if r['calibration'] == scheme]
        table_config = merge(config, {'fusion': {'wm_views': False, 'conditional_marginal': False}}) if scheme == 'functional_band' else config
        header, table = paper_table(selected, table_config, labels, tasks, args.methods)
        tables[scheme] = (header, table)
        markdown = markdown_table(header, table)
        # Keep the original filename for full-episode calibration.
        stem = 'paper_table' if scheme == 'episode' else f'paper_table_{scheme}'
        with (report_dir / f'{stem}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(header); writer.writerows(table)
        (report_dir / f'{stem}.md').write_text(markdown)
    (report_dir / "labels_snapshot.yaml").write_text(labels_text)
    write_json(report_dir / "benchmark_config.json", config)
    for name, rows in [("summary_metrics", metrics), ("episode_results", reports["episodes"]), ("score_distributions", reports["distributions"]),
                       ("calibration", reports["calibration"]), ("latency_episodes", reports["latencies"]), ("latency_summary", summary)]:
        write_csv(report_dir / f"{name}.csv", rows, ["task", "method", "view", "annotator_id", "status"])
    settings = [item['scoring_settings'] for item in inventories]
    write_csv(report_dir / 'scoring_settings.csv', settings, ['task', 'method', 'ode_steps_per_solve'])
    report = {"created_utc": utc_now(), "script_sha256": digest(Path(__file__).resolve()), "alpha": config['alpha'],
        "tasks": tasks, "methods": args.methods, "labels_sha256": hashlib.sha256(labels_text.encode()).hexdigest(),
        "score_caches": inventories, "protocols": {key: config[key] for key in ["paper_protocol", "latency_protocol"]},
        "calibration": reports["calibration"], "summary_metrics": metrics, "latency_summary": summary,
        "scoring_settings": settings,
        "latency_note": "Causal first-alarm delay in recorded wall seconds, conditional on detection; pre-onset alarms and misses counted separately. Not GPU inference runtime.",
        "onset_status": "No registered annotators" if not annotators else "See per-annotator/per-camera coverage in latency_summary.csv",
        "calibration_schemes": schemes,
        "wm_score_aggregation": config.get('wm_score_aggregation', 'per_view'),
        "reused_score_report": ({'path': str(args.reuse_score_report.resolve()),
                                 'sha256': digest(args.reuse_score_report/'report.json'),
                                 'note': 'Exact recorded cache hashes verified; no current-code rescoring claimed'}
                                if args.reuse_score_report else None),
        "AUROC_note": "One peak signed calibrated margin per test episode; calibration excluded. Overall AUROC is the equal-task macro mean, never pooled raw scores.",
        "fusion_note": "Same-frame min/max signed standardized margins implement AND/OR. Component alpha is unchanged: OR has no alpha-level familywise guarantee.",
        "time_varying_note": "Elapsed-wall-time bin maxima calibrated across available success episodes. Binwise comparison only, no simultaneous episode-level coverage claim; unsupported bins use +inf.",
        "references": {"FIPER": "https://arxiv.org/html/2510.09459v1", "Ward": "https://arxiv.org/html/2603.06987v1"},
        "todos": ["Inspector score overlays"]}
    write_json(report_dir / 'report.json', report)
    write_inspection_reports(report_dir, report, tables, reports['distributions'])
    # Pure post-processing; independent of cache fingerprints and inference.
    # Write alongside existing reports so the immutable archive includes them.
    from benchmark_roc import analyze as analyze_roc, write_reports as write_roc_reports
    write_roc_reports(report_dir, analyze_roc(reports['episodes']), {
        'created_utc': utc_now(), 'labels_sha256': report['labels_sha256'],
        'scoring_settings': settings, 'protocols': report['protocols'],
        'far_targets': [0., .05, .10, .20],
        'score_source': 'episode_results.csv',
        'score_source_sha256': digest(report_dir / 'episode_results.csv'),
        'analysis_script_sha256': digest(Path(__file__).with_name('benchmark_roc.py')),
    }, report_stem='roc_report')
    from results_inspector import archive_result
    archived = archive_result(report_dir, 'offline')
    print(f"Reported {len(metrics)} metric rows across {len(schemes)} calibration schemes.")
    print(f"Reports: {report_dir}")
    print(f"Empirical ROC and matched-FAR comparison: {report_dir / 'roc_report.html'}")
    print(f"Preserved result archive: {archived}")
    print("Latency uses its separately calibrated causal protocol. Check annotation coverage and missed/pre-onset alarm counts.")


if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--worker-spec":
            run_sampled_worker(sys.argv[2])
        else:
            main()
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        raise SystemExit(f"Benchmark stopped: {error}") from error
