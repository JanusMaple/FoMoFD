#!/usr/bin/env python3
"""CPU-only, reproducible descriptive analyses for FoMo-FD experiments.

First module: rollout-timing, for labeled successful policy rollouts.
  python3 lerobot/utils/analyze_experiments.py rollout-timing
  python3 lerobot/utils/analyze_experiments.py rollout-timing --tasks needle_pickup

Reads only label/preparation metadata and the timestamps.npy member of selected
raw NPZ files. Never imports Torch, loads images/features/models, or modifies data.
Uses recorded wall time for simulation AND real tasks; no nominal-FPS fallback.
Reports calibration/test cohorts separately, plus a descriptive all-success pool
within each task. Recording duration is NOT an annotated task-completion time.

Outputs CSVs, a Markdown report, PNG/PDF figures, and input provenance under
ignored logs/analysis/. Existing report directories are never overwritten.
Dependencies: numpy, PyYAML; matplotlib unless --no-plots is specified.

Second module: score-distributions, comparing cached expert-validation and
successful-policy WM-NC scores at window and episode-maximum levels. Use
--config and --output to create a separate experiment under tasks/benchmark_results.
This script does not select or calibrate detection thresholds.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
TASKS = {
    "needle_pickup": "Needle Pickup (Sim.)",
    "ring_over_post": "Ring Over Post (Sim.)",
    "tissue_retraction": "Tissue Retraction (Real)",
    "shunt_insertion": "Shunt Insertion (Real)",
}
COHORTS = ("calibration", "test", "all_success")
COLORS = {"calibration": "#2563a6", "test": "#d16a24"}
SOURCES = {
    "FAIL-Detect": "https://arxiv.org/html/2503.08558v2#A1.SS2",
    "FIPER": "https://arxiv.org/html/2510.09459v1#A4",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def label_index(node: dict) -> dict[int, dict]:
    """Validate the entire labeling partition, then select successes elsewhere."""
    labels = {}
    for group in node["groups"]:
        first, last = group["episode_range"]
        if type(first) is not int or type(last) is not int or not 0 <= first <= last:
            raise ValueError("Invalid inclusive episode range")
        if group["split"] not in {"calibration", "test"} or group["outcome"] not in {"success", "failure"}:
            raise ValueError("Unsupported split/outcome")
        for episode in range(first, last + 1):
            if episode in labels:
                raise ValueError(f"Overlapping episode labels: {episode}")
            labels[episode] = group
    if len(labels) != node["expected_episode_count"]:
        raise ValueError("Labeled episode count differs from expected_episode_count")
    return labels


def read_wall_timestamps(path: Path) -> tuple[np.ndarray, dict]:
    before = path.stat()
    with zipfile.ZipFile(path) as archive:
        if archive.namelist().count("timestamps.npy") != 1:
            raise ValueError(f"Expected exactly one timestamps.npy member: {path}")
        payload = archive.read("timestamps.npy")
    timestamps = np.asarray(np.load(io.BytesIO(payload), allow_pickle=False), dtype=np.float64)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"Recording changed while being read: {path}")
    if timestamps.ndim != 1 or len(timestamps) < 2 or not np.isfinite(timestamps).all():
        raise ValueError(f"Timestamps must be a finite 1D array with >=2 entries: {path}")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"Non-increasing wall timestamps; no FPS substitution is allowed: {path}")
    return timestamps, {"path": str(path), "file_bytes": before.st_size,
                        "mtime_ns": before.st_mtime_ns, "member": "timestamps.npy",
                        "member_sha256": sha256(payload)}


def describe(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Statistics require a nonempty finite vector")
    q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    return {"mean": mean, "std": std, "variance": None if std is None else std**2,
            "median": float(median), "q25": float(q25), "q75": float(q75),
            "iqr": float(q75 - q25), "p05": float(q05), "p95": float(q95),
            "min": float(values.min()), "max": float(values.max()),
            "cv_pct": 100 * std / mean if std is not None and mean > 0 else None,
            "max_min_ratio": float(values.max() / values.min()) if values.min() > 0 else None}


def bootstrap_cv(values, *, replicates: int, seed: int, key: str) -> tuple[float | None, float | None]:
    """Percentile interval, resampling whole episodes; descriptive, not a test."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return None, None
    stable_key = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, stable_key]))
    cvs = []
    for start in range(0, replicates, 256):
        sample = rng.choice(values, size=(min(256, replicates - start), len(values)), replace=True)
        cvs.extend(100 * sample.std(axis=1, ddof=1) / sample.mean(axis=1))
    return tuple(float(x) for x in np.quantile(cvs, [0.025, 0.975]))


def load_successes(repo: Path, labels: dict, tasks: list[str], prepared_name: str):
    episodes, timings, provenance = [], {}, []
    for task in tasks:
        base = repo / "fomofd/tasks" / task / "prl_data"
        preparation_path = base / prepared_name / "meta/preparation.json"
        payload = preparation_path.read_bytes()
        preparation = json.loads(payload)
        if preparation.get("source_timestamp") != "raw wall clock (simulation included)":
            raise ValueError(f"Unrecognized timestamp provenance: {preparation_path}")
        shift = preparation["action_shift"]
        if type(shift) is not int or shift not in (0, 1):
            raise ValueError("Only contiguous recordings with zero/one omitted terminal frame are supported")
        mapping = {}
        for row in preparation["episode_mapping"]:
            episode = row["dataset_episode_index"]
            if type(episode) is not int or episode < 0 or episode in mapping:
                raise ValueError(f"Invalid/duplicate prepared episode ID: {preparation_path}")
            mapping[episode] = row
        index = label_index(labels["tasks"][task])
        if set(mapping) != set(index):
            raise ValueError(f"Prepared and labeled episode IDs differ: {task}")
        task_source = {"task": task, "preparation_path": str(preparation_path),
                       "preparation_sha256": sha256(payload), "recordings": []}
        for episode, group in sorted(index.items()):
            if group["outcome"] != "success":
                continue
            row = mapping[episode]
            filename = row["raw_file"]
            if not filename or Path(filename).name != filename:
                raise ValueError(f"Invalid raw filename: {filename}")
            timestamps, source = read_wall_timestamps(base / "raw" / filename)
            n = len(timestamps)
            if n != row["raw_frames"] or row["prepared_frames"] != n - shift or n - shift < 2:
                raise ValueError(f"Frame count/alignment differs from preparation: {task}/{episode}")
            elapsed = timestamps - timestamps[0]
            if not math.isclose(float(elapsed[-1]), row["wall_duration_seconds"], abs_tol=1e-6, rel_tol=1e-9):
                raise ValueError(f"Wall duration differs from preparation: {task}/{episode}")
            intervals = np.diff(timestamps)
            median_interval = float(np.median(intervals))
            record = {"task": task, "split": group["split"], "episode_index": episode,
                      "raw_episode_id": row["raw_episode_id"], "raw_file": filename,
                      "raw_frames": n, "prepared_frames": n - shift,
                      "wall_duration_s": float(elapsed[-1]),
                      "prepared_wall_duration_s": float(elapsed[n - shift - 1]),
                      "start_unix_s": float(timestamps[0]), "end_unix_s": float(timestamps[-1]),
                      "mean_interval_s": float(intervals.mean()), "median_interval_s": median_interval,
                      "p05_interval_s": float(np.quantile(intervals, 0.05)),
                      "p95_interval_s": float(np.quantile(intervals, 0.95)),
                      "max_interval_s": float(intervals.max()),
                      "interval_cv_pct": describe(intervals)["cv_pct"],
                      "gaps_over_2x_episode_median": int(np.sum(intervals > 2 * median_interval)),
                      "effective_recorded_fps": float((n - 1) / elapsed[-1])}
            episodes.append(record)
            timings[task, episode] = elapsed
            task_source["recordings"].append({"episode_index": episode, **source})
        for cohort in COHORTS[:2]:
            if not any(r["task"] == task and r["split"] == cohort for r in episodes):
                raise ValueError(f"No labeled successes for {task}/{cohort}")
        provenance.append(task_source)
    return episodes, timings, provenance


def recording_support(endpoints, grid) -> np.ndarray:
    """Recorded interval is closed: a sample exactly at the endpoint is present."""
    ends = np.sort(np.asarray(endpoints))
    return len(ends) - np.searchsorted(ends, grid, side="left")


def summarize(episodes, timings, tasks, replicates, seed):
    summaries, support, alignment, comparisons = [], [], [], []
    for task in tasks:
        task_rows = [r for r in episodes if r["task"] == task]
        # Common grids across cohorts. Include exact ends and a point after each
        # end so support counts have unambiguous endpoint behavior in CSV/plots.
        grids = {}
        for clock in ("wall", "prepared_wall"):
            field = "wall_duration_s" if clock == "wall" else "prepared_wall_duration_s"
            ends = np.array([r[field] for r in task_rows])
            grids[clock] = np.unique(np.r_[0., ends, np.nextafter(ends, np.inf)])
        for clock, field in (("raw_frame", "raw_frames"), ("prepared_frame", "prepared_frames")):
            grids[clock] = np.arange(max(r[field] for r in task_rows) + 1)
        for cohort in COHORTS:
            rows = [r for r in task_rows if cohort == "all_success" or r["split"] == cohort]
            summary = {"task": task, "cohort": cohort, "n": len(rows)}
            for field in ("wall_duration_s", "prepared_wall_duration_s", "raw_frames", "median_interval_s"):
                summary.update({f"{field}_{k}": v for k, v in describe([r[field] for r in rows]).items()})
            low, high = bootstrap_cv([r["wall_duration_s"] for r in rows], replicates=replicates,
                                     seed=seed, key=f"{task}/{cohort}")
            summary.update(wall_duration_cv_ci95_low_pct=low, wall_duration_cv_ci95_high_pct=high)
            common_frame = min(r["raw_frames"] for r in rows) - 1
            common_times = [timings[task, r["episode_index"]][common_frame] for r in rows]
            summary.update(common_last_raw_frame=common_frame,
                           common_frame_elapsed_cv_pct=describe(common_times)["cv_pct"],
                           common_frame_elapsed_min_s=min(common_times), common_frame_elapsed_max_s=max(common_times))
            summaries.append(summary)
            for clock, grid in grids.items():
                field = {"wall": "wall_duration_s", "prepared_wall": "prepared_wall_duration_s",
                         "raw_frame": "raw_frames", "prepared_frame": "prepared_frames"}[clock]
                ends = [r[field] - int(clock.endswith("frame")) for r in rows]
                for x, count in zip(grid, recording_support(ends, grid)):
                    support.append({"task": task, "cohort": cohort, "clock": clock,
                                    "coordinate": float(x), "recordings_present": int(count),
                                    "total_recordings": len(rows), "fraction_present": float(count / len(rows))})
            for frame in range(max(r["raw_frames"] for r in rows)):
                values = [timings[task, r["episode_index"]][frame] for r in rows if r["raw_frames"] > frame]
                stats = describe(values)
                alignment.append({"task": task, "cohort": cohort, "raw_frame_index": frame,
                                  "recordings_present": len(values), "total_recordings": len(rows),
                                  "elapsed_median_s": stats["median"], "elapsed_p05_s": stats["p05"],
                                  "elapsed_p95_s": stats["p95"], "elapsed_min_s": stats["min"],
                                  "elapsed_max_s": stats["max"], "elapsed_std_s": stats["std"]})
        calibration = [r for r in task_rows if r["split"] == "calibration"]
        test = [r for r in task_rows if r["split"] == "test"]
        cmin, cmax = min(r["wall_duration_s"] for r in calibration), max(r["wall_duration_s"] for r in calibration)
        comparisons.append({"task": task, "n_calibration": len(calibration), "n_test": len(test),
                            "calibration_min_duration_s": cmin, "calibration_max_duration_s": cmax,
                            "test_minus_calibration_mean_s": float(np.mean([r["wall_duration_s"] for r in test]) - np.mean([r["wall_duration_s"] for r in calibration])),
                            "test_longer_than_max_calibration": sum(r["wall_duration_s"] > cmax for r in test),
                            "test_shorter_than_min_calibration": sum(r["wall_duration_s"] < cmin for r in test),
                            "test_frames_beyond_max_calibration": sum(r["raw_frames"] > max(c["raw_frames"] for c in calibration) for r in test)})
    return summaries, support, alignment, comparisons


def plot_timing(output, episodes, support, tasks):
    # All caches and rendering are CPU-only and scoped to the output directory.
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(math.ceil(len(tasks) / 2), 2, figsize=(10, 3.8 * math.ceil(len(tasks) / 2)), squeeze=False)
    for axis, task in zip(axes.flat, tasks):
        for pos, cohort in enumerate(COHORTS[:2], 1):
            values = [r["wall_duration_s"] for r in episodes if r["task"] == task and r["split"] == cohort]
            axis.boxplot([values], positions=[pos], widths=0.45, showfliers=False,
                         medianprops={"color": "black"}, patch_artist=True,
                         boxprops={"facecolor": COLORS[cohort], "alpha": 0.25})
            # Deterministic horizontal spread; every episode remains visible.
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(values))
            axis.scatter(pos + jitter, values, color=COLORS[cohort], s=20, alpha=0.8)
        counts = [sum(r["task"] == task and r["split"] == c for r in episodes) for c in COHORTS[:2]]
        axis.set_xticks([1, 2], [f"Calibration (n={counts[0]})", f"Test success (n={counts[1]})"])
        axis.set_title(TASKS[task]); axis.set_ylabel("Recorded wall duration (s)")
        axis.grid(axis="y", alpha=0.2)
    for axis in list(axes.flat)[len(tasks):]:
        axis.set_visible(False)
    fig.suptitle("Successful policy rollouts: recorded duration", fontsize=14)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"rollout_durations.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(tasks), 2, figsize=(11, 2.8 * len(tasks)), squeeze=False)
    for row, task in enumerate(tasks):
        for col, clock in enumerate(("wall", "raw_frame")):
            axis = axes[row, col]
            for cohort in COHORTS[:2]:
                values = [r for r in support if r["task"] == task and r["cohort"] == cohort and r["clock"] == clock]
                axis.step([r["coordinate"] for r in values], [100 * r["fraction_present"] for r in values],
                          where="post", color=COLORS[cohort], label=cohort.capitalize())
            axis.set_title(TASKS[task]); axis.set_ylim(-3, 103); axis.set_ylabel("Recordings present (%)")
            axis.set_xlabel("Elapsed recorded wall time (s)" if clock == "wall" else "Raw observation frame index")
            axis.grid(alpha=0.2)
            if row == 0:
                axis.legend()
    fig.suptitle("Recording support across time and frame index (not a conformal coverage estimate)", fontsize=12)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"recording_support.{extension}", dpi=180)
    plt.close(fig)


def markdown_report(summaries, comparisons, episodes, replicates):
    lines = ["# Successful policy rollout timing", "",
             "Recorded wall time is used for both simulation and real tasks. Duration is last timestamp minus first timestamp; no frames/FPS estimate is substituted.", "",
             "Only labeled successes are included. Calibration and test remain separate; all_success is a descriptive pool within each task. No detector was trained, scored, or calibrated.", "",
             "| Task | Cohort | n | Mean ± sample SD (s) | Median [Q1, Q3] (s) | Min–max (s) | Duration CV (%) | Frame-count CV (%) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "NA" if value is None else f"{value:.2f}"
    for s in summaries:
        prefix = "wall_duration_s_"
        lines.append(f"| {TASKS[s['task']]} | {s['cohort']} | {s['n']} | {fmt(s[prefix+'mean'])} ± {fmt(s[prefix+'std'])} | "
                     f"{fmt(s[prefix+'median'])} [{fmt(s[prefix+'q25'])}, {fmt(s[prefix+'q75'])}] | "
                     f"{fmt(s[prefix+'min'])}–{fmt(s[prefix+'max'])} | {fmt(s[prefix+'cv_pct'])} | {fmt(s['raw_frames_cv_pct'])} |")
    lines += ["", "CV = 100 × sample SD / mean; sample variance uses n−1. summary.csv also includes variance, IQR, percentiles, prepared spans, sampling intervals, and clock spread at the last raw frame shared by every episode in the cohort.", "",
              f"Duration-CV 95% percentile bootstrap intervals use {replicates} resamples of whole episodes, never individual frames. These are descriptive uncertainty estimates assuming independent episodes, not hypothesis tests or conformal guarantees. The pooled cohort may mix collection conditions.", "",
              "## Calibration/test timing support", "",
              "These counts describe recording availability; they do not estimate FAR or conformal coverage. Test data is not used to select thresholds or analysis cutoffs.", "",
              "| Task | Test successes ending after longest calibration | Test successes shorter than shortest calibration | Test successes with more frames than any calibration |",
              "|---|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {TASKS[r['task']]} | {r['test_longer_than_max_calibration']}/{r['n_test']} | "
                     f"{r['test_shorter_than_min_calibration']}/{r['n_test']} | {r['test_frames_beyond_max_calibration']}/{r['n_test']} |")
    lines += ["", "## Interpretation and limits", "",
              "- These measurements establish variation in recorded duration, observation count, and sampling cadence. They do not measure semantic task progress, retry times, or exact task-completion times. Manual start/stop delays and terminal dwell may contribute.",
              "- A shared observation index and a shared elapsed wall time are different alignments. recording_support.csv includes both raw/prepared spans; frame_alignment.csv measures wall-clock spread at the same raw frame index, without interpolation, padding, or normalization by future episode duration.",
              "- A time-varying threshold is a calibration choice, separate from the logpZO score model. FAIL-Detect fits a time-indexed band with separate subsets for band shape and calibration. Its construction does not impose identical semantic task progress at each index. [FAIL-Detect, Appendix B](" + SOURCES["FAIL-Detect"] + ")",
              "- Timing variability can reduce a time-indexed band's efficiency; it does not by itself disprove conformal validity or show that a constant threshold detects failures better. Validity still depends on exchangeability and the calibration/horizon construction.",
              "- FIPER discusses temporal shifts and sparse support late in variable-length rollouts as practical limitations of time-varying thresholds. [FIPER, Appendix D](" + SOURCES["FIPER"] + ")",
              "- With 19 successful calibration episodes per task, fitting band shape and calibrating its width on disjoint subsets further reduces calibration support. A future implementation must state its finite-sample quantile, horizon, and padding/masking rules; naive per-time quantiles do not automatically provide episode-level FAR control.",
              "- For a reviewer comparison, reuse the same cached logpZO scores, fit/calibrate both threshold rules without test data, and evaluate held-out FAR/FDR and onset delay. No logpZO retraining is required just to change the threshold. Success milestone annotations would directly test the proposed progress-alignment explanation.", "",
              "## Files", "",
              "- episodes.csv: per-success-episode duration, frame counts, and cadence diagnostics.",
              "- summary.csv: per-task, per-cohort descriptive statistics and CV intervals.",
              "- calibration_test_comparison.csv: duration/length support comparisons.",
              "- recording_support.csv: numbers of recordings present at each wall-time/frame coordinate. Terminal samples are included; no values are fabricated past episode ends.",
              "- frame_alignment.csv: wall-time distribution at each raw frame, with the number contributing.",
              "- rollout_durations.png/.pdf and recording_support.png/.pdf: shareable figures, unless plots disabled.",
              "- provenance.json and labels_snapshot.yaml: exact timestamp-member hashes, preparation hashes, label snapshot, arguments, versions, and metric definitions.", "",
              f"Analyzed {len(episodes)} successful recordings. Failures and expert demonstrations were not read by this timing analysis. Use the separate score-distributions subcommand for expert/policy scores.", ""]
    return "\n".join(lines)


def rollout_timing(args):
    repo = args.repo_root.resolve()
    labels_path = args.labels or repo / "lerobot/configs/benchmark/prl_episode_labels.yaml"
    labels_payload = labels_path.read_bytes()
    labels = yaml.safe_load(labels_payload)
    if labels.get("schema_version") != 2 or labels.get("episode_id_space") != "prepared_prl_episode_index":
        raise ValueError("Expected schema 2 labels in prepared PRL episode-index space")
    if len(set(args.tasks)) != len(args.tasks) or args.seed < 0:
        raise ValueError("Tasks must be unique and seed nonnegative")
    if Path(args.prepared_name).name != args.prepared_name or args.prepared_name in (".", ".."):
        raise ValueError("--prepared-name must be a single directory name")
    now = datetime.now(timezone.utc)
    output = (args.output or repo / "logs/analysis" / now.strftime("%Y-%m-%d") /
              now.strftime("rollout_timing_%H%M%S_%fZ")).resolve()
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    episodes, timings, sources = load_successes(repo, labels, args.tasks, args.prepared_name)
    summaries, support, alignment, comparisons = summarize(episodes, timings, args.tasks, args.bootstrap_replicates, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage all outputs so failed analysis/plotting leaves no partial final report.
    with tempfile.TemporaryDirectory(prefix=".rollout_timing_", dir=output.parent) as staging:
        stage = Path(staging)
        for name, rows in (("episodes", episodes), ("summary", summaries), ("recording_support", support),
                           ("frame_alignment", alignment), ("calibration_test_comparison", comparisons)):
            write_csv(stage / f"{name}.csv", rows)
        (stage / "labels_snapshot.yaml").write_bytes(labels_payload)
        (stage / "report.md").write_text(markdown_report(summaries, comparisons, episodes, args.bootstrap_replicates))
        if not args.no_plots:
            plot_timing(stage, episodes, support, args.tasks)
        write_json(stage / "provenance.json", {
            "schema_version": 1, "analysis": "rollout-timing", "created_utc": now.isoformat(),
            "script_sha256": sha256(Path(__file__).read_bytes()), "repo_root": str(repo),
            "labels_path": str(labels_path.resolve()), "labels_sha256": sha256(labels_payload),
            "tasks": args.tasks, "prepared_name": args.prepared_name, "seed": args.seed,
            "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_unit": "episode",
            "no_plots": args.no_plots, "numpy_version": np.__version__, "yaml_version": yaml.__version__,
            "definitions": {"duration": "last minus first raw recorded wall timestamp",
                            "prepared_duration": "last retained minus first raw wall timestamp",
                            "variance": "sample variance (ddof=1)", "cv_pct": "100 * sample std / mean",
                            "support": "recording interval includes its terminal sample; no padding",
                            "all_success": "descriptive within-task pool of calibration and test successes"},
            "sources": sources, "method_references": SOURCES})
        # mkdir is exclusive: concurrent analyses can never overwrite this run.
        output.mkdir(exist_ok=False)
        for item in stage.iterdir():
            item.rename(output / item.name)
    print("Task                  n   Mean ± SD (wall s)     Range (s)       CV%")
    for s in summaries:
        if s["cohort"] == "all_success":
            print(f"{s['task']:20} {s['n']:3}   {s['wall_duration_s_mean']:6.2f} ± {s['wall_duration_s_std']:5.2f}     "
                  f"{s['wall_duration_s_min']:5.2f}–{s['wall_duration_s_max']:5.2f}     {s['wall_duration_s_cv_pct']:5.2f}")
    print(f"Reports: {output}")
    print("Descriptive timing only; no scores, models, GPU use, or threshold calibration.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="analysis", required=True)
    timing = commands.add_parser("rollout-timing", help="Successful rollout wall durations, frame counts, cadence, and recording support")
    timing.add_argument("--repo-root", type=Path, default=REPO)
    timing.add_argument("--labels", type=Path)
    timing.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    timing.add_argument("--prepared-name", default="processed_le_v21_vfmf_vae_fix_wrist")
    timing.add_argument("--output", type=Path, help="New directory; default logs/analysis/YYYY-MM-DD/rollout_timing_<UTC>")
    timing.add_argument("--bootstrap-replicates", type=positive_int, default=2000)
    timing.add_argument("--seed", type=int, default=0)
    timing.add_argument("--no-plots", action="store_true")
    timing.set_defaults(func=rollout_timing)
    scores = commands.add_parser('score-distributions', help='Cached expert-validation versus policy-success WM-NC score distributions')
    scores.add_argument('--config', type=Path, default=REPO/'lerobot/configs/benchmark/expert_policy_score_distributions.yaml')
    scores.add_argument('--output', type=Path, required=True, help='Fresh experiment directory under tasks/benchmark_results/analyses')
    scores.add_argument('--refresh-plots', action='store_true', help='Update HTML/Markdown and density plots from an existing experiment\'s exported scores only')
    def score_distributions(args):
        from expert_policy_scores import run
        run(args)
    scores.set_defaults(func=score_distributions)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
