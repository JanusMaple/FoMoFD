#!/usr/bin/env python3
"""Failure Example Plot Selector: four score plots with two camera frames each.

From the repository root, using the lerobot environment::

    python lerobot/utils/failure_example_plot_selector.py
    python lerobot/utils/plot_failure_examples.py --port 8767 --no-browser
    python lerobot/utils/plot_failure_examples.py --export --png
    python lerobot/utils/failure_example_plot_selector.py --export --out logs/figures/figure4.pdf
    python lerobot/utils/plot_failure_examples.py \
        --config logs/figures/figure4.selection.json --export --out /tmp/figure4.svg

Select tasks, failure modes, prepared episode IDs, camera view, and two scored
raw frames in the local browser. Click a curve to position the active marker.
Starts with configs/figures/failure_example.yaml; Save defaults keeps new choices
there for the next launch. plot_failure_examples.py remains a supported launcher.
Exports SVG or PDF with vector curves/text and embedded original RGB images,
plus a review copy and a JSON selection recording provenance. Both copies use
7 pt Nimbus Roman labels, bold task/failure titles, and compact score plots at
the configured double-column width. One-line titles use the inter-panel gaps
to maximize their shared font size up to 7 pt; two-line titles also use 7 pt.
Onset lines default to the
three-annotator mean recorded time for the selected camera; individual labels
remain selectable. The current evaluation protocol supplies the annotator roster.
The seconds axis uses the exact mean; raw-frame axes use the first frame at/after
that mean, as in the benchmark's evidence rule. No GPU, inference,
HF conversion, label edits, score smoothing, or threshold refitting is performed.
Uses the main archived report's paper protocol and saved episode threshold.
Style constants follow replay_score_paper.py (without importing its HF stack).
Dependencies: numpy, PyYAML, Pillow, matplotlib; all in the lerobot environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
import tempfile
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
import webbrowser

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image
import yaml

from benchmark_consensus_onsets import REFERENCE, mean_annotation
from paper_plot_style import FONT_SIZE, RC_PARAMS, apply_font_files, font_metadata
from replay_inspector import Recordings, onset_views, score_report_path
from results_inspector import resolve_repo_path, sha256
from unified_benchmark import curves

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "lerobot/configs/figures/failure_example.yaml"
LABELS = REPO / "lerobot/configs/benchmark/prl_episode_labels.yaml"
TAXONOMY = REPO / "lerobot/configs/benchmark/failure_categories.yaml"
RESULTS_CATALOG = REPO / "lerobot/configs/benchmark/results_catalog.yaml"
TASK_TITLES = {"needle_pickup": "Needle Pickup", "ring_over_post": "Ring Over Post",
               "tissue_retraction": "Tissue Retraction", "shunt_insertion": "Shunt Insertion"}
CATEGORY_NUMBERS = {"grasp": "I", "spatial": "II", "scene": "III", "sensing_actuation": "IV"}
SUCCESS_COLOR, FAIL_COLOR, FAIL_ABOVE_COLOR, THRESHOLD_COLOR = "#1f77b4", "#d62728", "#8b0000", "0.25"
SUCCESS_LS, THRESHOLD_LS = (0, (4.0, 2.2)), "-."
FAIL_LW, FAIL_ABOVE_LW = .6, .8
ONSET_COLOR, ONSET_LS = "0.45", (0, (2.5, 2.5))


def read_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Figure selection must be a JSON/YAML object.")
    return config


def integer(value, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer.")
    return value


def split_curve(x, y, threshold):
    """Insert exact crossings so the thick above-threshold segment meets the line."""
    xx, yy = [float(x[0])], [float(y[0])]
    for x0, x1, y0, y1 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if (y0 - threshold) * (y1 - threshold) < 0:
            xx.append(float(x0 + (x1 - x0) * (threshold - y0) / (y1 - y0)))
            yy.append(float(threshold))
        xx.append(float(x1))
        yy.append(float(y1))
    xx, yy = np.asarray(xx), np.asarray(yy)
    return xx, np.where(yy <= threshold, yy, np.nan), np.where(yy >= threshold, yy, np.nan)


class FigureData:
    """Read-only recordings and content-verified, per-task/view score caches."""

    def __init__(self, report: Path, report_reference: str | None = None,
                 onset_protocol: Path | None = None):
        self.report_dir = report.resolve()
        self.report_reference = report_reference or str(self.report_dir)
        self.report_file = self.report_dir / "report.json"
        self.report = json.loads(self.report_file.read_text())
        self.report_hash = sha256(self.report_file)
        self.protocol = self.report["protocols"]["paper_protocol"]
        if self.protocol.get("window") != 1 or self.report.get("wm_score_aggregation", "per_view") != "per_view":
            raise ValueError("Choose an unsmoothed, per-view benchmark report (paper window=1).")
        self.recordings = Recordings(REPO, LABELS)
        self.manifest = self.recordings.labels.read()
        self.labels_hash = sha256(LABELS)
        self.onset_protocol_file = (onset_protocol or resolve_repo_path(
            read_config(RESULTS_CATALOG)["evaluation_protocol"])).resolve()
        self.onset_protocol = read_config(self.onset_protocol_file)
        self.onset_protocol_hash = sha256(self.onset_protocol_file)
        self.onset_rule = self.onset_protocol["detection"]["onset_reference"]
        self.consensus_annotators = self.onset_rule["annotator_ids"]
        if (self.onset_rule["kind"] != "mean_wall_time"
                or len(self.consensus_annotators) != self.onset_rule["required_annotations_per_view"]
                or len(set(self.consensus_annotators)) != len(self.consensus_annotators)
                or not self.consensus_annotators
                or set(self.consensus_annotators) - set(self.manifest["annotators"])):
            raise ValueError("Mean-onset protocol annotator roster does not match labels.")
        self.consensus_name = f"Mean onset ({len(self.consensus_annotators)} annotators)"
        taxonomy = read_config(TAXONOMY)
        self.categories = {}
        for key, category in taxonomy["categories"].items():
            for task, modes in category["modes"].items():
                for mode in modes:
                    self.categories[task, mode] = CATEGORY_NUMBERS[key]
        self.lock = threading.RLock()

    def catalog(self) -> dict:
        tasks = []
        for task in self.recordings.catalog["tasks"]:
            groups = [{"mode": g["failure_mode"], "episodes": g["episode_ids"],
                       "category": self.categories[task["id"], g["failure_mode"]]}
                      for g in task["groups"] if g["outcome"] == "failure"]
            tasks.append({"id": task["id"], "title": TASK_TITLES[task["id"]], "groups": groups})
        annotators = [{"id": REFERENCE, "name": self.consensus_name, "derived": True}]
        annotators += [{"id": key, "name": value["name"]}
                       for key, value in self.manifest.get("annotators", {}).items() if key != REFERENCE]
        return {"tasks": tasks, "report": str(self.report_dir), "annotators": annotators}

    @lru_cache(maxsize=8)
    def score_data(self, task: str, view: str):
        entries = [c for c in self.report["score_caches"]
                   if c["task"] == task and c["method"] == "wm_nc" and view in c["views"]]
        if len(entries) != 1:
            raise ValueError(f"Expected one WM-NC {view} score cache for {task}.")
        entry = entries[0]
        path = resolve_repo_path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Score cache failed SHA-256 verification: {path}")
        with np.load(path, allow_pickle=False) as source:
            arrays = {k: source[k] for k in ("episode_index", "frame_index", "raw_frame_index", "wall_timestamp", "scores")}
        settings = entry["scoring_settings"]
        job = SimpleNamespace(task=task, mode=True,
                              merged={"data": {"history_len": settings["L"], "pred_horizon": settings["K"]}})
        series = curves(arrays, job, entry["views"].index(view), self.protocol)
        records = [r for r in self.report["calibration"]
                   if r["task"] == task and r["method"] == "wm_nc" and r["view"] == view
                   and r["calibration"] == "episode" and r["protocol"] == "paper"]
        if len(records) != 1 or not np.isfinite(float(records[0]["threshold"])):
            raise ValueError(f"Missing finite saved episode threshold: {task}/{view}.")
        calibration_ids = [ep for g in self.manifest["tasks"][task]["groups"]
                           if g["split"] == "calibration" and g["outcome"] == "success"
                           for ep in range(g["episode_range"][0], g["episode_range"][1] + 1)]
        if len(calibration_ids) != records[0]["n_calibration"] or any(ep not in series for ep in calibration_ids):
            raise ValueError(f"Calibration labels differ from the saved report: {task}/{view}.")
        for curve in series.values():
            if (not len(curve["raw"]) or not np.isfinite(curve["score"]).all()
                    or not np.isfinite(curve["wall"]).all() or (np.diff(curve["raw"]) <= 0).any()):
                raise ValueError(f"Invalid score endpoints: {task}/{view}.")
        return series, records[0], calibration_ids, entry

    def episode(self, task: str, ep: int, view: str) -> dict:
        if view not in {"fix", "wrist"} or (task, ep) not in self.recordings.paths:
            raise ValueError("Select a known task, episode, and camera view.")
        series, record, calibration_ids, _ = self.score_data(task, view)
        curve = series[ep]
        with self.lock:
            recording = self.recordings.episode(task, ep)
            raw = curve["raw"]
            if raw[0] < 0 or raw[-1] >= len(recording.times):
                raise ValueError("Score endpoint is outside the raw recording.")
            # Check absolute timestamps as well as raw indices. In particular,
            # a shifted episode with an identical cadence must still be rejected.
            times = self.score_times(task, ep, view)
            onsets, onset_errors = self.episode_onsets(task, ep, view)
        # Each calibration rollout has its own clock; never align unrelated
        # absolute timestamps across episodes. Relative score times are shifted
        # using the exact recording's timestamp zero, cached once per episode.
        calibration = []
        for calibration_ep in calibration_ids:
            c = series[calibration_ep]
            calibration.append({"frames": c["raw"].tolist(), "times": self.score_times(task, calibration_ep, view),
                                "scores": c["score"].tolist()})
        return {"raw_frames": raw.tolist(), "times": times, "scores": curve["score"].tolist(),
                "threshold": float(record["threshold"]), "n_calibration": record["n_calibration"],
                "calibration": calibration, "raw_file": recording.raw_file, "onsets": onsets,
                "onset_errors": onset_errors}

    def raw_timestamps(self, task: str, ep: int) -> np.ndarray:
        # Read only the recorded clock, avoiding decompression of camera images.
        with np.load(self.recordings.paths[task, ep], allow_pickle=False) as source:
            wall = np.asarray(source["timestamps"], dtype=np.float64).reshape(-1)
        if not len(wall) or not np.isfinite(wall).all() or (np.diff(wall) < 0).any():
            raise ValueError(f"Invalid recorded timestamps: {task}/{ep}.")
        return wall

    def episode_onsets(self, task: str, ep: int, view: str) -> tuple[dict, dict]:
        """Derive only this camera's mean using the benchmark's exact time rule."""
        wall = self.raw_timestamps(task, ep)
        onsets, errors = {}, {}
        annotations = self.manifest["tasks"][task].get("failure_onsets", {}).get(ep, {})
        for annotator, entry in annotations.items():
            if annotator == REFERENCE:
                continue  # Always derive from the protocol's individual sources.
            annotation = onset_views(entry).get(view)
            if annotation is not None:
                frame = integer(annotation["frame_index"], "Onset frame")
                if not 0 <= frame < len(wall):
                    raise ValueError(f"Onset exceeds recording: {task}/{ep}/{annotator}/{view}.")
                onsets[annotator] = {"raw_frame": frame, "time": float(wall[frame] - wall[0]),
                                     "annotation": annotation}
        missing = [aid for aid in self.consensus_annotators if aid not in onsets]
        if missing:
            names = ", ".join(self.manifest["annotators"][aid]["name"] for aid in missing)
            errors[REFERENCE] = (f"{task}/{ep}/{view}: {self.consensus_name} requires every contributor; "
                                 f"missing onset from {names}. Select an individual or hide the onset line.")
        else:
            sources = {aid: onsets[aid]["annotation"] for aid in self.consensus_annotators}
            annotation = mean_annotation(sources, wall)
            onsets[REFERENCE] = {"raw_frame": annotation["frame_index"],
                "time": annotation["elapsed_seconds"], "annotation": annotation, "derived": True,
                "contributors": {aid: {"name": self.manifest["annotators"][aid]["name"], **onsets[aid]}
                                 for aid in self.consensus_annotators}}
        return onsets, errors

    @lru_cache(maxsize=256)
    def score_times(self, task: str, ep: int, view: str) -> list:
        curve = self.score_data(task, view)[0][ep]
        timestamps = self.raw_timestamps(task, ep)
        if (curve["raw"][0] < 0 or curve["raw"][-1] >= len(timestamps)
                or not np.allclose(curve["wall"], timestamps[curve["raw"]], rtol=0, atol=1e-5)):
            raise ValueError(f"Invalid or mismatched recorded timestamps: {task}/{ep}.")
        return (timestamps[curve["raw"]] - timestamps[0]).tolist()

    def normalize(self, config: dict) -> dict:
        if not isinstance(config, dict) or config.get("schema_version") != 1:
            raise ValueError("Expected a schema_version: 1 figure selection.")
        recorded_hash = config.get("provenance", {}).get("report_sha256")
        if recorded_hash and recorded_hash != self.report_hash:
            raise ValueError("Saved selection belongs to a different report SHA-256.")
        out = {k: config.get(k, default) for k, default in {
            "schema_version": 1, "view": "wrist", "time_axis": "seconds", "show_ticks": False,
            "show_legend": True, "show_onset": True, "onset_annotator": None, "title_layout": "one_line",
            "transparent": False, "width": 7.16, "height": 2.05}.items()}
        if out["view"] not in {"fix", "wrist"} or out["time_axis"] not in {"seconds", "frames"}:
            raise ValueError("Choose fix/wrist and seconds/frames.")
        if out["title_layout"] not in {"one_line", "two_lines"}:
            raise ValueError("Choose one_line or two_lines for the titles.")
        for flag in ("show_ticks", "show_legend", "show_onset", "transparent"):
            if type(out[flag]) is not bool:
                raise ValueError(f"{flag} must be true or false.")
        annotators = {a["id"] for a in self.catalog()["annotators"]}
        if out["onset_annotator"] is None:
            out["onset_annotator"] = REFERENCE
        if out["onset_annotator"] is not None and out["onset_annotator"] not in annotators:
            raise ValueError("Select the combined mean or a registered onset annotator.")
        for key, limits in (("width", (5, 16)), ("height", (1.8, 8))):
            value = out[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not limits[0] <= value <= limits[1]:
                raise ValueError(f"{key} must be between {limits[0]} and {limits[1]} inches.")
        if not isinstance(config.get("panels"), list) or len(config["panels"]) != 4:
            raise ValueError("Choose exactly four panels.")
        out["report"] = str(self.report_dir)
        out["panels"] = []
        for panel in config["panels"]:
            if not isinstance(panel, dict):
                raise ValueError("Each panel must be an object.")
            task, mode = panel.get("task"), panel.get("failure_mode")
            group = next((g for t in self.catalog()["tasks"] if t["id"] == task
                          for g in t["groups"] if g["mode"] == mode), None)
            if group is None:
                raise ValueError(f"Unknown task/failure mode: {task}/{mode}.")
            ep = panel.get("episode")
            ep = group["episodes"][0] if ep is None else integer(ep, "Episode")
            if ep not in group["episodes"]:
                raise ValueError(f"Episode {ep} is not labeled {mode} for {task}.")
            if out["show_onset"] and out["onset_annotator"] == REFERENCE:
                _, errors = self.episode_onsets(task, ep, out["view"])
                if REFERENCE in errors:
                    raise ValueError(errors[REFERENCE])
            raw = self.score_data(task, out["view"])[0][ep]["raw"]
            marks = panel.get("frames")
            if marks is None:
                peak = int(np.argmax(self.score_data(task, out["view"])[0][ep]["score"]))
                marks = [int(raw[0]), int(raw[peak if peak else -1])]
            if not isinstance(marks, list) or len(marks) != 2:
                raise ValueError("Select exactly two scored raw frames per panel.")
            marks = [integer(f, "Frame") for f in marks]
            if marks[0] >= marks[1] or any(f not in raw for f in marks):
                raise ValueError("Frames must be distinct scored raw endpoints in increasing order.")
            out["panels"].append({"task": task, "failure_mode": mode, "category": group["category"],
                                  "episode": ep, "frames": marks})
        return out

    def frame_png(self, task: str, ep: int, frame: int, view: str) -> bytes:
        if view not in {"fix", "wrist"} or (task, ep) not in self.recordings.paths:
            raise ValueError("Unknown recording or view.")
        with self.lock:
            recording = self.recordings.episode(task, ep)
            if not 0 <= frame < len(recording.times):
                raise ValueError("Frame is outside the raw recording.")
            buffer = io.BytesIO()
            Image.fromarray(getattr(recording, view)[frame]).save(buffer, format="PNG")
            return buffer.getvalue()


def label_next_to_dot(ax, x: float, y: float, label: str, occupied: list) -> None:
    """Choose a nearby corner with the fewest curve crossings; draw no leader."""
    scale = ax.figure.dpi / 72
    px, py = ax.transData.transform((x, y))
    width = (TextPath((0, 0), label, size=FONT_SIZE).get_extents().width + 1.5) * scale
    height = (FONT_SIZE + 2) * scale
    curves = [line.get_transform().transform(line.get_path().vertices) for line in ax.lines]
    candidates = []
    for rank, (dx, dy) in enumerate(((4, 4), (4, -4), (-4, 4), (-4, -4))):
        left = px + dx * scale - (width if dx < 0 else 0)
        bottom = py + dy * scale - (height if dy < 0 else 0)
        box = Bbox.from_bounds(left, bottom, width, height)
        outside = not (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1))
        cost = 100_000 * outside + 10_000 * sum(box.overlaps(b) for b in occupied) + rank * .01
        for points in curves:
            a, b = points[:-1], points[1:]
            hits = ((np.minimum(a[:, 0], b[:, 0]) <= box.x1) & (np.maximum(a[:, 0], b[:, 0]) >= box.x0)
                    & (np.minimum(a[:, 1], b[:, 1]) <= box.y1) & (np.maximum(a[:, 1], b[:, 1]) >= box.y0))
            cost += int(hits.sum())
        candidates.append((cost, dx, dy, box))
    _, dx, dy, box = min(candidates, key=lambda c: c[0])
    occupied.append(box)
    ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=FONT_SIZE,
                color=FAIL_ABOVE_COLOR, ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top",
                zorder=11, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .1})


def fit_title_row(fig: Figure, axes: list, renderer) -> float:
    """Use inter-panel whitespace for one-line titles with a common font size."""
    scale = fig.dpi / 72
    padding, gap = 1 * scale, 4 * scale
    available = fig.bbox.width - 2 * padding - gap * (len(axes) - 1)
    widths = [ax.title.get_window_extent(renderer).width for ax in axes]
    size = min(FONT_SIZE, FONT_SIZE * available / sum(widths))
    size = np.floor(size * 10) / 10  # A little room for PDF/SVG font metrics.
    for ax in axes:
        ax.title.set_fontsize(size)
    widths = [ax.title.get_window_extent(renderer).width for ax in axes]
    lefts = [ax.bbox.x0 + ax.bbox.width / 2 - width / 2 for ax, width in zip(axes, widths)]
    # Stay near the panel centers, then resolve collisions and canvas overflow.
    for i in range(len(axes)):
        lefts[i] = max(lefts[i], lefts[i-1] + widths[i-1] + gap if i else padding)
    for i in reversed(range(len(axes))):
        right = lefts[i+1] - gap if i+1 < len(axes) else fig.bbox.width - padding
        lefts[i] = min(lefts[i], right - widths[i])
    for ax, left, width in zip(axes, lefts, widths):
        ax.title.set_x((left + width / 2 - ax.bbox.x0) / ax.bbox.width)
    return float(size)


def render_figure(data: FigureData, selection: dict, fmt: str = "svg") -> bytes:
    """Use the same compact layout for browser preview and both saved copies."""
    config = data.normalize(selection)
    with data.lock, matplotlib.rc_context({**RC_PARAMS, "svg.image_inline": True, "axes.linewidth": .75}):
        fig = Figure(figsize=(config["width"], config["height"]), dpi=400, facecolor="white")
        fig.patch.set_alpha(0 if config["transparent"] else 1)
        grid = fig.add_gridspec(2, 4, left=.027, right=.998, bottom=.15 if config["show_legend"] else .055,
                               top=.88, wspace=.20, hspace=.60 if config["show_ticks"] else .30,
                               height_ratios=[.64, 1])
        separator = ":\n" if config["title_layout"] == "two_lines" else ": "
        titles = [f"{TASK_TITLES[p['task']]}{separator}{p['failure_mode']}" for p in config["panels"]]
        has_onset = False
        image_axes = []
        score_axes = []
        for col, panel in enumerate(config["panels"]):
            task, ep, view = panel["task"], panel["episode"], config["view"]
            payload = data.episode(task, ep, view)
            time_key = "times" if config["time_axis"] == "seconds" else "frames"
            x = np.asarray(payload["times"] if time_key == "times" else payload["raw_frames"])
            y, threshold = np.asarray(payload["scores"]), payload["threshold"]
            ax = fig.add_subplot(grid[0, col])
            score_axes.append(ax)
            ax.set_facecolor("white")
            ax.patch.set_alpha(0 if config["transparent"] else 1)
            for c in payload["calibration"]:
                ax.plot(c[time_key], c["scores"], color=SUCCESS_COLOR, ls=SUCCESS_LS, lw=.65, alpha=.55, zorder=2)
            xx, below, above = split_curve(x, y, threshold)
            ax.plot(xx, below, color=FAIL_COLOR, lw=FAIL_LW, alpha=.7, zorder=3)
            ax.axhline(threshold, color=THRESHOLD_COLOR, ls=THRESHOLD_LS, lw=.9, alpha=.9, zorder=4)
            ax.plot(xx, above, color=FAIL_ABOVE_COLOR, lw=FAIL_ABOVE_LW, alpha=.95, zorder=5)
            onset = payload["onsets"].get(config["onset_annotator"]) if config["show_onset"] else None
            if onset is not None:
                onset_x = onset["time"] if config["time_axis"] == "seconds" else onset["raw_frame"]
                line = ax.axvline(onset_x, color=ONSET_COLOR, ls=ONSET_LS, lw=.6, alpha=.5, zorder=2.5)
                line.set_gid(f"failure-onset-{col}")
                has_onset = True
            ax.set_xlim(left=0)
            ax.margins(y=.16)
            ax.grid(True, lw=.35, alpha=.20)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(axis="both", which="both", bottom=config["show_ticks"], left=config["show_ticks"],
                           labelbottom=config["show_ticks"], labelleft=config["show_ticks"], length=2, pad=1, labelsize=FONT_SIZE)
            ax.set_xlabel("time", labelpad=3, fontsize=FONT_SIZE)
            if col == 0:
                ax.set_ylabel("score", fontsize=FONT_SIZE, labelpad=3)
            ax.set_title(titles[col], fontsize=FONT_SIZE, fontweight="bold", pad=4)
            images = grid[1, col].subgridspec(1, 2, wspace=.045)
            with data.lock:
                recording = data.recordings.episode(task, ep)
            indices = [int(np.searchsorted(payload["raw_frames"], f)) for f in panel["frames"]]
            occupied_labels = []
            for j, (raw_frame, index) in enumerate(zip(panel["frames"], indices)):
                label = f"({chr(97 + 2*col + j)})"
                ax.plot(x[index], y[index], "o", ms=3, mfc=FAIL_ABOVE_COLOR, mec="white", mew=.5, zorder=10)
                label_next_to_dot(ax, x[index], y[index], label, occupied_labels)
                frame_ax = fig.add_subplot(images[0, j])
                image_axes.append(frame_ax)
                frame_ax.imshow(getattr(recording, view)[raw_frame], interpolation="none")
                frame_ax.set_anchor("N")
                frame_ax.set_axis_off()
                frame_ax.text(.025, .975, label, transform=frame_ax.transAxes, ha="left", va="top",
                              fontsize=FONT_SIZE, color="white", weight="bold",
                              bbox={"facecolor": "black", "alpha": .65, "pad": 1, "edgecolor": "none"})
        if config["show_legend"]:
            handles = [Line2D([], [], color=SUCCESS_COLOR, ls=SUCCESS_LS, lw=.65, label="Successful calibration"),
                Line2D([], [], color=FAIL_COLOR, lw=FAIL_LW, label="Selected failure"),
                Line2D([], [], color=FAIL_ABOVE_COLOR, lw=FAIL_ABOVE_LW, label="Above threshold"),
                Line2D([], [], color=THRESHOLD_COLOR, ls=THRESHOLD_LS, lw=.9, label="Threshold")]
            if has_onset:
                handles.append(Line2D([], [], color=ONSET_COLOR, ls=ONSET_LS, lw=.6, alpha=.5, label="Failure onset"))
            # Follow the actual image bottoms rather than the figure edge: an
            # equal-aspect image can leave unused space in its grid cell.
            image_bottom = min(ax.get_position().y0 for ax in image_axes)
            legend_top = image_bottom - 3 / (72 * config["height"])
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, legend_top),
                       ncol=len(handles), frameon=False, borderaxespad=0, borderpad=.15,
                       fontsize=FONT_SIZE, handlelength=1.8, columnspacing=1., handletextpad=.5)
        apply_font_files(fig)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        if config["title_layout"] == "one_line":
            fit_title_row(fig, score_axes, canvas.get_renderer())
            canvas.draw()
        # Preserve the exported double-column width, so 7 pt stays 7 pt at
        # insertion. Trim vertical whitespace with one point of outer padding.
        content = fig.get_tightbbox(canvas.get_renderer())
        crop = Bbox.from_extents(0, content.y0 - 1/72, config["width"], content.y1 + 1/72)
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=400, transparent=config["transparent"],
                    facecolor=fig.get_facecolor(), bbox_inches=crop, pad_inches=0,
                    metadata={"Creator": "FoMo-FD Figure 4 selector"})
        fig.clear()
        return buffer.getvalue()


def export_figure(data: FigureData, selection: dict, output: Path, png: bool = False) -> dict:
    fmt = output.suffix.lower().lstrip(".")
    if fmt not in {"svg", "pdf"}:
        raise ValueError("Figure output must end with .svg or .pdf.")
    config = data.normalize(selection)
    figure_bytes = render_figure(data, config, fmt)
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "report_sha256": data.report_hash,
                  "labels_sha256": data.labels_hash, "taxonomy_sha256": sha256(TAXONOMY),
                  "paper_protocol": data.protocol, "score_method": "wm_nc", "smoothing": "none",
                  "export_format": fmt, "typography": {**font_metadata(), "titles": config["title_layout"],
                      "title_weight": "bold"},
                  "panels": []}
    if config["show_onset"] and config["onset_annotator"] == REFERENCE:
        provenance["onset_reference"] = {"id": REFERENCE, "protocol": str(data.onset_protocol_file),
            "protocol_id": data.onset_protocol["protocol_id"], "protocol_sha256": data.onset_protocol_hash,
            "rule": data.onset_rule}
    for p in config["panels"]:
        _, record, calibration_ids, entry = data.score_data(p["task"], config["view"])
        provenance["panels"].append({"task": p["task"], "episode": p["episode"],
            "raw_file": data.recordings.paths[p["task"], p["episode"]].name,
            "raw_frames": p["frames"], "score_cache": entry["path"], "score_cache_sha256": entry["sha256"],
            "checkpoint": entry["scoring_settings"]["checkpoint"], "threshold": record["threshold"],
            "calibration_episodes": calibration_ids, "alpha": record["alpha"],
            "onset": data.episode(p["task"], p["episode"], config["view"])["onsets"].get(config["onset_annotator"])
                     if config["show_onset"] else None})
    config["provenance"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".selection.json")
    output.write_bytes(figure_bytes)
    review_path = output.with_suffix(f".review.{fmt}")
    review_path.write_bytes(figure_bytes)
    sidecar.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if png:
        png_bytes = render_figure(data, config, "png")
        output.with_suffix(".png").write_bytes(png_bytes)
        output.with_suffix(".review.png").write_bytes(png_bytes)
    return {fmt: str(output), f"review_{fmt}": str(review_path), "selection": str(sidecar)}


def save_defaults(data: FigureData, selection: dict, path: Path = DEFAULT_CONFIG) -> dict:
    """Persist the validated selection without copying export-only provenance."""
    config = data.normalize(selection)
    # Preserve the configured report alias (usually "main"). Export JSONs
    # retain the exact report path and hash for reproduction of each figure.
    config["report"] = data.report_reference
    for panel in config["panels"]:
        panel.pop("category", None)  # Derived from the reporting taxonomy.
    header = ("# Default selection for Failure Example Plot Selector.\n"
              "# Run: python lerobot/utils/failure_example_plot_selector.py\n"
              "# Episodes use prepared IDs; frames are zero-based raw recording indices.\n"
              "# consensus_mean uses the current protocol's three annotators, per camera.\n")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(header + yaml.safe_dump(config, sort_keys=False, default_flow_style=None))
        os.chmod(temporary, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return config


def make_handler(data: FigureData, initial: dict, output: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, mime, body, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def error(self, error):
            self.send(400, "application/json", json.dumps({"error": str(error)}).encode())

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                if url.path == "/":
                    return self.send(200, "text/html; charset=utf-8", Path(__file__).with_suffix(".html").read_bytes())
                if url.path == "/api/catalog":
                    return self.send(200, "application/json", json.dumps({**data.catalog(), "initial": initial,
                                                                        "output": str(output)}).encode())
                downloads = {"/download/figure.svg": output.with_suffix(".svg"),
                             "/download/review.svg": output.with_suffix(".review.svg"),
                             "/download/figure.pdf": output.with_suffix(".pdf"),
                             "/download/review.pdf": output.with_suffix(".review.pdf"),
                             "/download/selection.json": output.with_suffix(".selection.json")}
                if url.path in downloads:
                    path = downloads[url.path]
                    mime = {".svg": "image/svg+xml", ".pdf": "application/pdf", ".json": "application/json"}
                    return self.send(200, mime[path.suffix],
                                     path.read_bytes(), path.name)
                query = parse_qs(url.query)
                task, ep, view = query["task"][0], int(query["episode"][0]), query["view"][0]
                if url.path == "/api/episode":
                    return self.send(200, "application/json", json.dumps(data.episode(task, ep, view), allow_nan=False).encode())
                if url.path == "/api/frame":
                    return self.send(200, "image/png", data.frame_png(task, ep, int(query["frame"][0]), view))
                self.send(404, "text/plain", b"Unknown endpoint")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (KeyError, ValueError, TypeError, OSError) as error:
                self.error(error)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    raise ValueError("Use this selector's own browser page.")
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Expected application/json.")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 100_000:
                    raise ValueError("Invalid figure selection size.")
                selection = json.loads(self.rfile.read(length))
                # Matplotlib and the shared recording cache are serialized.
                with data.lock:
                    if self.path == "/api/preview":
                        return self.send(200, "image/svg+xml", render_figure(data, selection))
                    if self.path in {"/api/export", "/api/export-pdf"}:
                        suffix = ".pdf" if self.path == "/api/export-pdf" else ".svg"
                        result = export_figure(data, selection, output.with_suffix(suffix))
                        return self.send(200, "application/json", json.dumps(result).encode())
                    if self.path == "/api/defaults":
                        config = save_defaults(data, selection)
                        initial.clear()
                        initial.update(data.normalize(config))
                        return self.send(200, "application/json", json.dumps({"initial": initial}).encode())
                self.send(404, "text/plain", b"Unknown endpoint")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (KeyError, ValueError, TypeError, OSError) as error:
                self.error(error)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", help="main or an offline report directory; overrides config")
    parser.add_argument("--out", type=Path, default=REPO / "logs/figures/figure4.svg")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--export", action="store_true", help="Save without starting the interactive server")
    parser.add_argument("--png", action="store_true", help="Also save a PNG with --export")
    args = parser.parse_args()
    if args.out.suffix.lower() not in {".svg", ".pdf"}:
        parser.error("--out must end with .svg or .pdf")
    try:
        config = read_config(args.config)
        report = args.report or config.get("report", "main")
        data = FigureData(score_report_path(REPO, report), report_reference=report)
        initial = data.normalize(config)
        if args.export:
            print(json.dumps(export_figure(data, initial, args.out.resolve(), args.png), indent=2))
            return
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(data, initial, args.out.resolve()))
    except (KeyError, ValueError, TypeError, OSError) as error:
        parser.exit(1, f"Failure Example Plot Selector: {error}\n")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Failure Example Plot Selector: {url}\nDefaults: {DEFAULT_CONFIG}\n"
          f"SVG output: {args.out.resolve().with_suffix('.svg')}\n"
          f"PDF output: {args.out.resolve().with_suffix('.pdf')}\nCtrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
