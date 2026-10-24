#!/usr/bin/env python3
"""Single-column WM-NC ablations from measured, retrospective test-FAR counts.

Run from the repository root in the lerobot environment (CPU only)::

    python lerobot/utils/plot_sensitivity.py
    python lerobot/utils/plot_sensitivity.py --font "Times New Roman"
    python lerobot/utils/plot_sensitivity.py --font-path /path/to/times.ttf

Writes logs/figures/ablation.{pdf,svg,png,csv,provenance.json}. The canvas is
exactly 3.5 x 1.35 inches; PNG resolution is 400 dpi. Nimbus Roman at 7 pt
provides compact typography. A requested unavailable font raises an error.

Each task/view/variant already has a deterministic threshold selected from
successful TEST episode maxima at FAR <= 5%, with strict score > threshold.
Failure detections use post-onset maxima and the three-annotator mean reference.
Pool saved matched_detected_n / population_n across tasks AFTER this per-task
threshold selection. These are retrospective operating points, not deployment
calibration guarantees. No scores, thresholds, error bars or inference are fit.
Training fractions are exactly 1/3, 2/3 and full, with fixed VAE features.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from paper_plot_style import FONT_FAMILY, FONT_SIZE, RC_PARAMS, apply_font_files, embedded_font_file


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "docs/results/details/offline_by_failure.csv"
if (REPO / "logs/benchmarks/result_tables/details/offline_by_failure.csv").is_file():
    DEFAULT_SOURCE = REPO / "logs/benchmarks/result_tables/details/offline_by_failure.csv"
TASKS = ("needle_pickup", "ring_over_post", "tissue_retraction", "shunt_insertion")
SIZE = (3.5, 1.35)
DPI = 400
# Preserve the compact plot geometry with 7 pt labels and header.
MARGINS = dict(left=.137, right=.975, bottom=.40/SIZE[1], top=1-.255/SIZE[1], wspace=.16)
VIEWS = {"wrist": dict(label="Wrist", color="#B83A35", linestyle="-", marker="o"),
         "fix": dict(label="Fixed", color="#3476A8", linestyle=(0, (4, 2.2)), marker="s")}
# setting, printed tick label, measured variant. K/L sweeps keep the other at 4.
PANELS = (
    ("K", "(a) Horizon K", ((1, "1", "wm_nc_k1_l4"), (2, "2", "wm_nc_k2_l4"),
        (3, "3", "wm_nc_k3_l4"), (4, "4", "main"),
        (6, "6", "wm_nc_k6_l4"), (8, "8", "wm_nc_k8_l4"))),
    ("L", "(b) History L", ((1, "1", "wm_nc_k4_l1"), (2, "2", "wm_nc_k4_l2"),
        (4, "4", "main"), (6, "6", "wm_nc_k4_l6"), (8, "8", "wm_nc_k4_l8"))),
    ("training_data", "(c) Training data", ((1/3, "1/3", "wm_k4_l4_data33"),
        (2/3, "2/3", "wm_k4_l4_data66"), (1., "Full", "main"))),
)
CSV_FIELDS = ("panel", "setting", "tick_label", "variant", "view", "is_main",
    "detected_n", "total_n", "FDR_percent", "test_false_alarm_n", "success_test_n",
    "test_FAR_percent", "test_FAR_target_percent", "evaluation_protocol", "onset_reference")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_values(source: Path) -> tuple[list[dict], list[dict]]:
    """Require complete, identical cohorts; never mix modes or average rates."""
    variants = {v for _, _, points in PANELS for _, _, v in points}
    selected = {}
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["variant"] not in variants or row["method"] != "wm_nc"
                    or row["calibration"] != "episode" or row["group_type"] != "all_failures"
                    or row["view"] not in VIEWS):
                continue
            key = row["variant"], row["view"], row["task"]
            if key in selected:
                raise ValueError(f"Duplicate task aggregate: {key}")
            selected[key] = row
    expected = {(variant, view, task) for variant in variants for view in VIEWS for task in TASKS}
    if set(selected) != expected:
        raise ValueError(f"Incomplete/unexpected task coverage: missing {sorted(expected-set(selected))}; "
                         f"extra {sorted(set(selected)-expected)}")
    protocols = {r["evaluation_protocol"] for r in selected.values()}
    if len(protocols) != 1 or not next(iter(protocols)):
        raise ValueError("Do not mix evaluation protocols in an ablation.")
    for (variant, view, task), row in selected.items():
        n, d = int(row["population_n"]), int(row["matched_detected_n"])
        ns, fp = int(row["success_test_n"]), int(row["matched_false_alarm_n"])
        if (row["measurement"] != "dense_offline" or row["annotator_id"] != "consensus_mean"
                or int(row["recorded_n"]) != n or int(row["not_recorded_n"]) or int(row["unknown_n"])):
            raise ValueError(f"Need complete offline mean-onset results: {variant}/{view}/{task}")
        if not (n > 0 and 0 <= d <= n and ns > 0 and 0 <= fp <= ns):
            raise ValueError("Invalid measured episode counts.")
        if (not math.isclose(float(row["matched_FAR_target"]), .05, abs_tol=1e-12)
                or not math.isclose(float(row["matched_FDR"]), d/n, abs_tol=1e-12)
                or not math.isclose(float(row["matched_FAR"]), fp/ns, abs_tol=1e-12)
                or fp/ns > .05 + 1e-12):
            raise ValueError("Inconsistent retrospective 5% test-FAR counts/rates.")
        ids = row["episode_ids"].split(";")
        reference = selected["main", "wrist", task]
        if (len(ids) != n or len(set(ids)) != n
                or set(ids) != set(reference["episode_ids"].split(";"))
                or ns != int(reference["success_test_n"])):
            raise ValueError(f"Ablation populations differ: {variant}/{view}/{task}")
    values = []
    for panel, _, points in PANELS:
        for setting, label, variant in points:
            for view in VIEWS:
                rows = [selected[variant, view, task] for task in TASKS]
                n = sum(int(r["population_n"]) for r in rows)
                d = sum(int(r["matched_detected_n"]) for r in rows)
                ns = sum(int(r["success_test_n"]) for r in rows)
                fp = sum(int(r["matched_false_alarm_n"]) for r in rows)
                values.append(dict(panel=panel, setting=setting, tick_label=label, variant=variant,
                    view=view, is_main=variant == "main", detected_n=d, total_n=n, FDR_percent=100*d/n,
                    test_false_alarm_n=fp, success_test_n=ns, test_FAR_percent=100*fp/ns,
                    test_FAR_target_percent=5., evaluation_protocol=rows[0]["evaluation_protocol"],
                    onset_reference=rows[0]["annotator_id"]))
    return values, list(selected.values())


def resolve_font(family: str, path: Path | None) -> tuple[str, Path]:
    if path:
        font_manager.fontManager.addfont(path)
        family = font_manager.FontProperties(fname=path).get_name()
    font_file = Path(font_manager.findfont(font_manager.FontProperties(family=[family]),
                                         fallback_to_default=False))
    return family, font_file


def make_figure(values: list[dict]) -> Figure:
    fig = Figure(figsize=SIZE, dpi=DPI, facecolor="white")
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(**MARGINS)
    for i, (ax, (panel, label, points)) in enumerate(zip(axes, PANELS)):
        for view, style in VIEWS.items():
            rows = [r for r in values if r["panel"] == panel and r["view"] == view]
            x, y = [r["setting"] for r in rows], [r["FDR_percent"] for r in rows]
            ax.plot(x, y, color=style["color"], ls=style["linestyle"], lw=1.05,
                    marker=style["marker"], ms=3.1, mfc="white", mec=style["color"], mew=.8,
                    markevery=[j for j, r in enumerate(rows) if not r["is_main"]], zorder=3)
            main = next(r for r in rows if r["is_main"])
            ax.plot(main["setting"], main["FDR_percent"], marker="*", ls="none", ms=6.5,
                    mfc=style["color"], mec=style["color"], mew=.5, zorder=4)
        ax.set_ylim(25, 104)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_xticks([p[0] for p in points], [p[1] for p in points])
        ax.margins(x=.09)
        ax.set_xlabel(label, fontsize=FONT_SIZE, labelpad=4)
        ax.tick_params(axis="both", labelsize=FONT_SIZE, length=2.3, width=.6, pad=2)
        ax.tick_params(axis="y", labelleft=i == 0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(.6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.87", lw=.4)
    axes[0].set_ylabel("FDR (%)", fontsize=FONT_SIZE, labelpad=3)
    fig.text(MARGINS["left"], 1-.085/SIZE[1], "5% FAR budget", fontsize=FONT_SIZE, ha="left", va="center")
    handles = [Line2D([], [], color=s["color"], ls=s["linestyle"], lw=1.05, marker=s["marker"],
        ms=3.1, mfc="white", mec=s["color"], mew=.8, label=s["label"]) for s in VIEWS.values()]
    handles.append(Line2D([], [], color="0.25", marker="*", ls="none", ms=6.5, mew=.5, label="Main"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(MARGINS["right"], 1-.015/SIZE[1]),
               ncol=3, fontsize=FONT_SIZE, frameon=False, handlelength=1.4,
               columnspacing=.8, handletextpad=.4, borderpad=0, borderaxespad=0)
    return fig


def check_layout(fig: Figure) -> dict:
    """Check visible labels/legend for clipping and mutual overlap at export DPI."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    artists = [("header", fig.texts[0]), ("legend", fig.legends[0])]
    for i, ax in enumerate(fig.axes):
        artists += [(f"panel {i+1} x label", ax.xaxis.label)]
        if i == 0:
            artists.append(("y label", ax.yaxis.label))
        artists += [(f"panel {i+1} tick {j}", t) for j, t in
                    enumerate(ax.get_xticklabels() + ax.get_yticklabels()) if t.get_visible()]
    boxes = [(name, artist.get_window_extent(renderer)) for name, artist in artists]
    for name, box in boxes:
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
            raise ValueError(f"Clipped label: {name}")
    for i, (name, box) in enumerate(boxes):
        for other_name, other in boxes[i+1:]:
            if box.overlaps(other):
                raise ValueError(f"Overlapping labels: {name} / {other_name}")
    return {"canvas_inches": list(SIZE), "png_pixels": list(canvas.get_width_height()),
            "margins": MARGINS, "y_limits": list(fig.axes[0].get_ylim()),
            "y_ticks": fig.axes[0].get_yticks().tolist(),
            "visible_labels_checked": len(boxes), "clipping_or_label_overlaps": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=REPO / "logs/figures/ablation", help="Output basename")
    parser.add_argument("--font", default=FONT_FAMILY)
    parser.add_argument("--font-path", type=Path, help="Register a local TTF/OTF font")
    args = parser.parse_args()
    try:
        values, task_rows = load_values(args.source)
        manifest_path = args.source.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            entry = manifest.get("files", {}).get(f"{args.source.parent.name}/{args.source.name}")
            if entry and sha256(args.source) != entry["sha256"]:
                raise ValueError("Measured source CSV no longer matches its report manifest.")
        family, font_path = resolve_font(args.font, args.font_path)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with matplotlib.rc_context({**RC_PARAMS, "font.family": family, "savefig.bbox": None,
                "figure.autolayout": False, "figure.constrained_layout.use": False}):
            fig = make_figure(values)
            apply_font_files(fig)
            layout = check_layout(fig)
            for fmt in ("pdf", "svg", "png"):
                fig.savefig(args.out.with_suffix(f".{fmt}"), format=fmt, dpi=DPI,
                            bbox_inches=None, facecolor="white", metadata={"Creator": "FoMo-FD WM-NC ablations"})
            fig.clear()
        with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(values)
        provenance = dict(created_at=datetime.now(timezone.utc).isoformat(),
            source=str(args.source.resolve()), source_sha256=sha256(args.source), script_sha256=sha256(Path(__file__)),
            font=dict(family=family, size_pt=FONT_SIZE, file=str(font_path), sha256=sha256(font_path),
                      embedded_file=embedded_font_file(str(font_path))), layout=layout,
            aggregation="100 * sum(matched_detected_n) / sum(population_n); per-task/view test thresholds",
            operating_point="Retrospective deterministic strict crossing at test FAR <= 5%; post-onset FDR",
            error_bars=None, training_fractions=["1/3", "2/3", "1"],
            task_counts=[{k: r[k] for k in ("variant", "task", "view", "population_n", "matched_detected_n",
                "success_test_n", "matched_false_alarm_n", "matched_threshold", "evaluation_protocol", "annotator_id")}
                for r in task_rows])
        args.out.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        print(json.dumps({fmt: str(args.out.with_suffix(f".{fmt}")) for fmt in
                          ("pdf", "svg", "png", "csv", "provenance.json")}, indent=2))
    except (KeyError, ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Ablation plot: {error}\n")


if __name__ == "__main__":
    main()
