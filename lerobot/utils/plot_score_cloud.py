from __future__ import annotations

"""
Qualitative score-cloud plot for task/method comparison.

Behavior:
- Automatically resolves score files from task + method.
- Uses calibration successful episodes from replay.yaml.
- Uses failure episodes exactly as listed in replay.yaml.
- Plots:
    successful calibration curves: light blue dashed
    failure curves: red solid
    conformal threshold: gray dash-dot
- No text labels, titles, or legend.
- Press "s" to save as SVG.

Examples:

Single panel:
python utils/plot_score_cloud.py \
  --task shunt_insertion \
  --method wm_nc \
  --view wrist \
  --smooth 7 \
  --out shunt_wm_nc_wrist_cloud.svg

Full 4 x 4 wrist-view grid:
python utils/plot_score_cloud.py \
  --task all \
  --method all \
  --view wrist \
  --smooth 7 \
  --out score_cloud_all_wrist.svg
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


plt.rcParams["svg.fonttype"] = "none"

TASKS = ["needle_pickup", "ring_over_post", "tissue_retraction", "shunt_insertion"]
METHODS = ["wm_nc", "logpzo", "rnd", "wm_pe"]

SUCCESS_COLOR = "#2F7FB8"
FAIL_COLOR = "#C03532"
THRESHOLD_COLOR = "0.22"

SUCCESS_LS = (0, (4.0, 2.2))
FAIL_LS = "-"
THRESHOLD_LS = "-."

SUCCESS_LW = 0.65
FAIL_LW = 0.85
THRESHOLD_LW = 0.9

SUCCESS_ALPHA = 0.42
FAIL_ALPHA = 0.65
THRESHOLD_ALPHA = 0.90
GRID_ALPHA = 0.20


# ---------------------------------------------------------------------
# YAML / episode helpers
# ---------------------------------------------------------------------

def load_text_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config extension: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a dict: {path}")
    return data


def parse_episode_ids(x: Any) -> list[int]:
    """
    Supports:
      - None
      - space-separated string: "1 2 3"
      - comma-separated string: "1,2,3"
      - list/tuple/set of ints or strings
      - dict of groups, recursively flattened
    """
    if x is None:
        return []

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        s = s.replace(",", " ")
        return [int(v) for v in s.split()]

    if isinstance(x, (list, tuple, set)):
        out: list[int] = []
        for v in x:
            out.extend(parse_episode_ids(v))
        return out

    if isinstance(x, dict):
        out: list[int] = []
        for v in x.values():
            out.extend(parse_episode_ids(v))
        return out

    return [int(x)]


def dedupe_episode_order(order: list[int]) -> list[int]:
    seen = set()
    out = []
    for ep in order:
        if ep not in seen:
            seen.add(ep)
            out.append(ep)
    return out


def load_episode_info(info_yaml: str | Path) -> dict[str, Any]:
    raw = load_text_config(info_yaml)
    return {
        "ID": raw.get("ID", {}) or {},
        "OOD": raw.get("OOD", {}) or {},
    }


def get_success_eps(info: dict[str, Any], include_ood_success: bool) -> list[int]:
    out = []
    out.extend(parse_episode_ids(info["ID"].get("success")))

    if include_ood_success:
        out.extend(parse_episode_ids(info["OOD"].get("success")))

    return dedupe_episode_order(out)


def get_failure_eps(
    info: dict[str, Any],
    *,
    include_task_level_fail: bool,
    include_ood_fail: bool,
) -> list[int]:
    out = []

    out.extend(parse_episode_ids(info["ID"].get("state_level_fail")))

    if include_task_level_fail:
        out.extend(parse_episode_ids(info["ID"].get("task_level_fail")))

    if include_ood_fail:
        out.extend(parse_episode_ids(info["OOD"].get("state_level_fail")))
        if include_task_level_fail:
            out.extend(parse_episode_ids(info["OOD"].get("task_level_fail")))

    return dedupe_episode_order(out)


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def resolve_score_result_dir(root_dir: Path, task: str, prl: bool) -> Path:
    base = "prl_data" if prl else "data"
    return root_dir / "fomofd" / "tasks" / task / base / "score_result"


def resolve_info_yaml(root_dir: Path, task: str, prl: bool) -> Path:
    return resolve_score_result_dir(root_dir, task, prl) / "replay.yaml"


def resolve_score_npz(root_dir: Path, task: str, method: str, prl: bool) -> Path:
    base = resolve_score_result_dir(root_dir, task, prl)

    if method == "wm_nc":
        return base / "score_val_reactive" / "window_scores_reactive.npz"

    if method == "wm_pe":
        return base / "score_val_predictive" / "window_scores_predictive.npz"

    if method == "logpzo":
        return base / "faildetect_logpzo" / "frame_scores.npz"

    if method == "rnd":
        return base / "fiper_rnd" / "frame_scores.npz"

    raise ValueError(f"Unknown method: {method}")


def method_to_score_method(method: str) -> str:
    if method in {"wm_nc", "wm_pe"}:
        return "fm_wm"
    if method == "logpzo":
        return "faildetect_logpzo"
    if method == "rnd":
        return "fiper_rnd"
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------
# Score loading
# ---------------------------------------------------------------------

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return x.astype(np.float64, copy=False)

    kernel = np.ones(w, dtype=np.float64)
    num = np.convolve(x.astype(np.float64), kernel, mode="same")
    den = np.convolve(np.ones_like(x, dtype=np.float64), kernel, mode="same")
    return num / np.maximum(den, 1e-12)


def load_score_run_config(score_npz: str | Path) -> dict[str, Any]:
    path = Path(score_npz).resolve().parent / "score_run_config.json"
    if not path.exists():
        return {}
    return load_text_config(path)


def load_merged_config(score_npz: str | Path) -> dict[str, Any]:
    path = Path(score_npz).resolve().parent / "merged_flow_matching_wm_config.json"
    if not path.exists():
        return {}
    return load_text_config(path)


def load_history_and_horizon_from_score(score_npz: str | Path) -> tuple[int, int]:
    cfg = load_merged_config(score_npz)
    data = cfg.get("data", {})
    if not isinstance(data, dict):
        return 1, 1

    history_len = int(data.get("history_len", 1))
    pred_horizon = int(data.get("pred_horizon", 1))
    return history_len, pred_horizon


def compute_valid_target_range(
    anchor_x: np.ndarray,
    history_len: int,
    pred_horizon: int,
) -> tuple[int, int]:
    if len(anchor_x) == 0:
        return 0, -1

    episode_last_frame = int(anchor_x.max()) + pred_horizon
    valid_lo = int(history_len)
    valid_hi = int(episode_last_frame - (pred_horizon - 1))
    return valid_lo, valid_hi


def load_selected_fm_wm_scores(
    score_npz: str | Path,
    score_key: str,
    single_view_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(score_npz)

    if "scores_union" not in data and "scores" in data:
        scores_union = data["scores"]
        scores_view = None
    else:
        scores_union = data["scores_union"]
        scores_view = data["scores_view"] if "scores_view" in data else None

    if score_key == "union":
        scores = scores_union
    else:
        if scores_view is None:
            if score_key != single_view_name:
                raise ValueError(
                    f"{score_npz} has only one score stream. "
                    f"Use --single_view_name {score_key} if intended."
                )
            scores = scores_union
        else:
            if scores_view.ndim != 2:
                raise ValueError(f"Expected scores_view [N,V], got {scores_view.shape}")

            num_views = scores_view.shape[1]
            if num_views == 1:
                if score_key != single_view_name:
                    raise ValueError(
                        f"{score_npz} has one score stream, interpreted as {single_view_name}."
                    )
                scores = scores_view[:, 0]
            elif num_views == 2:
                if score_key == "fix":
                    scores = scores_view[:, 0]
                elif score_key == "wrist":
                    scores = scores_view[:, 1]
                else:
                    raise ValueError(f"Unsupported score_key={score_key}")
            else:
                raise ValueError(f"Expected 1 or 2 views, got {num_views}")

    return (
        np.asarray(data["episode_index"], dtype=np.int64),
        np.asarray(data["anchor_frame_index"], dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
    )


def build_fm_wm_episode_score_curves(
    score_npz: str | Path,
    score_key: str,
    single_view_name: str,
    smooth: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    history_len, pred_horizon = load_history_and_horizon_from_score(score_npz)
    ep_idx, anchor_idx, scores = load_selected_fm_wm_scores(
        score_npz=score_npz,
        score_key=score_key,
        single_view_name=single_view_name,
    )

    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for ep in np.unique(ep_idx):
        mask = ep_idx == int(ep)

        a = anchor_idx[mask].astype(np.int64)
        y = scores[mask].astype(np.float64)

        order = np.argsort(a)
        a = a[order]
        y = y[order]

        x = a + pred_horizon
        valid_lo, valid_hi = compute_valid_target_range(a, history_len, pred_horizon)
        keep = (x >= valid_lo) & (x <= valid_hi)

        x = x[keep]
        y = y[keep]
        y = moving_average(y, smooth)

        curves[int(ep)] = (x, y)

    return curves


def load_frame_level_scores(
    score_npz: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(score_npz)

    required = ["episode_index", "frame_index", "scores"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{score_npz} missing keys {missing}; expected {required}")

    return (
        np.asarray(data["episode_index"], dtype=np.int64),
        np.asarray(data["frame_index"], dtype=np.int64),
        np.asarray(data["scores"], dtype=np.float64),
    )


def build_frame_level_episode_score_curves(
    score_npz: str | Path,
    smooth: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    ep_idx, frame_idx, scores = load_frame_level_scores(score_npz)
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for ep in np.unique(ep_idx):
        mask = ep_idx == int(ep)

        x = frame_idx[mask].astype(np.int64)
        y = scores[mask].astype(np.float64)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        y = moving_average(y, smooth)
        curves[int(ep)] = (x, y)

    return curves


def build_episode_score_curves(
    score_npz: str | Path,
    method: str,
    view: str,
    single_view_name: str,
    smooth: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    score_method = method_to_score_method(method)

    if score_method == "fm_wm":
        return build_fm_wm_episode_score_curves(
            score_npz=score_npz,
            score_key=view,
            single_view_name=single_view_name,
            smooth=smooth,
        )

    return build_frame_level_episode_score_curves(
        score_npz=score_npz,
        smooth=smooth,
    )


# ---------------------------------------------------------------------
# Threshold / plotting
# ---------------------------------------------------------------------

def compute_episode_max_scores(
    score_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_ids: list[int],
) -> np.ndarray:
    vals = []

    for ep in episode_ids:
        if ep not in score_curves:
            continue
        _, y = score_curves[ep]
        if len(y) > 0:
            vals.append(float(np.max(y)))

    return np.asarray(vals, dtype=np.float64)


def conformal_threshold_from_episode_maxima(
    episode_max_scores: np.ndarray,
    alpha: float,
) -> float:
    if episode_max_scores.size == 0:
        raise ValueError("No calibration episode maxima available.")

    scores = np.sort(episode_max_scores)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))

    if k > n:
        return float("inf")

    return float(scores[k - 1])


def compute_limits(
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    success_eps: list[int],
    failure_eps: list[int],
    threshold: float | None,
) -> tuple[float, float, float, float]:
    xs = []
    ys = []

    for ep in success_eps + failure_eps:
        if ep not in curves:
            continue

        x, y = curves[ep]
        if len(x) == 0 or len(y) == 0:
            continue

        xs.append(x)
        ys.append(y)

    if not xs:
        return 0.0, 1.0, 0.0, 1.0

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)

    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))

    if threshold is not None and np.isfinite(threshold):
        y_min = min(y_min, float(threshold))
        y_max = max(y_max, float(threshold))

    if x_max <= x_min:
        x_max = x_min + 1.0

    if y_max <= y_min:
        y_pad = 1.0 if y_max == 0 else 0.1 * abs(y_max)
    else:
        y_pad = 0.08 * (y_max - y_min)

    x_pad = 0.02 * max(1e-9, x_max - x_min)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def style_axis(ax, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # Even grid spacing, but no numeric tick labels.
    ax.set_xticks(np.linspace(x_min, x_max, 5))
    ax.set_yticks(np.linspace(y_min, y_max, 4))

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    ax.grid(True, linewidth=0.35, alpha=GRID_ALPHA)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def plot_score_cloud_panel(
    ax,
    *,
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    success_eps: list[int],
    failure_eps: list[int],
    threshold: float | None,
) -> None:
    for ep in success_eps:
        if ep not in curves:
            continue

        x, y = curves[ep]
        if len(x) == 0:
            continue

        ax.plot(
            x,
            y,
            color=SUCCESS_COLOR,
            linestyle=SUCCESS_LS,
            linewidth=SUCCESS_LW,
            alpha=SUCCESS_ALPHA,
            zorder=1,
        )

    for ep in failure_eps:
        if ep not in curves:
            continue

        x, y = curves[ep]
        if len(x) == 0:
            continue

        ax.plot(
            x,
            y,
            color=FAIL_COLOR,
            linestyle=FAIL_LS,
            linewidth=FAIL_LW,
            alpha=FAIL_ALPHA,
            zorder=3,
        )

    if threshold is not None and np.isfinite(threshold):
        ax.axhline(
            threshold,
            color=THRESHOLD_COLOR,
            linestyle=THRESHOLD_LS,
            linewidth=THRESHOLD_LW,
            alpha=THRESHOLD_ALPHA,
            zorder=2,
        )

    x_min, x_max, y_min, y_max = compute_limits(curves, success_eps, failure_eps, threshold)
    style_axis(ax, x_min, x_max, y_min, y_max)


def load_panel_data(
    *,
    root_dir: Path,
    task: str,
    method: str,
    prl: bool,
    view: str,
    single_view_name: str,
    smooth: int,
    alpha: float,
    include_task_level_fail: bool,
    include_ood_success: bool,
    include_ood_fail: bool,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[int], list[int], float]:
    info_yaml = resolve_info_yaml(root_dir, task, prl)
    score_npz = resolve_score_npz(root_dir, task, method, prl)

    if not info_yaml.exists():
        raise FileNotFoundError(f"Missing info yaml: {info_yaml}")
    if not score_npz.exists():
        raise FileNotFoundError(f"Missing score npz: {score_npz}")

    info = load_episode_info(info_yaml)

    success_eps = get_success_eps(info, include_ood_success=include_ood_success)
    failure_eps = get_failure_eps(
        info,
        include_task_level_fail=include_task_level_fail,
        include_ood_fail=include_ood_fail,
    )

    curves = build_episode_score_curves(
        score_npz=score_npz,
        method=method,
        view=view,
        single_view_name=single_view_name,
        smooth=smooth,
    )

    calib_max_scores = compute_episode_max_scores(curves, success_eps)
    threshold = conformal_threshold_from_episode_maxima(calib_max_scores, alpha=alpha)

    print(
        f"[PANEL] task={task} method={method} view={view} "
        f"n_success={len(success_eps)} n_fail={len(failure_eps)} "
        f"threshold={threshold:.6g}"
    )

    missing_success = [ep for ep in success_eps if ep not in curves]
    missing_failure = [ep for ep in failure_eps if ep not in curves]
    if missing_success:
        print(f"[WARN] Missing success score curves: task={task} method={method} eps={missing_success}")
    if missing_failure:
        print(f"[WARN] Missing failure score curves: task={task} method={method} eps={missing_failure}")

    return curves, success_eps, failure_eps, threshold


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, choices=TASKS + ["all"])
    parser.add_argument("--method", type=str, required=True, choices=METHODS + ["all"])
    parser.add_argument("--view", type=str, default="wrist", choices=["fix", "wrist", "union"])

    parser.add_argument("--smooth", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--include_task_level_fail", action="store_true")
    parser.add_argument("--include_ood_success", action="store_true")
    parser.add_argument("--no_ood_fail", action="store_true")

    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--prl", dest="prl", action="store_true", default=True)
    parser.add_argument("--no_prl", dest="prl", action="store_false")
    parser.add_argument("--single_view_name", type=str, default="wrist", choices=["fix", "wrist"])

    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)

    args = parser.parse_args()

    root_dir = (
        Path(__file__).resolve().parents[2]
        if args.root_dir is None
        else Path(args.root_dir).expanduser().resolve()
    )

    tasks = TASKS if args.task == "all" else [args.task]
    methods = METHODS if args.method == "all" else [args.method]

    n_rows = len(tasks)
    n_cols = len(methods)

    width = float(args.width) if args.width is not None else 3.0 * n_cols
    height = float(args.height) if args.height is not None else 1.85 * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width, height),
        squeeze=False,
    )
    fig.patch.set_alpha(0.0)

    for ax in axes.ravel():
        ax.patch.set_alpha(0.0)

    for r, task in enumerate(tasks):
        for c, method in enumerate(methods):
            ax = axes[r, c]

            curves, success_eps, failure_eps, threshold = load_panel_data(
                root_dir=root_dir,
                task=task,
                method=method,
                prl=bool(args.prl),
                view=args.view,
                single_view_name=args.single_view_name,
                smooth=int(args.smooth),
                alpha=float(args.alpha),
                include_task_level_fail=bool(args.include_task_level_fail),
                include_ood_success=bool(args.include_ood_success),
                include_ood_fail=not bool(args.no_ood_fail),
            )

            plot_score_cloud_panel(
                ax,
                curves=curves,
                success_eps=success_eps,
                failure_eps=failure_eps,
                threshold=threshold,
            )

    fig.subplots_adjust(
        left=0.035,
        right=0.995,
        bottom=0.045,
        top=0.995,
        wspace=0.015,
        hspace=0.035,
    )

    if args.out is None:
        task_name = "all" if args.task == "all" else args.task
        method_name = "all" if args.method == "all" else args.method
        out_path = Path(f"score_cloud_{task_name}_{method_name}_{args.view}.svg").resolve()
    else:
        out_path = Path(args.out).expanduser().resolve()

    def on_key(event):
        if event.key == "s":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                out_path,
                format="svg",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.01,
            )
            print(f"[SAVE] {out_path}")
        elif event.key in {"q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("[INFO] Press s to save SVG. Press q/Esc to close.")
    plt.show()


if __name__ == "__main__":
    main()