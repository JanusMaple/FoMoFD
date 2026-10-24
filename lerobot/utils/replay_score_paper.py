from __future__ import annotations

"""
Paper-ready score curve plot + replay frame selector.

Main functions:
1. Load score curves from:
   - FM-WM window_scores_*.npz
   - faildetect_logpzo frame_scores.npz
   - fiper_rnd frame_scores.npz
2. Compute conformal threshold from successful calibration episodes.
3. Save transparent paper-ready score plot:
   - successful curves: dashed, thin
   - failed curves: solid
   - failed portions above threshold: thicker
   - threshold: dash-dot horizontal line
4. Open replay UI:
   - two camera views side-by-side
   - score curves shown below
   - space: pause/resume
   - left/right: step when paused
   - s: save current fix/wrist frames separately
   - q/esc: quit

Example:
python utils/replay_score_paper.py \
  --task shunt_insertion \
  --prl \
  --info_yaml ../fomofd/tasks/shunt_insertion/prl_data/score_result/replay.yaml \
  --score_npz ../fomofd/tasks/shunt_insertion/prl_data/score_result/score_val_reactive/window_scores_reactive.npz \
  --score_key wrist \
  --single_view_name wrist \
  --smooth 7 \
  --skip_id_success
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from datasets import load_from_disk


FPS = 30.0

SUCCESS_COLOR = "#1f77b4"      # blue
FAIL_COLOR = "#d62728"         # red
FAIL_ABOVE_COLOR = "#8b0000"   # dark red
THRESHOLD_COLOR = "0.25"       # dark gray

SUCCESS_LS = (0, (4.0, 2.2))   # dashed, B/W readable
FAIL_LS = "-"
THRESHOLD_LS = "-."

SUCCESS_LW = 0.65
FAIL_LW = 0.85
FAIL_ABOVE_LW = 1.2
THRESHOLD_LW = 0.9

# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return x.astype(np.float64, copy=False)

    kernel = np.ones(w, dtype=np.float64)
    num = np.convolve(x.astype(np.float64), kernel, mode="same")
    den = np.convolve(np.ones_like(x, dtype=np.float64), kernel, mode="same")
    return num / np.maximum(den, 1e-12)


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
    if x is None:
        return []
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return []
        return [int(v) for v in x.split()]
    if isinstance(x, (list, tuple, set)):
        return [int(v) for v in x]
    raise ValueError(f"Unsupported episode id list type: {type(x)}")


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------

def resolve_processed_hf_dir(root_dir: Path, task: str, prl: bool) -> Path:
    base = "prl_data" if prl else "data"
    return root_dir / "fomofd" / "tasks" / task / base / "processed_hf"


def chw_u8_to_hwc_u8(x: Any) -> np.ndarray:
    x = np.asarray(x)

    if x.dtype != np.uint8:
        x = x.astype(np.uint8, copy=False)

    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[1] > 4 and x.shape[2] > 4:
        return np.transpose(x, (1, 2, 0))

    if x.ndim == 3 and x.shape[-1] in (1, 3):
        return x

    raise ValueError(f"Unexpected image shape {x.shape}; expected CHW or HWC.")


def resize_nn(img: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
    h, w = img.shape[:2]
    if (h, w) == (h_out, w_out):
        return img

    ys = np.linspace(0, h - 1, h_out).astype(np.int64)
    xs = np.linspace(0, w - 1, w_out).astype(np.int64)
    return img[ys][:, xs]


def stack_side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]

    if (ha, wa) != (hb, wb):
        b = resize_nn(b, ha, wa)

    return np.concatenate([a, b], axis=1)


def get_episode_indices(ds, episode_idx: int) -> np.ndarray:
    ep = np.asarray(ds["episode_index"], dtype=np.int64)
    idx = np.where(ep == int(episode_idx))[0]

    if idx.size == 0:
        raise ValueError(f"No rows found for episode_idx={episode_idx}")

    steps = np.asarray([ds[int(i)]["step_index"] for i in idx], dtype=np.int64)
    return idx[np.argsort(steps)]


# ---------------------------------------------------------------------
# Episode info
# ---------------------------------------------------------------------

def load_episode_info(info_yaml: str | Path) -> dict[str, list[int]]:
    raw = load_text_config(info_yaml)
    id_node = raw.get("ID", {})
    ood_node = raw.get("OOD", {})

    return {
        "id_success": parse_episode_ids(id_node.get("success")),
        "id_state_level_fail": parse_episode_ids(id_node.get("state_level_fail")),
        "id_task_level_fail": parse_episode_ids(id_node.get("task_level_fail")),
        "ood_success": parse_episode_ids(ood_node.get("success")),
        "ood_state_level_fail": parse_episode_ids(ood_node.get("state_level_fail")),
        "ood_task_level_fail": parse_episode_ids(ood_node.get("task_level_fail")),
    }


def dedupe_episode_order(order: list[int]) -> list[int]:
    seen = set()
    out = []
    for ep in order:
        if ep not in seen:
            seen.add(ep)
            out.append(ep)
    return out


def build_plot_episode_order(
    info: dict[str, list[int]],
    include_task_level_fail: bool,
) -> list[int]:
    order = []
    order.extend(info["id_success"])
    order.extend(info["ood_success"])
    order.extend(info["id_state_level_fail"])
    order.extend(info["ood_state_level_fail"])

    if include_task_level_fail:
        order.extend(info["id_task_level_fail"])
        order.extend(info["ood_task_level_fail"])

    return dedupe_episode_order(order)


def build_replay_episode_order(
    info: dict[str, list[int]],
    include_task_level_fail: bool,
    skip_id_success: bool,
) -> list[int]:
    order = []

    if not skip_id_success:
        order.extend(info["id_success"])

    order.extend(info["id_state_level_fail"])

    if include_task_level_fail:
        order.extend(info["id_task_level_fail"])

    order.extend(info["ood_success"])
    order.extend(info["ood_state_level_fail"])

    if include_task_level_fail:
        order.extend(info["ood_task_level_fail"])

    return dedupe_episode_order(order)


def is_success_episode(ep: int, info: dict[str, list[int]]) -> bool:
    return ep in info["id_success"] or ep in info["ood_success"]


def is_failure_episode(
    ep: int,
    info: dict[str, list[int]],
    include_task_level_fail: bool,
) -> bool:
    if ep in info["id_state_level_fail"]:
        return True
    if ep in info["ood_state_level_fail"]:
        return True
    if include_task_level_fail and ep in info["id_task_level_fail"]:
        return True
    if include_task_level_fail and ep in info["ood_task_level_fail"]:
        return True
    return False


def episode_tag(
    ep: int,
    info: dict[str, list[int]],
    include_task_level_fail: bool,
) -> str:
    if ep in info["id_success"]:
        return "ID success"
    if ep in info["ood_success"]:
        return "OOD success"
    if ep in info["id_state_level_fail"]:
        return "ID state fail"
    if ep in info["ood_state_level_fail"]:
        return "OOD state fail"
    if include_task_level_fail and ep in info["id_task_level_fail"]:
        return "ID task fail"
    if include_task_level_fail and ep in info["ood_task_level_fail"]:
        return "OOD task fail"
    return "unknown"


# ---------------------------------------------------------------------
# Score loading
# ---------------------------------------------------------------------

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(score_npz)
    cfg = load_score_run_config(score_npz)

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
                    f"This NPZ only has a single score stream. "
                    f"Use --single_view_name {score_key} if intended."
                )
            scores = scores_union
        else:
            if scores_view.ndim != 2:
                raise ValueError(f"Expected scores_view shape [N,V], got {scores_view.shape}")

            num_views = scores_view.shape[1]
            if num_views == 1:
                if score_key != single_view_name:
                    raise ValueError(
                        f"This NPZ only has one score view, interpreted as {single_view_name}."
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
                raise ValueError(f"Expected 1 or 2 score views, got {num_views}")

    return (
        np.asarray(data["episode_index"], dtype=np.int64),
        np.asarray(data["anchor_frame_index"], dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        cfg,
    )


def build_fm_wm_episode_score_curves(
    score_npz: str | Path,
    score_key: str,
    single_view_name: str,
    smooth: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], str]:
    history_len, pred_horizon = load_history_and_horizon_from_score(score_npz)
    ep_idx, anchor_idx, scores, run_cfg = load_selected_fm_wm_scores(
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

    mode_name = str(run_cfg.get("proactive_score_mode", run_cfg.get("mode", "score")))
    return curves, mode_name


def load_frame_level_scores(
    score_npz: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(score_npz)
    cfg = load_score_run_config(score_npz)

    required = ["episode_index", "frame_index", "scores"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"Frame-level score NPZ is missing keys {missing}. "
            f"Expected keys: {required}"
        )

    return (
        np.asarray(data["episode_index"], dtype=np.int64),
        np.asarray(data["frame_index"], dtype=np.int64),
        np.asarray(data["scores"], dtype=np.float64),
        cfg,
    )


def build_frame_level_episode_score_curves(
    score_npz: str | Path,
    smooth: int,
    method_name: str,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], str]:
    ep_idx, frame_idx, scores, run_cfg = load_frame_level_scores(score_npz)
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

    if method_name == "faildetect_logpzo":
        mode_name = str(run_cfg.get("score_kind", "faildetect_logpzo"))
    elif method_name == "fiper_rnd":
        mode_name = str(run_cfg.get("score_reduction", "fiper_rnd"))
    else:
        mode_name = method_name

    return curves, mode_name


def normalize_score_method(score_method: str) -> str:
    if score_method == "faildetect_lopzo":
        return "faildetect_logpzo"
    if score_method == "fiper":
        return "fiper_rnd"
    return score_method


def build_episode_score_curves(
    score_npz: str | Path,
    score_key: str,
    single_view_name: str,
    smooth: int,
    score_method: str,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], str]:
    score_method = normalize_score_method(score_method)

    if score_method == "fm_wm":
        return build_fm_wm_episode_score_curves(
            score_npz=score_npz,
            score_key=score_key,
            single_view_name=single_view_name,
            smooth=smooth,
        )

    if score_method in {"faildetect_logpzo", "fiper_rnd"}:
        return build_frame_level_episode_score_curves(
            score_npz=score_npz,
            smooth=smooth,
            method_name=score_method,
        )

    raise ValueError(f"Unsupported score_method={score_method}")


# ---------------------------------------------------------------------
# Threshold helpers
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

    # Split-conformal finite-sample quantile index.
    k = int(np.ceil((n + 1) * (1.0 - alpha)))

    if k > n:
        return float("inf")

    return float(scores[k - 1])


def split_curve_at_threshold(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) == 0:
        return x, y.copy(), y.copy()

    x_aug = [float(x[0])]
    y_aug = [float(y[0])]

    for i in range(1, len(x)):
        x0, x1 = float(x[i - 1]), float(x[i])
        y0, y1 = float(y[i - 1]), float(y[i])

        crosses = (
            np.isfinite(y0)
            and np.isfinite(y1)
            and (y0 - threshold) * (y1 - threshold) < 0.0
            and y1 != y0
        )

        if crosses:
            frac = (threshold - y0) / (y1 - y0)
            x_cross = x0 + frac * (x1 - x0)
            x_aug.append(float(x_cross))
            y_aug.append(float(threshold))

        x_aug.append(x1)
        y_aug.append(y1)

    x_aug = np.asarray(x_aug, dtype=np.float64)
    y_aug = np.asarray(y_aug, dtype=np.float64)

    y_below = y_aug.copy()
    y_above = y_aug.copy()

    y_below[y_aug > threshold] = np.nan
    y_above[y_aug < threshold] = np.nan

    return x_aug, y_below, y_above


def collect_exceeding_episodes(
    score_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_ids: list[int],
    threshold: float,
) -> list[tuple[int, float]]:
    out = []

    for ep in episode_ids:
        if ep not in score_curves:
            continue

        _, y = score_curves[ep]
        if len(y) == 0:
            continue

        peak = float(np.max(y))
        if peak > threshold:
            out.append((int(ep), peak))

    return out


def compute_plot_limits(
    score_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_order: list[int],
    threshold: float | None,
) -> tuple[int, tuple[float, float]]:
    y_vals = []
    x_vals = []

    for ep in episode_order:
        if ep not in score_curves:
            continue

        x, y = score_curves[ep]
        if len(x) == 0 or len(y) == 0:
            continue

        x_vals.append(x)
        y_vals.append(y)

    if not y_vals:
        return 1, (0.0, 1.0)

    x_all = np.concatenate(x_vals)
    y_all = np.concatenate(y_vals)

    x_max = int(np.max(x_all)) if len(x_all) > 0 else 1
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))

    if threshold is not None and np.isfinite(threshold):
        y_min = min(y_min, float(threshold))
        y_max = max(y_max, float(threshold))

    if y_max <= y_min:
        y_pad = 1.0 if y_max == 0 else 0.1 * abs(y_max)
    else:
        y_pad = 0.08 * (y_max - y_min)

    return x_max, (y_min - y_pad, y_max + y_pad)


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_camera_views(
    out_dir: Path,
    task: str,
    ep: int,
    frame: int,
    img_fix: np.ndarray,
    img_wrist: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{safe_name(task)}_ep{int(ep):03d}_frame{int(frame):05d}"

    fix_path = out_dir / f"{stem}_fix.png"
    wrist_path = out_dir / f"{stem}_wrist.png"

    plt.imsave(fix_path, img_fix)
    plt.imsave(wrist_path, img_wrist)

    print(f"[SAVE] {fix_path}")
    print(f"[SAVE] {wrist_path}")


def parse_mark_frames(values: list[str] | None) -> list[int]:
    if not values:
        return []
    frames = []
    for v in values:
        for p in str(v).replace(",", " ").split():
            frames.append(int(p))
    return frames


def int_to_letters(i: int) -> str:
    out = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        out = chr(ord("a") + r) + out
    return out


def make_mark_labels(n: int, style: str) -> list[str]:
    if style == "none":
        return [""] * n
    if style == "letters":
        return [f"({int_to_letters(i)})" for i in range(n)]
    return [f"({i + 1})" for i in range(n)]


def select_first_failure_episode(
    episode_order: list[int],
    score_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    info: dict[str, list[int]],
    include_task_level_fail: bool,
) -> int | None:
    for ep in episode_order:
        if ep in score_curves and is_failure_episode(ep, info, include_task_level_fail):
            x, y = score_curves[ep]
            if len(x) > 0 and len(y) > 0:
                return int(ep)
    return None


def draw_mark_points(
    ax,
    score_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    mark_episode: int | None,
    mark_frames: list[int],
    mark_label_style: str,
) -> None:
    if mark_episode is None or not mark_frames:
        return

    if mark_episode not in score_curves:
        print(f"[MARK][WARN] episode {mark_episode} not found in score curves.")
        return

    x, y = score_curves[mark_episode]
    if len(x) == 0 or len(y) == 0:
        return

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_unique, idx_unique = np.unique(x, return_index=True)
    y_unique = y[idx_unique]

    valid_x = []
    valid_y = []

    for f in mark_frames:
        if f < x_unique[0] or f > x_unique[-1]:
            print(f"[MARK][WARN] frame {f} outside [{int(x_unique[0])}, {int(x_unique[-1])}], skipped.")
            continue
        valid_x.append(float(f))
        valid_y.append(float(np.interp(float(f), x_unique, y_unique)))

    labels = make_mark_labels(len(valid_x), mark_label_style)

    for xm, ym, label in zip(valid_x, valid_y, labels):
        ax.plot(
            [xm],
            [ym],
            marker="o",
            linestyle="None",
            markersize=5.0,
            markerfacecolor=FAIL_ABOVE_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=50,
        )
        if label:
            ax.annotate(
                label,
                xy=(xm, ym),
                xytext=(4, 4),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                color=FAIL_ABOVE_COLOR,
                zorder=51,
            )


def save_paper_score_figure(
    out_dir: Path,
    task: str,
    score_label: str,
    score_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_order: list[int],
    info: dict[str, list[int]],
    include_task_level_fail: bool,
    threshold: float | None,
    y_lim: tuple[float, float],
    x_max: int,
    dpi: int,
    figsize: tuple[float, float],
    mark_episode: int | None = None,
    mark_frames: list[int] | None = None,
    mark_label_style: str = "numbers",
    hide_legend: bool = False
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for ep in episode_order:
        if ep not in score_curves:
            continue

        x, y = score_curves[ep]
        if len(x) == 0:
            continue

        failure = is_failure_episode(ep, info, include_task_level_fail)

        if failure:
            if threshold is not None and np.isfinite(threshold):
                x_split, y_below, y_above = split_curve_at_threshold(x, y, threshold)

                ax.plot(
                    x_split,
                    y_below,
                    color=FAIL_COLOR,
                    linestyle=FAIL_LS,
                    linewidth=FAIL_LW,
                    alpha=0.70,
                    zorder=3,
                )
                ax.plot(
                    x_split,
                    y_above,
                    color=FAIL_ABOVE_COLOR,
                    linestyle=FAIL_LS,
                    linewidth=FAIL_ABOVE_LW,
                    alpha=0.95,
                    zorder=5,
                )
            else:
                ax.plot(
                    x,
                    y,
                    color=FAIL_COLOR,
                    linestyle=FAIL_LS,
                    linewidth=FAIL_LW,
                    alpha=0.75,
                    zorder=3,
                )

        elif is_success_episode(ep, info):
            ax.plot(
                x,
                y,
                color=SUCCESS_COLOR,
                linestyle=SUCCESS_LS,
                linewidth=SUCCESS_LW,
                alpha=0.55,
                zorder=2,
            )

    if threshold is not None and np.isfinite(threshold):
        ax.axhline(
            threshold,
            color=THRESHOLD_COLOR,
            linestyle=THRESHOLD_LS,
            linewidth=THRESHOLD_LW,
            alpha=0.9,
            zorder=4,
        )

    # Keep schematic-style axes: no numeric tick labels.
    # Hide tick marks and tick labels.
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

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    ax.grid(True, linewidth=0.35, alpha=0.20)

    draw_mark_points(
        ax=ax,
        score_curves=score_curves,
        mark_episode=mark_episode,
        mark_frames=mark_frames or [],
        mark_label_style=mark_label_style,
    )

    def draw_handle_only_legend(ax, threshold: float | None) -> None:
        x0 = 0.015
        x1 = 0.125
        y0 = 0.92
        dy = 0.13

        specs = [
            (SUCCESS_COLOR, SUCCESS_LS, 1.4),
            (FAIL_COLOR, FAIL_LS, 1.4),
            (FAIL_ABOVE_COLOR, FAIL_LS, 1.8),
        ]

        if threshold is not None and np.isfinite(threshold):
            specs.append((THRESHOLD_COLOR, THRESHOLD_LS, 1.4))

        for i, (color, linestyle, linewidth) in enumerate(specs):
            y = y0 - i * dy
            ax.plot(
                [x0, x1],
                [y, y],
                transform=ax.transAxes,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                solid_capstyle="butt",
                clip_on=False,
                zorder=100,
            )

    if not hide_legend:
        draw_handle_only_legend(ax, threshold)

    fig.tight_layout(pad=0.15)

    png_path = out_dir / f"{safe_name(task)}_score_curves_paper.png"
    pdf_path = out_dir / f"{safe_name(task)}_score_curves_paper.pdf"
    svg_path = out_dir / f"{safe_name(task)}_score_curves_paper.svg"

    fig.savefig(
        png_path,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    fig.savefig(
        pdf_path,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    fig.savefig(
        svg_path,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close(fig)

    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {pdf_path}")
    print(f"[SAVE] {svg_path}")

    return png_path, pdf_path


# ---------------------------------------------------------------------
# Replay UI
# ---------------------------------------------------------------------

class PlayerState:
    def __init__(self) -> None:
        self.paused = False
        self.quit = False
        self.step_delta = 0

        self.current_ep: int | None = None
        self.current_frame: int | None = None
        self.current_fix: np.ndarray | None = None
        self.current_wrist: np.ndarray | None = None


def init_replay_score_axis(
    ax,
    score_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_order: list[int],
    info: dict[str, list[int]],
    include_task_level_fail: bool,
    threshold: float | None,
    x_max: int,
    y_lim: tuple[float, float],
    score_label: str,
) -> None:
    # Replay background: successful rollouts only.
    for ep in episode_order:
        if ep not in score_curves:
            continue
        if not is_success_episode(ep, info):
            continue

        x, y = score_curves[ep]
        if len(x) == 0:
            continue

        ax.plot(
            x,
            y,
            color=SUCCESS_COLOR,
            linestyle=SUCCESS_LS,
            linewidth=SUCCESS_LW,
            alpha=0.32,
            zorder=1,
        )

    if threshold is not None and np.isfinite(threshold):
        ax.axhline(
            threshold,
            color=THRESHOLD_COLOR,
            linestyle=THRESHOLD_LS,
            linewidth=THRESHOLD_LW,
            alpha=0.9,
            zorder=4,
        )

    ax.set_xlim(0, max(1, x_max))
    ax.set_ylim(*y_lim)
    ax.set_xlabel("Frame index")
    ax.set_ylabel(score_label)
    ax.grid(True, linewidth=0.4, alpha=0.25)


def run_replay_ui(
    ds,
    task: str,
    save_dir: Path,
    score_label: str,
    score_mode_name: str,
    score_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    replay_episode_order: list[int],
    plot_episode_order: list[int],
    info: dict[str, list[int]],
    include_task_level_fail: bool,
    threshold: float | None,
    x_max: int,
    y_lim: tuple[float, float],
    fps: float,
    speed: float,
    wall_time: bool,
    episode_pause_s: float,
) -> None:
    if len(replay_episode_order) == 0:
        print("[REPLAY] No episodes to replay.")
        return

    plt.ion()

    fig = plt.figure(figsize=(7.0, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.05], hspace=0.18)

    ax_img = fig.add_subplot(gs[0])
    ax_score = fig.add_subplot(gs[1])

    fig.canvas.manager.set_window_title(f"Replay+PaperScore: {task}")

    ax_img.axis("off")

    init_replay_score_axis(
        ax=ax_score,
        score_curves=score_curves,
        episode_order=plot_episode_order,
        info=info,
        include_task_level_fail=include_task_level_fail,
        threshold=threshold,
        x_max=x_max,
        y_lim=y_lim,
        score_label=score_label,
    )

    state = PlayerState()

    def on_key(event):
        if event.key == " ":
            state.paused = not state.paused
        elif event.key in ("q", "escape"):
            state.quit = True
        elif event.key == "left":
            if state.paused:
                state.step_delta = -1
        elif event.key == "right":
            if state.paused:
                state.step_delta = +1
        elif event.key == "s":
            if (
                state.current_ep is not None
                and state.current_frame is not None
                and state.current_fix is not None
                and state.current_wrist is not None
            ):
                save_camera_views(
                    out_dir=save_dir,
                    task=task,
                    ep=state.current_ep,
                    frame=state.current_frame,
                    img_fix=state.current_fix,
                    img_wrist=state.current_wrist,
                )
            else:
                print("[SAVE] No current frame available.")

    def on_close(_event):
        state.quit = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    ep0 = replay_episode_order[0]
    idx0 = get_episode_indices(ds, ep0)
    row0 = ds[int(idx0[0])]

    img_fix0 = chw_u8_to_hwc_u8(row0["observation.images.fix_camera"])
    img_wrist0 = chw_u8_to_hwc_u8(row0["observation.images.wrist_camera"])
    im = ax_img.imshow(stack_side_by_side(img_fix0, img_wrist0))

    current_vline = ax_score.axvline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.75, zorder=10)
    current_dot, = ax_score.plot([], [], marker="o", color="black", linestyle="None", markersize=4.5, zorder=11)
    current_line_success, = ax_score.plot(
        [], [],
        color=SUCCESS_COLOR,
        linestyle=SUCCESS_LS,
        linewidth=1.05,
        alpha=0.95,
        zorder=9,
    )

    current_line_fail_below, = ax_score.plot(
        [], [],
        color=FAIL_COLOR,
        linestyle=FAIL_LS,
        linewidth=1.05,
        alpha=0.95,
        zorder=9,
    )

    current_line_fail_above, = ax_score.plot(
        [], [],
        color=FAIL_ABOVE_COLOR,
        linestyle=FAIL_LS,
        linewidth=1.65,
        alpha=0.98,
        zorder=10,
    )

    print("[REPLAY] Controls: space=pause/resume, left/right=step when paused, s=save current fix/wrist views, q=quit")

    for ep in replay_episode_order:
        if state.quit or not plt.fignum_exists(fig.number):
            break

        if not is_success_episode(ep, info) and not is_failure_episode(ep, info, include_task_level_fail):
            continue

        idx = get_episode_indices(ds, ep)
        x_score, y_score = score_curves.get(
            ep,
            (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)),
        )

        tag = episode_tag(ep, info, include_task_level_fail)
        dt = 1.0 / float(fps)
        next_t = time.time()

        ts_prev = None
        k = 0

        while True:
            if state.quit or not plt.fignum_exists(fig.number):
                break

            if state.paused:
                if state.step_delta != 0:
                    k = int(np.clip(k + state.step_delta, -1, len(idx)))
                    if k == -1:
                        k = len(idx) - 1
                    elif k == len(idx):
                        k = 0
                    state.step_delta = 0
                else:
                    plt.pause(0.03)
                    continue
            else:
                now = time.time()
                if now < next_t:
                    plt.pause(min(0.01, next_t - now))
                    continue

            row = ds[int(idx[k])]

            if wall_time:
                ts = float(row["timestamp"])
                if ts_prev is None:
                    ts_prev = ts
                dt_data = max(0.0, ts - ts_prev)
                ts_prev = ts
                next_t = time.time() + dt_data / max(speed, 1e-6)
            else:
                next_t = time.time() + dt / max(speed, 1e-6)

            img_fix = chw_u8_to_hwc_u8(row["observation.images.fix_camera"])
            img_wrist = chw_u8_to_hwc_u8(row["observation.images.wrist_camera"])

            current_frame = int(row["frame_index"]) if "frame_index" in row else int(k)

            state.current_ep = int(ep)
            state.current_frame = int(current_frame)
            state.current_fix = img_fix.copy()
            state.current_wrist = img_wrist.copy()

            im.set_data(stack_side_by_side(img_fix, img_wrist))
            current_vline.set_xdata([current_frame, current_frame])

            if len(x_score) > 0:
                visible = x_score <= current_frame
                x_vis = x_score[visible]
                y_vis = y_score[visible]
            else:
                x_vis = np.asarray([], dtype=np.int64)
                y_vis = np.asarray([], dtype=np.float64)

            current_failure = is_failure_episode(ep, info, include_task_level_fail)

            current_line_success.set_data([], [])
            current_line_fail_below.set_data([], [])
            current_line_fail_above.set_data([], [])

            if current_failure:
                if threshold is not None and np.isfinite(threshold):
                    x_split, y_below, y_above = split_curve_at_threshold(x_vis, y_vis, threshold)
                    current_line_fail_below.set_data(x_split, y_below)
                    current_line_fail_above.set_data(x_split, y_above)
                else:
                    current_line_fail_below.set_data(x_vis, y_vis)
            else:
                current_line_success.set_data(x_vis, y_vis)

            if len(x_vis) > 0:
                if current_failure and threshold is not None and np.isfinite(threshold) and y_vis[-1] > threshold:
                    current_dot.set_color(FAIL_ABOVE_COLOR)
                elif current_failure:
                    current_dot.set_color(FAIL_COLOR)
                else:
                    current_dot.set_color(SUCCESS_COLOR)

                current_dot.set_data([x_vis[-1]], [y_vis[-1]])
            else:
                current_dot.set_data([], [])

            ax_img.set_title(
                f"episode {ep} | {tag} | {score_label} | {score_mode_name} | frame {current_frame}",
                fontsize=11,
            )

            fig.canvas.draw_idle()
            plt.pause(0.001)

            if not state.paused and k < len(idx) - 1:
                k += 1
            else:
                if not state.paused:
                    break

        if state.quit or not plt.fignum_exists(fig.number):
            break

        current_line_success.set_data([], [])
        current_line_fail_below.set_data([], [])
        current_line_fail_above.set_data([], [])
        current_dot.set_data([], [])

        pause_end = time.time() + float(episode_pause_s)
        while not state.quit and not state.paused and time.time() < pause_end:
            plt.pause(0.03)

    print("[REPLAY] Finished. Close window or press q.")
    while plt.fignum_exists(fig.number) and not state.quit:
        plt.pause(0.05)

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--prl", action="store_true")
    parser.add_argument("--info_yaml", type=str, required=True)
    parser.add_argument("--score_npz", type=str, required=True)

    parser.add_argument("--score_method", type=str, default="fm_wm",
                        choices=["fm_wm", "faildetect_logpzo", "faildetect_lopzo", "fiper", "fiper_rnd"])
    parser.add_argument("--score_key", type=str, default="wrist", choices=["union", "fix", "wrist"])
    parser.add_argument("--single_view_name", type=str, default="wrist", choices=["fix", "wrist"])

    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--include_task_level_fail", action="store_true")

    parser.add_argument("--conformal_alpha", type=float, default=0.05)
    parser.add_argument("--disable_conformal_threshold", action="store_true")

    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--paper_dpi", type=int, default=1800)
    parser.add_argument("--paper_width", type=float, default=3.38)
    parser.add_argument("--paper_height", type=float, default=1.14)
    parser.add_argument("--no_save_paper_fig", action="store_true")

    parser.add_argument("--no_replay", action="store_true")
    parser.add_argument("--skip_id_success", action="store_true")
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--wall_time", action="store_true")
    parser.add_argument("--episode_pause_s", type=float, default=0.5)

    parser.add_argument("--mark_episode", type=int, default=None)
    parser.add_argument("--mark_frames", nargs="*", default=[])
    parser.add_argument(
        "--mark_label_style",
        type=str,
        default="numbers",
        choices=["numbers", "letters", "none"],
    )

    parser.add_argument("--hide_legend", action="store_true")

    args = parser.parse_args()

    score_method = normalize_score_method(args.score_method)

    if args.root_dir is None:
        root_dir = Path(__file__).resolve().parents[2]
    else:
        root_dir = Path(args.root_dir).expanduser().resolve()

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else Path(args.score_npz).resolve().parent / "paper_figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dpath = resolve_processed_hf_dir(root_dir, args.task, args.prl)
    if not dpath.exists():
        raise FileNotFoundError(f"Processed HF dataset not found: {dpath}")

    ds = load_from_disk(str(dpath))
    if len(ds) == 0:
        raise RuntimeError(f"Dataset is empty: {dpath}")

    info = load_episode_info(args.info_yaml)

    plot_episode_order = build_plot_episode_order(
        info=info,
        include_task_level_fail=args.include_task_level_fail,
    )
    replay_episode_order = build_replay_episode_order(
        info=info,
        include_task_level_fail=args.include_task_level_fail,
        skip_id_success=args.skip_id_success,
    )

    available_eps = set(np.asarray(ds["episode_index"], dtype=np.int64).tolist())

    plot_episode_order = [ep for ep in plot_episode_order if ep in available_eps]
    replay_episode_order = [ep for ep in replay_episode_order if ep in available_eps]

    if len(plot_episode_order) == 0:
        raise RuntimeError("No plot episodes found in dataset after filtering.")
    if len(replay_episode_order) == 0 and not args.no_replay:
        raise RuntimeError("No replay episodes found in dataset after filtering.")

    score_curves, score_mode_name = build_episode_score_curves(
        score_npz=args.score_npz,
        score_key=args.score_key,
        single_view_name=args.single_view_name,
        smooth=args.smooth,
        score_method=score_method,
    )

    if score_method == "fm_wm":
        score_label = f"{args.score_key} score"
    else:
        score_label = f"{score_method} score"

    threshold = None
    if not args.disable_conformal_threshold:
        calib_eps = info["id_success"]
        calib_max_scores = compute_episode_max_scores(score_curves, calib_eps)
        threshold = conformal_threshold_from_episode_maxima(
            episode_max_scores=calib_max_scores,
            alpha=float(args.conformal_alpha),
        )

        print(
            f"[CONFORMAL] alpha={args.conformal_alpha} "
            f"n_calib={len(calib_max_scores)} "
            f"threshold={threshold:.6g}"
        )

        if np.isfinite(threshold):
            exceeded = collect_exceeding_episodes(
                score_curves=score_curves,
                episode_ids=plot_episode_order,
                threshold=threshold,
            )
            print(f"[THRESHOLD] episodes with peak > threshold: {len(exceeded)}/{len(plot_episode_order)}")
            if exceeded:
                print("[THRESHOLD] " + ", ".join([f"{ep}(peak={peak:.6g})" for ep, peak in exceeded]))

    x_max, y_lim = compute_plot_limits(
        score_curves=score_curves,
        episode_order=plot_episode_order,
        threshold=threshold,
    )

    mark_frames = parse_mark_frames(args.mark_frames)
    mark_episode = args.mark_episode

    if mark_frames and mark_episode is None:
        mark_episode = select_first_failure_episode(
            episode_order=plot_episode_order,
            score_curves=score_curves,
            info=info,
            include_task_level_fail=args.include_task_level_fail,
        )
        print(f"[MARK] mark_episode not specified; using first failed episode: {mark_episode}")

    if mark_frames:
        print(f"[MARK] episode={mark_episode}, frames={mark_frames}, labels={args.mark_label_style}")

    print(f"[INFO] task={args.task}")
    print(f"[INFO] score_method={score_method}, score_label={score_label}, mode={score_mode_name}")
    print(f"[INFO] plot episodes={plot_episode_order}")
    print(f"[INFO] replay episodes={replay_episode_order}")
    print(f"[INFO] out_dir={out_dir}")

    if not args.no_save_paper_fig:
        save_paper_score_figure(
            out_dir=out_dir,
            task=args.task,
            score_label=score_label,
            score_curves=score_curves,
            episode_order=plot_episode_order,
            info=info,
            include_task_level_fail=args.include_task_level_fail,
            threshold=threshold,
            y_lim=y_lim,
            x_max=x_max,
            dpi=int(args.paper_dpi),
            figsize=(float(args.paper_width), float(args.paper_height)),
            mark_episode=mark_episode,
            mark_frames=mark_frames,
            mark_label_style=args.mark_label_style,
            hide_legend=args.hide_legend
        )

    if not args.no_replay:
        run_replay_ui(
            ds=ds,
            task=args.task,
            save_dir=out_dir,
            score_label=score_label,
            score_mode_name=score_mode_name,
            score_curves=score_curves,
            replay_episode_order=replay_episode_order,
            plot_episode_order=plot_episode_order,
            info=info,
            include_task_level_fail=args.include_task_level_fail,
            threshold=threshold,
            x_max=x_max,
            y_lim=y_lim,
            fps=float(args.fps),
            speed=float(args.speed),
            wall_time=bool(args.wall_time),
            episode_pause_s=float(args.episode_pause_s),
        )


if __name__ == "__main__":
    main()