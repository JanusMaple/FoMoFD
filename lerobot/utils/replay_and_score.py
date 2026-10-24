from __future__ import annotations

"""
Replay PRL HF episodes while progressively drawing score curves.

Example:
 ---Needle Pickup---
  python utils/replay_and_score.py \
    --task needle_pickup \
    --prl \
    --info_yaml ../fomofd/tasks/needle_pickup/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/needle_pickup/prl_data/score_result/score_val_reactive/window_scores_reactive.npz \
    --score_key wrist \
    --single_view_name wrist \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task needle_pickup \
    --prl \
    --info_yaml ../fomofd/tasks/needle_pickup/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/needle_pickup/prl_data/score_result/score_val_predictive/window_scores_predictive.npz \
    --score_key wrist \
    --single_view_name wrist \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task needle_pickup \
    --prl \
    --info_yaml ../fomofd/tasks/needle_pickup/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/needle_pickup/prl_data/score_result/faildetect_logpzo/frame_scores.npz \
    --score_method faildetect_logpzo \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task needle_pickup \
    --prl \
    --info_yaml ../fomofd/tasks/needle_pickup/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/needle_pickup/prl_data/score_result/fiper_rnd/frame_scores.npz \
    --score_method fiper_rnd \
    --smooth 7 \
    --skip_id_success

 ---Ring Over Post---
  python utils/replay_and_score.py \
    --task ring_over_post \
    --prl \
    --info_yaml ../fomofd/tasks/ring_over_post/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/ring_over_post/prl_data/score_result/score_val_reactive/window_scores_reactive.npz \
    --score_key wrist \
    --single_view_name wrist \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task ring_over_post \
    --prl \
    --info_yaml ../fomofd/tasks/ring_over_post/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/ring_over_post/prl_data/score_result/score_val_predictive/window_scores_predictive.npz \
    --score_key wrist \
    --single_view_name wrist \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task ring_over_post \
    --prl \
    --info_yaml ../fomofd/tasks/ring_over_post/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/ring_over_post/prl_data/score_result/faildetect_logpzo/frame_scores.npz \
    --score_method faildetect_logpzo \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task ring_over_post \
    --prl \
    --info_yaml ../fomofd/tasks/ring_over_post/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/ring_over_post/prl_data/score_result/fiper_rnd/frame_scores.npz \
    --score_method fiper_rnd \
    --smooth 7 \
    --skip_id_success

 ---Tissue Retraction---
  python utils/replay_and_score.py \
    --task tissue_retraction \
    --prl \
    --info_yaml ../fomofd/tasks/tissue_retraction/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/tissue_retraction/prl_data/score_result/score_val_reactive/window_scores_reactive.npz \
    --score_key wrist \
    --single_view_name wrist \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task tissue_retraction \
    --prl \
    --info_yaml ../fomofd/tasks/tissue_retraction/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/tissue_retraction/prl_data/score_result/score_val_predictive/window_scores_predictive.npz \
    --score_key wrist \
    --single_view_name wrist \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task tissue_retraction \
    --prl \
    --info_yaml ../fomofd/tasks/tissue_retraction/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/tissue_retraction/prl_data/score_result/faildetect_logpzo/frame_scores.npz \
    --score_method faildetect_logpzo \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task tissue_retraction \
    --prl \
    --info_yaml ../fomofd/tasks/tissue_retraction/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/tissue_retraction/prl_data/score_result/fiper_rnd/frame_scores.npz \
    --score_method fiper_rnd \
    --smooth 7 \
    --skip_id_success

 ---Shunt Insertion---
  python utils/replay_and_score.py \
    --task shunt_insertion \
    --prl \
    --info_yaml ../fomofd/tasks/shunt_insertion/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/shunt_insertion/prl_data/score_result/score_val_reactive/window_scores_reactive.npz \
    --score_key wrist \
    --single_view_name wrist \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task shunt_insertion \
    --prl \
    --info_yaml ../fomofd/tasks/shunt_insertion/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/shunt_insertion/prl_data/score_result/score_val_predictive/window_scores_predictive.npz \
    --score_key wrist \
    --single_view_name wrist \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task shunt_insertion \
    --prl \
    --info_yaml ../fomofd/tasks/shunt_insertion/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/shunt_insertion/prl_data/score_result/faildetect_logpzo/frame_scores.npz \
    --score_method faildetect_logpzo \
    --smooth 7 \
    --skip_id_success

  python utils/replay_and_score.py \
    --task shunt_insertion \
    --prl \
    --info_yaml ../fomofd/tasks/shunt_insertion/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/shunt_insertion/prl_data/score_result/fiper_rnd/frame_scores.npz \
    --score_method fiper_rnd \
    --smooth 7 \
    --skip_id_success

 ---Proactive---
  python utils/replay_and_score.py \
    --task needle_pickup \
    --prl \
    --info_yaml ../fomofd/tasks/needle_pickup/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/needle_pickup/prl_data/score_result/score_val_proactive/window_scores_proactive.npz \
    --score_key wrist \
    --single_view_name wrist \
    --smooth 7 \
    --skip_id_success
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from datasets import load_from_disk

FPS = 30.0


# ----------------------------
# Dataset helpers
# ----------------------------

def _resolve_processed_hf_dir(root_dir: Path, task: str, prl: bool = False) -> Path:
    base = "prl_data" if prl else "data"
    return root_dir / "fomofd" / "tasks" / task / base / "processed_hf"


def _chw_u8_to_hwc_u8(x) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = x.astype(np.uint8, copy=False)

    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[1] > 4 and x.shape[2] > 4:
        return np.transpose(x, (1, 2, 0))
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        return x
    raise ValueError(f"Unexpected image shape {x.shape} (expected CHW or HWC).")


def _resize_nn(img: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
    h, w = img.shape[:2]
    if (h, w) == (h_out, w_out):
        return img
    ys = (np.linspace(0, h - 1, h_out)).astype(np.int64)
    xs = (np.linspace(0, w - 1, w_out)).astype(np.int64)
    return img[ys][:, xs]


def _stack_side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if (ha, wa) != (hb, wb):
        b = _resize_nn(b, ha, wa)
    return np.concatenate([a, b], axis=1)


def _get_episode_indices(ds, episode_idx: int) -> np.ndarray:
    ep = np.asarray(ds["episode_index"], dtype=np.int64)
    idx = np.where(ep == int(episode_idx))[0]
    if idx.size == 0:
        raise ValueError(f"No rows found for episode_idx={episode_idx}")
    steps = np.asarray([ds[int(i)]["step_index"] for i in idx], dtype=np.int64)
    return idx[np.argsort(steps)]


# ----------------------------
# Score/info helpers
# ----------------------------

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return x.astype(np.float64, copy=False)
    kernel = np.ones(w, dtype=np.float64)
    num = np.convolve(x.astype(np.float64), kernel, mode="same")
    den = np.convolve(np.ones_like(x, dtype=np.float64), kernel, mode="same")
    return num / np.maximum(den, 1e-12)


def _parse_episode_ids(x: Any) -> list[int]:
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


def load_episode_info(info_yaml: str | Path) -> dict[str, list[int]]:
    raw = _load_text_config(info_yaml)
    id_node = raw.get("ID", {})
    ood_node = raw.get("OOD", {})
    return {
        "id_success": _parse_episode_ids(id_node.get("success")),
        "id_state_level_fail": _parse_episode_ids(id_node.get("state_level_fail")),
        "id_task_level_fail": _parse_episode_ids(id_node.get("task_level_fail")),
        "ood_success": _parse_episode_ids(ood_node.get("success")),
        "ood_state_level_fail": _parse_episode_ids(ood_node.get("state_level_fail")),
        "ood_task_level_fail": _parse_episode_ids(ood_node.get("task_level_fail")),
    }


def build_episode_order(info: dict[str, list[int]], include_task_level_fail: bool = False) -> list[int]:
    order: list[int] = []
    order.extend(info["id_success"])
    order.extend(info["id_state_level_fail"])
    if include_task_level_fail:
        order.extend(info["id_task_level_fail"])
    order.extend(info["ood_success"])
    order.extend(info["ood_state_level_fail"])
    if include_task_level_fail:
        order.extend(info["ood_task_level_fail"])
    seen = set()
    deduped = []
    for ep in order:
        if ep not in seen:
            seen.add(ep)
            deduped.append(ep)
    return deduped


def dedupe_episode_order(order: list[int]) -> list[int]:
    seen = set()
    deduped = []
    for ep in order:
        if ep not in seen:
            seen.add(ep)
            deduped.append(ep)
    return deduped


def build_replay_episode_order(
    info: dict[str, list[int]],
    include_task_level_fail: bool = False,
    skip_id_success: bool = False,
) -> list[int]:
    order: list[int] = []

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


def get_episode_style(
    ep_int: int,
    info: dict[str, list[int]],
    include_task_level_fail: bool,
) -> dict[str, Any]:
    if ep_int in info["id_success"]:
        return {
            "plot": True,
            "color": "blue",
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.85,
            "zorder": 2,
            "label": "ID success",
            "tag": "ID success",
        }
    if ep_int in info["id_state_level_fail"]:
        return {
            "plot": True,
            "color": "red",
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.85,
            "zorder": 3,
            "label": "ID state fail",
            "tag": "ID state fail",
        }
    if ep_int in info["ood_success"]:
        return {
            "plot": True,
            "color": "blue",
            "linestyle": "--",
            "linewidth": 1.8,
            "alpha": 0.85,
            "zorder": 2,
            "label": "OOD success",
            "tag": "OOD success",
        }
    if ep_int in info["ood_state_level_fail"]:
        return {
            "plot": True,
            "color": "red",
            "linestyle": "--",
            "linewidth": 1.8,
            "alpha": 0.85,
            "zorder": 3,
            "label": "OOD state fail",
            "tag": "OOD state fail",
        }
    if ep_int in info["id_task_level_fail"]:
        return {
            "plot": bool(include_task_level_fail),
            "color": "gray",
            "linestyle": "-",
            "linewidth": 1.6,
            "alpha": 0.7,
            "zorder": 1,
            "label": "ID task fail",
            "tag": "ID task fail",
        }
    if ep_int in info["ood_task_level_fail"]:
        return {
            "plot": bool(include_task_level_fail),
            "color": "gray",
            "linestyle": "--",
            "linewidth": 1.6,
            "alpha": 0.7,
            "zorder": 1,
            "label": "OOD task fail",
            "tag": "OOD task fail",
        }
    return {
        "plot": False,
        "color": "black",
        "linestyle": "-",
        "linewidth": 1.5,
        "alpha": 0.6,
        "zorder": 1,
        "label": "unknown",
        "tag": "unknown",
    }


def load_score_run_config(score_npz: str | Path) -> dict[str, Any]:
    path = Path(score_npz).resolve().parent / "score_run_config.json"
    if not path.exists():
        return {}
    return _load_text_config(path)


def load_merged_config(score_npz: str | Path) -> dict[str, Any]:
    path = Path(score_npz).resolve().parent / "merged_flow_matching_wm_config.json"
    if not path.exists():
        return {}
    return _load_text_config(path)


def load_history_and_horizon_from_score(score_npz: str | Path) -> tuple[int, int]:
    cfg = load_merged_config(score_npz)
    data = cfg.get("data", {})
    if not isinstance(data, dict):
        return 1, 1
    history_len = int(data.get("history_len", 1))
    pred_horizon = int(data.get("pred_horizon", 1))
    return history_len, pred_horizon


def compute_valid_target_range(anchor_x: np.ndarray, history_len: int, pred_horizon: int) -> tuple[int, int]:
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
                    f"This NPZ only has a single score stream. Use --single_view_name {score_key} if intended."
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
    ep_idx, anchor_idx, scores, run_cfg = load_selected_fm_wm_scores(score_npz, score_key, single_view_name)
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

        # Frame-level baselines: no history/prediction-window truncation.
        y = moving_average(y, smooth)
        curves[int(ep)] = (x, y)

    if method_name == "faildetect_logpzo":
        mode_name = str(run_cfg.get("score_kind", "faildetect_logpzo"))
    elif method_name == "fiper_rnd":
        mode_name = str(run_cfg.get("score_reduction", "fiper_rnd"))
    else:
        mode_name = method_name

    return curves, mode_name


def build_episode_score_curves(
    score_npz: str | Path,
    score_key: str,
    single_view_name: str,
    smooth: int,
    score_method: str,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], str]:
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


# ----------------------------
# Threshold Helpers
# ----------------------------

def is_failure_episode(ep: int, info: dict[str, list[int]], include_task_level_fail: bool) -> bool:
    if ep in info["id_state_level_fail"]:
        return True
    if ep in info["ood_state_level_fail"]:
        return True
    if include_task_level_fail and ep in info["id_task_level_fail"]:
        return True
    if include_task_level_fail and ep in info["ood_task_level_fail"]:
        return True
    return False


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


def collect_threshold_exceeding_episodes(
    score_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_ids: list[int],
    threshold: float,
) -> list[tuple[int, float]]:
    exceeded = []
    for ep in episode_ids:
        if ep not in score_curves:
            continue
        _, y = score_curves[ep]
        if len(y) == 0:
            continue
        peak = float(np.max(y))
        if peak > threshold:
            exceeded.append((int(ep), peak))
    return exceeded


def conformal_threshold_from_episode_maxima(
    episode_max_scores: np.ndarray,
    alpha: float,
) -> float:
    if episode_max_scores.size == 0:
        raise ValueError("No calibration episode maxima available.")

    scores = np.sort(episode_max_scores)
    n = len(scores)

    # split-conformal quantile index: ceil((n + 1) * (1 - alpha))
    k = int(np.ceil((n + 1) * (1.0 - alpha)))

    if k > n:
        # Conservative finite-sample behavior would be +inf.
        # For n=19, alpha=0.05, k=19 exactly, so this is not triggered.
        return float("inf")

    return float(scores[k - 1])


def split_curve_at_threshold(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Insert interpolated crossing points where y crosses threshold.

    Returns:
      x_aug
      y_below: values above threshold replaced by nan
      y_above: values below threshold replaced by nan

    The interpolated threshold point is kept in both curves so the color
    transition visually touches the horizontal threshold line.
    """
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


# ----------------------------
# Replay UI
# ----------------------------

class PlayerState:
    def __init__(self):
        self.paused = False
        self.quit = False
        self.step_delta = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--prl", action="store_true")
    parser.add_argument("--info_yaml", type=str, required=True)
    parser.add_argument("--score_npz", type=str, required=True)
    parser.add_argument("--score_key", type=str, default="wrist", choices=["union", "fix", "wrist"])
    parser.add_argument("--single_view_name", type=str, default="wrist", choices=["fix", "wrist"])
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--include_task_level_fail", action="store_true")
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--wall_time", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--episode_pause_s", type=float, default=0.5)
    parser.add_argument(
        "--score_method",
        type=str,
        default="fm_wm",
        choices=["fm_wm", "faildetect_logpzo", "faildetect_lopzo", "fiper", "fiper_rnd"],
        help="Score file format. fm_wm uses window scores; faildetect_logpzo/fiper use frame-level scores.",
    )
    parser.add_argument(
    "--skip_id_success",
        action="store_true",
        help="Draw ID-success curves as background, but start replay from non-ID-success episodes.",
    )
    parser.add_argument(
        "--conformal_alpha",
        type=float, default=0.05
    )
    parser.add_argument(
        "--conformal_calib_group",
        type=str,
        default="id_success",
        choices=["id_success"],
        help="Episodes used to calibrate conformal threshold.",
    )
    parser.add_argument(
        "--disable_conformal_threshold",
        action="store_true",
    )
    parser.add_argument(
        "--show_legend",
        action="store_true",
        help="Show plot legend. Hidden by default.",
    )
    parser.add_argument(
        "--legend_outside",
        action="store_true",
        help="Place legend outside the score axis when --show_legend is enabled.",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    dpath = _resolve_processed_hf_dir(root_dir, args.task, args.prl)
    if not dpath.exists():
        raise FileNotFoundError(f"Processed dataset dir not found: {dpath}")

    ds = load_from_disk(str(dpath))
    if len(ds) == 0:
        raise RuntimeError(f"Dataset is empty: {dpath}")

    info = load_episode_info(args.info_yaml)
    all_plot_episode_order = build_episode_order(
        info,
        include_task_level_fail=args.include_task_level_fail,
    )
    episode_order = build_replay_episode_order(
        info,
        include_task_level_fail=args.include_task_level_fail,
        skip_id_success=args.skip_id_success,
    )
    background_episode_order = info["id_success"] if args.skip_id_success else []

    if len(episode_order) == 0:
        raise RuntimeError("No episodes found in info_yaml after filtering.")

    score_curves, score_mode_name = build_episode_score_curves(
        score_npz=args.score_npz,
        score_key=args.score_key,
        single_view_name=args.single_view_name,
        smooth=args.smooth,
        score_method=args.score_method,
    )

    conformal_threshold = None
    if not args.disable_conformal_threshold:
        calib_eps = info["id_success"]

        calib_max_scores = compute_episode_max_scores(score_curves, calib_eps)
        conformal_threshold = conformal_threshold_from_episode_maxima(
            calib_max_scores,
            alpha=float(args.conformal_alpha),
        )

        print(
            f"[CONFORMAL] alpha={args.conformal_alpha} "
            f"n_calib={len(calib_max_scores)} "
            f"threshold={conformal_threshold:.6g}"
        )

    available_eps = set(np.asarray(ds["episode_index"], dtype=np.int64).tolist())
    missing_in_ds = [ep for ep in all_plot_episode_order if ep not in available_eps]
    if missing_in_ds:
        print(f"[WARN] Skipping episodes missing from HF dataset: {missing_in_ds}")
    all_plot_episode_order = [ep for ep in all_plot_episode_order if ep in available_eps]
    episode_order = [ep for ep in episode_order if ep in available_eps]
    background_episode_order = [ep for ep in background_episode_order if ep in available_eps]
    if len(episode_order) == 0:
        raise RuntimeError("No replay episodes exist in HF dataset.")
    
    if conformal_threshold is not None and np.isfinite(conformal_threshold):
        exceeded = collect_threshold_exceeding_episodes(
            score_curves=score_curves,
            episode_ids=episode_order,
            threshold=conformal_threshold,
        )
        print(
            f"[THRESHOLD] replay episodes with peak > threshold: "
            f"{len(exceeded)}/{len(episode_order)}"
        )
        if exceeded:
            ep_str = ", ".join([f"{ep}(peak={peak:.6g})" for ep, peak in exceeded])
            print(f"[THRESHOLD] exceeded episodes: {ep_str}")

    y_vals = []
    x_vals = []
    for ep in all_plot_episode_order:
        if ep in score_curves:
            x_ep, y_ep = score_curves[ep]
            if len(y_ep) > 0:
                y_vals.append(y_ep)
                x_vals.append(x_ep)
    if y_vals:
        y_all = np.concatenate(y_vals, axis=0)
        x_all = np.concatenate(x_vals, axis=0)
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        x_max = int(np.max(x_all)) if len(x_all) > 0 else 1
        if y_max <= y_min:
            y_pad = 1.0 if y_max == 0 else 0.1 * abs(y_max)
        else:
            y_pad = 0.08 * (y_max - y_min)
        y_lim = (y_min - y_pad, y_max + y_pad)
    else:
        x_max = 1
        y_lim = (0.0, 1.0)

    if conformal_threshold is not None and np.isfinite(conformal_threshold):
        y0, y1 = y_lim
        y0 = min(y0, conformal_threshold)
        y1 = max(y1, conformal_threshold)
        pad = 0.08 * max(1e-9, y1 - y0)
        y_lim = (y0 - pad, y1 + pad)

    print(f"[REPLAY+SCORE] task={args.task} replay_episodes={episode_order}")
    if args.skip_id_success:
        print(f"[REPLAY+SCORE] ID-success background episodes={background_episode_order}")
    print("[REPLAY+SCORE] Controls: space=pause/resume, q=quit, left/right=step when paused")

    plt.ion()
    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.18)
    ax_img = fig.add_subplot(gs[0])
    ax_score = fig.add_subplot(gs[1])
    fig.canvas.manager.set_window_title(f"Replay+Score: {args.task}")

    ax_img.axis("off")
    ax_score.set_xlabel("frame index")
    if args.score_method == "fm_wm":
        score_label = f"{args.score_key} score"
    else:
        score_label = f"{args.score_method} score"
    ax_score.set_ylabel(score_label)
    ax_score.set_xlim(0, max(1, x_max))
    ax_score.set_ylim(*y_lim)
    ax_score.grid(True, alpha=0.3)

    if conformal_threshold is not None and np.isfinite(conformal_threshold):
        ax_score.axhline(
            conformal_threshold,
            color="black",
            linestyle="-.",
            linewidth=1.4,
            alpha=0.5,
            zorder=1,
            label=f"CP threshold α={args.conformal_alpha}",
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

    def on_close(event):
        state.quit = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    used_labels: set[str] = set()
    BG_ALPHA = 0.22
    BG_LINEWIDTH = 1.0
    BG_ZORDER = 0
    FAIL_BELOW_COLOR = "tab:orange"
    FAIL_ABOVE_COLOR = "crimson"

    if args.skip_id_success:
        for ep in background_episode_order:
            if ep not in score_curves:
                continue
            x_bg, y_bg = score_curves[ep]
            if len(x_bg) == 0:
                continue
            label = "ID success background" if "ID success background" not in used_labels else None
            if label is not None:
                used_labels.add(label)
            ax_score.plot(
                x_bg,
                y_bg,
                color="blue",
                linestyle="-",
                linewidth=BG_LINEWIDTH,
                alpha=BG_ALPHA,
                zorder=BG_ZORDER,
                label=label,
            )

    # initial image
    ep0 = episode_order[0]
    idx0 = _get_episode_indices(ds, ep0)
    row0 = ds[int(idx0[0])]
    img_fix0 = _chw_u8_to_hwc_u8(row0["observation.images.fix_camera"])
    img_wrist0 = _chw_u8_to_hwc_u8(row0["observation.images.wrist_camera"])
    im = ax_img.imshow(_stack_side_by_side(img_fix0, img_wrist0))

    for ep in episode_order:
        if state.quit or not plt.fignum_exists(fig.number):
            break

        style = get_episode_style(ep, info, include_task_level_fail=args.include_task_level_fail)
        if not style.get("plot", False):
            continue

        idx = _get_episode_indices(ds, ep)
        x_score, y_score = score_curves.get(ep, (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)))

        label = style["label"] if style["label"] not in used_labels else None
        if label is not None:
            used_labels.add(style["label"])

        failure_ep = is_failure_episode(ep, info, args.include_task_level_fail)

        if failure_ep and conformal_threshold is not None and np.isfinite(conformal_threshold):
            below_label = "failure below threshold" if "failure below threshold" not in used_labels else None
            above_label = "failure above threshold" if "failure above threshold" not in used_labels else None

            if below_label is not None:
                used_labels.add("failure below threshold")
            if above_label is not None:
                used_labels.add("failure above threshold")

            line_below, = ax_score.plot(
                [], [],
                color=FAIL_BELOW_COLOR,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                label=below_label,
            )
            line_above, = ax_score.plot(
                [], [],
                color=FAIL_ABOVE_COLOR,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"] + 0.4,
                alpha=style["alpha"],
                zorder=style["zorder"] + 1,
                label=above_label,
            )
            line = None
        else:
            label = style["label"] if style["label"] not in used_labels else None
            if label is not None:
                used_labels.add(style["label"])

            line, = ax_score.plot(
                [], [],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                label=label,
            )
            line_below = None
            line_above = None
        
        dot, = ax_score.plot(
            [], [],
            marker="o",
            markersize=6,
            linestyle="None",
            color=style["color"],
            zorder=style["zorder"] + 1,
        )
        if args.show_legend and used_labels:
            if args.legend_outside:
                ax_score.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=8,
                    framealpha=0.85,
                )
                fig.subplots_adjust(right=0.78)
            else:
                ax_score.legend(
                    loc="lower right",
                    fontsize=8,
                    framealpha=0.85,
                )

        dt = 1.0 / float(args.fps)
        next_t = time.time()
        t_init = None
        ts_prev = None
        k = 0

        while True:
            if not plt.fignum_exists(fig.number) or state.quit:
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
            ts = float(row["timestamp"])
            if t_init is None:
                t_init = ts
                ts_prev = ts

            if args.wall_time:
                dt_data = max(0.0, ts - (ts_prev if ts_prev is not None else ts))
                ts_prev = ts
                next_t = time.time() + (dt_data / max(args.speed, 1e-6))
            else:
                next_t = time.time() + (dt / max(args.speed, 1e-6))

            img_fix = _chw_u8_to_hwc_u8(row["observation.images.fix_camera"])
            img_wrist = _chw_u8_to_hwc_u8(row["observation.images.wrist_camera"])
            im.set_data(_stack_side_by_side(img_fix, img_wrist))

            current_frame = int(row["frame_index"]) if "frame_index" in row else int(k)
            if len(x_score) > 0:
                visible = x_score <= current_frame
                x_vis = x_score[visible]
                y_vis = y_score[visible]
            else:
                x_vis = np.asarray([], dtype=np.int64)
                y_vis = np.asarray([], dtype=np.float64)

            if failure_ep and conformal_threshold is not None and np.isfinite(conformal_threshold):
                x_split, y_below, y_above = split_curve_at_threshold(
                    x_vis,
                    y_vis,
                    conformal_threshold,
                )

                line_below.set_data(x_split, y_below)
                line_above.set_data(x_split, y_above)
            else:
                line.set_data(x_vis, y_vis)
            if len(x_vis) > 0:
                dot_color = (
                    FAIL_ABOVE_COLOR
                    if failure_ep
                    and conformal_threshold is not None
                    and np.isfinite(conformal_threshold)
                    and y_vis[-1] > conformal_threshold
                    else style["color"]
                )
                dot.set_color(dot_color)
                dot.set_data([x_vis[-1]], [y_vis[-1]])
            else:
                dot.set_data([], [])

            ax_img.set_title(
                f"episode {ep} | {style['tag']} | {score_label} | {score_mode_name} | frame {current_frame}",
                fontsize=12,
            )

            fig.canvas.draw_idle()
            plt.pause(0.001)

            if (not state.paused) and (k < len(idx) - 1):
                k += 1
            else:
                if not state.paused:
                    break

        # keep finished line, remove active dot and weaken line
        dot.remove()
        for finished_line in (line, line_below, line_above):
            if finished_line is None:
                continue
            finished_line.set_alpha(BG_ALPHA)
            finished_line.set_linewidth(BG_LINEWIDTH)
            finished_line.set_zorder(BG_ZORDER)
        fig.canvas.draw_idle()
        plt.pause(0.001)

        if state.quit or not plt.fignum_exists(fig.number):
            break

        pause_end = time.time() + float(args.episode_pause_s)
        while (not state.quit) and (not state.paused) and time.time() < pause_end:
            plt.pause(0.03)

    if plt.fignum_exists(fig.number) and not state.quit:
        print("[REPLAY+SCORE] Finished. Close window or press q to exit.")
        while plt.fignum_exists(fig.number) and not state.quit:
            plt.pause(0.05)

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)


if __name__ == "__main__":
    main()