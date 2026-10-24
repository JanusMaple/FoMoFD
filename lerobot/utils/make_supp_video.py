from __future__ import annotations

"""
Generate supplementary-video material for FoMo-FD.

Default exported layout has no text overlays:
  fixed-view RGB         | wrist-view RGB
  fixed-view score curve | wrist-view score curve

The script reads replay.yaml, follows the same replay ordering as replay_and_score.py
when --episode_ids is omitted, and writes one MP4 containing all selected episodes.

Example:
  python utils/make_supp_video.py \
    --task shunt_insertion \
    --prl \
    --info_yaml ../fomofd/tasks/shunt_insertion/prl_data/score_result/replay.yaml \
    --score_npz ../fomofd/tasks/shunt_insertion/prl_data/score_result/score_val_predictive/window_scores_predictive.npz \
    --smooth 7 \
    --skip_id_success \
    --out videos/supp/shunt_all_fixed_wrist_scores.mp4

Optional:
  --episode_ids 60             export only selected episode(s)
  --show_text                  add titles, axis labels, and threshold text
  --hide_success_background    hide faint ID-success background curves
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import yaml
from datasets import load_from_disk

try:
    # Works when executed as: python -m utils.make_supp_video
    from utils.replay_and_score import (  # type: ignore
        _chw_u8_to_hwc_u8,
        _get_episode_indices,
        _resolve_processed_hf_dir,
        build_episode_order,
        build_replay_episode_order,
        compute_episode_max_scores,
        conformal_threshold_from_episode_maxima,
        compute_valid_target_range,
        get_episode_style,
        is_failure_episode,
        load_episode_info,
        load_history_and_horizon_from_score,
        load_score_run_config,
        moving_average,
        split_curve_at_threshold,
    )
except ModuleNotFoundError:
    # Works when executed as: python utils/make_supp_video.py
    from replay_and_score import (  # type: ignore
        _chw_u8_to_hwc_u8,
        _get_episode_indices,
        _resolve_processed_hf_dir,
        build_episode_order,
        build_replay_episode_order,
        compute_episode_max_scores,
        conformal_threshold_from_episode_maxima,
        compute_valid_target_range,
        get_episode_style,
        is_failure_episode,
        load_episode_info,
        load_history_and_horizon_from_score,
        load_score_run_config,
        moving_average,
        split_curve_at_threshold,
    )

FPS = 30.0
VIEW_NAMES = ("fix", "wrist")
IMAGE_KEYS = {
    "fix": "observation.images.fix_camera",
    "wrist": "observation.images.wrist_camera",
}


# ----------------------------
# Score helpers
# ----------------------------

def build_fm_wm_two_view_episode_score_curves(
    score_npz: str | Path,
    smooth: int,
) -> tuple[dict[str, dict[int, tuple[np.ndarray, np.ndarray]]], str]:
    """Load fixed/wrist FM-WM scores from a two-view score NPZ.

    Expected NPZ keys:
      episode_index, anchor_frame_index, scores_view

    Convention:
      scores_view[:, 0] = fixed-view score
      scores_view[:, 1] = wrist-view score
    """
    score_npz = Path(score_npz)
    data = np.load(score_npz)

    required = ["episode_index", "anchor_frame_index", "scores_view"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"{score_npz} is missing {missing}. This script expects a two-view FM-WM score file."
        )

    scores_view = np.asarray(data["scores_view"], dtype=np.float64)
    if scores_view.ndim != 2 or scores_view.shape[1] < 2:
        raise ValueError(
            f"Expected scores_view with shape [N, 2] or [N, >=2], got {scores_view.shape}."
        )

    episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    anchor_index = np.asarray(data["anchor_frame_index"], dtype=np.int64)
    history_len, pred_horizon = load_history_and_horizon_from_score(score_npz)
    run_cfg = load_score_run_config(score_npz)

    curves: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        "fix": {},
        "wrist": {},
    }

    for ep in np.unique(episode_index):
        mask = episode_index == int(ep)
        a = anchor_index[mask].astype(np.int64)
        y_fix = scores_view[mask, 0].astype(np.float64)
        y_wrist = scores_view[mask, 1].astype(np.float64)

        order = np.argsort(a)
        a = a[order]
        y_fix = y_fix[order]
        y_wrist = y_wrist[order]

        x = a + pred_horizon
        valid_lo, valid_hi = compute_valid_target_range(a, history_len, pred_horizon)
        keep = (x >= valid_lo) & (x <= valid_hi)
        x = x[keep]

        curves["fix"][int(ep)] = (x, moving_average(y_fix[keep], smooth))
        curves["wrist"][int(ep)] = (x, moving_average(y_wrist[keep], smooth))

    mode_name = str(run_cfg.get("proactive_score_mode", run_cfg.get("mode", "score")))
    return curves, mode_name


def compute_view_thresholds(
    curves_by_view: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    calib_eps: list[int],
    alpha: float,
    disabled: bool,
    quiet: bool,
) -> dict[str, float | None]:
    thresholds: dict[str, float | None] = {"fix": None, "wrist": None}
    if disabled:
        return thresholds

    for view in VIEW_NAMES:
        calib_max = compute_episode_max_scores(curves_by_view[view], calib_eps)
        thresholds[view] = conformal_threshold_from_episode_maxima(calib_max, alpha)
        if not quiet:
            print(
                f"[CONFORMAL:{view}] alpha={alpha} "
                f"n_calib={len(calib_max)} threshold={thresholds[view]:.6g}"
            )
    return thresholds


def compute_replay_and_score_axis_limits(
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    all_plot_episode_order: list[int],
    threshold: float | None,
) -> tuple[int, tuple[float, float]]:
    """Match replay_and_score.py axis-limit logic for one score stream.

    replay_and_score.py computes x/y limits from all_plot_episode_order,
    not only from the replayed episodes. This function intentionally mirrors
    that behavior so each exported score panel looks like the corresponding
    single-view replay window.
    """
    y_vals = []
    x_vals = []
    for ep in all_plot_episode_order:
        if ep in curves:
            x_ep, y_ep = curves[ep]
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

    if threshold is not None and np.isfinite(threshold):
        y0, y1 = y_lim
        y0 = min(y0, float(threshold))
        y1 = max(y1, float(threshold))
        pad = 0.08 * max(1e-9, y1 - y0)
        y_lim = (y0 - pad, y1 + pad)

    return x_max, y_lim


def compute_global_xmax(
    ds,
    episode_ids: list[int],
    curves_by_view: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> int:
    xmax = 1
    for ep in episode_ids:
        try:
            idx = _get_episode_indices(ds, ep)
        except ValueError:
            continue

        frame_vals = []
        for i in idx:
            row = ds[int(i)]
            frame_vals.append(int(row["frame_index"]) if "frame_index" in row else int(row["step_index"]))
        if frame_vals:
            xmax = max(xmax, max(frame_vals))

        for view in VIEW_NAMES:
            if ep in curves_by_view[view]:
                x, _ = curves_by_view[view][ep]
                if len(x) > 0:
                    xmax = max(xmax, int(np.max(x)))
    return xmax


def success_background_episode_ids(info: dict[str, list[int]], include_ood_success: bool) -> list[int]:
    eps = list(info["id_success"])
    if include_ood_success:
        eps.extend(info["ood_success"])
    return list(dict.fromkeys(int(ep) for ep in eps))


# ----------------------------
# Layout helpers
# ----------------------------

def compute_axes_layout(
    fig_w: float,
    fig_h: float,
    image_shape: tuple[int, int],
    content_width: float,
    image_gap: float,
    score_h: float,
    score_gap: float,
    image_score_gap: float,
    bottom: float,
    top: float,
) -> dict[str, list[float]]:
    """Return a tight 2x2 layout.

    Layout:
      fixed RGB  | wrist RGB
      fixed score| wrist score

    Each score axis is directly below its corresponding image axis.  The
    score_gap argument is kept only for CLI backward compatibility; this
    two-column layout uses image_score_gap as the vertical gap between image
    and score rows.
    """
    _ = score_gap  # backward-compatible unused argument

    img_h_px, img_w_px = image_shape[:2]
    image_aspect = float(img_h_px) / max(1.0, float(img_w_px))

    available_h = top - bottom
    if available_h <= image_score_gap + score_h:
        raise ValueError("Invalid layout: no vertical room for image and score axes.")

    img_w_norm = (content_width - image_gap) / 2.0
    img_h_norm = img_w_norm * (fig_w / fig_h) * image_aspect

    max_img_h = available_h - image_score_gap - score_h
    if img_h_norm > max_img_h:
        img_h_norm = max_img_h
        img_w_norm = img_h_norm * (fig_h / fig_w) / max(image_aspect, 1e-12)

    total_w = 2.0 * img_w_norm + image_gap
    left = 0.5 - total_w / 2.0

    block_h = img_h_norm + image_score_gap + score_h
    block_bottom = bottom + max(0.0, available_h - block_h) / 2.0

    score_bottom = block_bottom
    image_bottom = score_bottom + score_h + image_score_gap

    return {
        "fix_img": [left, image_bottom, img_w_norm, img_h_norm],
        "wrist_img": [left + img_w_norm + image_gap, image_bottom, img_w_norm, img_h_norm],
        "fix_score": [left, score_bottom, img_w_norm, score_h],
        "wrist_score": [left + img_w_norm + image_gap, score_bottom, img_w_norm, score_h],
    }


# ----------------------------
# Plot helpers
# ----------------------------

def setup_image_axis(ax, title: str, show_text: bool) -> None:
    if show_text:
        ax.set_title(title, fontsize=11, pad=3)
    ax.axis("off")


def setup_score_axis(
    ax,
    title: str,
    ylabel: str,
    xlim: tuple[int, int],
    ylim: tuple[float, float],
    show_text: bool,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.18, linewidth=0.6)

    for spine in ax.spines.values():
        spine.set_alpha(0.45)
        spine.set_linewidth(0.8)

    if show_text:
        ax.set_title(title, fontsize=10, pad=2)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8, length=3)
    else:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labelleft=False,
            bottom=False,
            left=False,
            length=0,
        )


def draw_threshold(ax, threshold: float | None, show_text: bool) -> None:
    if threshold is None or not np.isfinite(threshold):
        return

    ax.axhline(
        float(threshold),
        color="black",
        linestyle="-.",
        linewidth=1.0,
        alpha=0.45,
        zorder=1,
    )

    if show_text:
        ax.text(
            0.995,
            0.90,
            f"CP threshold: {threshold:.3g}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=7,
            color="black",
            alpha=0.75,
        )


def draw_success_background(
    ax,
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_ids: list[int],
    color: str,
    alpha: float,
    linewidth: float,
) -> None:
    for ep in episode_ids:
        if ep not in curves:
            continue
        x_bg, y_bg = curves[ep]
        if len(x_bg) == 0:
            continue
        ax.plot(
            x_bg,
            y_bg,
            color=color,
            linestyle="--",
            linewidth=linewidth,
            alpha=alpha,
            zorder=0,
        )


def make_score_artists(ax, style: dict[str, Any], failure_ep: bool, threshold: float | None):
    fail_split = failure_ep and threshold is not None and np.isfinite(threshold)
    if fail_split:
        line_below, = ax.plot(
            [],
            [],
            color="tab:orange",
            linestyle=style["linestyle"],
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )
        line_above, = ax.plot(
            [],
            [],
            color="crimson",
            linestyle=style["linestyle"],
            linewidth=2.2,
            alpha=1.0,
            zorder=4,
        )
        line = None
    else:
        line, = ax.plot(
            [],
            [],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )
        line_below = None
        line_above = None

    dot, = ax.plot(
        [],
        [],
        marker="o",
        markersize=4,
        linestyle="None",
        color=style["color"],
        zorder=5,
    )
    vline = ax.axvline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.25, zorder=1)
    return {"line": line, "below": line_below, "above": line_above, "dot": dot, "vline": vline}


def update_score_artists(
    artists: dict[str, Any],
    x_score: np.ndarray,
    y_score: np.ndarray,
    current_frame: int,
    style: dict[str, Any],
    failure_ep: bool,
    threshold: float | None,
) -> None:
    if len(x_score) > 0:
        visible = x_score <= current_frame
        x_vis = x_score[visible]
        y_vis = y_score[visible]
    else:
        x_vis = np.asarray([], dtype=np.float64)
        y_vis = np.asarray([], dtype=np.float64)

    fail_split = failure_ep and threshold is not None and np.isfinite(threshold)
    if fail_split:
        x_split, y_below, y_above = split_curve_at_threshold(x_vis, y_vis, float(threshold))
        artists["below"].set_data(x_split, y_below)
        artists["above"].set_data(x_split, y_above)
    else:
        artists["line"].set_data(x_vis, y_vis)

    if len(x_vis) > 0:
        above = threshold is not None and np.isfinite(threshold) and y_vis[-1] > threshold
        artists["dot"].set_color("crimson" if failure_ep and above else style["color"])
        artists["dot"].set_data([x_vis[-1]], [y_vis[-1]])
    else:
        artists["dot"].set_data([], [])

    artists["vline"].set_xdata([current_frame, current_frame])


def remove_artists(artists: dict[str, Any]) -> None:
    for artist in artists.values():
        if artist is not None:
            artist.remove()


def weaken_finished_artists(artists: dict[str, Any]) -> None:
    for key, artist in artists.items():
        if artist is None:
            continue
        if key in {"dot", "vline"}:
            artist.remove()
        else:
            artist.set_alpha(0.16)
            artist.set_linewidth(0.8)
            artist.set_zorder(0)


# ----------------------------
# CLI / main
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--prl", action="store_true")
    parser.add_argument("--info_yaml", type=str, required=True)
    parser.add_argument("--score_npz", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--episode_ids", type=int, nargs="*", default=None)
    parser.add_argument("--skip_id_success", action="store_true")
    parser.add_argument("--include_task_level_fail", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)

    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--conformal_alpha", type=float, default=0.05)
    parser.add_argument("--disable_conformal_threshold", action="store_true")

    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--episode_pause_frames", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--fig_w", type=float, default=7.2)
    parser.add_argument("--fig_h", type=float, default=4.8)
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--bitrate", type=int, default=5000)
    parser.add_argument("--crf", type=int, default=18)

    # Layout defaults use a tight two-column / two-row canvas: each score panel
    # sits directly below the corresponding camera view, while each score axis
    # keeps the same scaling logic as replay_and_score.py.
    parser.add_argument("--content_width", type=float, default=0.72)
    parser.add_argument("--image_gap", type=float, default=0.010)
    parser.add_argument("--score_h", type=float, default=0.180)
    parser.add_argument("--score_gap", type=float, default=0.000)
    parser.add_argument("--image_score_gap", type=float, default=0.012)
    parser.add_argument("--bottom", type=float, default=0.055)
    parser.add_argument("--top", type=float, default=0.955)

    parser.add_argument(
        "--hide_success_background",
        action="store_true",
        help="Hide faint dashed ID-success background curves.",
    )
    parser.add_argument(
        "--include_ood_success_background",
        action="store_true",
        help="Also include OOD-success episodes as faint background curves.",
    )
    parser.add_argument("--success_bg_alpha", type=float, default=0.34)
    parser.add_argument("--success_bg_linewidth", type=float, default=0.85)
    parser.add_argument("--success_bg_color", type=str, default="cornflowerblue")

    parser.add_argument(
        "--keep_finished_traces",
        action="store_true",
        help="Keep finished replay traces faintly when exporting multiple episodes.",
    )
    parser.add_argument(
        "--show_text",
        action="store_true",
        help="Show view labels, score-panel titles, axis labels, threshold text, and episode title.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title prefix used only when --show_text is enabled.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    if not (0.0 < args.content_width <= 1.0):
        raise ValueError("--content_width must be in (0, 1].")

    root_dir = Path(__file__).resolve().parents[2]
    dpath = _resolve_processed_hf_dir(root_dir, args.task, args.prl)
    if not dpath.exists():
        raise FileNotFoundError(f"Processed dataset dir not found: {dpath}")

    ds = load_from_disk(str(dpath))
    if len(ds) == 0:
        raise RuntimeError(f"Dataset is empty: {dpath}")

    info = load_episode_info(args.info_yaml)
    if args.episode_ids is not None and len(args.episode_ids) > 0:
        episode_order = list(dict.fromkeys(int(ep) for ep in args.episode_ids))
    else:
        episode_order = build_replay_episode_order(
            info,
            include_task_level_fail=args.include_task_level_fail,
            skip_id_success=args.skip_id_success,
        )
    if args.max_episodes is not None:
        episode_order = episode_order[: int(args.max_episodes)]

    available_eps = set(np.asarray(ds["episode_index"], dtype=np.int64).tolist())
    missing = [ep for ep in episode_order if ep not in available_eps]
    if missing and not args.quiet:
        print(f"[WARN] Skipping episodes missing from HF dataset: {missing}")
    episode_order = [ep for ep in episode_order if ep in available_eps]
    if len(episode_order) == 0:
        raise RuntimeError("No requested replay episodes exist in the HF dataset.")

    curves_by_view, score_mode_name = build_fm_wm_two_view_episode_score_curves(
        score_npz=args.score_npz,
        smooth=args.smooth,
    )
    thresholds = compute_view_thresholds(
        curves_by_view=curves_by_view,
        calib_eps=info["id_success"],
        alpha=float(args.conformal_alpha),
        disabled=bool(args.disable_conformal_threshold),
        quiet=bool(args.quiet),
    )

    bg_eps = success_background_episode_ids(
        info,
        include_ood_success=bool(args.include_ood_success_background),
    )
    bg_eps = [ep for ep in bg_eps if ep in available_eps]

    all_plot_episode_order = build_episode_order(
        info,
        include_task_level_fail=args.include_task_level_fail,
    )
    all_plot_episode_order = [ep for ep in all_plot_episode_order if ep in available_eps]

    axis_limits = {
        view: compute_replay_and_score_axis_limits(
            curves_by_view[view],
            all_plot_episode_order,
            thresholds[view],
        )
        for view in VIEW_NAMES
    }
    xlims = {view: (0, axis_limits[view][0]) for view in VIEW_NAMES}
    ylims = {view: axis_limits[view][1] for view in VIEW_NAMES}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize from the first frame so the axes can be sized to the image aspect ratio.
    first_idx = _get_episode_indices(ds, episode_order[0])
    first_row = ds[int(first_idx[0])]
    first_fix = _chw_u8_to_hwc_u8(first_row[IMAGE_KEYS["fix"]])
    first_wrist = _chw_u8_to_hwc_u8(first_row[IMAGE_KEYS["wrist"]])

    fig = plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
    layout = compute_axes_layout(
        fig_w=float(args.fig_w),
        fig_h=float(args.fig_h),
        image_shape=first_fix.shape[:2],
        content_width=float(args.content_width),
        image_gap=float(args.image_gap),
        score_h=float(args.score_h),
        score_gap=float(args.score_gap),
        image_score_gap=float(args.image_score_gap),
        bottom=float(args.bottom),
        top=float(args.top),
    )

    ax_fix_img = fig.add_axes(layout["fix_img"])
    ax_wrist_img = fig.add_axes(layout["wrist_img"])
    ax_fix_score = fig.add_axes(layout["fix_score"])
    ax_wrist_score = fig.add_axes(layout["wrist_score"])

    setup_image_axis(ax_fix_img, "Fixed view", show_text=bool(args.show_text))
    setup_image_axis(ax_wrist_img, "Wrist view", show_text=bool(args.show_text))

    setup_score_axis(
        ax_fix_score,
        "Fixed-view nonconformity score",
        "score",
        xlims["fix"],
        ylims["fix"],
        show_text=bool(args.show_text),
    )
    setup_score_axis(
        ax_wrist_score,
        "Wrist-view nonconformity score",
        "score",
        xlims["wrist"],
        ylims["wrist"],
        show_text=bool(args.show_text),
    )
    if args.show_text:
        ax_fix_score.set_xlabel("frame index", fontsize=9)
        ax_wrist_score.set_xlabel("frame index", fontsize=9)

    for view, ax in [("fix", ax_fix_score), ("wrist", ax_wrist_score)]:
        draw_threshold(ax, thresholds[view], show_text=bool(args.show_text))

    if not args.hide_success_background:
        for view, ax in [("fix", ax_fix_score), ("wrist", ax_wrist_score)]:
            draw_success_background(
                ax=ax,
                curves=curves_by_view[view],
                episode_ids=bg_eps,
                color=str(args.success_bg_color),
                alpha=float(args.success_bg_alpha),
                linewidth=float(args.success_bg_linewidth),
            )

    im_fix = ax_fix_img.imshow(first_fix, aspect="equal")
    im_wrist = ax_wrist_img.imshow(first_wrist, aspect="equal")

    title_artist = None
    if args.show_text:
        title_artist = fig.suptitle("", fontsize=11, y=0.992)

    writer = FFMpegWriter(
        fps=float(args.fps),
        codec=args.codec,
        bitrate=int(args.bitrate),
        metadata={"title": f"FoMo-FD supplementary video: {args.task}"},
        extra_args=["-pix_fmt", "yuv420p", "-crf", str(int(args.crf))],
    )

    if not args.quiet:
        print(f"[VIDEO] Writing {out_path}")
        print(f"[VIDEO] task={args.task} episodes={episode_order} fps={args.fps} stride={args.frame_stride}")
        if not args.hide_success_background:
            print(f"[VIDEO] success background episodes={bg_eps}")

    with writer.saving(fig, str(out_path), dpi=int(args.dpi)):
        for ep_i, ep in enumerate(episode_order, start=1):
            style = get_episode_style(ep, info, include_task_level_fail=args.include_task_level_fail)
            if not style.get("plot", False):
                if not args.quiet:
                    print(f"[WARN] Skipping episode {ep}: not enabled by info_yaml/style.")
                continue

            failure_ep = is_failure_episode(ep, info, args.include_task_level_fail)
            idx = _get_episode_indices(ds, ep)
            idx = idx[:: int(args.frame_stride)]

            artists_fix = make_score_artists(ax_fix_score, style, failure_ep, thresholds["fix"])
            artists_wrist = make_score_artists(ax_wrist_score, style, failure_ep, thresholds["wrist"])

            x_fix, y_fix = curves_by_view["fix"].get(
                ep, (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64))
            )
            x_wrist, y_wrist = curves_by_view["wrist"].get(
                ep, (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64))
            )

            for ds_idx in idx:
                row = ds[int(ds_idx)]
                im_fix.set_data(_chw_u8_to_hwc_u8(row[IMAGE_KEYS["fix"]]))
                im_wrist.set_data(_chw_u8_to_hwc_u8(row[IMAGE_KEYS["wrist"]]))

                current_frame = int(row["frame_index"]) if "frame_index" in row else int(row["step_index"])

                update_score_artists(
                    artists_fix,
                    x_fix,
                    y_fix,
                    current_frame,
                    style,
                    failure_ep,
                    thresholds["fix"],
                )
                update_score_artists(
                    artists_wrist,
                    x_wrist,
                    y_wrist,
                    current_frame,
                    style,
                    failure_ep,
                    thresholds["wrist"],
                )

                if title_artist is not None:
                    prefix = f"{args.title} | " if args.title else ""
                    title_artist.set_text(
                        f"{prefix}{args.task.replace('_', ' ').title()} | "
                        f"episode {ep} ({ep_i}/{len(episode_order)}) | {style['tag']} | "
                        f"{score_mode_name} | frame {current_frame}"
                    )

                writer.grab_frame()

            for _ in range(max(0, int(args.episode_pause_frames))):
                writer.grab_frame()

            if args.keep_finished_traces:
                weaken_finished_artists(artists_fix)
                weaken_finished_artists(artists_wrist)
            else:
                remove_artists(artists_fix)
                remove_artists(artists_wrist)

            if not args.quiet:
                print(f"[VIDEO] Finished episode {ep} ({ep_i}/{len(episode_order)})")

    plt.close(fig)
    if not args.quiet:
        print(f"[VIDEO] Saved: {out_path}")


if __name__ == "__main__":
    main()