from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["svg.fonttype"] = "none"

K = np.asarray([1, 2, 3, 4], dtype=float)


def pct(num, den):
    return 100.0 * np.asarray(num, dtype=float) / float(den)


TASKS = {
    "needle_pickup": {
        "fdr_fix": pct([84, 65, 104, 57], 120),
        "fdr_wrist": pct([98, 103, 109, 116], 120),
        "far_fix": pct([1, 2, 2, 2], 20),
        "far_wrist": pct([0, 0, 0, 0], 20),
    },
    "ring_over_post": {
        "fdr_fix": pct([30, 40, 38, 26], 120),
        "fdr_wrist": pct([113, 112, 107, 115], 120),
        "far_fix": pct([2, 0, 0, 0], 20),
        "far_wrist": pct([6, 3, 2, 1], 20),
    },
    "tissue_retraction": {
        "fdr_fix": pct([35, 39, 38, 38], 40),
        "fdr_wrist": pct([38, 40, 40, 40], 40),
        "far_fix": pct([0, 0, 0, 0], 20),
        "far_wrist": pct([0, 0, 0, 0], 20),
    },
    "shunt_insertion": {
        "fdr_fix": pct([6, 21, 15, 26], 40),
        "fdr_wrist": pct([30, 29, 34, 38], 40),
        "far_fix": pct([0, 1, 0, 6], 20),
        "far_wrist": pct([0, 0, 0, 0], 20),
    },
}

TOTAL = {
    "fdr_fix": pct([155, 165, 195, 147], 320),
    "fdr_wrist": pct([279, 284, 290, 309], 320),
    "far_fix": pct([3, 3, 2, 8], 80),
    "far_wrist": pct([6, 3, 2, 1], 80),
}

# Muted, paper-friendly colors
FIX_COLOR = "#2F6B9A"      # muted deep blue
WRIST_COLOR = "#B23A48"    # muted rose red

FIX_LS = (0, (4.0, 2.2))
WRIST_LS = "-"

FIX_MARKER = "s"
WRIST_MARKER = "o"


def style_axis(ax, ylim, with_text: bool, metric: str, is_bottom: bool):
    ax.set_xlim(0.85, 4.15)
    ax.set_ylim(*ylim)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))

    ax.grid(True, linewidth=0.45, alpha=0.25)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    ax.tick_params(axis="both", width=0.9, length=3.5)

    if with_text:
        ax.set_ylabel("FDR (%)" if metric == "fdr" else "FAR (%)", fontsize=10)
        if is_bottom:
            ax.set_xlabel("Action Conditioning Modes", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def plot_metric(ax, metric: str, ylim: tuple[float, float], show_task_curves: bool,
                with_text: bool, is_bottom: bool):
    if show_task_curves:
        for data in TASKS.values():
            ax.plot(
                K,
                data[f"{metric}_wrist"],
                color=WRIST_COLOR,
                linestyle=WRIST_LS,
                marker=WRIST_MARKER,
                markersize=3.0,
                linewidth=0.8,
                alpha=0.23,
                zorder=1,
            )
            ax.plot(
                K,
                data[f"{metric}_fix"],
                color=FIX_COLOR,
                linestyle=FIX_LS,
                marker=FIX_MARKER,
                markersize=3.0,
                linewidth=0.8,
                alpha=0.23,
                zorder=1,
            )

    wrist_line, = ax.plot(
        K,
        TOTAL[f"{metric}_wrist"],
        color=WRIST_COLOR,
        linestyle=WRIST_LS,
        marker=WRIST_MARKER,
        markersize=5.0,
        linewidth=1.85,
        alpha=0.98,
        zorder=5,
        label="Wrist view",
    )
    fix_line, = ax.plot(
        K,
        TOTAL[f"{metric}_fix"],
        color=FIX_COLOR,
        linestyle=FIX_LS,
        marker=FIX_MARKER,
        markersize=5.0,
        linewidth=1.85,
        alpha=0.98,
        zorder=6,
        label="Fixed view",
    )

    style_axis(ax, ylim, with_text=with_text, metric=metric, is_bottom=is_bottom)

    return fix_line, wrist_line


def draw_handle_only_legend(ax):
    """Draw legend samples only, no labels, at bottom right."""
    x0 = 0.58
    x1 = 0.68
    y0 = 0.11
    dy = 0.14

    specs = [
        (FIX_COLOR, FIX_LS, FIX_MARKER),
        (WRIST_COLOR, WRIST_LS, WRIST_MARKER),
    ]

    for i, (color, linestyle, marker) in enumerate(specs):
        y = y0 + i * dy
        ax.plot(
            [x0, x1],
            [y, y],
            transform=ax.transAxes,
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            marker=marker,
            markersize=4.8,
            markevery=[1],
            clip_on=False,
            zorder=100,
        )


def make_figure(width: float, height: float, show_task_curves: bool,
                fdr_only: bool, with_text: bool):
    if fdr_only:
        fig, ax = plt.subplots(figsize=(width, height))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        handles = plot_metric(
            ax=ax,
            metric="fdr",
            ylim=(0.0, 100.0),
            show_task_curves=show_task_curves,
            with_text=with_text,
            is_bottom=True,
        )

        axes = [ax]
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(width, height),
            sharex=True,
            gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12},
        )
        fig.patch.set_alpha(0.0)

        for ax in axes:
            ax.patch.set_alpha(0.0)

        handles = plot_metric(
            ax=axes[0],
            metric="fdr",
            ylim=(0.0, 100.0),
            show_task_curves=show_task_curves,
            with_text=with_text,
            is_bottom=False,
        )
        plot_metric(
            ax=axes[1],
            metric="far",
            ylim=(0.0, 15.0),
            show_task_curves=show_task_curves,
            with_text=with_text,
            is_bottom=True,
        )

    if with_text:
        axes[0].legend(
            handles=list(handles),
            labels=["Fixed view", "Wrist view"],
            loc="lower right",
            fontsize=9,
            frameon=False,
            handlelength=2.8,
        )
    else:
        draw_handle_only_legend(axes[0])

    fig.tight_layout(pad=0.15)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="action_ablation_clean.svg")
    parser.add_argument("--width", type=float, default=4.8)
    parser.add_argument("--height", type=float, default=3.0)
    parser.add_argument("--show_task_curves", action="store_true")
    parser.add_argument("--fdr_only", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()

    fig_text = make_figure(
        width=args.width,
        height=args.height,
        show_task_curves=args.show_task_curves,
        fdr_only=args.fdr_only,
        with_text=True,
    )
    fig_clean = make_figure(
        width=args.width,
        height=args.height,
        show_task_curves=args.show_task_curves,
        fdr_only=args.fdr_only,
        with_text=False,
    )

    fig_text.canvas.manager.set_window_title("Preview with text")
    fig_clean.canvas.manager.set_window_title("Clean figure to save")

    def on_key(event):
        if event.key == "s":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig_clean.savefig(
                out_path,
                format="svg",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.01,
            )
            print(f"[SAVE] {out_path}")
        elif event.key in {"q", "escape"}:
            plt.close(fig_text)
            plt.close(fig_clean)

    fig_text.canvas.mpl_connect("key_press_event", on_key)
    fig_clean.canvas.mpl_connect("key_press_event", on_key)

    print("[INFO] Two windows opened: text preview + clean figure.")
    print("[INFO] Press s in either window to save the clean SVG. Press q/Esc to close.")
    plt.show()


if __name__ == "__main__":
    main()