"""
Example usage:
  python -m models.fiper_rnd.plot_scores \
    --npz ../fomofd/tasks/needle_pickup/prl_data/score_result/fiper_rnd/frame_scores.npz \
    --out ../fomofd/tasks/needle_pickup/prl_data/score_result/fiper_rnd/frame_scores.png \
    --smooth 7 \
    --info_yaml ../fomofd/tasks/needle_pickup/prl_data/score_result/info.yaml

  python -m models.fiper_rnd.plot_scores \
    --npz ../fomofd/tasks/ring_over_post/prl_data/score_result/fiper_rnd/frame_scores.npz \
    --out ../fomofd/tasks/ring_over_post/prl_data/score_result/fiper_rnd/frame_scores.png \
    --smooth 7 \
    --info_yaml ../fomofd/tasks/ring_over_post/prl_data/score_result/info.yaml

Fallback without info_yaml:
  python -m models.fiper_rnd.plot_scores \
    --npz ../fomofd/tasks/needle_pickup/prl_data/score_result/fiper_rnd/frame_scores.npz \
    --out ../fomofd/tasks/needle_pickup/prl_data/score_result/fiper_rnd/frame_scores.png \
    --smooth 7 \
    --failed_episodes "[0, 1, 4, 6, 9]"
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Please `pip install pyyaml`.") from exc


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return x.astype(np.float64, copy=False)

    kernel = np.ones(w, dtype=np.float64)
    num = np.convolve(x.astype(np.float64), kernel, mode="same")
    den = np.convolve(np.ones_like(x, dtype=np.float64), kernel, mode="same")
    return num / np.maximum(den, 1e-12)


def parse_failed_episodes(text: str) -> set[int]:
    if text is None or text.strip() == "":
        return set()

    try:
        value = ast.literal_eval(text)
    except Exception as e:
        raise ValueError(
            f"Could not parse --failed_episodes={text!r}. "
            f"Use a Python-style list like '[0,5,6,7]'."
        ) from e

    if not isinstance(value, (list, tuple, set)):
        raise ValueError("--failed_episodes must parse to a list/tuple/set of ints.")

    return {int(v) for v in value}


def _parse_episode_ids(x: Any) -> set[int]:
    if x is None:
        return set()

    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return set()
        return {int(v) for v in x.split()}

    if isinstance(x, (list, tuple, set)):
        return {int(v) for v in x}

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


def load_episode_info(info_yaml: str | Path) -> dict[str, set[int]]:
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


def get_episode_style(
    ep_int: int,
    episode_info: dict[str, set[int]] | None,
    failed_episodes: set[int],
    include_task_level_fail: bool,
    inspect_episode: int | None,
) -> dict[str, Any]:
    if inspect_episode is not None and ep_int == inspect_episode:
        return {
            "plot": True,
            "color": "purple",
            "linestyle": "-",
            "linewidth": 2.8,
            "alpha": 1.0,
            "zorder": 6,
            "label": "inspect",
        }

    if episode_info is None:
        is_failed = ep_int in failed_episodes
        return {
            "plot": True,
            "color": "red" if is_failed else "blue",
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.8,
            "zorder": 3 if is_failed else 2,
            "label": "failed" if is_failed else "success",
        }

    if ep_int in episode_info["id_success"]:
        return {
            "plot": True,
            "color": "blue",
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.8,
            "zorder": 2,
            "label": "ID success",
        }

    if ep_int in episode_info["id_state_level_fail"]:
        return {
            "plot": True,
            "color": "red",
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.8,
            "zorder": 3,
            "label": "ID state fail",
        }

    if ep_int in episode_info["ood_success"]:
        return {
            "plot": True,
            "color": "blue",
            "linestyle": "--",
            "linewidth": 1.8,
            "alpha": 0.8,
            "zorder": 2,
            "label": "OOD success",
        }

    if ep_int in episode_info["ood_state_level_fail"]:
        return {
            "plot": True,
            "color": "red",
            "linestyle": "--",
            "linewidth": 1.8,
            "alpha": 0.8,
            "zorder": 3,
            "label": "OOD state fail",
        }

    if ep_int in episode_info["id_task_level_fail"]:
        if include_task_level_fail:
            return {
                "plot": True,
                "color": "gray",
                "linestyle": "-",
                "linewidth": 1.6,
                "alpha": 0.7,
                "zorder": 1,
                "label": "ID task fail",
            }
        return {"plot": False}

    if ep_int in episode_info["ood_task_level_fail"]:
        if include_task_level_fail:
            return {
                "plot": True,
                "color": "gray",
                "linestyle": "--",
                "linewidth": 1.6,
                "alpha": 0.7,
                "zorder": 1,
                "label": "OOD task fail",
            }
        return {"plot": False}

    return {"plot": False}


def plot_one_axis(
    ax,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    scores: np.ndarray,
    failed_episodes: set[int],
    episode_info: dict[str, set[int]] | None,
    include_task_level_fail: bool,
    inspect_episode: int | None,
    smooth: int,
    ylabel: str,
):
    unique_eps = np.unique(episode_index)
    used_labels: set[str] = set()

    for ep in unique_eps:
        ep_int = int(ep)

        style = get_episode_style(
            ep_int=ep_int,
            episode_info=episode_info,
            failed_episodes=failed_episodes,
            include_task_level_fail=include_task_level_fail,
            inspect_episode=inspect_episode,
        )

        if not style.get("plot", False):
            continue

        mask = episode_index == ep

        x = frame_index[mask].astype(np.int64)
        y = scores[mask].astype(np.float64)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if len(x) == 0:
            continue

        y_s = moving_average(y, smooth)

        label = style["label"]
        if label in used_labels:
            label = None
        else:
            used_labels.add(style["label"])

        ax.plot(
            x,
            y_s,
            linewidth=style["linewidth"],
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=style["alpha"],
            label=label,
            zorder=style["zorder"],
        )

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if used_labels:
        ax.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument(
        "--failed_episodes",
        type=str,
        default="[]",
        help="Fallback when --info_yaml is not provided. Python-style list like '[0,5,6,7]'.",
    )
    parser.add_argument("--info_yaml", type=str, default=None)
    parser.add_argument("--include_task_level_fail", action="store_true")
    parser.add_argument(
        "--inspect_episode",
        type=int,
        default=None,
        help="Episode id to highlight in purple.",
    )
    parser.add_argument(
        "--score_key",
        type=str,
        default="scores",
        choices=["scores"],
    )
    args = parser.parse_args()

    failed_episodes = parse_failed_episodes(args.failed_episodes)
    episode_info = load_episode_info(args.info_yaml) if args.info_yaml is not None else None

    data = np.load(args.npz)
    episode_index = data["episode_index"]
    frame_index = data["frame_index"]

    if args.score_key not in data:
        raise KeyError(
            f"{args.score_key!r} not found in NPZ. "
            f"Available keys: {list(data.keys())}"
        )

    values = data[args.score_key]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_one_axis(
        ax=ax,
        episode_index=episode_index,
        frame_index=frame_index,
        scores=values,
        failed_episodes=failed_episodes,
        episode_info=episode_info,
        include_task_level_fail=args.include_task_level_fail,
        inspect_episode=args.inspect_episode,
        smooth=args.smooth,
        ylabel=args.score_key,
    )

    ax.set_xlabel("frame index")
    ax.set_title(f"FIPER RND {args.score_key} by rollout")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"saved plot to: {out_path}")


if __name__ == "__main__":
    main()