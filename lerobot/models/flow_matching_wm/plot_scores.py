"""
Example usage:
  python -m models.flow_matching_wm.plot_scores \
    --config ../fomofd/tasks/needle_pickup/model/fm_wm/fm_wm_vfmfvae_z8_both_state_off_k4_l4_action_both_data100_seed0/flow_matching_wm.yaml \
    --npz_dir ../fomofd/tasks/needle_pickup/prl_data/score_result \
    --out_dir ../fomofd/tasks/needle_pickup/prl_data/score_result \
    --smooth 7 \
    --info_yaml ../fomofd/tasks/needle_pickup/prl_data/score_result/info.yaml \
    --score_mode reactive
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
    """
    Boundary-safe moving average.
    Unlike np.convolve(..., mode='same') with zero padding, this re-normalizes
    by the number of valid samples near the edges.
    """
    if w <= 1 or len(x) == 0:
        return x.astype(np.float64, copy=False)

    kernel = np.ones(w, dtype=np.float64)
    num = np.convolve(x.astype(np.float64), kernel, mode="same")
    den = np.convolve(np.ones_like(x, dtype=np.float64), kernel, mode="same")
    return num / np.maximum(den, 1e-12)


def parse_failed_episodes(text: str):
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


def _extract_node(raw: dict[str, Any], config_key: str | None = None) -> dict[str, Any]:
    if config_key is not None:
        if config_key not in raw:
            raise KeyError(f"config_key='{config_key}' not found in config file.")
        node = raw[config_key]
    elif "fm_wm" in raw:
        node = raw["fm_wm"]
    else:
        node = raw

    if not isinstance(node, dict):
        raise ValueError("Config node must be a dict.")
    return dict(node)


def load_history_and_horizon(config_path: str | Path, config_key: str | None = None) -> tuple[int, int]:
    raw = _load_text_config(config_path)
    node = _extract_node(raw, config_key=config_key)

    data = node.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("Config must contain a 'data' dict.")

    history_len = int(data["history_len"])
    pred_horizon = int(data["pred_horizon"])

    if history_len < 1:
        raise ValueError("history_len must be >= 1")
    if pred_horizon < 1:
        raise ValueError("pred_horizon must be >= 1")

    return history_len, pred_horizon


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


def load_score_run_config(npz_dir: str | Path, score_mode: str) -> dict[str, Any]:
    path = Path(npz_dir) / f"score_val_{score_mode}" / "score_run_config.json"
    if not path.exists():
        return {}
    return _load_text_config(path)


def load_score_arrays(
    data: np.lib.npyio.NpzFile,
    score_key: str,
    single_view_name: str,
    collapse_single_view_joint: bool = False,
):
    """
    Returns a dict[name -> score_array_of_shape_(N,)].

    Supports:
      - old single-score format: scores
      - new format: scores_union, scores_view

    For single-view runs, scores_view has shape [N,1].
    In that case, we label the single column using --single_view_name.
    """
    if "scores_union" not in data and "scores" in data:
        scores_union = data["scores"]
        scores_view = None
    else:
        scores_union = data["scores_union"]
        scores_view = data["scores_view"] if "scores_view" in data else None

    if scores_view is None:
        if score_key in {"fix", "wrist"}:
            if score_key != single_view_name:
                raise ValueError(
                    f"This NPZ only has a single score stream. "
                    f"Use --single_view_name {score_key} if that is the intended view."
                )
            return {score_key: scores_union}

        if score_key == "union":
            return {"union": scores_union}

        if score_key == "joint":
            return {"union": scores_union, single_view_name: scores_union}

        raise ValueError(f"Unsupported score_key: {score_key}")

    if scores_view.ndim != 2:
        raise ValueError(f"Expected scores_view to have shape [N,V], got {scores_view.shape}")

    num_views = scores_view.shape[1]

    if num_views == 1:
        view_dict = {single_view_name: scores_view[:, 0]}
    elif num_views == 2:
        view_dict = {
            "fix": scores_view[:, 0],
            "wrist": scores_view[:, 1],
        }
    else:
        raise ValueError(f"Expected 1 or 2 views, got scores_view.shape[1]={num_views}")

    if score_key == "union":
        return {"union": scores_union}
    if score_key == "fix":
        if "fix" not in view_dict:
            raise ValueError("This NPZ does not contain a fix view.")
        return {"fix": view_dict["fix"]}
    if score_key == "wrist":
        if "wrist" not in view_dict:
            raise ValueError("This NPZ does not contain a wrist view.")
        return {"wrist": view_dict["wrist"]}
    if score_key == "joint":
        if collapse_single_view_joint and num_views == 1:
            return {"union": scores_union}
        out = {"union": scores_union}
        out.update(view_dict)
        return out

    raise ValueError(f"Unsupported score_key: {score_key}")


def compute_valid_target_range(
    anchor_x: np.ndarray,
    history_len: int,
    pred_horizon: int,
) -> tuple[int, int]:
    """
    We plot on the future target-frame axis:
        target_t = anchor_t + pred_horizon

    Dataset construction uses:
        anchor_t in [history_len - 1, ..., episode_len - pred_horizon - 1]

    From the scored anchors, estimate:
        episode_last_frame = max(anchor_t) + pred_horizon

    User-requested plotting policy:
      - drop target times t in {0, 1, ..., history_len - 1}
      - drop the last (pred_horizon - 1) target times

    So keep:
        target_t in [history_len, episode_last_frame - (pred_horizon - 1)]
    """
    if len(anchor_x) == 0:
        return 0, -1

    episode_last_frame = int(anchor_x.max()) + pred_horizon
    valid_lo = int(history_len)
    valid_hi = int(episode_last_frame - (pred_horizon - 1))
    return valid_lo, valid_hi


def get_episode_style(
    ep_int: int,
    info: dict[str, set[int]] | None,
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

    if info is None:
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

    if ep_int in info["id_success"]:
        return {
            "plot": True,
            "color": "blue",
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.8,
            "zorder": 2,
            "label": "ID success",
        }

    if ep_int in info["id_state_level_fail"]:
        return {
            "plot": True,
            "color": "red",
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.8,
            "zorder": 3,
            "label": "ID state fail",
        }

    if ep_int in info["ood_success"]:
        return {
            "plot": True,
            "color": "blue",
            "linestyle": "--",
            "linewidth": 1.8,
            "alpha": 0.8,
            "zorder": 2,
            "label": "OOD success",
        }

    if ep_int in info["ood_state_level_fail"]:
        return {
            "plot": True,
            "color": "red",
            "linestyle": "--",
            "linewidth": 1.8,
            "alpha": 0.8,
            "zorder": 3,
            "label": "OOD state fail",
        }

    if ep_int in info["id_task_level_fail"]:
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

    if ep_int in info["ood_task_level_fail"]:
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
    episode_index,
    anchor_frame_index,
    scores,
    failed_episodes,
    episode_info,
    include_task_level_fail,
    inspect_episode,
    smooth,
    history_len,
    pred_horizon,
    ylabel,
):
    unique_eps = np.unique(episode_index)
    used_labels: set[str] = set()

    for ep in unique_eps:
        ep_int = int(ep)
        style = get_episode_style(
            ep_int=ep_int,
            info=episode_info,
            failed_episodes=failed_episodes,
            include_task_level_fail=include_task_level_fail,
            inspect_episode=inspect_episode,
        )
        if not style.get("plot", False):
            continue

        mask = episode_index == ep

        anchor_x = anchor_frame_index[mask].astype(np.int64)
        y = scores[mask].astype(np.float64)

        order = np.argsort(anchor_x)
        anchor_x = anchor_x[order]
        y = y[order]

        x = anchor_x + pred_horizon

        valid_lo, valid_hi = compute_valid_target_range(
            anchor_x=anchor_x,
            history_len=history_len,
            pred_horizon=pred_horizon,
        )
        keep = (x >= valid_lo) & (x <= valid_hi)

        x = x[keep]
        y = y[keep]

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
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_key", type=str, default=None)
    parser.add_argument("--npz_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
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
        "--score_key",
        type=str,
        default="joint",
        choices=["union", "fix", "wrist", "joint"],
        help="Which score to plot.",
    )
    parser.add_argument(
        "--single_view_name",
        type=str,
        default="fix",
        choices=["fix", "wrist"],
        help="When the NPZ only has one view in scores_view[:,0], interpret it as this view.",
    )
    parser.add_argument(
        "--inspect_episode",
        type=int,
        default=None,
        help="Episode id to highlight in purple.",
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        default="reactive",
        choices=["reactive", "proactive"],
    )
    args = parser.parse_args()

    failed_episodes = parse_failed_episodes(args.failed_episodes)
    episode_info = load_episode_info(args.info_yaml) if args.info_yaml is not None else None
    history_len, pred_horizon = load_history_and_horizon(args.config, args.config_key)

    data = np.load(Path(args.npz_dir) / f"score_val_{args.score_mode}" / f"window_scores_{args.score_mode}.npz")
    score_run_cfg = load_score_run_config(args.npz_dir, args.score_mode)
    proactive_score_mode = score_run_cfg.get("proactive_score_mode", None)
    episode_index = data["episode_index"]
    anchor_frame_index = data["anchor_frame_index"]

    mode_name = args.score_mode

    collapse_single_view_joint = (
        args.score_mode == "proactive"
        and proactive_score_mode == "ace"
    )

    score_dict = load_score_arrays(
        data=data,
        score_key=args.score_key,
        single_view_name=args.single_view_name,
        collapse_single_view_joint=collapse_single_view_joint,
    )

    if args.score_key == "joint":
        n_panels = len(score_dict)
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.5 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        for ax, (name, scores) in zip(axes, score_dict.items()):
            plot_one_axis(
                ax=ax,
                episode_index=episode_index,
                anchor_frame_index=anchor_frame_index,
                scores=scores,
                failed_episodes=failed_episodes,
                episode_info=episode_info,
                include_task_level_fail=args.include_task_level_fail,
                inspect_episode=args.inspect_episode,
                smooth=args.smooth,
                history_len=history_len,
                pred_horizon=pred_horizon,
                ylabel=f"{name} score",
            )

        axes[-1].set_xlabel("frame index")
        fig.suptitle(f"{mode_name} scores by rollout", y=0.995)
        fig.tight_layout()
    else:
        name, scores = next(iter(score_dict.items()))
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_one_axis(
            ax=ax,
            episode_index=episode_index,
            anchor_frame_index=anchor_frame_index,
            scores=scores,
            failed_episodes=failed_episodes,
            episode_info=episode_info,
            include_task_level_fail=args.include_task_level_fail,
            inspect_episode=args.inspect_episode,
            smooth=args.smooth,
            history_len=history_len,
            pred_horizon=pred_horizon,
            ylabel=f"{name} score",
        )
        ax.set_xlabel("frame index")
        ax.set_title(f"{mode_name} {name} scores by rollout")
        fig.tight_layout()

    out_path = Path(args.out_dir) / f"score_val_{mode_name}" / f"{mode_name}_scores.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"saved plot to: {out_path}")


if __name__ == "__main__":
    main()