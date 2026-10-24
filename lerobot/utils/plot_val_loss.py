"""
Example Usage:
  python utils/plot_val_loss.py --task needle_pickup --model fm_wm --view total
  python utils/plot_val_loss.py --task ring_over_post --model faildetect --view wrist
  python utils/plot_val_loss.py --task shunt_insertion --model fiper --view fix
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


TASKS = {
    "needle_pickup",
    "ring_over_post",
    "tissue_retraction",
    "shunt_insertion",
}

MODELS = {
    "fm_wm",
    "faildetect",
    "fiper",
}

VIEWS = {
    "wrist",
    "fix",
    "total",
}


def get_fomofd_root() -> Path:
    """Resolve the repository relative to this script."""
    return Path(__file__).resolve().parents[2]


def get_history_path(task: str, model: str, view: str) -> Path:
    root = get_fomofd_root()
    task_model_dir = root / "fomofd" / "tasks" / task / "model"

    if model == "fm_wm":
        return (
            task_model_dir
            / "fm_wm"
            / "fm_wm_vfmfvae_z8_both_state_off_k4_l4_action_both_data100_seed0"
            / "history.jsonl"
        )

    if model == "faildetect":
        if view == "total":
            raise ValueError("view='total' is only valid for model='fm_wm'.")

        return (
            task_model_dir
            / "faildetect_logpzo"
            / f"logpzo_vfmfvae_z8_{view}_state_on_data100_seed0"
            / "history.jsonl"
        )

    if model == "fiper":
        if view == "total":
            raise ValueError("view='total' is only valid for model='fm_wm'.")

        return (
            task_model_dir
            / "fiper_rnd"
            / f"rnd_vfmfvae_z8_{view}_state_off_data100_seed0"
            / "history.jsonl"
        )

    raise ValueError(f"Unknown model: {model}")


def get_loss_key(model: str, view: str) -> str:
    if model == "fm_wm":
        if view == "wrist":
            return "flow_wrist_loss"
        if view == "fix":
            return "flow_fix_loss"
        if view == "total":
            return "flow_loss"
        raise ValueError(f"Unknown view: {view}")

    return "loss_total"


def load_jsonl(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_val_row(row: dict) -> bool:
    return row.get("split") == "val" or row.get("kind") == "val"


def extract_val_loss(rows, loss_key: str):
    steps = []
    losses = []

    for row in rows:
        if not is_val_row(row):
            continue
        if loss_key not in row:
            continue

        steps.append(row["step"])
        losses.append(row[loss_key])

    return steps, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--view", required=True, choices=sorted(VIEWS))
    args = parser.parse_args()

    history_path = get_history_path(args.task, args.model, args.view)
    loss_key = get_loss_key(args.model, args.view)

    if not history_path.exists():
        raise FileNotFoundError(f"Cannot find history file:\n{history_path}")

    rows = load_jsonl(history_path)
    steps, losses = extract_val_loss(rows, loss_key)

    if len(steps) == 0:
        raise RuntimeError(
            f"No validation rows with key '{loss_key}' found in:\n{history_path}"
        )

    best_idx = min(range(len(losses)), key=lambda i: losses[i])
    best_step = steps[best_idx]
    best_loss = losses[best_idx]

    print(f"History: {history_path}")
    print(f"Plotting: {loss_key}")
    print(f"Best val {loss_key}: {best_loss:.6f} at step {best_step}")

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", linewidth=1.5, markersize=3)

    plt.scatter([best_step], [best_loss], s=60)
    plt.annotate(
        f"best {best_loss:.4f}\nstep {best_step}",
        xy=(best_step, best_loss),
        xytext=(10, 10),
        textcoords="offset points",
    )

    plt.xlabel("Training step")
    plt.ylabel(f"Validation {loss_key}")
    plt.title(f"{args.task} | {args.model} | {args.view}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
