#!/usr/bin/env python3
"""Deployment-style per-frame runtime benchmark for FAIL-Detect logpZO."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.faildetect_logpzo.score import (  # noqa: E402
    get_amp_dtype,
    get_device,
    load_merged_config,
    load_model_from_checkpoint,
)
from utils._benchmark_single_frame_e2e_runtime import (  # noqa: E402
    build_arg_parser,
    run_benchmark,
)


def score_frame(
    model: torch.nn.Module,
    fix_obs: torch.Tensor | None,
    wrist_obs: torch.Tensor | None,
    state_obs: torch.Tensor | None,
) -> torch.Tensor:
    out = model.compute_nonconformity_scores(
        fix_obs=fix_obs,
        wrist_obs=wrist_obs,
        state_obs=state_obs,
    )
    return out["scores"]


def main() -> None:
    parser = build_arg_parser(
        "Benchmark deployment-style DINOv2/VFMF-VAE + FAIL-Detect logpZO latency."
    )
    args = parser.parse_args()

    config, train_config_path, _, _, _ = load_merged_config(
        score_config_path=args.config,
        checkpoint_path=args.checkpoint,
        score_config_key=args.config_key,
        train_config_key=None,
    )
    device = get_device(args.device or config.runtime.device)
    amp_dtype = get_amp_dtype(config.runtime.amp_dtype)
    model = load_model_from_checkpoint(config, args.checkpoint, device=device)

    run_benchmark(
        args=args,
        method="faildetect_logpzo",
        config=config,
        model=model,
        device=device,
        amp_dtype=amp_dtype,
        score_frame=score_frame,
        method_summary={
            "config": args.config,
            "train_config": str(train_config_path),
            "score_kind": config.fm_head.score_kind,
            "score_transport_mode": config.fm_head.score_transport_mode,
            "ode_solver": config.fm_head.ode_solver,
            "ode_rtol": config.fm_head.ode_rtol,
            "ode_atol": config.fm_head.ode_atol,
        },
    )


if __name__ == "__main__":
    main()
