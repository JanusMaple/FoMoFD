# Fused-view models and combined monitors

These are two distinct comparisons:

* **Fused-view architecture:** each task's K4/L4, full-data, seed-0
  `concat_channels` checkpoint, scored with WM-NC, WM-PE and WM-PU. The model
  consumes both views jointly and uses **one joint endpoint score and threshold**.
  WM-NC/WM-PE sum coordinate squared norms/errors across both views; WM-PU
  averages coordinate standard deviations across the concatenated latent.
  No OR/AND alarm combinations are added for these models. The October 10
  per-view results are retained as historical component diagnostics.
* **Combined monitors:** the original separate-branch WM-NC, fixed OR/AND wrist;
  and WM-NC OR/AND native logpZO within each view. Existing dense component
  scores and thresholds are reused.

WM-PU retains 16 samples, one RK4 step, FP32 with high matmul precision and
offline window batch 8. It is a Ward-inspired uncertainty score on FoMo-FD's
WM, not a full Ward reproduction or a claim of integration convergence.
Smoothing stays disabled. Original calibration/test IDs and primary reports
are preserved. Offline reports retain episode and legacy wall-bin calibration;
the timed online comparison uses constant episode thresholds.

Fusion uses `max` (OR) or `min` (AND) of `(score - threshold) / scale`, evaluated
at the **same raw frame**. Alarm comparison is strictly `> 0`. AND does not
latch component alarms from different times. Component scales and thresholds
come only from calibration successes. OR may increase FAR; it does not retain
the individual detector's 5% familywise guarantee. Use the existing matched-FAR
analysis for comparisons at comparable FAR. Functional-band fusion is not
defined by this experiment.

Online method fusion runs both resident encoders/scorers sequentially on a
single selected endpoint, charges both costs, then chooses the latest eligible
window again. View fusion shares one WM call. Raw images are preloaded into CPU
RAM; visual features are computed during replay. Successes and offline misses
are skipped, so online results report conditional latency/retention and misses,
not measured overall online FAR/FDR.

The October 10 queue is under
`logs/benchmarks/2026-10-10/fused_and_combinations/`. Its `prepare.py` creates
local path adapters for imported training configs without changing the source
checkpoints, restores the original WM scorer package by matching every recorded
source hash, and freezes input manifests. The old package loads under a private
Python package name; the fused models use current code. Before timing, each
online model group checks raw/prepared feature and score parity.

Results and frozen inputs live under
`fomofd/tasks/benchmark_results/ablations/2026-10-10_fused_and_combinations/`.
The corrected joint-score report and new online configuration are under
`fomofd/tasks/benchmark_results/ablations/2026-10-11_fused_joint/`.
The corrected offline report reuses the original SHA-256-verified score caches
with `--cache-only --reuse-score-report`; it sums/averages each endpoint BEFORE
recalibrating, never adds separate episode maxima or thresholds. No inference
or retraining is needed. Raw cached component scores remain unchanged.

The new online configuration uses a 2% relative score tolerance (absolute
tolerance remains 1e-4). A 22-window numerical probe reproduced the original
1.10% discrepancy: identical saved inputs differ by 1.007% between batch 1/8
in bfloat16, versus 0.000040% in float32. The largest fresh/cached per-view
discrepancy checked was 1.19%. All twelve joint task/method adapters passed
one normal calibration-window check each. Feature tolerances are unchanged.
This is a sampled numerical validation, not a bound on every window or proof
that near-threshold decisions cannot change. The previous failed run remains
archived; the corrected online plan requires a fresh output directory.

The queue runs four offline task reports, a cached all-task summary, and then
the online suite. It generates `comparison.md` / `comparison.yaml` and registers
the results in the HTML inspector's **Combined monitors** and **Fused-view
model** sections. Raw score caches stay under each task's `prl_data/score_result`.

Monitor from the repository root:

```bash
tail -f logs/benchmarks/2026-10-10/fused_and_combinations/queue.log
```

The detached process survives closing VS Code or its inspection terminal.
Do not launch another GPU job during timed online replay. A failed stage stops
the queue; inspect its error before using `run_sequence.py --resume`.
