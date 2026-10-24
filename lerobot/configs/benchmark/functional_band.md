# Functional conformal benchmark

This is a **FAIL-Detect-style adaptation with an expert-derived template**, not
an exact reproduction. It compares `episode`, `functional_band`, and diagnostic
`wall_time_bin` thresholds on identical verified policy rollouts.
`time_varying` is an alias for the binwise algorithm.

## Data and computation

1. For each frozen checkpoint/view, read the exact `val_episode_ids` in its saved
   `splits.json`, and use its expert dataset under the task's `data/`. The scorer's
   reconstructed selection must match these IDs before inference; every scored
   ID/frame is checked afterwards. ACE uses the saved expert preparation split
   because it has no separately trained detector/validation split. These are
   never validation subsets of `prl_data/`.
2. Fit the template mean using only these expert score trajectories. The time
   coordinate is elapsed **recorded wall time**, including simulation, relative
   to the first prepared observation. The grid contains one-second knots plus
   each expert trajectory's first/last valid scoring time. Interpolate each
   expert only inside its scoring support and average available contributors
   equally. Fill unsupported mean-grid knots by linear interpolation; beyond
   support, hold the template boundary values. No score zero-padding or
   normalization by an evaluation episode's final duration is used.
3. Freeze the template. Default modulation is `g(t)=1`. Optional `robust_mad`
   uses 1.4826 × the median absolute deviation of expert contributors, with the
   fixed positive `scale_floor`; no calibration/test tuning is performed.
4. Use **all original 19 successful ACT calibration rollouts, IDs 0–18**:

   `R_i = max_t((score_i(t) - mu(t)) / g(t))`

   `k = ceil((n_cal + 1) * (1 - alpha))`

   `h = sorted(R)[k - 1]`, or `+infinity` with a warning if `k > n_cal`.

   At `n_cal=19, alpha=0.05`, `h` is the largest calibration residual.
5. Alarm strictly when `score(t) > mu(t) + h*g(t)`, implemented in residual
   space as `(score(t)-mu(t))/g(t) > h`. Each episode contributes one maximum
   residual, regardless of its duration. Late monitoring is not disabled when
   fewer than 19 trajectories reach a time point. Empty/nonfinite score traces
   produce explicit errors/availability entries; no episode is silently dropped.

Constant mean zero and unit modulation recover the existing episode-maximum
threshold. Functional ROC uses **R**, sweeping `h` with the template frozen;
retrospective matched-FAR thresholds remain separate from the deployed band.

The existing paper endpoint restriction and causal all-endpoint latency protocol
are retained. Each gets its own expert template and calibrated multiplier.
Latency is dense offline onset-to-first-alarm delay, not end-to-end online
inference latency. Pre-onset alarms, misses, missing labels and camera-specific
onsets retain their established meanings. Deadline rates at 1/2/5/10 wall seconds
use annotated failures as denominator, retaining misses and pre-onset alarms.

Nominal marginal trajectory coverage assumes exchangeable successful calibration
and test trajectories conditional on the frozen model/template and scoring
protocol. It does **not** require every observed test batch to have FAR ≤ 5%.

## Differences from the reference

[FAIL-Detect, IV-B and Appendix B](https://arxiv.org/html/2503.08558v3#A2) uses
separate successful policy-rollout subsets for the template and band multiplier.
We use saved expert validation for the template to retain all 19 policy
calibration rollouts for the multiplier. Variable-length wall-time interpolation
and boundary hold are also adaptations.

The released
[`FunctionalPredictor.get_one_sided_prediction_band`](https://github.com/CXU-TRI/FAIL-Detect/blob/main/UQ_test/timeseries_cp/methods/functional_predictor.py)
uses `np.quantile(calibration_scores, 1-alpha)`, with NumPy's default linear
interpolation; the finite-sample order-statistic implementation is commented out
in that function. We use the corrected order statistic and explicit infinity
fallback. The code's upper-sided residual is `(score-mean)/g`, which we follow;
we do not use absolute residuals or the lower-band sign appearing in the paper's
Appendix B expression. Its constant modulation is `1/T`; `g=1` differs only by
an overall positive scaling of h when T is fixed, and avoids an evaluation-length
dependency here. Its plotting script uses `>=` for upper-band crossing, whereas
we retain strict `>` (ties do not alarm).

Applicable scorers: WM-NC/PE/PU, VAE and ACT-native logpZO/RND, and ACT ACE,
including already scored K/L variants. Checkpoints, ODE settings, state inclusion,
normalization, score orientation and input representations stay fixed.
Component-calibrated AND/OR fusions lack an independent expert-only raw score
under their existing definition; this comparison does not assign them a joint
functional band or claim an unchanged-alpha OR guarantee. Binary-only outputs
without continuous traces are not applicable. Constant ACE traces remain
constant, rather than being replaced with a different score.

## Commands and artifacts

Run from the repository root using the existing `lerobot` environment:

```bash
# Inspect model and dataset prerequisites.
python lerobot/utils/functional_benchmark.py \
  --output logs/functional_plan --plan-only

# Generate missing expert-validation scores for one task and method.
python lerobot/utils/functional_benchmark.py \
  --output fomofd/tasks/benchmark_results/functional/needle_rnd \
  --variants main --tasks needle_pickup --methods rnd --allow-inference

# All curated scorer/checkpoint variants, one scorer at a time.
python lerobot/utils/functional_benchmark.py \
  --output fomofd/tasks/benchmark_results/functional/comparison \
  --allow-inference
```

Omit `--allow-inference` to require existing expert caches and explicitly report
missing ones. `--resume` resumes score acquisition with the same configuration;
completed report directories cannot be overwritten. Choose a fresh output for
a new analysis. The ordinary unified writer also accepts `--calibrations
functional_band --functional-templates <variant>/templates_manifest.json`, but
the dedicated runner is preferable when reusing archived PRL cache identities.

Expert scores live at `<task>/data/score_result/unified_benchmark/...`. Existing
PRL score caches remain under `<task>/prl_data/score_result/...`.
Each output has separate `<variant>/{episode,wall_time_bin,functional_band}/`
reports, plus frozen templates/manifests, calibration residuals/h, first-alarm
records, per-checkpoint/seed provenance, ODE settings, onset summaries, deadline
rates, ROC/matched-FAR tables, and a combined HTML/Markdown comparison. FAR/FDR
include episode counts and descriptive 95% Wilson intervals.
`availability.json` and `state.json` record input availability and job status.

```bash
PYTHONPATH=lerobot/utils:lerobot/tests CUDA_VISIBLE_DEVICES='' \
  python -m unittest test_functional_conformal test_unified_benchmark
```
