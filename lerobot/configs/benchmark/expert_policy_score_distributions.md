# Expert-validation versus policy-success score distributions

Run the separate, cache-only main K4/L4 WM-NC experiment from the repo root:

```bash
conda activate fomofd-lerobot
python lerobot/utils/analyze_experiments.py score-distributions \
  --config lerobot/configs/benchmark/expert_policy_score_distributions.yaml \
  --output fomofd/tasks/benchmark_results/analyses/2026-10-20/expert_policy_score_distributions
```

This analysis requires existing score caches and a fresh output directory.
Missing or changed caches cause an error; score generation is a separate step.

The source is the frozen functional-band experiment, which already scored the
main models' exact saved expert validation splits: 25 episodes for each simulation
task and 10 for each real task. Cached raw scores are used without applying the
functional template. Their checkpoint, scoring configuration, batching, score
seed, shared source hashes, validation IDs, and frame/timestamp coverage are
checked against the paired policy cache. Report all 39 policy successes, with
the 19 calibration and 20 test episodes also shown separately.

Three distributions are reported per task and view:

- Pooled window scores: every eligible window has equal weight.
- Episode-balanced window scores: each episode contributes equal total weight.
- Episode maxima: one maximum over all eligible windows per episode.

The report's primary comparison is the **median (IQR)** of episode-maximum raw
WM-NC scores, computed separately for every task, camera, and cohort. Expert
validation, policy calibration (19), policy test-success (20), and their combined
policy-success cohort (39) are all included. Every episode contributes one value.
Q1, median, and Q3 use NumPy's `method="linear"` (index `(n-1)p`); IQR is
`Q3-Q1`. In particular, even-sized cohorts use the midpoint of the two middle
values. These describe between-episode spread, not confidence intervals.

The frozen visual-validity correction excludes Needle Pickup policy episode 5
from raw frame 122 onward in both views. Expert episodes are not truncated by
PRL episode IDs. No other endpoint selection or score transformation is changed.

The figures show normalized probability-density histograms with shared bin edges
across cohorts in each panel. Area under each histogram is one; the probability
in a bin equals its height times its width. Blue/green filled histograms compare
expert validation and all policy successes. Calibration/test subsets have dashed
outlines. There is no KDE or score smoothing. Episode maxima use fewer bins to
reflect the smaller sample counts. Exact bins/densities are in `density_histograms.csv`.

Refresh the HTML/Markdown and plots of the completed experiment using its exported
scores, leaving all numeric results and input provenance unchanged:

```bash
python lerobot/utils/analyze_experiments.py score-distributions --refresh-plots \
  --output fomofd/tasks/benchmark_results/analyses/2026-10-20/expert_policy_score_distributions
```

The analysis retains all valid endpoints for both populations. Scores are raw
and unsmoothed, with no per-task/view rescaling.

Outputs include `index.html`, `report.md`, `results.json`, distribution/comparison
CSV tables, complete window/episode CSV exports, exact episode selections, input
hashes, `episode_maximum_summary.csv`, and PNG/PDF/SVG figures. The fourth panel plots episode maxima versus
scored-window count. Window overlap, episode length, and execution phase affect
these descriptive distributions; they are not independent-window significance
tests or proof of calibration validity. Expert validation was also used to
select the model checkpoint.

Refresh this section of the main reports without recomputing primary results:

```bash
python lerobot/utils/results_inspector.py --refresh-studies
python lerobot/utils/export_benchmark_tables.py --studies-only
```
