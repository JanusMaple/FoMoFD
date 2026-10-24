# Calibration-split sensitivity

The paper study uses 500 distinct 19/20 partitions of the 39 successful policy
rollouts per task, RNG seed 20261013, and the main K4/L4 WM-NC checkpoint.
The same task-specific partitions are used across views and calibration methods;
all failure recordings remain fixed. All saved splits contribute to the summary.

The constant statistic is the episode maximum. The functional statistic uses
residuals from the frozen expert-derived band. Thresholds and raw scores from
the original split are verified before evaluating new splits. Current results
use valid endpoints and the three-annotation mean onset.

[calibration_sensitivity_main_500.yaml](calibration_sensitivity_main_500.yaml)
defines the calibrated study. The separate
[matched-FAR configuration](calibration_matched_far_main_500.yaml) uses each
split's 20 test successes to select a retrospective 5% FAR budget threshold.
It does not use the other 19 successes for that threshold.

```bash
python lerobot/utils/calibration_sensitivity.py \
  --config lerobot/configs/benchmark/calibration_sensitivity_main_500.yaml \
  --output fomofd/tasks/benchmark_results/analyses/NEW_RUN
```

This cache-only calculation requires no GPU or new scoring. It saves split
membership, exact counts, summary statistics, and source hashes. Use a new
output directory for a new run.

FAR and FDR pool task counts within each split before computing the mean and
sample SD. AUROC averages tasks equally. These overlapping resamples describe
sensitivity, not independent experiments or confidence intervals. A mean FAR
near 5% follows rank symmetry and is not a calibration-validity guarantee.
