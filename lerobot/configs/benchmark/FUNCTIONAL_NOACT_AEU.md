# NoACT and AEU functional bands

The functional-band comparison uses the K4/L4 NoACT world model with separate
camera scores and the positional action-ensemble uncertainty (AEU) score with
joint camera input/output.

Templates use the saved expert validation split: 25 episodes per simulation
task and 10 per real task. ACT members were trained on all expert episodes, so
the AEU template set is not held out from policy training. Each template uses
a one-second expert-derived mean curve and constant modulation, frozen before
policy-rollout evaluation. Thresholds use 19 successful calibration episodes,
strict crossings, and unsmoothed scores on all valid endpoints.

Detection uses the per-view mean annotated onset; joint AEU uses the earlier
view mean. Onset annotations do not enter threshold fitting.

With the required checkpoints, datasets, and score caches available, inspect
the jobs from the repository root:

```bash
python lerobot/utils/functional_benchmark.py \
  --config lerobot/configs/benchmark/functional_band_noact_aeu.yaml \
  --variants wm_k4_l4_action_none act_ensemble --methods wm_nc act_ensemble_pos \
  --output logs/plans/noact_aeu --plan-only
```

To generate missing expert scores and evaluate in a fresh output directory:

```bash
python lerobot/utils/functional_benchmark.py \
  --config lerobot/configs/benchmark/functional_band_noact_aeu.yaml \
  --variants wm_k4_l4_action_none act_ensemble --methods wm_nc act_ensemble_pos \
  --output fomofd/tasks/benchmark_results/functional/noact_aeu \
  --allow-inference
```

Functional-band online latency is not measured by an episode-threshold replay;
it requires a separate timed evaluation of the calibrated monitor.
