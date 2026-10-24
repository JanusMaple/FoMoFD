# NoAct and AEU functional bands

Completed on **2026-10-20 at 19:55 EDT**, including all eight scorer jobs and
the report/site refresh. Expert scores were generated; existing PRL scores were
reused. Do not substitute failure scores or success-test scores for the
expert template. State, frozen inputs and logs are under
`logs/experiments/2026-10-20/functional_noact_aeu/`.

- **NoAct:** existing K4/L4 WM-NC action-none checkpoint, separate camera scores.
- **AEU:** existing positional action-ensemble uncertainty score, joint input/output;
  unchanged three ACT checkpoints, signed adjacent-observation discrepancy and score orientation.
- Expert IDs: exact saved validation manifests, 25 episodes per simulation task,
  10 per real task. ACT members themselves used all expert episodes: the AEU
  template set follows the saved preparation split but is **not held out from
  ACT training**. Do not describe it as policy-held-out validation.
- Match existing functional bands: 1-second expert-derived mean curve, constant
  modulation, frozen before PRL evaluation. Use all valid endpoints, the existing
  visual correction, no smoothing, strict threshold crossing and 19 calibration
  successes. PRL scores are reused; no retraining or PRL rescoring.
- New onset reference: per-view mean annotated wall time; combined AEU uses the
  earlier view mean. This affects detection/latency, not threshold fitting.

CPU-only planning (choose a fresh output directory):

```bash
CUDA_VISIBLE_DEVICES='' python lerobot/utils/functional_benchmark.py \
  --config lerobot/configs/benchmark/functional_band_noact_aeu.yaml \
  --variants wm_k4_l4_action_none act_ensemble --methods wm_nc act_ensemble_pos \
  --output logs/experiments/2026-10-20/functional_noact_aeu_plan_new --plan-only
```

The completed detached queue used frozen copies of the configuration, catalog,
labels and evaluation protocol. To reproduce in a fresh result directory:

```bash
python lerobot/utils/functional_benchmark.py \
  --config lerobot/configs/benchmark/functional_band_noact_aeu.yaml \
  --variants wm_k4_l4_action_none act_ensemble --methods wm_nc act_ensemble_pos \
  --output fomofd/tasks/benchmark_results/functional/noact_aeu_reproduction \
  --allow-inference
```

All eight jobs and artifact checks passed. The completed result
`functional/2026-10-20/noact_aeu` is registered in `functional_band_runs` in
`results_catalog.yaml`; HTML, Markdown/YAML, CSV exports and the review site
were rebuilt. Inspect the completion log with:

```bash
tail -n 30 logs/experiments/2026-10-20/functional_noact_aeu/run.log
```

Functional-band **online latency**
remains unmeasured: the existing episode-threshold replay schedule cannot be
claimed to measure a newly calibrated detector. A separate replay is needed for
that result. The current charts leave missing functional results unavailable.
