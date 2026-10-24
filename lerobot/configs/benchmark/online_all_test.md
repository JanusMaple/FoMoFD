# Full-test streaming evaluation

This optional schema-2 experiment is separate from the schema-1 ablation
pipeline, which skips successes and offline-undetected failures. It runs only
when explicitly selected with `--config` and `--run`.

`online_all_test.yaml` selects all verified test successes and failures: 80
successes and 320 failures across four tasks. Calibration IDs 0–18 are excluded.
The eight main methods and six WM-NC K/L variants require 7,200 raw-image replays;
the main K4/L4 reference is used once. There is no offline-detection filtering.

Each method has one synchronous worker at recorded wall speed. It processes the
latest eligible window, including visual encoding and scoring, and skips stale
windows. Constant and functional-band thresholds use the same actually computed
scores, with independent first alarms for each protocol/view. Replay continues
until all protocols/views alarm or the source is exhausted. No outcomes or onset
annotations affect scheduling or stopping. This is one paired stream per method,
not two independent scheduling measurements.

Both thresholds are frozen from the offline **causal/all-endpoint** calibration.
The functional template uses elapsed time from the first prepared observation;
stream arrival and alarm times use elapsed raw-recording wall time. The offset
is handled explicitly. The exact older sampled-score helpers recorded by the
PRL caches are restored from Git and hash-checked; other input hashes stay strict.
Every resident detector receives raw/prepared feature and score parity checks
before timed trials. Model loading, warm-up, and preloading are outside the timer.

## Metrics

- FAR: successes with any first alarm / all tested successes.
- FDR_any_alarm: failures with any first alarm / all tested failures, including
  alarms before onset. No-alarm failures are reported separately.
- Pre-onset alarms: alarm completion before onset. Pre-onset evidence: the
  triggering frame precedes onset, even if processing completes later. Both
  counts are reported; early evidence is never credited as post-onset detection.
- Deadline rates at 1/2/5/10 seconds: qualifying post-onset detections by the
  deadline / **all annotated failures**, including misses and early evidence.
- Conditional delay mean, sample SD, median, p95, and n use qualifying post-onset
  detections only. Missing annotations are excluded from onset metrics explicitly,
  but their episode outcomes still enter FAR/FDR. Annotators and camera onsets
  are never pooled.

Partial reports include completed and planned denominators; a missing result is
not treated as a miss or zero FAR. Successes never receive onset/latency labels.

## Run

From the repo root in the `lerobot` environment:

```bash
# CPU-only plan; select a new output directory.
python lerobot/utils/online_benchmark.py \
  --config lerobot/configs/benchmark/online_all_test.yaml \
  --output logs/experiments/2026-10-08/all_test_streaming/new_plan

# Explicit GPU run; no concurrent training/scoring on this desktop.
python -u lerobot/utils/online_benchmark.py \
  --config lerobot/configs/benchmark/online_all_test.yaml \
  --output fomofd/tasks/benchmark_results/streaming/new_full_test_run \
  --run
```

Add `--resume` only for the same output and unchanged code/config/input hashes.
Completed trial JSONs are retained and skipped. Reports refresh about once per
minute outside timed replay; the log and state update after each episode.

The result directory contains `report.html`, `report.md`, `metrics.csv`,
`onset_metrics.csv`, `deadline_rates.csv`, per-episode outcome/latency CSVs,
per-trial score/timing traces, exact plan, model validation, and hardware metadata.
Old conditional-latency archives are unchanged and are not merged into this run.
