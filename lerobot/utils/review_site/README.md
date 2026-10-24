# Project website

The static page contains two interactive replay panels, detection results, and
collapsed sensitivity/robustness and latency sections. Open the generated page:

```bash
xdg-open fomofd/tasks/review_site/index.html
```

It works from disk without a server. Replay defaults to Needle Drop and Shunt
Insertion Misalignment, wrist view, 2× speed, paused, with successful calibration
curves visible. Task/camera switches preserve each recording's controls. Drag a
score plot or timeline to seek.

Build or refresh from the current frozen local benchmark archive:

```bash
python lerobot/utils/export_benchmark_tables.py
python lerobot/utils/export_review_site.py
# Presentation changes reuse all videos and score curves:
python lerobot/utils/export_review_site.py --refresh-presentation
python lerobot/utils/export_public_results.py
```

The first command writes full provenance to `logs/benchmarks/result_tables`.
Public exports select current runs, documented supplementary baselines, and the
paper's 500-split/four-seed studies. They omit development reports and historical
runs. The sensitivity curves default to the **5% FAR budget** used in Figure 4;
the switch exposes the calibrated operating points. FAR/FDR pool counts, and
public values retain their original precision.

[Evaluation definitions](protocol.md) cover thresholds, onset, latency, fusion,
and rounding. Qwen detection includes late returned decisions; paired latency
uses on-time shared detections. Recording IDs in URLs and CSVs retain source
IDs; the replay display numbers recordings from zero within each failure mode.

The exporter verifies input hashes and reuses paired-camera videos only when
source bytes and timestamps match. It does no model inference, calibration,
label editing, commit, upload, or deployment. Full archives and local review
notes remain outside the generated page. The website needs only the files in
its generated directory; no remote scripts, fonts, or analytics are used.

Build the source/site release using `python lerobot/utils/export_public_release.py`.
That directory excludes Git history and local development assets. Text scanning
does not inspect identifying details that might be visible inside recordings.
