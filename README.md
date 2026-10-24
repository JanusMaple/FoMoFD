# FoMo-FD

Research code and companion results for **Failure Detection for Surgical Robot
Imitation Policies via Flow-Matching World Modeling**.

FoMo-FD learns nominal visual dynamics from successful demonstrations and
scores deviations using camera observations and robot actions.

- [Current result tables and evaluation protocol](docs/results/README.md)
- [World-model configurations](lerobot/configs/fm_wm_experiments.txt)
- [Model directory conventions](lerobot/configs/model_organization.md)
- [Project website instructions](lerobot/utils/review_site/README.md)
- [VLM environment](lerobot/environments/README.md)

Training, scoring, and replay require the corresponding datasets and model
checkpoints. These assets are not bundled with the source code. Configuration
paths are relative to the repository; see the task-specific files under
`lerobot/configs/` and the command-line help of the relevant utility.

The public tables use the paper's current evaluation protocol. Supplementary
VAE-feature baselines are labeled separately from ACT-native Policy baselines.
Blank numerical entries mean unavailable, not zero. Full run histories and local
review records are stored outside the public outputs.

To build a release directory containing the current source and static website:

```bash
python lerobot/utils/export_public_release.py
```

The release contains no Git history, local logs, credentials, datasets, or model
weights. Refresh results before rebuilding it. Local development archives remain
available in ignored directories for reproducibility.
