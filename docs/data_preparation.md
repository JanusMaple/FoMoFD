# Prepare datasets from the raw FoMo-FD release

Download raw recordings and main checkpoints into a fresh FoMoFD code checkout.
The data release includes 700 expert episodes and 476 policy rollouts. VAE and
ACT-token feature caches are intentionally omitted; the commands below rebuild
both representations for all four tasks without retraining any model.

## Download

The verified release contains about **66.5 GB** of raw data, metadata and main
checkpoints. Activate `fomofd-lerobot` (which already includes the `hf` CLI), then
run from the code repository root:

```bash
hf download Januxes/FoMoFD-datasets --repo-type dataset \
  --revision 9b98d062e8300f59eca4d6a437077180a10cac68 \
  --local-dir . --include "fomofd/**"
hf download Januxes/FoMoFD-models --repo-type model \
  --revision adeb528c5514087d7203949bc1be000a52946011 \
  --local-dir . --include "fomofd/**"
```

These asset-only downloads preserve this checkout's code, labels, protocol,
README and licenses. The current code includes the conversion helper and frozen
benchmark definitions. The commands pin verified Hub revisions; repeating them
continues interrupted downloads. Reserve additional disk space for generated
features. See Hugging Face's [download documentation](https://huggingface.co/docs/huggingface_hub/guides/download)
for alternative download methods.

Download the release cards, licenses, attribution notices, checksums and model
index into separate ignored directories:

```bash
hf download Januxes/FoMoFD-datasets README.md LICENSE NOTICE FILES.json RELEASE.json \
  --repo-type dataset --revision 9b98d062e8300f59eca4d6a437077180a10cac68 \
  --local-dir logs/releases/dataset
hf download Januxes/FoMoFD-models README.md LICENSE NOTICE FILES.json RELEASE.json MODEL_INDEX.csv \
  docs/licenses/DINOv2-LICENSE.txt docs/licenses/LeRobot-LICENSE.txt \
  --repo-type model --revision adeb528c5514087d7203949bc1be000a52946011 \
  --local-dir logs/releases/model
```

Datasets are licensed under **CC BY 4.0** and released checkpoints under
**Apache 2.0**, with applicable upstream notices. Preserve these notices when
redistributing assets. `FILES.json` records payload byte sizes and SHA-256 hashes;
`MODEL_INDEX.csv` lists the exact checkpoint selections. The preparation helper
verifies its raw recordings and feature-encoder checkpoints against saved hashes
before extraction. Only main/supporting models are included in this first release.

Files use the repository's existing layout:

```text
fomofd/tasks/TASK/data/raw/                         # expert recordings
fomofd/tasks/TASK/prl_data/raw/                     # policy rollouts
fomofd/tasks/TASK/{data,prl_data}/release_metadata/  # frozen reference metadata
fomofd/tasks/TASK/model/                           # released checkpoints
```

Reference metadata is deliberately separate from generated datasets. It includes
episode mappings, fixed normalization statistics, preparation settings and saved
expert splits. It is not a prepared feature dataset. Do not move it into an output
directory before conversion or use its reference Parquet metadata as observations.

## Convert

Activate `fomofd-lerobot` and install its dependencies and the editable ACT plugin
as documented in `lerobot/environments/README.md`. DINOv2 extraction also needs
its Torch Hub code/weights available locally or downloadable on first use.
A GPU is recommended. From the repository root:

```bash
# Preview all 16 jobs; no model is loaded and no data is written.
python lerobot/utils/prepare_released_datasets.py

# Generate features; interruption-safe per completed dataset.
python lerobot/utils/prepare_released_datasets.py --execute --resume

# Or prepare one task/representation first:
python lerobot/utils/prepare_released_datasets.py \
  --task needle_pickup --representations vae --execute --resume
```

Outputs retain the locations referenced by the scoring configs:

```text
fomofd/tasks/TASK/{data,prl_data}/processed_le_v21_vfmf_vae_fix_wrist/
fomofd/tasks/TASK/{data,prl_data}/processed_le_v21_tokens/
```

The helper verifies raw/checkpoint SHA-256 hashes before extraction, uses the
existing `npz_to_le_v21.py` converter, and checks episode counts, frame counts,
schema and action alignment. It then restores released statistics and episode
splits, retaining newly calculated stats as `meta/stats_regenerated.json` for
inspection. It never substitutes recalculated latent statistics for those used
by the released checkpoints. Existing outputs are not overwritten: `--resume`
accepts only matching completed conversions recorded by this helper. A failed
job's temporary output is discarded, while prior completed datasets remain.

Source recordings and checkpoint bytes are never modified. Keep the inputs
unchanged while conversion is running. Deterministic VAE encoding is used, but
GPU/library differences can still cause small floating-point differences; this
recipe does not promise byte-identical regenerated feature Parquet files.

## Frozen conventions

- Expert episodes and simulated policy rollouts pair observation `t` with action
  `t+1`; the last raw observation is omitted by that alignment. Real policy
  rollouts retain the pre-action observation/action pairing, without this shift.
- Preserve original episode numbering and raw-frame indices. Use wall timestamps
  for timing; dataset nominal FPS is not a replacement for recorded wall time.
- Use gripper standard-deviation scale 10. VAE extraction uses the released
  fixed/wrist encoders in deterministic posterior-mode inference. ACT tokens use
  each task's released 200000-step policy and learned camera compressors.
- Visual-VAE/FM-WM expert splits are episode-level: 225/25 for each simulation
  task and 90/10 for each real task. Original ACT policies used all expert episodes.
- Keep the full raw recordings. The evaluation protocol applies Needle Pickup
  policy-rollout episode 5's validity mask (exclude raw frame 122 onward) without
  deleting/renumbering raw data during preparation. Mean onset and score rules
  remain in `lerobot/configs/benchmark/evaluation_protocol_v3.yaml`.

After preparation, run the benchmark dry-run in the main README to check paths.
Use matching current scoring configs; historical paths in saved training records
are retained as provenance and should not be treated as executable local paths.

To retrain visual VAEs or image-input policies from scratch, the base converter
also provides `--extract_latents_L_fix`, `--extract_latents_L_wrist`, and its default
image mode. Those optional intermediate datasets are not needed to evaluate the
released monitors and are not generated by this helper.
