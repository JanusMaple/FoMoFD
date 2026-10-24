# Baselines on ACT visual tokens

Each task's `policy_native/` directory contains `logpzo_fix.yaml`,
`logpzo_wrist.yaml`, `rnd_fix.yaml`, `rnd_wrist.yaml`, and one scoring override
per method. Train on expert `data/`; use `prl_data/` only for calibration and
benchmarking. The fixed and wrist models remain separate.

## Representation

Use the task's saved `model/act_dinov2/checkpoints/200000/pretrained_model`.
Its frozen DINOv2 ViT-B/14-register backbone produces 256 final-layer patch
tokens of dimension 768 per 224x224 image. The ACT camera compressor then applies:

1. LayerNorm(768), Linear(768,768), GELU, Linear(768,768) to every patch.
2. A learned camera embedding to these features and to 16 learned query tokens.
3. Eight-head cross-attention: the 16 queries attend to all 256 projected patches.
4. A query residual followed by LayerNorm.
5. A feed-forward block 768 → 3072 → 768 with GELU and dropout, followed by
   another residual and LayerNorm.

The resulting `[16,768]` tokens per camera are the actual visual inputs to ACT's
transformer. The two camera compressors have separate trained weights and share
the DINOv2 backbone. These checkpoints use neither the optional extra 2D positional
embedding nor an appended CLS token. DINOv2 already carries its own positional
information. Dropout was 0.1 during policy training and is disabled during export.
The compressor was learned jointly with the policy; DINOv2 was frozen.

Export restores the DINOv2 weights AND camera-compressor weights from the ACT
checkpoint with strict key checking. The exporter uses cached DINOv2 code when
available and does not download pretrained backbone weights to replace those
saved in ACT. Inference is frozen float32/eval mode. RGB is divided by 255 and
ImageNet-normalized exactly inside the policy tokenizer. No extra feature L2
normalization, VAE, or saved VAE latent statistics are applied.

## Dataset layout and preparation

```text
fomofd/tasks/TASK/data/processed_le_v21_tokens/
fomofd/tasks/TASK/prl_data/processed_le_v21_tokens/
```

Feature keys are `observation.tokens.fix_camera` and
`observation.tokens.wrist_camera`, each `[16,768]` float32. The dataset also
preserves robot state, actions, episode IDs, raw-frame indices, and wall timestamps.
The `1x16x768` shape in baseline configs is only an adapter for the existing
flattening loader; query tokens do not form a spatial image grid.

The local preparation orchestrator can generate all eight datasets from the repo
root. It is currently a development tool under ignored `logs/`:

```bash
conda activate lerobot
python logs/development/dataset_preparation/prepare_datasets.py \
  --task all --representations native --splits data prl_data
# After inspecting that dry-run plan:
python logs/development/dataset_preparation/prepare_datasets.py \
  --task all --representations native --splits data prl_data --execute --resume
```

`--resume` skips completed datasets only after checking metadata, checkpoint and
extraction-code hashes, episode mappings, and expert splits. Partial conversion
staging directories are never mistaken for completed datasets. Existing VAE
datasets are not modified. `meta/preparation.json` records the actual checkpoint,
representation, normalization, and extraction source hashes;
`meta/preparation_splits.json` records the expert episode split.

The reusable converter itself is tracked at `utils/npz_to_le_v21.py`. Without the
local orchestrator, its `--extract_camera_tokens --act_ckpt_dir PATH` mode writes
the same features; the orchestrator additionally validates and records preparation
provenance and splits. Action alignment must follow the recording convention:
expert and simulated PRL use `--shift_actions_to_next_obs`; real PRL does not.

## Training configuration

- logpZO: flattened ACT tokens plus the same 10D robot state as the prior models;
  input dimension 12,298 for one camera.
- RND: flattened ACT tokens without robot state; input dimension 12,288.
  Preserve the prior three-layer predictor/student (`12288 → 2048 → 1024 → 256`)
  and frozen one-layer target/teacher (`12288 → 256`), MSE loss, and mean score
  reduction. This is the existing adapted RND baseline, not the full FIPER method.
- No VAE latent normalization. State remains the same stored state used by the
  existing logpZO loader; no new state normalization is introduced.
- Seed 0, episode split seed 0, validation ratio 0.1, batch sizes 128/256.
  Model settings, optimizer, scheduler, and budgets were copied from each saved
  reference run, including the corrected expert-trained shunt models.
- Step caps: needle 80k, ring 200k, tissue 80k, shunt 80k for both methods.
  These configs do not add early stopping to the baselines.

Run training from `lerobot/`, for example:

```bash
python -m models.faildetect_logpzo.train --config configs/needle_pickup/policy_native/logpzo_fix.yaml
python -m models.fiper_rnd.train --config configs/needle_pickup/policy_native/rnd_fix.yaml
```

Use the corresponding wrist configs for wrist models. Checkpoints go to the
existing task-level model families with new names such as
`logpzo_act_tokens_q16_d768_fix_state_on_data100_seed0` and
`rnd_act_tokens_q16_d768_fix_state_off_data100_seed0`.

The scoring overrides select native PRL features and inherit camera/state/model
settings from the matching trained checkpoint. They must not be used with VAE
checkpoints. The unified benchmark still points to the existing VAE baselines;
switch its paths explicitly after native models have actually been trained.

A comparison of episode-maximum calibration and time-varying conformal bands is
pending. Both should reuse the same raw score cache and reserve test data for
evaluation; changing the threshold alone does not require retraining logpZO.
At the current 5% target, retain all 19 successful calibration episodes for the
conformal quantile. Splitting those 19 further leaves too few calibration episodes
for a finite distribution-free threshold using the usual `(n+1)` rank correction.
Fit the time-dependent center/scale on separate data (for example, expert
validation data), or collect additional successful rollouts for fitting. The
clock, variable-length masking, and treatment of times beyond the fitting
support must be specified before inspecting test results.
