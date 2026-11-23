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

Prepare expert ACT-token features from the repository root:

```bash
conda activate fomofd-lerobot
python lerobot/utils/npz_to_le_v21.py \
  --task needle_pickup --including_gripper \
  --gripper_transform std_scale --gripper_std_scale 10 \
  --shift_actions_to_next_obs --extract_camera_tokens \
  --act_ckpt_dir fomofd/tasks/needle_pickup/model/act_dinov2/checkpoints/200000/pretrained_model
```

Use `--prl` to prepare policy rollouts. Expert demonstrations and simulated
policy rollouts use `--shift_actions_to_next_obs`; real policy rollouts omit
that flag. The converter also accepts explicit `--raw_dir` and `--output_dir`
paths. Use the matching task's frozen ACT checkpoint for feature extraction.

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
task-level model families with names such as
`logpzo_act_tokens_q16_d768_fix_state_on_data100_seed0` and
`rnd_act_tokens_q16_d768_fix_state_off_data100_seed0`.

The scoring overrides select ACT-token policy-rollout features and inherit
camera, state, and model settings from the trained checkpoint. The unified
benchmark registers these baselines as `logpzo_native` and `rnd_native`.
They use their own checkpoints, separate from supplementary VAE-feature
variants. Calibration and functional-band definitions are documented in
[the evaluation protocol](benchmark/PROTOCOL.md).
