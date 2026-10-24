# ACT policy ensemble: additional training seeds

Eight configs are prepared under `<task>/policy_ensemble/seed1001.json` and
`seed1002.json`, for needle pickup, ring over post, tissue retraction, and shunt
insertion. The existing ACT member for every task used seed **1000**.

Each config copies that task's saved
`fomofd/tasks/<task>/model/act_dinov2/checkpoints/200000/pretrained_model/train_config.json`.
Only the training seed, local dataset path, output path, job name and checkpoint
refresh interval change.
The derived `checkpoint_path` field is omitted. These are fresh independent
trainings (`resume: false`, `policy.pretrained_path: null`), not fine-tuning runs.
JSON is intentional: the installed `lerobot-train --config_path` loader parses
training configs as JSON.

Preserved settings include both camera images and weak robot state, a four-action
chunk, 16 visual tokens per camera, the original ACT architecture and VAE,
batch size 64, 200,000 training steps, AdamW at 1e-5 and no LR scheduler.
Use `last` after 200,000 steps for the budget-matched comparison
with the existing member. All expert episodes are selected, as in the saved ACT
training config (250 per simulation task, 100 per real task); this is not the
FM-WM 90/10 episode split. PRL episodes are not training inputs.

## Dataset prerequisite

Training requires RGB expert demonstrations in `data/processed_le_v21`.
VAE and ACT-token datasets cannot substitute for the RGB inputs: each policy
learns its own visual token compressor.

Before launching training, prepare the RGB expert dataset once per task using
the original next-observation action alignment and gripper standard-deviation
scale of 10. From the repository root, an example is:

```bash
conda activate lerobot
TASK=needle_pickup
python lerobot/utils/npz_to_le_v21.py \
  --task "$TASK" \
  --repo_id "local/$TASK" \
  --raw_dir "fomofd/tasks/$TASK/data/raw" \
  --output_dir "fomofd/tasks/$TASK/data/processed_le_v21" \
  --fps auto --including_gripper \
  --shift_actions_to_next_obs \
  --gripper_transform std_scale --gripper_std_scale 10
```

Omitting feature-extraction flags selects RGB output. Before training, verify
the episode counts, both image columns, action alignment, and action/state
normalization statistics against the existing ACT checkpoint processors.
Do not overwrite an existing dataset to run this command.

## Training command

Use the repository runner, which resolves config paths relative to `lerobot/`:

```bash
conda activate lerobot
cd /path/to/FoMoFD
bash lerobot/run_act_ensemble.sh needle_pickup 1001
```

Substitute the task and seed to run another member after dataset preparation.
Outputs are separate from the existing policy:

```text
fomofd/tasks/<task>/model/act_dinov2_ensemble/
  act_dinov2_weak_state_both_chunk4_seed1001/
  act_dinov2_weak_state_both_chunk4_seed1002/
```

Do not launch these configs directly with `lerobot-train` if you want best/last
retention: that command still writes numbered checkpoints. The wrapper uses
LeRobot's existing training loop, preprocessing, optimizer and checkpoint writer,
and replaces only the checkpoint-retention and logging callbacks. It imports
the ACT plugin from this checkout.

## Best and last checkpoints

- `checkpoints/last/`: refreshed every 5,000 updates and at the final update.
- `checkpoints/best/`: refreshed only when the mean loss over the most recent
  1,000 training updates strictly improves at a checkpoint event. This is total
  ACT training loss (L1 + weighted KL), not validation loss or rollout success.
- Both use standard LeRobot `pretrained_model/` and `training_state/` contents,
  including processors, optimizer, step and RNG state. No numbered checkpoint
  history is retained. Best and last share hard-linked files when identical;
  completed directories are replaced rather than editing their files in place.
  A temporary staging copy can increase disk use while saving.
- `history.jsonl` records every update. `run_report.json` records timing, peak
  VRAM, selection criterion and source hashes.
- Fresh output folders are mandatory. This launcher currently starts fresh runs;
  automated resume is not implemented. A hard interruption loses updates since
  the most recent saved `last`; existing complete checkpoints remain available.

All expert episodes are used for training with a fixed step budget. See the
[AEU evaluation protocol](benchmark/ACT_ENSEMBLE_PROTOCOL.md) for ensemble
disagreement scoring and benchmark commands.
