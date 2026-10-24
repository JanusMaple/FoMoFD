# Single-branch fixed+wrist FM-WM (K4 L4)

The `view_fusion_ablation_k4_l4/concat_channels.yaml` config under each task
changes only `backbone.view_fusion` and the experiment name relative to that
task's `k_ablation_l4/k4.yaml`. Existing configs default to `view_fusion: separate`.

## Architecture and data

Each prepared observation already contains fixed and wrist VAE latents. The
existing dataset normalizes each camera with its saved per-channel statistics.
The model then concatenates fixed channels followed by wrist channels:

```text
fixed history [B, L=4, 16, 16, 8] ─┐
                                  ├─ channel concatenation [B, 4, 16, 16, 16]
wrist history [B, L=4, 16, 16, 8] ─┘
                                               │
joint noisy endpoint [B, 16, 16, 16] ────────────┤
                                               ▼
                   one visual branch: projection 16→504,
                   same 10-layer backbone + action conditioning,
                   velocity output 504→16 and one auxiliary head
```

This concatenates channels at each grid position, not spatial tokens or whole
flattened maps. It permits mixing the two views without increasing the number
of spatial/time tokens. It does not assume that matching pixel coordinates
correspond to the same physical point across cameras.

**No new dataset, VAE, or latent-statistics file is needed.** `latent_channels`
remains **8 per camera** in the data config; the branch internally uses 16.
The same expert episode split, full training subset, normalized actions,
action cross-attention plus adaLN, disabled state input, learning-rate schedule,
batch size, validation schedule, seed and loss weights are retained.

The flow objective remains `MSE_fixed + MSE_wrist`; the one-step auxiliary
objective uses the same sum, weighted by 0.5. Taking only the mean over all
16 channels would halve the previous objective, so the implementation explicitly
preserves the sum. Both output channel groups now depend on both views.

The public endpoint tensor remains `[B, 2, 16, 16, 8]`, fixed then wrist, so the
existing trainer, ODE transport, endpoint sampling and low-level scorer work
with the new checkpoint. The scorer already saves `scores_union`, the joint
score over all channels. Its per-view scores are contributions from the
jointly conditioned model, not independent single-camera detectors. Future
benchmark registration should identify this as a distinct joint-input model
and calibrate its joint score separately from the existing AND/OR fusions.

## Capacity and training budgets

Keeping hidden width 504 and depth 10 gives **67,592,344 parameters** for the
single-branch model versus **129,760,476** for the separate-branch K4/L4 model.
This is a same-width/depth architectural comparison, not a parameter-matched
comparison. Parameter counts alone do not establish runtime or memory savings.

| Task | Maximum optimizer steps | Batch / validation batch |
| --- | ---: | ---: |
| Needle pickup | 120,000 | 8 / 8 |
| Ring over post | 80,000 | 8 / 8 |
| Tissue retraction | 80,000 | 8 / 8 |
| Shunt insertion | 100,000 | 8 / 8 |

These configs exactly preserve the canonical K4/L4 runtime settings, including
their unset `early_stop_patience_steps` (fixed step cap). The existing optional
early-stop implementation remains available; this ablation does not introduce
a new patience or shorten the reference budgets.

## Commands

From the repository root, activate the `lerobot` environment, then:

```bash
cd lerobot
python -m models.flow_matching_wm.train \
  --config configs/needle_pickup/view_fusion_ablation_k4_l4/concat_channels.yaml
```

Substitute `ring_over_post`, `tissue_retraction`, or `shunt_insertion` for the
other tasks. Run directories stay in the shared per-task `model/fm_wm/` folder:

```text
fomofd/tasks/<task>/model/fm_wm/
  fm_wm_vfmfvae_z8_both_concat_channels_state_off_k4_l4_action_both_data100_seed0/
    config.json
    splits.json
    history.jsonl
    checkpoints/{best,last}.pt
```

The default names are distinct from every separate-branch model. Train from
scratch; separate-branch weights/optimizer states are incompatible. To resume
an interrupted **joint** run, append `--resume <that-run>/checkpoints/last.pt`
to the same training command. The saved effective `config.json` preserves the
architecture for scoring and reloading.

The existing task score override can be reused. For example, from `lerobot/`,
after training has completed:

```bash
python -m models.flow_matching_wm.score \
  --config configs/needle_pickup/flow_matching_wm_score.yaml \
  --checkpoint ../fomofd/tasks/needle_pickup/model/fm_wm/fm_wm_vfmfvae_z8_both_concat_channels_state_off_k4_l4_action_both_data100_seed0/checkpoints/best.pt \
  --mode reactive --split all \
  --output_dir ../fomofd/tasks/needle_pickup/prl_data/score_result/joint_wm_k4_l4
```

CPU tests cover the joint forward/backward pass, loss scale, cross-camera
dependence, ODE transport, score reduction, dataset normalization and checkpoint
reload. Legacy fixed-only, wrist-only and separate dual-view models were also
compared against the pre-change code for exact weights, losses and gradients.

```bash
# From repository root:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=lerobot:lerobot/tests \
python -m unittest test_fm_wm_view_fusion
```
