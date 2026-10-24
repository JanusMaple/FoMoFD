# FM-WM training-data sensitivity

Use `configs/TASK/data_size_ablation_k4_l4/data33.yaml` and `data66.yaml` for
the one-third and two-thirds training subsets. The names abbreviate the exact
fractions: `train_episode_fraction` is `1/3` or `2/3`, rather than `0.33` or `0.66`.
The experiment varies the number of expert episodes used for FM-WM training.

| Task | One third (`data33`) | Two thirds (`data66`) | Full training reference | Fixed validation |
|---|---:|---:|---:|---:|
| Needle Pickup | 75 | 150 | 225 | 25 |
| Ring Over Post | 75 | 150 | 225 | 25 |
| Tissue Retraction | 30 | 60 | 90 | 10 |
| Shunt Insertion | 30 | 60 | 90 | 10 |

The trainer first makes the original episode-level split with `split_seed=0`.
It then shuffles only the expert training IDs using `train_subset_seed=0` and
selects a prefix. The one-third subset is contained in the two-thirds subset;
both are contained in the full training split and disjoint from validation.
The selected and unused training IDs are recorded in each run's `splits.json`.
All windows from selected episodes are eligible for training.

Keep K=4, L=4, both cameras, both action-conditioning mechanisms, state off,
model seed 0, and the full-reference architecture, optimizer, batch sizes,
validation schedule, and cosine learning-rate schedule. Step caps remain 120k
for Needle, 80k for Ring/Tissue, and 100k for Shunt. Early stopping uses
`early_stop_patience_steps=47001`, with strict improvement in validation flow
loss and checks after validation. Validation every 1,000 steps means a stagnant
run first crosses this patience threshold at a validation check after 48,000
steps without improvement.

The saved VAE features, VAE weights, and latent/action normalization statistics
come from the original preparation and are reused across all fractions. This
measures FM-WM training-data sensitivity with a fixed representation; it does
not measure retraining the entire representation pipeline with fewer demos.
No dataset regeneration is required. PRL data remain for calibration/evaluation.

Train each fraction from scratch. The full-data reference is the existing
`fomofd/tasks/TASK/model/fm_wm/fm_wm_vfmfvae_z8_both_state_off_k4_l4_action_both_data100_seed0/checkpoints/best.pt`;
`data100.yaml` describes its settings and is not a request to retrain it.
The earlier 25/50/75% configs are older drafts and are outside this proposed sweep.

From `lerobot/`, for example:

```bash
python -m models.flow_matching_wm.train --config configs/needle_pickup/data_size_ablation_k4_l4/data33.yaml
python -m models.flow_matching_wm.train --config configs/needle_pickup/data_size_ablation_k4_l4/data66.yaml
```

New checkpoints use the shared task-level `model/fm_wm/` folder with names such as
`fm_wm_vfmfvae_z8_both_state_off_k4_l4_action_both_data33_seed0` and its `data66`
counterpart. Changing the data fraction does not directly reduce GPU memory or
runtime per optimizer step; runtime depends on the cap and when early stopping
triggers.
