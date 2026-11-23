# FoMo-FD

**Failure Detection for Surgical Robot Imitation Policies via Flow-Matching World Modeling**

[Paper](https://arxiv.org/pdf/2607.27511) · [Website](https://janusmaple.github.io/FoMoFD/)

## Install

Run from the repository root:

```bash
conda env create -f lerobot/environments/fomofd-lerobot.yml
conda activate fomofd-lerobot
python -m pip install --no-deps -e lerobot/policies/act_dinov2
```

For Isaac Sim or Qwen, follow the [environment setup](lerobot/environments/README.md).

## Download data and models

| Download | Size | License |
| --- | --- | --- |
| [Raw datasets](https://huggingface.co/datasets/Januxes/FoMoFD-datasets) | 35.33 GB | CC BY 4.0 |
| [Main models](https://huggingface.co/Januxes/FoMoFD-models) | 31.15 GB | Apache 2.0 |

The repositories are currently private; log in with an account that has access.

```bash
hf auth login
hf download Januxes/FoMoFD-datasets --repo-type dataset \
  --revision 9b98d062e8300f59eca4d6a437077180a10cac68 \
  --local-dir . --include "fomofd/**"
hf download Januxes/FoMoFD-models --repo-type model \
  --revision adeb528c5514087d7203949bc1be000a52946011 \
  --local-dir . --include "fomofd/**"
```

## Prepare features

Generate VAE and ACT-token features for all four tasks (GPU recommended):

```bash
python lerobot/utils/prepare_released_datasets.py --execute --resume
```

For a preview, omit `--execute --resume`. For one task, add `--task needle_pickup`.
See [data preparation](docs/data_preparation.md) for options and release notices.

Task names: `needle_pickup`, `ring_over_post`, `tissue_retraction`, `shunt_insertion`.

## Evaluate

Score the released Needle Pickup world model and generate benchmark reports:

```bash
python lerobot/utils/unified_benchmark.py \
  --tasks needle_pickup --methods wm_nc --calibrations episode --allow-inference
```

Replace the task name as needed. Replace `--allow-inference` with `--dry-run` to
preview jobs. Reports are saved under `logs/benchmarks/`.
See the [evaluation protocol](lerobot/configs/benchmark/PROTOCOL.md) for metric definitions.

## Train

Train a new world model with a separate experiment name:

```bash
cd lerobot
python -m models.flow_matching_wm.train \
  --config configs/needle_pickup/flow_matching_wm.yaml \
  --experiment_name wm_k4_l4_run1
cd ..
```

Set a new experiment name for each run. To benchmark it, update the checkpoint
path in [unified_benchmark.yaml](lerobot/configs/benchmark/unified_benchmark.yaml).
See [experiment configs](lerobot/configs/fm_wm_experiments.txt) for ablations.

## Figures

Generate the sensitivity plots from the included results:

```bash
python lerobot/utils/plot_sensitivity.py --out logs/figures/ablation
```
