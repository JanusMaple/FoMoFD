# FoMo-FD

**Failure Detection for Surgical Robot Imitation Policies via Flow-Matching
World Modeling**

FoMo-FD learns nominal visual dynamics from successful-only demonstrations,
using camera images and robot actions to detect execution failures. A frozen
visual encoder and feature-space VAE provide compact observations; an
action-conditioned flow-matching world model learns how these observations
evolve. At execution time, the WM-NC anomaly score is compared with a threshold
calibrated on successful policy rollouts.

The evaluation covers **Needle Pickup** and **Ring Over Post** in simulation,
and **Tissue Retraction** and **Shunt Insertion** on a surgical robot. This
repository includes the models, task configurations, evaluation tools, paper
figure scripts, and an interactive project website.

## Explore the project

[**Explore the interactive project website →**](https://janusmaple.github.io/FoMoFD/)

Explore episodes, compare both camera views, and drag the score curves to compare
visual events with detection scores. The website also presents detection results,
sensitivity experiments, and latency comparisons. No installation is required.

## Installation

The code uses three separate environments:

| Environment | Python | Purpose |
| --- | --- | --- |
| `fomofd-lerobot` | 3.10 | Policy/world-model training, scoring, evaluation, and figures |
| `fomofd-isaac` | 3.11 | Isaac Sim / Isaac Lab task simulation and recording |
| `fomofd-vlm` | 3.11 | Qwen video-baseline inference |

For training, evaluation, and plotting, run from the repository root:

```bash
conda env create -f lerobot/environments/fomofd-lerobot.yml
conda activate fomofd-lerobot
python -m pip install --no-deps -e lerobot/policies/act_dinov2
```

The root [requirements.txt](requirements.txt) provides the same training/scoring
dependencies for an existing Python environment. See the
[environment guide](lerobot/environments/README.md) for all three requirements
files, simulation setup, and the optional Qwen environment.

## Data and checkpoints

Training and model inference require datasets and checkpoints, which are not
bundled with this repository. The included website and sensitivity figure can
be used without them. Place research assets under the corresponding task:

```text
fomofd/tasks/<task>/
  data/       # Successful expert demonstrations and prepared training features
  prl_data/   # Policy rollouts and prepared evaluation features
  model/      # Policy, visual VAE, world-model, and baseline checkpoints
```

Task identifiers are `needle_pickup`, `ring_over_post`, `tissue_retraction`, and
`shunt_insertion`. The main world model uses eight-channel VAE features with
prediction horizon K=4 and observation history L=4. Dataset conversion and
feature-extraction options are available with:

```bash
python lerobot/utils/npz_to_le_v21.py --help
```

## Train and score a world model

Activate `fomofd-lerobot`, then run the following commands from the `lerobot/` directory;
the paths in model configuration files are relative to that directory.
Training requires the prepared expert VAE features specified by the config.

```bash
cd lerobot
python -m models.flow_matching_wm.train \
  --config configs/needle_pickup/flow_matching_wm.yaml
```

To compute WM-NC scores on prepared policy rollouts after training:

```bash
python -m models.flow_matching_wm.score \
  --config configs/needle_pickup/flow_matching_wm_score.yaml \
  --checkpoint ../fomofd/tasks/needle_pickup/model/fm_wm/fm_wm_vfmfvae_z8_both_state_off_k4_l4_action_both_data100_seed0/checkpoints/best.pt \
  --mode reactive --split all \
  --output_dir ../fomofd/tasks/needle_pickup/prl_data/score_result/wm_nc
```

Replace the task name to use another task. Visual VAE training uses
`python -m models.vfmf_vae.train --config configs/needle_pickup/vfmf_vae_fix.yaml`
and its `vfmf_vae_wrist.yaml` counterpart. Each trainer exposes additional
options through `--help`.

See [experiment configurations](lerobot/configs/fm_wm_experiments.txt) for
horizon, history, action-conditioning, and training-data variants, and
[model locations](lerobot/configs/model_organization.md) for checkpoint layout.

## Evaluation and figures

Run these commands from the repository root in `fomofd-lerobot`.
Once checkpoints and their saved training configurations are installed, inspect
benchmark jobs before running inference:

```bash
python lerobot/utils/unified_benchmark.py \
  --tasks needle_pickup --methods wm_nc --calibrations episode --dry-run
```

The [evaluation protocol](lerobot/configs/benchmark/PROTOCOL.md) defines
calibration, onset-aware detection, and latency. The **5% FAR budget** comparison
uses retrospective test-success thresholds; it is separate from the calibrated
operating point.

Generate the sensitivity figure using the included measured results:

```bash
python lerobot/utils/plot_sensitivity.py --out logs/figures/ablation
```

This produces PDF, SVG, PNG, and the plotted values. For the failure-example
figure, the interactive selector lets you choose recordings and frame markers:

```bash
python lerobot/utils/failure_example_plot_selector.py
```

The selector requires the original recordings and offline score reports. Its
defaults are stored in `lerobot/configs/figures/failure_example.yaml`. See the
[website guide](lerobot/utils/review_site/README.md) for presentation files and
rebuilding the replay page from benchmark outputs.
