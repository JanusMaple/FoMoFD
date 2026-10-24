# Environments

Use separate environments for learning, simulation, and the optional video
baseline. Commands below start at the FoMoFD repository root. The requirements
pin the Python packages used by the project on Linux with NVIDIA CUDA 12.8.
System drivers and robot middleware are installed separately.

| Environment | Python | PyTorch | Main dependencies | Requirements |
| --- | --- | --- | --- | --- |
| `lerobot` | 3.10 | 2.7.1 | LeRobot 0.4.3, timm, torchdiffeq, Matplotlib | [requirements-lerobot.txt](requirements-lerobot.txt) |
| `EMILS` | 3.11 | 2.7.0 | Isaac Sim 5.1, pinned Isaac Lab source | [requirements-emils.txt](requirements-emils.txt) |
| `fomofd-vlm` | 3.11 | 2.7.1 | Transformers 4.57.6, Qwen VL utilities | [requirements-vlm.txt](requirements-vlm.txt) |

## Training, scoring, and plotting

```bash
conda env create -f lerobot/environments/lerobot.yml
conda activate lerobot
python -m pip install --no-deps -e lerobot/policies/act_dinov2
python -m pip check
```

The editable ACT plugin provides the DINOv2 policy used by policy training,
rollout serving, and ACT-feature baselines. Install it from this checkout.

For an existing Python 3.10 environment, install FFmpeg 7.1 and use
`python -m pip install -r requirements.txt`, followed by the plugin installation
above. FFmpeg supplies the shared libraries used by TorchCodec for video
decoding. The separate requirements files should not be combined: simulation
uses different PyTorch, NumPy, and Pillow versions.

Paper figures use Nimbus Roman. On Ubuntu/Debian, the `fonts-urw-base35` package
provides it. The sensitivity plot also accepts a different installed family,
for example `--font "DejaVu Serif"`, or a font file through `--font-path`.
The static project website needs only a browser.

## Simulation

```bash
conda env create -f lerobot/environments/emils.yml
conda activate EMILS
git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab-FoMoFD
git -C ../IsaacLab-FoMoFD checkout 22a00e46cfabae61f2e9b05c5384162ae69b4df4
python -m pip install -e ../IsaacLab-FoMoFD/source/isaaclab \
  -e ../IsaacLab-FoMoFD/source/isaaclab_assets \
  -e ../IsaacLab-FoMoFD/source/isaaclab_tasks
python -m pip check
```

The requirements install Isaac Sim 5.1. The subsequent commands install the
Isaac Lab core, asset, and task packages at commit
`22a00e46cfabae61f2e9b05c5384162ae69b4df4` (core package version 0.52.1).
Keep that checkout: the simulator launcher also loads its `apps/` files.
Follow NVIDIA's
[Isaac Sim 5.1 system and installation requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html),
including the supported driver and the license prompt at first launch.

A simulated policy rollout uses two terminals. In the first, serve a trained
ACT policy from the repository root:

```bash
conda activate lerobot
python lerobot/eval_server.py --task needle_pickup --method act_dinov2 --including_gripper
```

In the second, start the simulator from the same repository:

```bash
conda activate EMILS
python fomofd/eval.py --task needle_pickup --enable_cameras --report_gripper
```

Use `ring_over_post` for the other simulation task. The server requires the
task's ACT checkpoint. Teleoperation recording uses `fomofd/record.py`; physical
robot operation additionally requires a configured dVRK/ROS installation and
the bridge packages in `dvrk_ws/`.

## Qwen video baseline

```bash
conda env create -f lerobot/environments/fomofd-vlm.yml
conda activate fomofd-vlm
python -m pip check
python lerobot/utils/vlm_online_benchmark.py --help
```

This environment runs Qwen3-VL-8B-Instruct with BF16 and PyTorch SDPA;
FlashAttention and vLLM are not required. Store the downloaded model and
processor under `fomofd/model/vlm/Qwen/Qwen3-VL-8B-Instruct/`, or update the model
path in the protocol configuration. The checkpoint is not bundled.

The [Qwen baseline guide](../configs/benchmark/vlm/README.md) identifies the
paper's v9 prompt and protocol. Inputs are RGB recordings with frame indices
and timestamps, rather than VAE latents or ACT tokens.

PyTorch wheel combinations follow the
[official previous-version installation instructions](https://pytorch.org/get-started/previous-versions/).
