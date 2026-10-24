# VLM inference environment

Use `fomofd-vlm` for Qwen3-VL-8B-Instruct and Cosmos-Reason2-8B inference.
Keep existing FoMo-FD training and scoring in `lerobot`.

From the repository root, create the environment on another Linux/CUDA desktop:

```bash
conda env create -f lerobot/environments/fomofd-vlm.yml
conda activate fomofd-vlm
python -m pip check
```

On a desktop where it is already installed, only activation is needed:

```bash
conda activate fomofd-vlm
cd /path/to/fomofd
```

The environment pins Python 3.11, PyTorch 2.7.1 with CUDA 12.8,
torchvision 0.22.1, Transformers 4.57.6, and the dataset readers. Both models use
`Qwen3VLForConditionalGeneration` with `AutoProcessor`. Start with BF16 and
`attn_implementation="sdpa"`; FlashAttention and vLLM are not required.
Load one model at a time from its shared local folder:

```text
fomofd/model/vlm/Qwen/Qwen3-VL-8B-Instruct/
fomofd/model/vlm/nvidia/Cosmos-Reason2-8B/
```

Use `local_files_only=True` to load the downloaded checkpoints and processors.
Their `HF_REVISION.txt` files record the pinned Hugging Face revisions.

VLM inputs are RGB frames from each task's `data/raw` or `prl_data/raw` NPZ
recordings. Use `meta/preparation.json` for episode mappings and prepared
Parquet columns for raw-frame indices and wall timestamps. The existing
`utils.online_benchmark.preload_episode` reader can read these with
`method="vlm"`, which does not require action alignment. VAE latents and ACT
tokens are not the VLM visual inputs. No dataset regeneration is needed.
