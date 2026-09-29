# FreqCa: Accelerating Image Generation and Editing via Frequency-Aware Caching

Official implementation of **FreqCa**, accepted at **NeurIPS 2026**.

Jiacheng Liu\*, Peiliang Cai\*, Qinming Zhou, Yuqi Lin, Deyang Kong, Benhao
Huang, Yupei Pan, Haowen Xu, Chang Zou, Junshu Tang, Shikang Zheng, Linfeng
Zhang†

\* Equal contribution. † Corresponding author.

FreqCa is a training-free cache-and-forecast method for accelerating diffusion
transformers. It separates cached features into frequency bands: stable
low-frequency components are reused, while temporally continuous high-frequency
components are forecast with a second-order Hermite predictor. Caching only the
Cumulative Residual Feature (CRF) keeps cache memory independent of transformer
depth.

## Highlights

- **Training-free:** no fine-tuning, calibration set, or architecture-specific
  training is required.
- **Generation and editing:** supports FLUX.1-dev, FLUX.1-schnell,
  FLUX.1-Kontext-dev, Qwen-Image, and Qwen-Image-Edit.
- **Fast:** up to 5.47× measured latency speedup on FLUX.1-dev and 5.68× on
  Qwen-Image in the paper's 50-step setting.
- **Memory-efficient:** CRF caching has `O(1)` memory complexity with respect to
  model depth; the FLUX.1-dev experiment used 0.18 GB additional cache memory.

## Repository layout

```text
FreqCa/
├── flux/                       # FLUX generation/editing and evaluation
│   ├── src/flux/               # FLUX model and FreqCa implementation
│   ├── src/sample.py           # Single-GPU sampling
│   ├── src/sample_ddp.py       # Data-parallel prompt sampling
│   └── src/sample_gedit.py     # FLUX Kontext on GEdit-Bench
├── qwen_image/                 # Qwen-Image generation/editing
│   ├── cache_functions/        # Frequency decomposition and forecasting
│   ├── pipeline/               # FreqCa-enabled Diffusers pipelines
│   ├── sample.py               # Single-GPU sampling
│   ├── sample_ddp.py           # Data-parallel prompt sampling
│   └── sample_edit.py          # Qwen-Image-Edit on GEdit-Bench
└── scripts/check_release.py    # Public-release hygiene check
```

The two backends are intentionally kept separate because they build on
different upstream inference stacks. Their DDP scripts replicate one model per
GPU and shard prompts across processes; they do not tensor-parallelize a single
sample.

## Installation

Python 3.10 or 3.11 and a CUDA-capable PyTorch environment are recommended.
Create separate environments for the two backends if your existing Diffusers
or Transformers versions are constrained.

### FLUX

```bash
conda create -n freqca-flux python=3.10 -y
conda activate freqca-flux
pip install -e "./flux[torch]"
```

For the provided quality-evaluation script, install the optional dependencies:

```bash
pip install -e "./flux[eval]" calflops
```

### Qwen-Image

```bash
conda create -n freqca-qwen python=3.10 -y
conda activate freqca-qwen
pip install -r qwen_image/requirements.txt
```

Weights are downloaded from Hugging Face on first use. Authenticate first for
gated checkpoints and accept the checkpoint's terms on its model page:

```bash
hf auth login
```

## Quick start

All commands below are run from the corresponding backend directory. Setting
`--interval 1` runs every transformer step and serves as the no-cache baseline.

### FLUX.1-dev text-to-image

The paper used DCT for FLUX models. `N=6` is a strong quality/speed setting;
`N=9` is the aggressive setting.

```bash
cd flux
CUDA_VISIBLE_DEVICES=0 python src/sample.py \
  --model_name flux-dev \
  --prompt_file prompts/DrawBench200.txt \
  --num_steps 50 \
  --interval 6 \
  --max_order 2 \
  --forecast_method hermite \
  --decompose_method DCT \
  --output_dir samples/flux-dev-n6
```

For FLUX.1-schnell, use `--model_name flux-schnell --num_steps 4 --interval 2`.

### FLUX.1-Kontext image editing

```bash
cd flux
CUDA_VISIBLE_DEVICES=0 python src/sample.py \
  --model_name flux-dev-kontext \
  --input_image img.jpg \
  --prompt_file prompts/parti_prompts.txt \
  --num_steps 50 \
  --interval 6 \
  --decompose_method DCT \
  --output_dir samples/kontext-n6
```

### Qwen-Image text-to-image

The paper used FFT for Qwen models.

```bash
cd qwen_image
CUDA_VISIBLE_DEVICES=0 python sample.py \
  --model_name qwen-image \
  --prompt_file prompts/DrawBench200.txt \
  --num_steps 50 \
  --interval 6 \
  --max_order 2 \
  --forecast_method hermite \
  --decompose_method FFT \
  --output_dir samples/qwen-image-n6
```

### Qwen-Image-Edit

```bash
cd qwen_image
CUDA_VISIBLE_DEVICES=0 python sample.py \
  --model_name qwen-image-edit \
  --input_image img.jpg \
  --prompt_file prompts/prompts_for_edit.txt \
  --num_steps 50 \
  --interval 6 \
  --decompose_method FFT \
  --output_dir samples/qwen-image-edit-n6
```

### Multi-GPU prompt sampling

Use the matching `sample_ddp.py` entry point. Each process handles a disjoint
prompt shard:

```bash
# Run from flux/
torchrun --standalone --nproc_per_node=8 src/sample_ddp.py \
  --model_name flux-dev --interval 6 --decompose_method DCT \
  --prompt_file prompts/DrawBench200.txt \
  --output_dir samples/flux-dev-n6

# Run from qwen_image/
torchrun --standalone --nproc_per_node=8 sample_ddp.py \
  --model_name qwen-image --interval 6 --decompose_method FFT \
  --prompt_file prompts/DrawBench200.txt \
  --output_dir samples/qwen-image-n6
```

## GEdit-Bench

The benchmark entry points accept either a Hugging Face dataset ID or a local
directory previously written by `datasets.save_to_disk`.

```bash
# FLUX.1-Kontext-dev (run from flux/)
torchrun --standalone --nproc_per_node=8 src/sample_gedit.py \
  --dataset_path stepfun-ai/GEdit-Bench \
  --interval 6 --decompose_method DCT \
  --output_dir samples/gedit/flux-kontext-n6

# Qwen-Image-Edit (run from qwen_image/)
CUDA_VISIBLE_DEVICES=0 python sample_edit.py \
  --dataset_path stepfun-ai/GEdit-Bench \
  --interval 6 --decompose_method FFT \
  --output_dir samples/gedit/qwen-edit-n6
```

Add `--english_only` for the English subset. Existing output images are skipped,
so interrupted runs can be resumed.

## Cache parameters

| Argument | Meaning | Paper default |
|---|---|---:|
| `--interval N` | Perform one full computation every `N` denoising steps | 6 or 9 |
| `--first_enhance` | Number of initial full-computation steps | 3 |
| `--max_order` | Maximum forecasting order | 2 |
| `--min_order` | Minimum forecasting order while history is short | 0 |
| `--forecast_method` | High-frequency predictor (`hermite` or `taylor`) | `hermite` |
| `--decompose_method` | Frequency transform (`DCT`, `FFT`, or `None`) | DCT for FLUX; FFT for Qwen |

The frequency cutoff is 0.1 in `cache_utils.py`. Larger intervals are faster but
more aggressive. Always compare against `--interval 1` using the same prompts,
seeds, resolution, step count, warm-up, and hardware. `--use_z_cache` exposes an
experimental look-ahead schedule; when enabled, `forecast_steps` must be at
least `interval`.

## Paper results

Representative 50-step results are shown below. FLOPs speedup and measured
latency speedup are different quantities. FLUX experiments used an NVIDIA A100;
Qwen experiments used an NVIDIA H20.

| Model / task | Interval | FLOPs speedup | Latency speedup | Quality |
|---|---:|---:|---:|---:|
| FLUX.1-dev / DrawBench | 6 | 4.99× | 4.48× | ImageReward 1.01 (+2.0%) |
| FLUX.1-dev / DrawBench | 9 | 6.24× | 5.47× | ImageReward 0.97 (-2.0%) |
| Qwen-Image / DrawBench | 6 | 5.00× | 4.28× | ImageReward 1.20 (-4.0%) |
| Qwen-Image / DrawBench | 9 | 7.14× | 5.68× | ImageReward 1.02 (-18.4%) |
| FLUX.1-Kontext-dev / GEdit-EN | 9 | 6.24× | 5.74× | Overall 6.190 (-0.4%) |
| Qwen-Image-Edit / GEdit-CN | 9 | 6.24× | 5.57× | Overall 7.27 (-1.9%) |
| Qwen-Image-Edit / GEdit-EN | 9 | 6.24× | 5.57× | Overall 7.21 (-4.3%) |

Latency depends on software versions, attention kernels, resolution, batch size,
and GPU. Report both absolute latency and speedup against a locally measured
baseline.

## Evaluation

DrawBench prompts are included in both backends. From `flux/`, compute
CLIPScore, ImageReward, PSNR, SSIM, and LPIPS with:

```bash
python evaluate.py \
  --test_folder samples/flux-dev-n6 \
  --prompt_file prompts/DrawBench200.txt \
  --reference_folder samples/flux-dev-baseline
```

Use `--test_FLOPs` to invoke `calflops` profiling and
`--monitor_gpu_usage` to print memory statistics. FLOPs profiling adds overhead
and should not be used for latency measurement.

The GEdit evaluator is `flux/evaluate_gedit.py`. Its VIEScore backends may
require additional model packages or API credentials; keep credentials in an
ignored `secret.env` file and never commit them.

## License

The FreqCa source code is released under the [Apache License 2.0](LICENSE).
This does **not** relicense model weights or datasets. Review the terms for every
checkpoint and dataset you download. See [third-party notices](THIRD_PARTY_NOTICES.md)
for upstream code attribution, including the separate licenses governing FLUX
model variants.

## Citation

If FreqCa is useful in your research, please cite:

```bibtex
@inproceedings{liu2026freqca,
  title     = {FreqCa: Accelerating Image Generation and Editing via Frequency-Aware Caching},
  author    = {Liu, Jiacheng and Cai, Peiliang and Zhou, Qinming and Lin, Yuqi and
               Kong, Deyang and Huang, Benhao and Pan, Yupei and Xu, Haowen and
               Zou, Chang and Tang, Junshu and Zheng, Shikang and Zhang, Linfeng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

This repository builds on FLUX, Hugging Face Diffusers, Qwen-Image, DrawBench,
GEdit-Bench, ImageReward, and VIEScore. We thank their authors and maintainers.
