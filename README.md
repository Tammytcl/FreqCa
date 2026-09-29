<div align="center">

# [NeurIPS 2026] FreqCa

### Accelerating Image Generation and Editing via Frequency-Aware Caching

Jiacheng Liu<sup>1,2,3,*</sup>, Peiliang Cai<sup>1,*</sup>, Qinming Zhou<sup>1,4</sup>,
Yuqi Lin<sup>1,5</sup>, Deyang Kong<sup>1,7</sup>, Benhao Huang<sup>1,6</sup>,
Yupei Pan<sup>1,7</sup>, Haowen Xu<sup>1</sup>, Chang Zou<sup>1,2,7</sup>,
Junshu Tang<sup>2</sup>, Shikang Zheng<sup>1,8</sup>, Linfeng Zhang<sup>1,†</sup>

<sup>1</sup>EPIC Lab, SJTU &nbsp; <sup>2</sup>Tencent Hunyuan &nbsp;
<sup>3</sup>SDU &nbsp; <sup>4</sup>THU &nbsp; <sup>5</sup>JLU &nbsp;
<sup>6</sup>CMU &nbsp; <sup>7</sup>UESTC &nbsp; <sup>8</sup>SCUT

<sup>*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding author

<p>
  <img src="https://img.shields.io/badge/NeurIPS-2026-4b44ce" alt="NeurIPS 2026">
  <img src="https://img.shields.io/badge/Paper-coming_soon-lightgrey" alt="Paper coming soon">
  <a href="https://github.com/Tammytcl/FreqCa"><img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="Code"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License"></a>
  <a href="https://github.com/Tammytcl/FreqCa/stargazers"><img src="https://img.shields.io/github/stars/Tammytcl/FreqCa?style=social" alt="GitHub stars"></a>
</p>

**Training-free · 6–7× FLOPs acceleration · Generation and editing · O(1) cache memory**

[News](#-news) · [Method](#-method) · [Results](#-main-results) ·
[Installation](#-installation) · [FLUX](#-freqca-flux) ·
[Qwen-Image](#-freqca-qwen-image) · [Evaluation](#-evaluation)

</div>

## 🔥 News

- **2026/09/29** — The inference code for FLUX and Qwen-Image is released.
- **2026** — FreqCa is accepted at **NeurIPS 2026**. 🎉

<details>
<summary><strong>Abstract</strong></summary>

Diffusion transformers deliver strong image generation and editing quality, but
their iterative inference is expensive. Existing feature-caching methods
typically assume that adjacent diffusion features are uniformly similar or
continuous. FreqCa shows that this assumption behaves differently across
frequency bands: low-frequency components are highly similar but difficult to
extrapolate, while high-frequency components are less similar yet more
temporally continuous. Based on this observation, FreqCa directly reuses the
low-frequency component and forecasts the high-frequency component with a
second-order Hermite predictor. It additionally caches only the Cumulative
Residual Feature (CRF), instead of layer-wise activations, reducing cache memory
by approximately 99%. Experiments on FLUX.1-dev, FLUX.1-Kontext-dev,
Qwen-Image, and Qwen-Image-Edit demonstrate strong acceleration for both image
generation and editing without additional training.

</details>

## 💡 Method

FreqCa applies a different strategy to each frequency band of the Cumulative
Residual Feature:

```text
                         Cumulative Residual Feature (CRF)
                                      │
                              Frequency split
                          ┌───────────┴───────────┐
                          │                       │
                  Low-frequency CRF      High-frequency CRF
                  stable structure       continuous details
                          │                       │
                     direct reuse         Hermite forecasting
                          └───────────┬───────────┘
                                      │
                              predicted CRF
```

- **Frequency-aware caching:** reuse stable low-frequency structure and
  forecast evolving high-frequency details.
- **Second-order Hermite predictor:** preserves local trajectory curvature under
  aggressive cache intervals.
- **CRF-only cache:** changes cache-memory complexity from `O(L)` to `O(1)` with
  respect to transformer depth.
- **Plug-and-play:** requires no training or calibration data.

## 📊 Main Results

Representative results from the 50-step experiments are shown below. FLUX was
measured on an NVIDIA A100 and Qwen-Image on an NVIDIA H20.

| Model / benchmark | Interval | FLOPs speedup | Latency speedup | Quality |
|:--|--:|--:|--:|--:|
| FLUX.1-dev / DrawBench | 6 | 4.99× | 4.48× | ImageReward **1.01** (+2.0%) |
| FLUX.1-dev / DrawBench | 9 | 6.24× | **5.47×** | ImageReward **0.97** (-2.0%) |
| Qwen-Image / DrawBench | 6 | 5.00× | 4.28× | ImageReward **1.20** (-4.0%) |
| Qwen-Image / DrawBench | 9 | **7.14×** | **5.68×** | ImageReward **1.02** (-18.4%) |
| FLUX.1-Kontext-dev / GEdit-EN | 9 | 6.24× | **5.74×** | Overall **6.190** (-0.4%) |
| Qwen-Image-Edit / GEdit-CN | 9 | 6.24× | **5.57×** | Overall **7.27** (-1.9%) |
| Qwen-Image-Edit / GEdit-EN | 9 | 6.24× | **5.57×** | Overall **7.21** (-4.3%) |

> [!NOTE]
> FLOPs speedup and wall-clock speedup are different quantities. Latency varies
> with GPU, attention kernels, resolution, batch size, and software versions.
> Always compare with a baseline measured in the same environment.

## 🛠 Installation

```bash
git clone https://github.com/Tammytcl/FreqCa.git
cd FreqCa
```

Python 3.10 or 3.11 and a CUDA-capable PyTorch environment are recommended.
Because the two model families use different upstream inference stacks, separate
environments are the safest setup.

### FLUX environment

```bash
conda create -n freqca-flux python=3.10 -y
conda activate freqca-flux
pip install -e "./flux[torch]"
```

### Qwen-Image environment

```bash
conda create -n freqca-qwen python=3.10 -y
conda activate freqca-qwen
pip install -r qwen_image/requirements.txt
```

Weights are downloaded from Hugging Face on first use. For gated checkpoints,
accept the model terms and authenticate with:

```bash
hf auth login
```

## ⚡ FreqCa-FLUX

FreqCa supports **FLUX.1-dev**, **FLUX.1-schnell**, **FLUX.1-Kontext-dev**, and
**FLUX.1-Fill-dev**. The paper uses DCT decomposition for FLUX models.

### Text-to-image generation

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

Use `--interval 9` for the aggressive 6.24× FLOPs setting. For FLUX.1-schnell,
use `--model_name flux-schnell --num_steps 4 --interval 2`.

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

## ⚡ FreqCa-Qwen-Image

FreqCa supports **Qwen-Image** and **Qwen-Image-Edit**. The paper uses FFT
decomposition for Qwen models.

### Text-to-image generation

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

### Image editing

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

Use `--interval 9` for the aggressive 7.14× FLOPs Qwen-Image setting.

## 🚀 Multi-GPU Sampling

The DDP entry points replicate one model per GPU and shard prompts across
processes. They accelerate benchmark generation throughput; they do not
tensor-parallelize one sample.

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

## 🧪 Evaluation

### GEdit-Bench generation

The editing entry points accept either a Hugging Face dataset ID or a local
directory created by `datasets.save_to_disk`.

```bash
# FLUX.1-Kontext-dev — run from flux/
torchrun --standalone --nproc_per_node=8 src/sample_gedit.py \
  --dataset_path stepfun-ai/GEdit-Bench \
  --interval 6 --decompose_method DCT \
  --output_dir samples/gedit/flux-kontext-n6

# Qwen-Image-Edit — run from qwen_image/
CUDA_VISIBLE_DEVICES=0 python sample_edit.py \
  --dataset_path stepfun-ai/GEdit-Bench \
  --interval 6 --decompose_method FFT \
  --output_dir samples/gedit/qwen-edit-n6
```

Add `--english_only` for the English subset. Existing output images are skipped,
so interrupted runs can be resumed.

### DrawBench metrics

Install the FLUX evaluation dependencies and compute CLIPScore, ImageReward,
PSNR, SSIM, and LPIPS:

```bash
pip install -e "./flux[eval]" calflops
cd flux
python evaluate.py \
  --test_folder samples/flux-dev-n6 \
  --prompt_file prompts/DrawBench200.txt \
  --reference_folder samples/flux-dev-baseline
```

The GEdit evaluator is `flux/evaluate_gedit.py`. Its VIEScore backends may need
additional model packages or API credentials. Keep credentials in the ignored
`secret.env` file and never commit them.

<details>
<summary><strong>Cache parameters and profiling options</strong></summary>

| Argument | Meaning | Paper setting |
|:--|:--|--:|
| `--interval N` | Perform one full computation every `N` denoising steps | 6 or 9 |
| `--first_enhance` | Number of initial full-computation steps | 3 |
| `--max_order` | Maximum forecasting order | 2 |
| `--min_order` | Minimum forecasting order while history is short | 0 |
| `--forecast_method` | High-frequency predictor | `hermite` |
| `--decompose_method` | Frequency transform | DCT for FLUX; FFT for Qwen |

The frequency cutoff is 0.1 in `cache_utils.py`. `--interval 1` disables
skipping and provides the full-compute baseline. Larger intervals are faster
but more aggressive. Use the same prompts, seeds, resolution, denoising steps,
warm-up, and hardware for comparisons.

Use `--test_FLOPs` for `calflops` profiling and `--monitor_gpu_usage` for memory
statistics. FLOPs profiling adds overhead and should not be used for latency
measurement. `--use_z_cache` enables an experimental look-ahead schedule; its
`forecast_steps` value must be at least `interval`.

</details>

<details>
<summary><strong>Repository structure</strong></summary>

```text
FreqCa/
├── flux/
│   ├── src/flux/               # FLUX model and FreqCa implementation
│   ├── src/sample.py           # Single-GPU generation and editing
│   ├── src/sample_ddp.py       # Data-parallel prompt sampling
│   └── src/sample_gedit.py     # FLUX Kontext on GEdit-Bench
├── qwen_image/
│   ├── cache_functions/        # Frequency split, cache, and forecast
│   ├── pipeline/               # FreqCa-enabled Diffusers pipelines
│   ├── sample.py               # Single-GPU generation and editing
│   ├── sample_ddp.py           # Data-parallel prompt sampling
│   └── sample_edit.py          # Qwen-Image-Edit on GEdit-Bench
└── scripts/check_release.py    # Public-release hygiene check
```

</details>

## 👍 Acknowledgements

We thank the authors and maintainers of
[FLUX](https://github.com/black-forest-labs/flux),
[Diffusers](https://github.com/huggingface/diffusers),
[Qwen-Image](https://github.com/QwenLM/Qwen-Image),
[DrawBench](https://arxiv.org/abs/2205.11487),
[GEdit-Bench](https://github.com/stepfun-ai/Step1X-Edit),
[ImageReward](https://github.com/THUDM/ImageReward), and VIEScore.

The FreqCa source code is released under the [Apache License 2.0](LICENSE).
Model weights and datasets retain their own licenses and acceptable-use terms.
See [third-party notices](THIRD_PARTY_NOTICES.md) for details.

## 📌 Citation

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

## 📧 Contact

For questions and discussions, please open a GitHub issue or contact
[Linfeng Zhang](mailto:zhanglinfeng@sjtu.edu.cn).

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tammytcl/FreqCa&type=Date)](https://www.star-history.com/#Tammytcl/FreqCa&Date)
