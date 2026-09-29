import os
from pathlib import Path
import torch
from PIL import Image, ExifTags
from tqdm import tqdm
from dataclasses import dataclass
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import pipeline
from typing import cast

from pipeline.pipeline_qwenimage_edit import QwenImageEditPipeline
from cache_functions.cache_utils import pipeline_with_taylorseer
from cache_functions import cache_init


@dataclass
class SamplingOptions:
    dataset_path: str           # Path to the GEdit dataset
    negative_prompt: str        # Negative prompt for guidance
    num_steps: int              # Number of sampling steps
    guidance_scale: float       # Guidance scale
    seed: int                   # Random seed
    model_name: str             # Model name
    output_dir: str             # Output directory
    english_only: bool          # Whether to process only English tasks
    test_FLOPs: bool            # Whether in FLOPs test mode
    monitor_gpu_usage: bool     # Whether to monitor GPU memory usage
    interval: int               # Cache period length
    max_order: int              # Maximum order of Taylor expansion
    min_order: int              # Minimum order of Taylor expansion
    first_enhance: int          # Initial enhancement steps
    forecast_method: str        # Forecast method
    decompose_method: str       # Decomposition method
    use_z_cache: bool           # Use Z cache
    forecast_steps: int         # Forecast steps


def create_folders(output_dir: str, task_types: list[str], languages: list[str]):
    """Create directory structure"""
    base_dir = Path(output_dir) / "fullset"

    for task_type in task_types:
        for lang in languages:
            task_dir = base_dir / task_type / lang
            task_dir.mkdir(parents=True, exist_ok=True)


def check_images(output_dir: str, task_type: str, instruction_language: str, key: str) -> bool:
    """Check if image has already been generated"""
    image_path = Path(output_dir) / "fullset" / task_type / instruction_language / f"{key}.png"
    return image_path.exists()


def save_image(image: Image.Image, output_dir: str, task_type: str, instruction_language: str, key: str):
    """Save generated image to corresponding directory"""
    save_dir = Path(output_dir) / "fullset" / task_type / instruction_language
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{key}.png"
    image.save(save_path)


def load_gedit_dataset(dataset_path: str):
    """Load GEdit-Bench from a local datasets cache or the Hugging Face Hub."""
    if Path(dataset_path).expanduser().exists():
        dataset = load_from_disk(str(Path(dataset_path).expanduser()))
    else:
        dataset = load_dataset(dataset_path)

    if isinstance(dataset, DatasetDict):
        split = "train" if "train" in dataset else next(iter(dataset))
        dataset = dataset[split]
    return dataset


def main(opts: SamplingOptions):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    print(f"Loading dataset: {opts.dataset_path}")
    dataset = load_gedit_dataset(opts.dataset_path)
    print(f"Dataset loaded, total {len(dataset)} samples")

    # Get task types and languages
    task_types: list[str] = list(set(str(dataset[i]["task_type"]) for i in range(len(dataset))))
    languages = ["en", "cn"] if not opts.english_only else ["en"]

    print(f"Found task types: {task_types}")
    if opts.english_only:
        print("English-only mode enabled: will skip Chinese tasks")

    # Create output directory structure
    create_folders(opts.output_dir, task_types, languages)

    # Load pipeline
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16
    ).to(device=device)

    pipe = pipeline_with_taylorseer(pipe)

    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)

    # Process dataset
    processed_count = 0
    skipped_count = 0

    progress_bar = tqdm(total=len(dataset), desc="Generating images")

    for i in range(len(dataset)):
        item = dataset[i]
        task_type: str = str(item["task_type"])
        instruction: str = str(item["instruction"])
        instruction_language: str = str(item["instruction_language"])
        key: str = str(item["key"])
        input_image = item["input_image"]

        # Skip Chinese tasks (if english_only enabled)
        if opts.english_only and instruction_language == "cn":
            progress_bar.update(1)
            skipped_count += 1
            continue

        # Check if already exists
        if check_images(opts.output_dir, task_type, instruction_language, key):
            progress_bar.update(1)
            skipped_count += 1
            continue

        # Set random seed
        seed = opts.seed + i

        batch_prompts = [instruction]
        input_image = cast(Image.Image, input_image)

        # Initialize cache
        cache_dic, current = cache_init(kwargs={
            'num_steps': opts.num_steps,
            'test_FLOPs': opts.test_FLOPs,
            'monitor_gpu_usage': opts.monitor_gpu_usage,
            'interval': opts.interval,
            'max_order': opts.max_order,
            'min_order': opts.min_order,
            'first_enhance': opts.first_enhance,
            'forecast_method': opts.forecast_method,
            'decompose_method': opts.decompose_method,
            'use_z_cache': opts.use_z_cache,
            'forecast_steps': opts.forecast_steps
        })

        # Generate images
        result = pipe(
            image=input_image,
            prompt=batch_prompts,
            negative_prompt=opts.negative_prompt,
            num_inference_steps=opts.num_steps,
            guidance_scale=opts.guidance_scale,
            generator=torch.Generator(device).manual_seed(int(seed)),
            cache_dic=cache_dic,
            current=current
        )

        images = getattr(result, 'images', None)
        if images is None:
            if isinstance(result, (list, tuple)):
                images = list(result)
            else:
                images = [result]

        output_image = images[0]
        save_image(output_image, opts.output_dir, task_type, instruction_language, key)
        processed_count += 1

        progress_bar.update(1)

    progress_bar.close()

    print(f"Generation complete! Processed {processed_count} images, skipped {skipped_count} images")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate GEdit-Bench images using Qwen Image Edit model.")
    parser.add_argument('--dataset_path', type=str, default='stepfun-ai/GEdit-Bench', help='Local dataset path or Hugging Face dataset ID.')
    parser.add_argument('--negative_prompt', type=str, default=' ', help='Negative prompt for guidance.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps.')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--model_name', type=str, default='qwen-image-edit', help='Model name.')
    parser.add_argument('--output_dir', type=str, default='results/test', help='Directory to save images.')
    parser.add_argument('--english_only', action='store_true', help='Whether to process only English tasks.')
    parser.add_argument('--test_FLOPs', action='store_true', help='Test inference computation cost.')
    parser.add_argument('--monitor_gpu_usage', action='store_true', help='Monitor GPU memory usage during sampling.')

    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--max_order', type=int, default=2)
    parser.add_argument('--min_order', type=int, default=0)
    parser.add_argument('--first_enhance', type=int, default=3)
    parser.add_argument('--forecast_method', type=str, default='hermite', choices=['hermite', 'taylor'])
    parser.add_argument('--decompose_method', type=str, default='FFT', choices=['None', 'FFT', 'DCT'])
    parser.add_argument('--use_z_cache', action='store_true')
    parser.add_argument('--forecast_steps', type=int, default=10)

    args = parser.parse_args()

    opts = SamplingOptions(
        dataset_path=args.dataset_path,
        negative_prompt=args.negative_prompt,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        model_name=args.model_name,
        output_dir=args.output_dir,
        english_only=args.english_only,
        test_FLOPs=args.test_FLOPs,
        monitor_gpu_usage=args.monitor_gpu_usage,
        interval=args.interval,
        max_order=args.max_order,
        min_order=args.min_order,
        first_enhance=args.first_enhance,
        forecast_method=args.forecast_method,
        decompose_method=args.decompose_method,
        use_z_cache=args.use_z_cache,
        forecast_steps=args.forecast_steps,
    )

    main(opts)
    # CUDA_VISIBLE_DEVICES=6 python sample_edit.py --model_name qwen-image-edit --interval 6 --output_dir results/N6
    # CUDA_VISIBLE_DEVICES=7 python sample_edit.py --model_name qwen-image-edit --interval 9 --output_dir results/N9
