import argparse
import csv
import math
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, cast

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from viescore import VIEScore

FIELDNAMES = [
    "key",
    "edited_image",
    "instruction",
    "sementics_score",
    "quality_score",
    "overall_score",
    "intersection_exist",
    "instruction_language",
]


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    new_area = width * height
    return int(width), int(height), int(new_area)


def process_item(item, vie_score, max_retries=10000):
    for _ in range(max_retries):
        pil_image = item["input_image_raw"].convert("RGB")
        pil_image_edited = Image.open(item["edited_image_path"]).convert("RGB")
        source_img_width, source_img_height, source_img_area = calculate_dimensions(
            512 * 512, pil_image.width / pil_image.height
        )
        edited_img_width, edited_img_height, edited_img_area = calculate_dimensions(
            512 * 512, pil_image_edited.width / pil_image_edited.height
        )
        pil_image = pil_image.resize((int(source_img_width), int(source_img_height)))
        pil_image_edited = pil_image_edited.resize((int(edited_img_width), int(edited_img_height)))
        text_prompt = item["instruction"]
        score_list = vie_score.evaluate([pil_image, pil_image_edited], text_prompt)
        sementics_score, quality_score, overall_score = score_list
        assert not np.isnan(sementics_score) and not np.isnan(quality_score) and not np.isnan(overall_score), f"sementics_score or quality_score or overall_score is nan!"

        return {
            "key": item["key"],
            "edited_image": item["edited_image_path"],
            "instruction": item["instruction"],
            "sementics_score": sementics_score,
            "quality_score": quality_score,
            "overall_score": overall_score,
            "intersection_exist": item["Intersection_exist"],
            "instruction_language": item["instruction_language"],
        }


def append_result(tmp_dir, group_name, rank, row):
    group_dir = os.path.join(tmp_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)
    tmp_file = os.path.join(group_dir, f"rank_{rank}.csv")
    file_exists = os.path.exists(tmp_file)
    with open(tmp_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_results(tmp_dir, groups):
    processed = {group: set() for group in groups}

    for group_name in groups:
        group_tmp_dir = os.path.join(tmp_dir, group_name)
        if os.path.isdir(group_tmp_dir):
            rank_files = glob.glob(os.path.join(group_tmp_dir, "rank_*.csv"))
            for path in rank_files:
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    key = row["key"]
                    instruction_language = row["instruction_language"]
                    processed[group_name].add((str(key), instruction_language))

    return processed


def analyze_scores(save_path_dir, language, groups):
    group_scores_semantics = {}
    group_scores_quality = {}
    group_scores_overall = {}

    group_scores_semantics_intersection = {}
    group_scores_quality_intersection = {}
    group_scores_overall_intersection = {}

    for group_name in groups:
        csv_path = os.path.join(save_path_dir, "score", f"{group_name}.csv")

        if not os.path.exists(csv_path):
            print(f"Skipping group {group_name}")
            continue

        df = pd.read_csv(csv_path)

        filtered_semantics_scores = []
        filtered_quality_scores = []
        filtered_overall_scores = []
        filtered_semantics_scores_intersection = []
        filtered_quality_scores_intersection = []
        filtered_overall_scores_intersection = []

        for _, row in df.iterrows():
            semantics_score = row["sementics_score"]
            quality_score = row["quality_score"]
            overall_score = row["overall_score"]
            intersection_exist = row["intersection_exist"]
            instruction_language = row["instruction_language"]

            if instruction_language != language:
                continue

            filtered_semantics_scores.append(semantics_score)
            filtered_quality_scores.append(quality_score)
            filtered_overall_scores.append(overall_score)
            if bool(intersection_exist):
                filtered_semantics_scores_intersection.append(semantics_score)
                filtered_quality_scores_intersection.append(quality_score)
                filtered_overall_scores_intersection.append(overall_score)

        group_scores_semantics[group_name] = np.mean(filtered_semantics_scores)
        group_scores_quality[group_name] = np.mean(filtered_quality_scores)
        group_scores_overall[group_name] = np.mean(filtered_overall_scores)

        group_scores_semantics_intersection[group_name] = np.mean(filtered_semantics_scores_intersection)
        group_scores_quality_intersection[group_name] = np.mean(filtered_quality_scores_intersection)
        group_scores_overall_intersection[group_name] = np.mean(filtered_overall_scores_intersection)

    def avg_of(dct):
        vals = []
        for g in groups:
            if g in dct:
                v = dct[g]
                vals.append(v)
        return np.mean(vals)

    group_scores_semantics["avg_semantics"] = avg_of(group_scores_semantics)
    group_scores_quality["avg_quality"] = avg_of(group_scores_quality)
    group_scores_overall["avg_overall"] = avg_of(group_scores_overall)

    group_scores_semantics_intersection["avg_semantics"] = avg_of(group_scores_semantics_intersection)
    group_scores_quality_intersection["avg_quality"] = avg_of(group_scores_quality_intersection)
    group_scores_overall_intersection["avg_overall"] = avg_of(group_scores_overall_intersection)

    return (
        group_scores_semantics,
        group_scores_quality,
        group_scores_overall,
        group_scores_semantics_intersection,
        group_scores_quality_intersection,
        group_scores_overall_intersection,
    )


def save_scores(
    save_dir,
    language,
    groups,
    group_scores_semantics,
    group_scores_quality,
    group_scores_overall,
    group_scores_semantics_intersection,
    group_scores_quality_intersection,
    group_scores_overall_intersection,
):
    print(f"\nLanguage: {language}")
    records = []

    print("Overall:")
    for group_name in groups:
        if group_name in group_scores_semantics:
            semantics = group_scores_semantics[group_name]
            quality = group_scores_quality[group_name]
            overall = group_scores_overall[group_name]
            print(f"{group_name}: {semantics:.4f}, {quality:.4f}, {overall:.4f}")
            records.append({"Language": language, "Type": "Overall", "Group": group_name, "Semantics": semantics, "Quality": quality, "Overall": overall})
        else:
            print(f"{group_name}: No data available")

    avg_semantics = group_scores_semantics["avg_semantics"]
    avg_quality = group_scores_quality["avg_quality"]
    avg_overall = group_scores_overall["avg_overall"]
    print(f"Average: {avg_semantics:.4f}, {avg_quality:.4f}, {avg_overall:.4f}")
    records.append({"Language": language, "Type": "Overall", "Group": "Average", "Semantics": avg_semantics, "Quality": avg_quality, "Overall": avg_overall})


    print("\nIntersection:")
    for group_name in groups:
        if group_name in group_scores_semantics_intersection:
            semantics = group_scores_semantics_intersection[group_name]
            quality = group_scores_quality_intersection[group_name]
            overall = group_scores_overall_intersection[group_name]
            print(f"{group_name}: {semantics:.4f}, {quality:.4f}, {overall:.4f}")
            records.append({"Language": language, "Type": "Intersection", "Group": group_name, "Semantics": semantics, "Quality": quality, "Overall": overall})
        else:
            print(f"{group_name}: No intersection data available")

    avg_semantics_intersection = group_scores_semantics_intersection["avg_semantics"]
    avg_quality_intersection = group_scores_quality_intersection["avg_quality"]
    avg_overall_intersection = group_scores_overall_intersection["avg_overall"]
    print(f"Average Intersection: {avg_semantics_intersection:.4f}, {avg_quality_intersection:.4f}, {avg_overall_intersection:.4f}")
    records.append({"Language": language, "Type": "Intersection", "Group": "Average", "Semantics": avg_semantics_intersection, "Quality": avg_quality_intersection, "Overall": avg_overall_intersection})

    df_scores = pd.DataFrame(records)
    output_path = os.path.join(save_dir, "score", "scores.csv")
    df_scores.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False, float_format='%.4f')


def main(args):
    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    rank = 0
    world_size = 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)


    if args.task_type == "all":
        groups = [
            "background_change",
            "color_alter",
            "material_alter",
            "motion_change",
            "ps_human",
            "style_change",
            "subject-add",
            "subject-remove",
            "subject-replace",
            "text_change",
            "tone_transfer",
        ]
    else:
        groups = [args.task_type]

    score_dir = os.path.join(args.save_dir, "score")
    tmp_dir = os.path.join(args.save_dir, "tmp")
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    processed_by_group = load_existing_results(tmp_dir, groups)

    vie_score = VIEScore(backbone=args.backbone, task="tie", key_path="secret.env")
    dataset = load_dataset("stepfun-ai/GEdit-Bench", split="train")

    target_groups = set(groups)
    filtered_items = [
        it
        for it in cast(Iterable[Dict[str, Any]], dataset)
        if (args.instruction_language == "all" or it["instruction_language"] == args.instruction_language)
        and it["task_type"] in target_groups
    ]
    local_items = [it for i, it in enumerate(filtered_items) if i % world_size == rank]

    remaining_items = []
    for item in local_items:
        group_name = item["task_type"]
        key = item["key"]
        instruction_language = item["instruction_language"]
        identifier = (str(key), instruction_language)

        if identifier and identifier in processed_by_group.get(group_name, set()):
            continue
        remaining_items.append(item)

    skipped = len(local_items) - len(remaining_items)
    if skipped > 0:
        print(f"[Rank {rank}] Skipped {skipped} items already processed.")

    if args.backbone == "gpt4o":
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_item = {}
            for item in remaining_items:
                key = item["key"]
                group_name = item["task_type"]
                edited_images_path = os.path.join(args.save_dir, "fullset", group_name, item["instruction_language"])
                item["edited_image_path"] = os.path.join(edited_images_path, f"{key}.png")
                future = executor.submit(process_item, item, vie_score)
                future_to_item[future] = item
            for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Processing items"):
                item = future_to_item[future]
                group_name = item["task_type"]
                res = future.result()
                if res:
                    append_result(tmp_dir, group_name, rank, res)
                    key = res["key"]
                    instruction_language = res["instruction_language"]
                    identifier = (str(key), instruction_language)
                    processed_by_group[group_name].add(identifier)
    else:
        for item in tqdm(remaining_items, desc="Processing items"):
            key = item["key"]
            group_name = item["task_type"]
            edited_images_path = os.path.join(args.save_dir, "fullset", group_name, item["instruction_language"])
            item["edited_image_path"] = os.path.join(edited_images_path, f"{key}.png")
            res = process_item(item, vie_score)
            if res:
                append_result(tmp_dir, group_name, rank, res)
                key = res["key"]
                instruction_language = res["instruction_language"]
                identifier = (str(key), instruction_language)
                processed_by_group[group_name].add(identifier)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if (not dist.is_available() or not dist.is_initialized()) or rank == 0:
        for group_name in groups:
            group_tmp_dir = os.path.join(tmp_dir, group_name)
            if not os.path.isdir(group_tmp_dir):
                continue
            rank_files = glob.glob(os.path.join(group_tmp_dir, "rank_*.csv"))
            if not rank_files:
                continue
            dfs = [pd.read_csv(p) for p in rank_files]
            df_all = pd.concat(dfs, ignore_index=True)
            if "key" in df_all.columns:
                df_all = df_all.sort_values(["key", "instruction_language"])
                df_all = df_all.drop_duplicates(subset=["key", "instruction_language"], keep="last")
            final_csv = os.path.join(score_dir, f"{group_name}.csv")
            with open(final_csv, "w", newline="") as f:
                df_all.to_csv(f, index=False)

        languages = ["en", "cn"] if args.instruction_language == "all" else [args.instruction_language]
        for language in languages:
            save_path_new = args.save_dir
            (
                group_scores_semantics,
                group_scores_quality,
                group_scores_overall,
                group_scores_semantics_intersection,
                group_scores_quality_intersection,
                group_scores_overall_intersection,
            ) = analyze_scores(
                save_path_new,
                language=language,
                groups=groups,
            )

            save_scores(
                args.save_dir,
                language,
                groups,
                group_scores_semantics,
                group_scores_quality,
                group_scores_overall,
                group_scores_semantics_intersection,
                group_scores_quality_intersection,
                group_scores_overall_intersection,
            )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="path to edited images") # For example, samples/GEdit/test, which contains samples/GEdit/test/fullset.
    parser.add_argument("--instruction_language", type=str, default="all", choices=["all", "en", "cn"])
    parser.add_argument("--task_type", type=str, default="all", choices=["all", "background_change", "color_alter", "material_alter", "motion_change", "ps_human", "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"])
    parser.add_argument("--backbone", type=str, default="qwen25vl", choices=["gpt4o", "qwen25vl"])

    args = parser.parse_args()

    main(args)
    # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=0 evaluate_gedit.py