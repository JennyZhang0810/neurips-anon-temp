#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

import cv2
import numpy as np


COMPATIBILITY = {
    "airplane": [0, 11],
    "airport": [0, 11],
    "baseballfield": [1],
    "basketballcourt": [1],
    "bridge": [2, 3, 9],
    "chimney": [8, 11, 4, 6],
    "dam": [2, 11, 9],
    "expressway-service-area": [3, 8],
    "expressway-toll-station": [3],
    "golffield": [1, 6],
    "groundtrackfield": [1],
    "harbor": [2, 8],
    "overpass": [3, 9],
    "ship": [2],
    "stadium": [1, 8],
    "storagetank": [11, 8, 5, 7],
    "tenniscourt": [1],
    "trainstation": [3, 8],
    "vehicle": [3, 11, 8, 0],
    "windmill": [11, 5, 6, 7],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dior-root", default=os.environ.get("DIOR_ROOT", "."))
    parser.add_argument(
        "--split-json",
        default="outputs/splits/k3/k3_selected_instances.json",
    )
    parser.add_argument(
        "--fg-manifest",
        default="outputs/foregrounds/k3/manifest.json",
    )
    parser.add_argument(
        "--bg-train-list",
        default="",
    )
    parser.add_argument(
        "--exclude-image-ids",
        default="outputs/splits/k3/k3_train_image_ids.txt",
    )
    parser.add_argument(
        "--images-dir",
        default="",
    )
    parser.add_argument(
        "--mask-dir",
        default="",
    )
    parser.add_argument(
        "--out-plan",
        default="outputs/plans/strict_k3_n200_semantic.json",
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tries", type=int, default=20)
    parser.add_argument("--bg-pool-per-class", type=int, default=1200)
    parser.add_argument("--fallback-label", type=int, default=11)
    args = parser.parse_args()

    args.bg_train_list = args.bg_train_list or os.path.join(args.dior_root, "ImageSets/Main/train.txt")
    args.images_dir = args.images_dir or os.path.join(args.dior_root, "DIOR_dataset/images/train")
    args.mask_dir = args.mask_dir or os.path.join(args.dior_root, "DIOR_semantic_masks")

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out_plan), exist_ok=True)

    with open(args.split_json, "r") as f:
        split = json.load(f)
    with open(args.fg_manifest, "r") as f:
        manifest = json.load(f)
    with open(args.bg_train_list, "r") as f:
        all_bg_ids = [line.strip() for line in f if line.strip()]
    with open(args.exclude_image_ids, "r") as f:
        exclude_ids = {line.strip() for line in f if line.strip()}

    bg_ids = [x for x in all_bg_ids if x not in exclude_ids]
    if not bg_ids:
        bg_ids = list(all_bg_ids)

    fg_by_class = defaultdict(list)
    for item in manifest:
        fg_by_class[item["class"]].append(item)

    classes = list(split["classes"].keys())
    per_class = max(1, args.n // max(1, len(classes)))
    target_counts = {cls: per_class for cls in classes}
    remainder = args.n - per_class * len(classes)
    for cls in classes[:remainder]:
        target_counts[cls] += 1

    def append_plan_item(plan_list, cls_name, fg_item, bg_path, bg_id, x, y, w, h):
        plan_list.append(
            {
                "class": cls_name,
                "foreground_path": fg_item["foreground_path"],
                "background_path": bg_path,
                "bg_id": bg_id,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
            }
        )

    def try_place(mask, allowed, w, h, rng_obj):
        H, W = mask.shape
        valid_coords = np.where(np.isin(mask, allowed))
        if len(valid_coords[0]) == 0:
            return None
        idxs = list(range(len(valid_coords[0])))
        rng_obj.shuffle(idxs)
        for idx in idxs[: args.max_tries]:
            cy = int(valid_coords[0][idx])
            cx = int(valid_coords[1][idx])
            x = max(0, min(W - w, cx - w // 2))
            y = max(0, min(H - h, cy - h // 2))
            roi = mask[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            if float(np.mean(np.isin(roi, allowed))) < 0.3:
                continue
            return x, y
        return None

    def random_place(mask, w, h, rng_obj):
        H, W = mask.shape
        return (
            rng_obj.randint(0, max(0, W - w)),
            rng_obj.randint(0, max(0, H - h)),
        )

    plan = []
    mask_cache = {}
    for cls in classes:
        allowed = COMPATIBILITY.get(cls, [11])
        fg_pool = fg_by_class.get(cls, [])
        if not fg_pool:
            continue
        created = 0
        max_bg_trials = max(args.bg_pool_per_class, target_counts[cls] * 10)
        for _ in range(max_bg_trials):
            if created >= target_counts[cls]:
                break
            bg_id = rng.choice(bg_ids)
            mask_path = os.path.join(args.mask_dir, f"{bg_id}.npy")
            bg_path = os.path.join(args.images_dir, f"{bg_id}.jpg")
            if not os.path.exists(mask_path) or not os.path.exists(bg_path):
                continue
            if mask_path not in mask_cache:
                mask_cache[mask_path] = np.load(mask_path, mmap_mode="r")
            mask = mask_cache[mask_path]

            fg_item = rng.choice(fg_pool)
            base_w, base_h = fg_item["bbox_wh"]
            if base_w <= 0 or base_h <= 0:
                continue
            scale = rng.uniform(0.9, 1.1)
            w = max(4, int(round(base_w * scale)))
            h = max(4, int(round(base_h * scale)))

            placed_xy = try_place(mask, allowed, w, h, rng)
            if placed_xy is None:
                placed_xy = try_place(mask, [args.fallback_label], w, h, rng)
            if placed_xy is None:
                placed_xy = random_place(mask, w, h, rng)

            append_plan_item(plan, cls, fg_item, bg_path, bg_id, placed_xy[0], placed_xy[1], w, h)
            created += 1

        while created < target_counts[cls]:
            bg_id = rng.choice(bg_ids)
            bg_path = os.path.join(args.images_dir, f"{bg_id}.jpg")
            mask_path = os.path.join(args.mask_dir, f"{bg_id}.npy")
            if not os.path.exists(bg_path):
                continue
            fg_item = rng.choice(fg_pool)
            base_w, base_h = fg_item["bbox_wh"]
            if base_w <= 0 or base_h <= 0:
                continue
            scale = rng.uniform(0.9, 1.1)
            w = max(4, int(round(base_w * scale)))
            h = max(4, int(round(base_h * scale)))
            if os.path.exists(mask_path):
                if mask_path not in mask_cache:
                    mask_cache[mask_path] = np.load(mask_path, mmap_mode="r")
                mask = mask_cache[mask_path]
                placed_xy = try_place(mask, allowed, w, h, rng)
                if placed_xy is None:
                    placed_xy = try_place(mask, [args.fallback_label], w, h, rng)
                if placed_xy is None:
                    placed_xy = random_place(mask, w, h, rng)
            else:
                placed_xy = (0, 0)
            append_plan_item(plan, cls, fg_item, bg_path, bg_id, placed_xy[0], placed_xy[1], w, h)
            created += 1

    with open(args.out_plan, "w") as f:
        json.dump(plan, f, indent=2)
    print(json.dumps({"count": len(plan), "out_plan": args.out_plan}, indent=2))


if __name__ == "__main__":
    main()
