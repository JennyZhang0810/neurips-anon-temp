#!/usr/bin/env python3
import argparse
import json
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dior-root", default=os.environ.get("DIOR_ROOT", "."))
    parser.add_argument(
        "--split-json",
        default="outputs/splits/k3/k3_selected_instances.json",
    )
    parser.add_argument(
        "--images-dir",
        default="",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/foregrounds/k3",
    )
    parser.add_argument("--pad", type=int, default=0)
    args = parser.parse_args()
    args.images_dir = args.images_dir or os.path.join(args.dior_root, "DIOR_dataset/images/train")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.split_json, "r") as f:
        split = json.load(f)

    manifest = []
    for cls, items in split["classes"].items():
        cls_dir = os.path.join(args.out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for local_idx, item in enumerate(items):
            image_path = os.path.join(args.images_dir, f"{item['image_id']}.jpg")
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            x1, y1, x2, y2 = item["bbox_xyxy"]
            x1 = max(0, x1 - args.pad)
            y1 = max(0, y1 - args.pad)
            x2 = min(img.shape[1], x2 + args.pad)
            y2 = min(img.shape[0], y2 + args.pad)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            alpha = np.full((crop.shape[0], crop.shape[1], 1), 255, dtype=np.uint8)
            rgba = np.concatenate([crop, alpha], axis=2)
            out_name = f"{item['image_id']}_obj{item['object_index']}_{local_idx}.png"
            out_path = os.path.join(cls_dir, out_name)
            cv2.imwrite(out_path, rgba)
            manifest.append(
                {
                    "class": cls,
                    "foreground_path": out_path,
                    "source_image_id": item["image_id"],
                    "object_index": item["object_index"],
                    "bbox_xyxy": item["bbox_xyxy"],
                    "bbox_wh": item["bbox_wh"],
                }
            )

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"count": len(manifest), "manifest": manifest_path}, indent=2))


if __name__ == "__main__":
    main()
