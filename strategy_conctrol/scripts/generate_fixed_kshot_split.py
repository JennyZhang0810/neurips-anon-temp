#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import defaultdict

import xml.etree.ElementTree as ET


CANONICAL_CLASSES = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "expressway-service-area",
    "expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]


NAME_MAP = {
    "Expressway-Service-area": "expressway-service-area",
    "Expressway-toll-station": "expressway-toll-station",
    "expressway-service-area": "expressway-service-area",
    "expressway-toll-station": "expressway-toll-station",
    "groundtrackfield": "groundtrackfield",
}


def canonicalize(name: str) -> str:
    name = name.strip()
    return NAME_MAP.get(name, name.lower())


def parse_xml(xml_path: str):
    root = ET.parse(xml_path).getroot()
    objects = []
    for idx, obj in enumerate(root.findall("object")):
        cls = canonicalize(obj.find("name").text)
        if cls not in CANONICAL_CLASSES:
            continue
        b = obj.find("bndbox")
        xmin = int(float(b.find("xmin").text))
        ymin = int(float(b.find("ymin").text))
        xmax = int(float(b.find("xmax").text))
        ymax = int(float(b.find("ymax").text))
        w = max(0, xmax - xmin)
        h = max(0, ymax - ymin)
        objects.append(
            {
                "class": cls,
                "object_index": idx,
                "bbox_xyxy": [xmin, ymin, xmax, ymax],
                "bbox_wh": [w, h],
            }
        )
    return objects


def build_candidates(train_ids, ann_dir):
    per_class = defaultdict(list)
    for image_id in train_ids:
        xml_path = os.path.join(ann_dir, f"{image_id}.xml")
        if not os.path.exists(xml_path):
            continue
        for obj in parse_xml(xml_path):
            per_class[obj["class"]].append(
                {
                    "image_id": image_id,
                    "xml_path": xml_path,
                    "object_index": obj["object_index"],
                    "bbox_xyxy": obj["bbox_xyxy"],
                    "bbox_wh": obj["bbox_wh"],
                }
            )
    return per_class


def select_k_instances(candidates, k, seed):
    rng = random.Random(seed)
    selected = {}
    repeated_within_class = []
    for cls in CANONICAL_CLASSES:
        pool = list(candidates.get(cls, []))
        if len(pool) < k:
            raise RuntimeError(f"class {cls} only has {len(pool)} candidates, need {k}")
        rng.shuffle(pool)

        grouped = defaultdict(list)
        for item in pool:
            grouped[item["image_id"]].append(item)

        unique_ids = list(grouped.keys())
        rng.shuffle(unique_ids)

        picks = []
        used_images = set()
        for image_id in unique_ids:
            if len(picks) >= k:
                break
            picks.append(grouped[image_id][0])
            used_images.add(image_id)

        if len(picks) < k:
            leftovers = [x for x in pool if x["image_id"] not in used_images]
            leftovers += [x for x in pool if x["image_id"] in used_images]
            seen = {(p["image_id"], p["object_index"]) for p in picks}
            for item in leftovers:
                key = (item["image_id"], item["object_index"])
                if key in seen:
                    continue
                picks.append(item)
                seen.add(key)
                if len(picks) >= k:
                    break

        if len({p["image_id"] for p in picks}) < k:
            repeated_within_class.append(cls)
        selected[cls] = picks[:k]
    return selected, repeated_within_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dior-root", default=os.environ.get("DIOR_ROOT", "."))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-list",
        default="",
    )
    parser.add_argument(
        "--train-label-dir",
        default="",
    )
    parser.add_argument(
        "--images-dir",
        default="",
    )
    parser.add_argument(
        "--ann-dir",
        default="",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/splits/k3",
    )
    args = parser.parse_args()

    args.train_label_dir = args.train_label_dir or os.path.join(args.dior_root, "DIOR_dataset/labels/train")
    args.images_dir = args.images_dir or os.path.join(args.dior_root, "DIOR_dataset/images/train")
    args.ann_dir = args.ann_dir or os.path.join(args.dior_root, "Annotations")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.train_list:
        with open(args.train_list, "r") as f:
            train_ids = [line.strip() for line in f if line.strip()]
    else:
        train_ids = sorted(
            os.path.splitext(x)[0]
            for x in os.listdir(args.train_label_dir)
            if x.endswith(".txt")
        )

    candidates = build_candidates(train_ids, args.ann_dir)
    selected, repeated_classes = select_k_instances(candidates, args.k, args.seed)

    selected_instances = {
        "meta": {
            "seed": args.seed,
            "k": args.k,
            "num_classes": len(CANONICAL_CLASSES),
            "selection_rule": "instance-level; prefer unique images within each class; training uses full annotations of selected images",
        },
        "classes": selected,
    }

    train_image_ids = sorted({x["image_id"] for items in selected.values() for x in items})
    train_images = [os.path.join(args.images_dir, f"{image_id}.jpg") for image_id in train_image_ids]

    summary = {
        "seed": args.seed,
        "k": args.k,
        "num_classes": len(CANONICAL_CLASSES),
        "num_selected_instances": len(CANONICAL_CLASSES) * args.k,
        "num_train_images": len(train_image_ids),
        "selection_rule": "instance-level; prefer unique images within each class; training uses full annotations of selected images",
        "all_classes_have_3_instances": all(len(v) == args.k for v in selected.values()),
        "classes_with_repeated_image_usage_within_class": repeated_classes,
        "class_stats": {
            cls: {
                "num_instances": len(items),
                "unique_images": len({x["image_id"] for x in items}),
                "image_ids": [x["image_id"] for x in items],
            }
            for cls, items in selected.items()
        },
    }

    with open(os.path.join(args.out_dir, "k3_selected_instances.json"), "w") as f:
        json.dump(selected_instances, f, indent=2)
    with open(os.path.join(args.out_dir, "k3_train_image_ids.txt"), "w") as f:
        f.write("\n".join(train_image_ids) + "\n")
    with open(os.path.join(args.out_dir, "k3_train_images.txt"), "w") as f:
        f.write("\n".join(train_images) + "\n")
    with open(os.path.join(args.out_dir, "k3_split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
