#!/usr/bin/env python3
import argparse
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm


def generate_target_mask(img_h, img_w, x, y, w, h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    mask[y1:y2, x1:x2] = 255
    return mask


DIOR_CLASSES = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
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
CLASS_TO_ID = {x.lower(): i for i, x in enumerate(DIOR_CLASSES)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--anydoor-root", default=os.environ.get("ANYDOOR_ROOT", ""))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not args.anydoor_root:
        raise RuntimeError("Set --anydoor-root or ANYDOOR_ROOT to the external inpainting/composition code directory.")
    sys.path.append(args.anydoor_root)
    os.chdir(args.anydoor_root)
    import demo_20251117  # type: ignore

    inference_single_image = demo_20251117.inference_single_image

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.label_dir, exist_ok=True)

    with open(args.plan, "r") as f:
        plan = json.load(f)

    for idx, item in enumerate(tqdm(plan)):
        bg_path = item["background_path"]
        fg_path = item["foreground_path"]
        cls = item["class"].lower()
        out_id = f"strict_{idx:05d}"
        out_img = os.path.join(args.output_dir, f"{out_id}.jpg")
        out_lab = os.path.join(args.label_dir, f"{out_id}.txt")
        if args.skip_existing and os.path.exists(out_img) and os.path.exists(out_lab):
            continue

        bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        if bg is None or fg is None:
            continue

        if fg.shape[2] == 4:
            fg_mask = (fg[:, :, 3] > 128).astype(np.uint8)
            fg_rgb = cv2.cvtColor(fg[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            fg_mask = np.ones(fg.shape[:2], dtype=np.uint8)
            fg_rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)

        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        h_bg, w_bg = bg.shape[:2]
        tar_mask = generate_target_mask(h_bg, w_bg, item["x"], item["y"], item["w"], item["h"])

        try:
            result = inference_single_image(fg_rgb, fg_mask, bg_rgb, tar_mask)
            result_rgb = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            print(f"render error {out_id}: {e}")
            continue

        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_img, result_bgr)

        cid = CLASS_TO_ID.get(cls)
        if cid is None:
            continue
        xc = (item["x"] + item["w"] / 2.0) / w_bg
        yc = (item["y"] + item["h"] / 2.0) / h_bg
        wn = item["w"] / w_bg
        hn = item["h"] / h_bg
        with open(out_lab, "w") as f:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")


if __name__ == "__main__":
    main()
