#!/usr/bin/env python3
import argparse
import os


DIOR_CLASSES = {
    0: "airplane",
    1: "airport",
    2: "baseballfield",
    3: "basketballcourt",
    4: "bridge",
    5: "chimney",
    6: "dam",
    7: "Expressway-Service-area",
    8: "Expressway-toll-station",
    9: "golffield",
    10: "groundtrackfield",
    11: "harbor",
    12: "overpass",
    13: "ship",
    14: "stadium",
    15: "storagetank",
    16: "tenniscourt",
    17: "trainstation",
    18: "vehicle",
    19: "windmill",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--out-yaml", required=True)
    parser.add_argument("--val-path", default="")
    parser.add_argument("--root-path", default=os.environ.get("DIOR_ROOT", "."))
    args = parser.parse_args()
    args.val_path = args.val_path or os.path.join(args.root_path, "DIOR_dataset/images/test")

    os.makedirs(os.path.dirname(args.out_yaml), exist_ok=True)
    lines = [
        f"path: {args.root_path}",
        f"train: {args.train_list}",
        f"val: {args.val_path}",
        "",
        "names:",
    ]
    for idx, name in DIOR_CLASSES.items():
        lines.append(f"  {idx}: {name}")

    with open(args.out_yaml, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(args.out_yaml)


if __name__ == "__main__":
    main()
