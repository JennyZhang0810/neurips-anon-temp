# Strategy Control Minimal Release

This folder contains only the code needed to reproduce the three control strategies:

- semantic control: class-compatible semantic-mask placement
- density control: `--n` controls the number of generated instances and balances them per class
- size control: target boxes come from selected K-shot instance sizes with a small perturbation

No dataset files, logs, trained weights, generated images, or experiment runs are included.

## Files

- `scripts/generate_fixed_kshot_split.py`: select K annotated instances per class.
- `scripts/extract_strict_foregrounds.py`: crop the selected instances into a K-shot foreground pool.
- `scripts/generate_strict_semantic_plan.py`: generate the semantic/density/size controlled synthesis plan.
- `scripts/render_strict_anydoor.py`: render a generated plan with an external composition model.
- `scripts/build_yolo_yaml_from_list.py`: build a YOLO dataset yaml from a train list.
- `examples/*.json`: schemas/templates only, not data.

## Expected Inputs

Set `DIOR_ROOT` to a local DIOR-style dataset root containing:

```text
${DIOR_ROOT}/Annotations/
${DIOR_ROOT}/DIOR_dataset/images/train/
${DIOR_ROOT}/DIOR_dataset/labels/train/
${DIOR_ROOT}/DIOR_dataset/images/test/
${DIOR_ROOT}/ImageSets/Main/train.txt
${DIOR_ROOT}/DIOR_semantic_masks/
```

The semantic masks should be `.npy` files named by image id, for example `000001.npy`.

## Minimal Pipeline

```bash
export DIOR_ROOT=/path/to/DIOR

python scripts/generate_fixed_kshot_split.py \
  --dior-root "$DIOR_ROOT" \
  --k 3 \
  --out-dir outputs/splits/k3

python scripts/extract_strict_foregrounds.py \
  --dior-root "$DIOR_ROOT" \
  --split-json outputs/splits/k3/k3_selected_instances.json \
  --out-dir outputs/foregrounds/k3

python scripts/generate_strict_semantic_plan.py \
  --dior-root "$DIOR_ROOT" \
  --split-json outputs/splits/k3/k3_selected_instances.json \
  --fg-manifest outputs/foregrounds/k3/manifest.json \
  --exclude-image-ids outputs/splits/k3/k3_train_image_ids.txt \
  --n 200 \
  --out-plan outputs/plans/strict_k3_n200_semantic.json
```

Rendering requires an external composition model:

```bash
export ANYDOOR_ROOT=/path/to/composition_model

python scripts/render_strict_anydoor.py \
  --anydoor-root "$ANYDOOR_ROOT" \
  --plan outputs/plans/strict_k3_n200_semantic.json \
  --output-dir outputs/rendered/images/train \
  --label-dir outputs/rendered/labels/train
```
