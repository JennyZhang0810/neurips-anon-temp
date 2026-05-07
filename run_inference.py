import argparse
import os
import random
import cv2
import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity
from datasets.data_utils import (get_bbox_from_mask,
                                 expand_image_mask,
                                 pad_to_square,
                                 sobel,
                                 expand_bbox,
                                 box2squre,
                                 box_in_box)

# --------------------- model loading ---------------------
def load_model(config_path='./configs/inference.yaml'):
    """Load the pre-trained ControlNet model and DDIM sampler."""
    disable_verbosity()
    config = OmegaConf.load(config_path)
    model_ckpt = config.pretrained_model
    model_config = config.config_file
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
    model = model.cuda()
    sampler = DDIMSampler(model)
    return model, sampler

model, ddim_sampler = load_model()

# --------------------- core processing ---------------------
def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    """
    Prepare reference, target, and hint tensors for the diffusion model.

    Returns a dict containing:
        ref     : reference image array (H, W, 3) in [0, 1]
        jpg     : target background in [-1, 1]
        hint    : collage (target with pasted reference) + mask channel
        extra_sizes : (H1, W1, H2, W2) for post-processing
        tar_box_yyxx_crop : bounding box in original target coordinates
    """
    # ---------- reference ----------
    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]

    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)

    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
    masked_ref_image = cv2.resize(masked_ref_image, (224, 224)).astype(np.uint8)
    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224, 224)).astype(np.uint8)
    ref_mask = ref_mask_3[:, :, 0]

    # collage preparation
    masked_ref_image_aug = masked_ref_image  # can add augmentations here if needed
    ref_collage = sobel(masked_ref_image, ref_mask / 255)

    # ---------- target ----------
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.3, 1.5])
    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 2])
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop
    cropped_target = tar_image[y1:y2, x1:x2, :]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx

    t_w = int(max(1, x2 - x1))
    t_h = int(max(1, y2 - y1))
    fg_h, fg_w = ref_collage.shape[:2]
    scale = min(t_w / fg_w, t_h / fg_h)
    if scale <= 0:
        scale = min(cropped_target.shape[1] / fg_w, cropped_target.shape[0] / fg_h)

    new_w = max(1, int(fg_w * scale))
    new_h = max(1, int(fg_h * scale))
    fg_resized = cv2.resize(ref_collage, (new_w, new_h))

    center_x = int((x1 + x2) // 2)
    center_y = int((y1 + y2) // 2)
    x1_fg = center_x - new_w // 2
    y1_fg = center_y - new_h // 2
    x2_fg = x1_fg + new_w
    y2_fg = y1_fg + new_h

    Ht, Wt = cropped_target.shape[:2]
    x1_clip = max(x1_fg, 0)
    y1_clip = max(y1_fg, 0)
    x2_clip = min(x2_fg, Wt)
    y2_clip = min(y2_fg, Ht)

    fg_x1 = x1_clip - x1_fg
    fg_y1 = y1_clip - y1_fg
    fg_x2 = fg_x1 + (x2_clip - x1_clip)
    fg_y2 = fg_y1 + (y2_clip - y1_clip)

    collage = cropped_target.copy()
    if x2_clip > x1_clip and y2_clip > y1_clip and fg_x2 > fg_x1 and fg_y2 > fg_y1:
        collage[y1_clip:y2_clip, x1_clip:x2_clip, :] = fg_resized[fg_y1:fg_y2, fg_x1:fg_x2, :]

    collage_mask = np.zeros_like(cropped_target, dtype=np.float32)
    if x2_clip > x1_clip and y2_clip > y1_clip:
        collage_mask[y1_clip:y2_clip, x1_clip:x2_clip, :] = 1.0

    H1, W1 = collage.shape[:2]
    cropped_target = pad_to_square(cropped_target, pad_value=0, random=False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value=-1, random=False).astype(np.uint8)

    H2, W2 = collage.shape[:2]
    cropped_target = cv2.resize(cropped_target, (512, 512)).astype(np.float32)
    collage = cv2.resize(collage, (512, 512)).astype(np.float32)
    collage_mask = (cv2.resize(collage_mask, (512, 512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug / 255.0
    cropped_target = cropped_target / 127.5 - 1.0
    collage = collage / 127.5 - 1.0
    collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)

    return dict(
        ref=masked_ref_image_aug,
        jpg=cropped_target,
        hint=collage,
        extra_sizes=np.array([H1, W1, H2, W2]),
        tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
    )


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    """Place the generated patch back into the original target image."""
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 5  # margin to avoid edge artifacts

    if W1 == H1:
        tar_image[y1+m:y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1:-pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1:-pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m:y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
    return gen_image


def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale=5.0):
    """
    Perform object-level image composition using ControlNet.

    Args:
        ref_image: RGB reference image (uint8)
        ref_mask:  binary mask of the reference object (uint8)
        tar_image: RGB target background (uint8)
        tar_mask:  binary mask indicating where to place the object (uint8)
        guidance_scale: classifier-free guidance scale

    Returns:
        gen_image: the composited image (uint8, RGB)
        tar_box_yyxx_crop: bounding box (y1, y2, x1, x2) of the inserted region
    """
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item['ref']
    tar = item['jpg']
    hint = item['hint']
    extra_sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop']

    # convert to torch
    control = torch.from_numpy(hint.copy()).float().cuda().unsqueeze(0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    clip_input = torch.from_numpy(ref.copy()).float().cuda().unsqueeze(0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    H, W = 512, 512
    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning(clip_input)]
    }
    un_cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning(
            torch.zeros((1, 3, 224, 224), device=clip_input.device))]
    }
    shape = (4, H // 8, W // 8)

    strength = 1.0
    model.control_scales = [strength] * 13

    samples, _ = ddim_sampler.sample(
        S=50,
        batch_size=1,
        shape=shape,
        conditioning=cond,
        verbose=False,
        eta=0.0,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=un_cond,
    )

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5)
    pred = x_samples[0].cpu().numpy().clip(0, 255).astype(np.uint8)

    gen_image = crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop)
    return gen_image, tar_box_yyxx_crop


# --------------------- simple CLI ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_image', required=True, help='Path to reference RGBA image')
    parser.add_argument('--tar_image', required=True, help='Path to target background RGB image')
    parser.add_argument('--tar_mask', required=True, help='Path to target mask (grayscale)')
    parser.add_argument('--output', required=True, help='Path to save the composited image')
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load reference (expects RGBA png with alpha channel as mask)
    fg_img = cv2.imread(args.ref_image, cv2.IMREAD_UNCHANGED)
    if fg_img is None:
        raise FileNotFoundError(f"Could not read {args.ref_image}")
    ref_mask = (fg_img[:, :, -1] > 128).astype(np.uint8)
    ref_image = cv2.cvtColor(fg_img[:, :, :-1], cv2.COLOR_BGR2RGB)

    # load target background
    tar_img = cv2.imread(args.tar_image)
    if tar_img is None:
        raise FileNotFoundError(f"Could not read {args.tar_image}")
    tar_image = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)

    # load target mask (grayscale, white where object should be placed)
    tar_mask = cv2.imread(args.tar_mask, cv2.IMREAD_GRAYSCALE)
    if tar_mask is None:
        raise FileNotFoundError(f"Could not read {args.tar_mask}")
    tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))
    tar_mask = (tar_mask > 128).astype(np.uint8)

    gen_image, bbox = inference_single_image(
        ref_image, ref_mask, tar_image, tar_mask,
        guidance_scale=args.guidance_scale
    )

    # save result
    out_bgr = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, out_bgr)
    print(f"Saved composition to {args.output}")
    print(f"Insertion bounding box (y1,y2,x1,x2): {bbox}")


if __name__ == '__main__':
    main()
