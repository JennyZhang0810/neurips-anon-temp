import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import cv2
import einops
import pyvips
from omegaconf import OmegaConf
import numpy as np
import torch
from cldm.hack import disable_verbosity

from datasets.data_utils import (get_bbox_from_mask, expand_image_mask,
                                 pad_to_square, sobel, expand_bbox,
                                 box2squre, box_in_box)

from skimage.exposure import match_histograms

save_memory = False


CLASS_CONFIG = {
    # 陆地单位
    "MIM-104": {"type": "land", "base_ratio_wh": (11.2, 2.6), "scale": 12},
    "地-M1A2坦克": {"type": "land", "base_ratio_wh": (9.8, 5.3), "scale": 12},
    "地-发电厂": {"type": "land", "base_ratio_wh": (50, 50), "scale": 12},
    "地-发射车-M142": {"type": "land", "base_ratio_wh": (7, 3), "scale": 12},
    "地-发射车-THAAD": {"type": "land", "base_ratio_wh": (12, 4), "scale": 12},
    "地-雷达站-AN_FPS-117": {"type": "land", "base_ratio_wh": (5.5, 5), "scale": 12},
    "地-坦克-M10": {"type": "land", "base_ratio_wh": (7, 3), "scale": 12},
    "地-运输车-M977": {"type": "land", "base_ratio_wh": (9.3, 2.5), "scale": 12},
    "地-运输车-M978": {"type": "land", "base_ratio_wh": (9, 3.4), "scale": 12},
    "地-运输车-M1070": {"type": "land", "base_ratio_wh": (9.1, 2.6), "scale": 12},
    "地-阵地-MIM-104爱国者AN_MPQ-53_65 雷达": {"type": "land", "base_ratio_wh": (8, 2.5), "scale": 12},
    "地-阵地路障": {"type": "land", "base_ratio_wh": (4, 2), "scale": 12},
    "地-装甲车-LAV50": {"type": "land", "base_ratio_wh": (5.7, 2.5), "scale": 12},
    "地-装甲车-M2A3布拉德利": {"type": "land", "base_ratio_wh": (6.6, 3.6), "scale": 12},
    "地-装甲车-M1134": {"type": "land", "base_ratio_wh": (7, 3), "scale": 12},
    "M142-2": {"type": "land", "base_ratio_wh": (7, 3), "scale": 12},
    # 海上单位
    "海-大型-提康德罗加": {"type": "maritime", "base_ratio_wh": (173, 17), "scale": 1},
    "海-登陆舰-黄蜂登陆舰": {"type": "maritime", "base_ratio_wh": (160, 32), "scale": 1},
    "海-航母-福特": {"type": "maritime", "base_ratio_wh": (337, 78), "scale": 1},
    "海-尼米兹航母Nimitz": {"type": "maritime", "base_ratio_wh": (333, 77), "scale": 1},
    "海-小型-康定级护卫舰": {"type": "maritime", "base_ratio_wh": (125, 15.4), "scale": 1},
    "海-运输舰-供应级综合补给舰": {"type": "maritime", "base_ratio_wh": (230, 32), "scale": 1},
    "海-中型-基隆级导弹驱逐舰": {"type": "maritime", "base_ratio_wh": (172, 17), "scale": 1},
    # 空中单位
    "空-大型-P8反潜机": {"type": "land", "base_ratio_wh": (39.4, 37.6), "scale": 5},
    "空-大型-E-2D预警机": {"type": "land", "base_ratio_wh": (18, 24.6), "scale": 6},
    "空-黑鹰直升机-UH-60": {"type": "land", "base_ratio_wh": (19.7, 15.2), "scale": 7},
    "空-小型-RQ-4全球鹰": {"type": "land", "base_ratio_wh": (14.5, 39.9), "scale": 6},
    "空-中型-F22": {"type": "land", "base_ratio_wh": (18, 13.6), "scale": 8},
    "空-中型-F35B": {"type": "land", "base_ratio_wh": (15.6, 12), "scale": 8},
    "MQ-9-Reaper": {"type": "land", "base_ratio_wh": (11, 18), "scale": 8},
    # 设施
    "地-道路立交桥": {"type": "land", "base_ratio_wh": (80, 60), "scale": 2},
    "地-电线塔": {"type": "land", "base_ratio_wh": (45, 20), "scale": 3},
    "地-炼油化工厂": {"type": "land", "base_ratio_wh": (200, 150), "scale": 1},
    "地-球形雷达": {"type": "land", "base_ratio_wh": (1, 1), "scale": 80},
    # 修正键
    "空-大-P8反潜机": {"type": "land", "base_ratio_wh": (39.4, 37.6), "scale": 5},
}

ALIAS_TO_CLASS = {
    "黄蜂登陆舰": "海-登陆舰-黄蜂登陆舰",
    "KangDing": "海-小型-康定级护卫舰",
    "Ticonderoga": "海-大型-提康德罗加",
    "基隆级驱逐舰": "海-中型-基隆级导弹驱逐舰",
    "FORD-AC": "海-航母-福特",
    "Nimitz_CVN-77": "海-尼米兹航母Nimitz",
    "补给舰": "海-运输舰-供应级综合补给舰",
    "M1070": "地-运输车-M1070",
    "M977": "地-运输车-M977",
    "M978": "地-运输车-M978",
    "LAV50": "地-装甲车-LAV50",
    "M1134": "地-装甲车-M1134",
    "M2A3": "地-装甲车-M2A3布拉德利",
    "UH-60": "空-黑鹰直升机-UH-60",
    "F-22": "空-中型-F22",
    "MQ-9": "MQ-9-Reaper",
    "E2D": "空-大型-E-2D预警机",
    "F35B": "空-中型-F35B",
    "RQ-4全球鹰": "空-小型-RQ-4全球鹰",
    "P8A反潜机": "空-大型-P8反潜机",
}


def adjust_ratio_advanced(w, h, b_w, b_h, method='area_preserve'):
    """
    高级版本：提供多种调整方法
    
    参数:
        w, h, b_w, b_h: 同上
        method: 调整方法
            'min_change' - 最小化单边变化
            'area_preserve' - 保持面积不变
            'balanced' - 平衡调整两边
    """
    original_ratio = w / h
    target_ratio1 = b_w / b_h
    target_ratio2 = b_h / b_w
    
    # 选择更接近的目标比例
    diff1 = abs(original_ratio - target_ratio1)
    diff2 = abs(original_ratio - target_ratio2)
    target_ratio = target_ratio1 if diff1 <= diff2 else target_ratio2
    
    if method == 'area_preserve':
        # 保持面积不变
        area = w * h
        new_w = (area * target_ratio) ** 0.5
        new_h = new_w / target_ratio
        return new_w, new_h
    
    elif method == 'balanced':
        # 平衡调整两边，使相对变化最小
        # 使用几何平均来平衡调整
        scale_factor = ((w / target_ratio) / h) ** 0.5
        if scale_factor > 1:
            new_w = w
            new_h = w / target_ratio
        else:
            new_w = h * target_ratio
            new_h = h
        return new_w, new_h

def load_model():
    disable_verbosity()
    config = OmegaConf.load('./configs/inference.yaml')
    model_ckpt = config.pretrained_model
    model_config = config.config_file
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler

model, ddim_sampler = None, None
model, ddim_sampler = load_model()


@dataclass
class SynthesisObject:
    """封装单个待合成对象的所有信息"""
    ref_image: np.ndarray
    ref_mask: np.ndarray
    tar_mask: np.ndarray


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]

    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
    masked_ref_image = cv2.resize(masked_ref_image, (224, 224)).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224, 224)).astype(np.uint8)
    ref_mask = ref_mask_3[:, :, 0]

    # ref aug
    masked_ref_image_aug = masked_ref_image

    # collage aug
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.3, 1.5])

    # crop
    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 2])
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)  # crop box
    y1, y2, x1, x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2, x1:x2, :]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx

    # 使用目标 mask 在 crop 内的 bbox 来决定前景放置位置与缩放，
    # 避免将 ref (224 空间) 的像素坐标直接用作 crop 的坐标。
    # tar_box_yyxx 在之前已经被 box_in_box 映射为相对于 cropped_target_image 的坐标
    t_y1, t_y2, t_x1, t_x2 = tar_box_yyxx
    Ht, Wt = cropped_target_image.shape[:2]

    # 目标 mask bbox 的宽高（crop 坐标系）
    t_w = int(max(1, t_x2 - t_x1))
    t_h = int(max(1, t_y2 - t_y1))

    fg_h, fg_w = ref_image_collage.shape[:2]
    # 以目标 bbox 为参考缩放前景，使其尽量填满 bbox（保留纵横比）
    scale_w = t_w / float(fg_w)
    scale_h = t_h / float(fg_h)
    scale = min(scale_w, scale_h)
    if scale <= 0:
        scale = min(Wt / float(fg_w), Ht / float(fg_h))

    new_w = max(1, int(fg_w * scale))
    new_h = max(1, int(fg_h * scale))
    fg_resized = cv2.resize(ref_image_collage, (new_w, new_h))

    # 把前景居中到目标 mask 的中心（crop 坐标系）
    center_x = int((t_x1 + t_x2) // 2)
    center_y = int((t_y1 + t_y2) // 2)
    x1_fg = center_x - new_w // 2
    y1_fg = center_y - new_h // 2
    x2_fg = x1_fg + new_w
    y2_fg = y1_fg + new_h

    # 背景的范围
    Ht, Wt = cropped_target_image.shape[:2]

    # 前景和背景的交集（裁剪以免越界）
    x1_clip = max(x1_fg, 0)
    y1_clip = max(y1_fg, 0)
    x2_clip = min(x2_fg, Wt)
    y2_clip = min(y2_fg, Ht)

    # 对应的前景区域
    fg_x1 = x1_clip - x1_fg
    fg_y1 = y1_clip - y1_fg
    fg_x2 = fg_x1 + (x2_clip - x1_clip)
    fg_y2 = fg_y1 + (y2_clip - y1_clip)

    collage = cropped_target_image.copy()
    if x2_clip > x1_clip and y2_clip > y1_clip and fg_x2 > fg_x1 and fg_y2 > fg_y1:
        collage[y1_clip:y2_clip, x1_clip:x2_clip, :] = fg_resized[fg_y1:fg_y2, fg_x1:fg_x2, :]

    collage_mask = np.zeros_like(cropped_target_image, dtype=np.float32)
    # 使用裁剪后的坐标来标记 mask，保证与实际贴上的区域一致
    if x2_clip > x1_clip and y2_clip > y1_clip:
        collage_mask[y1_clip:y2_clip, x1_clip:x2_clip, :] = 1.0

    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value=0, random=False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value=-1, random=False).astype(np.uint8)

    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512, 512)).astype(np.float32)
    collage = cv2.resize(collage, (512, 512)).astype(np.float32)
    collage_mask = (cv2.resize(collage_mask, (512, 512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug / 255
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0
    collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)
    
    item = dict(
        ref=masked_ref_image_aug.copy(),
        jpg=cropped_target_image.copy(),
        hint=collage.copy(),
        extra_sizes=np.array([H1, W1, H2, W2]),
        tar_box_yyxx_crop=np.array(tar_box_yyxx_crop)
    )

    return item


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale = 5.0):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    # 返回生成图和用于定位的 tar_box_yyxx_crop (y1,y2,x1,x2)
    return gen_image, tar_box_yyxx_crop


def inference_single_image_batch(reference_batch, hint_batch, guidance_scale = 5.0):
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    num_samples = reference_batch.shape[0]

    control = torch.from_numpy(hint_batch).float().cuda()
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    clip_input = torch.from_numpy(reference_batch).float().cuda() 
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {
        "c_concat": [control], 
        "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {
        "c_concat": None if guess_mode else [control], 
        "c_crossattn": [model.get_learned_conditioning([torch.zeros((num_samples,3,224,224), device=clip_input.device)])]}
    
    
    shape = (4, H // 8, W // 8)
    # ====
    # num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)


    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, _ = ddim_sampler.sample(
         S=ddim_steps,
        batch_size=num_samples,
        shape=shape,
        conditioning=cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)
    pred = np.clip(x_samples, 0, 255)
    return pred


def single_pair_inference(bg_dir, fg_dir, mask_dir, save_dir):
    gen_save_dir = os.path.join(save_dir, 'gen')
    vis_save_dir = os.path.join(save_dir, 'vis')
    os.makedirs(gen_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    # 读取前景
    image = cv2.imread(fg_dir, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"读取失败：{fg_dir}，请检查路径")
        return
    mask = (image[:, :, -1] > 128).astype(np.uint8)
    image = image[:, :, :-1]
    ref_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    if ref_image.dtype == np.uint16:
        ref_image = (ref_image / 256).astype(np.uint8)
    ref_mask = mask

    # 读取背景
    back_image = cv2.imread(bg_dir, cv2.IMREAD_UNCHANGED)
    if back_image is None:
        print(f"读取失败：{bg_dir}，请检查路径")
        return
    back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

    # 读取mask
    tar_mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    tar_mask = cv2.resize(tar_mask, (back_image.shape[1], back_image.shape[0]))
    tar_mask = (tar_mask > 128).astype(np.uint8)

    # 推理
    try:
        gen_image, tar_box = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
    except ValueError as e:
        print(f"尺寸不匹配，跳过: {e}")
        return

    # 调整尺寸
    h, w = back_image.shape[:2]
    ref_image_vis = cv2.resize(ref_image, (w, h))
    gen_image_vis = cv2.resize(gen_image, (w, h))
    mask_vis = cv2.resize(tar_mask * 255, (w, h))  
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    # 拼接可视化
    vis_image = cv2.hconcat([
        cv2.cvtColor(ref_image_vis, cv2.COLOR_RGB2BGR),   # fg
        cv2.cvtColor(back_image, cv2.COLOR_RGB2BGR),      # bg
        mask_vis,                                         # mask
        cv2.cvtColor(gen_image_vis, cv2.COLOR_RGB2BGR)    # 生成结果
    ])

    # 保存
    fg_name = os.path.splitext(os.path.basename(fg_dir))[0]
    bg_name = os.path.splitext(os.path.basename(bg_dir))[0]
    gen_save_path = os.path.join(gen_save_dir, f"{bg_name}_from_{fg_name}.png")
    vis_save_path = os.path.join(vis_save_dir, f"{bg_name}_from_{fg_name}.png")

    cv2.imwrite(gen_save_path, cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(vis_save_path, vis_image)

    # 保存YOLO和可视化红框（单图模式）
    y1, y2, x1, x2 = tar_box
    abs_x1 = x1
    abs_y1 = y1
    abs_x2 = x2
    abs_y2 = y2
    Hbg, Wbg = back_image.shape[:2]
    cx = (abs_x1 + abs_x2) / 2.0
    cy = (abs_y1 + abs_y2) / 2.0
    ww = abs_x2 - abs_x1
    hh = abs_y2 - abs_y1
    norm_cx = cx / Wbg
    norm_cy = cy / Hbg
    norm_w = ww / Wbg
    norm_h = hh / Hbg
    yolo_label_path = os.path.join(gen_save_dir, f"{bg_name}_from_{fg_name}.txt")
    with open(yolo_label_path, 'w') as f:
        f.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    vis_img = cv2.cvtColor(back_image.copy(), cv2.COLOR_RGB2BGR)
    pt1 = (int(abs_x1), int(abs_y1))
    pt2 = (int(abs_x2), int(abs_y2))
    cv2.rectangle(vis_img, pt1, pt2, (0,0,255), 6)
    visbox_path = os.path.join(gen_save_dir, f"{bg_name}_from_{fg_name}_yolobox_red.png")
    cv2.imwrite(visbox_path, vis_img)


def batch_inference(bg_dir, fg_dir, mask_dir, save_dir):
    gen_save_dir = os.path.join(save_dir, 'gen')
    vis_save_dir = os.path.join(save_dir, 'vis')
    os.makedirs(gen_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    fg_files = sorted([f for f in os.listdir(fg_dir) if f.endswith('.png')])
    bg_files = sorted([f for f in os.listdir(bg_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    for fg_file, bg_file, mask_file in zip(fg_files, bg_files, mask_files):
        # 路径
        reference_image_path = os.path.join(fg_dir, fg_file)
        bg_image_path = os.path.join(bg_dir, bg_file)
        bg_mask_path = os.path.join(mask_dir, mask_file)

        # 保存路径
        gen_save_path = os.path.join(gen_save_dir, f"{os.path.splitext(bg_file)[0]}_from_{os.path.splitext(fg_file)[0]}.png")
        vis_save_path = os.path.join(vis_save_dir, f"{os.path.splitext(bg_file)[0]}_from_{os.path.splitext(fg_file)[0]}.png")

        try:
            # 读取前景
            image = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"读取失败：{reference_image_path}，跳过")
                continue
            mask = (image[:, :, -1] > 128).astype(np.uint8)
            image = image[:, :, :-1]
            ref_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            if ref_image.dtype == np.uint16:
                ref_image = (ref_image / 256).astype(np.uint8)
            ref_mask = mask

            # 读取背景
            back_image = cv2.imread(bg_image_path)
            if back_image is None:
                print(f"读取失败：{bg_image_path}，跳过")
                continue
            back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

            # 读取mask
            tar_mask = cv2.imread(bg_mask_path, cv2.IMREAD_GRAYSCALE)
            tar_mask = cv2.resize(tar_mask, (back_image.shape[1], back_image.shape[0]))
            tar_mask = (tar_mask > 128).astype(np.uint8)

            # 推理
            try:
                gen_image, tar_box = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
            except ValueError as e:
                print(f"{fg_file} + {bg_file} 尺寸不匹配，跳过: {e}")
                continue

            # 调整尺寸
            h, w = back_image.shape[:2]
            ref_image_vis = cv2.resize(ref_image, (w, h))
            gen_image_vis = cv2.resize(gen_image, (w, h))
            mask_vis = cv2.resize(tar_mask * 255, (w, h))  
            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

            # 拼接可视化
            vis_image = cv2.hconcat([
                cv2.cvtColor(ref_image_vis, cv2.COLOR_RGB2BGR),   # fg
                cv2.cvtColor(back_image, cv2.COLOR_RGB2BGR),      # bg
                mask_vis,                                         # mask
                cv2.cvtColor(gen_image_vis, cv2.COLOR_RGB2BGR)    # 生成结果
            ])

            # 保存 
            cv2.imwrite(gen_save_path, cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(vis_save_path, vis_image)

            # --- 计算并保存YOLO标签（tar_box: y1,y2,x1,x2 相对于 back_image） ---
            y1, y2, x1, x2 = tar_box
            abs_x1 = x1
            abs_y1 = y1
            abs_x2 = x2
            abs_y2 = y2
            Hbg, Wbg = back_image.shape[:2]
            cx = (abs_x1 + abs_x2) / 2.0
            cy = (abs_y1 + abs_y2) / 2.0
            ww = abs_x2 - abs_x1
            hh = abs_y2 - abs_y1
            norm_cx = cx / Wbg
            norm_cy = cy / Hbg
            norm_w = ww / Wbg
            norm_h = hh / Hbg
            yolo_label_path = os.path.join(gen_save_dir, f"{os.path.splitext(bg_file)[0]}_from_{os.path.splitext(fg_file)[0]}.txt")
            with open(yolo_label_path, 'w') as f:
                f.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            # --- 可视化：在原始 back_image 上画红框并保存 ---
            vis_img = cv2.cvtColor(back_image.copy(), cv2.COLOR_RGB2BGR)
            pt1 = (int(abs_x1), int(abs_y1))
            pt2 = (int(abs_x2), int(abs_y2))
            cv2.rectangle(vis_img, pt1, pt2, (0,0,255), 6)
            visbox_path = os.path.join(gen_save_dir, f"{os.path.splitext(bg_file)[0]}_from_{os.path.splitext(fg_file)[0]}_yolobox_red.png")
            cv2.imwrite(visbox_path, vis_img)

        except Exception as e:
            print(f"处理 {fg_file} + {bg_file} 时发生未知错误：{e}")
            continue


def single_large_pair_inference(args):
    # YOLO标签可视化应在for循环内部，每个patch处理后画框
    bg_dir = args.bg_dir
    fg_dir = args.fg_dir
    # mask_dir = args.mask_dir
    save_dir = args.save_dir
    gen_save_dir = os.path.join(save_dir, 'gen')
    vis_save_dir = os.path.join(save_dir, 'vis')
    os.makedirs(gen_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    if args.input is None:
        print('need corrct input ')
        return
    resolution = args.resolution
    sample_factor = 0.5 / resolution

    input_coordinates = eval(args.input)
    input_image = pyvips.Image.new_from_file(bg_dir)
    sampled_image = input_image.resize(sample_factor)
    result_image = sampled_image
    
    
    # 设定模型输入尺寸和超尺寸处理的参数
    BASE_PATCH_SIZE = min(sampled_image.width, sampled_image.height) if min(sampled_image.width, sampled_image.height) < 512 else 512
    print(f"Base patch size：{BASE_PATCH_SIZE}")
    OVERSIZE_SCALE_FACTOR = 2.0
    is_IR = False

    for i, single_coord in enumerate(input_coordinates):
        print(single_coord)
        x, y, w, h = single_coord
        
        ####### mask修正 ##############################################
        fg_class = os.path.basename(fg_dir).split(".")[0] # Safely get class name
        if fg_class in ALIAS_TO_CLASS:
            fg_class = ALIAS_TO_CLASS[fg_class]
        if fg_class in CLASS_CONFIG:
            base_w, base_h = CLASS_CONFIG[fg_class]["base_ratio_wh"]
            w, h = adjust_ratio_advanced(w, h, base_w, base_h)
        ##############################################################

        # convert coordinate
        new_x = int(x * sample_factor)
        new_y = int(y * sample_factor)
        new_w = int(w * sample_factor)
        new_h = int(h * sample_factor)

        center_x = new_x + new_w / 2
        center_y = new_y + new_h / 2
        print(f"缩放后区域的中心点: ({center_x:.2f}, {center_y:.2f})")

        # 定义一个变量来存储最终用于推理的图像和mask
        final_input_patch_np = None
        final_tar_mask_np = None
        dynamic_patch_size = BASE_PATCH_SIZE
        # 定义一个变量来存储最终结果需要还原的尺寸
        final_output_size = (BASE_PATCH_SIZE, BASE_PATCH_SIZE)

        # ==================== 核心逻辑：根据尺寸选择不同处理路径 ====================
        if new_w > BASE_PATCH_SIZE or new_h > BASE_PATCH_SIZE:
            # ----- 超尺寸路径 -----
            print(f"检测到超尺寸对象 (w:{new_w}, h:{new_h})，启用动态缩放流程。")

            # a. 计算动态的、更大的 patch size
            dynamic_patch_size = int(max(new_w, new_h) * OVERSIZE_SCALE_FACTOR)
            final_output_size = (dynamic_patch_size, dynamic_patch_size) # 记录下来，用于最后放大结果

            # b. 计算这个大 patch 的裁剪坐标，并进行边界修正
            crop_x = round(center_x - dynamic_patch_size / 2)
            crop_y = round(center_y - dynamic_patch_size / 2)

            crop_x = max(0, crop_x)
            crop_x = min(crop_x, sampled_image.width - dynamic_patch_size)
            crop_y = max(0, crop_y)
            crop_y = min(crop_y, sampled_image.height - dynamic_patch_size)
            crop_x, crop_y = int(crop_x), int(crop_y)
            
            # c. 裁剪出这个大尺寸的 patch
            print(crop_x, crop_y)
            large_patch = sampled_image.crop(crop_x, crop_y, dynamic_patch_size, dynamic_patch_size)
            
            # d. 将大 patch 缩放到模型输入的 512x512
            resize_factor = BASE_PATCH_SIZE / dynamic_patch_size
            input_patch = large_patch.resize(resize_factor)
            final_input_patch_np = input_patch.numpy()
            if len(final_input_patch_np.shape) == 2:
                is_IR = True
                final_input_patch_np = np.stack((final_input_patch_np,) * 3, axis=-1)

            # e. 计算并创建缩放后的 mask
            #   - 首先计算 mask 在 large_patch 内的相对坐标
            mask_x_in_large_patch = new_x - crop_x
            mask_y_in_large_patch = new_y - crop_y
            #   - 然后根据 resize_factor 缩放这些坐标和尺寸
            scaled_mask_x = int(mask_x_in_large_patch * resize_factor)
            scaled_mask_y = int(mask_y_in_large_patch * resize_factor)
            scaled_mask_w = int(new_w * resize_factor)
            scaled_mask_h = int(new_h * resize_factor)
            #   - 创建 512x512 的 mask
            mask_background = pyvips.Image.black(BASE_PATCH_SIZE, BASE_PATCH_SIZE, bands=1)
            mask_foreground = (pyvips.Image.black(scaled_mask_w, scaled_mask_h, bands=1) + 255).cast('uchar')
            scaled_mask = mask_background.insert(mask_foreground, scaled_mask_x, scaled_mask_y)
            
            input_mask_np = scaled_mask.numpy()
            final_tar_mask_np = (input_mask_np > 128).astype(np.uint8)

        else:
            # ----- 常规路径 -----
            # c. 计算 512x512 图像块的左上角裁剪坐标
            crop_x = round(center_x - BASE_PATCH_SIZE / 2)
            crop_y = round(center_y - BASE_PATCH_SIZE / 2)

            # 边界修正
            crop_x = max(0, crop_x)
            crop_x = min(crop_x, sampled_image.width - BASE_PATCH_SIZE)
            crop_y = max(0, crop_y)
            crop_y = min(crop_y, sampled_image.height - BASE_PATCH_SIZE)
            crop_x, crop_y = int(crop_x), int(crop_y)
            
            # 裁剪与处理
            print(crop_x, crop_y)
            input_patch = sampled_image.crop(crop_x, crop_y, BASE_PATCH_SIZE, BASE_PATCH_SIZE)
            final_input_patch_np = input_patch.numpy()
            if len(final_input_patch_np.shape) == 2:
                is_IR = True
                final_input_patch_np = np.stack((final_input_patch_np,) * 3, axis=-1)

            # 创建 mask
            mask_x = new_x - crop_x
            mask_y = new_y - crop_y
            mask_background = pyvips.Image.black(BASE_PATCH_SIZE, BASE_PATCH_SIZE, bands=1)
            mask_foreground = (pyvips.Image.black(new_w, new_h, bands=1) + 255).cast('uchar')
            mask = mask_background.insert(mask_foreground, mask_x, mask_y)
            input_mask_np = mask.numpy()
            final_tar_mask_np = (input_mask_np > 128).astype(np.uint8)

        # ==============================================================================

        # 读取前景 (这部分逻辑不变)
        image = cv2.imread(fg_dir, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"读取失败：{fg_dir}，请检查路径")
            continue # 使用 continue 更合适
        
        mask = (image[:, :, -1] > 128).astype(np.uint8)
        image = image[:, :, :-1]
        ref_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        if ref_image.dtype == np.uint16:
            ref_image = (ref_image / 256).astype(np.uint8)
        ref_mask = mask

        try:
            # 推理函数使用最终准备好的 numpy array
            gen_image, _ = inference_single_image(ref_image, ref_mask, final_input_patch_np.copy(), final_tar_mask_np)

            # ==================== 关键：将结果缩放回原始尺寸 ====================
            # gen_image 是 512x512 的 numpy array
            if final_output_size != (BASE_PATCH_SIZE, BASE_PATCH_SIZE):
                print(f"推理完成，将结果从 {BASE_PATCH_SIZE}x{BASE_PATCH_SIZE} 放大回 {final_output_size[0]}x{final_output_size[1]}")
                # 使用 cv2.resize 将结果放大
                gen_image = cv2.resize(gen_image, final_output_size, interpolation=cv2.INTER_LANCZOS4)

        except ValueError as e:
            print(f"尺寸不匹配，跳过: {e}")
            continue # 使用 continue 更合适
        
    # vis
        h = w = BASE_PATCH_SIZE
        ref_image_vis = cv2.resize(ref_image, (w, h))
        gen_image_vis = cv2.resize(gen_image, (w, h))
        mask_vis = cv2.resize(final_tar_mask_np * 255, (w, h))
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        # 保存
        fg_name = os.path.splitext(os.path.basename(fg_dir))[0]
        bg_name = os.path.splitext(os.path.basename(bg_dir))[0]
        gen_save_path = os.path.join(gen_save_dir, f"{bg_name}_from_{i}_{fg_name}.png")
        cv2.imwrite(gen_save_path, cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR))

        # 拼接可视化
        vis_save_path = os.path.join(vis_save_dir, f"{bg_name}_from_{i}_{fg_name}.png")
        vis_image = cv2.hconcat([
            cv2.cvtColor(ref_image_vis, cv2.COLOR_RGB2BGR),  # fg
            cv2.cvtColor(final_input_patch_np, cv2.COLOR_RGB2BGR),  # bg
            mask_vis,  # mask
            cv2.cvtColor(gen_image_vis, cv2.COLOR_RGB2BGR)  # 生成结果
        ])
        cv2.imwrite(vis_save_path, vis_image)

        # 插入原图
        if is_IR:
            output_gray = np.mean(gen_image, axis=2).astype(np.uint8)
        else:
            output_gray = gen_image
        processed_region = pyvips.Image.new_from_array(output_gray)  # 从 numypy转成image对象，然后插入大图对应位置
        result_image = result_image.insert(processed_region, crop_x, crop_y)


        ## vis part
        # vis_part = sampled_image.crop(crop_x-512, crop_y-512, 1024, 1024)
        # vis_part.write_to_file(f'{save_dir}/gen_vis_part_1.0.png', compression=6)
        temp_bg = result_image.copy()
        vis_part = temp_bg.crop(crop_x, crop_y, dynamic_patch_size, dynamic_patch_size)
        vis_part.write_to_file(f'{save_dir}/gen_vis_part_{i}.png', compression=6)

    # result_image.write_to_file(f'{save_dir}/result_resolution_{args.resolution}.png', compression=6)  # 修改保存地址


def main():
    parser = argparse.ArgumentParser(description="使用ControlNet进行图像合成推理")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3],  default = 3, help="模式: 1=单图配对, 2=批量处理, 3=80k大图")
    parser.add_argument("--bg_dir", type=str,  default = "/data5/zhangjiening/Data_Generation/demo/input/bg/SVN3-01_20250809.png", help="背景路径(文件夹或单图)")
    parser.add_argument("--fg_dir", type=str, default = "/data5/zhangjiening/Data_Generation/demo/input/fg/F-22_00_225.png", help="前景路径(文件夹或单图)")
    parser.add_argument("--mask_dir", type=str, default = 'demo/input/mask', help="掩码路径(文件夹或单图)")
    parser.add_argument("--save_dir", type=str, default = '20251118/large_pair', help="生成结果保存路径")
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--input", type=str, default='[(16194, 63718, 122, 139), (15999, 62869, 107, 130)]',
                        help='input information e. [(x,y,w,h),(x,y,w,h),..]')
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    disable_verbosity()    
    random.seed(args.seed)
    
    if args.mode == 1:
        single_pair_inference(args.bg_dir, args.fg_dir, args.mask_dir, args.save_dir)
    elif args.mode == 2:
        batch_inference(args.bg_dir, args.fg_dir, args.mask_dir, args.save_dir)
    elif args.mode == 3:
        single_large_pair_inference(args)
    else:
        raise NotImplementedError
        

if __name__ == '__main__':
    main()
    
    # disable_verbosity()
    # save_dir = "20251117/result"

    # while True:
    #     try:
    #         root = input("数据集根目录：")
    #         bg = os.path.join(root, "bg")
    #         fg = os.path.join(root, "fg")
    #         mask = os.path.join(root, "mask")

    #         batch_inference(bg, fg, mask, save_dir)
    #     except:
    #         print("error")





