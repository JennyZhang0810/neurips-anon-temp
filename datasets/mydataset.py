# # # import json
# # # import cv2
# # # import numpy as np
# # # import os
# # # from torch.utils.data import Dataset
# # # from PIL import Image
# # # import cv2
# # # from .data_utils import * 
# # # from .base import BaseDataset

# # # class MyDataset(BaseDataset):
# # #     def __init__(self, fg_dir, bg_dir):
# # #         super().__init__()

# # #         self.bg_dir = bg_dir

# # #         # bg_data = os.listdir(self.bg_dir)
# # #         # self.bg_data = [i for i in bg_data if 'mask' in i]
# # #         bg_data = os.listdir(mask_dir)
        

# # #         self.image_dir = fg_dir
# # #         self.data  = os.listdir(self.image_dir)
# # #         self.dynamic = 0

# # #     def __len__(self):
# # #         return len(self.data)
    
# # #     def __getitem__(self, idx):
# # #         item = self.get_sample(idx)
# # #         return item
    
# # #     def get_alpha_mask(self, mask_path):
# # #         image = cv2.imread( mask_path, cv2.IMREAD_UNCHANGED)
# # #         mask = (image[:,:,-1] > 128).astype(np.uint8)
# # #         return mask
    
# # #     def check_mask_area(self, mask):
# # #         H,W = mask.shape[0], mask.shape[1]
# # #         ratio = mask.sum() / (H * W)
# # #         if ratio > 0.8 * 0.8  or ratio < 0.05 * 0.05:
# # #             return False
# # #         else:
# # #             return True 
    

# # #     def get_sample(self, idx):
# # #         # dir_name = self.data[idx]
# # #         # dir_path = os.path.join(self.image_dir, dir_name)
# # #         # images = os.listdir(dir_path)
# # #         # image_name = [i for i in images if '.png' in i][0]
# # #         # image_path = os.path.join(dir_path, image_name)
# # #         image_path = os.path.join(self.image_dir, self.data[idx])

# # #         image = cv2.imread( image_path, cv2.IMREAD_UNCHANGED)
# # #         image = cv2.resize(image, (512, 512))
# # #         mask = (image[:,:,-1] > 128).astype(np.uint8)
# # #         image = image[:,:,:-1]

# # #         image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
# # #         ref_image = image 
# # #         ref_mask = mask
# # #         ref_image, ref_mask = expand_image_mask(image, mask, ratio=1.4)
# # #         bg_idx =  np.random.randint(0, len(self.bg_data)-1)
        
# # #         tar_mask_name = self.bg_data[bg_idx]
# # #         tar_mask_path = os.path.join(self.bg_dir, tar_mask_name)
# # #         tar_image_path = tar_mask_path.replace('_mask','')

# # #         tar_image = cv2.imread(tar_image_path).astype(np.uint8)
# # #         tar_image = cv2.resize(tar_image, (512, 512))
# # #         tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
# # #         tar_mask = (cv2.imread(tar_mask_path) > 128).astype(np.uint8)[:,:,0] 
# # #         tar_mask = cv2.resize(tar_mask, (512, 512))

# # #         item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
# # #         sampled_time_steps = self.sample_timestep()
# # #         item_with_collage['time_steps'] = sampled_time_steps
# # #         return item_with_collage

# # import os
# # import cv2
# # import numpy as np
# # from torch.utils.data import Dataset
# # from .data_utils import *
# # from .base import BaseDataset

# # class MyDataset(BaseDataset):
# #     def __init__(self, fg_dir, bg_dir, mask_dir):
# #         super().__init__()

# #         self.fg_dir = fg_dir
# #         self.bg_dir = bg_dir
# #         self.mask_dir = mask_dir

# #         # 统一的数据名列表（假设三者都有相同的文件名）
# #         self.data = os.listdir(self.fg_dir)
# #         self.data = [f for f in self.data if f.endswith('.png') or f.endswith('.jpg')]

# #         print(f"[Init] fg samples: {len(self.data)}")
# #         print(f"[Init] fg_dir: {fg_dir}")
# #         print(f"[Init] bg_dir: {bg_dir}")
# #         print(f"[Init] mask_dir: {mask_dir}")

# #         self.dynamic = 0

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         item = self.get_sample(idx)
# #         return item
# #         print(f"[Return keys] {item_with_collage.keys()}")


# #     def get_sample(self, idx):
# #         filename = self.data[idx]
# #         print(f"\n[Sample] Loading sample: {filename}")

# #         # 构建路径
# #         fg_path = os.path.join(self.fg_dir, filename)
# #         bg_path = os.path.join(self.bg_dir, filename)
# #         mask_path = os.path.join(self.mask_dir, filename)

# #         # 读取前景图
# #         fg_image = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
# #         if fg_image is None:
# #             raise FileNotFoundError(f"Cannot load fg image: {fg_path}")
# #         fg_image = cv2.resize(fg_image, (512, 512))

# #         if fg_image.shape[-1] == 4:
# #             ref_mask = (fg_image[:, :, -1] > 128).astype(np.uint8)
# #             fg_image = fg_image[:, :, :-1]
# #         else:
# #             raise ValueError(f"Foreground image does not have alpha channel: {fg_path}")

# #         fg_image = cv2.cvtColor(fg_image.copy(), cv2.COLOR_BGR2RGB)

# #         ref_image, ref_mask = expand_image_mask(fg_image, ref_mask, ratio=1.4)

# #         # 读取背景图
# #         bg_image = cv2.imread(bg_path)
# #         if bg_image is None:
# #             raise FileNotFoundError(f"Cannot load bg image: {bg_path}")
# #         bg_image = cv2.resize(bg_image, (512, 512))
# #         bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

# #         # 读取mask图
# #         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# #         if mask is None:
# #             raise FileNotFoundError(f"Cannot load mask image: {mask_path}")
# #         tar_mask = (mask > 128).astype(np.uint8)
# #         tar_mask = cv2.resize(tar_mask, (512, 512))

# #         # 加入调试输出
# #         print(f"[Paths] FG: {fg_path}")
# #         print(f"[Paths] BG: {bg_path}")
# #         print(f"[Paths] MASK: {mask_path}")
# #         print(f"[Shapes] ref_image: {ref_image.shape}, bg_image: {bg_image.shape}, mask: {tar_mask.shape}")

# #         item_with_collage = self.process_pairs(ref_image, ref_mask, bg_image, tar_mask)
# #         sampled_time_steps = self.sample_timestep()
# #         if item_with_collage is None:
# #         # 如果 item_with_collage 为 None，则初始化为一个空字典或其他默认值
# #              item_with_collage = {}
# #         item_with_collage['time_steps'] = sampled_time_steps

# #         return item_with_collage


# import os
# import cv2
# import numpy as np
# from torch.utils.data import Dataset
# from .data_utils import *
# from .base import BaseDataset

# class MyDataset(BaseDataset):
#     def __init__(self, fg_dir, bg_dir, mask_dir):
#         super().__init__()
#
#         self.fg_dir = fg_dir
#         self.bg_dir = bg_dir
#         self.mask_dir = mask_dir

#     def get_sample(self, idx):
#         filename = self.data[idx]
#         print(f"\n[Sample] Loading sample: {filename}")

#         # 构建路径
#         fg_path = os.path.join(self.fg_dir, filename)
#         bg_path = os.path.join(self.bg_dir, filename)
#         mask_path = os.path.join(self.mask_dir, filename)

#         try:
#             # 读取前景图
#             fg_image = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
#             if fg_image is None:
#                 raise FileNotFoundError(f"Cannot load fg image: {fg_path}")
#             fg_image = cv2.resize(fg_image, (512, 512))

#             if fg_image.shape[-1] == 4:
#                 ref_mask = (fg_image[:, :, -1] > 128).astype(np.uint8)
#                 fg_image = fg_image[:, :, :-1]
#             else:
#                 raise ValueError(f"Foreground image does not have alpha channel: {fg_path}")

#             fg_image = cv2.cvtColor(fg_image.copy(), cv2.COLOR_BGR2RGB)

#             ref_image, ref_mask = expand_image_mask(fg_image, ref_mask, ratio=1.4)

#             # 读取背景图
#             bg_image = cv2.imread(bg_path)
#             if bg_image is None:
#                 raise FileNotFoundError(f"Cannot load bg image: {bg_path}")
#             bg_image = cv2.resize(bg_image, (512, 512))
#             bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

#             # 读取mask图
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             if mask is None:
#                 raise FileNotFoundError(f"Cannot load mask image: {mask_path}")
#             tar_mask = (mask > 128).astype(np.uint8)
#             tar_mask = cv2.resize(tar_mask, (512, 512))

#             # 加入调试输出
#             print(f"[Paths] FG: {fg_path}")
#             print(f"[Paths] BG: {bg_path}")
#             print(f"[Paths] MASK: {mask_path}")
#             print(f"[Shapes] ref_image: {ref_image.shape}, bg_image: {bg_image.shape}, mask: {tar_mask.shape}")

#             item_with_collage = self.process_pairs(ref_image, ref_mask, bg_image, tar_mask)
#             sampled_time_steps = self.sample_timestep()
#             if item_with_collage is None:
#                 # 如果 item_with_collage 为 None，则初始化为一个空字典或其他默认值
#                 item_with_collage = {}
#             item_with_collage['time_steps'] = sampled_time_steps

#             return item_with_collage
#         except (FileNotFoundError, ValueError) as e:
#             print(f"[Error] Skipping file {filename}: {e}")
#             return None

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from .data_utils import *
from .base import BaseDataset

def mask_score(mask):
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    conc_score = np.max(cnt_area) / sum(cnt_area)
    return conc_score

def check_mask_area(mask):
    H, W = mask.shape
    ratio = mask.sum() / (H * W)
    # return 0.0005 * 0.0005 <= ratio <= 0.2 * 0.2
    return 0.0001 <= ratio <= 0.1

class MyDataset(BaseDataset):
    def __init__(self, fg_dir, bg_dir=None, mask_dir=None, masked_dir=None, valid_list_file=None):
        super().__init__()
        self.fg_dir = fg_dir
        self.bg_dir = bg_dir
        self.mask_dir = mask_dir
        self.masked_dir = masked_dir
        self.valid_list_file = valid_list_file

        # 预处理：筛选合格的样本
        all_files = [f for f in os.listdir(self.fg_dir) if f.endswith('.png') or f.endswith('.jpg')]
        valid_files = []
        for f in all_files:
            fg_path = os.path.join(self.fg_dir, f)
            fg_image = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
            if fg_image is None or fg_image.shape[-1] != 4:
                continue
            fg_image = cv2.resize(fg_image, (512, 512))
            ref_mask = (fg_image[:, :, -1] > 128).astype(np.uint8)

            if not check_mask_area(ref_mask):
                print(f"[Skip] {f}: Area not OK")
                continue
            if mask_score(ref_mask) < 0.9:
                print(f"[Skip] {f}: Connectivity not OK")
                continue

            valid_files.append(f)

        self.data = valid_files

        # 如果提供或存在有效样本列表文件，则进一步过滤（优先使用用户提供的 valid_list_file）
        auto_valid_path = os.path.join('lightning_logs', 'dataset_checks', 'valid_files.txt')
        use_path = None
        if valid_list_file is not None:
            use_path = valid_list_file
        elif os.path.exists(auto_valid_path):
            use_path = auto_valid_path

        if use_path is not None:
            try:
                with open(use_path, 'r') as f:
                    listed = [line.strip() for line in f if line.strip()]
                listed_set = set(listed)
                before = len(self.data)
                self.data = [f for f in self.data if f in listed_set]
                print(f"[Init] filtered by valid_list ({use_path}): {before} -> {len(self.data)} samples")
            except Exception as e:
                print(f"[Init][Warn] failed to read valid_list {use_path}: {e}")

        print(f"[Init] fg samples after filtering: {len(self.data)}")
        print(f"[Init] fg_dir: {self.fg_dir}")
        print(f"[Init] bg_dir: {self.bg_dir}")
        print(f"[Init] mask_dir: {self.mask_dir}")

        self.dynamic = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.get_sample(idx)
        if item is None:
            return self.__getitem__((idx + 1) % len(self))  # Skip invalid samples
        return item

    def get_sample(self, idx):
        filename = self.data[idx]
        print(f"\n[Sample] Loading sample: {filename}")

        # 构建路径
        fg_path = os.path.join(self.fg_dir, filename)
        bg_path = os.path.join(self.bg_dir, filename) if self.bg_dir is not None else None
        masked_path = os.path.join(self.masked_dir, filename) if (self.masked_dir is not None) else None
        mask_path = os.path.join(self.mask_dir, filename)

        try:
            # 读取前景图
            fg_image = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
            if fg_image is None:
                raise FileNotFoundError(f"Cannot load fg image: {fg_path}")
            fg_image = cv2.resize(fg_image, (512, 512))

            if fg_image.shape[-1] == 4:
                ref_mask = (fg_image[:, :, -1] > 128).astype(np.uint8)
                fg_image = fg_image[:, :, :-1]
            else:
                raise ValueError(f"Foreground image does not have alpha channel: {fg_path}")

            # 再次检查（保险起见）
            if not check_mask_area(ref_mask):
                print(f"[Filter] Area too small or too large for ref_mask in {filename}")
                return None
            if mask_score(ref_mask) < 0.9:
                print(f"[Filter] Low connectivity score for ref_mask in {filename}")
                return None

            fg_image = cv2.cvtColor(fg_image.copy(), cv2.COLOR_BGR2RGB)
            # 保持原始前景 crop 作为 reference identity（不要 expand）
            ref_image = fg_image.copy()
            ref_mask = ref_mask.copy()

            # 读取背景图
            bg_image = cv2.imread(bg_path)
            if bg_image is None:
                raise FileNotFoundError(f"Cannot load bg image: {bg_path}")
            bg_image = cv2.resize(bg_image, (512, 512))
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

            # 读取mask图
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Cannot load mask image: {mask_path}")
            tar_mask = (mask > 128).astype(np.uint8)
            tar_mask = cv2.resize(tar_mask, (512, 512))

            print(f"[Paths] FG: {fg_path}")
            print(f"[Paths] BG: {bg_path}")
            print(f"[Paths] MASK: {mask_path}")
            print(f"[Shapes] ref_image: {ref_image.shape}, bg_image: {bg_image.shape}, mask: {tar_mask.shape}")

            # item_with_collage = self.process_pairs(ref_image, ref_mask, bg_image, tar_mask)
            # sampled_time_steps = self.sample_timestep()
            # if item_with_collage is None:
            #     item_with_collage = {}
            # item_with_collage['time_steps'] = sampled_time_steps

            # return item_with_collage

            # NOTE: delay process_pairs call until masked_bg is created below
            # ==== 新增: 构建 AnyDoor 风格的数据项并保持向后兼容 ====
            # scene_gt: 原始完整场景 (GT)
            # scene_masked / bg_image: 场景中目标区域被擦除（用于模型输入与风格编码器）
            # fg, fg_mask: 目标前景 crop 与其 mask（用于作为 reference/ID）

            # 规范范围到 [-1,1] 与其它输入一致，避免风格分布尺度不匹配
            gt_norm = bg_image.astype(np.float32) / 127.5 - 1.0  # (H,W,3)

            # 创建/读取被遮挡的 scene：优先使用预计算的 masked image（若提供），否则把 mask 区域填充为中灰 (127)
            if masked_path is not None and os.path.exists(masked_path):
                masked_bg = cv2.imread(masked_path)
                if masked_bg is None:
                    masked_bg = bg_image.copy()
                else:
                    masked_bg = cv2.resize(masked_bg, (512, 512))
                    masked_bg = cv2.cvtColor(masked_bg, cv2.COLOR_BGR2RGB)
            else:
                masked_bg = bg_image.copy()
                try:
                    mask_bool = (tar_mask > 0)
                    # tar_mask shape: (H,W)
                    masked_bg[mask_bool] = 127
                except Exception:
                    # 退回到把全部擦灰，以防 tar_mask 形状有问题
                    h, w = masked_bg.shape[:2]
                    masked_bg[:] = 127

            masked_norm = masked_bg.astype(np.float32) / 127.5 - 1.0

            # 前景与前景 mask（保持与 process_pairs 生成的一致）
            # 使用原始前景 crop（未 expand）作为 fg
            fg_norm = fg_image.astype(np.float32) / 127.5 - 1.0
            fg_mask = (ref_mask > 0).astype(np.uint8)

            # 现在用正确语义调用 process_pairs：第三个参数必须是完整的 GT 背景 (bg_image)，
            # 因为 process_pairs 会基于 tar_image 提取 GT crop ('jpg')。
            item_with_collage = self.process_pairs(ref_image, ref_mask, bg_image, tar_mask)
            if item_with_collage is None:
                return self.__getitem__((idx + 1) % len(self))

            sampled_time_steps = self.sample_timestep()
            item_with_collage['time_steps'] = sampled_time_steps

            # 放入 item，按照 AnyDoor 要求返回关键字段，并保留兼容字段
            # jpg: GT supervising target (如果 process_pairs 已经生成则保留其裁切结果，否则回退到完整 GT)
            # masked_image: 被遮挡的 scene（inpainting 输入）
            # fg: 前景 crop
            # mask: tar_mask，指示要填充的位置
            # bg_image: 原始完整背景（供 style encoder 使用）
            if 'jpg' not in item_with_collage:
                # 只有在 process_pairs 没有生成 jpg 时才回退到完整 GT
                item_with_collage['jpg'] = gt_norm.copy()
            # 同时显式返回 masked_image 以满足 AnyDoor 输入要求（训练时用作 inpainting 输入）
            item_with_collage['masked_image'] = masked_norm.copy()
            item_with_collage['fg'] = fg_norm.copy()
            # tar_mask 目前是二值数组 (0/1)，转换为 uint8
            item_with_collage['mask'] = (tar_mask.astype(np.uint8)).copy()
            item_with_collage['bg_image'] = gt_norm.copy()
            # 兼容旧字段
            item_with_collage['scene_masked'] = masked_norm.copy()
            item_with_collage['scene_gt'] = gt_norm.copy()
            item_with_collage['fg_mask'] = fg_mask.copy()

            return item_with_collage


        except (FileNotFoundError, ValueError) as e:
            print(f"[Error] Skipping file {filename}: {e}")
            return None
