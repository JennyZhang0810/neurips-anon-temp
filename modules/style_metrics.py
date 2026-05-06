import numpy as np
import torch
from typing import Dict, Tuple

try:
    from skimage.color import rgb2lab
except Exception:
    rgb2lab = None

__all__ = [
    "compute_lab_delta",
    "histogram_distance",
    "style_alignment_score",
    "batch_style_report"
]

def _to_numpy(img: torch.Tensor) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:  # (B,C,H,W)
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        elif img.dim() == 3:  # (C,H,W)
            img = img.detach().cpu().permute(1, 2, 0).numpy()
    return img

def _normalize_to_01(arr: np.ndarray) -> np.ndarray:
    # 输入可能是 [-1,1] 或 [0,1]，统一到 [0,1]
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    return np.clip(arr, 0.0, 1.0)

# =============== Metric 1: ΔLab 平均距离 ===============

def compute_lab_delta(bg: torch.Tensor, gen: torch.Tensor) -> float:
    """Compute average ΔE (approximated by L2 in Lab) between background and generated region.
    Args:
        bg: (B,C,H,W) or (C,H,W)
        gen: same shape as bg (broadcast B permitted)
    Returns:
        mean_delta: scalar float
    Notes:
        - 若 skimage 不可用, 回退到 RGB 空间差异。
    """
    bg_np = _to_numpy(bg)
    gen_np = _to_numpy(gen)
    if bg_np.shape != gen_np.shape:
        raise ValueError(f"Shape mismatch bg {bg_np.shape} vs gen {gen_np.shape}")
    bg_np = _normalize_to_01(bg_np)
    gen_np = _normalize_to_01(gen_np)

    if rgb2lab is not None:
        # 展平 batch 维度
        if bg_np.ndim == 4:
            B = bg_np.shape[0]
            diffs = []
            for i in range(B):
                lab_bg = rgb2lab(bg_np[i])
                lab_gen = rgb2lab(gen_np[i])
                diff = lab_bg - lab_gen
                diffs.append(np.sqrt((diff ** 2).sum(axis=-1)).mean())
            return float(np.mean(diffs))
        else:
            lab_bg = rgb2lab(bg_np)
            lab_gen = rgb2lab(gen_np)
            diff = lab_bg - lab_gen
            return float(np.sqrt((diff ** 2).sum(axis=-1)).mean())
    else:
        # 回退: RGB 空间 L2
        diff = bg_np - gen_np
        return float(np.sqrt((diff ** 2).sum(axis=-1)).mean())

# =============== Metric 2: 颜色直方图距离 (对三通道取平均 KL) ===============

def histogram_distance(bg: torch.Tensor, gen: torch.Tensor, bins: int = 32, eps: float = 1e-6) -> float:
    bg_np = _normalize_to_01(_to_numpy(bg))
    gen_np = _normalize_to_01(_to_numpy(gen))
    if bg_np.shape != gen_np.shape:
        raise ValueError("Histogram distance shape mismatch")
    if bg_np.ndim == 4:
        B = bg_np.shape[0]
        scores = []
        for i in range(B):
            scores.append(_hist_kl(bg_np[i], gen_np[i], bins, eps))
        return float(np.mean(scores))
    else:
        return float(_hist_kl(bg_np, gen_np, bins, eps))

def _hist_kl(a: np.ndarray, b: np.ndarray, bins: int, eps: float) -> float:
    # a,b: (H,W,3)
    kls = []
    for c in range(3):
        ha, _ = np.histogram(a[..., c], bins=bins, range=(0,1), density=True)
        hb, _ = np.histogram(b[..., c], bins=bins, range=(0,1), density=True)
        ha = ha + eps; hb = hb + eps
        ha = ha / ha.sum(); hb = hb / hb.sum()
        kl = np.sum(ha * np.log(ha / hb))
        kls.append(kl)
    return float(np.mean(kls))

# =============== Metric 3: 风格向量余弦相似度对齐 ===============

def style_alignment_score(S: torch.Tensor, S_hat: torch.Tensor) -> float:
    """Compute mean cosine similarity between style vectors.
    Inputs:
        S, S_hat: (B, D)
    Returns:
        mean cosine similarity in [ -1, 1 ].
    """
    if S.shape != S_hat.shape:
        raise ValueError("Style vector shape mismatch")
    S_n = torch.nn.functional.normalize(S, p=2, dim=1)
    S_hat_n = torch.nn.functional.normalize(S_hat, p=2, dim=1)
    cos = (S_n * S_hat_n).sum(dim=1)
    return float(cos.mean().item())

# =============== Batch 汇总报告 ===============

def batch_style_report(batch: Dict[str, torch.Tensor], style_vecs: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
    """汇总一批样本的风格一致性指标。
    Args:
        batch: 包含 'bg_image', 'jpg' 键 (形状 B,H,W,3 或 tensor B,C,H,W)。
        style_vecs: (S, S_hat)
    Returns:
        dict with keys: lab_delta, hist_kl, style_cos
    """
    bg = batch.get('bg_image')
    gen = batch.get('jpg')
    if bg is None or gen is None:
        raise KeyError("batch 必须包含 'bg_image' 与 'jpg'")
    if isinstance(bg, torch.Tensor) and bg.dim() == 4 and bg.shape[1] == 3:
        bg_t = bg
    else:
        # (B,H,W,3) -> (B,3,H,W)
        bg_t = torch.tensor(bg).permute(0,3,1,2)
    if isinstance(gen, torch.Tensor) and gen.dim() == 4 and gen.shape[1] == 3:
        gen_t = gen
    else:
        gen_t = torch.tensor(gen).permute(0,3,1,2)
    S, S_hat = style_vecs

    lab_delta = compute_lab_delta(bg_t, gen_t)
    hist_kl = histogram_distance(bg_t, gen_t)
    style_cos = style_alignment_score(S, S_hat)
    return {"lab_delta": lab_delta, "hist_kl": hist_kl, "style_cos": style_cos}
