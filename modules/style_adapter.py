import torch
import torch.nn as nn
from typing import Optional

__all__ = ["StyleAdapter", "build_style_adapter"]

class StyleAdapter(nn.Module):
    """Style Adapter Block

    功能:
        将风格向量 S -> (B, D) 映射为特征调制参数 (gamma, beta)，并对输入特征 f 进行仿 AdaIN 的线性调制。
        标准公式: f_out = (1 + gamma) * f_in + beta

    设计原则:
        1. 低侵入: 不修改原 UNet 结构，只在 decoder 指定位置插入。
        2. 通道对齐: gamma/beta 通道数与被调制的特征通道一致。
        3. 稳定性: 使用 LayerNorm/GroupNorm 前置可选规范化 (预留)。
        4. 可扩展: 未来可替换为多头注意或门控机制。

    参数:
        in_dim (int): 风格向量维度 (与 StyleEncoder 输出一致)。
        feat_channels (int): 目标特征的通道数。
        hidden (int): 中间映射宽度 (MLP)。
        use_residual (bool): 是否在输出加上 f_in 残差 (默认 True)。

    接口:
        forward(S, f) -> f_modulated
            S: (B, in_dim)
            f: (B, C, H, W)
            返回: 同 shape 特征张量。
    """
    def __init__(self, in_dim: int, feat_channels: int, hidden: int = 256, use_residual: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.feat_channels = feat_channels
        self.use_residual = use_residual
        self.mlp_gamma = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, feat_channels)
        )
        self.mlp_beta = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, feat_channels)
        )

    def forward(self, S: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        if S.dim() != 2:
            raise ValueError("Style vector S must be (B, D)")
        if f.dim() != 4:
            raise ValueError("Feature f must be (B, C, H, W)")
        if f.shape[0] != S.shape[0]:
            raise ValueError("Batch size mismatch between S and f")
        # Defensive device move: if adapter params are not on same device as S, move them.
        try:
            first_param = next(self.parameters(), None)
            if first_param is not None:
                param_dev = first_param.device
                if param_dev != S.device:
                    # move module to the device of S to avoid cross-device matmuls
                    try:
                        self.to(S.device)
                    except Exception:
                        pass
        except Exception:
            pass

        gamma = self.mlp_gamma(S).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        beta = self.mlp_beta(S).unsqueeze(-1).unsqueeze(-1)
        f_out = (1 + gamma) * f + beta
        if self.use_residual:
            f_out = f_out + f  # 进一步稳健，避免初期过度偏移
        return f_out


def build_style_adapter(cfg: Optional[dict] = None) -> StyleAdapter:
    cfg = cfg or {}
    return StyleAdapter(
        in_dim=cfg.get("in_dim", 512),
        feat_channels=cfg.get("feat_channels", 320),  # 例: 与中层 UNet 通道匹配
        hidden=cfg.get("hidden", 256),
        use_residual=cfg.get("use_residual", True),
    )

# 未来接入点示例 (不在此实现):
# 在 cldm/cldm.py 的 ControlledUnetModel.forward 输出 blocks 中:
#   if self.style_adapter is not None:
#       h = self.style_adapter(S, h)
