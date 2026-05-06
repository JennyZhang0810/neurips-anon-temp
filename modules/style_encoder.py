import torch
import torch.nn as nn
from typing import Optional, Tuple

__all__ = ["StyleEncoder", "build_style_encoder"]

class StyleEncoder(nn.Module):
    """Background Style Encoder (BSE)

    核心功能:
        从背景图 (或背景裁剪) 中抽取语义 + 色调风格向量 S。

    forward 输入/输出:
        输入: bg -> (B,C,H,W) 期望 C=3, value range 由外部统一, 推荐 [-1,1] 或 [0,1]。
        输出: S -> (B, style_dim) L2 归一化后的风格向量。

    设计要点:
        1. 轻量卷积金字塔 + GroupNorm + SiLU 保持稳定。
        2. GAP 提取全局特征避免定位敏感性。
        3. 可选颜色统计 (RGB mean/std) 作为附加低维风格提示, 通过 concat 后再投影。
        4. 输出统一归一化便于 L_style (欧氏距离) 与后续调制。

    可配置参数:
        in_channels(int): 输入通道数。
        base_channels(int): 第一层基宽度。
        num_down(int): 下采样次数 (stride=2)。
        style_dim(int): 输出风格向量维度。
        use_clip(bool): 预留开关, 未来使用外部视觉编码器时启用。
        dropout(float): 中间随机失活。
        include_color_stats(bool): 是否拼接颜色统计 (每通道 mean/std 共 6 维)。

    扩展占位:
        - 多尺度 GAP: 可在未来维护一个 list 收集各层特征 GAP 后 concat。
        - LightEncoder 联合返回 (S, light_vec)。

    示例:
        S = style_encoder(bg)               # 背景风格
        S_hat = style_encoder(gen_patch)    # 生成区域风格
        L_style = ((S - S_hat)**2).sum(1).mean()

    注意: 只做特征提取, 不含损失实现。
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_down: int = 4,
        style_dim: int = 512,
        use_clip: bool = False,
        dropout: float = 0.0,
        include_color_stats: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_down = num_down
        self.style_dim = style_dim
        self.use_clip = use_clip
        self.include_color_stats = include_color_stats

        layers = []
        c_in = in_channels
        c = base_channels
        for i in range(num_down):
            layers.append(nn.Conv2d(c_in, c, kernel_size=3, stride=2, padding=1))  # 下采样
            layers.append(nn.GroupNorm(num_groups=8, num_channels=c))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            c_in = c
            c = min(c * 2, base_channels * 8)  # 限制最大宽度
        self.encoder = nn.Sequential(*layers)

        # MLP 投影到 style_dim
        # 输入到 MLP 的维度 (卷积编码 + 可选颜色统计 6 维)
        mlp_in_dim = c_in + (6 if self.include_color_stats else 0)
        self.proj = nn.Sequential(
            nn.Linear(mlp_in_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, bg: torch.Tensor) -> torch.Tensor:
        """Compute normalized style vector.
        Args:
            bg: (B,C,H,W) background image tensor.
        Returns:
            S: (B, style_dim) L2-normalized style vector.
        """
        x = self.encoder(bg)              # (B, C_enc, H', W')
        x = torch.mean(x, dim=[2, 3])     # GAP -> (B, C_enc)

        if self.include_color_stats:
            # 颜色统计在原分辨率上计算，避免过度平滑
            # 如果输入范围是 [-1,1] 可以临时映射到 [0,1] 计算统计再返回。
            if bg.min() < 0.0:  # 简单判断范围
                bg_for_stats = (bg + 1.0) / 2.0
            else:
                bg_for_stats = bg
            mean = bg_for_stats.mean(dim=[2,3])          # (B,C)
            std  = bg_for_stats.std(dim=[2,3]) + 1e-6     # (B,C)
            color_stats = torch.cat([mean, std], dim=1)   # (B, 2C) -> 6
            feat = torch.cat([x, color_stats], dim=1)
        else:
            feat = x

        S = self.proj(feat)                  # (B, style_dim)
        S = torch.nn.functional.normalize(S, p=2, dim=1)  # L2 归一化
        return S


def build_style_encoder(cfg: Optional[dict] = None) -> StyleEncoder:
    """根据配置构建 StyleEncoder。
    cfg 字段示例:
        {
          "in_channels": 3,
          "base_channels": 64,
          "num_down": 4,
          "style_dim": 512,
          "use_clip": false,
          "dropout": 0.0
        }
    若 cfg 为 None 使用默认参数。
    """
    cfg = cfg or {}
    return StyleEncoder(
        in_channels=cfg.get("in_channels", 3),
        base_channels=cfg.get("base_channels", 64),
        num_down=cfg.get("num_down", 4),
        style_dim=cfg.get("style_dim", 512),
        use_clip=cfg.get("use_clip", False),
        dropout=cfg.get("dropout", 0.0),
        include_color_stats=cfg.get("include_color_stats", True),
    )
