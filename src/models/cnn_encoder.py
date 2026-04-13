"""
1D-CNN 局部特征提取器

将多变量时间序列沿时间轴提取局部卷积特征。
输入：[B, 52, W]（channel-first，符合 Conv1d 要求）
输出（pool=True） ：[B, 256]（全局平均池化后展平）
输出（pool=False）：[B, 256, W]（保留时间维度，供 Transformer 使用）
"""

import torch
import torch.nn as nn
from typing import List


class CNNEncoder(nn.Module):
    """1D-CNN 特征提取器

    网络结构：
        Conv1d(52→128, k=3, p=1) → BN → ReLU → Dropout(0.2)
        Conv1d(128→256, k=3, p=1) → BN → ReLU
        （可选）AdaptiveAvgPool1d(1) → Flatten → [B, 256]

    Args:
        in_channels: 输入通道数（TEP 传感器数量 = 52）
        channels: 各卷积层输出通道数列表，默认 [128, 256]
        dropout: Dropout 比率，默认 0.2
    """

    def __init__(
        self,
        in_channels: int = 52,
        channels: List[int] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [128, 256]

        # 第一卷积块
        self.conv1 = nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        # 第二卷积块
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.relu2 = nn.ReLU(inplace=True)

        # 全局平均池化（standalone 模式使用）
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor, pool: bool = True) -> torch.Tensor:
        """前向传播

        Args:
            x:    输入张量，shape = [B, in_channels, W]
            pool: True  → AdaptiveAvgPool1d(1) → [B, out_channels]
                  False → 保留时间维度 → [B, out_channels, W]

        Returns:
            pool=True ：[B, out_channels]
            pool=False：[B, out_channels, W]
        """
        # 第一卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        # 第二卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        if pool:
            x = self.pool(x)        # [B, out_channels, 1]
            x = x.squeeze(-1)       # [B, out_channels]

        return x
