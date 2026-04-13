"""
Transformer 长程依赖编码器

在 CNN 局部特征基础上捕获时序全局依赖关系。
输入：[B, W, d_model]（batch_first，来自 CNN 输出 reshape）
输出：[B, d_model]（时间步均值聚合）
"""

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """Transformer 时序编码器

    网络结构：
        Linear(d_model, d_model) 输入投影
        + 可学习位置嵌入 [1, W, d_model]
        TransformerEncoderLayer × num_layers（batch_first=True）
        LayerNorm
        时间步均值聚合 → [B, d_model]

    Args:
        d_model:         特征嵌入维度，默认 256
        nhead:           多头注意力头数，默认 8
        num_layers:      Transformer 编码层数，默认 4
        dim_feedforward: FFN 隐层维度，默认 512
        dropout:         Dropout 比率，默认 0.1
        window_size:     滑动窗口大小（决定位置嵌入维度），默认 50
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        window_size: int = 50,
    ) -> None:
        super().__init__()

        # 输入投影：将 CNN 输出特征映射到 Transformer 特征空间
        self.input_proj = nn.Linear(d_model, d_model)

        # 可学习位置嵌入：shape = [1, window_size, d_model]
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer 编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,       # 重要：输入格式 [B, W, d_model]
            norm_first=False,       # Post-LN（原始 Transformer 结构）
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: [B, W, d_model]，CNN 输出 permute 后的时序特征序列

        Returns:
            [B, d_model]，全局语义特征（时间步均值聚合）
        """
        # 输入投影
        x = self.input_proj(x)                               # [B, W, d_model]

        # 加入位置嵌入（兼容任意序列长度 ≤ window_size）
        x = x + self.pos_embed[:, :x.size(1), :]            # [B, W, d_model]

        # Transformer 编码
        x = self.transformer(x)                              # [B, W, d_model]
        x = self.norm(x)

        # 均值聚合：将时序维度压缩为全局特征
        x = x.mean(dim=1)                                    # [B, d_model]

        return x
