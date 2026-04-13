"""
故障分类模型：1D-CNN + Transformer + 分类头

前向流程：
    x [B, W, 52]
    → permute(0,2,1) → [B, 52, W]
    → CNNEncoder(pool=False) → [B, 256, W]
    → permute(0,2,1) → [B, W, 256]
    → TransformerEncoder → [B, 256]
    → Linear(256, num_classes) → logits [B, num_classes]
"""

from __future__ import annotations

import yaml
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from src.models.cnn_encoder import CNNEncoder
from src.models.transformer_encoder import TransformerEncoder


class FaultClassifier(nn.Module):
    """TEP 故障分类模型

    整合 1D-CNN（局部特征）+ Transformer（全局依赖）的端到端分类器。

    Args:
        num_classes: 分类类别数，默认 22（TEP 正常 + 21 种故障）
        config:      超参数字典，键值见下方 Config 结构说明

    Config 结构：
        d_model           (int) : 特征嵌入维度，默认 256
        cnn_channels      (list): CNN 各层通道数，默认 [128, 256]
        transformer_layers(int) : Transformer 层数，默认 4
        transformer_heads (int) : 注意力头数，默认 8
        dropout           (float): Dropout 比率，默认 0.1
        window_size       (int) : 滑动窗口大小，默认 50
    """

    def __init__(
        self,
        num_classes: int = 22,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = {}

        d_model = config.get("d_model", 256)
        cnn_channels = config.get("cnn_channels", [128, 256])
        transformer_layers = config.get("transformer_layers", 4)
        transformer_heads = config.get("transformer_heads", 8)
        dropout = config.get("dropout", 0.1)
        window_size = config.get("window_size", 50)

        # 1D-CNN 局部特征提取器
        self.cnn = CNNEncoder(
            in_channels=52,
            channels=cnn_channels,
            dropout=0.2,
        )

        # Transformer 长程依赖编码器
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            window_size=window_size,
        )

        # 分类头
        self.head = nn.Linear(d_model, num_classes)

        self._d_model = d_model
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化分类头权重"""
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """端到端前向传播

        Args:
            x: [B, W, 52]  滑动窗口时序样本（batch_size, window_size, n_features）

        Returns:
            logits: [B, num_classes]
        """
        # [B, W, 52] → [B, 52, W]（Conv1d 要求 channel-first）
        x = x.permute(0, 2, 1)
        # CNN 提取局部特征（不做全局池化，保留时间维度）
        x = self.cnn(x, pool=False)          # [B, 256, W]
        # [B, 256, W] → [B, W, 256]（Transformer 要求 batch_first）
        x = x.permute(0, 2, 1)
        # Transformer 捕获全局时序依赖
        x = self.transformer(x)              # [B, 256]
        # 分类输出
        logits = self.head(x)                # [B, num_classes]
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取分类头前的中间特征向量（EWC Fisher 矩阵计算使用）

        Args:
            x: [B, W, 52]

        Returns:
            features: [B, d_model]
        """
        x = x.permute(0, 2, 1)
        x = self.cnn(x, pool=False)
        x = x.permute(0, 2, 1)
        features = self.transformer(x)
        return features

    def count_params(self) -> int:
        """统计并打印可训练参数量

        Returns:
            total: 可训练参数数量
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数量: {total:,} ({total / 1e6:.2f}M)")
        return total

    @classmethod
    def from_config(cls, config_path: str) -> "FaultClassifier":
        """从 YAML 配置文件创建模型

        Args:
            config_path: config.yaml 路径

        Returns:
            FaultClassifier 实例
        """
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_cfg = dict(cfg.get("model", {}))
        model_cfg["window_size"] = cfg.get("training", {}).get("window_size", 50)
        num_classes = cfg.get("evaluation", {}).get("num_classes", 22)

        return cls(num_classes=num_classes, config=model_cfg)
