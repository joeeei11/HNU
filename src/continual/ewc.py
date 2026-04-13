"""
EWC（弹性权重巩固）模块

实现对角线近似 Fisher 信息矩阵的 EWC 正则化，防止灾难性遗忘。

核心公式：
    penalty = (λ/2) * Σ_i F_i * (θ_i - θ*_i)²

其中 F_i 是参数 θ_i 的对角 Fisher 估计，θ*_i 是上一个任务训练完毕后的最优参数值。
多任务时 Fisher 累积（F_total = F_old + F_new），不替换，防止旧任务保护退化。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EWC:
    """弹性权重巩固（Elastic Weight Consolidation）

    使用对角线近似 Fisher 信息矩阵估计参数重要性，
    通过二次正则项约束关键参数不发生大幅偏移。

    Args:
        model:      待保护的神经网络模型
        lambda_ewc: EWC 正则项强度，默认 5000.0
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 5000.0) -> None:
        self.model = model
        self.lambda_ewc = lambda_ewc

        # 累积 Fisher 信息矩阵（对角线）：{name: Tensor}
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        # 最优参数快照（上一个任务训练完毕时的参数值）：{name: Tensor}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 公开 API
    # ──────────────────────────────────────────────────────────────────────────

    def compute_fisher(
        self,
        dataloader: DataLoader,
        n_samples: int = 500,
    ) -> None:
        """计算当前任务的对角 Fisher 信息矩阵并保存参数快照。

        采样 n_samples 条数据，对每条样本计算：
            F_i = E[(∂log p(y|x)/∂θ_i)²]

        注意：
          - 调用时模型应处于 eval() 状态
          - 不调用 optimizer.step()（仅计算梯度，不更新参数）
          - 关闭 AMP autocast，避免 float16 精度不足导致 Fisher 估计偏差

        Args:
            dataloader: 当前任务训练 DataLoader
            n_samples:  用于估计 Fisher 的最大样本数，默认 500
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # 初始化临时 Fisher 累加器（float32，不跟随模型精度）
        temp_fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, dtype=torch.float32)
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

        n_counted = 0
        criterion = nn.CrossEntropyLoss()

        for x_batch, y_batch in dataloader:
            if n_counted >= n_samples:
                break

            # 取当前 batch 中所需数量
            remain = n_samples - n_counted
            x_batch = x_batch[:remain].to(device, dtype=torch.float32)
            y_batch = y_batch[:remain].to(device, dtype=torch.long)

            # 逐样本计算梯度（不使用 AMP）
            for i in range(x_batch.size(0)):
                x_i = x_batch[i].unsqueeze(0)  # [1, W, 52]
                y_i = y_batch[i].unsqueeze(0)   # [1,]

                self.model.zero_grad()
                logits = self.model(x_i)            # [1, num_classes]
                loss = criterion(logits, y_i)
                loss.backward()

                for name, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        temp_fisher[name] += p.grad.float().pow(2)

                n_counted += 1
                if n_counted >= n_samples:
                    break

        # 对样本数取平均，得到期望值
        if n_counted > 0:
            for name in temp_fisher:
                temp_fisher[name] /= n_counted

        # 保存当前任务的 Fisher（计算完成后再调用 update_task 进行累积）
        self._current_fisher: Dict[str, torch.Tensor] = temp_fisher

        # 保存当前参数快照（脱离计算图）
        self.optimal_params = {
            name: p.detach().clone().float()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

    def update_task(self) -> None:
        """将当前任务的 Fisher 累积合并到历史 Fisher 中。

        采用累加策略：F_total = F_old + F_new
        不做替换，确保旧任务的参数保护不被降级。

        调用时机：当前任务的 compute_fisher() 完成后立即调用。
        """
        if not hasattr(self, "_current_fisher"):
            return

        for name, f_new in self._current_fisher.items():
            if name in self.fisher_dict:
                self.fisher_dict[name] = self.fisher_dict[name] + f_new.cpu()
            else:
                self.fisher_dict[name] = f_new.cpu().clone()

        # 清理临时缓存
        del self._current_fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """计算 EWC 正则惩罚项。

        penalty = (λ/2) * Σ_i F_i * (θ_i - θ*_i)²

        若尚未完成任何任务的 Fisher 估计，返回 0 标量（不产生梯度）。

        Args:
            model: 当前正在训练的模型（参数在 GPU 上）

        Returns:
            loss_ewc: EWC 正则损失标量 Tensor（与 model 的参数共享设备）
        """
        if not self.fisher_dict:
            # 第一个任务时无历史约束，返回零
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=False)

        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)

        for name, p in model.named_parameters():
            if name not in self.fisher_dict:
                continue

            fisher = self.fisher_dict[name].to(device)
            opt_param = self.optimal_params[name].to(device)
            loss = loss + (fisher * (p.float() - opt_param).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * loss

    def is_ready(self) -> bool:
        """返回 EWC 是否已完成至少一个任务的 Fisher 估计。"""
        return len(self.fisher_dict) > 0
