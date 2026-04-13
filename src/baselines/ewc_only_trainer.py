"""
基线 2：纯 EWC 增量训练器

每个增量任务的 Loss = CrossEntropy(新数据) + EWC.penalty(model)
训练完每个任务后立即计算 Fisher 矩阵并累积到 EWC 实例中。

与 FineTuningTrainer 的唯一区别在于是否添加 EWC 正则损失。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.fault_classifier import FaultClassifier
from src.continual.ewc import EWC


class EWCOnlyTrainer:
    """纯 EWC 增量训练器

    支持跨任务序列的增量训练，使用 EWC 正则化防止灾难性遗忘。

    Args:
        model:         待训练的 FaultClassifier 实例
        ewc:           EWC 实例（与 model 绑定）
        device:        训练设备（None 则自动选择 CUDA/CPU）
        lr:            Adam 学习率，默认 1e-3
        use_amp:       是否开启 AMP 混合精度（仅 CUDA 生效），默认 True
        fisher_samples: 每个任务训练完后估计 Fisher 的样本数，默认 500
    """

    def __init__(
        self,
        model: FaultClassifier,
        ewc: EWC,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        use_amp: bool = True,
        fisher_samples: int = 500,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.ewc = ewc
        self.lr = lr
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.fisher_samples = fisher_samples

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ──────────────────────────────────────────────────────────────────────────
    # 公开 API
    # ──────────────────────────────────────────────────────────────────────────

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int = 50,
        log_every: int = 5,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict:
        """在指定增量任务上训练模型。

        损失 = CrossEntropy + EWC.penalty（第一个任务时 penalty=0）

        训练结束后自动调用 EWC.compute_fisher() 和 EWC.update_task()。

        Args:
            task_id:      当前增量任务编号（0~3）
            train_loader: 当前任务训练 DataLoader
            epochs:       训练轮数
            log_every:    每隔多少 epoch 打印日志
            val_loader:   验证 DataLoader（None 则不评估）

        Returns:
            history: 列表，每元素为 {"epoch", "loss_ce", "loss_ewc", "loss_total", ["val_acc"]}
        """
        print(f"\n{'='*55}")
        print(f"[EWC-Only] Task {task_id} 训练开始（共 {epochs} epochs）")
        print(f"{'='*55}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr * 0.01
        )

        history: List[Dict] = []

        for epoch in range(1, epochs + 1):
            loss_ce, loss_ewc = self._train_epoch(train_loader)
            loss_total = loss_ce + loss_ewc
            record: Dict = {
                "epoch": epoch,
                "loss_ce": loss_ce,
                "loss_ewc": loss_ewc,
                "loss_total": loss_total,
            }

            if val_loader is not None:
                val_acc = self.evaluate_on_loader(val_loader)
                record["val_acc"] = val_acc

            history.append(record)
            scheduler.step()

            if epoch % log_every == 0 or epoch == epochs:
                msg = (
                    f"  [Epoch {epoch:3d}/{epochs}] "
                    f"ce={loss_ce:.4f}  ewc={loss_ewc:.4f}  total={loss_total:.4f}"
                )
                if "val_acc" in record:
                    msg += f"  val_acc={record['val_acc'] * 100:.2f}%"
                print(msg)

        # 训练完当前任务后，计算并累积 Fisher 信息矩阵
        print(f"\n  正在计算 Task {task_id} 的 Fisher 矩阵（样本数={self.fisher_samples}）...")
        self.ewc.compute_fisher(train_loader, n_samples=self.fisher_samples)
        self.ewc.update_task()
        print(f"  Fisher 矩阵已更新，当前保护参数层数：{len(self.ewc.fisher_dict)}")

        return history

    @torch.no_grad()
    def evaluate_on_loader(self, loader: DataLoader) -> float:
        """在给定 DataLoader 上评估模型准确率（全类别）。

        Args:
            loader: 测试/验证 DataLoader

        Returns:
            accuracy: 正确率 [0, 1]
        """
        self.model.eval()
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.long)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        return correct / max(total, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader):
        """执行单 epoch 训练，返回 (avg_loss_ce, avg_loss_ewc)。"""
        self.model.train()
        total_ce = 0.0
        total_ewc = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x)
                loss_ce = self.criterion(logits, y)

            # EWC 正则（在 float32 下计算，避免 autocast 影响）
            loss_ewc = self.ewc.penalty(self.model)

            loss = loss_ce + loss_ewc

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_ce += loss_ce.item()
            total_ewc += loss_ewc.item()
            n_batches += 1

        n_batches = max(n_batches, 1)
        return total_ce / n_batches, total_ewc / n_batches
