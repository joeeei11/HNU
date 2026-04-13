"""
基线 1：静态全量训练器 & Fine-tuning 基线

StaticTrainer：在给定 DataLoader 上对 FaultClassifier 做静态（非增量）训练。
              支持 AMP 混合精度，每 log_every epoch 打印 train_loss / val_acc。

FineTuningTrainer：无正则化的朴素增量训练器（Fine-tuning 基线）。
                   与 EWCOnlyTrainer 相同结构，但 Loss 中不包含 EWC penalty。
                   用于与 EWC 对比，展示灾难性遗忘程度。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.fault_classifier import FaultClassifier


class StaticTrainer:
    """静态全量训练器

    Args:
        model:    待训练的 FaultClassifier 实例
        device:   训练设备（None 则自动选择 CUDA/CPU）
        lr:       Adam 学习率，默认 1e-3
        use_amp:  是否开启 AMP 混合精度（仅 CUDA 生效），默认 True
    """

    def __init__(
        self,
        model: FaultClassifier,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        use_amp: bool = True,
        task_classes=None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.lr = lr
        # AMP 仅在 CUDA 下生效
        self.use_amp = use_amp and (self.device.type == "cuda")

        if task_classes is not None:
            self._cls_idx = torch.tensor(task_classes, dtype=torch.long)
            self._label_map = {g: l for l, g in enumerate(task_classes)}
        else:
            self._cls_idx = None
            self._label_map = None

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.scheduler = None

    # ──────────────────────────────────────────────────────────────────────────
    # 公开 API
    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        log_every: int = 5,
        save_path: Optional[str] = None,
    ) -> List[Dict]:
        """训练模型

        Args:
            train_loader: 训练数据 DataLoader
            val_loader:   验证数据 DataLoader（None 则不评估）
            epochs:       训练总轮数
            log_every:    每隔多少 epoch 打印一次日志
            save_path:    模型权重保存路径（None 则不保存）

        Returns:
            history: 每 epoch 的记录列表，格式 [{"epoch", "train_loss", "val_acc"}, ...]
        """
        history: List[Dict] = []

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr * 0.01
        )

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            record: Dict = {"epoch": epoch, "train_loss": train_loss}

            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                record["val_acc"] = val_acc

            history.append(record)
            self.scheduler.step()

            if epoch % log_every == 0 or epoch == epochs:
                msg = f"[Epoch {epoch:3d}/{epochs}]  loss={train_loss:.4f}"
                if "val_acc" in record:
                    msg += f"  val_acc={record['val_acc'] * 100:.2f}%"
                print(msg)

        if save_path is not None:
            parent = os.path.dirname(save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"模型已保存至 {save_path}")

        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """评估模型准确率

        Args:
            loader: 测试/验证 DataLoader

        Returns:
            accuracy: 正确率，范围 [0, 1]
        """
        self.model.eval()
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.long)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(x)

            if self._cls_idx is not None:
                cls_idx = self._cls_idx.to(self.device)
                pred_local = logits[:, cls_idx].argmax(dim=1)
                pred = cls_idx[pred_local]
            else:
                pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        return correct / max(total, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        """执行单 epoch 训练，返回平均 loss"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(x)
                if self._cls_idx is not None:
                    cls_idx = self._cls_idx.to(self.device)
                    logits_m = logits[:, cls_idx]
                    y_local = torch.tensor(
                        [self._label_map[yi.item()] for yi in y.cpu()],
                        dtype=torch.long, device=self.device
                    )
                    loss = self.criterion(logits_m, y_local)
                else:
                    loss = self.criterion(logits, y)

            # AMP 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 基线 3：Fine-tuning 朴素增量训练器（对照组）
# ─────────────────────────────────────────────────────────────────────────────

class FineTuningTrainer:
    """Fine-tuning 朴素增量训练器（灾难性遗忘对照组）

    与 EWCOnlyTrainer 接口一致，但 Loss 中不含任何正则约束。
    用于定量衡量不加保护时的灾难性遗忘程度。

    Args:
        model:   待训练的 FaultClassifier 实例
        device:  训练设备（None 则自动选择 CUDA/CPU）
        lr:      Adam 学习率，默认 1e-3
        use_amp: 是否开启 AMP 混合精度（仅 CUDA 生效），默认 True
    """

    def __init__(
        self,
        model: "FaultClassifier",
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        use_amp: bool = True,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.lr = lr
        self.use_amp = use_amp and (self.device.type == "cuda")

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
        train_loader: "DataLoader",
        epochs: int = 50,
        log_every: int = 5,
        val_loader: Optional["DataLoader"] = None,
    ) -> List[Dict]:
        """在指定增量任务上朴素 Fine-tuning（无任何正则化）。

        Args:
            task_id:      当前增量任务编号（0~3）
            train_loader: 当前任务训练 DataLoader
            epochs:       训练轮数
            log_every:    每隔多少 epoch 打印日志
            val_loader:   验证 DataLoader（None 则不评估）

        Returns:
            history: 列表，每元素为 {"epoch", "loss_ce", ["val_acc"]}
        """
        print(f"\n{'='*55}")
        print(f"[Fine-tuning] Task {task_id} 训练开始（共 {epochs} epochs）")
        print(f"{'='*55}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr * 0.01
        )

        history: List[Dict] = []

        for epoch in range(1, epochs + 1):
            loss_ce = self._train_epoch(train_loader)
            record: Dict = {"epoch": epoch, "loss_ce": loss_ce}

            if val_loader is not None:
                val_acc = self.evaluate_on_loader(val_loader)
                record["val_acc"] = val_acc

            history.append(record)
            scheduler.step()

            if epoch % log_every == 0 or epoch == epochs:
                msg = f"  [Epoch {epoch:3d}/{epochs}]  ce={loss_ce:.4f}"
                if "val_acc" in record:
                    msg += f"  val_acc={record['val_acc'] * 100:.2f}%"
                print(msg)

        return history

    @torch.no_grad()
    def evaluate_on_loader(self, loader: "DataLoader") -> float:
        """在给定 DataLoader 上评估模型准确率（全类别）。

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

    def _train_epoch(self, loader: "DataLoader") -> float:
        """执行单 epoch 训练，返回平均 CE loss。"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
