"""
基线 3：纯 Replay 增量训练器

Loss = CrossEntropy(混合 batch)，不含 EWC 正则。
混合 batch 采用动态平衡策略（与 ContinualTrainer 一致），保证公平对比。

用于量化纯 Replay 的遗忘缓解效果（无 EWC 对照组）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.fault_classifier import FaultClassifier
from src.continual.replay_buffer import ReplayBuffer
from src.data.task_splitter import TASK_CLASSES


class ReplayOnlyTrainer:
    """纯 Replay 增量训练器（基线 3）

    与 ContinualTrainer 采用相同的动态平衡 replay 策略和类别平衡 CE，
    唯一区别是不使用 EWC 正则化，用于量化 EWC 的独立贡献。

    Args:
        model:       待训练的 FaultClassifier 实例
        replay_buffer: ReplayBuffer 实例
        device:      训练设备（None 则自动选择 CUDA/CPU）
        lr:          Adam 学习率，默认 1e-3
        use_amp:     是否开启 AMP 混合精度（仅 CUDA 生效），默认 True
        balanced_replay: 是否使用动态平衡 replay，默认 True
        num_classes: 总类别数，默认 22
    """

    def __init__(
        self,
        model: FaultClassifier,
        replay_buffer: ReplayBuffer,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        use_amp: bool = True,
        balanced_replay: bool = True,
        num_classes: int = 22,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.balanced_replay = balanced_replay
        self._num_classes = num_classes

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        # AMP scaler 跨 epoch 持久化
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # 当前任务新类数
        self._current_n_new_classes: int = 0

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
        X_train_np=None,
        y_train_np=None,
    ) -> List[Dict]:
        """在指定增量任务上训练模型（纯 Replay 策略）。

        训练结束后将当前任务样本加入 ReplayBuffer。

        Args:
            task_id:       当前增量任务编号（0~3）
            train_loader:  当前任务训练 DataLoader
            epochs:        训练轮数
            log_every:     每隔多少 epoch 打印日志
            val_loader:    验证 DataLoader（None 则不评估）
            X_train_np:    当前任务训练窗口样本 ndarray [N, W, F]
            y_train_np:    当前任务训练标签 ndarray [N,]

        Returns:
            history: 列表，每元素为 {"epoch", "loss_ce", ["val_acc"]}
        """
        self._current_n_new_classes = len(TASK_CLASSES.get(task_id, []))
        n_old_classes = len(self.replay_buffer.get_stats())

        print(f"\n{'='*55}")
        print(f"[Replay-Only] Task {task_id} 训练开始（共 {epochs} epochs）")
        print(f"  新类数: {self._current_n_new_classes}  旧类数: {n_old_classes}")
        print(f"  Buffer 状态: {self.replay_buffer.get_stats()}")
        print(f"{'='*55}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr * 0.01
        )

        history: List[Dict] = []

        for epoch in range(1, epochs + 1):
            loss_ce = self._train_epoch(train_loader, task_id)
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

        # 训练完当前任务后，将样本加入 ReplayBuffer
        if X_train_np is not None and y_train_np is not None:
            self.replay_buffer.add_task_samples(X_train_np, y_train_np)
            print(f"\n  Buffer 已更新: {self.replay_buffer.get_stats()}")
            print(f"  Buffer 总样本数: {len(self.replay_buffer)}")

        return history

    @torch.no_grad()
    def evaluate_on_loader(self, loader: DataLoader) -> float:
        """在给定 DataLoader 上评估模型准确率（全类别）。"""
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

    def _compute_dynamic_replay_size(self, new_batch_size: int) -> int:
        """计算动态平衡 replay 采样量（与 ContinualTrainer 一致）。"""
        n_old_classes = len(self.replay_buffer.get_stats())
        if n_old_classes == 0 or self._current_n_new_classes == 0:
            return 0
        per_class_target = max(2, new_batch_size // self._current_n_new_classes)
        return n_old_classes * per_class_target

    def _balanced_ce_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """类别平衡交叉熵（与 ContinualTrainer 一致）。"""
        with torch.no_grad():
            counts = torch.bincount(targets, minlength=self._num_classes).float()
            weights = torch.zeros_like(counts)
            mask = counts > 0
            weights[mask] = 1.0 / counts[mask]
            n_present = mask.sum().float()
            if n_present > 0:
                weights = weights * (n_present / weights.sum())
        return F.cross_entropy(logits, targets, weight=weights)

    def _train_epoch(self, loader: DataLoader, task_id: int) -> float:
        """执行单 epoch 训练（动态平衡混合 batch），返回平均 CE loss。"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        device_str = str(self.device)

        for x_new, y_new in loader:
            x_new = x_new.to(self.device, dtype=torch.float32)
            y_new = y_new.to(self.device, dtype=torch.long)

            # 动态平衡 replay
            if task_id > 0 and len(self.replay_buffer) > 0:
                if self.balanced_replay:
                    replay_size = self._compute_dynamic_replay_size(x_new.size(0))
                else:
                    replay_size = 64  # 固定回退
                x_rep, y_rep = self.replay_buffer.sample_replay_batch(
                    batch_size=replay_size,
                    device=device_str,
                )
                x_mix = torch.cat([x_new, x_rep], dim=0)
                y_mix = torch.cat([y_new, y_rep], dim=0)
            else:
                x_mix = x_new
                y_mix = y_new

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x_mix)
                loss = self._balanced_ce_loss(logits, y_mix)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
