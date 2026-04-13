"""
核心混合增量训练器（EWC + Experience Replay）

修复版：
  - 动态平衡 replay：根据旧/新类数自适应调整 replay 采样量，
    保证混合 batch 中每类样本数大致相等
  - 类别平衡 CE：对混合 batch 按类别频率反加权，防止多数类主导梯度
  - EWC lambda 已在 config 中从 5000 降至 1000，配合 Fisher 累积策略

每个 mini-batch 流程：
  1. 从 train_loader 取新数据（batch_size 条）
  2. 若 task_id > 0：从 ReplayBuffer 动态采样旧数据
     采样量 = n_old_classes × (batch_size / n_new_classes)
  3. Loss = CE_balanced(混合 batch) + EWC.penalty(model)
  4. AMP 反向传播 + optimizer.step()
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.fault_classifier import FaultClassifier
from src.continual.ewc import EWC
from src.continual.replay_buffer import ReplayBuffer
from src.data.task_splitter import TASK_CLASSES
from src.evaluation.metrics import compute_metrics


class ContinualTrainer:
    """EWC + Replay 混合增量训练器（核心方案）

    Args:
        model:             待训练的 FaultClassifier 实例
        ewc:               EWC 实例（与 model 绑定）
        replay_buffer:     ReplayBuffer 实例
        config:            超参数字典（来自 config.yaml）
        device:            训练设备（None 则自动选择 CUDA/CPU）
    """

    def __init__(
        self,
        model: FaultClassifier,
        ewc: EWC,
        replay_buffer: ReplayBuffer,
        config: Dict,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.ewc = ewc
        self.replay_buffer = replay_buffer
        self.config = config

        # 从 config 中提取超参数
        training_cfg = config.get("training", config)
        self.lr: float = training_cfg.get("lr", 1e-3)
        self.use_amp: bool = training_cfg.get("use_amp", True) and (
            self.device.type == "cuda"
        )
        self.fisher_samples: int = config.get("ewc", {}).get("fisher_samples", 500)

        # Replay 参数
        replay_cfg = config.get("replay", {})
        self.balanced_replay: bool = replay_cfg.get("balanced_replay", True)
        self.replay_batch_size: int = replay_cfg.get("replay_batch_size", 64)

        # 分类数（用于类别平衡 CE）
        self._num_classes: int = config.get("evaluation", {}).get("num_classes", 22)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        # AMP scaler 跨 epoch 持久化（不在 epoch 内重建）
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # 当前任务新类数（在 train_task 中设置）
        self._current_n_new_classes: int = 0
        # 记录已完成的任务数
        self._completed_tasks: int = 0

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
        X_train_np: Optional[np.ndarray] = None,
        y_train_np: Optional[np.ndarray] = None,
    ) -> Dict:
        """在指定增量任务上执行 EWC+Replay 混合训练。

        Args:
            task_id:       当前增量任务编号（0~3）
            train_loader:  当前任务训练 DataLoader
            epochs:        训练轮数
            log_every:     每隔多少 epoch 打印日志
            val_loader:    验证 DataLoader（None 则不评估）
            X_train_np:    当前任务滑动窗口样本 [N, W, F]（用于加入 buffer）
            y_train_np:    当前任务全局标签 [N,]（用于加入 buffer）

        Returns:
            {"task_id", "loss_history", "final_loss_ce", "final_loss_ewc"}
        """
        # 确定当前任务新类数
        self._current_n_new_classes = len(TASK_CLASSES.get(task_id, []))
        n_old_classes = len(self.replay_buffer.get_stats())

        # 计算本轮预期 replay 量
        if self.balanced_replay and n_old_classes > 0:
            per_class_target = max(2, train_loader.batch_size // self._current_n_new_classes)
            expected_replay = n_old_classes * per_class_target
        else:
            expected_replay = self.replay_batch_size if n_old_classes > 0 else 0

        print(f"\n{'='*60}")
        print(f"[ContinualTrainer] Task {task_id} 训练开始（共 {epochs} epochs）")
        print(f"  新类数: {self._current_n_new_classes}  旧类数: {n_old_classes}")
        print(f"  EWC ready: {self.ewc.is_ready()}  |  Buffer: {len(self.replay_buffer)} 条")
        print(f"  Replay 模式: {'动态平衡' if self.balanced_replay else '固定'}  "
              f"预期每步采样: {expected_replay} 条")
        print(f"{'='*60}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr * 0.01
        )

        loss_history: List[Dict] = []

        for epoch in range(1, epochs + 1):
            loss_ce, loss_ewc = self._train_epoch(train_loader, task_id)
            loss_total = loss_ce + loss_ewc

            record: Dict = {
                "epoch": epoch,
                "loss_ce": loss_ce,
                "loss_ewc": loss_ewc,
                "loss_total": loss_total,
            }

            if val_loader is not None:
                val_acc = self._evaluate_loader(val_loader)
                record["val_acc"] = val_acc

            loss_history.append(record)
            scheduler.step()

            if epoch % log_every == 0 or epoch == epochs:
                msg = (
                    f"  [Epoch {epoch:3d}/{epochs}] "
                    f"ce={loss_ce:.4f}  ewc={loss_ewc:.4f}  total={loss_total:.4f}"
                )
                if "val_acc" in record:
                    msg += f"  val_acc={record['val_acc'] * 100:.2f}%"
                print(msg)

        # ── 训练后：更新 Fisher + 补充 ReplayBuffer ──────────────────────────
        print(f"\n  正在计算 Task {task_id} 的 Fisher 矩阵（n_samples={self.fisher_samples}）...")
        self.ewc.compute_fisher(train_loader, n_samples=self.fisher_samples)
        self.ewc.update_task()
        n_layers = len(self.ewc.fisher_dict)
        print(f"  Fisher 矩阵已更新，保护层数：{n_layers}")

        if X_train_np is not None and y_train_np is not None:
            self.replay_buffer.add_task_samples(X_train_np, y_train_np)
            stats = self.replay_buffer.get_stats()
            print(f"  ReplayBuffer 已更新: {stats}")
            print(f"  Buffer 总样本数: {len(self.replay_buffer)}")

        self._completed_tasks += 1

        final = loss_history[-1]
        return {
            "task_id": task_id,
            "loss_history": loss_history,
            "final_loss_ce": final["loss_ce"],
            "final_loss_ewc": final["loss_ewc"],
        }

    def evaluate_all_tasks(
        self,
        test_loaders: List[DataLoader],
    ) -> Dict[int, Dict]:
        """在所有已见任务的测试集上评估，返回各任务指标。

        Args:
            test_loaders: 按 task_id 顺序的测试 DataLoader 列表

        Returns:
            {task_id: {"acc": float, "far": float, "fdr": float, "per_class_acc": list}}
        """
        results: Dict[int, Dict] = {}

        for task_id, loader in enumerate(test_loaders):
            y_true_all: List[int] = []
            y_pred_all: List[int] = []

            self.model.eval()
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(self.device, dtype=torch.float32)
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        logits = self.model(x)
                    pred = logits.argmax(dim=1).cpu().numpy()
                    y_true_all.extend(y.numpy().tolist())
                    y_pred_all.extend(pred.tolist())

            metrics = compute_metrics(
                np.array(y_true_all),
                np.array(y_pred_all),
                normal_class=0,
            )
            results[task_id] = metrics

        return results

    def save_checkpoint(self, path: str) -> None:
        """保存完整训练状态到 checkpoint 文件。"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "ewc_fisher": self.ewc.fisher_dict,
            "ewc_optimal_params": self.ewc.optimal_params,
            "replay_buffer": self.replay_buffer.state_dict(),
            "completed_tasks": self._completed_tasks,
        }

        torch.save(checkpoint, path)
        print(f"  Checkpoint 已保存至 {path}")

    def load_checkpoint(self, path: str) -> None:
        """从 checkpoint 文件恢复完整训练状态。"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.ewc.fisher_dict = checkpoint["ewc_fisher"]
        self.ewc.optimal_params = checkpoint["ewc_optimal_params"]

        self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        self._completed_tasks = checkpoint.get("completed_tasks", 0)

        print(f"  Checkpoint 已加载：{path}")
        print(f"  已完成任务数: {self._completed_tasks}")
        print(f"  EWC 保护层数: {len(self.ewc.fisher_dict)}")
        print(f"  Buffer 总样本: {len(self.replay_buffer)}")

    # ──────────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_dynamic_replay_size(self, new_batch_size: int) -> int:
        """计算动态平衡 replay 采样量。

        目标：混合 batch 中每个类（无论新旧）获得大致相同的样本数。
        公式：replay_total = n_old_classes × (new_batch_size / n_new_classes)

        Returns:
            应从 replay buffer 采样的总样本数
        """
        n_old_classes = len(self.replay_buffer.get_stats())
        if n_old_classes == 0 or self._current_n_new_classes == 0:
            return 0
        per_class_target = max(2, new_batch_size // self._current_n_new_classes)
        return n_old_classes * per_class_target

    def _balanced_ce_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """类别平衡交叉熵：按混合 batch 中各类频率反加权。

        保证每个类对 loss 的贡献大致相等，无论该类在 batch 中有多少样本。
        缺失类权重为 0，不影响训练。
        """
        with torch.no_grad():
            counts = torch.bincount(targets, minlength=self._num_classes).float()
            # 有样本的类：权重 = 1/count；无样本的类：权重 = 0
            weights = torch.zeros_like(counts)
            mask = counts > 0
            weights[mask] = 1.0 / counts[mask]
            # 归一化：使有效类的平均权重 = 1.0
            n_present = mask.sum().float()
            if n_present > 0:
                weights = weights * (n_present / weights.sum())

        return F.cross_entropy(logits, targets, weight=weights)

    def _train_epoch(
        self, loader: DataLoader, task_id: int
    ) -> Tuple[float, float]:
        """执行单 epoch 训练，返回 (avg_loss_ce, avg_loss_ewc)。"""
        self.model.train()
        total_ce = 0.0
        total_ewc = 0.0
        n_batches = 0
        device_str = str(self.device)

        for x_new, y_new in loader:
            x_new = x_new.to(self.device, dtype=torch.float32)
            y_new = y_new.to(self.device, dtype=torch.long)

            # ── 动态平衡 replay（Task 0 时 buffer 为空，自动退化为纯新数据）──
            if task_id > 0 and len(self.replay_buffer) > 0:
                if self.balanced_replay:
                    replay_size = self._compute_dynamic_replay_size(x_new.size(0))
                else:
                    replay_size = self.replay_batch_size

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

            # 类别平衡 CE（AMP 加速）
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x_mix)
                loss_ce = self._balanced_ce_loss(logits, y_mix)

            # EWC penalty（float32，不在 autocast 下，避免精度问题）
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

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> float:
        """在单个 loader 上评估准确率（用于训练过程中的 val 监控）。"""
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
