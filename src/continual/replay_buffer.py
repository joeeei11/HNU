"""
经验回放缓冲区模块（Experience Replay Buffer）

采用蓄水池采样（Reservoir Sampling）保证每个历史样本被等概率保留。
以 numpy 存储，sample_replay_batch 时才转为 CUDA Tensor，节省显存。

设计原则：
  - 每个全局类别独立缓冲，每类最多保留 buffer_size_per_class 条窗口样本
  - 蓄水池采样保证无偏：无论输入顺序如何，每个样本被保留的概率相等
  - add_task_samples 支持增量追加（不重建整个 buffer）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class ReplayBuffer:
    """经验回放缓冲区

    为每个全局类别维护独立的蓄水池缓冲区，每类最多保留
    buffer_size_per_class 条滑动窗口样本（numpy 存储）。

    Args:
        buffer_size_per_class: 每个类别保留的最大样本数，默认 50
        random_seed:           随机数种子（None 则不固定），用于复现性
    """

    def __init__(
        self,
        buffer_size_per_class: int = 50,
        random_seed: Optional[int] = None,
    ) -> None:
        self.buffer_size_per_class = buffer_size_per_class
        self._rng = np.random.default_rng(random_seed)

        # 各类别的样本缓冲：{class_id: {"X": ndarray[N,W,F], "count": int}}
        # count 记录"见过的总样本数"（用于蓄水池采样概率计算，不是 buffer 大小）
        self._buffers: Dict[int, Dict] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 公开 API
    # ──────────────────────────────────────────────────────────────────────────

    def add_task_samples(
        self,
        X_windows: np.ndarray,
        y_windows: np.ndarray,
    ) -> None:
        """将新任务的窗口样本加入缓冲区（蓄水池采样）。

        对每个类别独立执行蓄水池采样：
          - 若 buffer 未满：直接插入
          - 若 buffer 已满：以 k/n 的概率替换 buffer 中的随机位置
            （k = buffer_size，n = 已见过的该类总样本数）

        Args:
            X_windows: 形状 [N, window_size, n_features]，滑动窗口样本
            y_windows: 形状 [N,]，对应的全局类别标签（int64）
        """
        assert X_windows.ndim == 3, f"X_windows 应为 3D，实际 shape={X_windows.shape}"
        assert y_windows.ndim == 1, f"y_windows 应为 1D，实际 shape={y_windows.shape}"
        assert len(X_windows) == len(y_windows), "X 与 y 数量不一致"

        classes = np.unique(y_windows)
        for cls in classes:
            cls = int(cls)
            mask = y_windows == cls
            X_cls = X_windows[mask]   # [N_cls, W, F]

            if cls not in self._buffers:
                # 初始化该类别的缓冲区
                self._buffers[cls] = {
                    "X": np.empty((0,) + X_cls.shape[1:], dtype=np.float32),
                    "count": 0,  # 已见过的总样本数
                }

            buf = self._buffers[cls]

            for i in range(len(X_cls)):
                buf["count"] += 1
                n = buf["count"]           # 已见过的样本总数（含当前）
                k = self.buffer_size_per_class

                if len(buf["X"]) < k:
                    # buffer 未满，直接追加
                    buf["X"] = np.concatenate(
                        [buf["X"], X_cls[i : i + 1]], axis=0
                    )
                else:
                    # buffer 已满：以 k/n 的概率决定是否替换
                    j = self._rng.integers(0, n)  # [0, n-1] 均匀采样
                    if j < k:
                        # 替换 buffer 中第 j 个位置
                        buf["X"][j] = X_cls[i]

    def sample_replay_batch(
        self,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> Tuple:
        """从所有历史类别中均匀采样，返回 (X_tensor, y_tensor)。

        均匀采样：先按类别均匀分配名额，再在各类内随机取样。
        若总样本不足 batch_size，则返回全部样本。

        Args:
            batch_size: 要采样的样本数
            device:     目标设备字符串（如 "cuda"、"cpu"），None 则返回 CPU Tensor

        Returns:
            (X_tensor, y_tensor)：
                X_tensor: float32 Tensor，形状 [N, window_size, n_features]
                y_tensor: int64 Tensor，形状 [N,]
        """
        import torch

        if len(self) == 0:
            # 缓冲区为空（Task 0 前调用），返回空 Tensor
            return (
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
            )

        classes = sorted(self._buffers.keys())
        n_classes = len(classes)
        quota_per_class = max(1, batch_size // n_classes)

        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []

        for cls in classes:
            X_cls = self._buffers[cls]["X"]   # [N_buf, W, F]
            n_avail = len(X_cls)
            n_sample = min(quota_per_class, n_avail)

            idx = self._rng.choice(n_avail, size=n_sample, replace=False)
            X_parts.append(X_cls[idx])
            y_parts.append(np.full(n_sample, cls, dtype=np.int64))

        X_np = np.concatenate(X_parts, axis=0)
        y_np = np.concatenate(y_parts, axis=0)

        # 打乱顺序，避免类别 block 效应
        perm = self._rng.permutation(len(X_np))
        X_np = X_np[perm]
        y_np = y_np[perm]

        X_tensor = torch.from_numpy(X_np).float()
        y_tensor = torch.from_numpy(y_np).long()

        if device is not None:
            X_tensor = X_tensor.to(device)
            y_tensor = y_tensor.to(device)

        return X_tensor, y_tensor

    def get_stats(self) -> Dict[int, int]:
        """返回各类别当前缓冲区大小（实际存储的样本数）。

        Returns:
            {class_id: n_samples_in_buffer}
        """
        return {cls: len(buf["X"]) for cls, buf in self._buffers.items()}

    def get_total_seen(self) -> Dict[int, int]:
        """返回各类别累计见过的总样本数（包括被替换掉的样本）。

        Returns:
            {class_id: total_samples_seen}
        """
        return {cls: buf["count"] for cls, buf in self._buffers.items()}

    def __len__(self) -> int:
        """返回缓冲区总样本数（所有类别之和）。"""
        return sum(len(buf["X"]) for buf in self._buffers.values())

    def state_dict(self) -> Dict:
        """序列化缓冲区状态，用于 checkpoint 保存。

        Returns:
            可被 torch.save 序列化的字典
        """
        return {
            "buffer_size_per_class": self.buffer_size_per_class,
            "buffers": {
                cls: {
                    "X": buf["X"].copy(),
                    "count": buf["count"],
                }
                for cls, buf in self._buffers.items()
            },
        }

    def load_state_dict(self, state: Dict) -> None:
        """从 state_dict 恢复缓冲区状态。

        Args:
            state: 由 state_dict() 序列化的字典
        """
        self.buffer_size_per_class = state["buffer_size_per_class"]
        self._buffers = {}
        for cls_str, buf_data in state["buffers"].items():
            cls = int(cls_str)
            self._buffers[cls] = {
                "X": buf_data["X"].copy(),
                "count": buf_data["count"],
            }
