"""
增量任务划分模块
将 TEP 22 类按预设方案划分为 4 个增量任务批次，并提供 DataLoader 接口。

设计说明：
  - TASK_CLASSES 常量和 TaskSplitter 类定义不依赖 torch
  - DataLoader 在 _build_task 内部懒加载，避免模块导入时的 DLL 崩溃
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from src.data.loader import (
    load_tep_dataset,
    TEPWindowDataset,
    TRAIN_SAMPLES_NORMAL,
    TRAIN_SAMPLES_FAULT,
)
from src.data.preprocessor import TEPPreprocessor


# 增量任务类别划分（全局标签 0~21）
TASK_CLASSES: Dict[int, List[int]] = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6, 7, 8],
    2: [9, 10, 11, 12, 13, 14],
    3: [15, 16, 17, 18, 19, 20, 21],
}


class TaskSplitter:
    """
    增量任务分割器：将 TEP 数据集按 TASK_CLASSES 划分为 4 个增量任务。

    每个增量任务包含若干新故障类别（全局标签），训练/测试数据均已做：
      1. 各任务独立的 Z-score 标准化（仅在该任务训练集上 fit）
      2. 滑动窗口切片

    DataLoader 采用懒加载，仅在服务器（AutoDL）环境调用 get_task() 时构建。

    Args:
        raw_dir: TEP .dat 文件目录
        window_size: 滑动窗口大小
        stride: 滑动窗口步长
        batch_size: DataLoader batch 大小
        num_workers: DataLoader 并行进程数
    """

    def __init__(
        self,
        raw_dir: str,
        window_size: int = 50,
        stride: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        self._raw_dataset = load_tep_dataset(raw_dir)
        self._window_size = window_size
        self._stride = stride
        self._batch_size = batch_size
        self._num_workers = num_workers

        # 缓存已构建的 DataLoader（避免重复滑窗）
        self._train_loaders: Dict = {}
        self._test_loaders: Dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 公开 API
    # ──────────────────────────────────────────────────────────────────────────

    def get_task(self, task_id: int) -> Tuple:
        """
        返回指定增量任务的训练/测试 DataLoader。

        标准化只在当前任务训练集上 fit，测试集用同一 scaler transform。

        Args:
            task_id: 0~3

        Returns:
            (train_loader, test_loader)
        """
        if task_id not in TASK_CLASSES:
            raise ValueError(f"task_id 须为 0~3，当前传入：{task_id}")

        if task_id not in self._train_loaders:
            self._build_task(task_id)

        return self._train_loaders[task_id], self._test_loaders[task_id]

    def get_all_test_loaders(self) -> List:
        """
        返回所有 4 个增量任务的测试 DataLoader（按 task_id 顺序）。
        若某任务尚未构建，将自动触发构建。

        Returns:
            List[DataLoader]，长度为 4
        """
        loaders = []
        for task_id in sorted(TASK_CLASSES.keys()):
            _, test_loader = self.get_task(task_id)
            loaders.append(test_loader)
        return loaders

    @property
    def task_classes(self) -> Dict[int, List[int]]:
        """返回各增量任务的类别列表（只读）。"""
        return TASK_CLASSES

    # ──────────────────────────────────────────────────────────────────────────
    # 内部构建
    # ──────────────────────────────────────────────────────────────────────────

    def _build_task(self, task_id: int) -> None:
        """
        构建单个增量任务的训练/测试 DataLoader 并缓存。
        DataLoader 在此处懒加载 torch，仅在服务器环境调用。
        """
        from torch.utils.data import DataLoader  # 懒加载

        classes = TASK_CLASSES[task_id]
        preprocessor = TEPPreprocessor()  # 每个任务独立 scaler

        # ── 1. 拼接该任务所有类别的训练集原始序列 ──────────────────────────
        X_train_parts, y_train_parts = [], []
        for cls in classes:
            X_raw = self._raw_dataset[cls]["train"]  # (480, 52)
            y_raw = np.full(X_raw.shape[0], cls, dtype=np.int64)
            X_train_parts.append(X_raw)
            y_train_parts.append(y_raw)

        X_train_raw = np.concatenate(X_train_parts, axis=0)
        y_train_raw = np.concatenate(y_train_parts, axis=0)

        # ── 2. 仅在训练集上 fit 标准化 ───────────────────────────────────────
        X_train_scaled = preprocessor.fit_transform(X_train_raw)

        # ── 3. 对每个类别分别做滑动窗口（不跨类别边界） ──────────────────────
        train_X_wins, train_y_wins = [], []
        test_X_wins, test_y_wins = [], []

        offset = 0
        for cls in classes:
            n = TRAIN_SAMPLES_NORMAL if cls == 0 else TRAIN_SAMPLES_FAULT
            X_cls = X_train_scaled[offset: offset + n]
            y_cls = y_train_raw[offset: offset + n]
            Xw, yw = TEPPreprocessor.sliding_window(
                X_cls, y_cls, self._window_size, self._stride
            )
            train_X_wins.append(Xw)
            train_y_wins.append(yw)
            offset += n

            X_test_raw = self._raw_dataset[cls]["test"]
            X_test_scaled = preprocessor.transform(X_test_raw)
            y_test_raw = np.full(X_test_raw.shape[0], cls, dtype=np.int64)
            Xw_te, yw_te = TEPPreprocessor.sliding_window(
                X_test_scaled, y_test_raw, self._window_size, self._stride
            )
            test_X_wins.append(Xw_te)
            test_y_wins.append(yw_te)

        X_train_win = np.concatenate(train_X_wins, axis=0)
        y_train_win = np.concatenate(train_y_wins, axis=0)
        X_test_win = np.concatenate(test_X_wins, axis=0)
        y_test_win = np.concatenate(test_y_wins, axis=0)

        # ── 4. 构建 Dataset & DataLoader ─────────────────────────────────────
        train_ds = TEPWindowDataset(X_train_win, y_train_win)
        test_ds = TEPWindowDataset(X_test_win, y_test_win)

        self._train_loaders[task_id] = DataLoader(
            train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=False,
        )
        self._test_loaders[task_id] = DataLoader(
            test_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数：从 config.yaml 快速创建 TaskSplitter
# ─────────────────────────────────────────────────────────────────────────────

def build_task_splitter_from_config(config_path: str = "config/config.yaml") -> "TaskSplitter":
    """
    读取 config.yaml 中的超参数，创建并返回 TaskSplitter 实例。

    Args:
        config_path: config.yaml 路径

    Returns:
        配置好的 TaskSplitter 实例
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return TaskSplitter(
        raw_dir=cfg["data"]["raw_dir"],
        window_size=cfg["training"]["window_size"],
        stride=cfg["training"]["stride"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )
