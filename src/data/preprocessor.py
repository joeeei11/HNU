"""
TEP 数据预处理模块
提供 Z-score 标准化和滑动窗口切片功能。
"""

from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


class TEPPreprocessor:
    """
    TEP 数据预处理器：Z-score 标准化 + 滑动窗口切片。

    使用规范：
      1. 仅在当前增量任务的训练集上调用 fit_transform()，不跨任务共享 scaler。
      2. 用同一 scaler 对对应任务的测试集调用 transform()。
      3. sliding_window() 可独立于标准化使用。
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        在训练集上拟合 StandardScaler，并返回标准化结果。

        Args:
            X_train: shape=(N, 52) 的原始训练特征

        Returns:
            shape=(N, 52) 的标准化后特征（float32）
        """
        X_scaled = self._scaler.fit_transform(X_train).astype(np.float32)
        self._fitted = True
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用已拟合的 scaler 标准化数据（用于测试集/验证集）。

        Args:
            X: shape=(N, 52) 的原始特征

        Returns:
            shape=(N, 52) 的标准化后特征（float32）

        Raises:
            RuntimeError: 尚未调用 fit_transform
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit_transform() 拟合训练集的 scaler。")
        return self._scaler.transform(X).astype(np.float32)

    @staticmethod
    def sliding_window(
        X: np.ndarray,
        y: np.ndarray,
        window_size: int = 50,
        stride: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单个类别的序列数据做滑动窗口切片。

        每次只对同一类别的连续序列做切片，不跨类别边界。
        如需多类别合并，请在外部循环调用后 np.concatenate。

        Args:
            X: shape=(N, 52) 的特征序列（已标准化）
            y: shape=(N,) 的标签数组（同一类别即可，也支持混合——按行逐窗口读取）
            window_size: 每个窗口的时间步数，默认 50
            stride: 相邻窗口的步长，默认 10

        Returns:
            X_win: shape=(N_win, window_size, 52) 的 float32 数组
            y_win: shape=(N_win,) 的 int64 标签数组
                   每个窗口的标签取该窗口最后一个时间步的标签

        Raises:
            ValueError: N < window_size 时无法生成任何窗口
        """
        N = X.shape[0]
        if N < window_size:
            raise ValueError(
                f"序列长度 {N} 小于 window_size={window_size}，无法生成窗口。"
            )

        # 计算起始索引列表
        starts = list(range(0, N - window_size + 1, stride))
        n_windows = len(starts)

        X_win = np.empty((n_windows, window_size, X.shape[1]), dtype=np.float32)
        y_win = np.empty(n_windows, dtype=np.int64)

        for i, start in enumerate(starts):
            end = start + window_size
            X_win[i] = X[start:end]
            # 取窗口末尾时间步的标签（若 y 全为同一值则等价）
            y_win[i] = y[end - 1]

        return X_win, y_win
