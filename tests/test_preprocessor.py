"""
tests/test_preprocessor.py
验证预处理器和任务分割器的正确性。

本机测试范围：TEPPreprocessor（纯 numpy）
服务器测试范围：额外含 TaskSplitter + DataLoader（需要 torch）
"""

import subprocess
import sys

import pytest
import numpy as np

from src.data.preprocessor import TEPPreprocessor
from src.data.task_splitter import TaskSplitter, TASK_CLASSES

# 数据目录
RAW_DIR = "Source/data/data/data"

# 检测 torch 是否可正常导入（用子进程避免 DLL 崩溃污染主进程）
def _check_torch_available() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        timeout=15,
    )
    return result.returncode == 0

_TORCH_AVAILABLE = _check_torch_available()

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="PyTorch 在本机不可用，跳过 DataLoader 相关测试（在 AutoDL 服务器运行）"
)


# ─────────────────────────────────────────────────────────────────────────────
# 测试 TEPPreprocessor - 标准化
# ─────────────────────────────────────────────────────────────────────────────

def test_fit_transform_mean_near_zero():
    """标准化后训练集均值绝对值 < 0.01"""
    rng = np.random.RandomState(42)
    X = rng.randn(480, 52).astype(np.float32) * 10 + 5
    preprocessor = TEPPreprocessor()
    X_scaled = preprocessor.fit_transform(X)
    mean_abs = np.abs(X_scaled.mean(axis=0)).max()
    assert mean_abs < 0.01, f"最大均值绝对值 {mean_abs:.4f} 超过阈值 0.01"


def test_fit_transform_std_near_one():
    """标准化后训练集标准差在 [0.99, 1.01] 范围内"""
    rng = np.random.RandomState(0)
    X = rng.randn(480, 52).astype(np.float32) * 3 - 2
    preprocessor = TEPPreprocessor()
    X_scaled = preprocessor.fit_transform(X)
    std = X_scaled.std(axis=0)
    assert std.min() >= 0.99, f"最小标准差 {std.min():.4f} < 0.99"
    assert std.max() <= 1.01, f"最大标准差 {std.max():.4f} > 1.01"


def test_transform_without_fit_raises():
    """未调用 fit_transform 直接调用 transform 应报错"""
    preprocessor = TEPPreprocessor()
    X = np.random.rand(10, 52).astype(np.float32)
    with pytest.raises(RuntimeError):
        preprocessor.transform(X)


def test_transform_output_dtype():
    """transform 输出应为 float32"""
    X = np.random.rand(100, 52).astype(np.float64)
    preprocessor = TEPPreprocessor()
    preprocessor.fit_transform(X)
    out = preprocessor.transform(X)
    assert out.dtype == np.float32, f"期望 float32，实际 {out.dtype}"


# ─────────────────────────────────────────────────────────────────────────────
# 测试 TEPPreprocessor - 滑动窗口
# ─────────────────────────────────────────────────────────────────────────────

def test_sliding_window_output_shape():
    """滑动窗口输出形状正确"""
    X = np.random.rand(480, 52).astype(np.float32)
    y = np.zeros(480, dtype=np.int64)
    X_win, y_win = TEPPreprocessor.sliding_window(X, y, window_size=50, stride=10)
    # 期望窗口数 = floor((480 - 50) / 10) + 1 = 44
    expected_n = (480 - 50) // 10 + 1
    assert X_win.shape == (expected_n, 50, 52), \
        f"X_win 形状错误：{X_win.shape}，期望 ({expected_n}, 50, 52)"
    assert y_win.shape == (expected_n,), \
        f"y_win 形状错误：{y_win.shape}"


def test_sliding_window_small_stride():
    """stride=1 时窗口数为 N - window_size + 1"""
    X = np.random.rand(100, 52).astype(np.float32)
    y = np.zeros(100, dtype=np.int64)
    X_win, _ = TEPPreprocessor.sliding_window(X, y, window_size=10, stride=1)
    assert X_win.shape[0] == 91  # 100 - 10 + 1


def test_sliding_window_too_short_raises():
    """序列长度 < window_size 应报 ValueError"""
    X = np.random.rand(20, 52).astype(np.float32)
    y = np.zeros(20, dtype=np.int64)
    with pytest.raises(ValueError):
        TEPPreprocessor.sliding_window(X, y, window_size=50)


def test_sliding_window_label_from_last_step():
    """每个窗口的标签应来自该窗口最后一个时间步"""
    X = np.random.rand(100, 52).astype(np.float32)
    y = np.arange(100, dtype=np.int64)
    _, y_win = TEPPreprocessor.sliding_window(X, y, window_size=10, stride=5)
    assert int(y_win[0]) == 9
    assert int(y_win[1]) == 14


# ─────────────────────────────────────────────────────────────────────────────
# 测试 TaskSplitter（需要 torch）
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def splitter():
    """模块级 fixture：创建 TaskSplitter（num_workers=0 避免 Windows 多进程问题）"""
    if not _TORCH_AVAILABLE:
        pytest.skip("PyTorch 不可用，跳过 TaskSplitter 测试")
    return TaskSplitter(
        raw_dir=RAW_DIR,
        window_size=50,
        stride=10,
        batch_size=64,
        num_workers=0,
    )


@requires_torch
@pytest.mark.parametrize("task_id", [0, 1, 2, 3])
def test_get_task_returns_loaders(splitter, task_id):
    """get_task() 应返回两个可迭代对象（DataLoader）"""
    train_loader, test_loader = splitter.get_task(task_id)
    assert hasattr(train_loader, "__iter__")
    assert hasattr(test_loader, "__iter__")


@requires_torch
@pytest.mark.parametrize("task_id", [0, 1, 2, 3])
def test_task_batch_shape(splitter, task_id):
    """训练 DataLoader 第一个 batch 形状应为 (<=64, 50, 52) 和 (<=64,)"""
    train_loader, _ = splitter.get_task(task_id)
    X_batch, y_batch = next(iter(train_loader))
    assert X_batch.shape[1:] == (50, 52), \
        f"Task {task_id} batch X 形状错误：{X_batch.shape}"
    assert y_batch.dim() == 1, \
        f"Task {task_id} batch y 应为 1D，实际 {y_batch.shape}"
    assert X_batch.shape[0] == y_batch.shape[0]


@requires_torch
@pytest.mark.parametrize("task_id", [0, 1, 2, 3])
def test_task_labels_in_range(splitter, task_id):
    """DataLoader 中的标签应在该任务的类别列表内"""
    train_loader, _ = splitter.get_task(task_id)
    valid_classes = set(TASK_CLASSES[task_id])
    for X_batch, y_batch in train_loader:
        unique_labels = set(y_batch.numpy().tolist())
        assert unique_labels.issubset(valid_classes), \
            f"Task {task_id} 出现意外标签：{unique_labels - valid_classes}"


@requires_torch
def test_get_all_test_loaders_count(splitter):
    """get_all_test_loaders() 应返回 4 个 DataLoader"""
    loaders = splitter.get_all_test_loaders()
    assert len(loaders) == 4


def test_invalid_task_id_raises():
    """无效 task_id 应抛出 ValueError（不依赖 torch，直接测）"""
    sp = TaskSplitter(raw_dir=RAW_DIR, num_workers=0)
    with pytest.raises(ValueError):
        sp.get_task(99)
