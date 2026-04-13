"""
tests/test_loader.py
验证 TEP 数据加载模块的正确性。

本机测试范围：纯 numpy 操作（load_single_file、load_tep_dataset、数据完整性）
服务器测试范围：额外含 TEPWindowDataset（需要 torch）
"""

import subprocess
import sys

import pytest
import numpy as np

from src.data.loader import (
    load_single_file,
    load_tep_dataset,
    TEPWindowDataset,
    NUM_FEATURES,
    NUM_TASKS,
    TRAIN_SAMPLES_NORMAL,
    TRAIN_SAMPLES_FAULT,
    TEST_SAMPLES,
)

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
    reason="PyTorch 在本机不可用，跳过 Dataset 相关测试（在 AutoDL 服务器运行）"
)

# 数据目录
RAW_DIR = "Source/data/data/data"


# ─────────────────────────────────────────────────────────────────────────────
# 测试 load_single_file
# ─────────────────────────────────────────────────────────────────────────────

def test_load_single_file_train_shape():
    """d00.dat 加载后（自动转置）应为 (500, 52)"""
    X = load_single_file(f"{RAW_DIR}/d00.dat")
    assert X.shape == (TRAIN_SAMPLES_NORMAL, NUM_FEATURES), \
        f"期望 ({TRAIN_SAMPLES_NORMAL}, {NUM_FEATURES})，实际 {X.shape}"


def test_load_single_file_test_shape():
    """测试文件 d00_te.dat 应为 (960, 52)"""
    X = load_single_file(f"{RAW_DIR}/d00_te.dat")
    assert X.shape == (TEST_SAMPLES, NUM_FEATURES), \
        f"期望 ({TEST_SAMPLES}, {NUM_FEATURES})，实际 {X.shape}"


def test_load_single_file_dtype():
    """加载结果应为 float32"""
    X = load_single_file(f"{RAW_DIR}/d01.dat")
    assert X.dtype == np.float32, f"期望 float32，实际 {X.dtype}"


def test_load_single_file_not_found():
    """不存在的文件应抛出 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        load_single_file("nonexistent_path/d99.dat")


# ─────────────────────────────────────────────────────────────────────────────
# 测试 load_tep_dataset（批量加载）
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_dataset():
    """模块级 fixture：加载全部 22 个任务（只加载一次，复用）"""
    return load_tep_dataset(RAW_DIR)


def test_all_tasks_loaded(full_dataset):
    """应加载全部 22 个任务（0~21）"""
    assert len(full_dataset) == NUM_TASKS, \
        f"期望 {NUM_TASKS} 个任务，实际 {len(full_dataset)}"


@pytest.mark.parametrize("task_id", range(NUM_TASKS))
def test_train_shape(full_dataset, task_id):
    """task_id=0 训练集应为 (500,52)，其余为 (480,52)"""
    X = full_dataset[task_id]["train"]
    expected_n = TRAIN_SAMPLES_NORMAL if task_id == 0 else TRAIN_SAMPLES_FAULT
    assert X.shape == (expected_n, NUM_FEATURES), \
        f"Task {task_id} 训练集形状错误：{X.shape}，期望 ({expected_n}, {NUM_FEATURES})"


@pytest.mark.parametrize("task_id", range(NUM_TASKS))
def test_test_shape(full_dataset, task_id):
    """每个任务的测试集 shape 应为 (960, 52)"""
    X = full_dataset[task_id]["test"]
    assert X.shape == (TEST_SAMPLES, NUM_FEATURES), \
        f"Task {task_id} 测试集形状错误：{X.shape}"


@pytest.mark.parametrize("task_id", range(NUM_TASKS))
def test_no_nan(full_dataset, task_id):
    """训练/测试集均不含 NaN"""
    assert not np.isnan(full_dataset[task_id]["train"]).any(), \
        f"Task {task_id} 训练集含 NaN"
    assert not np.isnan(full_dataset[task_id]["test"]).any(), \
        f"Task {task_id} 测试集含 NaN"


@pytest.mark.parametrize("task_id", range(NUM_TASKS))
def test_no_inf(full_dataset, task_id):
    """训练/测试集均不含 Inf"""
    assert not np.isinf(full_dataset[task_id]["train"]).any(), \
        f"Task {task_id} 训练集含 Inf"
    assert not np.isinf(full_dataset[task_id]["test"]).any(), \
        f"Task {task_id} 测试集含 Inf"


# ─────────────────────────────────────────────────────────────────────────────
# 测试 TEPWindowDataset（需要 torch）
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
def test_window_dataset_len():
    """Dataset 长度应与窗口数一致"""
    X = np.random.rand(100, 50, 52).astype(np.float32)
    y = np.zeros(100, dtype=np.int64)
    ds = TEPWindowDataset(X, y)
    assert len(ds) == 100


@requires_torch
def test_window_dataset_item_shapes():
    """__getitem__ 应返回 (Tensor[50,52], Tensor[scalar])"""
    X = np.random.rand(10, 50, 52).astype(np.float32)
    y = np.arange(10, dtype=np.int64)
    ds = TEPWindowDataset(X, y)
    x_item, y_item = ds[0]
    assert x_item.shape == (50, 52), f"特征形状错误：{x_item.shape}"
    assert y_item.shape == (), f"标签应为标量，实际：{y_item.shape}"


@requires_torch
def test_window_dataset_label_value():
    """标签值应与输入一致"""
    X = np.random.rand(5, 50, 52).astype(np.float32)
    y = np.array([3, 7, 11, 15, 20], dtype=np.int64)
    ds = TEPWindowDataset(X, y)
    for i, expected in enumerate(y):
        _, y_item = ds[i]
        assert int(y_item) == expected, \
            f"idx={i} 标签期望 {expected}，实际 {int(y_item)}"
