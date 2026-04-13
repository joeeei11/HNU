"""
EWC 模块单元测试

测试范围：
  - EWC.compute_fisher() 后 fisher_dict 非空，值非负
  - EWC.penalty() 在首个任务前返回 0；任务后返回正数
  - EWC.update_task() 累积合并正确（F_total = F_old + F_new）
  - metrics.compute_metrics() 基本正确性
  - metrics.compute_bwt() 数值正确性

若 PyTorch 不可用（本机 DLL 问题），整个模块跳过。
"""

import subprocess
import sys

import pytest
import numpy as np


# ─── 检测 torch 是否可正常导入（子进程隔离 DLL 崩溃） ──────────
def _check_torch_available() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        timeout=15,
    )
    return result.returncode == 0


if not _check_torch_available():
    pytest.skip(
        "PyTorch 在本机不可用（DLL 加载失败），EWC 测试模块跳过，"
        "请在 AutoDL 服务器运行。",
        allow_module_level=True,
    )

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.continual.ewc import EWC  # noqa: E402
from src.models.fault_classifier import FaultClassifier  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_metrics,
    compute_bwt,
    compute_avg_acc,
)


# ─── 公共测试参数 ─────────────────────────────────────────────
BATCH_SIZE = 4
WINDOW_SIZE = 50
N_FEATURES = 52
NUM_CLASSES = 22
N_SAMPLES = 20  # 测试时使用小样本量


# ─── 辅助函数 ─────────────────────────────────────────────────

def _make_model():
    """创建一个小规模 FaultClassifier（减少测试耗时）。"""
    cfg = {
        "d_model": 64,
        "cnn_channels": [32, 64],
        "transformer_layers": 1,
        "transformer_heads": 4,
        "dropout": 0.0,
        "window_size": WINDOW_SIZE,
    }
    return FaultClassifier(num_classes=NUM_CLASSES, config=cfg)


def _make_loader(n: int = N_SAMPLES, n_classes: int = 4):
    """生成随机测试 DataLoader。"""
    x = torch.randn(n, WINDOW_SIZE, N_FEATURES, dtype=torch.float32)
    y = torch.randint(0, n_classes, (n,), dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


# ─── 测试用例 ─────────────────────────────────────────────────

class TestEWC:
    """EWC 核心功能测试。"""

    def test_penalty_zero_before_first_task(self):
        """首个任务训练前（fisher_dict 为空），penalty 应为 0。"""
        model = _make_model()
        ewc = EWC(model=model, lambda_ewc=5000.0)

        penalty = ewc.penalty(model)
        assert penalty.item() == pytest.approx(0.0), (
            "首个任务前 EWC penalty 应为 0，实际为 {penalty.item()}"
        )

    def test_compute_fisher_populates_dict(self):
        """compute_fisher 后 fisher_dict 应包含所有可训练参数层。"""
        model = _make_model()
        ewc = EWC(model=model, lambda_ewc=5000.0)
        loader = _make_loader()

        ewc.compute_fisher(loader, n_samples=16)
        ewc.update_task()

        # 检查 fisher_dict 非空
        assert len(ewc.fisher_dict) > 0, "compute_fisher 后 fisher_dict 不应为空"

        # 检查所有 Fisher 值非负（F = E[grad²] ≥ 0）
        for name, f in ewc.fisher_dict.items():
            assert (f >= 0).all(), f"参数 {name} 的 Fisher 值含负数"

    def test_penalty_positive_after_first_task(self):
        """compute_fisher + update_task 后，若模型参数发生改变，penalty 应为正数。"""
        model = _make_model()
        ewc = EWC(model=model, lambda_ewc=5000.0)
        loader = _make_loader()

        ewc.compute_fisher(loader, n_samples=16)
        ewc.update_task()

        # 随机修改一个参数，使其偏离 optimal_params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
                break  # 只改第一层足矣

        penalty = ewc.penalty(model)
        assert penalty.item() > 0.0, (
            f"参数偏移后 EWC penalty 应为正数，实际为 {penalty.item()}"
        )

    def test_fisher_accumulation(self):
        """两次 update_task 后，Fisher 值应等于两次 compute_fisher 之和。"""
        model = _make_model()
        ewc = EWC(model=model, lambda_ewc=1.0)
        loader = _make_loader()

        # Task 0
        ewc.compute_fisher(loader, n_samples=16)
        fisher_task0 = {k: v.clone() for k, v in ewc._current_fisher.items()}
        ewc.update_task()

        # Task 1
        ewc.compute_fisher(loader, n_samples=16)
        fisher_task1 = {k: v.clone() for k, v in ewc._current_fisher.items()}
        ewc.update_task()

        # 验证累积：fisher_dict = fisher_task0 + fisher_task1
        for name in fisher_task0:
            expected = fisher_task0[name] + fisher_task1[name]
            actual = ewc.fisher_dict[name]
            assert torch.allclose(actual.cpu(), expected.cpu(), atol=1e-6), (
                f"参数 {name} 的 Fisher 累积不正确"
            )

    def test_is_ready(self):
        """is_ready() 应在首次 update_task 后返回 True。"""
        model = _make_model()
        ewc = EWC(model=model)
        loader = _make_loader()

        assert not ewc.is_ready(), "任务前 is_ready 应为 False"
        ewc.compute_fisher(loader, n_samples=8)
        ewc.update_task()
        assert ewc.is_ready(), "update_task 后 is_ready 应为 True"


# ─── metrics 测试 ────────────────────────────────────────────

class TestMetrics:
    """评估指标计算测试。"""

    def test_compute_metrics_perfect(self):
        """完美预测时 ACC=1, FAR=0, FDR=1。"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        m = compute_metrics(y_true, y_pred, normal_class=0)
        assert m["acc"] == pytest.approx(1.0)
        assert m["far"] == pytest.approx(0.0)
        assert m["fdr"] == pytest.approx(1.0)

    def test_compute_metrics_all_wrong(self):
        """全部误判时 FAR=1（正常全部误报）。"""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 2, 3])
        m = compute_metrics(y_true, y_pred, normal_class=0)
        assert m["acc"] == pytest.approx(0.0)
        assert m["far"] == pytest.approx(1.0)

    def test_compute_bwt_no_forgetting(self):
        """无遗忘时（R[T-1][j] == R[j][j]），BWT = 0。"""
        R = [
            [0.9],
            [0.9, 0.85],
            [0.9, 0.85, 0.8],
        ]
        bwt = compute_bwt(R)
        assert bwt == pytest.approx(0.0), f"无遗忘时 BWT 应为 0，实际 {bwt}"

    def test_compute_bwt_forgetting(self):
        """有遗忘时 BWT < 0。"""
        # 训练 Task 2 后，Task 0 从 0.9 降到 0.7，Task 1 从 0.8 降到 0.6
        R = [
            [0.9],
            [0.9, 0.8],
            [0.7, 0.6, 0.85],
        ]
        bwt = compute_bwt(R)
        # BWT = ((0.7-0.9) + (0.6-0.8)) / 2 = (-0.2 + -0.2) / 2 = -0.2
        assert bwt == pytest.approx(-0.2, abs=1e-6)

    def test_compute_avg_acc(self):
        """avg_acc 应等于最后一行均值。"""
        R = [
            [0.9],
            [0.8, 0.85],
            [0.7, 0.75, 0.8],
        ]
        avg = compute_avg_acc(R)
        expected = (0.7 + 0.75 + 0.8) / 3
        assert avg == pytest.approx(expected, abs=1e-6)

    def test_bwt_single_task(self):
        """只有 1 个任务时，BWT 应为 0。"""
        R = [[0.9]]
        assert compute_bwt(R) == pytest.approx(0.0)
