"""
Phase 4 单元测试：ReplayBuffer + ContinualTrainer

测试范围：
  - ReplayBuffer.add_task_samples()：蓄水池采样、容量限制、多类别
  - ReplayBuffer.sample_replay_batch()：返回正确形状、空 buffer 处理
  - ReplayBuffer.state_dict() / load_state_dict()：序列化/反序列化
  - ContinualTrainer.train_task()：正常执行、loss 分解记录
  - ContinualTrainer.save_checkpoint() / load_checkpoint()：状态持久化

若 PyTorch 不可用（本机 DLL 问题），整个模块跳过。
"""

import subprocess
import sys
import tempfile
import os

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
        "PyTorch 在本机不可用（DLL 加载失败），Replay 测试模块跳过，"
        "请在 AutoDL 服务器运行。",
        allow_module_level=True,
    )

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.continual.replay_buffer import ReplayBuffer  # noqa: E402
from src.continual.ewc import EWC  # noqa: E402
from src.continual.trainer import ContinualTrainer  # noqa: E402
from src.models.fault_classifier import FaultClassifier  # noqa: E402


# ─── 公共参数 ─────────────────────────────────────────────────
WINDOW_SIZE = 20
N_FEATURES = 52
NUM_CLASSES = 22


# ─── 辅助函数 ─────────────────────────────────────────────────

def _make_model():
    cfg = {
        "d_model": 32,
        "cnn_channels": [16, 32],
        "transformer_layers": 1,
        "transformer_heads": 4,
        "dropout": 0.0,
        "window_size": WINDOW_SIZE,
    }
    return FaultClassifier(num_classes=NUM_CLASSES, config=cfg)


def _make_windows(n: int, n_classes: int = 4, start_label: int = 0):
    """生成随机滑窗样本，类别标签从 start_label 开始。"""
    X = np.random.randn(n, WINDOW_SIZE, N_FEATURES).astype(np.float32)
    labels = list(range(start_label, start_label + n_classes))
    y = np.array([labels[i % n_classes] for i in range(n)], dtype=np.int64)
    return X, y


def _make_loader(n: int = 40, n_classes: int = 4):
    x = torch.randn(n, WINDOW_SIZE, N_FEATURES, dtype=torch.float32)
    y = torch.randint(0, n_classes, (n,), dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=8, shuffle=True)


# ─── ReplayBuffer 测试 ────────────────────────────────────────

class TestReplayBuffer:

    def test_add_samples_respects_capacity(self):
        """每类样本数不超过 buffer_size_per_class。"""
        buf = ReplayBuffer(buffer_size_per_class=10)
        X, y = _make_windows(n=200, n_classes=4)  # 每类约 50 条
        buf.add_task_samples(X, y)

        stats = buf.get_stats()
        for cls, cnt in stats.items():
            assert cnt <= 10, f"类别 {cls} 超出容量限制：{cnt}"

    def test_add_samples_small_dataset(self):
        """样本数少于容量时，全部保留。"""
        buf = ReplayBuffer(buffer_size_per_class=50)
        X, y = _make_windows(n=20, n_classes=4)  # 每类约 5 条
        buf.add_task_samples(X, y)

        stats = buf.get_stats()
        assert sum(stats.values()) == 20, "样本数少于容量时应全部保留"

    def test_multiple_tasks_accumulate(self):
        """多次 add_task_samples 累积不同类别。"""
        buf = ReplayBuffer(buffer_size_per_class=30)

        X0, y0 = _make_windows(n=40, n_classes=4, start_label=0)   # 类 0~3
        X1, y1 = _make_windows(n=50, n_classes=5, start_label=4)   # 类 4~8
        buf.add_task_samples(X0, y0)
        buf.add_task_samples(X1, y1)

        stats = buf.get_stats()
        assert len(stats) == 9, f"应有 9 个类别，实际 {len(stats)}"

    def test_total_len(self):
        """__len__ 返回总样本数。"""
        buf = ReplayBuffer(buffer_size_per_class=20)
        X, y = _make_windows(n=100, n_classes=5)
        buf.add_task_samples(X, y)
        assert len(buf) <= 5 * 20, "总样本数不超过 n_classes * capacity"

    def test_empty_buffer_sample(self):
        """空 buffer 采样返回空 Tensor，不报错。"""
        buf = ReplayBuffer(buffer_size_per_class=50)
        X, y = buf.sample_replay_batch(batch_size=32)
        assert X.numel() == 0 and y.numel() == 0

    def test_sample_batch_shape(self):
        """sample_replay_batch 返回正确形状的 Tensor。"""
        buf = ReplayBuffer(buffer_size_per_class=20, random_seed=0)
        X_np, y_np = _make_windows(n=100, n_classes=4)
        buf.add_task_samples(X_np, y_np)

        X_t, y_t = buf.sample_replay_batch(batch_size=16)
        assert X_t.ndim == 3, f"X 应为 3D，实际 {X_t.ndim}D"
        assert X_t.shape[1:] == (WINDOW_SIZE, N_FEATURES)
        assert X_t.shape[0] == y_t.shape[0]
        assert X_t.dtype == torch.float32
        assert y_t.dtype == torch.long

    def test_state_dict_roundtrip(self):
        """state_dict / load_state_dict 序列化后数据一致。"""
        buf = ReplayBuffer(buffer_size_per_class=15, random_seed=42)
        X, y = _make_windows(n=60, n_classes=3)
        buf.add_task_samples(X, y)

        state = buf.state_dict()

        buf2 = ReplayBuffer()
        buf2.load_state_dict(state)

        assert buf2.buffer_size_per_class == 15
        assert buf2.get_stats() == buf.get_stats()
        assert len(buf2) == len(buf)

        # 数据内容一致
        for cls in buf._buffers:
            np.testing.assert_array_equal(
                buf._buffers[cls]["X"],
                buf2._buffers[cls]["X"],
            )


# ─── ContinualTrainer 测试 ────────────────────────────────────

class TestContinualTrainer:

    def _make_trainer(self):
        model = _make_model()
        ewc = EWC(model=model, lambda_ewc=100.0)
        replay = ReplayBuffer(buffer_size_per_class=10)
        config = {
            "training": {"lr": 1e-3, "use_amp": False},
            "ewc": {"fisher_samples": 8},
            "replay": {"replay_batch_size": 4},
        }
        trainer = ContinualTrainer(
            model=model,
            ewc=ewc,
            replay_buffer=replay,
            config=config,
            device=torch.device("cpu"),
        )
        return trainer, model, replay

    def test_train_task0_returns_dict(self):
        """Task 0 训练后返回正确结构的 dict。"""
        trainer, _, _ = self._make_trainer()
        loader = _make_loader(n=24, n_classes=4)
        X_np, y_np = _make_windows(n=24, n_classes=4, start_label=0)

        result = trainer.train_task(
            task_id=0,
            train_loader=loader,
            epochs=2,
            log_every=1,
            X_train_np=X_np,
            y_train_np=y_np,
        )

        assert result["task_id"] == 0
        assert "loss_history" in result
        assert len(result["loss_history"]) == 2
        assert "final_loss_ce" in result
        assert "final_loss_ewc" in result

    def test_task0_ewc_penalty_zero(self):
        """Task 0 训练时 EWC penalty 为 0（Fisher 尚未计算）。"""
        trainer, model, _ = self._make_trainer()
        loader = _make_loader(n=16, n_classes=4)
        X_np, y_np = _make_windows(n=16, n_classes=4)

        result = trainer.train_task(
            task_id=0,
            train_loader=loader,
            epochs=1,
            X_train_np=X_np,
            y_train_np=y_np,
        )

        # Task 0 首个 epoch 的 loss_ewc 应接近 0（训练时 EWC 未就绪）
        first_epoch = result["loss_history"][0]
        assert first_epoch["loss_ewc"] == pytest.approx(0.0, abs=1e-6)

    def test_replay_buffer_populated_after_task0(self):
        """Task 0 训练后 ReplayBuffer 应包含样本。"""
        trainer, _, replay = self._make_trainer()
        loader = _make_loader(n=40, n_classes=4)
        X_np, y_np = _make_windows(n=40, n_classes=4)

        trainer.train_task(
            task_id=0,
            train_loader=loader,
            epochs=1,
            X_train_np=X_np,
            y_train_np=y_np,
        )

        assert len(replay) > 0, "Task 0 后 ReplayBuffer 应有样本"

    def test_checkpoint_roundtrip(self):
        """save_checkpoint / load_checkpoint 后状态一致。"""
        trainer, model, replay = self._make_trainer()
        loader = _make_loader(n=24, n_classes=4)
        X_np, y_np = _make_windows(n=24, n_classes=4)

        trainer.train_task(
            task_id=0,
            train_loader=loader,
            epochs=1,
            X_train_np=X_np,
            y_train_np=y_np,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
            trainer.save_checkpoint(ckpt_path)

            # 新建 trainer，从 checkpoint 加载
            trainer2, _, _ = self._make_trainer()
            trainer2.load_checkpoint(ckpt_path)

            assert trainer2._completed_tasks == trainer._completed_tasks
            assert len(trainer2.replay_buffer) == len(replay)

            # 模型参数一致
            for (n1, p1), (n2, p2) in zip(
                trainer.model.named_parameters(),
                trainer2.model.named_parameters(),
            ):
                assert torch.allclose(p1, p2), f"参数 {n1} 不一致"
