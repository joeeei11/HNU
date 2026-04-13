"""
模型单元测试（本机 CPU / 服务器 CUDA 均可运行）

本机测试范围：
  若 PyTorch DLL 可正常加载，运行全部 CPU 测试；
  若 torch 在本机不可用（DLL 崩溃），整个模块跳过（在 AutoDL 服务器运行）。

验收标准：
  - FaultClassifier.forward 输出 shape = [B, 22]
  - get_features 输出 shape = [B, 256]
  - count_params 在 [500_000, 5_000_000] 范围内
  - CUDA 测试需在服务器上运行
"""

import subprocess
import sys

import pytest


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
        "PyTorch 在本机不可用（DLL 加载失败），整个模型测试模块跳过，"
        "请在 AutoDL 服务器运行。",
        allow_module_level=True,
    )

# ─── PyTorch 可用，执行正式导入 ─────────────────────────────────
import torch  # noqa: E402

from src.models.cnn_encoder import CNNEncoder              # noqa: E402
from src.models.transformer_encoder import TransformerEncoder  # noqa: E402
from src.models.fault_classifier import FaultClassifier    # noqa: E402


# ─── 公共测试参数 ─────────────────────────────────────────────
BATCH_SIZE = 4          # 本机 CPU 内存友好
WINDOW_SIZE = 50
N_FEATURES = 52
NUM_CLASSES = 22
D_MODEL = 256

TEST_CONFIG = {
    "d_model": D_MODEL,
    "cnn_channels": [128, 256],
    "transformer_layers": 4,
    "transformer_heads": 8,
    "dropout": 0.1,
    "window_size": WINDOW_SIZE,
}


# ─── 辅助函数 ─────────────────────────────────────────────────

def _make_input(device: torch.device) -> torch.Tensor:
    """生成随机输入 [B, W, 52]"""
    return torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES, device=device)


def _make_model(device: torch.device) -> FaultClassifier:
    return FaultClassifier(num_classes=NUM_CLASSES, config=TEST_CONFIG).to(device)


# ─── CNNEncoder 测试 ──────────────────────────────────────────

class TestCNNEncoder:
    """CNN 编码器单元测试"""

    def test_output_shape_pool_true(self):
        """pool=True → [B, 256]"""
        cnn = CNNEncoder(in_channels=N_FEATURES, channels=[128, 256])
        x = torch.randn(BATCH_SIZE, N_FEATURES, WINDOW_SIZE)
        out = cnn(x, pool=True)
        assert out.shape == (BATCH_SIZE, 256), f"实际 shape: {out.shape}"

    def test_output_shape_pool_false(self):
        """pool=False → [B, 256, W]"""
        cnn = CNNEncoder(in_channels=N_FEATURES, channels=[128, 256])
        x = torch.randn(BATCH_SIZE, N_FEATURES, WINDOW_SIZE)
        out = cnn(x, pool=False)
        assert out.shape == (BATCH_SIZE, 256, WINDOW_SIZE), f"实际 shape: {out.shape}"

    def test_no_nan_output(self):
        """输出不含 NaN"""
        cnn = CNNEncoder()
        x = torch.randn(BATCH_SIZE, N_FEATURES, WINDOW_SIZE)
        assert not torch.isnan(cnn(x)).any()

    def test_batch_size_1_eval(self):
        """batch_size=1 时 eval 模式下 BatchNorm 不崩溃"""
        cnn = CNNEncoder().eval()
        x = torch.randn(1, N_FEATURES, WINDOW_SIZE)
        out = cnn(x)
        assert out.shape == (1, 256)


# ─── TransformerEncoder 测试 ─────────────────────────────────

class TestTransformerEncoder:
    """Transformer 编码器单元测试"""

    def test_output_shape(self):
        """输出 [B, d_model]"""
        enc = TransformerEncoder(
            d_model=D_MODEL, nhead=8, num_layers=4, window_size=WINDOW_SIZE
        )
        x = torch.randn(BATCH_SIZE, WINDOW_SIZE, D_MODEL)
        out = enc(x)
        assert out.shape == (BATCH_SIZE, D_MODEL), f"实际 shape: {out.shape}"

    def test_no_nan_output(self):
        """输出不含 NaN"""
        enc = TransformerEncoder(d_model=D_MODEL, nhead=8, num_layers=4, window_size=WINDOW_SIZE)
        x = torch.randn(BATCH_SIZE, WINDOW_SIZE, D_MODEL)
        assert not torch.isnan(enc(x)).any()

    def test_shorter_sequence(self):
        """序列长度短于 window_size 时，位置嵌入切片正常"""
        enc = TransformerEncoder(d_model=D_MODEL, nhead=8, num_layers=2, window_size=WINDOW_SIZE)
        x = torch.randn(BATCH_SIZE, 30, D_MODEL)    # 短于 window_size=50
        out = enc(x)
        assert out.shape == (BATCH_SIZE, D_MODEL)


# ─── FaultClassifier 测试 ────────────────────────────────────

class TestFaultClassifier:
    """整体分类模型单元测试"""

    def test_forward_shape_cpu(self):
        """forward 输出 [B, 22]（CPU）"""
        model = _make_model(torch.device("cpu"))
        x = _make_input(torch.device("cpu"))
        logits = model(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES), f"实际 shape: {logits.shape}"

    def test_get_features_shape_cpu(self):
        """get_features 输出 [B, 256]（CPU）"""
        model = _make_model(torch.device("cpu"))
        x = _make_input(torch.device("cpu"))
        features = model.get_features(x)
        assert features.shape == (BATCH_SIZE, D_MODEL), f"实际 shape: {features.shape}"

    def test_count_params_range(self):
        """参数量在 [500K, 5M] 范围内"""
        model = _make_model(torch.device("cpu"))
        n = model.count_params()
        assert 500_000 <= n <= 5_000_000, f"参数量 {n:,} 超出预期范围 [500K, 5M]"

    def test_no_nan_forward(self):
        """forward 输出不含 NaN"""
        model = _make_model(torch.device("cpu"))
        x = _make_input(torch.device("cpu"))
        assert not torch.isnan(model(x)).any()

    def test_no_nan_features(self):
        """get_features 输出不含 NaN"""
        model = _make_model(torch.device("cpu"))
        x = _make_input(torch.device("cpu"))
        assert not torch.isnan(model.get_features(x)).any()

    def test_eval_no_grad(self):
        """eval 模式 + no_grad 下 forward 正常"""
        model = _make_model(torch.device("cpu"))
        model.eval()
        x = _make_input(torch.device("cpu"))
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_from_config(self, tmp_path):
        """from_config 从 YAML 文件正确创建模型"""
        import yaml
        config = {
            "model": {
                "d_model": 64,
                "cnn_channels": [32, 64],
                "transformer_layers": 2,
                "transformer_heads": 4,
                "dropout": 0.1,
            },
            "training": {"window_size": 50},
            "evaluation": {"num_classes": 22},
        }
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config), encoding="utf-8")
        model = FaultClassifier.from_config(str(config_file))
        x = torch.randn(2, 50, 52)
        out = model(x)
        assert out.shape == (2, 22)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA 不可用（本机跳过，在 AutoDL 服务器运行）",
    )
    def test_forward_shape_cuda(self):
        """forward 输出 [B, 22]（CUDA）"""
        device = torch.device("cuda")
        model = _make_model(device)
        x = _make_input(device)
        logits = model(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA 不可用（本机跳过，在 AutoDL 服务器运行）",
    )
    def test_get_features_shape_cuda(self):
        """get_features 输出 [B, 256]（CUDA）"""
        device = torch.device("cuda")
        model = _make_model(device)
        x = _make_input(device)
        features = model.get_features(x)
        assert features.shape == (BATCH_SIZE, D_MODEL)
