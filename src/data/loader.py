"""
TEP 数据集加载模块
加载 Tennessee Eastman Process (TEP) 的 .dat 文件并封装为 PyTorch Dataset。

数据格式说明：
  - 训练集 d00.dat（正常工况）：存储为 (52, 500) 的转置格式，需 transpose → (500, 52)
  - 训练集 d01.dat~d21.dat（故障）：标准 (480, 52) 格式
  - 所有测试集 _te.dat：标准 (960, 52) 格式
  - load_single_file 自动识别并处理转置：若 shape[0]==52 且 shape[1]>52，则转置

设计说明：
  - load_single_file / load_tep_dataset 为纯 numpy 操作，无 torch 依赖，可在本机调试
  - TEPWindowDataset 需要 torch，仅在服务器（AutoDL）环境实例化
  - torch 采用懒加载（仅在 TEPWindowDataset.__init__ 内部导入），
    避免 Python 3.14 本机环境因 DLL 兼容问题导致进程崩溃
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# TEP 数据集常量
NUM_FEATURES = 52        # 传感器特征数
NUM_TASKS = 22           # 任务数（0=正常, 1~21=21种故障）
TRAIN_SAMPLES_NORMAL = 500   # d00.dat（正常工况）训练集样本数（转置后）
TRAIN_SAMPLES_FAULT = 480    # d01~d21.dat（故障）训练集样本数
TEST_SAMPLES = 960       # 所有任务测试集样本数


def load_single_file(path: str) -> np.ndarray:
    """
    加载单个 TEP .dat 文件（空格分隔，52列浮点数）。

    自动处理转置格式：若数组 shape 为 (52, N)（如 d00.dat），
    则自动转置为 (N, 52)，与其他文件格式保持一致。

    Args:
        path: .dat 文件的路径

    Returns:
        shape=(N, 52) 的 float32 数组，N 为样本数

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 转置后列数仍不为 52
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在：{path}")

    data = np.loadtxt(path, dtype=np.float32)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    # 若行数 == NUM_FEATURES 且列数 > NUM_FEATURES → 转置格式（如 d00.dat）
    if data.shape[0] == NUM_FEATURES and data.shape[1] > NUM_FEATURES:
        data = data.T  # (52, N) → (N, 52)

    if data.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"期望 {NUM_FEATURES} 列，实际 {data.shape[1]} 列（转置后）：{path}"
        )

    return data


def load_tep_dataset(raw_dir: str) -> Dict[int, Dict[str, np.ndarray]]:
    """
    批量加载全部 44 个 TEP .dat 文件。

    文件命名规则：
      训练集: d{task_id:02d}.dat        shape=(N_train, 52)
      测试集: d{task_id:02d}_te.dat     shape=(960, 52)

    说明：
      task_id=0 的训练集有 500 行（正常工况），
      task_id=1~21 的训练集各有 480 行（故障工况）。

    Args:
        raw_dir: 存放 .dat 文件的目录路径

    Returns:
        {
            task_id (int): {
                "train": np.ndarray shape=(N_train, 52),
                "test":  np.ndarray shape=(960, 52)
            }
        }
        task_id 范围 0~21，0 表示正常工况，1~21 表示 21 种故障
    """
    raw_dir = Path(raw_dir)
    dataset: Dict[int, Dict[str, np.ndarray]] = {}

    for task_id in range(NUM_TASKS):
        train_path = raw_dir / f"d{task_id:02d}.dat"
        test_path = raw_dir / f"d{task_id:02d}_te.dat"

        X_train = load_single_file(str(train_path))
        X_test = load_single_file(str(test_path))

        # 校验测试集行数（所有任务统一 960）
        if X_test.shape[0] != TEST_SAMPLES:
            raise ValueError(
                f"Task {task_id} 测试集期望 {TEST_SAMPLES} 行，实际 {X_test.shape[0]} 行"
            )

        # 训练集行数校验（task_id=0 允许 500，其余要求 480）
        expected_train = TRAIN_SAMPLES_NORMAL if task_id == 0 else TRAIN_SAMPLES_FAULT
        if X_train.shape[0] != expected_train:
            raise ValueError(
                f"Task {task_id} 训练集期望 {expected_train} 行，实际 {X_train.shape[0]} 行"
            )

        dataset[task_id] = {"train": X_train, "test": X_test}

    return dataset


class TEPWindowDataset:
    """
    将滑动窗口切片后的 TEP 数据封装为 PyTorch Dataset。

    本类在 __init__ 内部懒加载 torch，避免模块导入时的 DLL 崩溃问题。
    仅在服务器（AutoDL）环境实例化，本机勿直接使用。

    Args:
        X_windows: shape=(N_windows, window_size, 52) 的 float32 数组
        y_windows: shape=(N_windows,) 的整数标签数组（全局标签 0~21）
    """

    def __init__(self, X_windows: np.ndarray, y_windows: np.ndarray) -> None:
        import torch                           # 懒加载：仅在实例化时导入
        from torch.utils.data import Dataset   # noqa: F401（注册基类型）

        assert X_windows.shape[0] == y_windows.shape[0], \
            f"样本数不匹配：X={X_windows.shape[0]}, y={y_windows.shape[0]}"

        self.X = torch.from_numpy(X_windows.astype(np.float32))   # (N, W, 52)
        self.y = torch.from_numpy(y_windows.astype(np.int64))      # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple:
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 命令行快速验证入口
# python -m src.data.loader
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    import sys

    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("未找到 config/config.yaml，使用默认路径 Source/data/data/data", file=sys.stderr)
        raw_dir = "Source/data/data/data"
    else:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        raw_dir = cfg["data"]["raw_dir"]

    print(f"加载数据目录：{raw_dir}")
    dataset = load_tep_dataset(raw_dir)

    print(f"\n{'任务':>4}  {'训练集':>14}  {'测试集':>14}")
    print("-" * 40)
    for task_id, splits in dataset.items():
        print(
            f"  {task_id:>2}  "
            f"{str(splits['train'].shape):>14}  "
            f"{str(splits['test'].shape):>14}"
        )

    print(f"\n共 {len(dataset)} 个任务，数据加载完成，无错误。")
