"""
生成 proposed 方案的 22 类混淆矩阵

用法：
    python experiments/run_confusion_matrix.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.task_splitter import TaskSplitter, TASK_CLASSES
from src.models.fault_classifier import FaultClassifier
from src.evaluation.visualizer import plot_confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="生成混淆矩阵")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/proposed_final.pth",
        help="模型权重路径",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["window_size"] = cfg["training"]["window_size"]
    num_classes = cfg["evaluation"]["num_classes"]
    model = FaultClassifier(num_classes=num_classes, config=model_cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # ContinualTrainer 保存的是完整状态字典，取 model_state
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"已加载权重: {args.checkpoint}")

    # 加载所有任务测试集
    splitter = TaskSplitter(
        raw_dir=cfg["data"]["raw_dir"],
        window_size=cfg["training"]["window_size"],
        stride=cfg["training"]["stride"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 0),
    )

    all_y_true: list = []
    all_y_pred: list = []

    use_amp = cfg["training"].get("use_amp", True) and device.type == "cuda"

    for task_id in range(len(TASK_CLASSES)):
        _, test_loader = splitter.get_task(task_id)
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, dtype=torch.float32)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                pred = logits.argmax(dim=1).cpu().numpy()
                all_y_true.extend(y.numpy().tolist())
                all_y_pred.extend(pred.tolist())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    acc = (y_true == y_pred).mean() * 100
    print(f"全集准确率: {acc:.2f}%  样本数: {len(y_true)}")

    # 保存预测结果（供 Notebook 使用）
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "y_true.npy"), y_true)
    np.save(os.path.join(results_dir, "y_pred.npy"), y_pred)
    print(f"预测结果已保存: {results_dir}/y_true.npy, y_pred.npy")

    # 生成混淆矩阵
    save_path = os.path.join(results_dir, "figures", "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, save_path=save_path)
    print(f"混淆矩阵已保存: {save_path}")


if __name__ == "__main__":
    main()
