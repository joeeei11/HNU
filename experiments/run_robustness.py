"""
鲁棒性测试：在不同高斯噪声强度下评估各方法的性能

用法：
    # 使用默认噪声级别 [0, 0.1, 0.3, 0.5]
    python experiments/run_robustness.py --config config/config.yaml

    # 自定义噪声级别
    python experiments/run_robustness.py --noise_std 0 0.1 0.2 0.5 1.0

    # 指定 checkpoint 目录（默认 results/）
    python experiments/run_robustness.py --ckpt_dir results/

运行流程：
  1. 加载 4 种方法（finetune / ewc_only / replay_only / proposed）的最终权重
  2. 对每个噪声级别，在所有任务测试集上叠加高斯噪声（标准化后特征空间）
  3. 记录各方法在各噪声级别下的 ACC / FAR / FDR
  4. 保存到 results/robustness_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.task_splitter import TaskSplitter, TASK_CLASSES
from src.models.fault_classifier import FaultClassifier
from src.evaluation.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 加载映射
# ─────────────────────────────────────────────────────────────────────────────

# 各方法最终 checkpoint 文件名及加载方式
# finetune/ewc_only/replay_only: 直接保存 model.state_dict()
# proposed: ContinualTrainer checkpoint，模型权重在 "model_state" 键下
METHOD_CKPTS = {
    "finetune": {
        "filename": "finetune_task3.pth",
        "key": None,  # 直接是 state_dict
    },
    "ewc_only": {
        "filename": "ewc_only_task3.pth",
        "key": None,
    },
    "replay_only": {
        "filename": "replay_only_task3.pth",
        "key": None,
    },
    "proposed": {
        "filename": "proposed_final.pth",
        "key": "model_state",  # 需从 checkpoint dict 中取
    },
}


def build_model(cfg: dict) -> FaultClassifier:
    """根据 config 构建 FaultClassifier。"""
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["window_size"] = cfg["training"]["window_size"]
    num_classes = cfg["evaluation"]["num_classes"]
    return FaultClassifier(num_classes=num_classes, config=model_cfg)


def build_splitter(cfg: dict) -> TaskSplitter:
    """根据 config 构建 TaskSplitter。"""
    return TaskSplitter(
        raw_dir=cfg["data"]["raw_dir"],
        window_size=cfg["training"]["window_size"],
        stride=cfg["training"]["stride"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 0),
    )


def load_model_weights(
    model: FaultClassifier,
    ckpt_path: str,
    key: Optional[str],
    device: torch.device,
) -> None:
    """加载模型权重（兼容两种 checkpoint 格式）。

    Args:
        model:     FaultClassifier 实例
        ckpt_path: checkpoint 文件路径
        key:       若为 None 则 checkpoint 直接是 state_dict；
                   否则从 checkpoint[key] 取 state_dict
        device:    目标设备
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if key is not None:
        state_dict = checkpoint[key]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# 噪声注入 + 评估
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_noise(
    model: FaultClassifier,
    splitter: TaskSplitter,
    noise_std: float,
    device: torch.device,
    use_amp: bool = True,
) -> Dict:
    """在所有任务测试集上叠加指定强度的高斯噪声后评估模型。

    噪声叠加在标准化后的特征上：x_noisy = x + N(0, noise_std)

    Args:
        model:     已加载权重的模型
        splitter:  TaskSplitter 实例
        noise_std: 高斯噪声标准差（0 表示无噪声）
        device:    计算设备
        use_amp:   是否启用 AMP

    Returns:
        dict，包含 acc / far / fdr / per_class_acc
    """
    n_tasks = len(TASK_CLASSES)
    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    model.eval()
    with torch.no_grad():
        for task_id in range(n_tasks):
            _, test_loader = splitter.get_task(task_id)
            for x, y in test_loader:
                x = x.to(device, dtype=torch.float32)

                # 叠加高斯噪声（在标准化后的特征空间中）
                if noise_std > 0:
                    noise = torch.randn_like(x) * noise_std
                    x = x + noise

                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    logits = model(x)

                pred = logits.argmax(dim=1).cpu().numpy()
                all_y_true.extend(y.numpy().tolist())
                all_y_pred.extend(pred.tolist())

    metrics = compute_metrics(
        np.array(all_y_true),
        np.array(all_y_pred),
        normal_class=0,
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 主实验
# ─────────────────────────────────────────────────────────────────────────────

def run_robustness(
    cfg: dict,
    noise_levels: List[float],
    ckpt_dir: str = "results",
) -> dict:
    """运行鲁棒性测试，评估各方法在不同噪声强度下的性能。

    Args:
        cfg:          config.yaml 配置字典
        noise_levels: 噪声标准差列表，如 [0, 0.1, 0.3, 0.5]
        ckpt_dir:     checkpoint 所在目录

    Returns:
        results: 嵌套字典 {method: {noise_std: metrics_dict}}
    """
    print("=" * 65)
    print("鲁棒性测试：高斯噪声扰动下的模型性能评估")
    print(f"噪声级别: {noise_levels}")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    use_amp = cfg["training"]["use_amp"]
    splitter = build_splitter(cfg)

    # 检查 checkpoint 文件是否存在
    available_methods = {}
    for method_name, ckpt_info in METHOD_CKPTS.items():
        ckpt_path = os.path.join(ckpt_dir, ckpt_info["filename"])
        if os.path.exists(ckpt_path):
            available_methods[method_name] = ckpt_info
            print(f"  [OK] {method_name}: {ckpt_path}")
        else:
            print(f"  [SKIP] {method_name}: {ckpt_path} 不存在，跳过")

    if not available_methods:
        print("\n没有找到任何 checkpoint 文件，请先运行训练实验。")
        return {}

    # 逐方法、逐噪声级别评估
    results: Dict[str, Dict[str, dict]] = {}

    for method_name, ckpt_info in available_methods.items():
        print(f"\n{'─' * 55}")
        print(f"评估方法: {method_name}")
        print(f"{'─' * 55}")

        ckpt_path = os.path.join(ckpt_dir, ckpt_info["filename"])
        model = build_model(cfg)
        load_model_weights(model, ckpt_path, ckpt_info["key"], device)

        method_results: Dict[str, dict] = {}

        for noise_std in noise_levels:
            metrics = evaluate_with_noise(
                model, splitter, noise_std, device, use_amp
            )
            method_results[str(noise_std)] = metrics

            print(
                f"  noise_std={noise_std:.1f}  "
                f"ACC={metrics['acc'] * 100:.2f}%  "
                f"FAR={metrics['far'] * 100:.2f}%  "
                f"FDR={metrics['fdr'] * 100:.2f}%"
            )

        results[method_name] = method_results

    # ── 汇总打印 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("鲁棒性测试汇总（ACC %）")
    print("=" * 65)

    # 表头
    header = f"{'方法':<18}" + "".join(f"{'σ=' + str(n):<12}" for n in noise_levels)
    print(header)
    print("─" * len(header))

    for method_name in results:
        row = f"{method_name:<18}"
        for noise_std in noise_levels:
            acc = results[method_name][str(noise_std)]["acc"]
            row += f"{acc * 100:<12.2f}"
        print(row)

    # 性能下降率（noise=0 → noise=max）
    max_noise = str(max(noise_levels))
    if "0" in results.get(list(results.keys())[0], {}) or "0.0" in results.get(list(results.keys())[0], {}):
        print(f"\n性能衰减（σ=0 → σ={max(noise_levels)}）:")
        for method_name in results:
            noise_key_0 = "0" if "0" in results[method_name] else "0.0"
            acc_clean = results[method_name][noise_key_0]["acc"]
            acc_noisy = results[method_name][max_noise]["acc"]
            if acc_clean > 0:
                drop = (acc_clean - acc_noisy) / acc_clean * 100
                print(f"  {method_name}: {drop:.1f}% 下降")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "robustness_results.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至 {save_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="鲁棒性测试：高斯噪声扰动下的模型性能评估"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径（默认 config/config.yaml）",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        nargs="+",
        default=[0, 0.1, 0.3, 0.5],
        help="高斯噪声标准差列表（默认 0 0.1 0.3 0.5）",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="results",
        help="checkpoint 文件所在目录（默认 results/）",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 设置随机种子（保证噪声可复现）
    seed: int = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子: {seed}")

    run_robustness(
        cfg=cfg,
        noise_levels=args.noise_std,
        ckpt_dir=args.ckpt_dir,
    )


if __name__ == "__main__":
    main()
