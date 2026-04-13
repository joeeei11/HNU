"""
基线对比实验入口

用法：
    # 静态单任务验证（Task 0，4类，验证模型基础表达能力）
    python experiments/run_baselines.py --method static_task0 --config config/config.yaml

    # 纯 EWC 增量实验（Task 0→1→2→3）
    python experiments/run_baselines.py --method ewc_only --config config/config.yaml

    # Fine-tuning 朴素增量（对照组，展示灾难性遗忘）
    python experiments/run_baselines.py --method finetune --config config/config.yaml

    # 所有基线方法（ewc_only + finetune）
    python experiments/run_baselines.py --method all --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Dict, Optional

import torch
import yaml

# 确保项目根目录在 Python 路径中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.data.task_splitter import TaskSplitter, TASK_CLASSES
from src.models.fault_classifier import FaultClassifier
from src.baselines.static_trainer import StaticTrainer, FineTuningTrainer
from src.baselines.ewc_only_trainer import EWCOnlyTrainer
from src.baselines.replay_only_trainer import ReplayOnlyTrainer
from src.continual.ewc import EWC
from src.continual.replay_buffer import ReplayBuffer
from src.evaluation.metrics import (
    compute_metrics,
    compute_bwt,
    compute_avg_acc,
    format_results_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

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


def evaluate_all_tasks(
    trainer,
    splitter: TaskSplitter,
    n_tasks: int = 4,
) -> List[float]:
    """在所有已构建任务的测试集上评估模型，返回各任务准确率列表。

    Args:
        trainer:   拥有 evaluate_on_loader 方法的训练器
        splitter:  TaskSplitter 实例
        n_tasks:   评估的任务数量

    Returns:
        accs: [acc_task0, acc_task1, ..., acc_task(n-1)]
    """
    accs = []
    for tid in range(n_tasks):
        _, test_loader = splitter.get_task(tid)
        acc = trainer.evaluate_on_loader(test_loader)
        accs.append(acc)
        print(f"    Task {tid} 准确率: {acc * 100:.2f}%")
    return accs


def save_results(results: dict, path: str) -> None:
    """将结果字典保存为 JSON 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至 {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 实验函数
# ─────────────────────────────────────────────────────────────────────────────

def run_static_task0(cfg: dict) -> float:
    """静态训练 Task 0（4类），验证模型基础表达能力。

    验收标准：测试集准确率 ≥ 70%（TEP Fault3 公认难以检测，以70%作为下界）

    Returns:
        final_acc: Task 0 最终测试准确率
    """
    print("=" * 60)
    print("基线实验：静态单任务训练（Task 0，4类）")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    splitter = build_splitter(cfg)
    train_loader, test_loader = splitter.get_task(task_id=0)
    print(f"Task 0 训练批次: {len(train_loader)}  测试批次: {len(test_loader)}")

    model = build_model(cfg)
    model.count_params()

    trainer = StaticTrainer(
        model=model,
        device=device,
        lr=cfg["training"]["lr"],
        use_amp=cfg["training"]["use_amp"],
        task_classes=[0, 1, 2, 3],
    )
    results_dir = cfg.get("results_dir", "results")
    save_path = os.path.join(results_dir, "static_task0.pth")

    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=cfg["training"]["epochs_per_task"],
        log_every=5,
        save_path=save_path,
    )

    final_acc = trainer.evaluate(test_loader)
    print(f"\n最终 Task 0 测试准确率: {final_acc * 100:.2f}%")

    target = 0.70
    if final_acc >= target:
        print(f"✅ 验收通过（目标 ≥ {target * 100:.0f}%）")
    else:
        print(f"⚠️  未达标（目标 {target * 100:.0f}%，实际 {final_acc * 100:.2f}%）")

    return final_acc


def run_ewc_only(cfg: dict) -> dict:
    """纯 EWC 增量实验（Task 0→1→2→3）。

    每个任务训练完后：
      1. 在所有历史任务测试集上评估（填充 results_matrix）
      2. 计算 Fisher 矩阵并累积到 EWC

    Returns:
        results: 包含 results_matrix、BWT、Avg-ACC 等指标
    """
    print("=" * 60)
    print("基线实验：纯 EWC 增量训练（Task 0→1→2→3）")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    splitter = build_splitter(cfg)
    model = build_model(cfg)
    model.count_params()

    ewc = EWC(model=model, lambda_ewc=cfg["ewc"]["lambda"])
    trainer = EWCOnlyTrainer(
        model=model,
        ewc=ewc,
        device=device,
        lr=cfg["training"]["lr"],
        use_amp=cfg["training"]["use_amp"],
        fisher_samples=cfg["ewc"].get("fisher_samples", 500),
    )

    n_tasks = len(TASK_CLASSES)
    # results_matrix[i][j] = 训练完 Task i 后在 Task j 上的准确率
    results_matrix: List[List[Optional[float]]] = [
        [None] * n_tasks for _ in range(n_tasks)
    ]

    epochs = cfg["training"]["epochs_per_task"]
    results_dir = cfg.get("results_dir", "results")

    for task_id in range(n_tasks):
        train_loader, test_loader = splitter.get_task(task_id)
        print(f"\nTask {task_id} 训练集批次: {len(train_loader)}  测试集批次: {len(test_loader)}")

        trainer.train_task(
            task_id=task_id,
            train_loader=train_loader,
            epochs=epochs,
            log_every=5,
            val_loader=test_loader,
        )

        # 保存当前阶段模型权重
        ckpt_path = os.path.join(results_dir, f"ewc_only_task{task_id}.pth")
        os.makedirs(results_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        # 在所有历史任务（0~task_id）上评估
        print(f"\n  训练完 Task {task_id} 后，在所有历史任务上评估：")
        for eval_tid in range(task_id + 1):
            _, eval_test_loader = splitter.get_task(eval_tid)
            acc = trainer.evaluate_on_loader(eval_test_loader)
            results_matrix[task_id][eval_tid] = acc
            print(f"    → Task {eval_tid} 准确率: {acc * 100:.2f}%")

    # 计算汇总指标
    # 过滤掉 None，构建用于计算的有效矩阵
    valid_matrix = [
        [v for v in row if v is not None]
        for row in results_matrix
    ]
    bwt = compute_bwt(valid_matrix)
    avg_acc = compute_avg_acc(valid_matrix)

    print("\n" + "=" * 55)
    print("EWC-Only 实验完成")
    print(f"  BWT（后向转移）: {bwt:.4f}{'（遗忘）' if bwt < 0 else '（正向迁移）'}")
    print(f"  最终平均准确率: {avg_acc * 100:.2f}%")
    print("\n结果矩阵（行=训练完 Task i 后，列=在 Task j 上的准确率）：")
    print(format_results_matrix(results_matrix))

    results = {
        "method": "ewc_only",
        "lambda_ewc": cfg["ewc"]["lambda"],
        "results_matrix": results_matrix,
        "bwt": bwt,
        "avg_acc": avg_acc,
    }
    save_path = os.path.join(results_dir, "ewc_only_results.json")
    save_results(results, save_path)

    return results


def run_finetune(cfg: dict) -> dict:
    """Fine-tuning 朴素增量实验（灾难性遗忘对照组）。

    每个任务训练完后在所有历史任务上评估，展示无保护时的遗忘程度。

    Returns:
        results: 包含 results_matrix、BWT、Avg-ACC 等指标
    """
    print("=" * 60)
    print("基线实验：Fine-tuning 朴素增量（灾难性遗忘对照组）")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    splitter = build_splitter(cfg)
    model = build_model(cfg)
    model.count_params()

    trainer = FineTuningTrainer(
        model=model,
        device=device,
        lr=cfg["training"]["lr"],
        use_amp=cfg["training"]["use_amp"],
    )

    n_tasks = len(TASK_CLASSES)
    results_matrix: List[List[Optional[float]]] = [
        [None] * n_tasks for _ in range(n_tasks)
    ]

    epochs = cfg["training"]["epochs_per_task"]
    results_dir = cfg.get("results_dir", "results")

    for task_id in range(n_tasks):
        train_loader, test_loader = splitter.get_task(task_id)
        print(f"\nTask {task_id} 训练集批次: {len(train_loader)}  测试集批次: {len(test_loader)}")

        trainer.train_task(
            task_id=task_id,
            train_loader=train_loader,
            epochs=epochs,
            log_every=5,
            val_loader=test_loader,
        )

        ckpt_path = os.path.join(results_dir, f"finetune_task{task_id}.pth")
        os.makedirs(results_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        print(f"\n  训练完 Task {task_id} 后，在所有历史任务上评估：")
        for eval_tid in range(task_id + 1):
            _, eval_test_loader = splitter.get_task(eval_tid)
            acc = trainer.evaluate_on_loader(eval_test_loader)
            results_matrix[task_id][eval_tid] = acc
            print(f"    → Task {eval_tid} 准确率: {acc * 100:.2f}%")

    valid_matrix = [
        [v for v in row if v is not None]
        for row in results_matrix
    ]
    bwt = compute_bwt(valid_matrix)
    avg_acc = compute_avg_acc(valid_matrix)

    print("\n" + "=" * 55)
    print("Fine-tuning 实验完成")
    print(f"  BWT（后向转移）: {bwt:.4f}{'（遗忘）' if bwt < 0 else '（正向迁移）'}")
    print(f"  最终平均准确率: {avg_acc * 100:.2f}%")
    print("\n结果矩阵：")
    print(format_results_matrix(results_matrix))

    results = {
        "method": "finetune",
        "results_matrix": results_matrix,
        "bwt": bwt,
        "avg_acc": avg_acc,
    }
    save_path = os.path.join(results_dir, "finetune_results.json")
    save_results(results, save_path)

    return results


def run_replay_only(cfg: dict) -> dict:
    """纯 Replay 增量实验（Task 0→1→2→3）。

    Returns:
        results: 包含 results_matrix、BWT、Avg-ACC 等指标
    """
    print("=" * 60)
    print("基线实验：纯 Replay 增量训练（Task 0→1→2→3）")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    splitter = build_splitter(cfg)
    model = build_model(cfg)
    model.count_params()

    replay_buffer = ReplayBuffer(
        buffer_size_per_class=cfg["replay"]["buffer_size_per_class"]
    )
    num_classes = cfg["evaluation"]["num_classes"]

    trainer = ReplayOnlyTrainer(
        model=model,
        replay_buffer=replay_buffer,
        device=device,
        lr=cfg["training"]["lr"],
        use_amp=cfg["training"]["use_amp"],
        num_classes=num_classes,
    )

    n_tasks = len(TASK_CLASSES)
    results_matrix: List[List[Optional[float]]] = [
        [None] * n_tasks for _ in range(n_tasks)
    ]

    epochs = cfg["training"]["epochs_per_task"]
    results_dir = cfg.get("results_dir", "results")

    for task_id in range(n_tasks):
        train_loader, test_loader = splitter.get_task(task_id)
        print(f"\nTask {task_id} 训练集批次: {len(train_loader)}  测试集批次: {len(test_loader)}")

        # 提取 numpy 数组供 ReplayBuffer 使用
        X_parts, y_parts = [], []
        for x, y in train_loader:
            X_parts.append(x.numpy().astype(np.float32))
            y_parts.append(y.numpy().astype(np.int64))
        X_train_np = np.concatenate(X_parts, axis=0)
        y_train_np = np.concatenate(y_parts, axis=0)

        trainer.train_task(
            task_id=task_id,
            train_loader=train_loader,
            epochs=epochs,
            log_every=5,
            val_loader=test_loader,
            X_train_np=X_train_np,
            y_train_np=y_train_np,
        )

        ckpt_path = os.path.join(results_dir, f"replay_only_task{task_id}.pth")
        os.makedirs(results_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        print(f"\n  训练完 Task {task_id} 后，在所有历史任务上评估：")
        for eval_tid in range(task_id + 1):
            _, eval_test_loader = splitter.get_task(eval_tid)
            acc = trainer.evaluate_on_loader(eval_test_loader)
            results_matrix[task_id][eval_tid] = acc
            print(f"    → Task {eval_tid} 准确率: {acc * 100:.2f}%")

    valid_matrix = [
        [v for v in row if v is not None]
        for row in results_matrix
    ]
    bwt = compute_bwt(valid_matrix)
    avg_acc = compute_avg_acc(valid_matrix)

    print("\n" + "=" * 55)
    print("Replay-Only 实验完成")
    print(f"  BWT（后向转移）: {bwt:.4f}{'（遗忘）' if bwt < 0 else '（正向迁移）'}")
    print(f"  最终平均准确率: {avg_acc * 100:.2f}%")
    print("\n结果矩阵（行=训练完 Task i 后，列=在 Task j 上的准确率）：")
    print(format_results_matrix(results_matrix))

    results = {
        "method": "replay_only",
        "buffer_size_per_class": cfg["replay"]["buffer_size_per_class"],
        "results_matrix": results_matrix,
        "bwt": bwt,
        "avg_acc": avg_acc,
    }
    save_path = os.path.join(results_dir, "replay_only_results.json")
    save_results(results, save_path)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="基线对比实验")
    parser.add_argument(
        "--method",
        type=str,
        default="static_task0",
        choices=["static_task0", "ewc_only", "finetune", "replay_only", "all"],
        help=(
            "运行哪个基线方法：\n"
            "  static_task0 — 静态单任务（Task 0）\n"
            "  ewc_only     — 纯 EWC 增量（Task 0→3）\n"
            "  finetune     — Fine-tuning 基线（Task 0→3）\n"
            "  replay_only  — 纯 Replay 增量（Task 0→3）\n"
            "  all          — ewc_only + finetune + replay_only"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 设置全局随机种子
    seed: int = cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子: {seed}")

    if args.method == "static_task0":
        run_static_task0(cfg)

    elif args.method == "ewc_only":
        run_ewc_only(cfg)

    elif args.method == "finetune":
        run_finetune(cfg)

    elif args.method == "replay_only":
        run_replay_only(cfg)

    elif args.method == "all":
        print("\n>>> 运行所有基线方法：ewc_only + finetune + replay_only <<<\n")
        run_ewc_only(cfg)
        run_finetune(cfg)
        run_replay_only(cfg)


if __name__ == "__main__":
    main()
