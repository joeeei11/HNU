"""
主实验：EWC + Experience Replay 混合增量学习（核心方案）

用法：
    python experiments/run_proposed.py --config config/config.yaml
    python experiments/run_proposed.py --config config/config.yaml --resume results/proposed_task1.pth

运行流程：
  Task 0 → Task 1 → Task 2 → Task 3
  每个 task 训练完后：
    1. 打印当前 task 的 loss 分解（CE + EWC）
    2. 在所有历史任务测试集上评估（填充 results_matrix）
    3. 打印 ReplayBuffer 统计
    4. 保存当前阶段 checkpoint
  全部完成后：
    - 保存最终权重 results/proposed_final.pth
    - 保存完整指标 results/proposed_results.json
    - 打印 BWT / Avg-ACC / FWT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.task_splitter import TaskSplitter, TASK_CLASSES
from src.models.fault_classifier import FaultClassifier
from src.continual.ewc import EWC
from src.continual.replay_buffer import ReplayBuffer
from src.continual.trainer import ContinualTrainer
from src.evaluation.metrics import (
    compute_metrics,
    compute_bwt,
    compute_avg_acc,
    compute_fwt,
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


def get_numpy_arrays_from_loader(loader) -> tuple:
    """从 DataLoader 中提取全部 numpy 样本（用于 ReplayBuffer）。

    Returns:
        (X_np: ndarray [N, W, F], y_np: ndarray [N,])
    """
    X_parts = []
    y_parts = []
    for x, y in loader:
        X_parts.append(x.numpy().astype(np.float32))
        y_parts.append(y.numpy().astype(np.int64))
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def save_results(results: dict, path: str) -> None:
    """将结果字典保存为 JSON 文件。"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至 {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 主实验函数
# ─────────────────────────────────────────────────────────────────────────────

def run_proposed(cfg: dict, resume_path: Optional[str] = None) -> dict:
    """运行 EWC+Replay 混合增量实验（Task 0→1→2→3）。

    Args:
        cfg:         来自 config.yaml 的配置字典
        resume_path: 已有 checkpoint 路径（None 则从头开始）

    Returns:
        results: 包含 results_matrix / BWT / Avg-ACC / FWT 等指标
    """
    print("=" * 65)
    print("主实验：EWC + Experience Replay 混合增量学习（核心方案）")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir: str = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── 初始化组件 ────────────────────────────────────────────────────────────
    splitter = build_splitter(cfg)
    model = build_model(cfg)
    model.count_params()

    ewc = EWC(model=model, lambda_ewc=cfg["ewc"]["lambda"])
    replay_buffer = ReplayBuffer(
        buffer_size_per_class=cfg["replay"]["buffer_size_per_class"]
    )

    # trainer config 合并所有相关子配置
    trainer_config = {
        "training": cfg["training"],
        "ewc": cfg["ewc"],
        "replay": cfg.get("replay", {}),
        "evaluation": cfg.get("evaluation", {}),
    }
    trainer = ContinualTrainer(
        model=model,
        ewc=ewc,
        replay_buffer=replay_buffer,
        config=trainer_config,
        device=device,
    )

    # ── 从 checkpoint 恢复（可选）────────────────────────────────────────────
    start_task = 0
    if resume_path is not None and os.path.exists(resume_path):
        trainer.load_checkpoint(resume_path)
        start_task = trainer._completed_tasks
        print(f"\n从 Task {start_task} 继续训练...")

    n_tasks = len(TASK_CLASSES)
    epochs = cfg["training"]["epochs_per_task"]

    # results_matrix[i][j] = 训练完 Task i 后在 Task j 上的准确率
    results_matrix: List[List[Optional[float]]] = [
        [None] * n_tasks for _ in range(n_tasks)
    ]

    # ── 增量训练主循环 Task 0 → 3 ────────────────────────────────────────────
    for task_id in range(start_task, n_tasks):
        train_loader, test_loader = splitter.get_task(task_id)
        print(f"\nTask {task_id} 训练集批次: {len(train_loader)}  测试集批次: {len(test_loader)}")

        # 提取训练数据的 numpy 数组，用于 ReplayBuffer 更新
        print(f"  提取 Task {task_id} 训练数据 → numpy（用于更新 ReplayBuffer）...")
        X_train_np, y_train_np = get_numpy_arrays_from_loader(train_loader)
        print(f"  样本数: {len(X_train_np)}  shape: {X_train_np.shape}")

        # 训练当前任务
        task_result = trainer.train_task(
            task_id=task_id,
            train_loader=train_loader,
            epochs=epochs,
            log_every=5,
            val_loader=test_loader,
            X_train_np=X_train_np,
            y_train_np=y_train_np,
        )

        # ── 在所有历史任务测试集上评估 ────────────────────────────────────────
        print(f"\n  训练完 Task {task_id} 后，评估所有历史任务：")
        for eval_tid in range(task_id + 1):
            _, eval_test_loader = splitter.get_task(eval_tid)
            eval_results = trainer.evaluate_all_tasks([eval_test_loader])
            acc = eval_results[0]["acc"]
            far = eval_results[0]["far"]
            fdr = eval_results[0]["fdr"]
            results_matrix[task_id][eval_tid] = acc
            print(
                f"    → Task {eval_tid}:  acc={acc * 100:.2f}%  "
                f"far={far * 100:.2f}%  fdr={fdr * 100:.2f}%"
            )

        # ── 保存阶段性 checkpoint ────────────────────────────────────────────
        ckpt_path = os.path.join(results_dir, f"proposed_task{task_id}.pth")
        trainer.save_checkpoint(ckpt_path)

        # 打印 loss 分解
        print(
            f"\n  Task {task_id} 最终 loss — "
            f"CE: {task_result['final_loss_ce']:.4f}  "
            f"EWC: {task_result['final_loss_ewc']:.4f}"
        )

    # ── 最终 checkpoint（完整训练状态）──────────────────────────────────────
    final_ckpt = os.path.join(results_dir, "proposed_final.pth")
    trainer.save_checkpoint(final_ckpt)

    # ── 计算增量学习汇总指标 ──────────────────────────────────────────────────
    # 构建用于计算 BWT/FWT 的有效矩阵（去掉 None）
    valid_matrix = []
    for row in results_matrix:
        valid_row = []
        for v in row:
            valid_row.append(v if v is not None else 0.0)
        valid_matrix.append(valid_row)

    bwt = compute_bwt(valid_matrix)
    avg_acc = compute_avg_acc(valid_matrix)
    fwt = compute_fwt(valid_matrix)

    # 最终任务的详细指标（22类，全测试集）
    print("\n  计算最终全集指标（Task 3 训练完后，在所有任务上评估）...")
    all_test_loaders = [splitter.get_task(tid)[1] for tid in range(n_tasks)]
    final_task_metrics = trainer.evaluate_all_tasks(all_test_loaders)

    # 总体 acc/far/fdr（合并所有任务测试集的预测）
    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    model.eval()
    for loader in all_test_loaders:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, dtype=torch.float32)
                with torch.amp.autocast("cuda", enabled=trainer.use_amp):
                    logits = model(x)
                pred = logits.argmax(dim=1).cpu().numpy()
                all_y_true.extend(y.numpy().tolist())
                all_y_pred.extend(pred.tolist())

    overall_metrics = compute_metrics(
        np.array(all_y_true),
        np.array(all_y_pred),
        normal_class=0,
    )

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("主实验完成 — EWC + Replay 混合增量学习")
    print(f"  BWT（后向转移）:   {bwt:.4f}{'（遗忘）' if bwt < 0 else '（正向迁移）'}")
    print(f"  Avg-ACC（最终均值）: {avg_acc * 100:.2f}%")
    print(f"  FWT（前向转移）:   {fwt:.4f}")
    print(f"  总体准确率（全集）: {overall_metrics['acc'] * 100:.2f}%")
    print(f"  总体误报率（FAR）:  {overall_metrics['far'] * 100:.2f}%")
    print(f"  总体检测率（FDR）:  {overall_metrics['fdr'] * 100:.2f}%")

    print("\n结果矩阵（行=训练完 Task i 后，列=在 Task j 上的准确率）：")
    print(format_results_matrix(results_matrix))

    # 验收检查
    _check_acceptance(avg_acc, overall_metrics["far"])

    # ── 保存 JSON 结果 ────────────────────────────────────────────────────────
    results = {
        "method": "ewc_replay_hybrid",
        "lambda_ewc": cfg["ewc"]["lambda"],
        "buffer_size_per_class": cfg["replay"]["buffer_size_per_class"],
        "results_matrix": results_matrix,
        "bwt": bwt,
        "avg_acc": avg_acc,
        "fwt": fwt,
        "overall_metrics": overall_metrics,
        "per_task_metrics": {
            str(k): v for k, v in final_task_metrics.items()
        },
    }

    json_path = os.path.join(results_dir, "proposed_results.json")
    save_results(results, json_path)

    return results


def _check_acceptance(avg_acc: float, far: float) -> None:
    """验收标准检查（Phase 4）"""
    print("\n验收检查：")
    target_acc = 0.90
    target_far = 0.03

    acc_ok = avg_acc >= target_acc
    far_ok = far <= target_far

    print(
        f"  22类平均准确率 ≥ 90%：{'✅' if acc_ok else '❌'}  "
        f"（实际 {avg_acc * 100:.2f}%）"
    )
    print(
        f"  误报率 ≤ 3%：{'✅' if far_ok else '❌'}  "
        f"（实际 {far * 100:.2f}%）"
    )

    if acc_ok and far_ok:
        print("\n✅ Phase 4 验收通过！")
    else:
        print("\n⚠️  未完全达标，请调整超参数后重试。")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EWC + Replay 混合增量学习主实验"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径（默认 config/config.yaml）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从指定 checkpoint 恢复训练（例如 results/proposed_task1.pth）",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 设置全局随机种子
    seed: int = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子: {seed}")

    run_proposed(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
