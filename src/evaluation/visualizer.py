"""
实验可视化模块

提供以下绘图功能（输出论文级质量图表）：
  - plot_forgetting_curve:       4种方法遗忘曲线对比
  - plot_accuracy_heatmap:       单方法准确率热力图
  - plot_robustness_comparison:  噪声鲁棒性对比折线图
  - plot_confusion_matrix:       22类混淆矩阵
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互后端，适用于无显示器环境
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


# ─────────────────────────────────────────────────────────────────────────────
# 全局样式配置
# ─────────────────────────────────────────────────────────────────────────────

# 方法 → (显示名, 颜色, 标记)
METHOD_STYLE = {
    "finetune":    ("Fine-tuning",    "#D62728", "o"),   # 红
    "ewc_only":    ("EWC-Only",       "#FF7F0E", "s"),   # 橙
    "replay_only": ("Replay-Only",    "#2CA02C", "^"),   # 绿
    "proposed":    ("EWC+Replay",     "#1F77B4", "D"),   # 蓝
}

# 22 类标签：N（正常） + F01~F21
CLASS_LABELS = ["N"] + [f"F{i:02d}" for i in range(1, 22)]


def _setup_chinese_font() -> None:
    """尝试设置中文字体（Windows: 微软雅黑，macOS: PingFang，Linux: SimHei）。"""
    if not _HAS_MPL:
        return
    import platform
    system = platform.system()
    if system == "Windows":
        font_candidates = ["Microsoft YaHei", "SimHei"]
    elif system == "Darwin":
        font_candidates = ["PingFang SC", "Heiti SC"]
    else:
        font_candidates = ["SimHei", "WenQuanYi Micro Hei"]

    for font in font_candidates:
        try:
            plt.rcParams["font.family"] = [font]
            # 测试是否可用
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.set_title("测试")
            plt.close(fig)
            break
        except Exception:
            continue

    # 解决负号显示问题
    plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir(save_path: str) -> None:
    """确保保存路径的父目录存在。"""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 遗忘曲线
# ─────────────────────────────────────────────────────────────────────────────

def plot_forgetting_curve(
    all_results: Dict[str, dict],
    save_path: str = "results/figures/forgetting_curve.png",
    dpi: int = 300,
) -> None:
    """绘制遗忘曲线（各方法历史任务平均准确率随增量任务的变化）。

    横轴：任务序号 0→3（训练完 Task i 后）
    纵轴：历史任务平均准确率（%）
    每种方法一条折线。

    Args:
        all_results: {method_name: results_dict}，每个 results_dict 须含 "results_matrix"
        save_path:   输出图片路径
        dpi:         输出分辨率
    """
    if not _HAS_MPL:
        raise ImportError("plot_forgetting_curve 需要 matplotlib")

    _setup_chinese_font()
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method_key, style_info in METHOD_STYLE.items():
        if method_key not in all_results:
            continue

        label, color, marker = style_info
        rm = all_results[method_key].get("results_matrix")
        if rm is None:
            continue

        n_tasks = len(rm)
        # 计算每轮训练后历史任务的平均准确率
        avg_accs = []
        for i in range(n_tasks):
            # 取 rm[i][0..i] 的均值（训练完 Task i 后，在 Task 0~i 上的平均准确率）
            row_valid = [
                rm[i][j] for j in range(i + 1)
                if rm[i][j] is not None
            ]
            avg = np.mean(row_valid) * 100 if row_valid else 0.0
            avg_accs.append(avg)

        x = list(range(n_tasks))
        ax.plot(
            x, avg_accs,
            color=color, marker=marker, markersize=8,
            linewidth=2, label=label,
        )

    ax.set_xlabel("增量任务（训练完 Task i 后）", fontsize=12)
    ax.set_ylabel("历史任务平均准确率 (%)", fontsize=12)
    ax.set_title("遗忘曲线对比", fontsize=14, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"Task {i}" for i in range(4)])
    ax.set_ylim(0, 105)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"遗忘曲线已保存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 准确率热力图
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_heatmap(
    results_matrix: List[List[Optional[float]]],
    method_name: str = "EWC+Replay",
    save_path: str = "results/figures/heatmap_proposed.png",
    dpi: int = 300,
) -> None:
    """绘制准确率热力图。

    行 = 训练完 Task i 后，列 = 在 Task j 上的测试准确率。
    上三角（未见任务）显示为灰色。

    Args:
        results_matrix: T×T 列表，results_matrix[i][j] 为准确率（0~1）或 None
        method_name:    方法名称（用于标题）
        save_path:      输出图片路径
        dpi:            输出分辨率
    """
    if not _HAS_MPL or not _HAS_SNS:
        raise ImportError("plot_accuracy_heatmap 需要 matplotlib 和 seaborn")

    _setup_chinese_font()
    _ensure_dir(save_path)

    n_tasks = len(results_matrix)

    # 构建数据矩阵（百分比），None → NaN
    data = np.full((n_tasks, n_tasks), np.nan)
    annot = np.full((n_tasks, n_tasks), "", dtype=object)

    for i in range(n_tasks):
        for j in range(n_tasks):
            v = results_matrix[i][j]
            if v is not None:
                data[i][j] = v * 100
                annot[i][j] = f"{v * 100:.1f}%"

    # 创建 mask（上三角部分 = NaN 的位置）
    mask = np.isnan(data)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.heatmap(
        data,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        vmin=0, vmax=100,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "准确率 (%)"},
        ax=ax,
        square=True,
    )

    ax.set_xlabel("测试任务", fontsize=12)
    ax.set_ylabel("训练阶段（完成 Task i 后）", fontsize=12)
    ax.set_title(f"{method_name} — 准确率矩阵", fontsize=14, fontweight="bold")
    ax.set_xticklabels([f"Task {j}" for j in range(n_tasks)], rotation=0)
    ax.set_yticklabels([f"After T{i}" for i in range(n_tasks)], rotation=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"准确率热力图已保存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 鲁棒性对比
# ─────────────────────────────────────────────────────────────────────────────

def plot_robustness_comparison(
    robustness_results: Dict[str, Dict[str, dict]],
    save_path: str = "results/figures/robustness.png",
    dpi: int = 300,
) -> None:
    """绘制噪声鲁棒性对比折线图。

    横轴：高斯噪声标准差 σ
    纵轴：准确率（%）
    每种方法一条折线。

    Args:
        robustness_results: {method_name: {"noise_std": {acc, far, fdr, ...}}}
        save_path:          输出图片路径
        dpi:                输出分辨率
    """
    if not _HAS_MPL:
        raise ImportError("plot_robustness_comparison 需要 matplotlib")

    _setup_chinese_font()
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method_key, style_info in METHOD_STYLE.items():
        if method_key not in robustness_results:
            continue

        label, color, marker = style_info
        method_data = robustness_results[method_key]

        # 按噪声标准差排序
        noise_stds = sorted(method_data.keys(), key=lambda k: float(k))
        x_vals = [float(ns) for ns in noise_stds]
        y_vals = [method_data[ns]["acc"] * 100 for ns in noise_stds]

        ax.plot(
            x_vals, y_vals,
            color=color, marker=marker, markersize=8,
            linewidth=2, label=label,
        )

    ax.set_xlabel("高斯噪声标准差 (σ)", fontsize=12)
    ax.set_ylabel("准确率 (%)", fontsize=12)
    ax.set_title("噪声鲁棒性对比", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"鲁棒性对比图已保存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. 混淆矩阵
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    save_path: str = "results/figures/confusion_matrix.png",
    dpi: int = 300,
    normalize: bool = True,
) -> None:
    """绘制 22 类混淆矩阵。

    Args:
        y_true:    真实标签数组
        y_pred:    预测标签数组
        save_path: 输出图片路径
        dpi:       输出分辨率
        normalize: 是否按行归一化（显示百分比）
    """
    if not _HAS_MPL or not _HAS_SNS:
        raise ImportError("plot_confusion_matrix 需要 matplotlib 和 seaborn")

    _setup_chinese_font()
    _ensure_dir(save_path)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    n_classes = max(y_true.max(), y_pred.max()) + 1
    n_classes = max(n_classes, 22)  # 至少 22 类

    # 构建混淆矩阵
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # 归一化（按行 = 每类的预测分布）
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # 避免除零
        cm_norm = cm.astype(np.float64) / row_sums * 100
        fmt_str = ".1f"
        cbar_label = "百分比 (%)"
    else:
        cm_norm = cm.astype(np.float64)
        fmt_str = ".0f"
        cbar_label = "样本数"

    # 截取前 22 类
    cm_display = cm_norm[:22, :22]
    labels = CLASS_LABELS[:22]

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt_str,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.3,
        linecolor="gray",
        cbar_kws={"label": cbar_label},
        ax=ax,
        square=True,
        annot_kws={"size": 7},
    )

    ax.set_xlabel("预测类别", fontsize=12)
    ax.set_ylabel("真实类别", fontsize=12)
    ax.set_title("EWC+Replay 方法 — 22类混淆矩阵", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"混淆矩阵已保存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数：从 JSON 文件批量绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_from_json(
    results_dir: str = "results",
    figures_dir: str = "results/figures",
) -> None:
    """从 results/ 下的 JSON 文件批量生成所有图表。

    自动加载以下文件（存在则绘制，不存在则跳过）：
      - finetune_results.json
      - ewc_only_results.json
      - replay_only_results.json
      - proposed_results.json
      - robustness_results.json

    Args:
        results_dir: JSON 结果文件所在目录
        figures_dir: 输出图片目录
    """
    import json

    os.makedirs(figures_dir, exist_ok=True)

    # 加载各方法结果
    method_files = {
        "finetune":    "finetune_results.json",
        "ewc_only":    "ewc_only_results.json",
        "replay_only": "replay_only_results.json",
        "proposed":    "proposed_results.json",
    }

    all_results: Dict[str, dict] = {}
    for method_key, filename in method_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                all_results[method_key] = json.load(f)
            print(f"  [OK] 已加载 {filename}")
        else:
            print(f"  [SKIP] {filename} 不存在")

    # 1. 遗忘曲线
    if all_results:
        plot_forgetting_curve(
            all_results,
            save_path=os.path.join(figures_dir, "forgetting_curve.png"),
        )

    # 2. proposed 方法热力图
    if "proposed" in all_results:
        rm = all_results["proposed"].get("results_matrix")
        if rm is not None:
            plot_accuracy_heatmap(
                rm,
                method_name="EWC+Replay",
                save_path=os.path.join(figures_dir, "heatmap_proposed.png"),
            )

    # 3. 鲁棒性对比
    robustness_path = os.path.join(results_dir, "robustness_results.json")
    if os.path.exists(robustness_path):
        with open(robustness_path, "r", encoding="utf-8") as f:
            robustness_data = json.load(f)
        plot_robustness_comparison(
            robustness_data,
            save_path=os.path.join(figures_dir, "robustness.png"),
        )
    else:
        print("  [SKIP] robustness_results.json 不存在，跳过鲁棒性图")

    # 混淆矩阵需要模型预测结果（y_true, y_pred），无法从 JSON 直接生成
    print("\n注意：混淆矩阵需在服务器上运行模型推理生成 y_true/y_pred 后调用 plot_confusion_matrix()")

    print(f"\n全部图表已输出至 {figures_dir}/")
