"""
增量学习评估指标模块

实现以下指标：
  - ACC：整体准确率
  - FAR：误报率（False Alarm Rate），正常工况被误判为故障
  - FDR：故障检测率（Fault Detection Rate），故障被正确识别
  - BWT：后向转移（Backward Transfer），衡量灾难性遗忘程度
  - Avg-ACC：最终平均准确率（所有历史任务的均值）
  - per_class_acc：各类别准确率列表
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ─────────────────────────────────────────────────────────────────────────────
# 样本级指标
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: "np.ndarray | List[int]",
    y_pred: "np.ndarray | List[int]",
    normal_class: int = 0,
) -> Dict:
    """计算分类评估指标。

    Args:
        y_true:       真实标签数组，形状 [N,]
        y_pred:       预测标签数组，形状 [N,]
        normal_class: 正常工况对应的类别标签，默认为 0

    Returns:
        dict，包含以下键：
            acc          (float): 整体准确率
            far          (float): 误报率 = 正常样本被误判为故障 / 正常样本总数
            fdr          (float): 故障检测率 = 故障样本被正确检测 / 故障样本总数
            per_class_acc (list): 各类别准确率，顺序与 np.unique(y_true) 一致
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    assert y_true.shape == y_pred.shape, "y_true 与 y_pred 形状不一致"

    n = len(y_true)
    if n == 0:
        return {"acc": 0.0, "far": 0.0, "fdr": 0.0, "per_class_acc": []}

    # 整体准确率
    acc = float((y_true == y_pred).sum()) / n

    # FAR：正常样本中被误报（预测为非正常）的比例
    normal_mask = y_true == normal_class
    n_normal = normal_mask.sum()
    if n_normal > 0:
        far = float((y_pred[normal_mask] != normal_class).sum()) / n_normal
    else:
        far = 0.0

    # FDR：故障样本中被正确检测（预测为非正常）的比例
    fault_mask = y_true != normal_class
    n_fault = fault_mask.sum()
    if n_fault > 0:
        fdr = float((y_pred[fault_mask] != normal_class).sum()) / n_fault
    else:
        fdr = 0.0

    # 各类别准确率
    classes = np.unique(y_true)
    per_class_acc: List[float] = []
    for cls in classes:
        mask = y_true == cls
        cls_acc = float((y_pred[mask] == cls).sum()) / mask.sum()
        per_class_acc.append(cls_acc)

    return {
        "acc": acc,
        "far": far,
        "fdr": fdr,
        "per_class_acc": per_class_acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 增量学习专用指标
# ─────────────────────────────────────────────────────────────────────────────

def compute_bwt(R: List[List[float]]) -> float:
    """计算后向转移（Backward Transfer，BWT）。

    BWT < 0 表示遗忘（学习新任务后，旧任务性能下降）；
    BWT > 0 表示正向迁移。

    公式：
        BWT = (1 / (T-1)) * Σ_{j=0}^{T-2} (R[T-1][j] - R[j][j])

    其中 R[i][j] 为训练完 Task i 后，在 Task j 上的准确率。

    Args:
        R: T×T 下三角矩阵，R[i][j]（j ≤ i）= 训练完 Task i 后 Task j 的准确率。
           R[i][j] 对 j > i 可为 None 或 0（未见过的任务）。

    Returns:
        bwt: 后向转移标量；若任务数 T ≤ 1 则返回 0.0
    """
    T = len(R)
    if T <= 1:
        return 0.0

    bwt_sum = 0.0
    for j in range(T - 1):
        r_final = R[T - 1][j]  # 训练完所有任务后 Task j 的准确率
        r_diag = R[j][j]       # 刚训练完 Task j 时的准确率（最优时刻）
        bwt_sum += (r_final - r_diag)

    return bwt_sum / (T - 1)


def compute_avg_acc(R: List[List[float]]) -> float:
    """计算最终平均准确率（Average Accuracy）。

    取 R 最后一行（训练完所有任务后）各历史任务准确率的均值。

    Args:
        R: T×T 结果矩阵，同 compute_bwt 中定义

    Returns:
        avg_acc: 最终平均准确率；若 R 为空则返回 0.0
    """
    T = len(R)
    if T == 0:
        return 0.0

    last_row = R[T - 1]
    valid = [v for v in last_row if v is not None]
    if not valid:
        return 0.0
    return float(np.mean(valid))


def compute_fwt(R: List[List[float]]) -> float:
    """计算前向转移（Forward Transfer，FWT）。

    衡量已有知识对未来任务的正向迁移效果。
    此处使用零初始基线（random-init 模型在各任务的准确率设为 0.0）。

    公式：
        FWT = (1 / (T-1)) * Σ_{i=1}^{T-1} (R[i-1][i] - b[i])
        b[i] = 0（简化，无需额外 baseline 实验）

    Args:
        R: T×T 结果矩阵

    Returns:
        fwt: 前向转移标量；若任务数 T ≤ 1 则返回 0.0
    """
    T = len(R)
    if T <= 1:
        return 0.0

    fwt_sum = 0.0
    for i in range(1, T):
        # R[i-1][i] 为学完 Task i-1 后在 Task i（未见）上的准确率
        # 若矩阵中没有此值（上三角），视为 0
        r_transfer = R[i - 1][i] if len(R[i - 1]) > i else 0.0
        if r_transfer is None:
            r_transfer = 0.0
        fwt_sum += r_transfer  # baseline b[i] = 0

    return fwt_sum / (T - 1)


def format_results_matrix(R: List[List[Optional[float]]], precision: int = 4) -> str:
    """将结果矩阵格式化为可读字符串（用于日志输出）。

    Args:
        R:         T×T 结果矩阵
        precision: 小数位数

    Returns:
        多行字符串，每行对应训练完 Task i 后在各任务上的准确率
    """
    T = len(R)
    lines = []
    header = "       " + "  ".join(f"T{j}" for j in range(T))
    lines.append(header)
    for i, row in enumerate(R):
        vals = []
        for j in range(T):
            v = row[j] if j < len(row) else None
            if v is None:
                vals.append("  — ")
            else:
                vals.append(f"{v:.{precision}f}")
        lines.append(f"[T{i}]  " + "  ".join(vals))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 多方法汇总
# ─────────────────────────────────────────────────────────────────────────────

def summarize_results(
    all_results: Dict[str, dict],
) -> "pd.DataFrame":
    """汇总多个方法的实验指标为 DataFrame，用于论文表格。

    Args:
        all_results: {method_name: results_dict} 字典。
            每个 results_dict 应包含（可选字段缺失时填 NaN）：
                - results_matrix: T×T 列表（用于计算 Avg-ACC / BWT / FWT）
                - avg_acc: float（若已有则直接使用，否则从 results_matrix 计算）
                - bwt: float
                - fwt: float
                - overall_metrics: dict，含 acc / far / fdr

    Returns:
        pd.DataFrame，列为：Method / Avg-ACC(%) / FAR(%) / FDR(%) / BWT / FWT
        行按方法名排列。

    Raises:
        ImportError: 若 pandas 未安装
    """
    if not _HAS_PANDAS:
        raise ImportError("summarize_results 需要 pandas，请运行 pip install pandas")

    rows = []
    for method_name, res in all_results.items():
        # ── 从 results_matrix 计算增量指标（优先使用已有值）──────────────────
        rm = res.get("results_matrix")
        if rm is not None:
            # 将 None 替换为 0.0 用于计算
            valid = [
                [v if v is not None else 0.0 for v in row]
                for row in rm
            ]
        else:
            valid = None

        avg_acc = res.get("avg_acc")
        if avg_acc is None and valid is not None:
            avg_acc = compute_avg_acc(valid)

        bwt = res.get("bwt")
        if bwt is None and valid is not None:
            bwt = compute_bwt(valid)

        fwt = res.get("fwt")
        if fwt is None and valid is not None:
            fwt = compute_fwt(valid)

        # ── 提取 overall_metrics ──────────────────────────────────────────────
        om = res.get("overall_metrics", {})
        far = om.get("far")
        fdr = om.get("fdr")

        rows.append({
            "Method": method_name,
            "Avg-ACC(%)": round(avg_acc * 100, 2) if avg_acc is not None else float("nan"),
            "FAR(%)": round(far * 100, 2) if far is not None else float("nan"),
            "FDR(%)": round(fdr * 100, 2) if fdr is not None else float("nan"),
            "BWT": round(bwt, 4) if bwt is not None else float("nan"),
            "FWT": round(fwt, 4) if fwt is not None else float("nan"),
        })

    df = pd.DataFrame(rows)
    df = df.set_index("Method")
    return df
