# Phase 3：EWC 增量学习核心实现

## 目标
实现纯 EWC 方案的完整增量训练序列（Task 0→1→2→3），验证 EWC 参数保护机制有效，相比无保护的 Fine-tuning 遗忘更少。

## 前置条件
- Phase 2 全部验收标准已通过
- FaultClassifier 静态训练 Task 0 准确率已达 85%+

## 任务清单

### 3.1 EWC 核心模块
- [ ] 任务3.1.1：实现 `src/continual/ewc.py`
  - `EWC.__init__(model, lambda_ewc: float = 5000.0)`
  - `EWC.compute_fisher(dataloader, n_samples: int = 500) -> None`
    - 采样 n_samples 条数据，计算每个参数的对角 Fisher：
      `F_i = E[(∂log p(y|x)/∂θ_i)²]`
    - 保存 `self.fisher_dict[name]` 和 `self.optimal_params[name]`（均为 Tensor）
  - `EWC.penalty(model) -> Tensor`
    - 返回 `(λ/2) * Σ_i F_i * (θ_i - θ*_i)²`（标量）
  - `EWC.update_task() -> None`
    - 累积合并新旧 Fisher：`F_total = F_old + F_new`（不替换，防止旧任务保护退化）

### 3.2 纯 EWC 训练器
- [ ] 任务3.2.1：实现 `src/baselines/ewc_only_trainer.py`
  - `EWCOnlyTrainer.train_task(task_id, train_loader, epochs=50) -> Dict`
    - Loss = `CrossEntropy(新数据)` + `EWC.penalty(model)`
    - 支持 AMP
    - 训练结束后调用 `EWC.compute_fisher()` 和 `EWC.update_task()`
  - `EWCOnlyTrainer.evaluate_on_task(task_id) -> float`

### 3.3 Fine-tuning 基线（对照组）
- [ ] 任务3.3.1：在 `src/baselines/static_trainer.py` 中添加 `FineTuningTrainer`
  - 与 EWCOnlyTrainer 相同，但 Loss 中不包含 EWC.penalty
  - 用于与 EWC 对比，展示灾难性遗忘程度

### 3.4 增量评估指标
- [ ] 任务3.4.1：实现 `src/evaluation/metrics.py`
  - `compute_metrics(y_true, y_pred) -> Dict`：ACC / FAR / FDR
  - `compute_bwt(R: List[List[float]]) -> float`
    - `R[i][j]` = 训练完 Task i 后在 Task j 上的准确率
    - `BWT = (1/(T-1)) * Σ_{j=0}^{T-2} (R[T-1][j] - R[j][j])`
  - `compute_avg_acc(R) -> float`：最终平均准确率

### 3.5 实验脚本
- [ ] 任务3.5.1：实现 `experiments/run_baselines.py` 中 `ewc_only` 和 `finetune` 分支
  - 完整跑 Task 0→3 增量序列
  - 每个 task 训练完后在所有历史任务上评估，记录 `results_matrix`
  - 保存到 `results/ewc_only_results.json` 和 `results/finetune_results.json`

## 验收标准
- [ ] `pytest tests/ -v` 全部通过（含 EWC 模块单元测试）
- [ ] `results/ewc_only_results.json` 存在，包含完整 4×4 results_matrix
- [ ] EWC-only 在 Task 3 训练后，Task 0 准确率下降 < Fine-tuning 的 50%（BWT 绝对值更小）
- [ ] EWC.penalty() 在 Task 0 后返回值为 0（无历史任务时无惩罚）；Task 1 后返回正数

## 注意事项
- compute_fisher 时模型必须处于 `model.eval()` 状态，并对 loss 执行 backward，
  但不要调用 optimizer.step()
- Fisher 累积时对同名参数做加法，不能直接赋值替换（否则 Task 1 的保护会覆盖 Task 0）
- AMP 下 compute_fisher 建议关闭 autocast，避免 float16 精度不足导致 Fisher 估计偏差
- lambda_ewc 默认 5000，可通过 config.yaml 调整；调参时观察 loss_ewc 与 loss_ce 量级是否相当
