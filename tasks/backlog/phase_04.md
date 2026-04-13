# Phase 4：Experience Replay + 混合训练器（核心方案）

## 目标
完成 EWC+Replay 混合方案的完整增量训练流程（Task 0→3），Task 3 训练完成后 22 类平均故障检测准确率 ≥ 90%，误报率 ≤ 3%。

## 前置条件
- Phase 3 全部验收标准已通过
- EWC 模块单元测试全绿

## 任务清单

### 4.1 记忆库模块
- [ ] 任务4.1.1：实现 `src/continual/replay_buffer.py`
  - `ReplayBuffer.__init__(buffer_size_per_class: int = 50)`
  - `ReplayBuffer.add_task_samples(X_windows: np.ndarray, y_windows: np.ndarray) -> None`
    - 对每个类别：蓄水池采样，保留 ≤ 50 条窗口样本（numpy 存储节省显存）
  - `ReplayBuffer.sample_replay_batch(batch_size: int = 64) -> Tuple[Tensor, Tensor]`
    - 从所有历史类中均匀采样，返回 GPU Tensor
  - `ReplayBuffer.get_stats() -> Dict[int, int]`：各类当前 buffer 大小
  - `ReplayBuffer.__len__() -> int`：总样本数

### 4.2 纯 Replay 基线
- [ ] 任务4.2.1：实现 `src/baselines/replay_only_trainer.py`
  - Loss = `CrossEntropy(混合batch)`（无 EWC 惩罚项）
  - 混合 batch：256 条新数据 + 64 条 replay 样本
  - 用于对比纯 Replay vs EWC+Replay

### 4.3 混合增量训练器（核心）
- [ ] 任务4.3.1：实现 `src/continual/trainer.py`
  - `ContinualTrainer.__init__(model, ewc, replay_buffer, config, device)`
  - `ContinualTrainer.train_task(task_id, train_loader, epochs=50) -> Dict`
    - 每个 mini-batch 流程：
      1. 从 train_loader 取 256 条新数据
      2. 若 task_id > 0：从 ReplayBuffer 采样 64 条旧数据
      3. 拼接 → 320 条混合 batch
      4. Loss = `CE(混合batch)` + `EWC.penalty(model)`
      5. AMP 反向传播 + optimizer.step()
    - 训练完后：`EWC.compute_fisher()` → `EWC.update_task()` → `ReplayBuffer.add_task_samples()`
    - 返回：`{"task_id": int, "loss_history": [...], "final_loss_ce": float, "final_loss_ewc": float}`
  - `ContinualTrainer.evaluate_all_tasks() -> Dict[int, Dict]`
    - 遍历所有已见任务测试集，返回 `{task_id: {"acc": ..., "far": ..., "fdr": ...}}`
  - `ContinualTrainer.save_checkpoint(path: str) -> None`
    - 保存 model state_dict + ewc fisher_dict + replay_buffer
  - `ContinualTrainer.load_checkpoint(path: str) -> None`

### 4.4 主实验脚本
- [ ] 任务4.4.1：实现 `experiments/run_proposed.py`
  - 完整增量序列 Task 0→1→2→3
  - 每个 task 训练完：打印当前 task 的 loss 分解 + 所有历史任务准确率
  - 记录 4×4 results_matrix
  - 保存 `results/proposed_final.pth` 和 `results/proposed_results.json`

## 验收标准
- [ ] 全部 4 个增量任务训练完成，无 OOM 或 CUDA 错误
- [ ] `results/proposed_results.json` 存在，包含完整 results_matrix
- [ ] Task 3 训练后，22 类平均测试准确率 ≥ **90%**（核心指标）
- [ ] Task 3 训练后，误报率（正常类预测为故障）≤ **3%**
- [ ] Checkpoint 保存后重新加载，evaluate_all_tasks() 输出与保存前一致
- [ ] 打印的 loss_ewc 和 loss_ce 量级相当（若 loss_ewc >> loss_ce，需降低 lambda）

## 注意事项
- 混合 batch 中新旧数据的 label 必须是**全局标签**（0~21），非任务内局部标签
- ReplayBuffer 以 numpy 存储，sample_replay_batch 时才转为 CUDA Tensor
- Task 0 训练时 ReplayBuffer 为空，此时 Loss = CE(新数据) + 0，行为等同于普通训练
- AMP scaler 需跨 epoch 持久化，不要在 epoch 内重新创建
- 建议每个 task 结束后打印 ReplayBuffer.get_stats()，确认各类样本数 ≤ 50
