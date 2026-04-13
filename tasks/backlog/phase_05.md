# Phase 5：基线方法整合与对比实验框架

## 目标
4 种方法（静态模型 / Fine-tuning / 纯 EWC / 纯 Replay / EWC+Replay）可在同一套框架下一键运行和比较，结果完整保存且可重现。

## 前置条件
- Phase 4 全部验收标准已通过
- proposed 方案 Task 3 平均准确率已达 90%+

## 任务清单

### 5.1 统一实验入口
- [ ] 任务5.1.1：重构 `experiments/run_baselines.py`
  - 支持命令行参数：`--method [static|finetune|ewc_only|replay_only|proposed|all]`
  - `--method all`：依次运行全部 5 种方法
  - 所有方法在脚本开头统一固定随机种子：
    ```python
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    ```
  - 所有方法共享同一份预处理后的 DataLoader（避免随机性差异）
  - 结果统一保存到 `results/<method>_results.json`

### 5.2 完善评估指标
- [ ] 任务5.2.1：补全 `src/evaluation/metrics.py`
  - `compute_far(y_true, y_pred, normal_class=0) -> float`：正常样本误报率
  - `compute_fdr(y_true, y_pred, normal_class=0) -> float`：故障检测率
  - `compute_bwt(R) -> float`：Backward Transfer
  - `compute_avg_acc(R) -> float`：最终平均准确率（最后一行均值）
  - `summarize_results(results_dict: Dict) -> pd.DataFrame`
    - 输出列：方法名 / 平均ACC / FAR / FDR / BWT / 训练总时长 / 参数量
    - 可直接用于论文表格

### 5.3 鲁棒性测试
- [ ] 任务5.3.1：实现 `experiments/run_robustness.py`
  - 命令行参数：`--noise_std 0.1 0.3 0.5`（可传多个值）
  - 在标准化后的**测试集特征**上叠加高斯噪声
  - 对全部 5 种方法（加载已保存的 checkpoint）测试噪声下的准确率
  - 保存结果到 `results/robustness_results.json`

### 5.4 计算复杂度记录
- [ ] 任务5.4.1：在各 Trainer 中添加耗时统计
  - `time_per_task`：每个 task 训练秒数
  - `total_train_time`：全部 task 总耗时
  - 写入对应 `*_results.json`

## 验收标准
- [ ] `python experiments/run_baselines.py --method all` 完整运行，无报错
- [ ] `results/` 下存在 5 个 JSON 文件（static / finetune / ewc_only / replay_only / proposed）
- [ ] `summarize_results()` 输出 DataFrame 包含全部 5 种方法，列完整
- [ ] 准确率排序符合预期：`proposed ≥ {ewc_only 或 replay_only} ≥ finetune`
- [ ] 鲁棒性测试在 noise_std=0.5 时 proposed 方案准确率下降 < 15%

## 注意事项
- **静态模型是理论上界**（见过全部数据），增量方法不应期望超越
- 鲁棒性噪声叠加在**标准化后**的特征上（z-score 之后），不是原始值
- 加载 checkpoint 测试鲁棒性时，模型必须处于 `model.eval()` 状态
- 随机种子在每个方法开始前重置，确保各方法使用同等条件的初始化权重
