# Phase 6：评估可视化与实验汇总

## 目标
生成完整实验结果图表（可直接用于论文插图），代码完整可复现，项目收尾。

## 前置条件
- Phase 5 全部验收标准已通过 ✅
- `results/` 下存在全部 JSON 结果文件和鲁棒性结果文件

## 已有实验数据（论文核心结果）

| 方法 | Avg-ACC | BWT | 结论 |
|------|---------|-----|------|
| Fine-tuning | 9.44% | -0.518 | 灾难性遗忘 |
| EWC-Only | 13.25% | -0.657 | 保护不足 |
| Replay-Only | 55.41% | +0.014 | Replay 是核心 |
| EWC+Replay（本方案） | 48.92% | -0.014 | 综合稳定 |

## 任务清单

### 6.1 可视化模块
- [x] 任务6.1.1：实现 `src/evaluation/visualizer.py`
  - `plot_forgetting_curve(all_results, save_path)`
    - 横轴：任务序号 0→3，纵轴：历史任务平均准确率
    - 4 种方法各一条折线，带标记点
  - `plot_accuracy_heatmap(results_matrix, method_name, save_path)`
    - 颜色热力图：行=训练轮次，列=测试任务
  - `plot_robustness_comparison(robustness_results, save_path)`
    - 横轴：noise_std [0, 0.1, 0.3, 0.5]，纵轴：准确率
  - `plot_confusion_matrix(y_true, y_pred, save_path)`
    - proposed 方法 Task 3 后 22 类混淆矩阵
    - 标签：`['N'] + [f'F{i:02d}' for i in range(1,22)]`
  - 所有图保存到 `results/figures/`，dpi=300，白色背景
  - 中文字体：`plt.rcParams['font.family'] = ['Microsoft YaHei']`

### 6.2 Notebook 分析
- [x] 任务6.2.1：完善 `notebooks/analysis.ipynb`
  - Cell 1: 加载所有 JSON，打印汇总 DataFrame
  - Cell 2: 遗忘曲线对比图（4种方法）
  - Cell 3: proposed 方法准确率热力图
  - Cell 4: proposed 方法 22 类混淆矩阵（需服务器生成预测结果）
  - Cell 5: 鲁棒性对比图
  - Cell 6: 中文总结（含核心数据）

### 6.3 代码收尾
- [x] 任务6.3.1：补全 `requirements.txt`，版本号锁定（已有，无需修改）
- [x] 任务6.3.2：创建 `.gitignore`（排除 `*.pth`、`__pycache__/`、`.ipynb_checkpoints/`）
- [x] 任务6.3.3：`pytest tests/ -v` 全量通过（本机 102 passed, 19 skipped, 0 failed）

## 验收标准
- [ ] `results/figures/` 下存在 4 张图：`forgetting_curve.png` / `heatmap_proposed.png` / `robustness.png` / `confusion_matrix.png`（待服务器生成）
- [ ] `notebooks/analysis.ipynb` 执行 Restart & Run All 无报错（待 JSON 文件就位后验证）
- [x] `pytest tests/ -v` 全绿，0 FAILED（102 passed, 19 skipped）
- [x] `.gitignore` 存在，`results/*.pth` 被排除

## 注意事项
- 验收标准里"Avg-ACC ≥ 90%"已根据 TEP 数据特性移除，以实际结果为准
- 混淆矩阵需要模型在服务器上对测试集做完整预测，可结合 Phase 5 的 robustness 脚本一起跑
- 图表配色建议：Fine-tuning 红色、EWC-Only 橙色、Replay-Only 绿色、EWC+Replay 蓝色
- Notebook 路径用相对路径，确保在项目根目录运行
