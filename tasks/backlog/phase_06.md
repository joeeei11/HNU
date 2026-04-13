# Phase 6：评估可视化与实验汇总

## 目标
生成完整实验结果图表（可直接用于论文插图），代码完整可复现，项目收尾。

## 前置条件
- Phase 5 全部验收标准已通过
- `results/` 下存在全部 5 个 JSON 结果文件和鲁棒性结果文件

## 任务清单

### 6.1 可视化模块
- [ ] 任务6.1.1：实现 `src/evaluation/visualizer.py`
  - `plot_forgetting_curve(all_results: Dict, save_path) -> None`
    - 横轴：任务序号 0→3，纵轴：所有历史任务平均准确率
    - 5 种方法各一条折线，带标记点
  - `plot_accuracy_heatmap(results_matrix, method_name, save_path) -> None`
    - 横轴：测试任务 0~3，纵轴：训练到的任务轮次
    - 颜色热力图展示遗忘程度
  - `plot_robustness_comparison(robustness_results, save_path) -> None`
    - 横轴：噪声强度 [0, 0.1, 0.3, 0.5]，纵轴：准确率
    - 5 种方法各一条折线
  - `plot_confusion_matrix(y_true, y_pred, save_path) -> None`
    - proposed 方法 Task 3 后的 22 类混淆矩阵
    - 标签简写：N（正常）, F01~F21（故障1~21）
  - 所有图保存到 `results/figures/`，dpi=300，白色背景

### 6.2 Notebook 分析
- [ ] 任务6.2.1：完善 `notebooks/analysis.ipynb`
  - Cell 1: 加载所有 JSON 结果，打印汇总 DataFrame
  - Cell 2: 遗忘曲线对比图
  - Cell 3: proposed 方法准确率热力图
  - Cell 4: proposed 方法 22 类混淆矩阵
  - Cell 5: 鲁棒性对比图
  - Cell 6: 文字总结（中文，含核心数据：平均ACC/FAR/BWT）

### 6.3 代码收尾
- [ ] 任务6.3.1：补全 `requirements.txt`，确保版本号锁定
- [ ] 任务6.3.2：创建 `.gitignore`，排除 `results/*.pth`、`__pycache__/`、`.ipynb_checkpoints/`
- [ ] 任务6.3.3：`pytest tests/ -v` 全量通过

## 验收标准
- [ ] `results/figures/` 下存在 4 张图：`forgetting_curve.png` / `heatmap_proposed.png` / `robustness.png` / `confusion_matrix.png`
- [ ] `notebooks/analysis.ipynb` 执行 Restart & Run All 无报错
- [ ] 汇总 DataFrame 中 proposed 方法：平均 ACC ≥ **90%**，FAR ≤ **3%**
- [ ] `pytest tests/ -v` 全绿，0 FAILED
- [ ] 按 `CLAUDE.md` 启动方式从零运行，Phase 1~4 实验可完整复现

## 注意事项
- matplotlib 中文显示：`plt.rcParams['font.family'] = ['Microsoft YaHei']`（Windows）
- 论文图使用白色背景 + 高对比度配色（避免默认蓝色系）
- 混淆矩阵 22 类标签用简写，避免横轴拥挤：`['N'] + [f'F{i:02d}' for i in range(1,22)]`
- 最终检查 `results/` 不含 `.pth` 权重文件（体积过大，由 .gitignore 排除）
- Notebook 内的路径使用相对路径，确保在项目根目录下 `jupyter notebook` 可直接运行
