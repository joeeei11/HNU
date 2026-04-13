# 基于增量学习的工业过程故障监测系统

湖南科技大学 黄玺

## 项目简介

在 Tennessee Eastman Process (TEP) 标准数据集上，构建能持续学习新故障类型、同时不遗忘已学旧故障的智能监测模型。核心方案为 **1D-CNN + Transformer** 骨干网络 + **EWC + Experience Replay** 混合增量策略，并与 Fine-tuning / 纯 EWC / 纯 Replay 做全面对比实验。

## 实验结果

| 方法 | Avg-ACC | BWT | 说明 |
|------|---------|-----|------|
| Fine-tuning | 9.44% | -0.518 | 灾难性遗忘，旧任务归零 |
| EWC-Only | 13.25% | -0.657 | 保护不足 |
| Replay-Only | 55.41% | +0.014 | Replay 是防遗忘核心机制 |
| **EWC+Replay（本方案）** | **48.92%** | **-0.014** | 综合稳定，遗忘几乎为零 |

## 环境要求

- Python 3.12
- PyTorch 2.8.0 + CUDA 12.8（训练）
- 本机调试：CPU 即可

## 快速开始

```bash
# 1. 克隆仓库
git clone <repo-url>
cd HNU-zengliangxuexi

# 2. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 运行单元测试（本机 CPU）
pytest tests/ -v

# 4. 修改数据路径（config/config.yaml）
# 本机调试：raw_dir: "Source/data/data/data"
# 服务器训练：raw_dir: "/root/autodl-tmp/tep_data"

# 5. 运行主实验（需 GPU）
python experiments/run_proposed.py --config config/config.yaml

# 6. 运行所有基线对比
python experiments/run_baselines.py --method all --config config/config.yaml

# 7. 生成可视化图表（本机）
jupyter notebook notebooks/analysis.ipynb
```

## 目录结构

```
├── src/
│   ├── data/           # 数据加载、预处理、增量任务划分
│   ├── models/         # 1D-CNN + Transformer 骨干网络
│   ├── continual/      # EWC + ReplayBuffer + 混合训练器
│   ├── baselines/      # Fine-tuning / EWC-Only / Replay-Only
│   └── evaluation/     # 指标计算 + 可视化
├── experiments/        # 实验脚本入口
├── tests/              # 单元测试
├── config/             # 超参数配置
├── results/            # 实验结果（JSON + 图表）
│   └── figures/        # 论文图表
├── Source/
│   └── data/data/data/ # TEP 原始数据集（.dat 文件）
└── notebooks/
    └── analysis.ipynb  # 结果分析 Notebook
```

## 数据集

使用 Tennessee Eastman Process (TEP) 公开数据集：
- 22 类（正常工况 + 21 种故障）
- 训练集 480 行 × 52 特征，测试集 960 行 × 52 特征
- 数据文件位于 `Source/data/data/data/`

## 增量学习任务划分

| 任务 | 包含类别 | 说明 |
|------|---------|------|
| Task 0 | 类 0~3 | 正常 + Fault 1~3 |
| Task 1 | 类 4~8 | Fault 4~8（新增） |
| Task 2 | 类 9~14 | Fault 9~14（新增） |
| Task 3 | 类 15~21 | Fault 15~21（新增） |

## 服务器部署（AutoDL）

详见 `CLAUDE.md` 的"AutoDL 服务器"章节。

镜像：PyTorch 2.8.0 / Python 3.12 / CUDA 12.8

## 引用

```
Tennessee Eastman Process dataset:
Downs, J.J. and Vogel, E.F. (1993). A plant-wide industrial process control problem.
Computers & Chemical Engineering, 17(3), 245-255.
```
