# 基于增量学习的工业过程故障监测系统

## 项目概述

本项目是湖南师范大学信息科学与工程学院人工智能22级王程的本科毕业设计，研究题目为《基于增量学习的工业过程故障监测方法研究》。系统在 Tennessee Eastman Process (TEP) 标准数据集上，构建能持续学习新故障类型、同时不遗忘已学旧故障的智能监测模型，核心方案为 1D-CNN + Transformer 骨干网络 + EWC + Experience Replay 混合增量策略，并与静态模型等基线做全面对比实验。

## 技术栈

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12.x | 主语言 |
| PyTorch | 2.8.x | 深度学习框架（Blackwell/5090 完整支持） |
| NumPy | 1.26.x | 数值计算 |
| Pandas | 2.1.x | 数据操作 |
| Scikit-learn | 1.3.x | 预处理、评估指标 |
| Matplotlib | 3.8.x | 绘图 |
| Seaborn | 0.13.x | 统计可视化 |
| tqdm | 4.66.x | 进度条 |
| PyYAML | 6.0.x | 配置文件解析 |
| TensorBoard | 2.15.x | 实验曲线记录（可选） |
| pytest | 7.4.x | 单元测试 |

**本机（编写代码）：** Intel Core i7-12700H，Windows 11  
**训练服务器：** AutoDL 租用，NVIDIA RTX 5090（32GB VRAM），Ubuntu，按需开机  
**AutoDL 镜像：** PyTorch 2.8.0 / Python 3.12 / CUDA 12.8

## 目录结构

```
HNU-zengliangxuexi/
├── CLAUDE.md                    # 本文件
├── requirements.txt             # Python依赖
├── config/
│   └── config.yaml              # 全局超参数配置
├── Source/
│   ├── data/data/data/          # 原始TEP .dat 文件（只读，勿修改）
│   ├── docx/                    # 开题报告
│   └── Web.ai/                  # AI顾问建议
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # 加载 .dat 文件 → numpy
│   │   ├── preprocessor.py      # StandardScaler + 滑动窗口切片
│   │   └── task_splitter.py     # 按故障类型划分增量任务批次
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_encoder.py       # 1D-CNN 局部特征提取器
│   │   ├── transformer_encoder.py # Transformer 长程依赖编码器
│   │   └── fault_classifier.py  # 整体分类模型（CNN+Transformer+Head）
│   ├── continual/
│   │   ├── __init__.py
│   │   ├── ewc.py               # EWC：Fisher信息矩阵计算 + 正则损失
│   │   ├── replay_buffer.py     # 记忆库：蓄水池采样
│   │   └── trainer.py           # 混合增量训练主循环
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── static_trainer.py    # 基线1：静态全量训练
│   │   ├── ewc_only_trainer.py  # 基线2：纯EWC
│   │   └── replay_only_trainer.py # 基线3：纯Replay
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py           # ACC、FAR、FDR、BWT、FWT计算
│       └── visualizer.py        # 混淆矩阵、遗忘曲线、热力图
├── experiments/
│   ├── run_proposed.py          # 主实验：EWC+Replay混合
│   ├── run_baselines.py         # 对比实验：3个基线
│   └── run_robustness.py        # 鲁棒性：高斯噪声测试
├── results/
│   └── .gitkeep                 # 实验结果输出目录
├── notebooks/
│   └── analysis.ipynb           # 结果分析探索
├── tests/
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   └── test_model.py
└── tasks/
    ├── current.md               # 当前执行任务
    ├── progress.md              # 进度快照
    └── decisions.md             # 技术决策记录
```

## 环境变量

本项目无需网络服务，无外部 API Key。以下为可选配置（在 `config/config.yaml` 中管理，不使用 .env 文件）：

```yaml
# config/config.yaml 中的关键参数（非环境变量，统一在此配置）
data:
  # 本机路径（调试用）：Source/data/data/data
  # 服务器路径（训练用）：/root/autodl-tmp/tep_data
  raw_dir: "/root/autodl-tmp/tep_data"  # ← 服务器上运行时用此路径
  processed_dir: "/root/autodl-tmp/processed"  # 预处理缓存（高速盘）

training:
  window_size: 50       # 滑动窗口大小
  stride: 10            # 滑动窗口步长
  batch_size: 256       # 训练批大小（RTX 5090可大幅提升）
  lr: 1e-3              # 学习率
  epochs_per_task: 50   # 每个增量任务训练轮数
  use_amp: true         # 混合精度训练（AMP，加速+节省显存）
  num_workers: 4        # DataLoader 并行加载

ewc:
  lambda: 5000          # EWC正则强度

replay:
  buffer_size_per_class: 50  # 每类记忆样本数

model:
  cnn_channels: [128, 256]   # CNN通道数
  transformer_layers: 4      # Transformer层数
  transformer_heads: 8       # 注意力头数
  d_model: 256               # 特征维度
```

## 启动方式

### 本机（单元测试 / 小规模调试）
```bash
# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行单元测试
pytest tests/ -v
```

### AutoDL 服务器（正式训练）

**① 首次部署（只做一次）**
```bash
# 本机：上传数据集到服务器高速盘
scp -P <端口> -r "D:\MYSOFTWAREOFtechnology\ClaudeCodeFile\Projects\HNU-zengliangxuexi\Source\data\data\data" root@<host>:/root/autodl-tmp/tep_data

# 本机：上传代码
scp -P <端口> -r "D:\MYSOFTWAREOFtechnology\ClaudeCodeFile\Projects\HNU-zengliangxuexi" root@<host>:/root/fault-detection

# 服务器：安装依赖
cd /root/fault-detection
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 服务器：验证 GPU 与数据
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -m src.data.loader
```

**② 每次开机后（更新代码）**
```bash
# 本机：只同步代码变更（不重传数据）
scp -P <端口> -r "D:\...\src" root@<host>:/root/fault-detection/
scp -P <端口> -r "D:\...\experiments" root@<host>:/root/fault-detection/
scp -P <端口> "D:\...\config\config.yaml" root@<host>:/root/fault-detection/config/
```

**③ 后台运行实验（关闭 SSH 不中断）**
```bash
# 主实验
nohup python experiments/run_proposed.py --config config/config.yaml > results/proposed.log 2>&1 &

# 对比基线（全部方法）
nohup python experiments/run_baselines.py --method all --config config/config.yaml > results/baselines.log 2>&1 &

# 鲁棒性测试
nohup python experiments/run_robustness.py --noise_std 0.1 0.3 0.5 > results/robustness.log 2>&1 &

# 查看实时日志
tail -f results/proposed.log

# 查看 GPU 占用
watch -n 2 nvidia-smi
```

**④ 下载实验结果到本机**
```bash
# 本机执行：下载 results 目录（排除大体积权重文件）
scp -P <端口> -r root@<host>:/root/fault-detection/results "D:\MYSOFTWAREOFtechnology\ClaudeCodeFile\Projects\HNU-zengliangxuexi\"
```

**⑤ 节省费用**
- 实验跑完立即在 AutoDL 控制台**关机**（停止计费）
- 配好环境后在控制台保存**镜像快照**，下次开机直接用
- 数据和权重放 `/root/autodl-tmp/`（高速 + 不随实例删除）

## API 规范

本项目为研究型代码，模块接口如下：

### 数据层
```python
# src/data/loader.py
load_tep_data(raw_dir: str, task_id: int) -> Tuple[np.ndarray, np.ndarray]
# 返回 (X: [N, 52], y: [N,])，task_id=0表示加载正常+前4类故障

# src/data/preprocessor.py
class TEPPreprocessor:
    fit_transform(X_train) -> X_scaled  # 标准化（仅fit训练集）
    sliding_window(X, y, window_size, stride) -> (X_win, y_win)
    # X_win: [N_windows, window_size, 52]

# src/data/task_splitter.py
class TaskSplitter:
    get_task(task_id: int) -> (train_loader, test_loader)
    # Task划分: [0-3类] → [0-8类] → [0-14类] → [0-21类]
```

### 模型层
```python
# src/models/fault_classifier.py
class FaultClassifier(nn.Module):
    forward(x: Tensor[B, W, 52]) -> logits: Tensor[B, 22]
    get_features(x) -> features: Tensor[B, d_model]
```

### 增量学习层
```python
# src/continual/trainer.py
class ContinualTrainer:
    train_task(task_id: int, train_loader, epochs: int) -> dict  # 返回loss记录
    evaluate_all_tasks() -> dict  # 返回所有历史任务的当前准确率

# src/continual/ewc.py
class EWC:
    compute_fisher(model, loader, task_id) -> None   # 计算并保存Fisher矩阵
    penalty(model) -> Tensor  # 返回EWC正则损失标量

# src/continual/replay_buffer.py
class ReplayBuffer:
    add_task_samples(X, y, task_id) -> None  # 蓄水池采样加入记忆库
    sample_replay_batch(batch_size) -> (X, y)
```

### 评估层
```python
# src/evaluation/metrics.py
compute_metrics(y_true, y_pred) -> {
    "acc": float,          # 整体准确率
    "far": float,          # 误报率 (False Alarm Rate)
    "fdr": float,          # 故障检测率
    "bwt": float,          # Backward Transfer（遗忘度量）
    "per_class_acc": list  # 各类准确率
}
```

## 开发规范

### 命名规范
- 文件名：`snake_case.py`
- 类名：`PascalCase`（如 `FaultClassifier`）
- 函数/变量：`snake_case`（如 `compute_fisher`）
- 常量：`UPPER_SNAKE_CASE`（如 `NUM_CLASSES = 22`）
- 测试文件：`test_<模块名>.py`

### 代码风格
- 遵循 PEP 8
- 所有公开函数必须有类型注解和 docstring（中文或英文均可）
- 超参数统一从 `config.yaml` 读取，禁止硬编码在模型文件中
- 注释语言与现有代码保持一致（本项目以中文注释为主）

### 提交规范（仅供参考，无需强制）
```
feat: 新增 EWC Fisher矩阵计算模块
fix: 修复滑动窗口边界越界问题
exp: 运行Phase3实验，记录基线对比结果
docs: 更新 tasks/progress.md
```

## 任务管理

- `tasks/current.md`    — 当前阶段任务清单（逐项打勾）
- `tasks/progress.md`   — 各阶段完成情况快照
- `tasks/decisions.md`  — 关键技术决策及理由记录

### 每次新 session 开头：
读 `CLAUDE.md` → `tasks/progress.md` → `tasks/decisions.md` → `tasks/current.md`，执行当前任务。

### 每次结束前：
更新 `tasks/progress.md`（已完成项打勾），如有新技术决策更新 `tasks/decisions.md`，然后结束。

## 禁止事项

- 禁止修改 `Source/data/` 目录下的原始 .dat 文件（只读数据源）
- 禁止在模型代码中硬编码超参数（必须通过 config.yaml 传入）
- 禁止在训练集之外的数据上 fit StandardScaler（防止数据泄露）
- 禁止在评估对比实验时为不同方法使用不同超参数（保证公平对比）
- 禁止将 results/ 目录中的大型模型权重文件提交到 git
- 禁止在增量训练时直接访问历史任务的完整训练集（仅允许访问 ReplayBuffer 中的记忆样本）
