# Phase 2：骨干模型实现（1D-CNN + Transformer）

## 目标
完成 FaultClassifier 模型，能对单个增量任务批次做前向推理，静态训练 Task 0（4类）测试准确率达到 85% 以上，验证模型表达能力。

## 前置条件
- Phase 1 全部验收标准已通过
- `pytest tests/test_loader.py tests/test_preprocessor.py -v` 全绿

## 任务清单

### 2.1 CNN 编码器
- [ ] 任务2.1.1：实现 `src/models/cnn_encoder.py`
  - `Conv1d(in=52, out=128, kernel=3, padding=1)` → BN → ReLU → Dropout(0.2)
  - `Conv1d(128, 256, kernel=3, padding=1)` → BN → ReLU
  - `AdaptiveAvgPool1d(1)` → Flatten → 输出 `[B, 256]`
  - 输入 shape：`[B, 52, W]`（注意 Conv1d 要求 channel-first）

### 2.2 Transformer 编码器
- [ ] 任务2.2.1：实现 `src/models/transformer_encoder.py`
  - 输入投影：`Linear(256, 256)`（接收 CNN 特征，按时间步展开后输入）
  - 可学习位置嵌入：`nn.Parameter(torch.zeros(1, window_size, 256))`
  - `nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True)` × 4 层
  - CLS Token 聚合 → 输出 `[B, 256]`

### 2.3 整体分类模型
- [ ] 任务2.3.1：实现 `src/models/fault_classifier.py`
  - `FaultClassifier(num_classes=22, d_model=256)`
  - forward 流程：`x[B,50,52]` → permute → CNN → reshape → Transformer → `Linear(256,22)` → logits
  - `get_features(x) -> Tensor[B,256]`：返回分类头前的中间特征（EWC 使用）
  - `count_params() -> int`：打印参数量

### 2.4 静态基线验证
- [ ] 任务2.4.1：实现 `src/baselines/static_trainer.py`
  - `StaticTrainer.train(model, dataloader, epochs=50)` → Adam(lr=1e-3) + CrossEntropyLoss
  - 支持 AMP：`torch.cuda.amp.autocast()` + `GradScaler`
  - 每 5 epoch 打印 train_loss / val_acc
  - 训练完保存到 `results/static_task0.pth`
- [ ] 任务2.4.2：在 `experiments/run_baselines.py` 中添加静态单任务验证入口
  - 仅用 Task 0（4类）验证模型能力

### 2.5 单元测试
- [ ] 任务2.5.1：实现 `tests/test_model.py`
  - forward 输出 shape = `[B, 22]`
  - get_features 输出 shape = `[B, 256]`
  - count_params 在 `[500_000, 5_000_000]` 范围内
  - CPU 和 CUDA 均可运行

## 验收标准
- [ ] `pytest tests/test_model.py -v` 全部通过
- [ ] 静态训练 Task 0（4类，epoch=50），测试集准确率 ≥ 70%
- [ ] 模型 forward（batch=256）在 RTX 5090 上单次 < 50ms
- [ ] AMP 开启时无报错，loss 正常下降

## 注意事项
- CNN 输入需 permute：`x.permute(0, 2, 1)` 将 `[B, W, 52]` 转为 `[B, 52, W]`
- Transformer 的 `batch_first=True` 必须设置，否则维度顺序不符
- num_classes 全程固定为 22，Task 0 训练时用 label mask 忽略未见类（或直接让未见类 logit 不参与 loss）
- AMP 下 loss.backward() 需替换为 `scaler.scale(loss).backward()`
