# D2 已知局限、风险与降级方案

## 1. 证据边界

- **P0 只证明路径完整性。** 三个 epoch 的 mAP50-95 均为 0；它证明 teacher、tap、projector 和 KD loss 接入真实训练并进入反向传播，不支持任何精度主张。
- **P1 只在 VOC 完成。** off/A/B/C 各有 seeds 17、29、43 和 400 epochs。完整多 seed 结论不能直接外推到 COCO 或其他模型规模。
- **每格只有三个 seed。** A/B 的配对 t 区间很宽。A 的平均提升不能表述为稳定涨点。
- **D 未运行。** 当前数据不能估计完整的教师 × 尺度交互。

## 2. 实验解释限制

### 2.1 配方权重不同

A、B、C 使用不同的固定 `foundation_loss_weight`：

| 配方 | 权重 |
|---|---:|
| A：DINOv3 P4 | 3.3427745950933323 |
| B：DINOv3 multiscale | 3.0016106154954567 |
| C：SigLIP2 P4 | 3.1106560298222616 |

这些结果比较的是完整配方。A−B 同时改变层级和权重，A−C 同时改变教师和权重，因此不能把差值解释成纯尺度效应或纯教师效应。

### 2.2 多尺度包含插值

DINOv3 在 256×256 输入下产生 16×16 patch 网格，与学生 P4 对齐。B 还蒸馏 P3 32×32 和 P5 8×8，因此需要放大或缩小教师特征。若 B 表现较差，现有实验无法完全区分多尺度本身与插值近似的影响。

### 2.3 Foundation 教师未缓存

教师冻结但仍在每个 batch 前向。缓存可能降低训练开销，但会与几何增强和输入预处理产生额外一致性要求，因此本轮关闭缓存。若未来启用，缓存键至少要包含教师 model id、revision、权重 hash、预处理版本、样本 ID 和目标层级。

## 3. 环境与可复现性

### 3.1 DINOv3 访问受控

DINOv3 权重在 Hugging Face 上为 gated。运行者必须登录并接受模型许可；模型权重不提交到 Git。无法访问时应停止运行并记录失败原因，不能换用其他教师后沿用 DINOv3 标签。

### 3.2 依赖范围

Foundation extra 当前要求 `transformers>=5,<6`，与代码使用的 `DINOv3ViTBackbone` 接口一致。验证过的正式训练环境为 Python 3.12.14、PyTorch 2.11.0+cu128、Ultralytics 8.4.101 和 RTX 3090 24 GB；见 [`../env/README.md`](../env/README.md)。

### 3.3 教师 revision 只能记录，不能强制

当前 Foundation 配置没有 revision 字段，教师加载器不会把 revision 传给 `from_pretrained`。本轮在 [`../p1_voc_matrix.csv`](../p1_voc_matrix.csv) 和环境快照中记录：

| 教师 | revision |
|---|---|
| DINOv3 | `114c1379950215c8b35dfcd4e90a5c251dde0d32` |
| SigLIP2 patch16-256 | `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` |

复现时需要人工核对缓存 snapshot。未来若增加 revision 配置，multi-teacher 模式需要分别记录每个教师，不能共用一个含义不清的字段。

### 3.4 训练 Git SHA 缺失

现存 P1 训练日志没有可靠保存精确 Git SHA。逐运行配置、实际参数 manifest、教师 revision、软件版本和硬件已归档，但无法严格证明训练代码与最终 PR HEAD 字节级一致。manifest 保留了每份原始 `args.yaml` 的 SHA-256；完整文件不在 PR 中重复提交。未来每批运行必须先执行 `record_environment.py`，并要求 `experiment_inputs_dirty=false`。

## 4. 性能与部署 fallback

相对 off 的平均训练墙钟变化：

| 配方 | 平均增幅 |
|---|---:|
| A | +21.8% |
| B | +22.0% |
| C | +64.6% |

Foundation teacher 和对齐投影器只服务训练。部署 fallback 是导出 student-only 模型；推理不加载教师，也不承担上述训练开销。

## 5. 风险触发线

| 风险 | 触发条件 | 处理 |
|---|---|---|
| 教师不可访问 | gated 权限、token 或下载失败 | 停止并记录失败；不替换教师标签 |
| 显存不足 | CUDA OOM | 对所有对照组同步降低 batch 或 imgsz |
| 配置漂移 | `validate_pair.py` 失败 | 不启动训练，先恢复共享预算一致 |
| 实际参数漂移 | `collect_runs.py` 报 confound | 不合并统计，重跑受影响 seed |
| 指标不确定 | 配对 CI 跨 0 | 增加 seed，不移动 0.3 mAP 判读线 |
| 部署不需要教师 | 训练完成 | 使用 student-only 导出路径 |
