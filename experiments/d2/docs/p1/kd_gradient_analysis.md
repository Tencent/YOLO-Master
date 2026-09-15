# P1｜KD 梯度标定与可学性诊断

本页记录正式 P1 权重的来源，并回答两个前置问题：

1. 每种教师与层级组合应使用多大的 KD 权重？
2. 学生能否学到当前 relational KD 目标，还是只有投影器在吸收误差？

Probe 只用于诊断和标定，不读取 mAP。精度结论见 [`findings.md`](findings.md)。

## 1. 为什么不用 loss 数值占比

P0 使用的 `foundation_loss_weight=0.05` 让 KD loss 数值进入总目标，但 loss 的数值占比不能代表它对共享参数的更新强度。早期 weight sweep 中，权重扩大 16 倍也没有产生稳定、单调的检测 loss 变化，因此无法据此选择权重。

优化器响应的是梯度。正式标定改为直接比较共享 student 参数上的梯度范数：

```text
ratio(w=1) = ‖g_kd‖ / ‖g_task‖
w* = target_ratio / ratio(w=1)
```

目标比例在测量前固定为 `0.1`，即让加权 KD 梯度范数约为任务梯度范数的 10%。这是一项实验设计选择，不是搜索得到的最优超参数。

## 2. Probe A：VOC 四格权重标定

### 协议

- 数据：VOC，`imgsz=256`
- 模型：未预训练的 `yolo26-master-n`
- 硬件：RTX 3090
- 每格：50 batches
- 测量：分别反传未加权 KD 与 task loss，统计共享参数梯度
- 附加诊断：记录两个梯度的 cosine 与冲突 batch 比例

运行入口：

```bash
python experiments/d2/scripts/kd_gradient_probe.py --mode ratio \
  --batches 50 --target-ratio 0.1
```

机器可读证据：

- [A：DINOv3 P4](../../results/probe_a_voc_3090/a.json)
- [B：DINOv3 P3+P4+P5](../../results/probe_a_voc_3090/b.json)
- [C：SigLIP2 P4](../../results/probe_a_voc_3090/c.json)
- [D：SigLIP2 P3+P4+P5](../../results/probe_a_voc_3090/d.json)

### 结果

| 配方 | `w=1` 时梯度比 | 10% 目标权重 | cosine 均值（95% CI） | 冲突 batch |
|---|---:|---:|---|---:|
| A | 0.02992 | 3.34277 | +0.0122 [-0.0350, +0.0593] | 44% |
| B | 0.03332 | 3.00161 | +0.0118 [-0.0385, +0.0620] | 50% |
| C | 0.03215 | 3.11066 | -0.0137 [-0.0572, +0.0298] | 52% |
| D | 0.05381 | 1.85823 | -0.0014 [-0.0459, +0.0430] | 50% |

四种配方的未加权 KD 梯度强度不同，因此使用不同权重才能对齐到同一个 10% 目标。D 的 Probe A 已完成，但 D 没有进入 400-epoch 精度实验。

四格 cosine 的 95% CI 都包含 0，冲突比例接近 50%。训练起点没有检测到 KD 与任务梯度持续同向或反向的证据。这不能预测 mAP：近似正交的 KD 可能是有用正则，也可能只是无害噪声。

## 3. Probe B：KD 目标是否可学

Probe B 单独优化 KD 目标 300 steps，并比较完整 student+projector 与 projector-only 两个分支。证据见 [`probe_b_learnability.json`](../../results/probe_b_learnability.json)。

| 分支 | fit KD 降幅 | held-out KD 降幅 |
|---|---:|---:|
| student backbone + projector | 94.4% | 59.4% |
| projector only | 42.4% | 28.1% |

冻结 backbone 后的 held-out 残差是完整分支的 1.77 倍。当前目标在这个小型诊断中可达，而且 backbone 对改善有贡献；投影器同时吸收了部分误差。

这个探针使用 coco128、2 个 fit batches、8 个 held-out batches 和 KD-only Adam 优化。它只证明“原理上可学”，不能证明正式 P1 的 SGD 与 10% 梯度份额会在预算内学到相同程度。

## 4. 结论边界

- 四格权重是同一梯度目标的换算结果，不是最优权重。
- 权重测于未预训练模型的训练起点；梯度比例可能随训练变化。
- A/B/C 使用不同权重，因此正式结果比较的是完整校准配方。
- Probe A/B 不构成精度证据；mAP 判断只来自配对 P1 训练。
- D 只有权重标定结果，没有正式精度结果。
