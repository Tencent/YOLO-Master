# P1-VOC｜把对照矩阵从 COCO 换到 VOC

**状态：✅ A/B/C/off × 3 seeds 已完成**

矩阵 [`../../p1_voc_matrix.csv`](../../p1_voc_matrix.csv)、
配置 [`../../configs/p1_voc/`](../../configs/p1_voc/)。判读线不变，见 [`../design.md`](../design.md) §6。

## 1. 为什么换数据集

先在完整 COCO 上跑了两个 run 探路（`off-s17` 与 `a-s17`）。链路没问题，问题是预算：

| run | 数据 | mAP50-95 | 墙钟 |
|---|---|---|---|
| `off-s17` | COCO train2017（118 287 张） | **0.19009** | 28 030 s ≈ **7.8 h** |
| `a-s17` | 同上 + DINOv3 P4 蒸馏 | **0.18409** | 41 421 s ≈ **11.5 h** |

证据已归档：[`../../results/p1coco_summary.md`](../../results/p1coco_summary.md)，
逐 run 的 `metrics.csv` / `resolved_args.yaml` 在 `../../results/p1coco_off-s17/`
与 `../../results/p1coco_a-s17/`。

KD 让单次训练贵 **+48%**。按这个单价补完 15 格：

```
3 × 7.8 h  (off)  +  12 × 11.5 h  (四个 KD 格)  ≈  161 h  ≈  6.7 天
```

单卡 6.7 天换一个**探索型**课题的第一版矩阵，不划算——尤其是这个矩阵的
用途是筛掉不 work 的格子，而不是刷 SOTA 数字。VOC 训练集 16 551 张，
是 COCO 的 **1/7.1**，验证集（test2007，4 952 张）与 COCO val2017 基本同量级。

### 预估代价

从 COCO 试跑反推单图代价（假定每 epoch 的验证开销 ≈ 45 s，两臂相同）：

| | 单图训练代价 | VOC 每 epoch | VOC 50 epoch |
|---|---|---|---|
| off | 4.36 ms | ≈ 117 s | **≈ 1.6 h** |
| A / C（单尺度 KD） | 6.62 ms（1.52×） | ≈ 155 s | **≈ 2.2 h** |
| B / D（多尺度 KD） | — | ≈ 170 s | **≈ 2.4 h** |

15 格合计 **≈ 32 h ≈ 1.3 天**，比 COCO 快约 5 倍。

> 45 s 的验证开销是估计值，不是实测——COCO 的 `results.csv` 只记录累计墙钟，
> 拆不出 train/val。整表误差按 ±30% 读。第一格 `off-s17` 跑完即可用实测值校准。

## 2. 最终批次预算

五份配置除 Foundation 区块外逐字相同，由
`validate_pair.py --configs experiments/d2/configs/p1_voc --matrix experiments/d2/p1_voc_matrix.csv`
机械校验（已 PASS）。

```
data VOC.yaml · epochs 400 · imgsz 256 · batch 64 · workers 32
SGD lr0 0.01 lrf 0.01 warmup 3.0 · pretrained false · amp false · deterministic true
seed 17 / 29 / 43（由 run_p1.py 从矩阵覆盖）
```

最初的预算估算使用 50 epochs；正式运行扩展为 400 epochs。VOC 批次内部保持相同预算，
但训练长度与早期 COCO 试跑不同，因此两批结果不作直接精度比较。

`workers: 32` 是跑 COCO 试跑那台机器的值。换机器时**五份一起改**，
否则 `validate_pair.py` 会当场报出漂移。

## 3. 与 COCO 试跑的两处差异（必须在报告里声明）

1. **`foundation_loss_weight`**：`p1coco/a-s17/args.yaml` 记录的是 **3.0**；正式 VOC
   批次按实验格锁定为 A `3.3427745950933323`、B `3.0016106154954567`、
   C `3.1106560298222616`。因此 COCO 的 Δ 与 VOC 的 Δ 不可直接相减。

   `collect_runs.py` 同时检查跨实验格的声明轴和同一实验格内的 seed 一致性，避免把
   不同权重的运行错误合并成同一组三 seed 统计。
2. **数据集**：VOC 20 类、目标普遍更大，同预算下 mAP 会明显高于 COCO 的 0.19。
   判读线 `|Δ mAP50-95| < 0.003` 是绝对刻度，换数据集后**不移动**——
   见 `design.md` §6，结果模糊时只加 seed。

## 4. COCO 试跑说明了什么

只说明了**一件事**：n=1 时 KD 没有帮助。

```
Δ mAP50-95 = 0.18409 − 0.19009 = −0.0060
```

方向为负、幅度超过判读线，但**只有一个 seed，没有置信区间**，
按 `design.md` §6 不构成任何结论。它的作用是排除"KD 根本没接上"这种可能：

```
train/foundation_task_ratio = 0.113
```

KD 项占任务损失的 11.3%（P0 时是 0.4%），说明权重标定确实生效了，
梯度进到学生里了。所以如果 VOC 也判 no-go，归因清单可以直接跳过
"KD 影响力太小"这一条——见 `design.md` 的负结果归因顺序。

## 5. 怎么跑

```bash
# 1. 跑之前：确认配置层无混杂
python experiments/d2/scripts/validate_pair.py \
  --configs experiments/d2/configs/p1_voc \
  --matrix experiments/d2/p1_voc_matrix.csv

# 2. 开跑前：记录本次实验的身份
python experiments/d2/scripts/record_environment.py \
  --out experiments/d2/results/environment_p1voc.json

# 3. 先跑一对（off + A，seed 17），用实测墙钟校准 §1 的预估
python experiments/d2/scripts/run_p1.py --device 0 \
  --matrix experiments/d2/p1_voc_matrix.csv --configs p1_voc \
  --project d2/p1voc --only off-s17,a-s17

# 3b. 运行本轮验收所需的 12 格（已完成的会自动跳过）
python experiments/d2/scripts/run_p1.py --device 0 \
  --matrix experiments/d2/p1_voc_matrix.csv --configs p1_voc --project d2/p1voc \
  --only off-s17,off-s29,off-s43,a-s17,a-s29,a-s43,b-s17,b-s29,b-s43,c-s17,c-s29,c-s43

# 4. 跑完后：归档 + 汇总 + 事后混杂核查
python experiments/d2/scripts/collect_runs.py runs/detect/d2/p1voc/* --label p1voc
```

VOC 首次运行会自动下载并转换标注（约 2.8 GB），发生在第一格里，
不计入训练代价但会拉长第一格的墙钟。

## 6. 还有哪些提速手段（本批**未**采用）

按性价比排序，采用任何一条都必须**五份配置一起改**：

| 手段 | 预估收益 | 代价 |
|---|---|---|
| `amp: true` | 训练部分 −30~40% | 放弃 `deterministic` 意义上的逐位可复现；数值噪声进入 Δ |
| `foundation_cache_teacher_features: true` | KD 格的教师前向可摊销 | 需确认与数据增强的交互（增强后图像每 epoch 不同，缓存可能无效） |
| `imgsz` 降到 192 | 约 −40% | 判读线是在 256 上定的，改了要重新论证 |
| `fraction < 1.0` | 线性 | 相当于又换了一次数据集，抵消换 VOC 的意义 |

`amp: true` 是里面唯一近乎白拿的一项，但它与 `deterministic: true` 相冲——
这个取舍应在开跑前定，跑到一半再改会让前后两批不可比。
