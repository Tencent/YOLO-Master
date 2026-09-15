# D2 设计文档｜Foundation 特征蒸馏现状验证

**代码基线**：`e9ac08b`（当时的 `upstream/main`）

**主指标**：VOC `metrics/mAP50-95(B)`

**实验单位**：相同 seed 下的 treatment − off 配对差值

本文件记录最终执行的实验协议。完整结果见 [`p1/findings.md`](p1/findings.md)，已知局限与降级方案见 [`limitations.md`](limitations.md)。

## 1. 研究问题与边界

研究问题：冻结 Foundation 教师的中间特征，能否在训练预算相同的情况下改善小型 `yolo26-master-n`？

D2 不重新实现蒸馏框架。仓库已经提供 teacher、student feature tap、projector、dense relational loss 和训练 wrapper。D2 负责：

1. 用仓库真实训练入口验证链路完整性。
2. 固定训练预算并建立可审计对照。
3. 使用多个配对 seed 报告数字、区间和负结果。
4. 保存配置、实际解析参数、逐 epoch 指标、环境和复现命令。

P0 只回答“链路是否真的接通”；P1 才回答“完整蒸馏配方在当前预算下是否有效”。

## 2. 被验证的训练路径

| 组件 | 代码位置 | P0/P1 核对内容 |
|---|---|---|
| 教师后端 | `ultralytics/nn/foundation/teachers/` | 教师冻结，只在训练时前向 |
| 学生特征 | `ultralytics/nn/foundation/taps.py` | P-level 特征保留梯度 |
| 对齐投影 | `ultralytics/nn/foundation/projectors.py` | 学生投影器可训练，教师侧冻结 |
| 蒸馏损失 | `ultralytics/nn/foundation/losses.py` | 使用 relational KD |
| 训练 wrapper | `ultralytics/nn/foundation_distill_model.py` | KD 加权后进入总 loss |
| Trainer 集成 | `ultralytics/engine/trainer.py` | 从配置注入 wrapper 并记录指标 |

P0 的三轮真实训练满足：Foundation 指标列存在、raw KD 有限且非零、加权恒等式成立、结果写入 CSV。它证明实现链路可运行，不构成精度结论。证据见 [`p0/README.md`](p0/README.md)。

## 3. P1 最终实验设计

### 3.1 数据与共享预算

| 字段 | 固定值 |
|---|---|
| data | `VOC.yaml` |
| model | `yolo26-master-n` |
| epochs | 400 |
| imgsz | 256 |
| batch | 64 |
| optimizer | SGD |
| lr0 / lrf | 0.01 / 0.01 |
| pretrained / amp | false / false |
| deterministic | true |
| seeds | 17、29、43 |

VOC 用于在单卡预算内完成多 seed 对照。完整三 seed 结论只适用于这一数据与预算，不能直接外推到 COCO。

### 3.2 配方矩阵

| 配方 | 教师 | 学生层级 | 固定 KD 权重 | 状态 |
|---|---|---|---:|---|
| off | 无 | 无 | 0 | 3 seeds 完成 |
| A | DINOv3 ViT-S/16 | P4 | 3.3427745950933323 | 3 seeds 完成 |
| B | DINOv3 ViT-S/16 | P3+P4+P5 | 3.0016106154954567 | 3 seeds 完成 |
| C | SigLIP2 base patch16-256 | P4 | 3.1106560298222616 | 3 seeds 完成 |
| D | SigLIP2 base patch16-256 | P3+P4+P5 | 1.858231765126175 | 跳过 |

机器可读矩阵见 [`../p1_voc_matrix.csv`](../p1_voc_matrix.csv)，配置见 [`../configs/p1_voc/`](../configs/p1_voc/)。验收要求允许完成 2×2 中至少三格，因此在 off/A/B/C 已形成主证据后没有运行 D。

### 3.3 权重口径

早期扫描发现，relational loss 的数值占比不能稳定代表其梯度影响。权重因此通过梯度比探针确定，并在正式运行前固定；诊断过程见 [`p1/kd_gradient_analysis.md`](p1/kd_gradient_analysis.md)。

A、B、C 使用不同的固定权重。它们代表各自完整蒸馏配方，而不是只改变一个因素的严格因果消融：

- A 与 B 同为 DINOv3，但同时改变层级和 KD 权重。
- A 与 C 同为 P4，但同时改变教师和 KD 权重。

因此报告可以比较配方表现，但不能把 A−B 全部归因为尺度，也不能把 A−C 全部归因为教师。

最终权重来自 VOC、RTX 3090 上各 50 batches 的 Probe A：

| 配方 | `w=1` 时梯度比 | 10% 目标权重 | 平均 cosine（95% CI） | 冲突 batch |
|---|---:|---:|---|---:|
| A | 0.02992 | 3.34277 | +0.0122 [-0.0350, +0.0593] | 44% |
| B | 0.03332 | 3.00161 | +0.0118 [-0.0385, +0.0620] | 50% |
| C | 0.03215 | 3.11066 | -0.0137 [-0.0572, +0.0298] | 52% |
| D | 0.05381 | 1.85823 | -0.0014 [-0.0459, +0.0430] | 50% |

机器可读结果见 [`../results/probe_a_voc_3090/`](../results/probe_a_voc_3090/)。D 的探针完成只说明其权重已标定，不代表 D 完成正式训练。

### 3.4 教师身份

| 教师 | model id | 记录的 revision |
|---|---|---|
| DINOv3 | `facebook/dinov3-vits16-pretrain-lvd1689m` | `114c1379950215c8b35dfcd4e90a5c251dde0d32` |
| SigLIP2 | `google/siglip2-base-patch16-256` | `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` |

当前配置接口不能把 revision 强制传入教师加载器，因此 revision 作为审计信息保存。环境证据见 [`../env/`](../env/README.md)。

## 4. 对照完整性

### 4.1 训练前检查

[`../scripts/validate_pair.py`](../scripts/validate_pair.py) 检查：

- 五份配置中 Foundation 区块以外的预算字段一致。
- 矩阵中非实验字段在所有运行间一致。
- run id 没有重复。

```bash
python experiments/d2/scripts/validate_pair.py
```

### 4.2 训练后检查

[`../scripts/collect_runs.py`](../scripts/collect_runs.py) 读取每个完成运行的 `args.yaml`，归档为 `resolved_args.yaml`，并检查：

- 跨配方只在声明的 Foundation 字段和运行身份上变化。
- 同一配方的不同 seed 不得改变 teacher、层级、权重或其他实际解析参数。

这一步防止配置文件正确、实际命令却覆盖了不同参数。当前 12 个 P1 运行通过检查。

## 5. 统计与判读

主统计量是每个 seed 的：

```text
Δ = mAP50-95(treatment, seed) − mAP50-95(off, seed)
```

报告三个配对差值的均值、样本标准差和自由度 2 的双侧 t 区间。`n=3` 时区间不稳定，因此必须同时展示每个 seed，不能使用单次最好结果代替均值。

判读线在正式 P1 结果产生前固定：

```text
|平均 Δ| < 0.3 mAP 点，并且 95% CI 包含 0  →  no-go
```

Ultralytics CSV 使用 0–1 标度，所以 `0.3 mAP` 点等于 `0.003`。结果不确定时增加 seed，不移动阈值。

## 6. 最终结果与决策

| 配方 | 平均 mAP50-95 | 平均配对 Δ | 95% CI（mAP 点） | 决策 |
|---|---:|---:|---|---|
| off | 0.47295 | — | — | baseline |
| A | 0.47678 | +0.383 mAP | [-1.303, +2.069] | 方向性信号；不能宣称稳定提升 |
| B | 0.47088 | -0.207 mAP | [-1.682, +1.268] | no-go |
| C | 0.47013 | -0.282 mAP | [-0.362, -0.202] | 一致小幅下降，停止 |

A 的 seeds 17/29 为正，seed 43 为负。B 命中预设 no-go 规则。C 的三个 seed 都下降，虽然平均绝对差值略低于 0.3 点，其区间不含 0，说明当前三次运行中的负方向较一致。

训练动态显示所有 KD 配方在前中期落后于 off；只有 A 的三 seed 平均曲线在后期转正。完整逐 seed 数字、曲线和计算口径见 [`p1/findings.md`](p1/findings.md)。

## 7. 解释限制与风险

1. **样本量**：每格仅三个 seed，A/B 区间很宽。
2. **配方权重不同**：教师或尺度效应不能脱离权重单独归因。
3. **多尺度插值**：教师产生 16×16 网格；B 的 P3 32×32 和 P5 8×8 需要插值，因此层级与插值影响耦合。
4. **缺少 D**：无法估计完整的教师 × 尺度交互。
5. **数据外推**：VOC 结果不能直接代表完整 COCO。
6. **代码身份**：现存训练日志没有可靠保存精确 Git SHA；配置、解析参数、教师 revision 和环境版本已保存。

训练平均墙钟相对 off 增加约 21.8%（A）、22.0%（B）和 64.6%（C）。部署 fallback 是 student-only 模型，教师和投影器不进入推理产物。

## 8. 复现与证据索引

```bash
# 配置和矩阵检查
python experiments/d2/scripts/validate_pair.py

# 预览本轮 12 个运行
python experiments/d2/scripts/run_p1.py --dry-run \
  --only off-s17,off-s29,off-s43,a-s17,a-s29,a-s43,b-s17,b-s29,b-s43,c-s17,c-s29,c-s43

# 归档并重建图表
python experiments/d2/scripts/collect_runs.py runs/detect/d2/p1voc/* --label p1voc
python experiments/d2/scripts/plot_p1_findings.py
```

- P0 链路证据：[`../results/p0_train_ok/`](../results/p0_train_ok/)
- A/B/C/D 梯度标定：[`../results/probe_a_voc_3090/`](../results/probe_a_voc_3090/)
- P1 机器可读汇总：[`../results/p1voc_summary.md`](../results/p1voc_summary.md)
- P1 完整报告：[`p1/findings.md`](p1/findings.md)
- 训练环境：[`../env/README.md`](../env/README.md)
- 风险和 fallback：[`limitations.md`](limitations.md)
