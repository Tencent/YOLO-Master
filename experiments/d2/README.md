# D2｜Foundation 特征蒸馏现状验证

D2 验证冻结的 Foundation 教师特征能否在同一训练预算下改善小型 `yolo26-master-n`。课题允许负结果，交付重点是可复现配置、机器可读证据、配对统计和明确的结论边界。

| | |
|---|---|
| **基线 commit** | `e9ac08b`（= `upstream/main`） |
| **教师** | `facebook/dinov3-vits16-pretrain-lvd1689m`（冻结，仅训练期） |
| **学生** | `yolo26-master-n`，蒸馏 P4（第 19 层） |
| **P0 状态** | ✅ 已闭环（2026-08-25，CUDA）；证据 [`results/p0_train_ok/`](results/p0_train_ok/) |
| **权重标定** | ✅ 已闭环。最终 VOC 权重为 A `3.34277`、B `3.00161`、C `3.11066`；完整精度见 [`findings.md`](docs/p1/findings.md) |
| **P1 状态** | ✅ **已完成**：VOC A/B/C/off × 3 seeds，共 12 次 400-epoch 运行；结论见 [`findings.md`](docs/p1/findings.md) |

## 我们做了什么

1. **P0 跑通真实训练链路**：确认配置能够注入 teacher、tap、projector 和 Foundation loss，且 KD 项确实进入反向传播目标。
2. **诊断并标定 KD 强度**：原始损失值占比不能代表梯度影响，因此用梯度比探针确定各实验配方的固定权重。
3. **完成 P1 VOC 对照**：off、A、B、C 四组各运行 seeds `17/29/43`，共 12 次 400-epoch 训练。
4. **归档并审计证据**：保存每次运行的逐 epoch CSV、实际解析参数、环境信息，检查预算和同组 seed 配置一致性。

实验使用 VOC、`imgsz=256`、batch 64、SGD、`lr0=0.01`、`pretrained=false`、`amp=false`。训练教师仅在训练期使用，部署产物保留 student。

### Probe A：四格权重标定

正式训练前，在 RTX 3090 上分别对 A/B/C/D 运行 50 batches 的梯度比探针。探针测量未加权 KD 梯度与任务梯度的范数比，并求解让 KD 达到任务梯度 10% 的固定权重：

| 配方 | 教师与层级 | `‖g_kd‖ / ‖g_task‖`（`w=1`） | 10% 目标权重 | 平均梯度 cosine | 冲突 batch |
|---|---|---:|---:|---:|---:|
| A | DINOv3 P4 | 0.02992 | 3.34277 | +0.0122 | 44% |
| B | DINOv3 P3+P4+P5 | 0.03332 | 3.00161 | +0.0118 | 50% |
| C | SigLIP2 P4 | 0.03215 | 3.11066 | -0.0137 | 52% |
| D | SigLIP2 P3+P4+P5 | 0.05381 | 1.85823 | -0.0014 | 50% |

四格的平均 cosine 置信区间均包含 0，说明训练起点没有可检测的系统性同向或反向关系。该探针用于标定优化强度，不是精度实验；D 完成了 Probe A，但没有运行 400-epoch P1 训练。机器可读证据见 [`results/probe_a_voc_3090/`](results/probe_a_voc_3090/)，解释见 [`docs/p1/kd_gradient_analysis.md`](docs/p1/kd_gradient_analysis.md)。

## 结果

主指标是 epoch 400 的 `metrics/mAP50-95(B)`。每个 treatment 与相同 seed 的 off 配对；95% CI 使用三个配对差值的双侧 t 区间。

| 配方 | 教师与蒸馏位置 | 平均 mAP50-95 | 相对 off | 配对 95% CI（mAP 点） | 结论 |
|---|---|---:|---:|---|---|
| off | 无 KD | 0.47295 | — | — | baseline |
| A | DINOv3 P4 | **0.47678** | **+0.383 mAP** | [-1.303, +2.069] | 方向性信号；2/3 seeds 为正，不能宣称稳定提升 |
| B | DINOv3 P3+P4+P5 | 0.47088 | -0.207 mAP | [-1.682, +1.268] | 命中预设 no-go 规则 |
| C | SigLIP2 P4 | 0.47013 | -0.282 mAP | [-0.362, -0.202] | 三个 seeds 均下降，停止 |

![P1 最终三个 seed、均值与置信区间](docs/p1/assets/p1-final-seeds.png)

P1 没有证明 Foundation 特征蒸馏可以稳定涨点。A 的平均值最好，而且收益主要在训练后期出现，但 seed 43 为负、区间很宽，只能保留为待复现信号。B 和 C 没有继续投入的证据。完整逐 seed 数据、学习曲线和统计口径见 [`docs/p1/findings.md`](docs/p1/findings.md)。

这些组使用不同的固定 KD 权重，因此 A/B/C 应被解释为**完整配方之间的比较**。A−B 不能单独归因为尺度效应，A−C 也不能单独归因为教师效应。

## 从零复现

### 1. 安装

```bash
git clone -b d2 https://github.com/and-yliu/YOLO-Master.git
cd YOLO-Master
pip install -e ".[foundation]"
hf auth login
```

DINOv3 是 gated 模型；运行者需要在 Hugging Face 模型页接受许可。训练环境快照见 [`env/README.md`](env/README.md)。

### 2. 验证配置

```bash
python experiments/d2/scripts/validate_pair.py
```

该命令默认检查 [`configs/p1_voc/`](configs/p1_voc/) 和 [`p1_voc_matrix.csv`](p1_voc_matrix.csv)。

### 3. 记录环境并预览命令

```bash
python experiments/d2/scripts/record_environment.py \
  --out experiments/d2/results/environment_p1voc.json

python experiments/d2/scripts/run_p1.py --dry-run \
  --only off-s17,off-s29,off-s43,a-s17,a-s29,a-s43,b-s17,b-s29,b-s43,c-s17,c-s29,c-s43
```

### 4. 训练

```bash
python experiments/d2/scripts/run_p1.py --device 0 \
  --only off-s17,off-s29,off-s43,a-s17,a-s29,a-s43,b-s17,b-s29,b-s43,c-s17,c-s29,c-s43
```

已有完整 `results.csv` 的运行会被跳过。

### 5. 归档和重建图表

```bash
python experiments/d2/scripts/collect_runs.py runs/detect/d2/p1voc/* --label p1voc
python experiments/d2/scripts/plot_p1_findings.py
```

## 复现包

```text
experiments/d2/
├── configs/p1_voc/       P1 训练配置
├── scripts/              验证、环境记录、运行、归档和绘图入口
├── results/              逐 epoch 指标与 resolved args
├── env/                  训练依赖、教师 revision 与硬件快照
├── docs/design.md        最终实验设计与统计口径
├── docs/limitations.md   已知局限与 fallback
├── docs/p0/README.md     P0 链路证据入口
├── docs/p1/findings.md   P1 完整结果报告
└── p1_voc_matrix.csv     15 行预注册矩阵；12 行完成，D 的 3 行标记 skipped
```

关键证据：

- P0 真实训练：[`results/p0_train_ok/`](results/p0_train_ok/)
- A/B/C/D Probe A：[`results/probe_a_voc_3090/`](results/probe_a_voc_3090/)
- P1 汇总：[`results/p1voc_summary.md`](results/p1voc_summary.md)
- P1 结论与曲线：[`docs/p1/findings.md`](docs/p1/findings.md)
- 设计与判读线：[`docs/design.md`](docs/design.md)
- 已知局限：[`docs/limitations.md`](docs/limitations.md)
- PR 四节描述：[`docs/p1/pr_description.md`](docs/p1/pr_description.md)

## 已知边界

- 每格只有三个 seed，A/B 的置信区间很宽。
- B 的 P3/P5 必须与教师网格插值，尺度与插值影响无法分离。
- D（SigLIP2 multiscale）未运行，不能估计完整的教师 × 尺度交互。
- 完整三 seed 对照只在 VOC 上完成，不能直接外推到 COCO。
- 训练日志没有可靠保存精确 Git SHA；教师 revision、配置、解析参数和环境版本已归档。

训练平均墙钟相对 off 增加约 21.8%（A）、22.0%（B）和 64.6%（C）。更完整的限制与降级方案见 [`docs/limitations.md`](docs/limitations.md)。
