# MoT 同检查点跨域路由审计与低预算混合架构

本目录对应 2026 犀牛鸟 Issue #54 的增量实验方案。它复用仓库已有 MoT/MoA/EsMoE
实现，重点解决一个会影响路由结论可信度的问题：

> 跨数据域比较时固定同一个 MoT checkpoint，只改变输入图像域，避免把“训练数据不同导致的参数差异”
> 误解释成“场景导致的路由差异”。

实验包含完整消融、路由解释、混合架构、统计检验、医疗 TIFF 兼容、稳定性测试和一键编排。
所有结果文件由脚本生成，文档不预填未运行的指标。

完整的问题发现、修复与再实验过程见
[`experiment_journey_zh.md`](experiment_journey_zh.md)。

## 1. 贡献与创新

### 1.1 同检查点跨域审计

`scripts/analyze_mot_cross_domain.py` 在一个进程内只加载一次模型，并为实验清单写入 checkpoint
的 SHA-256。VisDrone、COCO128 和 brain-tumor 均通过这一模型实例，消除 checkpoint 混杂因素。

统计单位是“图像”：先在图像内跨 MoT 层聚合，再做 bootstrap 和 permutation test，避免把大量 token
当成独立样本造成伪显著。

### 1.2 MoT-P5 低计算预算混合架构

新增配置：

```text
ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-p5-n.yaml
```

该结构保留 EsMoE backbone，仅在最低分辨率 P5 neck 放置一个 MoTBlock。它用于检验：

- 全局 Transformer 专家是否能在低分辨率特征上带来增益；
- 相比三尺度完整 MoT，是否能显著降低 Params、FLOPs 和 P95/P99 延迟；
- 若精度未提升，能否用真实数据确定该设计不值得继续扩展。

预构建结果（不含训练指标）：

| 变体 | Params (M) | 静态 FLOPs (G, 640) | MoTBlock |
|---|---:|---:|---:|
| EsMoE-N | 3.450 | 7.999 | 0 |
| MoT-N | 4.055 | 11.281 | 6 |
| MoA-N | 3.577 | 8.273 | 0 |
| **MoT-P5-N** | **3.524** | **8.695** | **1** |

这些数值来自 `--check-build`，完整实验仍须以同一机器上的实际 FLOPs 与 latency 为准。

### 1.3 医疗图像与稳定性

路由脚本支持 `.tif/.tiff`、16-bit 灰度图和浮点科研图像：

1. 有限像素的 0.5/99.5 分位数稳健归一化；
2. 灰度复制为三通道；
3. 保持宽高比的 letterbox，不直接拉伸；
4. 原图不被修改。

此外对水平翻转、亮度降低和亮度升高计算：

- Jensen-Shannon divergence；
- 路由概率 L1 距离；
- 主导专家一致率；
- 归一化路由熵变化。

## 2. 实验矩阵

训练数据统一为 VisDrone，随机种子、epoch、分辨率和增强参数保持一致。

| key | 模型 | 角色 |
|---|---|---|
| `v10` | YOLO-Master-EsMoE-N | MoE 基线 |
| `v10_mot` | YOLO-Master-MoT-N | 完整 MoT |
| `v10_moa` | YOLO-Master-MoA-N | MoA 对照 |
| `v10_mot_p5` | YOLO-Master-MoT-P5-N | 新增低预算混合架构 |

统一记录：

- mAP50-95、mAP50；
- latency mean/P50/P95/P99；
- 实际 FLOPs、Params；
- loss 曲线、NaN 和发散检测；
- 路由 top-1 share、dense probability、entropy、effective experts、margin；
- bootstrap 95% CI、双侧置换检验、Benjamini-Hochberg FDR；
- 扰动前后的路由稳定性。

## 3. 数据

使用仓库官方 YAML 自动下载：

| 数据集 | 用途 | 规模 |
|---|---|---:|
| VisDrone | 四模型训练、验证、密集/稀疏及大小目标场景审计 | 6,471 train / 548 val |
| COCO128 | 通用自然图像域外路由审计 | 128 |
| brain-tumor | 稀疏灰度医疗域外路由审计 | 893 train / 223 val |

brain-tumor 只用于观察固定 VisDrone checkpoint 面对医疗图像时的路由变化，不报告其医疗检测 mAP，
也不提出临床结论。实验无需使用私人线粒体数据；私人数据不得提交到公开仓库。

## 4. 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install seaborn
```

建议使用 4 张 NVIDIA GPU。单卡也能运行，但四个训练任务会按设备列表轮转，不建议并发共享显存。

## 5. 一键运行

### 5.1 端到端 smoke test

先用 COCO128、160 像素和 1 epoch 验证完整链路：

```bash
python scripts/run_mot_cross_domain_experiment.py \
  --smoke \
  --project runs/mot_cross_domain_smoke
```

### 5.2 完整 30 epoch 实验

```bash
python scripts/run_mot_cross_domain_experiment.py \
  --devices 0 1 2 3 \
  --epochs 30 \
  --imgsz 640 \
  --batch 16 \
  --no-amp \
  --project runs/mot_cross_domain
```

四个训练进程并行运行；benchmark 在 GPU 0 串行执行，保证延迟可比。

本机 RTX 5090 + PyTorch 2.9 校准中，四个变体的首个 AMP epoch 都触发非有限梯度保护，
训练器随后恢复健康 checkpoint 并降级到 FP32。正式参考协议因此显式使用 `--no-amp`，
避免隐藏的 epoch 重跑。其他硬件可将 `--amp` 作为单独稳定性实验，但必须报告恢复日志。

### 5.3 分阶段与断点续跑

```bash
# 先下载并校验数据、构建模型
python scripts/run_mot_cross_domain_experiment.py \
  --stages prepare check \
  --project runs/mot_cross_domain

# 四卡训练；若 last.pt 存在则续跑
python scripts/run_mot_cross_domain_experiment.py \
  --stages train \
  --resume \
  --no-amp \
  --project runs/mot_cross_domain

# 串行测速与路由审计
python scripts/run_mot_cross_domain_experiment.py \
  --stages benchmark audit \
  --project runs/mot_cross_domain
```

## 6. 单独运行路由审计

```bash
python scripts/analyze_mot_cross_domain.py \
  --model runs/mot_cross_domain/training/v10_mot/weights/best.pt \
  --domain VisDrone=/path/to/VisDrone/images/val \
  --domain COCO128=/path/to/coco128/images/train2017 \
  --domain brain-tumor=/path/to/brain-tumor/images/val \
  --device 0 \
  --imgsz 640 \
  --max-images 128 \
  --equalize \
  --output runs/mot_cross_domain/routing/cross_domain
```

默认三个域各抽取相同数量图像，并运行三种确定性扰动。图像抽样、bootstrap 和 permutation test 都由
`--seed` 控制。

## 7. 输出

```text
runs/mot_cross_domain/
├── experiment_protocol.json
├── prepared_datasets.json
├── build/
│   └── build_summary.csv
├── training/
│   ├── summary.csv
│   ├── latency_0_640.csv
│   ├── v10/
│   ├── v10_mot/
│   ├── v10_moa/
│   └── v10_mot_p5/
├── routing/
│   ├── cross_domain/
│   │   ├── experiment_manifest.json
│   │   ├── sample_manifest.csv
│   │   ├── routing_detailed.csv
│   │   ├── domain_summary.csv
│   │   ├── pairwise_statistics.csv
│   │   ├── robustness_detailed.csv
│   │   ├── robustness_summary.csv
│   │   ├── routing_probability_heatmap.png
│   │   └── recommendations_zh.md
│   └── visdrone_scenes/
└── logs/
```

`experiment_manifest.json` 中的 `same_checkpoint_for_all_domains` 必须为 `true`，且
`model_sha256` 是所有跨域结论的模型身份。

## 8. 结果判定

混合架构协同增益沿用 Issue #54 标准：

- mAP50-95 绝对提升大于 1%；或
- 相同测量协议下延迟降低大于 10%。

若 MoT-P5 未达到任一标准，应如实报告为负结果。不能通过更换 seed、裁剪不利 epoch 或混用不同硬件
来选择性呈现结果。

路由解释采用以下约束：

- 只有 FDR 校正后 `q <= 0.05` 且 bootstrap CI 不跨 0，才称为稳定跨域差异；
- effect size 与差值必须同时报告，不能只写 p-value；
- DeformableTransformer 激活增加只能说明路由偏好，不证明遮挡检测更准确；
- brain-tumor 与 VisDrone 的语义标签不同，不做跨数据集 mAP 排名。

## 9. 测试

```bash
pytest tests/test_mot_cross_domain_analysis.py tests/test_mot.py -q
ruff check \
  scripts/analyze_mot_cross_domain.py \
  scripts/run_mot_cross_domain_experiment.py \
  scripts/compare_mot_ablation.py \
  tests/test_mot_cross_domain_analysis.py
```

测试覆盖：

- `window_size` 超过特征图、奇数尺寸 shift、eval 禁用 exploration 等既有边界；
- 16-bit TIFF 与灰度三通道转换；
- bootstrap/permutation/FDR 的确定性；
- JSD 与扰动稳定性；
- MoT-P5 配置仅包含一个 MoTBlock。

## 10. 文件索引

| 文件 | 作用 |
|---|---|
| `scripts/run_mot_cross_domain_experiment.py` | 数据准备、四卡训练、串行 benchmark、双路由审计 |
| `scripts/analyze_mot_cross_domain.py` | 同检查点 hook、统计检验、图表、中文观察 |
| `scripts/compare_mot_ablation.py` | 增加 `v10_mot_p5` 与并发安全的 `--no-summary` |
| `yolo-master-mot-p5-n.yaml` | 低预算 P5-only MoT 混合配置 |
| `tests/test_mot_cross_domain_analysis.py` | 新增统计、TIFF、配置测试 |
| `discussion_template_zh.md` | 实验完成后发布 Discussion 的结构化模板 |
| `experiment_journey_zh.md` | 问题、设计、失败、修复与再实验链路 |
