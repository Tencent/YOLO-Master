# MoT 同检查点跨域路由审计与低预算混合架构

本目录对应 2026 犀牛鸟 Issue #54 的增量实验方案。它复用仓库已有 MoT/MoA/EsMoE
实现，重点解决一个会影响路由结论可信度的问题：

> 跨数据域比较时固定同一个 MoT checkpoint，只改变输入图像域，避免把“训练数据不同导致的参数差异”
> 误解释成“场景导致的路由差异”。

实验包含完整消融、路由解释、混合架构、序列级统计、真实遮挡配对、医疗 TIFF 兼容、稳定
benchmark 和一键编排。30 epoch 结果、修正后的原始 CSV 和图表见
[`results/README.md`](results/README.md)。

完整的问题发现、修复与再实验过程见
[`experiment_journey_zh.md`](experiment_journey_zh.md)。
目标级匹配、检测效用监督、漂移保护与 adaptive K 的后续链路见
[`utility_router_adaptive_k_zh.md`](utility_router_adaptive_k_zh.md)。

## 0. 与已合并工作的关系

上游 PR [#96](https://github.com/Tencent/YOLO-Master/pull/96) 已提供 MoT/MoA+MoT 配置、
基础消融和边界测试；PR [#146](https://github.com/Tencent/YOLO-Master/pull/146) 已提供
VisDrone 消融及跨域路由报告，并明确记录其 COCO/VisDrone 比较使用了不同训练域的模型。这里
不重复声明这些基础能力，增量聚焦于：

- 用同一个 checkpoint 消除 #146 已指出的模型参数混杂；
- 用视频序列 cluster 修正连续帧伪重复；
- 用原始 occlusion 标注和协变量匹配直接复验遮挡假设；
- 用 P5-only MoT 探索不同于既有 MoA+MoT 的低预算组合；
- 用确定性输入、时长预热、顺序轮换和 3 轮结果量化 benchmark 波动。

## 1. 贡献与创新

### 1.1 同检查点跨域审计

`scripts/analyze_mot_cross_domain.py` 在一个进程内只加载一次模型，并为实验清单写入 checkpoint
的 SHA-256。VisDrone、COCO128 和 brain-tumor 均通过这一模型实例，消除 checkpoint 混杂因素。

基础统计单位是“图像”：先在图像内跨 MoT 层聚合，避免把大量 token 当成独立样本。对于
VisDrone 连续视频帧，再以视频序列为 cluster；场景间有共有序列时使用配对 cluster bootstrap
和 sign-flip permutation，避免把相邻帧当作独立重复。

脚本同时审计精确图像重叠和 cluster 重叠。若组间既不能独立比较也不能按共有 cluster 配对，
只输出描述统计并标记 `comparison_valid=false`。显著性必须同时满足 BH-FDR 和 bootstrap CI。

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

### 1.4 真实遮挡标注与协变量平衡

`scripts/prepare_mot_routing_scenes.py` 可读取 VisDrone 原始 8 列标注中的 occlusion 字段。它在
每个视频序列内选择一对低/高遮挡图，并匹配目标数量和中位框面积；全局平衡步骤避免把“更遮挡”
误写成“目标更多或更小”。本轮得到 25 对序列，目标数和面积 SMD 分别为 0.027 和 -0.017。

### 1.5 可复验 benchmark

benchmark 使用固定 seed 的局部随机生成器，不修改全局 RNG 或梯度模式；每个模型每轮至少
预热 50 次且不少于 2 秒，执行 3 轮并轮换模型顺序。汇总值取各轮 percentile 的中位数，同时
保留 run min/max。

### 1.6 检测效用路由与 Adaptive K

新增目标框内路由审计、单层强制专家反事实、utility router、KL 漂移保护和 inference-only
adaptive K。`K=max` 保持原逐 token Top-K，只有低 K 样本收缩到图像级专家池；调度统计记录
实际专家-样本调用，而不是理论 K。

当前结果是受控负结论：目标层调用最多下降 50.52%，但 128 图 mAP50-95 下降 0.00048，P50
未改善；utility router 也未通过独立 test-dev 泛化。代码默认关闭这些实验能力。

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
- 图像或视频序列级 bootstrap 95% CI、双侧置换检验、Benjamini-Hochberg FDR；
- 扰动前后的路由稳定性。

## 3. 数据

使用仓库官方 YAML 自动下载：

| 数据集 | 用途 | 规模 |
|---|---|---:|
| VisDrone | 四模型训练、验证、场景及真实遮挡审计 | 6,471 train / 548 val |
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

四个训练进程并行运行；benchmark 在 GPU 0 串行执行 3 轮，轮间旋转模型顺序。

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

# 三轮串行测速、序列级场景审计与真实遮挡审计
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

### 6.1 目标审计、效用路由与 Adaptive K

```bash
# 目标级匹配审计
python scripts/analyze_mot_object_causal.py \
  --model runs/mot_cross_domain/training/v10_mot/weights/best.pt \
  --dataset /path/to/VisDrone \
  --max-images 0 \
  --output runs/mot_object_causal

# 对目标层构建强制专家检测效用矩阵
python scripts/build_mot_detection_utility.py \
  --model runs/mot_cross_domain/training/v10_mot/weights/best.pt \
  --data ultralytics/cfg/datasets/VisDrone.yaml \
  --split train \
  --layer model.14.m.0 \
  --max-images 2048 \
  --output runs/mot_detection_utility/train_l14_m0_2048

# 冻结检测器与专家，只训练 utility router
python scripts/train_mot_utility_router.py \
  --model runs/mot_cross_domain/training/v10_mot/weights/best.pt \
  --data ultralytics/cfg/datasets/VisDrone.yaml \
  --matrix runs/mot_detection_utility/train_l14_m0_2048/detection_utility_matrix.csv \
  --split train \
  --layer model.14.m.0 \
  --enable-scene-head \
  --output runs/mot_utility_router/l14_m0_scene_2048

# 同图比较 mAP、三轮延迟和实际专家调用
python scripts/benchmark_mot_adaptive_k.py \
  --model runs/mot_cross_domain/training/v10_mot/weights/best.pt \
  --data ultralytics/cfg/datasets/VisDrone.yaml \
  --matrix runs/mot_detection_utility/val_l14_m0_128/detection_utility_matrix.csv \
  --router-bundle runs/mot_utility_router/l14_m0_scene_2048/utility_router.pt \
  --split val \
  --layer model.14.m.0 \
  --blend-alpha 0.4 \
  --rounds 3 \
  --output runs/mot_adaptive_k/val
```

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
│   ├── latency_rounds_0_640.csv
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
│   │   ├── sample_overlap.csv
│   │   ├── robustness_detailed.csv
│   │   ├── robustness_summary.csv
│   │   ├── routing_probability_heatmap.png
│   │   ├── routing_layer_probability_delta_heatmap.png
│   │   ├── routing_layer_top1_share_heatmap.png
│   │   └── recommendations_zh.md
│   ├── visdrone_scenes/
│   └── visdrone_occlusion/
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
- VisDrone 以视频序列为 cluster；有共有序列时使用配对检验；
- `comparison_valid=false` 表示没有合法的独立或配对推断设计；
- effect size 与差值必须同时报告，不能只写 p-value；
- DeformableTransformer 激活增加只能说明路由偏好，不证明遮挡检测更准确；
- brain-tumor 与 VisDrone 的语义标签不同，不做跨数据集 mAP 排名。

### 8.1 本轮最终判定

- 当前代码复验中，MoT-P5 相比完整 MoT：mAP50-95 增加 0.129 个百分点，P50 降低
  45.01%，FLOPs 降低 32.22%；
- MoT-P5 相比 EsMoE：mAP50-95 增加 0.099 个百分点，P50 增加 28.03%；
- 结论：MoT-P5 是完整 MoT 的低预算替代方案，但未证明相对 EsMoE 的协同；
- 图像级场景差异在视频序列配对后均未通过 FDR+CI；
- 25 对图像级真实遮挡复验中，Deformable 激活未显著上升；
- 更细的目标级匹配复验显示遮挡产生层相关组合重分配，不支持“统一切到 Deformable”；
- utility router 未通过独立 test-dev，KL guard 能回退基线但不能创造增益；
- adaptive K 将单层实际调用降低 50.52%，未达到 10% 端到端延迟改善标准。

## 9. 测试

```bash
pytest \
  tests/test_mot_audit_results.py \
  tests/test_mot_cross_domain_analysis.py \
  tests/test_mot.py \
  tests/test_mot_object_causal.py \
  tests/test_mot_detection_utility.py \
  tests/test_mot_utility_router.py \
  tests/test_mot_utility_evaluation.py \
  tests/test_mot_utility_deployment.py \
  tests/test_mot_adaptive_benchmark.py \
  -q
ruff check \
  scripts/analyze_mot_cross_domain.py \
  scripts/run_mot_cross_domain_experiment.py \
  scripts/compare_mot_ablation.py \
  scripts/prepare_mot_routing_scenes.py \
  tests/test_mot_cross_domain_analysis.py
```

测试覆盖：

- `window_size` 超过特征图、奇数尺寸 shift、eval 禁用 exploration 等既有边界；
- 16-bit TIFF 与灰度三通道转换；
- 图像级和视频序列配对 bootstrap/permutation/FDR 的确定性；
- 原始 VisDrone 遮挡字段解析、序列内配对与协变量平衡；
- 确定性 benchmark 输入、梯度状态保持和多轮汇总；
- JSD 与扰动稳定性；
- MoT-P5 配置仅包含一个 MoTBlock。
- 公开结果表与逐 epoch、逐轮 latency、mAP 和 utility guard 原始证据一致，且不含本地路径或凭据。

## 10. 文件索引

| 文件 | 作用 |
|---|---|
| `scripts/run_mot_cross_domain_experiment.py` | 数据准备、四卡训练、三轮 benchmark、三类路由审计 |
| `scripts/analyze_mot_cross_domain.py` | 同检查点 hook、序列级统计、图表、中文观察 |
| `scripts/prepare_mot_routing_scenes.py` | 场景划分、真实遮挡解析与序列内协变量匹配 |
| `scripts/compare_mot_ablation.py` | MoT-P5、checkpoint 指纹、实际 FLOPs 和多轮 latency |
| `scripts/analyze_mot_object_causal.py` | 目标框投影、遮挡匹配、序列级统计 |
| `scripts/build_mot_detection_utility.py` | 单层强制专家检测效用矩阵 |
| `scripts/train_mot_utility_router.py` | 冻结检测器的序列隔离 utility-router 训练 |
| `scripts/evaluate_mot_utility_router.py` | 独立 split、信任混合与 KL 漂移保护 |
| `scripts/benchmark_mot_adaptive_k.py` | mAP、三轮延迟与实际调度联合复验 |
| `yolo-master-mot-p5-n.yaml` | 低预算 P5-only MoT 混合配置 |
| `tests/test_mot_cross_domain_analysis.py` | 统计、TIFF、遮挡配对、配置与 benchmark 测试 |
| `tests/test_mot_audit_results.py` | 公开汇总与原始 CSV/JSON 的一致性及脱敏检查 |
| `discussion_template_zh.md` | 已回填真实结果的 GitHub Discussion 发布草稿 |
| `pr_description_zh.md` | 上游 Pull Request 说明草稿 |
| `experiment_journey_zh.md` | 问题、设计、失败、修复与再实验链路 |
| `utility_router_adaptive_k_zh.md` | 目标审计到 adaptive K 的完整实验链 |
| `results/` | 脱敏后的原始 CSV、图表、协议与正式结论 |
