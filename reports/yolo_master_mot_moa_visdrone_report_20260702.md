# YOLO-Master MoT/MoA VisDrone 消融实验正式报告

## 摘要

本报告基于 VisDrone DET 数据集，对 YOLO-Master v0.10 系列中的 MoE 基线、MoT 模块、MoA 模块以及 MoA+MoT 混合结构进行消融实验。实验完成了四个模型变体的 50 epoch 训练、精度评估、实际 FLOPs 和参数量统计、延迟基准测试、训练稳定性检查，以及 MoTBlock 专家路由行为分析。

主要结论如下：

- `YOLO-Master-v0.10-MoA-MoT-N` 获得本轮最高精度，mAP50-95 为 0.12158，mAP50 为 0.22157。
- 相比 `YOLO-Master-EsMoE-N`，MoA+MoT 的 mAP50-95 绝对提升为 +0.00135，相对提升为 +1.12%，但 P50 latency 增加 +67.9%。
- `YOLO-Master-v0.10-MoT-N` 单独使用时未带来 mAP50-95 增益，且 latency 成本较高。
- 四个模型训练均未出现 NaN 或 loss 发散。
- MoT 路由分析显示，专家激活更接近“层位置相关”：早中层更偏向 DeformableTransformer，末端 MoTBlock 更偏向 WindowTransformer；当前数据不支持“密集、小目标或不规则目标场景下 DeformableTransformer 激活显著上升”的结论。

## 实验目标

本次实验围绕以下问题展开：

1. 在 VisDrone 数据集上比较 `YOLO-Master-EsMoE-N`、`YOLO-Master-v0.10-MoT-N`、`YOLO-Master-v0.10-MoA-N` 三类核心变体。
2. 额外评估 MoA+MoT 混合结构，判断组合是否产生协同增益。
3. 测量每个模型的 mAP50-95、mAP50、Latency P50/P95/P99、实际 FLOPs、Params 和训练稳定性。
4. 对 MoTBlock 中 `LocalConvTransformer`、`WindowTransformer`、`DeformableTransformer` 的路由行为进行解释性分析。
5. 补充 MoT 边界测试并修复训练或验证中暴露的稳定性问题。

## 实验环境

实验运行在 K8s 训练系统中，使用单节点 8 卡 H200：

| 项目 | 配置 |
| --- | --- |
| 工作目录 | `/jpfs/huangyidan3/Rhinoceros——Bird/YOLO-Master-main` |
| 实验目录 | `runs/mot_ablation_k8s/20260702_144333` |
| 数据集 | VisDrone DET |
| 训练集 | 6471 images |
| 验证集 | 548 images |
| GPU | 8 x NVIDIA H200 |
| imgsz | 640 |
| epochs | 50 |
| batch | 64 |
| workers | 8 |
| AMP | disabled |
| PyTorch | 2.9.0+cu128 |

关键脚本和配置：

- `scripts/compare_mot_ablation.py`: 模型构建、训练、FLOPs、latency 和 summary 汇总。
- `scripts/run_mot_ablation_k8s.sh`: K8s 内部实验入口脚本。
- `scripts/analyze_mot_routing.py`: MoT 路由 hook、专家激活统计和热力图生成。
- `k8s/run_yolo_master_mot_ablation.yaml`: K8s TrainingJob 配置。
- `ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml`: MoT 配置。
- `ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml`: MoA+MoT 混合配置。

## 模型变体

| Key | 模型 | 说明 |
| --- | --- | --- |
| `v010_esmoe` | YOLO-Master-EsMoE-N | v0.10 MoE 基线 |
| `v010_mot` | YOLO-Master-v0.10-MoT-N | MoT 实验模块 |
| `v010_moa` | YOLO-Master-v0.10-MoA-N | MoA 对比组 |
| `v010_moa_mot` | YOLO-Master-v0.10-MoA-MoT-N | MoA+MoT 混合结构 |

## 评估方法

精度使用 VisDrone validation split 评估，指标为 mAP50-95 和 mAP50。FLOPs 使用 `thop` 在 640x640 输入上统计。Latency 使用合成单图输入在 `cuda:0` 上进行 10 次 warmup 和 100 次重复测试，报告 P50/P95/P99。训练稳定性通过 `results.csv` 扫描 loss 是否出现 NaN/Inf，以及最终 loss 是否出现明显发散。

路由分析使用 `v010_mot/weights/best.pt`，在 VisDrone val 前 512 张图像上运行。脚本 hook 每个 MoTBlock router，记录每个专家的平均权重和 top-1 token 占比，并按场景标签聚合：

- 密度：dense / medium / sparse。
- 尺度：small / mixed / large。
- 形状：regular / irregular。
- 遮挡：当前为 unknown，因为准备后的数据集只保留 YOLO labels，没有原始 VisDrone occlusion 字段。

## 主结果

| Variant | mAP50-95 | mAP50 | Latency P50/P95/P99 ms | GFLOPs | Params M | Stability |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO-Master-EsMoE-N | 0.12023 | 0.21868 | 13.534 / 14.065 / 17.295 | 7.850 | 3.450 | no NaN, no divergence |
| YOLO-Master-v0.10-MoT-N | 0.12011 | 0.21938 | 22.025 / 22.460 / 24.303 | 8.813 | 4.055 | no NaN, no divergence |
| YOLO-Master-v0.10-MoA-N | 0.12019 | 0.21893 | 20.515 / 20.860 / 21.621 | 8.121 | 3.577 | no NaN, no divergence |
| YOLO-Master-v0.10-MoA-MoT-N | 0.12158 | 0.22157 | 22.720 / 23.299 / 24.161 | 8.743 | 3.897 | no NaN, no divergence |

相对 EsMoE 基线的变化：

| Variant | Delta mAP50-95 | Delta mAP50-95 Relative | Delta mAP50 | Delta P50 Latency |
| --- | ---: | ---: | ---: | ---: |
| MoT | -0.00012 | -0.10% | +0.00070 | +62.7% |
| MoA | -0.00004 | -0.03% | +0.00025 | +51.6% |
| MoA+MoT | +0.00135 | +1.12% | +0.00289 | +67.9% |

## 结果解读

`v010_moa_mot` 是本轮最优精度模型，但收益较小。若“有效协同增益”定义为 mAP50-95 相对提升超过 1%，MoA+MoT 可视为刚刚达标；若定义为绝对 mAP 提升超过 1 个百分点，则本轮结果不达标。与此同时，MoA+MoT 的 P50 latency 从 13.534 ms 增加到 22.720 ms，部署代价明显。

`v010_mot` 单独使用时，mAP50 相比基线提升 +0.00070，但 mAP50-95 略降 -0.00012，说明该配置未在更严格 IoU 范围上获得稳定收益。考虑到其 P50 latency 增加 +62.7%，不建议作为默认替换方案。

`v010_moa` 的精度与基线基本持平，mAP50-95 仅低 -0.00004，但 latency 增加 +51.6%。从本轮 VisDrone 设置看，MoA 单独引入不具备明显性价比优势。

## MoT 路由分析

路由分析共处理 512 张验证图像，捕获 6 个 MoTBlock，共生成 3072 条路由记录和 432 张专家热力图。输出文件位于：

- `runs/mot_ablation_k8s/20260702_144333/routing/routing_records.csv`
- `runs/mot_ablation_k8s/20260702_144333/routing/routing_summary_by_scene.csv`
- `runs/mot_ablation_k8s/20260702_144333/routing/heatmaps/`

按模块统计结果：

| Module | Local mean | Window mean | Deformable mean | Top expert |
| --- | ---: | ---: | ---: | --- |
| `model.14.m.0` | 0.000 | 0.435 | 0.565 | Deformable |
| `model.14.m.1` | 0.230 | 0.252 | 0.519 | Deformable |
| `model.20.m.0` | 0.428 | 0.001 | 0.572 | Deformable |
| `model.20.m.1` | 0.495 | 0.000 | 0.505 | Deformable |
| `model.23.m.0` | 0.000 | 0.501 | 0.499 | Window |
| `model.23.m.1` | 0.000 | 0.500 | 0.500 | Window |

该结果说明，MoT 的专家选择主要受层级位置影响：`model.14` 和 `model.20` 中的 MoTBlock 更偏向 DeformableTransformer，`model.23` 中的 MoTBlock 更偏向 WindowTransformer。

按场景聚合后，DeformableTransformer 的平均权重差异很小：

| 分组维度 | Deformable mean range |
| --- | ---: |
| dense / medium / sparse | 0.000009 |
| small / mixed / large | 0.000050 |
| regular / irregular | 0.000013 |

因此，本轮实验不能证明 DeformableTransformer 在密集、小目标或不规则目标场景中有显著更高激活。遮挡场景无法做强结论，因为当前准备后的 VisDrone 数据缺少原始 occlusion annotation。

## 场景化建议

1. 精度优先且可以接受延迟上升时，优先选择 `YOLO-Master-v0.10-MoA-MoT-N`。该模型 mAP50-95 为 0.12158，mAP50 为 0.22157，均为四组最高。
2. 延迟敏感场景应保留 `YOLO-Master-EsMoE-N`。它的 P50/P95/P99 latency 为 13.534/14.065/17.295 ms，显著低于其他三组，同时 mAP50-95 与单独 MoT、MoA 基本持平。
3. 不建议在当前 VisDrone DET 设置下单独部署 `YOLO-Master-v0.10-MoT-N`。它的 mAP50-95 比基线低 0.00012，P50 latency 却增加 62.7%。
4. MoT 的路由解释应优先按层级分析，而不是直接按场景归因。早中层 Deformable mean 为 0.505-0.572，末端 Window mean 约 0.500，场景分组差异则小于 0.002。
5. 若后续要验证遮挡场景假设，必须保留原始 VisDrone annotation 中的 occlusion 字段，并将其传入 `scripts/analyze_mot_routing.py --visdrone-ann-dir`。

## 稳定性与边界修复

本轮补充并验证了 MoT 相关边界测试，覆盖：

- `MoTBlock` 在 `window_size` 大于 feature map 时的降级处理。
- `_WindowTransformerExpert` shift 操作在奇数尺寸输入时的边界。
- MoT `exploration_eps` 在 eval 模式下正确禁用。

测试结果：

```text
tests/test_mot.py tests/test_moa.py: 21 passed
```

训练过程中还暴露并修复了 DDP final validation 路径中的两个稳定性问题：

- `ultralytics/engine/validator.py` 缺少 `LOCAL_RANK` 和 `torch_distributed_zero_first` 导入。
- `validator.py` 调用了未定义的 `convert_ndjson_to_yolo_if_needed`，已补充 YAML 原样返回、NDJSON 转换的兼容实现。

这些修复使 8 卡 DDP 训练能够完成 final validation 和 summary 汇总。

## PR 建议

建议拆成两个 PR：

1. 稳定性与测试 PR。包含 `validator.py` DDP final validation 修复、MoT 边界测试补全。这部分是明确 bugfix，建议优先提交。
2. 实验配置 PR。包含 `yolo-master-moa-mot-n.yaml` 和相关脚本。该 PR 需要明确说明：MoA+MoT 在本轮 VisDrone 上有 +1.12% 相对 mAP50-95 提升，但 latency 上升 +67.9%。如果维护者接受相对提升作为有效增益标准，可以提交；如果要求绝对 +1 mAP point，则建议先继续优化。

## 局限性

- 本轮实验只覆盖 VisDrone DET train/val，未覆盖 COCO。
- Latency 是单图合成输入的模型前向 benchmark，不包含真实 dataloader、预处理和后处理全链路成本。
- 当前 VisDrone 数据准备流程未保留原始 occlusion 字段，因此无法验证遮挡场景下 DeformableTransformer 激活是否显著上升。
- 只完成一次 seed=42 的训练，未做多 seed 方差分析。
- MoA+MoT 精度提升较小，仍需进一步验证统计稳定性。

## 产物清单

核心结果文件：

- `runs/mot_ablation_k8s/20260702_144333/summary.csv`
- `runs/mot_ablation_k8s/20260702_144333/build_summary.csv`
- `runs/mot_ablation_k8s/20260702_144333/latency_0,1,2,3,4,5,6,7_640.csv`
- `runs/mot_ablation_k8s/20260702_144333/routing/routing_records.csv`
- `runs/mot_ablation_k8s/20260702_144333/routing/routing_summary_by_scene.csv`
- `runs/mot_ablation_k8s/20260702_144333/routing/heatmaps/`

模型权重：

- `runs/mot_ablation_k8s/20260702_144333/v010_esmoe/weights/best.pt`
- `runs/mot_ablation_k8s/20260702_144333/v010_mot/weights/best.pt`
- `runs/mot_ablation_k8s/20260702_144333/v010_moa/weights/best.pt`
- `runs/mot_ablation_k8s/20260702_144333/v010_moa_mot/weights/best.pt`

报告文件：

- `runs/mot_ablation_k8s/20260702_144333/technical_summary.md`
- `reports/yolo_master_mot_moa_visdrone_report_20260702.md`

## 结论

本次 VisDrone 消融实验已经完成训练、评估、稳定性检查和路由解释分析。MoA+MoT 是当前最优精度配置，但其延迟代价较大，暂不建议作为默认部署模型。MoT 的路由行为在本轮实验中呈现明显层级差异，但未呈现场景分组上的显著差异。后续若要进一步确认 MoT 对遮挡和不规则目标的价值，应保留原始 VisDrone annotation，并进行多 seed、多数据集复现实验。
