# [Issue #54] MoT 消融、同检查点跨域路由解释与低预算混合架构

> 发布前请用真实输出替换所有 `<...>`，删除本提示。不得估算或补写未运行结果。

## 摘要

本实验在 VisDrone 上统一训练 YOLO-Master-EsMoE-N、MoT-N、MoA-N 和新增的 MoT-P5-N，
并使用同一个 MoT-N checkpoint 分析 VisDrone、COCO128、brain-tumor 的路由行为。

- 代码分支/仓库：`<URL>`
- commit：`<SHA>`
- checkpoint SHA-256：`<SHA256>`
- GPU：`<GPU>`
- 协议：`30 epoch / imgsz=640 / seed=42 / FP32`

## 消融结果

| 模型 | mAP50-95 | mAP50 | P50 ms | P95 ms | P99 ms | FLOPs G | Params M | NaN/发散 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EsMoE-N | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| MoT-N | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| MoA-N | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| MoT-P5-N | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |

## 路由解释

### 同检查点跨域结果

粘贴 `routing/cross_domain/recommendations_zh.md` 中经人工核对的观察，并附：

- `routing_probability_heatmap.png`
- `pairwise_statistics.csv`
- `experiment_manifest.json`

说明 bootstrap CI、双侧 permutation p-value 和 Benjamini-Hochberg FDR 的统计单位均为图像。

### VisDrone 场景结果

对 dense/sparse、small/large、irregular_occluded 分别报告三类专家的 mean probability 和 top-1 share。
重点核验 DeformableTransformer 在 irregular_occluded 中的变化是否同时满足：

1. 95% bootstrap CI 不跨 0；
2. FDR q-value ≤ 0.05；
3. effect size 具有实际意义。

## 混合架构结论

MoT-P5 相对 EsMoE 的：

- mAP50-95 差值：`<...>`
- P50/P95 延迟差值：`<...>`
- Params/FLOPs 差值：`<...>`

根据 Issue 标准（mAP > +1% 或 latency < -10%），结论为：`<正协同 / 负结果>`。

## 三条数据支撑的场景建议

1. `<场景、专家、数值、CI/q-value>`
2. `<场景、专家、数值、CI/q-value>`
3. `<模型选择建议、mAP 与 latency 数值>`

## 局限

- brain-tumor 是 OOD 路由审计，不是医疗检测性能实验；
- 30 epoch 结论不代表充分收敛后的上限；
- 路由激活与检测因果关系仍需专家干预或冻结路由消融验证；
- 单硬件 latency 不直接外推到 TensorRT、ncnn 或其他 GPU。

## 复现

```bash
python scripts/run_mot_cross_domain_experiment.py \
  --devices 0 1 2 3 \
  --epochs 30 \
  --no-amp \
  --project runs/mot_cross_domain
```

测试结果：`<pytest summary>`。
