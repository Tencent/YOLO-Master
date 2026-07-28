# [Issue #54] MoT 消融、同检查点跨域路由解释与低预算混合架构

本文是可直接发布到 GitHub Discussion 的技术总结草稿。数据来自公开结果包，不包含未运行的估计值。

## 摘要

本实验在 VisDrone 上统一训练 YOLO-Master-EsMoE-N、MoT-N、MoA-N 和新增的 MoT-P5-N，
并使用同一个 MoT-N checkpoint 分析 VisDrone、COCO128、brain-tumor 的路由行为。

- 代码分支：`blues-kun/YOLO-Master:rhino-2026/issue-54-medical-routing`
- 训练 commit：`58cb439407e2f5f7a6e1c4b6a3a9382499713e88`
- 分析 commit：`b22af3b2d03b7f3262c2ceb0cb6c2207a779c1ac`
- checkpoint SHA-256：`a1857c81b7aebd0efb5a56f9d5b37405ef83edcc68890add15c9c480e9fee629`
- GPU：`4 x NVIDIA GeForce RTX 5090 32GB`
- 协议：`30 epoch / imgsz=640 / seed=42 / FP32`

## 消融结果

| 模型 | mAP50-95 | mAP50 | P50 ms | P95 ms | P99 ms | FLOPs G | Params M | NaN/发散 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EsMoE-N | 0.09014 | 0.17446 | 18.563 | 18.671 | 18.959 | 8.505 | 3.420 | 否 |
| MoT-N | 0.08912 | 0.16984 | 47.467 | 47.714 | 51.414 | 12.503 | 4.025 | 否 |
| MoA-N | 0.08710 | 0.16837 | 40.650 | 40.870 | 43.381 | 9.102 | 3.546 | 否 |
| MoT-P5-N | **0.09164** | 0.17309 | **19.628** | **19.707** | **20.539** | **8.632** | **3.494** | 否 |

四组均从 YAML 初始化，在相同预算下训练；最佳指标均出现在第 30 轮。该协议用于架构控制对比，
不代表充分收敛后的 SOTA 性能。

## 路由解释

### 同检查点跨域结果

同一 MoT checkpoint 从 VisDrone 切换到 brain-tumor 后，Deformable top-1 share 增加
0.009824，mean probability 增加 0.000620（95% CI `[0.000467, 0.000774]`，
`q=0.00035`）。差异统计单位为图像，但绝对幅度小；这是 OOD 路由信号，不是医疗检测结论。

水平翻转和亮度扰动下，主导专家一致率为 96.74% 至 99.87%，平均 JSD 不超过 `5.30e-7`。
结合路由概率接近均匀，这种稳定性不能单独作为专家质量证据。

### VisDrone 场景结果

互斥的 dense/sparse 比较中，Deformable mean probability 增加 0.000590
（95% CI `[0.000451, 0.000734]`, `q=0.00105`），top-1 share 增加 0.004937
（`q=0.04684`）。互斥的 large/small 比较中，Window top-1 share 增加 0.004537
（`q=0.03474`），但 mean probability 未通过 FDR。

初次统计把 `irregular_occluded` 与 dense 作为独立样本比较；指纹审计发现两组共享 104/128 张图。
修复后该比较被标记为无效。因此本实验没有验证“遮挡场景优先激活 Deformable”这一原假设。

## 混合架构结论

MoT-P5 相对完整 MoT，mAP50-95 增加 0.252 个百分点，P50/P95/P99 分别降低
58.65%/58.70%/60.05%，FLOPs 降低 30.96%。相对 EsMoE，mAP50-95 仅增加 0.150 个百分点，
P50 增加 5.74%。

Issue 的“mAP 提升 > 1%”若按相对比例计算为 1.66%，若按绝对百分点计算则只有 0.150 pp；
P50 没有降低。它是完整 MoT 的有效低预算替代方案，但在维护者确认阈值口径前，不声称相对 EsMoE
已产生正协同。

## 三条数据支撑的场景建议

1. 资源受限且确需 MoT 时，优先 MoT-P5：相比完整 MoT，P50 降低 58.65%、FLOPs 降低
   30.96%，mAP50-95 反而增加 0.252 个百分点。
2. 稀疏航拍只显示弱 Deformable 偏好：mean probability 增加 0.000590、top-1 share
   增加 0.004937，不能称为强专家专业化。
3. 小目标出现弱 Window argmax 偏移：top-1 share 增加 0.004537，但 mean probability
   不显著。
4. 遮挡结论暂缓：代理组与 dense 重叠 81.25%，需互斥或配对实验重新验证。

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

测试结果：`82 passed, 4 warnings`；warning 来自既有 MoA head channel adjustment。
