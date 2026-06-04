# MoE Stable Version Analysis

## 结论

当前建议把 `v0_stable` 作为正式稳定训练入口。该入口采用 `v0.6` 的 `HybridAdaptiveGateMoE`，对应配置文件：

- `/Users/gatilin/PycharmProjects/YOLO-Master-v260601-for-MoE/ultralytics/cfg/models/master/exp/yolo-master-v0_stable.yaml`
- 基线来源：`/Users/gatilin/PycharmProjects/YOLO-Master-v260601-for-MoE/ultralytics/cfg/models/master/exp/yolo-master-v0_6.yaml`

`v0.9` 作为研究候选保留，因为峰值最好；但它在 30 epoch COCO128 筛选中末轮明显回落，暂不适合作为默认稳定训练版本。`v0.10` 模块最丰富，但复杂度和候选框质量风险更高，也不建议作为当前正式入口。

## 模块分层

| 版本 | MoE 模块 | 主要设计 | 稳定性判断 |
|---|---|---|---|
| v0.1 | `ModularRouterExpertMoE` / `OptimizedMOEImproved` | 老版模块化路由与专家 | 参数大，作为历史基线 |
| v0.2 | `UltraOptimizedMoE` | 更激进的高效路由 | 参数大，作为历史基线 |
| v0.3 | `UltimateOptimizedMoE` | 通道拆分、融合专家、动态温度 | 有 NaN/坍缩风险 |
| v0.4 | `AdaptiveGateMoE` | 双流路由、SE split、稳定复杂度估计 | 稳定性改善，但计算不如 v0.6 均衡 |
| v0.5 | `FusedAdaptiveGateMoE` | v0.4 + 全融合专家候选 | 小专家数高效，专家数大时成本偏高 |
| v0.6 | `HybridAdaptiveGateMoE` | 融合专家 + shared inverted 专家混合后端、channel shuffle | 当前稳定版 |
| v0.7 | `LowRankHybridAdaptiveGateMoE` | v0.6 + 低秩融合专家 | 峰值接近 v0.6，但训练失败 |
| v0.8 | `RefinedLowRankHybridAdaptiveGateMoE` | v0.7 + 轻量特征 refinement | 可跑完但效果弱 |
| v0.9 | `DetailAwareLowRankHybridAdaptiveGateMoE` | v0.7 + detail gate | 峰值最高，波动大 |
| v0.10 | `VisualEnhancedAdaptiveGateMoE` | detail gate + pyramid context + refinement | 复杂度最高，暂不稳定 |

## COCO128 30e/320 筛选结果

| 版本 | epoch | best mAP50 | best mAP50-95 | last mAP50 | last mAP50-95 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| v0.3 | 29 | 0.00259 | 0.00077 | 0.00030 | 0.00003 | 后期坍缩，失败风险高 |
| v0.6 | 30 | 0.00283 | 0.00125 | 0.00163 | 0.00068 | 峰值中等，但完整稳定 |
| v0.7 | 23 | 0.00254 | 0.00126 | 0.00043 | 0.00009 | NaN persisted，失败 |
| v0.8 | 30 | 0.00267 | 0.00031 | 0.00023 | 0.00004 | 可跑完但效果弱 |
| v0.9 | 30 | 0.00845 | 0.00423 | 0.00021 | 0.00006 | 高上限，低稳定 |
| v0.10 | 30 | 0.00876 | 0.00091 | 0.00074 | 0.00028 | mAP50 高但定位弱，复杂度高 |

说明：COCO128 从头训练 30 epoch 的绝对 mAP 很低，只适合做相对筛选，不代表最终大数据训练精度。

## 为什么选择 v0.6

`HybridAdaptiveGateMoE` 在浅中层专家数较少时使用融合专家，减少 Python 调度和小 kernel 开销；在高层 16 experts 时切换到 shared inverted 后端，避免把全部专家密集算完。这个结构比 v0.5 更适合 4/8/16 experts 的三层插入方式，也比 v0.7-v0.10 少了低秩、细节门、多尺度上下文带来的额外不确定性。

训练器当前已经具备 MoE 稳定机制：MoE 超参注入、专家 warmup、router 独立学习率、路由坍缩检测与恢复。v0.6 与这些机制配合最好；v0.9/v0.10 虽然局部峰值更高，但在短训筛选里更容易出现后期路由回落。

## 推荐训练入口

正式训练优先使用：

```bash
yolo detect train model=/Users/gatilin/PycharmProjects/YOLO-Master-v260601-for-MoE/ultralytics/cfg/models/master/exp/yolo-master-v0_stable.yaml data=coco.yaml epochs=100 imgsz=640 batch=8
```

继续对比时可使用：

```bash
/usr/bin/python3 scripts/compare_moe_coco128.py --versions v0_stable v0_9 --check-build
```

## 后续研究方向

保留 `v0.9` 作为高上限研究线。下一步如果要把它转成稳定候选，优先处理路由坍缩和后期候选框质量，而不是继续增加视觉模块复杂度。
