# MoA 垂类训练验证报告 — Issue #53

## 训练配置

| 项目 | MoA | Baseline (MoE) |
|------|-----|----------|
| 模型 | yolo-master-moa-n.yaml | yolo-master-n.yaml |
| 数据集 | VisDrone2019-DET (6471 train, 548 val, 10 cls) | 同 |
| 参数量 | 3.55M | 3.42M |
| Image size | 640×640 | 640×640 |
| Batch size | 4 | 4 |
| Epochs | 50 | 50 |
| 设备 | RTX 5070 Laptop GPU (8GB) | 同 |
| AMP | ✅ | ✅ |

## 最终结果 (Epoch 50)

| 指标 | MoA | Baseline (MoE) | Delta |
|------|-----|----------|-------|
| **mAP50** | 0.1923 | **0.1982** | -0.0059 |
| **mAP50-95** | 0.1021 | **0.1065** | -0.0044 |
| Recall | 0.2378 | 0.2422 | -0.0044 |
| box_loss (train) | 1.7267 | 1.6486 | +0.0781 |
| cls_loss (train) | 1.3299 | 1.2534 | +0.0765 |

## 收敛曲线对比

```
Epoch  MoA mAP50   Baseline mAP50   MoA mAP50-95   Baseline mAP50-95
1      0.00001     0.00001          0.00001         0.00001
10     0.10266     0.10735          0.04823         0.05200
20     0.14551     0.15288          0.07245         0.08015
30     0.17104     0.17982          0.08813         0.09588
40     0.18836     0.19391          0.09812         0.10343
50     0.19232     0.19818          0.10209         0.10654
```

## 结论

1. **两者均成功收敛** — 50 epoch 无 NaN/Inf，损失曲线平滑下降
2. **Baseline (MoE) 略优** — mAP50 高 **0.59pp**，mAP50-95 高 **0.44pp**
3. **MoE 收敛更快** — 全程 MoE 在 mAP 上保持小幅领先
4. **MoA 训练正常** — router 辅助损失稳定收敛，注意力路由在 VisDrone 上可用
5. **MoA 验证了可行性**：注意力路由机制在真实数据集上可正常训练，但在当前规模下未超越 MoE FFN 路由策略

## 产物

- MoA results: `runs/issue53_visdrone/yolo_master_moa_n_20260723_002404/`
- Baseline results: `runs/issue53_visdrone/yolo_master_n_baseline_20260723_002404/`
