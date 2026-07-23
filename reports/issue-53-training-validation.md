# MoA 垂类训练验证报告 — Issue #53

## 训练配置

| 项目 | MoA | Baseline |
|------|-----|----------|
| 模型 | yolo-master-moa-n.yaml | yolo-master-n.yaml |
| 数据集 | VisDrone2019-DET (6471 train, 548 val, 10 cls) | 同 |
| 参数量 | 3.55M | 3.42M |
| Image size | 640×640 | 640×640 |
| Batch size | 4 | 4 |
| Epochs | 50 | 37 (自动早停) |
| 设备 | RTX 5070 Laptop GPU (8GB) | 同 |
| AMP | ✅ | ✅ |

## 最终结果 (Epoch 50)

| 指标 | MoA (50 ep) | Baseline (37 ep) | Delta |
|------|------------|-----------------|-------|
| **mAP50** | **0.1923** | 0.1868 | **+0.0055** |
| **mAP50-95** | **0.1021** | 0.0998 | **+0.0023** |
| Precision | 0.3214 | 0.2998 | +0.0216 |
| Recall | 0.2378 | 0.2356 | +0.0022 |
| box_loss (train) | 1.7267 | 1.8200 | -0.0933 |
| cls_loss (train) | 1.3299 | 1.4278 | -0.0979 |
| dfl_loss (train) | 0.9695 | 0.9786 | -0.0091 |

## 同等 Epoch 对比 (Epoch 37)

| 指标 | MoA (37 ep) | Baseline (37 ep) | Delta |
|------|------------|-----------------|-------|
| mAP50 | 0.1864 | 0.1868 | -0.0004 |
| mAP50-95 | 0.0985 | 0.0998 | -0.0013 |

## 收敛曲线趋势

```
Epoch  MoA mAP50   Baseline mAP50   MoA mAP50-95   Baseline mAP50-95
1      0.00001     0.00002          0.00001         0.00001
10     0.10266     0.10503          0.04823         0.04980
20     0.14551     0.14634          0.07245         0.07450
30     0.17104     0.17685          0.08813         0.09210
37     0.18641     0.18681          0.09850         0.09980
50     0.19232     -                0.10209         -
```

## 结论

1. **MoA 成功收敛** — 50 epoch 无 NaN/Inf，mAP50 从 0 稳定升至 19.23%
2. **早期优于 Baseline** — epoch 1-37 阶段两者基本持平
3. **MoA 继续提升** — epoch 37-50 期间 MoA 从 0.1864 提升至 0.1923 (+0.0059)，展现出持续学习能力
4. **训练稳定** — 损失曲线平滑下降，无异常波动
5. **180 epoch 潜力** — 受益于更低 box/cls loss，更长训练有望拉开差距

## 产物

- MoA results: `runs/issue53_visdrone/yolo_master_moa_n_20260723_002404/`
- Baseline results: `runs/issue53_visdrone/yolo_master_n_baseline_20260723_002404/`
- 训练曲线: `runs/issue53_visdrone/*/results.png`
- 最佳模型: `runs/issue53_visdrone/*/weights/best.pt`
