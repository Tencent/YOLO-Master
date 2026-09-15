# P0｜Foundation 蒸馏链路验证

**状态：完成**（CUDA；RTX 3090 复现）

P0 验证仓库现有 teacher、student feature tap、projector 和 relational KD loss 能通过真实训练入口运行，并且 KD 项进入反向传播目标。

## 结果

- Foundation 指标列成功写入逐 epoch CSV。
- raw KD 在三个 epoch 中均为有限非零值。
- `foundation_relational_raw × loss_weight × batch` 与记录的加权 Foundation loss 一致。
- smoke probe 确认梯度到达 student 和 projector，teacher 保持冻结且不进入 optimizer。

这是链路证据。该运行只有 3 epochs、`pretrained=false`，mAP50-95 全程为 0，不构成精度结论。

## 复现

```bash
yolo train model=ultralytics/cfg/models/26/yolo26-master-n.yaml \
  data=coco128.yaml epochs=3 imgsz=256 batch=4 workers=0 device=0 \
  seed=17 deterministic=true pretrained=false amp=false plots=false \
  foundation_enabled=true foundation_teacher=dinov3 \
  foundation_model=facebook/dinov3-vits16-pretrain-lvd1689m \
  foundation_target_levels=p4 foundation_loss=relational \
  foundation_loss_weight=0.05 project=d2/p0 name=train_ok
```

## 证据

- 真实训练指标：[`../../results/p0_train_ok/metrics.csv`](../../results/p0_train_ok/metrics.csv)
- 实际参数与源文件哈希：[`../../results/p0_manifest.csv`](../../results/p0_manifest.csv)
- 梯度和冻结检查：[`../../results/p0_smoke_dinov3.json`](../../results/p0_smoke_dinov3.json)
- 后续权重标定：[`../p1/kd_gradient_analysis.md`](../p1/kd_gradient_analysis.md)
