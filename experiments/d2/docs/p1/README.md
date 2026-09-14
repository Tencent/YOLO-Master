# P1｜Foundation 蒸馏同预算对照

**状态：✅ VOC 实验完成**

P1 使用 VOC、400 epochs 和三个配对随机种子，对比共享 baseline 与三个 Foundation 蒸馏配置。已完成 12 次运行：off、A、B、C 各 `17`、`29`、`43` 三个 seed。2×2 矩阵中的 D 格依据“至少完成三格”的验收要求跳过。

| 实验格 | 教师 | 蒸馏位置 | `foundation_loss_weight` |
|---|---|---|---:|
| off | 无 | 无 | 0 |
| A | DINOv3 | P4 | 3.3427745950933323 |
| B | DINOv3 | P3+P4+P5 | 3.0016106154954567 |
| C | SigLIP2 | P4 | 3.1106560298222616 |

完整实验结论、配对置信区间和学习曲线见 [`findings.md`](findings.md)。矩阵见 [`../../p1_voc_matrix.csv`](../../p1_voc_matrix.csv)，运行配置见 [`../../configs/p1_voc/`](../../configs/p1_voc/)，逐 epoch 指标和实际解析参数见 [`../../results/`](../../results/)。

## 复现

```bash
# 验证配置和矩阵预算一致
python experiments/d2/scripts/validate_pair.py \
  --configs experiments/d2/configs/p1_voc \
  --matrix experiments/d2/p1_voc_matrix.csv

# 预览本轮完成的 12 个命令
python experiments/d2/scripts/run_p1.py --dry-run \
  --only off-s17,off-s29,off-s43,a-s17,a-s29,a-s43,b-s17,b-s29,b-s43,c-s17,c-s29,c-s43

# 去掉 --dry-run 开始训练；已存在的结果会自动跳过
python experiments/d2/scripts/run_p1.py --device 0 \
  --only off-s17,off-s29,off-s43,a-s17,a-s29,a-s43,b-s17,b-s29,b-s43,c-s17,c-s29,c-s43

# 归档结果并重建报告图
python experiments/d2/scripts/collect_runs.py runs/detect/d2/p1voc/* --label p1voc
python experiments/d2/scripts/plot_p1_findings.py
```

## 设计和前置证据

- [`kd_gradient_analysis.md`](kd_gradient_analysis.md)：权重标定、梯度规模和可学性诊断。
- [`voc_batch.md`](voc_batch.md)：从 COCO 切换到 VOC 的预算原因和运行记录。
- [`../design.md`](../design.md)：实验轴、判读线和归因顺序。
