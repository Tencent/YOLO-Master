# D2 阶段汇报：Foundation 蒸馏 P1 VOC 对照

## 1. 问题与假设

问题：冻结 Foundation 教师的特征蒸馏，能否在同一 VOC 训练预算下改善小型 `yolo26-master-n`？

假设：DINOv3 或 SigLIP2 的中层空间表征可以约束学生 backbone，使最终 mAP50-95 相对同 seed 的 KD-off baseline 提升。预设 no-go 规则为 `|ΔmAP| < 0.3` 点且配对 95% CI 包含 0。

## 2. Commit、数据、预算、对照与复现

- 当前工作树：分支 `d2`，基础 HEAD `4e2bdb0`，包含尚未提交的 P1 结果更新和待删除的 P2 内容；最终汇报需替换为 PR HEAD。
- 数据：VOC，训练集 16,551 张，验证使用 test2007 4,952 张。
- 学生：`yolo26-master-n`；教师：DINOv3 ViT-S/16 或 SigLIP2 base patch16-256，训练期间冻结。
- 预算：400 epochs、`imgsz=256`、batch 64、SGD、`lr0=0.01`、`pretrained=false`、`amp=false`。
- 对照：共享 off baseline，以及 A=DINOv3 P4、B=DINOv3 P3/P4/P5、C=SigLIP2 P4；每格 seeds 17/29/43。

```bash
python experiments/d2/scripts/validate_pair.py \
  --configs experiments/d2/configs/p1_voc \
  --matrix experiments/d2/p1_voc_matrix.csv

python experiments/d2/scripts/run_p1.py --device 0 \
  --only off-s17,off-s29,off-s43,a-s17,a-s29,a-s43,b-s17,b-s29,b-s43,c-s17,c-s29,c-s43

python experiments/d2/scripts/collect_runs.py runs/detect/d2/p1voc/* --label p1voc
python experiments/d2/scripts/plot_p1_findings.py
```

## 3. 已完成证据

- 12 个运行均完成 400 epochs；`metrics.csv` 与 `resolved_args.yaml` 已归档。
- 配置和矩阵预算检查通过；归档后 resolved args 无混杂检查通过。
- 同一实验格的 teacher、层级和 `foundation_loss_weight` 在三个 seed 间一致。
- A/B/C 三格和共享 baseline 的数字、配对区间与三张曲线图见 [`p1/findings.md`](p1/findings.md)。
- 训练环境：Ultralytics 8.4.101、Python 3.12.14、PyTorch 2.11.0+cu128、RTX 3090 24 GB；见 [`../env/`](../env/README.md)。

| 配置 | seed 17 | seed 29 | seed 43 | 平均 mAP50-95 | 相对 off |
|---|---:|---:|---:|---:|---:|
| off | 0.47239 | 0.47440 | 0.47207 | 0.47295 | — |
| A | 0.48138 | 0.48075 | 0.46821 | 0.47678 | +0.383 mAP |
| B | 0.46349 | 0.47524 | 0.47392 | 0.47088 | -0.207 mAP |
| C | 0.46943 | 0.47135 | 0.46962 | 0.47013 | -0.282 mAP |

![最终 seed 结果](p1/assets/p1-final-seeds.png)

![逐 epoch 配对差值](p1/assets/p1-paired-deltas.png)

## 4. 结论与不确定性

A 的平均提升为 `+0.383 mAP`，但只有 2/3 seeds 为正，配对 95% CI 为 `[-1.303, +2.069] mAP`，因此只能记录为方向性信号。B 平均 `-0.207 mAP` 且 CI 包含 0，命中预设 no-go 规则。C 三个 seeds 均小幅下降，平均 `-0.282 mAP`，当前没有继续投入的正向证据。

实验没有证明 Foundation 蒸馏能够稳定涨点。样本只有三个 seeds，A/B 区间很宽；B 的 P3/P5 依赖插值，使“多尺度本身”和“插值影响”无法完全分离；D 格未运行，因此不能估计完整交互。

## 5. 下一阶段、风险触发线与所需协作

- 本轮以 P1 证据包和 PR 收尾，保留 A 的方向性结果以及 B/C 的负结果。
- 若未来需要对外声称 A 提升，应先增加配对 seeds；风险触发线仍使用预设 `0.3 mAP + 95% CI` 规则，不因当前结果调整。
- 合并前需要删除当前分支中的 P2 配置、脚本和实现改动，并把最终 PR SHA 回填到本报告和环境说明。
- 需要 reviewer 核对训练代码身份未被日志记录这一限制，以及 student-only 导出是否满足部署 fallback。
