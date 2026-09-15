## 改动摘要

D2 P1 验证冻结 Foundation 教师特征能否在同预算下改善 `yolo26-master-n`。本 PR 提供 VOC 的 off/A/B/C 三种子对照所需配置和运行脚本、12 组机器可读结果、环境快照、可重复生成的统计图以及结论报告。归档器新增同一实验格内部的参数一致性检查，避免把不同权重的运行合并成多 seed 统计。

## 测试证据

训练环境：Python 3.12.14、PyTorch 2.11.0+cu128、Ultralytics 8.4.101、单张 RTX 3090 24 GB。PR 整理和单测在 macOS arm64 的独立 Python 环境完成。

```bash
python experiments/d2/scripts/validate_pair.py \
  --configs experiments/d2/configs/p1_voc \
  --matrix experiments/d2/p1_voc_matrix.csv
# PASS: config and matrix budgets; no-confound check

pytest -q tests/test_foundation_dinov3.py tests/test_foundation_siglip2.py \
  tests/test_foundation_distill_model.py tests/test_foundation_config.py
# 55 passed

ruff check experiments/d2/scripts/collect_runs.py \
  experiments/d2/scripts/plot_p1_findings.py experiments/d2/scripts/run_p1.py
ruff format --check experiments/d2/scripts/collect_runs.py \
  experiments/d2/scripts/plot_p1_findings.py experiments/d2/scripts/run_p1.py
codespell experiments/d2/README.md experiments/d2/docs/p1/ \
  experiments/d2/scripts/collect_runs.py experiments/d2/scripts/plot_p1_findings.py \
  experiments/d2/scripts/run_p1.py experiments/d2/results/p1voc_summary.md
# PASS
```

归档校验确认 12 个运行均包含 400 epochs，且 seed、teacher、multiscale 和 loss weight 与矩阵一致。图表脚本已从归档 CSV 成功重建三张报告图片。

正式训练权重由 VOC/RTX 3090 上的 Probe A 标定：A/B/C/D 各测量 50 batches，并求解 KD 梯度达到任务梯度 10% 时的权重。四格结果保存在 `experiments/d2/results/probe_a_voc_3090/`；D 仅完成探针，没有纳入正式精度统计。

## 消融数据

VOC、400 epochs、`imgsz=256`、batch 64、SGD、三个配对 seed（17/29/43）；统计单位为每个 seed 的 `treatment - off`，95% CI 使用自由度 2 的双侧 t 区间。

| 配置 | 平均 mAP50-95 | 相对 off | 95% CI | 判读 |
|---|---:|---:|---|---|
| off | 0.47295 | — | — | baseline |
| A：DINOv3 P4 | 0.47678 | +0.383 mAP | [-1.303, +2.069] mAP | 方向性信号，2/3 seeds 为正 |
| B：DINOv3 multiscale | 0.47088 | -0.207 mAP | [-1.682, +1.268] mAP | 命中预设 no-go 规则 |
| C：SigLIP2 P4 | 0.47013 | -0.282 mAP | [-0.362, -0.202] mAP | 三个 seeds 均下降，停止 |

完整逐 seed 数值、曲线和统计口径见 `experiments/d2/docs/p1/findings.md`。

## 已知局限

- 每格只有三个 seed，A/B 的置信区间很宽；A 的 seed 43 为负，不能声称稳定涨点。
- A/B/C 使用不同的固定 KD 权重，因此只能比较完整配方，不能把 A−B 或 A−C 解释为纯尺度或纯教师效应。
- B 的 P3/P5 需要相对教师 P4 网格插值，因此无法完全区分多尺度本身与插值带来的影响。
- 未运行 SigLIP2 multiscale 的 D 格；现有结果不能估计完整的教师 × 尺度交互。
- 未完成完整 COCO 三种子矩阵，VOC 结果不能直接外推到 COCO。
- Foundation 教师只用于训练。部署 fallback 为导出 student-only 模型，不携带教师和投影器。
- 训练日志未保存精确 Git SHA。模型 revision、解析参数和环境版本已归档，但代码身份仍需以最终 PR HEAD 为准。

训练墙钟相对 off 平均增加约 21.8%（A）、22.0%（B）和 64.6%（C）。
