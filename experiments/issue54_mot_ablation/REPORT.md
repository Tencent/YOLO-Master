# Issue #54 MoT/MoA 消融实验权威摘要

早期 A40 报告已保存在 `REPORT.legacy.md`。它只依据 router bias 推断场景偏好，
不能替代逐 token、逐场景统计，因此不再作为最终结论来源。

权威实验入口：

- 完整报告：`examples/mot_hybrid_architecture/README.md`
- Discussion 文章：`examples/mot_hybrid_architecture/technical_summary.md`
- 四模型主表：`examples/mot_hybrid_architecture/results/mot_model_comparison.csv`
- 训练曲线：`examples/mot_hybrid_architecture/results/mot_training_curves.csv`
- 路由空间图与统计：`examples/mot_hybrid_architecture/results/routing_interpretability/`

## 受控四模型对比（VisDrone，50 epochs）

| 模型 | mAP50-95 | mAP50 | P50/P95/P99 (ms) | 实际 GFLOPs | Params (M) | NaN/发散 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| EsMoE-N | 0.12324 | 0.22356 | 34.355/35.184/37.371 | 8.671 | 3.450 | 否/否 |
| MoT-N | 0.12081 | 0.22248 | 61.971/63.168/64.935 | 12.270 | 4.055 | 否/否 |
| MoA-N | 0.11933 | 0.21697 | 58.788/59.924/63.708 | 10.072 | 3.577 | 否/否 |
| MoA+MoT-N | 0.11789 | 0.21942 | 63.807/64.557/65.267 | 15.568 | 4.057 | 否/否 |

MoA+MoT 相对 EsMoE 的 mAP50-95 下降 4.34%，P50 延迟增加 85.73%，
没有达到“mAP 提升 >1% 或延迟下降 >10%”的协同增益门槛。

## 路由机制分析（独立 100-epoch MoT checkpoint）

路由分析使用另一轮训练更充分的 MoT checkpoint，覆盖 548 张 VisDrone val 图像
和真实遮挡标签。它与上面的 50-epoch 四模型 run 不是同一 checkpoint，指标不可混用。

`MoTBlock.router` hook 保存三类 expert 的 token 权重，并进行密集/稀疏、小/大目标、
高/低遮挡、尺寸匹配单框 token 对比及 BH-FDR 校正。最强的 Deformable 结果位于
`model.20.m.1`：遮挡均值 0.154569，清晰均值 0.136347，相对提升 13.36%，
`p=8.16e-05`、`q=2.35e-04`（488 对）。该假设在特定层获得支持，但不能推广到
所有层；`model.23.m.1` 已完全坍缩到 LocalConv。

逐 token 空间图：
`examples/mot_hybrid_architecture/results/routing_interpretability/heatmap_spatial.png`。

## 当前验证

边界修复覆盖 window 大于 feature map、奇数尺寸 shift、eval 禁用 exploration，
以及非法 window/n_points、shift 边缘泄漏和 dtype 混合等回归：

```text
144 passed, 1 skipped
```

## 场景建议

1. VisDrone 小模型默认优先 EsMoE-N：精度最高、P50 最低、实际 FLOPs 最少。
2. MoT 不是无成本升级：mAP50-95 下降 1.97%，P50 增加 80.38%。
3. 遮挡目标重点查看 `model.20.m.1`：尺寸匹配后 Deformable 权重 +13.36%。
4. 不要声称所有层都偏 Deformable：最后一个 block 已坍缩到 LocalConv。
5. 当前 MoA+MoT 重型组合无协同增益，不建议作为性能配置提交。
