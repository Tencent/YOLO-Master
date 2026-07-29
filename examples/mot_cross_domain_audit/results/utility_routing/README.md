# Utility Routing 脱敏结果

本目录只保留可复核的小型统计，不包含 checkpoint、原始图像、逐图效用矩阵、本地路径或凭据。

| 路径 | 内容 |
|---|---|
| `object_audit/` | 12,296 对目标的序列级统计与实验协议 |
| `layer_screen.csv` | 同一 16 图上的六层强制专家筛选 |
| `utility_matrix_summary.csv` | train/val/test-dev 效用矩阵汇总 |
| `router_iterations.csv` | utility router 成功、失败和保护版本 |
| `utility_router/` | 最终 scene router 的训练、val 与 test-dev 报告 |
| `adaptive_k/` | 同一 128 图的检测指标、三轮原始延迟与实际调用 |

所有结果均对应 checkpoint SHA-256
`a1857c81b7aebd0efb5a56f9d5b37405ef83edcc68890add15c9c480e9fee629`。
解释和限制见 [`../../utility_router_adaptive_k_zh.md`](../../utility_router_adaptive_k_zh.md)。
