# MoT 真动态专家 548 张证据摘要

完整原始证据包未直接展开进 Git 历史，由以下哈希锚定：

```text
mot_dynamic_full_val_20260912_evidence.zip
bytes  16182
SHA256 0B05BACB5FAB5A91ACABE4211A6CF6852ABD01B7C37588E04A874E7BAE18CC72

mot_dynamic_route_margin_20260912_evidence.zip
bytes  18728
SHA256 818607439F165D63369A1839F365BCC3C266D8EBD5F828759073C4D78BD2C63D

mot_dynamic_deterministic_topk_20260912_235511_evidence.zip
bytes  20503
SHA256 4A379E1FD07AA334442DE38F6F2D8128CB6CCEDA02D87F3BAA647AA326F9ABE6
```

本目录的 `full_val_summary.json` 是从原始 `DYNAMIC_MOT_FULL_VAL_THOP_FIXED.json` 提取的审查尺寸摘要。
完整 ZIP 本地归档于：

```text
D:/YOLO_Master/P1_dynamic_runtime_20260912/mot_dynamic_full_val_20260912_evidence.zip
D:/YOLO_Master/P1_dynamic_runtime_20260912/mot_dynamic_route_margin_20260912_evidence.zip
D:/YOLO_Master/P1_dynamic_runtime_20260912/mot_dynamic_deterministic_topk_20260912_235511_evidence.zip
```

第二个证据包完成了 173 个漂移位置的 Top-K 边界裕量审计：两个漂移块的错位裕量最大值均未超过该块
dense 概率跨后端最大绝对误差，支持“近似并列浮点翻转”结论。结论和限制见
`reports/a3_dynamic_mot_full_val_20260912.md`。

第三个证据包是在提交 `d7c0085` 上用 `max_deadband_then_lowest_expert_id`（deadband=`1e-6`）重新导出
6 个真实 MoT 块后的 548 张 GPU 复验。精度门禁和动态执行门禁通过，但严格路由门禁失败：漂移为
`196 / 3,945,600`，集中于 `model.14.m.0`（174）和 `model.20.m.0`（22）。错位裕量落在
`8.05e-7` 到 `1.19e-6`，说明固定 deadband 把离散边界移到了 `1e-6` 附近，并未消除跨后端不连续性。
仓库中的 `deterministic_topk_full_val_summary.json` 保留本轮审查尺寸摘要；不得把本轮写成严格一致通过。
