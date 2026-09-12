# MoT 真动态专家 548 张证据摘要

完整原始证据包未直接展开进 Git 历史，由以下哈希锚定：

```text
mot_dynamic_full_val_20260912_evidence.zip
bytes  16182
SHA256 0B05BACB5FAB5A91ACABE4211A6CF6852ABD01B7C37588E04A874E7BAE18CC72

mot_dynamic_route_margin_20260912_evidence.zip
bytes  18728
SHA256 818607439F165D63369A1839F365BCC3C266D8EBD5F828759073C4D78BD2C63D
```

本目录的 `full_val_summary.json` 是从原始 `DYNAMIC_MOT_FULL_VAL_THOP_FIXED.json` 提取的审查尺寸摘要。
完整 ZIP 本地归档于：

```text
D:/YOLO_Master/P1_dynamic_runtime_20260912/mot_dynamic_full_val_20260912_evidence.zip
D:/YOLO_Master/P1_dynamic_runtime_20260912/mot_dynamic_route_margin_20260912_evidence.zip
```

第二个证据包完成了 173 个漂移位置的 Top-K 边界裕量审计：两个漂移块的错位裕量最大值均未超过该块
dense 概率跨后端最大绝对误差，支持“近似并列浮点翻转”结论。结论和限制见
`reports/a3_dynamic_mot_full_val_20260912.md`。
