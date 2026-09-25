# 06 - 结果证据索引

> 提交日期：2026-08-24
> 锁定版本：main @ e9ac08b
> 对应登记表列：结果证据

> 按规则 3.2："截图不能单独作证"；本目录提供 **可机验的 JSON 度量 + SHA256 校验值 + 可视化辅助 + 端到端预测样例**，三类证据互证。

## 文件清单

### 主指标 JSON
- `val_metrics.json` — PyTorch vs ONNX Runtime 全量指标对比（mAP50-95 / mAP50 / P / R / 推理耗时 / speedup）

### 可视化辅助（PNG / JPG，**非唯一证据**，仅作为度量趋势的视觉佐证）
- `val_pt_plots/` — PyTorch 端的 BoxF1/PR/P/R 曲线 + 混淆矩阵 + 批次预测
- `val_onnx_plots/` — ONNX Runtime 端的同类曲线与样本
- `ort_predict_sample/bus_input.jpg` — 标准 COCO 测试图 bus.jpg（输入）
- `ort_predict_sample/bus_pred.jpg` — ONNX Runtime 推理后的带检测框 bus.jpg（**4 persons + 1 bus**）

### 端到端预测样例
- 同上 `ort_predict_sample/`：用 ONNX Runtime 对单张图跑出 4 persons + 1 bus，与 PyTorch 一致

## 关键数字（从 val_metrics.json 读出）

| 指标 | PyTorch | ONNX Runtime | Δ |
|---|---|---|---|
| mAP50-95 | 0.7432 | 0.7108 | **-0.0324** |
| mAP50 | 0.9562 | 0.9522 | -0.0040 |
| Precision | 0.8026 | 0.8546 | +0.0520 |
| Recall | 0.9270 | 0.8908 | -0.0362 |
| 推理耗时 (ms) | 233.6 | 71.5 | **3.27× 加速** |

## 证据强度

| 类型 | 强度 | 用途 |
|---|---|---|
| JSON 度量 | **强** | 可机验、可 diff、可在 CI 复跑 |
| SHA256 校验 | **强** | 锁产物不被替换（见 `09_artifact_checksums.txt`） |
| PR/Issue/commit | **强** | 第三方公开记录（待 push 后由 fork 仓库提供） |
| PNG 曲线 | 中 | 视觉确认趋势 |
| JPG 检测图 | 中 | 端到端流程可视化 |
| 截图 | **弱** | 仅做辅助；本目录已避免依赖单一截图 |

## 局限

- 8 张 COCO8 统计噪声大；如需严谨对比，应在云端重做 COCO val2017 完整集
- 缺少 NVIDIA GPU；TensorRT FP16 性能未做（P1 阶段补）
- MoT/MoA 路由族（已具备 yaml，缺公开预训练权重）端到端未做（P1 阶段补）
