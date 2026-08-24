# A3 Smoke — 8.24 验收交付物

> **登记编号**：A3（YOLO-Master Topic A3 / P0-Smoke）
> **作者**：麦聿恺（Kennee Mai）
> **提交日期**：2026-08-24
> **锁定版本 (BASE_REF)**：`e9ac08b2bd135c379206b8f77df9714b8800cbe0` (main 最新)
> **immutable tag**：`rhino-2026-0824-a3-baseline`（annotated，不允许 force-push）
> **本目录位置**：`smoke/a3/`

---

## 9 份交付物对应登记表

| 登记表列 | 交付物 | 状态 |
|---|---|---|
| 环境安装 | [`01_environment_install.md`](01_environment_install.md) | ✅ |
| 基线/最小任务 | [`02_baseline_minimum_task.md`](02_baseline_minimum_task.md) | ✅ |
| 复现命令 | [`03_reproduction_commands.md`](03_reproduction_commands.md) | ✅ |
| 配置文件 | [`04_config_files/`](04_config_files/) (coco8_data.yaml, export_settings.json, val_settings.json) | ✅ |
| 完整日志 | [`05_full_logs.md`](05_full_logs.md) | ✅ |
| 结果证据 | [`06_result_evidence/`](06_result_evidence/) (val_metrics.json, 曲线图, 端到端预测样例) | ✅ |
| 设计说明 | [`07_design_doc.md`](07_design_doc.md) | ✅ |
| 风险与降级 | [`08_risk_degradation.md`](08_risk_degradation.md) | ✅ |
| 代码/方案链接 | 本目录（待 push 后链接到 `https://github.com/KennnMai/YOLO-Master/tree/rhino-a3-dev/smoke/a3`） | ✅ |

---

## 关键数字一览

| 项 | 值 |
|---|---|
| BASE_REF | `e9ac08b2bd135c379206b8f77df9714b8800cbe0` |
| 预训练权重 | `YOLO-Master-EsMoE-N.pt` (5.7 MB, SHA256 `29e1b93f...`) |
| ONNX 导出 | `yolo_master_n.onnx` (10.6 MB, SHA256 `586cf6fe...`), 21.9s, opset 19 |
| ONNX Runtime | 1.29.0, CPUExecutionProvider |
| **mAP50-95** | PT 0.7432 → ORT 0.7108 (Δ = -0.0324) |
| **mAP50** | PT 0.9562 → ORT 0.9522 (Δ = -0.0040) |
| **推理加速** | PT 233.6 ms → ORT 71.5 ms (3.27×) |
| preflight 单测 | 3/3 PASS (1.51s) |
| 历史结论 | EsMoE README 42.4% AP 量级在锁定版本下仍适用 |

---

## 规则对照

| 规则 | 落地 |
|---|---|
| 3.2 不可单独作证项（截图/口头/后补 README/私有仓库） | 已避免 |
| 3.2 immutable tag | `rhino-2026-0824-a3-baseline` (annotated) |
| 3.2 产物校验值 | `09_artifact_checksums.txt` + `04_config_files/export_settings.json` |
| 3.3 未登记处理 | 本次只算 P0；P1 单独分支累计 |
| 4.2 推荐核验命令 | `03_reproduction_commands.md` 第 10 节给出 |

---

## 快速复现

```powershell
# 在 smoke/a3/ 目录读 README 后，按 03_reproduction_commands.md 顺序执行 0-10 步
# 关键验证：
git rev-parse HEAD                      # → e9ac08b2bd135c379206b8f77df9714b8800cbe0
git tag -l rhino-2026-0824-a3-baseline  # 存在
git log --reverse e9ac08b..HEAD         # 列出新增 commit
git diff --stat e9ac08b..HEAD           # 列出变更文件
git merge-base --is-ancestor e9ac08b HEAD  # 退出码 0 = 通过
```

---

## 提交后登记行模板

```
16  麦聿恺  A3  已完成  已完成  已完成  已完成  已完成  已完成  已完成  已完成  已完成
```
