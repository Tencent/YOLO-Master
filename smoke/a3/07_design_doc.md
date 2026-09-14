# 07 - 设计说明

> 提交日期：2026-08-24
> 锁定版本：main @ e9ac08b
> 对应登记表列：设计说明

## 1. 目标

按 A3 P0/Smoke 任务原话："在锁定的 YOLO-Master 版本上重新复现部署链路，提交环境、命令、成功/失败日志与精度差；验证历史结论是否仍适用。"

本文档说明本次复现采用的**方法学、判定标准、关键选择**与对应的设计意图。

## 2. 复现方法学

### 2.1 锁定 → 重跑 → 对比

```
上 游 公 共 基 线 (e9ac08b)
        ↓
   浅克隆 + ref 指向
        ↓
   本地工作目录
        ↓
   ① 装环境（已修复 onnxruntime==1.19.2 上游笔误）
   ② 拉权重（hf-mirror 镜像）
   ③ 装 COCO8（手动写 data.yaml 指向本地）
   ④ 跑 preflight 单测（验证 export_preflight 链路）
   ⑤ yolo export → ONNX（验证端到端导出）
   ⑥ yolo predict → bus.jpg（验证 ORT 推理）
   ⑦ PyTorch val（COCO8 精度基线）
   ⑧ ONNX val（COCO8 精度基线）
   ⑨ 差值对比 + 历史结论验证
        ↓
   9 份交付物 + tag + 推送
```

### 2.2 判定标准

| 项 | 阈值 | 来源 |
|---|---|---|
| ONNX 导出 mAP50 差 | < 0.01 | 行业惯例（"导出无损"） |
| ONNX 导出 mAP50-95 差 | < 0.05 | 行业惯例（"可接受"） |
| 历史结论量级验证 | ±20% 区间 | 工程实践（小数据集噪声） |
| immovable tag | annotated tag + 不 force-push | 规则 3.2 |

### 2.3 关键设计选择

1. **使用 COCO8 而非 COCO val2017 完整集**：本机无 GPU，CPU 跑 5000 张耗时不经济（数小时），且 8.24 时间窗不允许。按任务规则"无 GPU 降级"原则，用 8 张子集做 smoke。
2. **手动写 COCO8 data.yaml**：ultralytics 官方 release 里的 coco8.zip 不含 data.yaml，本机网络拉不到，需要根据标准 COCO 80 类手写并指向本地路径。
3. **fork 提交策略**：从 e9ac08b 拉新分支 `rhino-a3-dev/smoke/a3`，不动 fork 现有 main；新 commit 上叠 smoke/ 目录。
4. **immutable tag**：用 annotated tag（`git tag -a`）+ 规则声明，不允许 force-push 改写。

## 3. 与规则对照

| 规则章节 | 要求 | 本次落地 |
|---|---|---|
| 3.2 可接受证据 | commit 页、原始日志、产物校验值、可复现配置 | ✅ 全部具备（commit e9ac08b / `03_reproduction_commands.md` / `09_artifact_checksums.txt` / `04_config_files/`）|
| 3.2 不可单独 | 截图、口头说明 | ✅ 已避免作为唯一证据，PNG 仅做辅助 |
| 3.2 私有成果 | 需导师只读访问 / 不可变归档 | N/A（本次走公开 fork 路径）|
| 3.2 不允许 | 登记后 force-push 改写 | ✅ 用 annotated tag；规则声明不 force-push |
| 3.3 未登记处理 | 锁定前既有工作不得算 P1/P2 | ✅ 本次只算 P0/Smoke；P1 待锁定后新建分支单独累计 |
| 4.1 P0 边界 | 历史功能经当前版本复现 + 完整证据 | ✅ 全部具备 |
| 4.1 P1 边界 | 锁定后实质新增 | N/A（本次只做 P0）|
| 4.2 推荐核验 | `git diff` / `git log` / `git merge-base` | ✅ 命令在 `03_reproduction_commands.md` 第 10 节 |
| 5 A3 专项 | 不重复认定既有跨平台/EsMoE 混合 INT8 | ✅ 公共 baseline 已声明，本次只补 smoke 端到端链路 |

## 4. 不在本次范围（明确声明）

- P1：MoT 路由族 FP32/FP16/INT8 闭环 + 路由漂移分析
- P2：五族统一量化矩阵、QAT 精度恢复、专家选择一致率、自动敏感层回退、真实端侧验证
- TensorRT FP16 / TensorRT INT8（无 GPU）
- COCO val2017 完整集精度（CPU 跑全集耗时不经济）
- 上游 PR/Issue 提交（截止时间窗口内不申请）

## 5. 后续路线

1. P1 阶段挂云端 GPU 重做 COCO val2017 完整集 + INT8 PTQ
2. P1 阶段补 MoT yaml 初始化权重端到端
3. P2 阶段实现量化矩阵 + 漂移分析自动化
