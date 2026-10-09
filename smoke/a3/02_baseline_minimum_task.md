# 02 - 基线 / 最小任务

> 提交日期：2026-08-24
> 锁定版本：main @ e9ac08b
> 对应登记表列：基线/最小任务

## A3 范围

- **P0/Smoke**：在锁定的 YOLO-Master 版本上重新复现部署链路，提交环境、命令、成功/失败日志与精度差；验证历史结论是否仍适用。

## 最小任务

复现 YOLO-Master EsMoE-N 从 PyTorch 权重到 ONNX Runtime CPU 后端的端到端部署链路，包含：

1. **导出预检**：`tests/test_export_preflight.py` 三个用例通过
2. **ONNX 导出**：`yolo export` 在 locked version 上成功生成 ONNX 模型
3. **端到端推理**：ONNX Runtime 跑通标准 COCO 测试图，输出预期类别
4. **精度对比**：PyTorch vs ONNX Runtime 在 COCO8 上的 mAP50-95 差 < 0.05（导出无损阈值）
5. **历史结论验证**：EsMoE README 描述的精度/架构口径在锁定版本下仍可复现

## 任务边界

| 项 | 在/不在范围 |
|---|---|
| PyTorch → ONNX → ONNX Runtime CPU 端到端 | ✅ 在范围 |
| 导出预检机制（export_preflight）验证 | ✅ 在范围 |
| EsMoE-N（YOLO-Master-EsMoE-N）全链路 | ✅ 在范围 |
| TensorRT FP16 / GPU 部署 | ❌ 本次 smoke 走 CPU（8.24 当时未启用独显）；2026-08-28 起本机 RTX 5060 8GB 可用，P1 阶段本机+云端 GPU 补 |
| COCO 完整 val2017 (5000 张) 精度 | ❌ CPU 跑全集耗时不经济；用 COCO8 (4 train + 4 val) 降级 |
| INT8 PTQ 量化 | ❌ P1 范围 |
| MoT/MoA/MoA-MOT 路由族（已具备 yaml，缺预训练权重） | ❌ P1 范围 |
| 上游 PR/Issue 提交 | ❌ 截止时间窗口内不申请，仅本地交付 |

## 锁定的基线

- **BASE_REF**：`e9ac08b2bd135c379206b8f77df9714b8800cbe0`（main 最新）
- **FINAL_REF**：本次提交在 BASE_REF 之上叠加 smoke 交付物的新 commit（待 push 后填入）
- **Tag**：`rhino-2026-0824-a3-baseline`（annotated，immutable；待 push 后填入 SHA）

## 基线可达性验证

```powershell
git merge-base --is-ancestor e9ac08b2bd135c379206b8f77df9714b8800cbe0 HEAD
# 退出码 0 = 满足
```
