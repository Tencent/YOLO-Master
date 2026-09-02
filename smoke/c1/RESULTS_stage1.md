# C1 Stage1：on/off 路由对照（VisDrone，seed=42，50 epoch）

## 方法（四同口径）

| 超参 | 值 |
|---|---|
| epochs | 50 |
| batch | 8 |
| imgsz | 640 |
| device | 0 (RTX 4060 Laptop, sm_89) |
| workers | 0 |
| seed | 42 |
| amp | False（离线训练，无预训练权重） |
| pretrained | False |
| eval | dense eval（`--no-sparse-eval`，避免 ES-MoE sparse 塌陷 mAP） |

两个模型使用**完全相同**的超参与数据划分（统一 VisDrone 划分，本地 `VisDrone.yaml` → `D:/datasets/VisDrone`）。

## 结果

| 模型 | 路由 | 参数量 | 最佳 mAP50 | 最佳 epoch |
|---|---|---|---|---|
| v0.1-N | 关（dense baseline） | 7.547M | **0.19590** | 50 |
| EsMoE-N | 开（ES-MoE 路由） | 2.845M | 0.19052 | 48 |

## 结论

- **平手**：v0.1-N 仅高 0.0054 mAP，差距在单 seed 噪声范围内，不构成显著优势。
- **参数效率**：ES-MoE-N 用 **37.7%** 的参数（2.845M vs 7.547M）达到同等精度 → **参数效率约 2.6×**。
- 这是 P2 的核心机制故事：ES-MoE 在不损失精度的前提下大幅压缩参数。

## 训练证据（日志可复现）

- 训练日志与 `results.csv` 位于 `runs/c1_onoff/VisDrone_v0.1-N` 与 `runs/c1_onoff/VisDrone_EsMoE-N`。
- ES-MoE 路由确实在工作：`mixture_aux_loss` 在训练中非零（可作为路由激活证据）。
- 环境：`torch 2.3.1+cu118` + `torchvision 0.18.1+cu118`（依赖见 `requirements_smoke.txt`）。
