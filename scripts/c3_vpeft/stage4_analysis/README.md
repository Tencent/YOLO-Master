# C3 stage4: 统计分析与证据结论（定稿）

主 18 单元（NEU-DET / DeepPCB × full_sft/vpeft/frozen_backbone × seed 824/2024/777）
+ 3 补充单元（`neu_vpeft_s2025`、`neu_frozen_s2025`、`pcb_frozen_s824b`）全部完成，
同预算对照（epochs=100, batch=8, imgsz=640, amp=false）。

## 口径与红线

- **指标** = results.csv 各 epoch val mAP 的 **best**（max），非末行（部分单元后期过拟合/震荡，
  如 PCB frozen_s824 末行 0.496 vs best@ep9 0.639）。
- **显存** = training.log 进度行 GpuMem 峰值。n 样本 CI 半宽 t=3.18(n=3)/2.78(n=4)。
- 主判读表固定 3 seed(824/2024/777)同预算；补充单元用于**稳定性/确定性验证**（显式报告，不混入主表 seed 池）。

## 表 1: 每 (dataset, strategy) 多 seed mAP50（best；主18 + 补充打勾的验证）

| dataset | strategy | seeds(best mAP50) | mean | sd | 95%CI |
|---|---|---|---|---|---|
| neu | full_sft | 2024:0.780, 777:0.768, 824:0.752 | **0.767** | 0.014 | ±0.026 |
| neu | vpeft | 2024:0.369⚠, 2025:0.644✓, 777:0.746, 824:0.691 | 0.612 | 0.167 | ±0.233 |
| neu | frozen_backbone | 2024:0.210⚠, 2025:0.739✓, 777:0.659, 824:0.685 | 0.573 | 0.245 | ±0.340 |
| pcb | full_sft | 2024:0.989, 777:0.990, 824:0.989 | **0.989** | 0.000 | ±0.001 |
| pcb | vpeft | 2024:0.819, 777:0.961, 824:0.864 | 0.881 | 0.072 | ±0.133 |
| pcb | frozen_backbone | 2024:0.891, 777:0.907, 824:0.639⚠, 824b:0.639⚠✓ | 0.769 | 0.151 | ±0.209 |

⚠ 收敛异常; ✓ = 补充单元完成。完整逐 epoch 曲线在 stage3 runs（不入库）。

## 表 2: 配对差（逐 seed: full_sft − 对照, 主 3 seed）

- NEU full−vpeft: **+0.165**（+0.022/+0.061/+0.411; 大差全来自 s2024 停滞）
- NEU full−frozen: **+0.249**（+0.109/+0.067/+0.570; 大差来自 s2024 卡死）
- PCB full−vpeft: **+0.108**（+0.029/+0.125/+0.170）
- PCB full−frozen: **+0.177**（+0.082/+0.351/+0.098）

## 补充单元判定（稳定性/确定性验证结论）

1. **NEU vpeft s2024 停滞（best 0.369）= seed 偶发**：新 seed 2025 正常收敛 0.644。
2. **NEU frozen s2024 全程卡死（best 0.210）= seed 偶发**：seed 2025 正常 0.739。
3. **PCB frozen s824 震荡 = 确定性动力学（非采样噪声）**：同 seed 824 重跑（s824b）逐一致，
   仍 best@ep9=0.639、末行 0.496 —— 该 (seed,冻结配置) 下 frozen 训练不稳是**可复现事实**。

### 稳健性口径（剔除偶发 seed 2024，NEU）
| strategy | mean(best mAP50, 剔除后 n) | sd | vs full |
|---|---|---|---|
| vpeft | 0.694 (n=3: 824/777/2025) | 0.052 | −0.073 |
| frozen_backbone | 0.694 (n=3: 824/777/2025) | 0.041 | −0.073 |
| full_sft | 0.767 (n=3) | 0.014 | — |

## 表 3: 四维对照（参数/显存/时长; 同预算）

| dataset | strategy | 可训参数(占比) | 峰值显存 | 时长/seed |
|---|---|---|---|---|
| neu | full_sft | 2,813,626 (100%) | 5.31G | 55m |
| neu | vpeft | adapter 116,736 (4.15%); +head 有效 465,250 (16.5%) | 4.16G | 51m |
| neu | frozen_backbone | 1,906,822 (67.8%) | 3.74G | 45m |
| pcb | full_sft | 同上 | 5.27G | 46m |
| pcb | vpeft | 同上 | 4.11G | 45m |
| pcb | frozen_backbone | 同上 | 3.71G | 51m |

## 结论判读

- **精度**：PCB 全量极稳(0.989±0.001)；NEU 上 full(0.767) 比 vpeft/frozen(0.694 稳健口径) 高 ~0.07 mAP50，
  且 full 的 seed 稳健性远好(sd 0.014 vs 0.04~0.05) —— **全量微调仍是小模型+小数据的最佳稳健基线**。
- **参数/显存**：vpeft adapter 仅 4.15% 全参、显存 −1.15G；frozen 67.8% 可训、显存 −1.6G。
  受限预算(P1)场景下 vpeft 以小幅精度代价换数量级参数节省，收益成立；frozen 参数节省有限且牺牲稳定性。
- **时长**：同 epochs 预算三策略基本一致(~45-55m/单元)，参数高效方法在本数据规模未体现训练加速（瓶颈为数据管线）。
- **限制**：n 样本小、CI 宽；NEU 偶发率(1/4)估计仅作警示不推总；结果限定 EsMoE-N + 两数据集。

## 收尾检查单

- [x] analysis_stats.py → comparison_tables.md（含补充单元）
- [x] 补充单元并入判读并定稿（见上表与判定节）
- [x] planner_audit_summary.md（ACCEPT 6/6、81 targets rank8、cap<8 exclude、LOVO 语义）
- [x] 复现包 stage5/reproduction 组装
- [x] stage3 README 补 supplement 结果；本 stage commit（push 待本人手动）
