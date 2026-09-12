# 已知局限与风险（C3 结项）

> 口径：主 18 单元 + 3 补充单元，NEU-DET / DeepPCB × full_sft / vpeft / frozen_backbone，同预算
> （epochs=100, batch=8, imgsz=640, amp=false）。指标取 `results.csv` 里 val mAP50 的 **best**（非末行，
> 部分单元后期过拟合/震荡），显存取 `training.log` 进度行 GpuMem 峰值。
> 明细见 `results/comparison_tables.md`、`results/matrix.json`、`results/planner_audit_summary.md`。

## 1. 精度结论的边界

- NEU-DET：full_sft 0.767±0.014；vpeft 0.694±0.052、frozen_backbone 0.694±0.041（稳健口径：剔除偶发 seed 2024，n=3）。full 比两者高约 0.07 mAP50，且 seed 稳健性明显更好 —— **全量微调仍是"小模型 + 小数据"下最稳的基线，本项目不主张 PEFT 的精度收益**。
- DeepPCB：full_sft 0.989±0.000（三个 seed 几乎重合），vpeft 0.881±0.072，frozen_backbone 0.769±0.151。
- 样本量小：主表 n=3，CI 半宽 ±0.026 ~ ±0.34，差异只能作趋势判断。

## 2. seed 稳定性与确定性

- NEU-DET vpeft seed 2024 停滞（best 0.369），换 seed 2025 正常收敛到 0.644 → 判为 seed 偶发。
- NEU-DET frozen_backbone seed 2024 全程卡死（best 0.210），seed 2025 正常 0.739 → 判为 seed 偶发。
- DeepPCB frozen_backbone seed 824 的震荡**同 seed 重跑逐一致**（补充单元 s824b：best@ep9=0.639、末行 0.496）→ 该 (seed, 冻结配置) 下的训练不稳是**可复现事实**，不是采样噪声。
- 以上都只在当前 seed 池（824/2024/777/2025）里观察到，**不能用来估计失败概率**。

## 3. 缺陷规避：精度数字是"修复前"口径

- planner 的 `cap<8` 缺陷（窄层被投影成适配目标 → plan 校验抛错 → `vpeft` 静默回退 legacy planner）是本轮实验**用 `lora_exclude_modules` 手工排除清单规避**的，所以所有精度/显存数字都来自**修复前的规避方案**。
- 源码修复在独立分支 `fix/vpeft-capacity-guard`（另开的上游 PR，编号待创建：`C_cap` 硬约束 + `capacity_excluded` 审计 + 未适配层冻结 + 8 个单测），**修复后没有重跑训练**。

## 4. 数据与许可

- NEU-DET：镜像仓库与官方数据库页面**都没有显式 LICENSE** → 提交物里如实声明许可不确定性，仅学术/评测用途。来源、统计与产物路径见 `datasets/manifest.md`。
- DeepPCB：MIT（仓库 LICENSE）。
- 数据集本体、`split_report.json`（样本 SHA-256 记在里面）和 `runs/` 产物**都不入库**。要复核 SHA 需按 `datasets/prepare_neu_det.py` / `datasets/prepare_deeppcb.py` 重下重生成；脚本用 `set` 去重后迭代，跨进程顺序受 `PYTHONHASHSEED` 影响，**重跑可能得到不同子集**，因此实验一律以首次固化产物为准（`train.txt` / `val.txt` / `shots/`）。

## 5. 规模与可外推性

- EsMoE-N 只有 2.8M 参数，adapter 116,736 个（4.15%）——小基数会压缩"参数效率数量级优势"的叙事，**结论不能直接外推到更大模型**。
- 同 epochs 预算下三策略墙钟时长接近（45–55 min/单元，瓶颈在数据管线），本轮没有观察到参数高效方法带来的训练加速。
- 所有结论限定在 NEU-DET / DeepPCB 两个工业缺陷数据集 + 这一份 EsMoE-N 检查点。
