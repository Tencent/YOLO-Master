<!--
C3 最终交付 PR 正文（分支 c3-vpeft-smoke → base Tencent:main，**新建** PR，不要复用 #279：
#279 的 head 是 fix/vpeft-capacity-guard，GitHub 不允许改已有 PR 的 head 分支）。
标题：
[犀牛鸟-C3]：V-PEFT 小样本工业缺陷检测结项交付
粘贴范围：从 "## SUMMARY｜C3 阶段交付" 开始到文件末尾（含末尾 "## PPT"）。
-->

## SUMMARY｜C3 阶段交付

用几百张图回答一个问题：在工业缺陷检测这种小数据场景里，V-PEFT 相比全量微调和冻结主干值不值得用。模型是 YOLO-Master EsMoE-N（全参 2,813,626），数据是 NEU-DET（1800 张）与 DeepPCB（1500 张）；三种策略在严格同预算下（epochs 100 / batch 8 / imgsz 640 / amp 关闭，同一 seed 池 824 / 2024 / 777）跑 18 个单元，另有 3 个补跑单元，逐单元记录精度、可训参数、显存峰值与墙钟时长，并把 planner 的决策与护栏日志留档。

本工作的交付不止最终精度表：

- 一份可复现的同预算对照矩阵与统计口径（best 口径 + 显存峰值 + 配对差 + 稳健估计）。
- 可恢复的矩阵调度：只挑显存 <1.5G 且利用率 <30% 的卡、按矩阵状态续跑、已完成单元不重跑，逐单元原始产物落盘。
- planner 决策审计：3 策略 × 2 数据集 × 3 seed 全部 ACCEPT，记录 target 数、rank、容量排除层与检测头解冻参数。
- seed 稳定性判定：3 个异常单元经补跑区分为「偶发 seed」与「该 (seed, 配置) 下的确定性行为」。
- 一个真实上游缺陷的定位与修复：容量约束缺失导致 V-PEFT 静默降级 legacy planner，源码修复（2 commit / 8 文件 + 8 个单测）可另行提交 PR，不占用本 PR。

结论是有限定的：参数效率的优势是数量级的（vpeft 只训练 116,736 个参数，全参的 4.15%，显存低约 1.15G），代价是 NEU 上 mAP50 低 0.073、DeepPCB 低 0.108，且 seed 稳健性明显不如全量微调（sd 0.052 / 0.072 对 0.014 / 0.000）。同样 100 轮，三种策略墙钟时长接近（45–55 分钟/单元），没有取得训练加速，瓶颈在数据管线。因此本轮不主张「PEFT 全面优于全量微调」，只主张受限参数/显存预算下的数量级参数效率。

| 阶段 | 交付内容 | 核心结论 |
|---|---|---|
| P0 | 环境重建、两个数据集准备、服务器化 runner 与三策略冒烟、planner 源码笔记 | 两个数据集都跑通 V-PEFT，planner 决策与护栏日志可查 |
| P1 | 3 策略 × 3 seed × 2 数据集同预算矩阵、统计与配对差 | 参数效率数量级成立（4.15%）；精度与 seed 稳健性有代价；时长无差异 |
| P2 | 小样本曲线（未做）+ planner 真实缺陷定位与修复 | 走任务书允许的「发现并修复真实 bug」一支，源码修复另开 PR |

## P0：打通 V-PEFT 与规划日志

- 环境：独立 conda env（python 3.11），先 `torch==2.6.0` + `torchvision==0.21.0`（内网镜像），再 `requirements.txt` 与 `pip install -e` 仓库，8 张 A800 均可见。
- 数据：NEU-DET 1800 张灰度、6 类，train 1620 / val 180；DeepPCB 1500 张、6 类，train 1350 / val 150。NEU 另备 k5/10/50/100 的 few-shot 子集（对应 30 / 60 / 300 / 600 张）。
- 冒烟：`server_run.py` 把阶段一脚本服务器化（路径自动推断、输出目录唯一且不可覆盖、落 `command.sh` / `training.log` / resolved `args.yaml` / `results.csv` / `summary.json`），三策略在 NEU 全量划分（imgsz 640 / batch 8 / seed 824）全部 exit 0。这一步只验证链路，数字不作结论。
- 环境坑（已修，见 `bugfix_log.md`）：PATH 里的裸 `yolo` 指向其它项目的 ultralytics，8 个 `lora_*` 参数全部不识别、7.8 秒退 1，容易被误判成代码坏了；改用当前 env 的 `yolo` 后正常，`smoke/c3/run_smoke.py` 一并修。并发首次生成 `labels.cache` 会撞车，调度器清 `PYTHONPATH` 并预热规避。
- planner 源码事实记录在 `planner_solver_notes.md`：solver / OR-Tools / 容量护栏 / 审计路径 / LOVO 语义（带 文件:行号），后续读 `[V-PEFT]` 日志以此为准。

## P1：严格同预算的三策略对照

**(a) 配置与跑法**

矩阵为 `{NEU-DET, DeepPCB} × {full_sft, vpeft, frozen_backbone} × seed {824, 2024, 777}`，18 个单元，统一 epochs 100 / batch 8 / imgsz 640 / amp 关闭（100 轮远大于默认 warmup 3）。`make_matrix.py` 生成矩阵，`sweep_runner.py` 只挑显存 <1.5G 且利用率 <30% 的卡、按矩阵状态续跑、已完成单元不重跑，中途重启不影响。实测约 35 s/epoch，单单元约 1 小时，5 卡并行 18 个单元约 4 小时。

**(b) 结果（best 口径：各 epoch 验证 mAP 的最大值，非末行）**

| 数据集 | 策略 | mAP50（824 / 2024 / 777） | 峰值显存 | 单 seed 时长 |
|---|---|---|---|---|
| NEU | full_sft | 0.752 / 0.780 / 0.768 | 5.31G | ~3.3k s |
| NEU | vpeft | 0.691 / 0.369 / 0.746 | 4.15G | ~3.1k s |
| NEU | frozen_backbone | 0.685 / 0.210 / 0.659 | 3.74G | ~2.7k s |
| PCB | full_sft | 0.989 / 0.989 / 0.990 | 5.28G | ~2.8k s |
| PCB | vpeft | 0.864 / 0.819 / 0.961 | 4.11G | ~2.7k s |
| PCB | frozen_backbone | 0.639 / 0.891 / 0.907 | 3.70G | ~2.9k s |

**(c) 统计与判读（n=3, t=3.18）**

| 数据集 | 策略 | 可训练参数（占比） | 峰值显存 | mAP50 mean±sd | 与 full_sft 的配对差 |
|---|---|---|---|---|---|
| NEU | full_sft | 2,813,626（100%） | 5.31G | 0.767±0.014 | — |
| NEU | vpeft | adapter 116,736（4.15%），含解冻 head 共 465,250（16.5%） | 4.16G | 0.694±0.052 | −0.073 |
| NEU | frozen_backbone | 1,906,822（67.8%） | 3.74G | 0.694±0.041 | −0.073 |
| PCB | full_sft | 同上 | 5.27G | 0.989±0.000 | — |
| PCB | vpeft | 同上 | 4.11G | 0.881±0.072 | −0.108 |
| PCB | frozen_backbone | 同上 | 3.71G | 0.769±0.151 | −0.177 |

NEU 的 0.694 是剔除偶发 seed 2024 后的稳健口径；不剔除时 3 seed 为 vpeft 0.612±0.167、frozen 0.573±0.245。DeepPCB frozen 的 0.769 含 seed 824 的确定性震荡。

- 精度：DeepPCB 全量极稳（0.989±0.000）；NEU 上全量微调比 vpeft / frozen 高约 0.07 mAP50，且 seed 稳健性好得多。
- 参数与显存：vpeft 只占 4.15% 全参、显存低约 1.15G，用不大的精度代价换数量级的参数节省；frozen 省得少（67.8% 可训）而稳定性更差。
- 时间：三种策略同预算时长相当，此类方法在本数据规模下没有加速收益。
- planner：6/6 单元决策一致 ACCEPT（81 个 target，rank=8，base rank 未升），检测头因类别数变化（80→6）重新初始化并解冻约 348,514 个参数；容量不足的层本轮用 `lora_exclude_modules` 规避，明细见 `stage4_analysis/planner_audit_summary.md`。

**(d) 稳定性与确定性**

3 个单元收敛异常：NEU vpeft seed 2024 停在 0.369（曲线在 0.2–0.3 平台），NEU frozen seed 2024 全程 <0.06，DeepPCB frozen seed 824 剧烈震荡（best 出现在第 9 轮 0.639）。补跑 3 个单元：NEU vpeft / frozen 换 seed 2025 得 0.644 / 0.739，正常收敛，判为 seed 偶发；DeepPCB frozen 用 seed 824 重跑逐值一致（best 仍在第 9 轮），判为该 (seed, 冻结配置) 下的确定性行为，不是采样噪声。

## P2：planner 缺陷的定位与修复

小样本曲线没有画：k5/10/50/100 划分已备好（对应 30/60/300/600 张；任务书要求的是 10/50/100/500 张），且 DeepPCB 的 few-shot 划分类别严重不平衡（k5 仅 8 张、k10 仅 13 张，open / short 等稀有类凑不够），本轮矩阵跑的是两个数据集的全量划分。按任务书 P2 的两条路线，本课题走「发现并修复 planner 真实 bug」这一支：

- 现象：七个硬约束（`C_op` / `C_sem` / `C_budget` / `C_deploy` / `C_compat` / `C_moe` / `C_div`）都没有建模层容量，而 plan 校验会拒绝 `rank > min(in, out)` 的目标，投影与校验互相矛盾 —— `0.conv`、`routing_network.2`、`25.dfl.conv` 被选为目标后 `apply_lora` 抛 `ValueError`；`lora_planner_backend=vpeft` 时该异常被 `except (ValueError, TypeError)` 吞掉、静默回退 legacy planner（`vpeft_strict=True` 下直接失败），等于 V-PEFT 没有生效。
- 修复（独立分支 `fix/vpeft-capacity-guard`，基于官方 `af961b9`，2 个 commit、8 个文件）：新增 `C_cap` 硬约束（`RankCapacityConstraint`），被跳过的层写入 `metadata["capacity_excluded"]` 并打一行可 grep 的 `[V-PEFT] capacity-excluded: ...`，校验一次报出全部违规目标，显式目标列表中被跳过的层改为冻结；新增 8 个单测。修复正文见 `stage5_pr_delivery/pr_body_capacity_guard.md`，如需上报可另开一个源码 PR。
- 边界：本轮消融是在该缺陷存在的前提下用 `lora_exclude_modules` 规避跑的，**没有用修复后的代码重跑**，精度数字仍是修复前口径。

## REVIEWER GUIDE｜交付物与证据入口

- 结项报告（结论 / 证据表 / 局限 / 复现）：`scripts/c3_vpeft/stage5_pr_delivery/report_final.md`
- 环境与数据：`stage1_env_data/README.md`、`datasets/manifest.md`（来源 / 许可 / 统计 / SHA-256 留档位置）、`prepare_neu_det.py`、`prepare_deeppcb.py`
- 服务器 runner 与冒烟：`stage2_server_smoke/server_run.py`、`smoke_cmds.sh`、`bugfix_log.md`、`planner_solver_notes.md`
- 矩阵与逐单元证据：`stage3_matrix/matrix.json`、`evidence_summary.csv|json`、`make_matrix.py`、`sweep_runner.py`、`collect_evidence.py`、`run_supplement.py`
- 统计与审计：`stage4_analysis/comparison_tables.md`、`planner_audit_summary.md`、`analysis_stats.py`、`p2/`
- 复现包（命令模板 / 路径模板 / 结果快照 / 局限）：`stage5_pr_delivery/reproduction/`
- 汇报材料：`stage5_pr_delivery/C3_结项汇报_初稿.pptx|.pdf` 与生成脚本 `gen_slides.py`
- 上游修复的说明文案（如需另开源码 PR）：`stage5_pr_delivery/pr_body_capacity_guard.md`

原始 `runs/`（逐 epoch 曲线、checkpoint）、数据集二进制与缓存不入库（见 `scripts/c3_vpeft/.gitignore`），入库的是脚本 + 汇总表 + 文档；需要逐 epoch 曲线时按 `reproduction/configs/train_commands_example.sh` 重跑。

## VALIDATION

本 PR 只新增文档与脚本（`smoke/c3/` 5 个文件 + `scripts/c3_vpeft/` 50 个文件），不改动 `ultralytics/` 源码与 `tests/`，因此不涉及仓库测试集的增删。可复查的是训练与汇总链路本身：

- 逐单元汇总 `stage3_matrix/evidence_summary.csv|json` 记录每个单元的 exit_code、best_epoch、mAP50(best)、mAP50-95(best)、显存峰值、planner 开关与预算，矩阵定义在 `matrix.json`；`collect_evidence.py` 从本地原始日志重算这两份表。
- `stage4_analysis/analysis_stats.py` 从 `evidence_summary.json` 重算 mean / sd / 95%CI 与配对差，产物为 `comparison_tables.md`。
- 源码级验证在源码修复 PR（8 个新增单测 + 相关回归用例），不在本 PR 范围。

## 已知局限与结论边界

1. seed 稳定性：NEU 上 vpeft / frozen 各有 1 个 seed 收敛异常，换 seed 补跑正常，判为偶发；DeepPCB frozen 在 seed 824 的震荡同 seed 可复现，说明该配置下训练本身不稳。该观察仅限于本轮 seed 池，不能估计失败概率。
2. 时间口径：三种策略时长接近，说明瓶颈在数据管线；本轮没有做端到端吞吐或推理侧测量，不主张部署加速。
3. 规模：EsMoE-N 只有 2.8M 参数，LoRA 相对全量的参数优势被小基数压缩，该对比不能直接外推到更大模型。
4. 数据许可：NEU-DET 镜像无显式 LICENSE（按学术用途使用，来源与 SHA-256 已登记）；DeepPCB 为 MIT。
5. 缺陷规避口径：消融数字来自修复前的排除清单方案，修复后未重跑。

## 说明

本 PR 不改动 `ultralytics/` 源码与 `tests/`，新增 `smoke/c3/`（阶段一准入冒烟与数据准备，5 个文件）与 `scripts/c3_vpeft/`（阶段一至五交付物，50 个文件）。分支基线为 `ba7e4b8`（早于当前 main `af961b9`），合并前按需 rebase 到最新 main。源码修复另开 PR 提交，不影响本 PR 的交付内容。

## PPT

https://drive.google.com/file/d/1dg8tZyI11xJR1JrCCKBWqbC4_kFgmsVp/view?usp=sharing
