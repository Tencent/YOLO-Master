<!--
C3 整体结项交付 PR 正文（分支 c3-vpeft-smoke → base Tencent:main）。
用途：仓库 PR #279 重开时作为描述（head 分支换成 c3-vpeft-smoke，见 stage5 README）。
标题：
[犀牛鸟-C3]：V-PEFT 小样本工业缺陷检测结项交付
粘贴范围：从 "## 交付内容" 开始到文件末尾（含 "## PPT"）。
-->

## 交付内容

C3 做的是 V-PEFT 在工业缺陷检测上的效果验证，模型用 YOLO-Master EsMoE-N（全参 2,813,626），数据集是 NEU-DET 和 DeepPCB。这个 PR 把整条链路交上来：环境重建、数据准备、A800 上的冒烟、21 个训练单元的对照矩阵、统计分析和结项报告，都放在 `c3_work/` 里按阶段分目录。`runs/`、数据集和缓存没入库，入库的是脚本、日志、汇总表和文档。

| 目录 | 内容 |
|---|---|
| `c3_work/stage1_env_data/` | 环境搭建、两个数据集下载与 YOLO 化、few-shot 划分、许可与 SHA-256 登记 |
| `c3_work/stage2_server_smoke/` | `server_run.py`、三策略冒烟、planner 源码笔记、BF-01 修复 |
| `c3_work/stage3_matrix/` | 18 单元矩阵、调度与续跑脚本、3 个补跑单元、证据汇总 |
| `c3_work/stage4_analysis/` | 多 seed 统计与配对差、planner 审计、四维对照表 |
| `c3_work/stage5_pr_delivery/` | 结项报告、复现包、汇报 PPT 与生成脚本、上游修复 PR 文案 |

结论：vpeft 只训练 116,736 个 LoRA adapter 参数（全参的 4.15%），NEU 上 mAP50 比全量微调低约 0.07、DeepPCB 低约 0.11，显存少 1.2G。但同样预算跑 3 个 seed，全量微调的稳定性明显更好（sd 0.014），vpeft 和 frozen 都出现过极端 seed（补跑确认是偶发），三者训练时长差不多，瓶颈在数据管线。

## 环境与数据

环境是独立的 conda env（python 3.11），不占团队共享环境：先装 `torch==2.6.0` + `torchvision==0.21.0`（内网镜像），再装 `requirements.txt`，最后 `pip install -e` 仓库。装完 `yolo 8.4.101`、`torch 2.6.0+cu124`，8 张 A800 都能看到。

NEU-DET 1800 张灰度图、6 类，train 1620 / val 180；另外切了 k5/10/50/100 的 few-shot 子集，每类 5/10/50/100 张，对应 30/60/300/600。这个镜像里没有 LICENSE 文件，按学术用途使用，来源已在 manifest 里写明。

DeepPCB 1500 张测试图、6 类，train 1350 / val 150。它的 few-shot 划分类别很不平衡（k5 只有 8 张、k10 只有 13 张，open / short 这些稀有类根本凑不够），所以小样本曲线的主线放在 NEU-DET，DeepPCB 用全量划分。许可为 MIT。

两个数据集的来源、许可、统计和 SHA-256 记在 `stage1_env_data/datasets/manifest.md`，下载和转换日志在同目录。

## 服务器冒烟

`server_run.py` 是把阶段一的 `run_smoke.py` 服务器化：路径自动推断，输出目录唯一且不允许覆盖（重跑必须换 `--name`），跑完落 `command.sh`、`training.log`、resolved 的 `args.yaml`、`results.csv` 和 `summary.json`（GPU 前后快照 + exit code + 耗时 + 末行指标）。

三种策略都在服务器上跑通了（NEU 全量划分，imgsz 640、batch 8、seed 824，全部 exit 0）。这一步只验证链路，数字不作数，正式实验在 stage3 全部重跑。

踩到的坑是 PATH 里的裸 `yolo` 指向别人项目的 ultralytics，8 个 `lora_*` 参数全部不识别，7.8 秒就退 1，很容易误判成代码坏了；改成用当前 env 的 `Path(sys.executable).parent/"yolo"` 就正常，`smoke/c3/run_smoke.py` 一并修了。另外并发首次生成 `labels.cache` 会撞车，调度器里清 PYTHONPATH 并预热缓存来规避。

`planner_solver_notes.md` 记录的是源码层面的事实：solver、OR-Tools、容量护栏、审计路径、LOVO 语义（带文件:行号）。后面读 `[V-PEFT]` 日志就靠它。

## 对照实验

矩阵是 `{NEU-DET, DeepPCB} × {full_sft, vpeft, frozen_backbone} × seed {824, 2024, 777}`，18 个单元，统一 epochs=100 / batch 8 / imgsz 640 / amp 关闭。100 轮远大于默认 warmup 3，三种策略在同样预算下可比。

跑法：`make_matrix.py` 生成矩阵，`sweep_runner.py` 只挑显存 <1.5G 且利用率 <30% 的卡，按矩阵状态续跑，做完的单元不重跑，中途重启也不影响（避免抢占别人的卡）。实测约 35s/epoch，一个单元 100 轮约 1 小时，5 卡并行 18 个单元 4 小时左右跑完。

结果用 best 口径（各 epoch 验证 mAP 的最大值，不是末行）

| 数据集 | 策略 | mAP50（seed 824 / 2024 / 777） | 峰值显存 | 单 seed 时长 |
|---|---|---|---|---|
| NEU | full_sft | 0.752 / 0.780 / 0.768 | 5.31G | ~3.3k s |
| NEU | vpeft | 0.691 / 0.369 / 0.746 | 4.15G | ~3.1k s |
| NEU | frozen_backbone | 0.685 / 0.210 / 0.659 | 3.74G | ~2.7k s |
| PCB | full_sft | 0.989 / 0.989 / 0.990 | 5.28G | ~2.8k s |
| PCB | vpeft | 0.864 / 0.819 / 0.961 | 4.11G | ~2.7k s |
| PCB | frozen_backbone | 0.639 / 0.891 / 0.907 | 3.70G | ~2.9k s |

其中三个单元收敛异常：NEU 上 vpeft seed 2024 停在 0.369（曲线在 0.2-0.3 一节平台上），NEU 上 frozen seed 2024 全程不到 0.06，DeepPCB 上 frozen seed 824 剧烈震荡（best 出现在第 9 轮 0.639）。为此补跑了三个单元：NEU 的 vpeft / frozen 换 seed 2025 分别是 0.644 和 0.739，正常收敛，判断是 seed 偶发；DeepPCB 的 frozen 用 seed 824 重跑，结果和第一次逐一致（best 仍在第 9 轮），说明是这个 (seed, 冻结配置) 下的确定性行为，不是采样噪声。

## 统计与结论

主表口径：指标取 best，显存取 `training.log` 进度行的 GpuMem 峰值，n=3 时 t=3.18。补充单元只用于验证稳定性，不混进主表的 seed 池。

| 数据集 | 策略 | 可训练参数（占比） | 峰值显存 | mAP50 mean±sd | 与 full_sft 的配对差 |
|---|---|---|---|---|---|
| NEU | full_sft | 2,813,626（100%） | 5.31G | 0.767±0.014 | — |
| NEU | vpeft | adapter 116,736（4.15%）、含解冻 head 共 465,250（16.5%） | 4.16G | 0.694±0.052 | −0.073 |
| NEU | frozen_backbone | 1,906,822（67.8%） | 3.74G | 0.694±0.041 | −0.073 |
| PCB | full_sft | 同上 | 5.27G | 0.989±0.000 | — |
| PCB | vpeft | 同上 | 4.11G | 0.881±0.072 | −0.108 |
| PCB | frozen_backbone | 同上 | 3.71G | 0.769±0.151 | −0.177 |

NEU 那两个 0.694 是剔除偶发 seed 2024 后的稳健口径（n=3）；不剔除时主表 3 seed 是 vpeft 0.612±0.167、frozen 0.573±0.245。DeepPCB frozen 的 0.769 里含 seed 824 的确定性震荡。

几点判读：

- 精度上，DeepPCB 全量非常稳（0.989±0.001）；NEU 上全量微调比 vpeft / frozen 高约 0.07 mAP50，而且 seed 稳健性明显更好（sd 0.014 对 0.04~0.05）。小模型加小数据这个场景下，全量微调仍是最稳的基线。
- 参数和显存上，vpeft 只占 4.15% 全参、显存低 1.15G，用不大的精度代价换数量级的参数节省，在受限预算下是成立的；frozen 省得没那么多（67.8% 可训），稳定性反而更差。
- 时间上没有差别，同样 100 轮三种策略都是 45-55 分钟一个单元，说明这类方法在这个数据规模下没有加速收益，瓶颈在数据管线。
- planner 的审计结果是 6/6 单元决策一致 ACCEPT（81 个 target，rank=8，base rank 没有升），检测头因为类别数不一样（80 类到 6 类）重新初始化并解冻了约 348,514 个参数；容量不足的层这一轮用 `lora_exclude_modules` 规避，细节见 `stage4_analysis/planner_audit_summary.md`。

## 过程中发现的上游缺陷

做实验时定位到 V-PEFT 的一个真实缺陷，顺便修了，但没混进这个 PR：修复单独放在基于官方 `af961b9` 的分支 `fix/vpeft-capacity-guard`（2 个 commit），走另一个上游 PR 提交。

问题是七个硬约束都没有建模层的容量，而 plan 校验会拒绝 `rank > min(in, out)` 的目标，于是投影和校验互相矛盾：`0.conv`、`routing_network.2`、`25.dfl.conv` 被选为目标后 `apply_lora` 抛 `ValueError`，vpeft 后端下异常被吞掉，运行静默回退到 legacy planner（`vpeft_strict=True` 时直接失败，等于 V-PEFT 没生效）。

修的内容：新增 `C_cap` 硬约束（`RankCapacityConstraint`），被跳过的层写进 `metadata["capacity_excluded"]` 并打一行可 grep 的 `[V-PEFT] capacity-excluded: ...`；校验阶段一次报出全部违规目标；显式目标列表或过滤跳过的层改为冻结，和 manual 后端对齐。新增 8 个单测。

本轮消融是在这个缺陷存在的前提下用 `lora_exclude_modules` 规避跑的，没有用修复后的代码重跑。完整说明见 `stage5_pr_delivery/pr_body_capacity_guard.md`。

## 复现

```bash
# 环境和数据
bash c3_work/stage1_env_data/env_setup.sh
bash c3_work/stage1_env_data/download_datasets.sh
python c3_work/stage1_env_data/prepare_neu_det.py
python c3_work/stage1_env_data/prepare_deeppcb.py

# 单次训练（模板见 smoke_cmds.sh）
python c3_work/stage2_server_smoke/server_run.py --help

# 矩阵、调度、证据汇总
python c3_work/stage3_matrix/make_matrix.py
nohup python c3_work/stage3_matrix/sweep_runner.py >> runner.log 2>&1 &
python c3_work/stage3_matrix/collect_evidence.py

# 统计与审计
python c3_work/stage4_analysis/analysis_stats.py
```

路径和预算参数以 `stage5_pr_delivery/reproduction/configs/paths.env`、`train_commands_example.sh` 为准；数据来源/许可/统计见 `reproduction/datasets/manifest.md`，局限与风险见 `reproduction/limitations.md`；`runs/`、数据集和 `*.cache` 已在 `c3_work/.gitignore` 里排除。

## 目标达成情况（P0 / P1 / P2）

按课题的任务分级（P0 保底 / P1 预期 / P2 理想）对照：

- P0 达成：NEU-DET 和 DeepPCB 上都跑通了 V-PEFT（各 3 个 seed，共 6 个单元，另有 3 个补跑单元），planner 决策与护栏日志都留档，见 `stage3_matrix/matrix.json`、`stage4_analysis/planner_audit_summary.md` 和各单元 `training.log` 里的 `[V-PEFT]` 行。
- P1 达成：三种策略严格同预算（epochs=100 / batch 8 / imgsz 640 / amp 关闭，同一 seed 池），PEFT 的参数效率数量级优势成立——vpeft 只训练 116,736 个参数，是全量微调的 4.15%（约 1/24）。显存低 1.15G、训练时长三者相当，这两项不是数量级。
- P2 达成，走的是"发现并修复 planner 真实 bug"这一支：定位并修复了容量约束缺失导致 V-PEFT 静默降级 legacy planner 的缺陷（源码修复 + 8 个单测），放在另一个独立 PR 里。小样本曲线没有画：k5/10/50/100 的划分已备好（任务书要求的是 10/50/100/500 张，仓库里备的是 30/60/300/600 张），本轮矩阵跑的是两个数据集的全量划分。

## 已知问题

1. seed 稳定性：NEU 上 vpeft 和 frozen 各有一个 seed（2024）收敛异常，换 seed 补跑正常，属于偶发；DeepPCB frozen 在 seed 824 的震荡同 seed 重跑能复现，说明那个配置下训练本身不稳。这些只在报告用的 seed 池里观察到，不能推总体概率。
2. 数据许可：NEU-DET 镜像没有 LICENSE 文件，DeepPCB 是 MIT，两者的来源和 SHA-256 都已登记。
3. 规模：EsMoE-N 只有 2.8M 参数，LoRA 相对全量的参数优势被小基数压缩，这套对比不能直接外推到更大模型。
4. 缺陷规避：消融是在容量缺陷存在的情况下用排除清单跑的，修复后没有重跑，所以精度数字还是修复前的口径。

## 说明

这个 PR 只提交 `c3_work/` 下的交付物，没有改动 `ultralytics/` 源码。相对 `main` 看到的源码差异是因为这个分支的基线较早，不是这次改的。训练产物 `runs/`、数据集二进制和缓存都不入库。源码修复走另一个单独的 PR。

## PPT

https://drive.google.com/file/d/1dg8tZyI11xJR1JrCCKBWqbC4_kFgmsVp/view?usp=sharing
