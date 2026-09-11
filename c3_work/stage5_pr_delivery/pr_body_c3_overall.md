<!--
C3 整体结项交付 PR 正文（分支 c3-vpeft-smoke → base Tencent:main，14 个 commit）。
标题建议：
[犀牛鸟-C3]：V-PEFT 小样本工业缺陷检测结项交付（环境/数据 · 冒烟 · 18+3 单元对照 · 统计分析 · 结项报告）
粘贴范围：从 "## 概述" 开始到文件末尾（含 "## PPT"）。
-->

## 概述

本 PR 是 C3 课题「V-PEFT 小样本工业缺陷检测（YOLO-Master EsMoE-N）」的**整体结项交付**：从环境重建、数据准备，到 A800 服务器冒烟、21 单元全量对照矩阵、统计分析与结项报告，全部放在 `c3_work/` 下按阶段分目录，可逐条复现。训练产物（`runs/`）与数据集不入库，只入库脚本、日志、汇总表与文档。

**一句话结论**：在 EsMoE-N（全参 2.8M）+ NEU-DET / DeepPCB 两个工业缺陷数据集上，V-PEFT（planner 选层）用 **4.15% 全参的 LoRA adapter** 拿到与全量微调相差约 0.07（NEU）/ 0.11（PCB）的 mAP50，显存少 1.2G；但同预算 3 seed 下全量微调的 seed 稳健性最好（sd ≤ 0.014），参数高效/冻结两类方法存在极端 seed 偶发（已补跑验证），时间维度三者相当（瓶颈在数据管线）。

## 交付内容与目录

| 阶段 | 目录 | 主要内容 | 状态 |
|---|---|---|---|
| stage1 环境与数据 | `c3_work/stage1_env_data/` | conda env(py3.11) + editable 安装、NEU-DET / DeepPCB 下载与 YOLO 化、k5/10/50/100 few-shot 划分、许可与 SHA-256 登记 | ✅ |
| stage2 服务器化冒烟 | `c3_work/stage2_server_smoke/` | `server_run.py`（路径推断/拒绝覆盖/GPU 快照/summary.json）、三策略冒烟、planner 源码笔记、BF-01 修复 | ✅ |
| stage3 全量对照矩阵 | `c3_work/stage3_matrix/` | 18 单元矩阵（双数据集 × 三策略 × 3 seed，epochs=100）+ 3 个补充单元、调度/续跑/证据汇总工具 | ✅ |
| stage4 统计分析 | `c3_work/stage4_analysis/` | 3+seed 均值/sd/95%CI、逐 seed 配对差、稳健口径、planner 审计汇总、四维对照表 | ✅ |
| stage5 结项交付 | `c3_work/stage5_pr_delivery/` | 结项报告、复现包、汇报 PPT（含生成脚本）、上游修复 PR 文案 | ✅ |

## 1. 环境与数据（stage1）

- **环境**：独立 conda env `python 3.11`（不污染团队共享 env），`pip install torch==2.6.0 / torchvision==0.21.0`（内网镜像）→ `requirements.txt` → `pip install -e <repo>`；`yolo 8.4.101`、`torch 2.6.0+cu124`、`cuda True`、8×A800 可见。
- **NEU-DET**：1800 张灰度 / 6 类；train 1620 / val 180；few-shot k5=30、k10=60、k50=300、k100=600（每类 5/10/50/100）。镜像**无显式 LICENSE**（学术用途，已如实登记）。
- **DeepPCB**：1500 test 图（640×640 灰度 jpg）/ 6 类；train 1350 / val 150；标注 `x1 y1 x2 y2 cls` 已抽验匹配。few-shot **类别极不平衡**（k5=8、k10=13、k50=70、k100=142，稀有类 open/short 不足）→ 小样本曲线主线用 NEU-DET，DeepPCB 走全量划分。许可 **MIT**。
- **留档**：`datasets/manifest.md`（来源 / 许可 / 统计 / SHA-256 / 划分说明），下载与转换日志同目录。

## 2. 服务器化冒烟（stage2）

- `server_run.py`：数据/模型/输出全参数化，输出目录唯一且拒绝覆盖、复跑必须新 `--name`；落盘 `command.sh` / `training.log` / `args.yaml`(resolved config) / `results.csv` / `summary.json`（GPU 前后快照 + exit + elapsed + 末行指标）。
- 三策略冒烟全部 `exit 0`（NEU 全量 1620/180，imgsz 640、batch 8、seed 824）：vpeft 跑通 strict planner，frozen 显存明显低于全参。**冒烟只验证链路，数字不作数**，正式对照在 stage3 全部重跑。
- **BF-01**：共享机 PATH 上裸 `yolo` 命中他人项目的 ultralytics，导致 8 个 `lora_*` 参数全部不可识别、7.8s 退出 1（极易误判成代码 bug）。修复为 `Path(sys.executable).parent/"yolo"`，同步修 `smoke/c3/run_smoke.py`。另记录 BF-02（并发首建 `labels.cache` 竞争）与处置方式。
- `planner_solver_notes.md`：solver / OR-Tools / 容量护栏 / 审计路径 / LOVO 语义的参数级事实笔记（带文件:行号），是后续解读 `[V-PEFT]` 日志的依据。

## 3. 全量对照矩阵（stage3）

- **矩阵**：`{NEU-DET, DeepPCB} × {full_sft, vpeft, frozen_backbone} × seed {824, 2024, 777}`，共 18 单元，**epochs=100 / batch 8 / imgsz 640 / amp=false**（≫ warmup 3.0，保证三策略同预算可比）；模型 EsMoE-N。
- **工具**：`make_matrix.py` 生成矩阵、`sweep_runner.py` 按"显存 <1.5G 且 util <30%"探测空闲卡并断点续跑（done 单元不重跑，避免抢占他人任务）、`collect_evidence.py` 汇总四维证据（精度 / 成本 / 可训参数 / planner 行为）。
- **结果（best 口径 = 各 epoch val mAP 最大值）**：NEU full 0.752/0.780/0.768、vpeft 0.691/0.369⚠/0.746、frozen 0.685/0.210⚠/0.659；PCB full 0.989/0.989/0.990、vpeft 0.864/0.819/0.961、frozen 0.639⚠/0.891/0.907。峰值显存 full 5.3G > vpeft 4.15G > frozen 3.7G。
- **补充单元（3 个）**：`neu_vpeft_s2025` 0.644@82、`neu_frozen_s2025` 0.739@80 正常收敛 → NEU 上 s2024 的停滞/卡死判定为 **seed 偶发**；`pcb_frozen_s824b` 与 s824 **逐一致（best 仍在 ep9）** → 该 (seed, 冻结配置) 的震荡是**确定性动力学**，不是采样噪声。
- 实测约 35s/epoch（100ep ≈ 1h/单元），5 卡并行 18 单元约 4h 跑完。

## 4. 统计分析与结论（stage4）

主 18 单元 + 3 补充单元全部完成，同预算对照；口径：指标取 best（非末行，PCB frozen s824 末行 0.496 vs best@ep9 0.639）、显存取日志 GpuMem 峰值。

| 数据集 | 策略 | 可训参数（占比） | 峰值显存 | mAP50(best) mean±sd | 配对差 vs full |
|---|---|---|---|---|---|
| NEU | full_sft | 2,813,626（100%） | 5.31G | 0.767±0.014 | — |
| NEU | vpeft | adapter 116,736（4.15%）/ 含解冻 head 465,250（16.5%） | 4.16G | 0.694±0.052\* | −0.073 |
| NEU | frozen_backbone | 1,906,822（67.8%） | 3.74G | 0.694±0.041\* | −0.073 |
| PCB | full_sft | 同上 | 5.27G | 0.989±0.000 | — |
| PCB | vpeft | 同上 | 4.11G | 0.881±0.072 | −0.108 |
| PCB | frozen_backbone | 同上 | 3.71G | 0.769±0.151 | −0.177 |

\* NEU 稳健口径（剔除偶发 seed 2024，n=3）。不剔除时主表 3 seed：vpeft 0.612±0.167、frozen 0.573±0.245。

- **精度**：PCB 全量极稳（0.989±0.001）；NEU 上 full 比 vpeft/frozen 高约 0.07 mAP50，且 seed 稳健性明显更好（sd 0.014 vs 0.04~0.05）——小模型 + 小数据下全量微调仍是最稳基线。
- **参数/显存**：vpeft 仅 4.15% 全参、显存 −1.15G；frozen 67.8% 可训、显存 −1.6G。受限预算场景下 vpeft 以小幅精度代价换数量级参数节省，收益成立；frozen 省得有限还牺牲稳定性。
- **时长**：同 epochs 三策略基本一致（~45-55min/单元），参数高效方法在本数据规模未体现训练加速。
- **planner 审计**：6/6 单元决策一致 `ACCEPT`（81 targets，rank=8，base rank 不升），同类 mismatch 的检测头（80→6 类）重初始化解冻 ~348,514 参数；容量不足层经 `lora_exclude_modules` 规避，审计见 `stage4_analysis/planner_audit_summary.md`。

## 5. 结项报告 / 复现包 / 汇报 PPT（stage5）

- `report_final.md`：结项报告（结论速览 / 方法与设置 / 证据链四维 / 局限与风险 / 复现 / PR 映射）。
- `reproduction/`：复现包（env 与 paths 模板、数据转换脚本、三策略训练命令样例、matrix 与证据汇总、统计表与审计、许可与 SHA 记录），由 `assemble_reproduction.py` 从 stage1-4 收集。
- `C3_结项汇报_初稿.pptx` / `.pdf`：10 页汇报稿，`gen_slides.py` 为可重生成的生成脚本（链接见文末 `## PPT`）。

## 6. 期间定位的上游缺陷（独立源码 PR）

实验过程中定位到 V-PEFT 的一个真实缺陷并已做源码修复，**该修复不混进本 PR**，单独开在基于官方 `af961b9` 的分支 `fix/vpeft-capacity-guard`（2 commit）→ 上游候选 PR **#279**：

- 七个硬约束都没有建模层容量，而 plan 校验会拒绝 `rank > min(in, out)` 的目标 → 投影与校验互相矛盾（`0.conv` / `routing_network.2` / `25.dfl.conv` 被选中后 `apply_lora` 抛 `ValueError`），`vpeft` 后端下异常被吞、**静默回退 legacy planner**（strict 下直接失败）。
- 修复内容：新增 `C_cap` 硬约束（`RankCapacityConstraint`）+ `capacity_excluded` 审计字段与可 grep 日志 `[V-PEFT] capacity-excluded: ...`、校验阶段一次报全违规目标、显式目标列表/过滤跳过的层改为冻结对齐 manual 后端；新增 8 个单测。
- 本轮消融是用 `lora_exclude_modules` 规避该缺陷跑出来的，**未用修复后代码重跑**；PR #279 的完整说明见 `c3_work/stage5_pr_delivery/pr_body_capacity_guard.md`。

## 复现

```bash
# 1) 环境与数据（stage1）
bash c3_work/stage1_env_data/env_setup.sh
bash c3_work/stage1_env_data/download_datasets.sh
python c3_work/stage1_env_data/prepare_neu_det.py && python c3_work/stage1_env_data/prepare_deeppcb.py

# 2) 服务器化单次训练（stage2）：模板见 smoke_cmds.sh，runner 用法见 server_run.py --help
python c3_work/stage2_server_smoke/server_run.py --help

# 3) 矩阵与证据（stage3）
python c3_work/stage3_matrix/make_matrix.py
nohup python c3_work/stage3_matrix/sweep_runner.py >> runner.log 2>&1 &
python c3_work/stage3_matrix/collect_evidence.py

# 4) 统计与审计（stage4）
python c3_work/stage4_analysis/analysis_stats.py
```

路径与预算以 `c3_work/stage5_pr_delivery/reproduction/configs/paths.env`、`train_commands_example.sh` 为准；`runs/`、数据集与 `*.cache` 已在 `c3_work/.gitignore` 中排除。

## 已知局限与风险

1. **seed 稳定性**：NEU 上 vpeft / frozen 各出现 1/4 seed 极端收敛异常（frozen 卡死 <0.06、vpeft 停滞 0.37），seed2025 补跑确认属 seed 偶发；PCB frozen 在 seed 824 的震荡同 seed 重跑确定性复现，说明该 (seed, 配置) 下训练不稳。**结论限定在报告 seed 池，不推总偶发概率**。
2. **数据许可**：NEU-DET 镜像无显式 LICENSE；DeepPCB 为 MIT。均已登记来源与 SHA-256。
3. **规模局限**：EsMoE-N 仅 2.8M，LoRA 相对全量的参数量优势被小基数压缩；三策略时长无差异（数据管线瓶颈）。
4. **缺陷规避**：本轮消融在底层缺陷存在的前提下用 `lora_exclude_modules` 规避完成，修复后未重跑（见第 6 节）。

## 范围声明

- 本 PR 只提交课题交付物（`c3_work/` 下的脚本、日志、汇总表、文档、汇报稿），**不改动** `ultralytics/` 源码；相对 `main` 出现的源码差异来自本分支基线较早，不代表改动意图。
- 不提交训练产物 `runs/`、数据集二进制与缓存。
- 上游源码修复走独立 PR #279。

## PPT

https://drive.google.com/file/d/1dg8tZyI11xJR1JrCCKBWqbC4_kFgmsVp/view?usp=sharing
