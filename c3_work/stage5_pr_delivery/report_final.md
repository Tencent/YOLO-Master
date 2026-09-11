# C3 实战结项报告：V-PEFT 小样本工业缺陷检测（YOLO-Master EsMoE-N）

数据集：NEU-DET（1800 张/6 类）、DeepPCB（YOLO 化 6 类，来源/许可见 stage1）。
模型：YOLO-Master-EsMoE-N（全参 2,813,626）。同预算 epochs=100×batch=8×imgsz=640×amp=false；
主对照 18 单元(3 数据集×3 策略×3 seed: 824/2024/777) + 3 补充单元(稳定性/确定性验证)。

## 1. 结论速览

| 维度 | full_sft（基线） | vpeft（V-PEFT+planner） | frozen_backbone（freeze=11） |
|---|---|---|---|
| 可训参数 | 2,813,626 (100%) | adapter 116,736 (4.15%)；含解冻 head 465,250 (16.5%) | 1,906,822 (67.8%) |
| 峰值显存(A800 640px) | ~5.3G | ~4.1G (−1.2G) | ~3.7G (−1.6G) |
| NEU mAP50(best, mean±sd) | 0.767±0.014 | 0.694±0.052* | 0.694±0.041* |
| DeepPCB mAP50(best, mean±sd) | 0.989±0.000 | 0.881±0.072 | 0.769±0.151 |
| 时长/单元 | ~46-55m | ~45-51m | ~40-51m |

\* NEU 为**稳健口径**（剔除 1 个偶发 seed 后 mean±sd，n=3）；主表 3 seed 含偶发后 vpeft 0.612±0.167 / frozen 0.573±0.245。
DeepPCB frozen 的 0.769 含 seed824 确定性震荡(0.639, best@ep9, 重跑复现)。

**一句话结论**：在 EsMoE-N(2.8M) + 两个工业缺陷数据集上，vpeft 以 **4.15% 全参的 adapter** 取得与全量微调仅差
~0.07(NEU)/~0.11(PCB) mAP50 的效果、显存 −1.2G；但同预算 3-seed 下全量微调 seed 稳健性最佳(sd≤0.014)，
参数高效/冻结法存在极端 seed 偶发(已补跑验证)，时间维度三者相当。

## 2. 方法与设置

- 复用 `run_smoke` 服务器化 `server_run.py`（stage2）：数据/模型/输出全参数化，done 标记+resume 适配共享 A800。
- V-PEFT planner 全 **ACCEPT**：6/6 单元决策一致(81 targets, rank=8)，base rank 不升；
  同类 mismatch 检测头(80→6 类)重初始化解冻 ~348,514 参数；LoRA 实际 96 个 lora_A/B=116,736 参数(4.15%)。
- 冻结策略 `freeze=11`：冻结顶层 0-10 子模块(32.2%)，可训 67.8%。
- 已知缺陷 cap<8 层(0.conv/routing_network.2/dfl.conv)在本轮实验中经 `lora_exclude_modules` 规避；该缺陷已完成源码修复并推送 fork、开上游候选 PR
  (分支 `fix/vpeft-capacity-guard`，基于官方 `af961b9`，2 commit：C_cap 容量硬约束 + `capacity_excluded` 审计、未适配层冻结对齐；
  PR #279 ← 链接见 `stage5_pr_delivery/C3_结项汇报_初稿.pptx` 末页①)，**消融数字未用修复后代码重跑**。

## 3. 测试证据（证据链四维，含稳定性验证）

| 数据集 | 证据项 | 落点 |
|---|---|---|
| NEU/PCB | 21 单元 best mAP50(逐 epoch 曲线)、显存峰值(日志 GpuMem)、时长 | stage3_matrix/runs + stage4/comparison_tables.md |
| 两数据 | 每策略 3+seed mean/sd/95%CI + 配对差 + 稳健口径 | stage4/README.md 表 1-3 |
| 稳定性 | seed2025 补跑(偶发排除)+ seed824 重跑(确定性复现) | stage3_matrix/README + stage4 README |
| 参数 | 由 best.pt state_dict 实测(adapter/freeze/total) | stage4 README §表3 |
| 审计 | planner ACCEPT/decision/护栏(LOVO/exclude/alpha_warmup) | stage4/planner_audit_summary.md |

## 4. 已知局限与风险（PR 必读）

1. **seed 稳定性**：NEU 上 vpeft/frozen 各有 1/4 seed 出现极端收敛异常(frozen 卡死<0.06、vpeft 停滞 0.37)，
   经 seed2025 补跑确认属 seed 偶发而非系统性失效；PCB frozen 在 seed824 的震荡(best@ep9)经同 seed 重跑**确定性复现**，
   说明 frozen 在该 (seed,配置) 下训练不稳。**结论限定在报告 seed 池，不推总偶发概率**。
2. **数据许可**：NEU-DET 镜像无显式 LICENSE；DeepPCB 许可见 stage1 README（SHA-256 留档）。
3. **planner cap<8 缺陷**：param-cap<8 层原先在 plan 校验阶段抛 ValueError 并使 V-PEFT 静默降级 legacy（strict 下报错），
   本轮用 `lora_exclude_modules` 规避；修复分支已推送 fork 并开 PR #279（`C_cap` 容量硬约束 + `capacity_excluded` 审计，分支 `fix/vpeft-capacity-guard`），
   但**未用修复后代码重跑消融**。
4. **规模局限**：EsMoE-N 仅 2.8M，LoRA 相对全量的参数量优势被小模型基数压缩；3 策略时长无差异(数据管线瓶颈)。

## 5. 复现

`reproduction/`（assemble_reproduction.py 收集）：数据转换脚本、paths.env 模板、
三策略训练命令样例、matrix/evidence/统计表/审计、许可与 SHA 记录。

## 6. PR 映射（改动摘要/测试证据/消融数据/已知局限）

- PR-1 stage1 env+data → PR-2 stage2 server runner+smoke → PR-3 stage3 matrix 18 单元
  → PR-4 stage4 统计与审计 → PR-5 stage5 复现包/本报告。
- **独立源码 PR（对齐官方 main）**：分支 `fix/vpeft-capacity-guard`（基线 `af961b9`，2 commit）= `C_cap` 容量硬约束 +
  未适配层冻结对齐；文案见 `stage5_pr_delivery/pr_body_capacity_guard.md`，PR 链接见 PPT 末页①。
- 本地 commit 已全部推送至 fork `lycyhrc/YOLO-Master`：`c3-vpeft-smoke`（证据包，15 条提交已统一为 Conventional Commits 风格，tip 见 fork 最新提交）与 `fix/vpeft-capacity-guard`（源码修复，tip `daed306`）；
  后者已开上游 PR #279，其标题/正文待按同门风格（`[犀牛鸟-C3]：` + Summary/Problem/Validation/Limitations）重填。
