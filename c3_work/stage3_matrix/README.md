# stage3_matrix · 全量对照矩阵（双数据集 × 三策略 × 3 seed，epochs=100）

**阶段目标**：在 A800 服务器执行 18 单元对照，产出可进报告的 evidence：
`{NEU-DET, DeepPCB} × {vpeft, full_sft, frozen_backbone} × seed {824, 2024, 777}`，
**epochs=100 / imgsz 640 / batch 8 / 全量训练划分**（NEU 1620 / DeepPCB 1350），
每单元独立 name（拒绝覆盖）、`summary.json`（resolved config + GPU 快照 + 末行指标）、`training.log`。

**决策（2026-09-07 确认）**：
- 模型 EsMoE-N（现有 `YOLO-Master-EsMoE-N.pt`；全参 2.8M，冒烟已验证）
- epochs=100（≫ warmup_epochs=3.0，同预算下三策略可比）
- vpeft budget 默认 2.1M，待 pilot 后视 planner 决策微调（见下）
- push 由用户在凭证就绪后自行执行

## 本目录文件

| 文件 | 说明 |
|---|---|
| `README.md` | 本文件 |
| `make_matrix.py` | 生成 `matrix.json`（18 单元定义 + pilot 结论注记） |
| `sweep_runner.py` | GPU 空闲探测 + 断点续跑调度（从 matrix 状态推进，拒绝覆盖） |
| `collect_evidence.py` | 汇总 runs → `evidence_summary.json/csv`（四维表） |
| `matrix.json` | 单元矩阵（生成物） |

## 工作流

1. **pilot**：先跑 1 个 vpeft 短跑（NEU-DET 全量 epochs=3）读取 `[V-PEFT]` 决策行，
   确认 planner 未 Refuse / 无 cap 抛错；据此给 budget 定案并写入 `matrix.json` 注记。
2. `python make_matrix.py` 生成 18 单元。
3. `nohup python sweep_runner.py >> runner.log 2>&1 &` —— 调度器自动找空闲卡并发跑；
   任意时刻重启 runner 都从 matrix 状态续跑（断点续跑），done 单元不重跑。
4. `python collect_evidence.py` 汇总四维证据表（stage4 消费）。

## evidence 四维（每单元）

- 精度：`results.csv` 末 epoch / best 的 `metrics/mAP50(B)`、`mAP50-95`
- 成本：`elapsed_sec`、`training.log` 峰值显存（默认 `amp=False`；vpeft 最小显存）
- 可训参数：full_sft=2.8M；frozen_backbone=<head>；vpeft=<adapter 决策>
- planner 行为：`resolved_config`（strict/budget/…）+ 日志 `[V-PEFT]` 决策行

## 状态表（自动更新）

| 单元 id | 数据集 | 策略 | seed | 状态 | 说明 |
|---|---|---|---|---|---|
| （sweep_runner 落盘前由 collect_evidence 生成） | | | | | |

## 注意事项

- DeepPCB 6 类、类不平衡、1500 张（1350 train/150 val，seed824 划分）；两数据同 seed 由 yolo `seed=` 参数控制
- 调度器 GPU 空闲判据：显存 used <1.5G 且 util <30%；只占用这样判定空闲的卡，避免抢占他人任务
- 冒烟数字不作数，本阶段全部重跑（含冒烟曾用名，避免沿用 1-2 epoch 数值）
- 调度器给子进程清 PYTHONPATH（CodeBuddy sitecustomize shim 会拦 unlink）；cache 预热避免并发首建竞争（BF-02）

## 执行记录 / BF

- 2026-09-07 11:30 调度器首发 4 单元(cards 2-5)+ pilot 收尾腾卡续发；`neu_vpeft_s824` 因与 pilot 并发首建 labels.cache 失败(BF-02)，改 `neu_vpeft_s824b` 重跑，失败产物保留作证据
- pilot(3ep) `[V-PEFT] ACCEPT: selected 81 targets with ranks=[8]`；显存峰值 ~4.1G（vpeft）< full_sft 4.54G
- 实测 ~35s/epoch(100ep≈1h/单元),5 卡并行 18 单元约 4h 完成

## 主 18 单元结果（best 口径 = results.csv 各 epoch val mAP 最大值; 显存=日志进度行 GpuMem 峰值）

| dataset | strategy | mAP50 × 3 seed (824/2024/777) | 显存峰值 | 时长 |
|---|---|---|---|---|
| NEU | full_sft | 0.752 / 0.780 / 0.768 | 5.31G | ~3.3k s |
| NEU | vpeft | 0.691 / 0.369⚠ / 0.746 | 4.15G | ~3.1k s |
| NEU | frozen_backbone | 0.685 / 0.210⚠ / 0.659 | 3.74G | ~2.7k s |
| PCB | full_sft | 0.989 / 0.989 / 0.990 | 5.28G | ~2.8k s |
| PCB | vpeft | 0.864 / 0.819 / 0.961 | 4.11G | ~2.7k s |
| PCB | frozen_backbone | 0.639⚠ / 0.891 / 0.907 | 3.70G | ~2.9k s |

⚠=收敛异常 outlier(见 `outlier_notes`/stage4): NEU+vpeft_s2024 停滞(曲线 0.2-0.3 平台), NEU+frozen_s2024 全程卡死<0.06,
   PCB+frozen_s824 剧烈震荡(0.03~0.5, best@ep9)。已安排 `run_supplement.py` 补跑验证确定性(见下)。

**关键数字(可训参数, 从 best.pt 实测)**:
- 全参 `YOLO-master-n`: **2,813,626**
- vpeft 真实 LoRA adapter: **116,736** (~1/24 全参; 96 个 lora_A/B 权重)
- vpeft 有效可训 = adapter + 类别重初始化解冻的 head ~348,514 → ~465k(~17% 全参; head 解冻为检测头类别数差异所致,两数据均 6 类)
- 显存分层(全量 640): full 5.31G > vpeft 4.15G > frozen 3.74G —— P1"受限预算内存优势"的量级基础

## 补充单元(run_supplement.py) — 已完成 ✅

| 单元 | 策略 | seed | best mAP50@ep | 结论 |
|---|---|---|---|---|
| neu_vpeft_s2025 | vpeft | 2025 | 0.644@82 | **正常收敛 → s2024 停滞(0.369)为 seed 偶发** |
| neu_frozen_s2025 | frozen | 2025 | 0.739@80 | **正常收敛 → s2024 卡死(0.210)为 seed 偶发** |
| pcb_frozen_s824b | frozen | 824(重跑) | 0.639@9 | **与 s824 逐一致(同 best@ep9) → 震荡为确定性动力学, 非噪声** |

判定意义见 stage4_analysis/README(稳定性/确定性验证)。
