# stage2_server_smoke · 服务器化冒烟（GPU 8×A800）

**阶段目标**：stage1 环境/数据就绪后，在 GPU 服务器上做最小训练闭环冒烟——验证 torch+CUDA+`pip install -e` 链路、三策略配置可达性、vpeft planner 决策与 LOVO/护栏记录路径，并产出可进报告的证据格式样例。冒烟=快（1-3 epoch），**数字不作数**。

**关键纪律**（沿用准入与 stage1）：
- 运行目录唯一，拒绝覆盖；复跑必须新 `--name`
- strict 不关；失败即报错，不静默降级
- 参数以 `runs/<name>/train/<strategy>/args.yaml`（resolved config）为准
- CLI 一律用本 env 的 yolo（`Path(sys.executable).parent/"yolo"`），不依赖 PATH（BF-01）

## 本目录文件

| 文件 | 状态 | 说明 |
|---|---|---|
| `server_run.py` | ✅ 已实跑（vpeft/full/frozen 全通） | `run_smoke.py` 服务器化：路径推断、`--runs-root`、`--tag`、拒绝覆盖、GPU 快照 + training.log 落盘 + summary.json |
| `planner_solver_notes.md` | ✅ 完成 | 源码级事实笔记：solver/OR-Tools/容量护栏/审计路径/LOVO 语义/参数清单（带文件:行号） |
| `bugfix_log.md` | ✅ BF-01 | 共享机 PATH 裸 `yolo` 命中错误 ultralytics（修复+泛化结论） |
| `smoke_cmds.sh` | ✅ 模板 | env 激活/三策略冒烟/证据提取命令 |
| `README.md` | 本文件 | 阶段说明与结果 |

## 冒烟结果（2026-09-07，GPU5，NEU-DET **全量 1620** train / 180 val，imgsz 640 / batch 8 / seed 824）

> 说明：冒烟 yaml 指向主划分（全量）而非 k10（k 档在 `shots/k*/`，正式实验用对应 yaml）；冒烟仅验证链路/可达性，epochs 不同不可直接对比。

| 策略 | run | epochs | 结果 | 时长(s) | 训练日志峰值显存 | 备注 |
|---|---|---|---|---|---|---|
| vpeft | `smoke_neuk10_vpeft_824b` | 2 | exit 0 / mAP50=0.029@ep2 | 142 | — | strict planner 通过（exclude 方案生效）；2ep 在 warmup(3.0) 内故偏低 |
| full_sft | `smoke_neuk10_full_824` | 1 | exit 0 / mAP50=0.205 | 95 | 4.54G | 全参 YOLO-master-n 仅 2.8M 参数 |
| frozen_backbone | `smoke_neufull_frozen_824` | 1 | exit 0 / mAP50=0.045 | 71 | 1.37G | freeze=11 冻结主干，显存显著低于全参 |

产物均在 `runs/<name>/`：`command.sh` / `training.log`（后两次）/ `train/<strategy>/args.yaml`(resolved) / `results.csv` / `summary.json`(GPU前后快照+exit+elapsed+末行指标)。

## 发现（P2 bug 素材，详见 `bugfix_log.md`）

- **BF-01**：共享服务器 PATH 裸 `yolo` 命中 base conda/他人项目（`chenzhong1/xuexiji_ocr/ultralytics`），8 个 `lora_*` 参数全不可识别、7.8s exit 1（易误判为代码问题）。修复：`Path(sys.executable).parent/"yolo"`。同修 `smoke/c3/run_smoke.py`。
- **stage3 决策点**：`warmup_epochs=3.0` 默认，正式实验 epochs 需 ≫warmup（50+）保证三策略同预算可比；YOLO-master-n 全参仅 2.8M——是否换 EsMoE-S/M 以拉开全量微调参数体量（task 书 P1"数量级优势"论证），stage3 定。

## 收尾检查单（Stage2）

- [x] yolo 8.4.101 / torch 2.6.0+cu124 / cuda True / 8×A800 可见
- [x] vpeft 冒烟（strict planner 通过，exit 0）
- [x] 三策略可达性冒烟全 exit 0；training.log 落盘
- [x] 证据格式验证：args.yaml(resolved) / summary.json(GPU+metrics) / command.sh / training.log
- [x] planner_solver_notes.md / bugfix_log.md（BF-01）
- [ ] git commit（含 BF-01 bugfix 到 `smoke/c3/run_smoke.py`）；push 待 GitHub 凭证

## 与后续 stage 关系

- stage3 用本目录 `server_run.py` 作单训练单元，队列并行 3 seed × 3 策略 × 双数据集；正式实验 epochs≫warmup、模型与 yaml（k 档）按设计选。
