# Stage2 武器横评状态（C1，2026-08-28 更新）

## 配置
- 4 武器：EsMoE-P2-N / v0.1-P2-N / UoMoE-N / UoMoE-P2-N
- 同超参：epochs=50, batch=8, imgsz=640, device=0, workers=0, seed=42,
  deterministic=True, pretrained=False, lora_r=0, optimizer=auto(SGD@0.01),
  amp=False, --no-sparse-eval(dense eval 修正 ES-MoE sparse 塌陷), val=True
- 公平性：与 Stage1 baseline (v0.1-N=0.1959 / EsMoE-N=0.19052) 严格同预算

## 进度（2026-08-28 14:10 续跑）
| 武器 | 状态 | 当前 |
|---|---|---|
| EsMoE-P2-N | ✅ 完成 50/50 | 见 results.csv |
| v0.1-P2-N | 🔄 续跑 19→50（resume from epoch 20）| 进行中 |
| UoMoE-N | ⏳ 待跑 | 0/50 |
| UoMoE-P2-N | ⏳ 待跑 | 0/50 |

## 防杀软文件锁补丁（关键）
火绒 HipsDaemon 实时扫描会锁住训练输出文件导致 PermissionError 崩溃。已对
`ultralytics.engine.trainer.BaseTrainer` 打三重异常安全补丁（run_c1_weapons.py）：
1. `save_model`：best.pt 写入失败跳过（last.pt 仍写，带 5 次重试）；last_healthy.pt 失败非致命
2. `_save_run_args`：args.yaml 写入失败跳过（纯记录文件）
3. last.pt 写入锁了重试 5 次再放弃（仅丢当前 epoch，不丢断点）

补丁只影响"文件保存"，不改变任何训练超参/数值，公平性不受影响。
