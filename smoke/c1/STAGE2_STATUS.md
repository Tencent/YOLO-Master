# Stage2 武器横评状态（C1，2026-09-01 更新）

## 配置
- 4 武器：EsMoE-P2-N / v0.1-P2-N / UoMoE-N / UoMoE-P2-N
- 同超参：epochs=50, batch=8, imgsz=640, device=0, workers=0, seed=42,
  deterministic=True, pretrained=False, lora_r=0, optimizer=auto(SGD@0.01),
  amp=False, --no-sparse-eval(dense eval 修正 ES-MoE sparse 塌陷), val=True
- 公平性：与 Stage1 baseline (v0.1-N=0.1959 / EsMoE-N=0.19052) 严格同预算

## 进度（全部完成于 2026-08-29）
| 武器 | 状态 | 结果（mAP50 / mAP50-95） |
|---|---|---|
| EsMoE-P2-N | ✅ 完成 50/50 | 0.20773 / 0.11405 |
| v0.1-P2-N | ✅ 完成 50/50 | 0.20160 / 0.11081 |
| UoMoE-N | ✅ 完成 50/50 | 0.19626 / 0.10570 |
| UoMoE-P2-N | ✅ 完成 50/50 | 0.20503 / 0.11337 |

## 配套实验（同配置，已完成）
- 三 seed 稳健性（EsMoE-P2-N）：seed42=0.20773 / seed0=0.19990 / seed1=0.19673（mAP50）
- 基线对照（c1_onoff，50 epoch）：v0.1-N=0.19590 / EsMoE-N=0.19041（mAP50）
- 与 Stage1 5-epoch smoke（mAP50≈0.05）形成完整训练曲线，验证从零训练收敛趋势

## COCO APs 评估口径
- 已与导师确认口径正确：small = 实例面积 < 1024，maxDets = 500（标准 COCO 定义）
- EsMoE-P2-N 已生成 `aps_result.json`；其余武器 APs 评估为后续补充项（不影响主结论）

## 防杀软文件锁补丁（关键）
火绒 HipsDaemon 实时扫描会锁住训练输出文件导致 PermissionError 崩溃。已对
`ultralytics.engine.trainer.BaseTrainer` 打三重异常安全补丁（run_c1_weapons.py）：
1. `save_model`：best.pt 写入失败跳过（last.pt 仍写，带 5 次重试）；last_healthy.pt 失败非致命
2. `_save_run_args`：args.yaml 写入失败跳过（纯记录文件）
3. last.pt 写入锁了重试 5 次再放弃（仅丢当前 epoch，不丢断点）

补丁只影响"文件保存"，不改变任何训练超参/数值，公平性不受影响。
