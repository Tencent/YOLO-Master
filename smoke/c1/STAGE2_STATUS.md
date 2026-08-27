# C1 Stage2：武器横评状态快照（2026-08-27 21:42）

## 横评对象（4 个武器，同超参）

与 Stage1 完全一致的超参（epochs=50, batch=8, seed=42, amp=False, pretrained=False, dense_eval）。

| 武器 | 结构 | 参数量 | 进度 | 当前最佳 mAP50 |
|---|---|---|---|---|
| EsMoE-P2-N | P2 + 开路由 | 2.898M | 35/50 | 0.19471 @ ep34（上升中） |
| v0.1-P2-N | P2 + 关路由 | 7.666M | 19/50（待续跑） | 0.15211 @ ep19 |
| UoMoE-N | UoMoE + 关路由 | 7.448M | 0/50 | — |
| UoMoE-P2-N | UoMoE + P2 | 7.566M | 0/50 | — |

## 关键工程修复（代码增量证据）

1. **best.pt 文件锁**：火绒 HipsDaemon 实时扫描锁住 `.pt`，导致 EsMoE-P2-N 在 27/50 中断。
   修复：`run_c1_weapons.py` 中对 `BaseTrainer.save_model` 打异常安全补丁，best.pt 写入失败仅跳过、不中断训练（last.pt / results.csv 照常）。
2. **CUDA OOM**：P2 变体峰值显存 9.44G > 8.19G 物理显存，douyin.exe + 杀软抢显存触发 OOM。
   修复：关闭非必要占显存进程，释放余量。

## 参数量自审（6 配置，全 nano 合规）

| 配置 | 参数量 |
|---|---|
| v0.1-N | 7.547M |
| EsMoE-N | 2.845M |
| EsMoE-P2-N | 2.898M |
| v0.1-P2-N | 7.666M |
| UoMoE-N | 7.448M |
| UoMoE-P2-N | 7.566M |

> 注：仓库内 `BudgetConstraint` 实为 V-PEFT(LoRA) 适配器参数约束，非 C1 检测硬预算脚本；故采用 `audit_params.py` 直接统计 6 武器参数量作合规证据。

## ETA

Stage2 全部完成约 2026-08-29 凌晨–上午。
