# C3 stage4: 统计分析与证据结论

依据 stage3 主 18 单元（NEU-DET / DeepPCB × full_sft/vpeft/frozen_backbone × seed 824/2024/777）
同预算对照（epochs=100, batch=8, imgsz=640, amp=false, 同优化器）做统计与判读。

## 口径与红线

- **指标** = results.csv 各 epoch val mAP 的 **best**（max），非末行（部分单元后期过拟合/崩坏，
  如 NEU frozen s2024 末行 0.023 vs best 0.210，取 best 避免系统性低估）；
  mAP50-95 同步记录但主要判读用 mAP50。
- **显存** = training.log 进度行 GpuMem 峰值（进程退出后采样失效）。
- 提升/下降判读需 ≥3 seed；样本 n=3 时 95%CI 半宽用 t=3.18，**CI 含 0 不判显著**。
- 收敛异常（outlier）单元保留于统计（诚实呈现策略稳定性），并单独标注:
  `neu_vpeft_s2024` 停滞、`neu_frozen_backbone_s2024` 卡死、`pcb_frozen_backbone_s824` 震荡。
  补充验证：`neu_vpeft_s2025`/`neu_frozen_s2025`(新 seed) + `pcb_frozen_s824b`(同 seed 重跑)，
  由 `../stage3_matrix/run_supplement.py` 待卡自动执行。

## 使用方法

```bash
# 生成 comparison_tables.md(统计表), 结果含主18+已完成的补充单元
python analysis_stats.py
```

## 四维结论要点(待最终证据后定稿)

- 精度: PCB 三策略均高位且 full 极稳(0.989-0.990); NEU 上 full > vpeft > frozen 且 full/vpeft
  均值差 ~0.03-0.07, frozen 与 seed 强相关(稳定性不足)。
- 参数: full 2,813,626 (100%) / frozen(freeze=11) 1,906,822 (67.8%) /
  vpeft adapter 116,736 (4.15%) + 类别重初始化解冻 head 348,514 → 有效 465,250 (16.5%)。
- 显存: full 5.31G > vpeft 4.15G > frozen 3.74G (A800 64G 视角量级差小,但对低端卡有意义)。
- 时长: 同 epochs 预算下三策略相近(~1h/单元), vpeft 未显著省时(瓶颈是数据管线)。

## 待办(收尾检查单)

- [x] analysis_stats.py 生成 comparison_tables.md(主18基线)
- [ ] 补充单元(s2025/b824b)完成后并入统计并定稿 comparison_tables.md
- [ ] planner_audit_summary.md: Accept/Adapt/Refuse 分布 + LOVO/ΔmAP(从 vpeft 单元日志提取)
- [ ] outlier 分析段(曲线/原因/结论)写入 README
- [ ] commit push fork c3-vpeft-smoke
