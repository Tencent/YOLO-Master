# A2OR：baseline（P0）与小目标标签分配实验归档

> **实验报告：** [baseline（P0）进度与标签分配消融实验](EXPERIMENT_REPORT_2026-09-13.md)。实验统计截至2026-09-13，文档更新于2026-09-14。`EXPERIMENT_STATUS.yaml` 中的64轮状态仅适用于早期 `baseline_gpu6g_retry` 实验。

## 云端实验进度与主要结果

本节汇总云端baseline及标签分配改进实验。结果采用已有评估记录；完整训练配置、日志和权重的归档状态在实验报告中单独说明。

| 实验 | 已有证据与结果 | 判断 |
|---|---|---|
| 云端配对 baseline，800/120轮、batch16、AdamW | 完成记录及权重清单覆盖至`epoch119.pt`；best AP/APs=20.111/11.802，last=19.982/11.579 | 120轮训练与面积评估已完成；完整复现包待归档 |
| 固定扩张 `[8,16)→24`，120轮预算 | epoch101–110逐轮APs差均值+0.0046，范围−0.088至+0.112；last差值+0.007仅有结果摘要，缺少完整评估表 | 后期接近baseline，未达到预期；磁盘满中断的是逐轮评估 |
| `axis_decay_cloud_120e_b16`，前60轮扩张、后60轮线性收窄方案 | 已有best形状评估；部分细长桶改善，但`0–8 × 16–32`从3.6569降至2.0777 | 已有分桶评估；总体APs及最终衰减配置待核验 |
| `dtk_explicit_120e_b16` | 已有best形状评估；`8–16 × 16–32` APs从6.4614降至5.9040，`16–32 × 16–32`从11.9703降至11.2436 | 多个形状分桶性能下降；总体APs及完整参数待归档 |
| C1 coverage，30轮预实验及120轮实验 | 120轮完成记录及25个形状分桶结果；部分细长桶改善，部分主要桶下降；总体APs仅有未优于baseline的定性记录 | 局部改善未转化为总体达标；总体APs精确数值缺失 |
| C2 初始非配对实验 | 训练配置差异显示C2有效优化器MuSGD、baseline为AdamW，warmup bias LR也不同 | 对照存在混杂，不能归因于C2；随后重新配对训练 |
| C2 与新baseline配对重跑，120轮 | best：ΔAPs=−0.130、ΔAPl=−1.695；last：ΔAPs=+0.116、ΔAPl=−3.132 | 未达到小目标改进目标，保留为负向消融结果 |

单位均为AP点；形状桶AP不能相加或按GT数量加权成总体APs。C2差值保留原评估输出，避免用已四舍五入的数值重算产生0.001点偏差。上述云端baseline不同于早期本机`baseline_b4_nbs64_120e`，不能混用名称或指标。

详细记录见[实验报告第4.7–4.10节](EXPERIMENT_REPORT_2026-09-13.md)、[云端形状对照表](reports/experiment_audit_2026-09-13/cloud_shape_comparison.md)、[实验记录附录](reports/experiment_audit_2026-09-13/cloud_source_excerpts.md)与[指标与来源索引](reports/experiment_audit_2026-09-13/cloud_history_evidence.json)。D1模型尚未实现；分辨率探针已实现，评估结果待补充。SMP/RP/MP等最小正样本方案处于设计阶段，未纳入已完成实验统计。

## 早期 GPU 限制 baseline 的历史归档

> **状态：DEGRADED BASELINE（退化基线）**  
> 本节记录的早期GPU限制实验不得作为“官方 `batch=16`、120 epoch 完整复现”的结果引用。

## 退化原因

该实验原计划使用 YOLO-Master v0.1-N 在完整 VisDrone2019-DET 上按以下协议训练：

- `imgsz=800`
- `epochs=120`
- `batch=16`
- `nbs=64`
- 完整 548 张验证集
- 约 6 GiB PyTorch GPU 显存上限

实际运行中发生 CUDA OOM，训练器依次自动降低物理 batch：

```text
batch=16 -> batch=8 -> batch=4
```

最终有效运行的 `args.yaml` 记录为 `batch=4`，`results.csv` 仅包含 64 个 epoch。因此，它既不满足官方 `batch=16`，也未完成计划的 120 epoch。

`nbs=64` 使 batch=4 时通过梯度累积获得约 64 张图像的名义优化器更新规模，但不能使其与物理 `batch=16` 严格等价。

## 结果定位

实验输出目录：

```text
runs/baseline_gpu6g_retry/
```

归档记录包含以下产物；原运行目录目前未保存在本地仓库：

- `results.csv`：64 行，最后记录为 epoch 64
- `weights/best.pt`
- `weights/last.pt`
- `weights/last_healthy.pt`
- 实际运行参数、训练批次图和标签图

以下目录对应未成功启动的实验：

- `runs/baseline_gpu6g/`
- `runs/baseline_gpu6g_restart/`

## 使用边界

可以用于：

- 收敛趋势与训练管线分析
- 受限显存条件下的探索性结果
- 调试、可视化和 checkpoint 检查

不得用于：

- 声称完成官方 batch=16 baseline
- 作为严格的120-epoch最终结果
- 与物理 batch=16 实验进行无条件等价比较
- 在未披露协议退化的情况下作为正式消融基线

如需正式实验，应从同一初始权重重新开始，并在启动前固定实际 batch。若硬件只能支持 batch=4，则所有对照组都应使用统一的 batch=4 协议，并将其称为“受限显存协议基线”。

## 小目标分辨率瓶颈探针

`probe_resolution_bottleneck.py` 使用同一个冻结 checkpoint，在相同的完整验证集上分别以 `imgsz=800` 和
`imgsz=1280` 评估。它复用 `compare_aps.py` 的原图面积分箱，同时报告官方 COCO `maxDets=100` 和
VisDrone 密集场景补充口径 `maxDets=300`。这是一项诊断，不是新的训练结果或结构消融。

```bash
python A2OR/probe_resolution_bottleneck.py \
  --checkpoint A2OR/runs/baseline_matched_vd100pct_s0_120e_b16_adamw_w8/weights/best.pt \
  --data A2OR/.runtime_data/visdrone_full_0764528ce5ce.yaml \
  --images /infinite/datasets/yqy/VisDrone/images/val \
  --labels /infinite/datasets/yqy/VisDrone/labels/val \
  --imgsz 800 1280 \
  --reference-imgsz 800 \
  --batch 4 \
  --device 0 \
  --workers 8 \
  --max-det 300 \
  --aps-gate 0.5 \
  --output A2OR/runs/baseline_resolution_probe.json
```

先在命令末尾添加 `--print-config` 可只检查参数和路径。若 `1280` 验证显存不足，应降低评估 batch；评估
batch 不改变指标定义。添加 `--sparse-sahi` 会运行可选的集成 Sparse SAHI 探针，其结果会单独标记为
推理流程变化，不与普通 `imgsz` 结果混为同一协议。
