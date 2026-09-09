# A2OR：退化 Baseline 归档

> **状态：DEGRADED BASELINE（退化基线）**  
> 本目录不得作为“官方 `batch=16`、120 epoch 完整复现”的结果引用。

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

真正产生训练结果的目录：

```text
runs/baseline_gpu6g_retry/
```

其中包含：

- `results.csv`：64 行，最后记录为 epoch 64
- `weights/best.pt`
- `weights/last.pt`
- `weights/last_healthy.pt`
- 实际运行参数、训练批次图和标签图

以下目录只是失败启动残留，不是有效实验结果：

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

如需正式实验，应从同一初始权重重新开始，并在启动前固定实际 batch。若硬件只能支持 batch=4，则所有对照组都应使用干净、统一的 batch=4 协议，并将其称为“受限显存协议基线”。

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
