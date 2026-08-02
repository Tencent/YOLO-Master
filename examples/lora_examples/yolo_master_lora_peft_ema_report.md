# YOLO-Master PEFT LoRA EMA 同步实验报告

本报告记录 `peft_ema_sync_rtx4060_v1` 协议。该协议用于验证 PEFT LoRA 的非张量 `scaling`
状态同步到 EMA 后，在 Brain Tumor 和 VisDrone 垂类场景中的 rank 扫描结果。

六组正式实验基于仓库提交 `a510883` 加本地 PEFT EMA 修复运行；实验完成后，修复提交才重放到
更新后的 `upstream/main`。因此结果应以本报告列出的完整协议为准，不能套用后续默认配置解释。

## 问题与修复

启用 `lora_alpha_warmup` 后，在线模型的 LoRA `scaling` 会随 epoch 增长，但 PEFT 0.19.1 将该状态
保存在普通 Python 字典中，而不是 `state_dict` 张量。标准 EMA 更新因此不会复制它，导致在线模型使用
LoRA、EMA 验证模型却保持零缩放。典型现象是训练 loss 下降，但 mAP 持续下降或归零。

修复在以下生命周期边界同步在线模型与 EMA 的 LoRA `scaling`：

- 每个 epoch 更新 alpha warmup 后；
- 验证前；
- checkpoint 序列化前；
- 断点续训恢复后。

## 实验环境与协议

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（8188 MiB）
- Python：3.11.15
- PyTorch：2.13.0+cu126
- PEFT：0.19.1
- 模型：YOLO-Master-EsMoE-N 预训练权重
- Rank：`r=4,8,16`，保持 `lora_alpha=2*r`
- Backend：配置为 `auto`，实际解析为 `peft`
- AMP：关闭，避免把数值稳定性问题混入 EMA 修复验证
- Router/gating：不纳入 LoRA 目标模块

Brain Tumor 使用全部训练集、`imgsz=640`、`batch=8`、最多 40 epochs、`patience=15`、
`lora_alpha_warmup=3`。VisDrone 使用 20% 训练集、完整验证集、`imgsz=640`、`batch=2`、
30 epochs、`lora_alpha_warmup=5` 和多尺度训练。

## Rank 扫描结果

| 数据集 | Rank | 完成轮数 | 最佳轮次 | mAP50 | mAP50-95 | 可训练参数 | Adapter 参数 | 时间（分钟） | 日志峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Brain Tumor | 4 | 17 | 2 | 0.40754 | 0.26097 | 409,174 | 64,000 | 10.38 | 3.79 GB |
| Brain Tumor | 8 | 40 | 33 | 0.47810 | **0.34647** | 473,174 | 128,000 | 25.38 | 3.83 GB |
| Brain Tumor | 16 | 17 | 2 | 0.47357 | 0.31845 | 601,174 | 256,000 | 10.62 | 3.84 GB |
| VisDrone | 4 | 30 | 20 | 0.09152 | 0.04601 | 410,734 | 64,000 | 73.14 | 8.68 GB |
| VisDrone | 8 | 30 | 20 | 0.09799 | 0.04926 | 474,734 | 128,000 | 69.06 | 8.69 GB |
| VisDrone | 16 | 30 | 28 | 0.11454 | **0.05797** | 602,734 | 256,000 | 76.62 | 8.72 GB |

峰值显存来自训练日志的 `GPU_mem` 最大值；不同 CUDA/PyTorch 版本的内存统计口径可能不同。
完整机器可读结果见 `yolo_master_lora_peft_ema_results.csv`。

## 结论

- Brain Tumor 推荐 `r=8`：mAP50-95 最高，且比 `r=16` 少 128,000 个 Adapter 参数。
- VisDrone 推荐 `r=16`：密集小目标场景从更大的 LoRA 容量中获得了持续收益。
- 两个场景不存在统一最佳 rank，rank 应根据领域复杂度分别选择。
- `best.pt` 重新验证结果与训练记录一致，修复后未再出现 LoRA EMA 缩放为零导致的指标崩溃。

## 可比性限制

本协议不能与仓库中的历史协议直接合并。历史结果可能使用 fallback 后端、AMP、不同 batch、
不同图像尺寸或不同数据比例。跨协议数值只能作为背景参考，rank 结论应在同一协议内部比较。
