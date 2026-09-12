# A3 P1 — 路由决策漂移分析（EsMoE-N 首轮）

> 日期：2026-08-29
> 状态：首轮数字（548 张 VisDrone val）
> 脚本：`scripts/a3_router_drift_analysis.py`
> 模型：`yolo_master_n.pt`（EsMoE-N，COCO 80 类，4 个 ES_MOE 块）
> 设备：本机 CPU（i5-1240P），torch 2.10.0

---

## 一、实验目的

回答 A3 的核心问题：**MoE 模型的路由层在低精度（FP16 / INT8）下，专家决策会漂移多少？**

本实验是第一刀，对象是 EsMoE-N（唯一有公开权重的 MoE 模型）。

## 二、关键前提（决定度量方式）

EsMoE-N 的 4 个 ES_MOE 块**全部是 dense softmax 路由**（`use_top_k=False`，各 3 专家），
不是 top-k 稀疏路由。因此本实验的"专家决策"定义为 **softmax 权重的 argmax 赢家专家**，
而非离散 top-k 选择集合。真正的 top-k 选择一致率要留给 MoT（见 §五）。

## 三、方法

- **隔离变量**：全程 fp32 计算，唯一变量是「路由层权重精度」。
- **量化模拟**（纯 PyTorch，无需 calibration）：
  - FP16 = `w.half().float()`（先降半精度再回 fp32）
  - INT8 = per-tensor 对称量化（`round(w/scale)·scale`，scale = max|w|/127）
- 只量化路由层（`DynamicRoutingLayer`，合计约 13K 参数），专家/骨干保持 FP32。
- hook 4 个 ES_MOE routing 层输出（softmax 权重 `[B,E,H,W]`），三档各 forward 一遍 548 张图。
- 度量：argmax 专家一致率 + 权重平均绝对误差（MAE）。

## 四、结果

| ES_MOE 块 | FP16 argmax 一致率 | FP16 MAE | INT8 argmax 一致率 | INT8 MAE |
|-----------|:---:|:---:|:---:|:---:|
| model.3 (256ch) | 100.00% | 0.00e+00 | 100.00% | 5.16e-04 |
| model.6 (512ch) | 100.00% | 0.00e+00 | 100.00% | 1.36e-04 |
| model.9 (512ch) | 100.00% | 0.00e+00 | 100.00% | 2.15e-05 |
| model.12 (1024ch) | 100.00% | 0.00e+00 | 100.00% | 8.73e-05 |

**两个发现：**

1. **FP16 完全无损（MAE=0）**——根因：EsMoE-N 权重在 HuggingFace 上本就是以半精度
   保存的（加载后 cast 到 fp32），FP16 量化是"无损还原"。结论：**EsMoE-N 走 FP16 部署
   无路由精度代价**。

2. **INT8 引入 ~1e-4 量级权重扰动，但 548 张图 argmax 专家零翻转**。per-tensor 对称量化
   是误差上界（真实部署 per-channel + calibration 误差更小），此结果初步表明 EsMoE-N 的
   dense softmax 路由对 INT8 鲁棒。

### 决策 margin 归因（零翻转是鲁棒还是运气？）

量化是否翻转 argmax，取决于 top-1 与 top-2 的权重差距（margin）相对量化扰动的量级。
FP32 下逐层统计（548 张图 × 全部空间 token）：

| ES_MOE 块 | margin 均值 | margin 最小 | <1e-3 占比 | <1e-4 占比 |
|-----------|:---:|:---:|:---:|:---:|
| model.3 (256ch) | 0.5434 | 0.5141 | 0% | 0% |
| model.6 (512ch) | 0.0777 | 0.0717 | 0% | 0% |
| model.9 (512ch) | 0.0186 | 0.0186 | 0% | 0% |
| model.12 (1024ch) | 0.0238 | 0.0238 | 0% | 0% |

**结论：零翻转是结构性鲁棒。** 最小 margin（1.86e-2）比 INT8 扰动（2e-5 ~ 5e-4）大 2~3 个
数量级，没有任何 token 的 margin 低于 1e-3。

**附带发现**：路由"自信度"随深度递减——浅层 model.3 近乎单一专家通吃（margin≈0.54），
深层 model.9/12 趋于三专家等权（margin≈0.02）。这暗示深层路由对更激进的量化（如 2/4-bit）
会最先失稳，是后续"量化位宽 × 漂移"扫描的重点关注区。

### 量化位宽扫描（16/8/4/2-bit，548 张图）—— 验证「深层最先失稳」

对路由层做对称量化，argmax 专家一致率（FP32 基线 = 100%）：

| ES_MOE 块 | fp16 | int8 | int4 | **int2** |
|-----------|:---:|:---:|:---:|:---:|
| model.3 (浅层) | 100% | 100% | 100% | **100%** |
| model.6 | 100% | 100% | 100% | **100%** |
| model.9 (深层) | 100% | 100% | 100% | **0.00%** |
| model.12 (深层) | 100% | 100% | 100% | **0.00%** |

权重 MAE（对照 margin 判断失稳阈值）：

| ES_MOE 块 | int8 | int4 | int2 | FP32 margin 最小值 |
|-----------|:---:|:---:|:---:|:---:|
| model.3 | 5.16e-4 | 2.44e-3 | 6.15e-2 | 0.514 |
| model.6 | 1.36e-4 | 7.81e-4 | 1.83e-2 | 0.072 |
| model.9 | 2.15e-5 | 8.78e-4 | 1.20e-2 | 0.0186 |
| model.12 | 8.73e-5 | 1.60e-3 | 1.20e-2 | 0.0238 |

**结论（预判成立）**：
1. **INT8 / INT4 全层鲁棒**——一致率 100%，量化扰动远小于 margin。
2. **INT2 出现清晰层分化**：深层 model.9/12 的 argmax 一致率暴跌至 0%（完全失稳），
   浅层 model.3/6 仍 100%。归因——深层 margin≈0.02 而 INT2 MAE≈1.2e-2 已逼近 margin，
   浅层 margin≈0.54 远大于 INT2 MAE≈6e-2，故仍鲁棒。
3. **工程含义**：路由层可安全量到 INT4；若要激进到 2-bit，必须对深层路由做精度保护
   （保留 fp16 或混合位宽）。这为 `quantize.py` 现有的"路由层保 fp16"策略提供了定量依据，
   并进一步指出：**同一模型内不同深度的路由层对量化的敏感度不同**，可做分层位宽分配。

## 五、局限与下一步

1. **EsMoE 是 dense softmax，非稀疏 top-k**——argmax（top-1）天然比 top-k 集合稳定。
   A3 的真正对象是 MoT 的 top-k 稀疏路由，其离散决策边界对量化更敏感。
2. 位宽扫描用的是 per-tensor 对称量化（误差上界），真实部署 per-channel + calibration
   误差更小，故本报告的失稳阈值是保守上界。
3. 每 token 一致率已记录于 `per_image_agree` 字段，逐图均 100%，样本内无翻转。

**下一步顺序：**
- ① ~~margin 分析~~ ✅ 已完成（见上节，零翻转是结构性鲁棒）
- ② ~~量化位宽扫描~~ ✅ 已完成（INT4 全层鲁棒；INT2 深层 model.9/12 失稳，预判成立）
- ③ MoT top-k 漂移分析（随机初始化 `yolo26-master-mot-n.yaml` 可 forward，方法学验证；
   真实数字待 MoT 权重训出后重跑）
- ④ 真实 INT8 PTQ（ONNX Runtime，云端 GPU）

## 六、复现命令

```bash
# 三档漂移分析
/d/Anaconda3/envs/yolo_env/python.exe scripts/a3_router_drift_analysis.py \
  --source data/visdrone/VisDrone2019-DET-val/images --limit 548 \
  --out runs/a3_router_drift_visdrone548

# margin 归因
/d/Anaconda3/envs/yolo_env/python.exe scripts/a3_router_margin_analysis.py \
  --source data/visdrone/VisDrone2019-DET-val/images --limit 548

# 位宽扫描（16/8/4/2-bit）
/d/Anaconda3/envs/yolo_env/python.exe scripts/a3_router_bitwidth_sweep.py \
  --source data/visdrone/VisDrone2019-DET-val/images --limit 548 \
  --out runs/a3_router_bitwidth_sweep
```
