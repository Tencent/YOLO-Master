# YOLO-Master Mixture-of-Transformers 集成方案与实验验证补充报告

> 日期：2026-06-25  
> 范围：基于现有 MoT 评估文档，补充工程接入、网络融合、参数配置、训练策略、可复现实验入口和本机实测结果。  
> 当前结论：MoT 已完成 YOLO-Master 解析、训练 loss 接入、单元测试、参数量统计、CPU latency smoke 和 COCO8 标准小数据集训练 smoke。参数增量可控但高于 MoA；CPU 推理成本明显增加；最终检测精度收益仍需 COCO128 50e 或 COCO 300e 长训确认。

---

## 1. 集成目标

MoT 的目标不是替代已有 MoE/MoA，而是在 Neck 的多尺度融合阶段引入架构级多样性：

- MoE：对 token/特征分配不同 FFN/卷积专家，解决“谁来处理”；
- MoA：在注意力头组之间软路由，解决“用什么感受野看”；
- MoT：在完整 Transformer 架构之间路由，解决“用什么结构范式理解”。

因此 MoT 适合作为 YOLO-Master 的 Stage 2 增强：在保持 backbone MoE 动态路由的基础上，将 Neck 中 P4/P5 的部分 `C3k2` 替换为 `C2fMoT`，利用 LocalConv / Window / Deformable 三类专家补足局部纹理、规则结构和不规则遮挡目标建模能力。

---

## 2. 模块设计

### 2.1 MoTBlock

`MoTBlock` 输入输出均为 `[B, C, H, W]`，核心数据流如下：

```text
Input
  -> Router: Conv1x1 + GN + SiLU + Conv1x1 -> [B, 3, H, W]
  -> Top-K routing weights
  -> LocalConvTransformerExpert
  -> WindowTransformerExpert
  -> DeformableTransformerExpert
  -> weighted sum
  -> GroupNorm + Conv1x1
  -> residual add
Output, z_loss
```

三个专家分工：

| Expert | 结构 | 复杂度 | 主要收益 |
|---|---|---:|---|
| LocalConvTransformer | DW 3x3 预混合 QKV + DW 7x7 V 位置编码 + gated FFN | `O(N^2)` | 小目标边缘、纹理、局部连续性 |
| WindowTransformer | Swin-style window attention，forward 间隔切换 shifted window | `O(N * win^2)` | 中等目标、规则结构、大图高效 |
| DeformableTransformer | 每 query/head 预测 K 个采样点并用 `grid_sample` 聚合 | `O(N * K)` | 遮挡、细长目标、不规则形状 |

### 2.2 Router 与 z-loss

Router 使用空间级 softmax，默认 `top_k=2`。本次补强包含两个训练稳定性修正：

- `router_z_loss` 改为沿专家维度 `logsumexp(logits, dim=1)`，即约束每个 token 的专家 logits，而不是沿空间维度约束每个专家的响应。
- 训练期增加很小的 dense exploration floor（默认 `0.02`），避免零初始化 + Top-K 在第一步固定只选择 expert 0/1，导致 expert 2 永久拿不到梯度；推理期仍使用严格 Top-K 稀疏路由。

MoT aux loss 由每个 `C2fMoT.last_aux_loss` 暴露，并通过 `collect_mot_aux_loss(model)` 汇总。

### 2.3 C2fMoT

`C2fMoT` 是 YAML 中实际使用的 C2f-style 封装：

```python
C2fMoT(c1, c2, n=1, num_heads=6, top_k=2, window_size=7,
       n_points=4, mlp_ratio=2.0, temperature=1.0,
       balance_loss_coeff=0.01, e=0.5)
```

结构：

```text
cv1(c1 -> 2*c)
  -> split(identity, dynamic)
  -> dynamic branch passes through n MoTBlock
  -> concat(identity + all dynamic outputs)
cv2((2+n)*c -> c2)
```

YAML 参数顺序：

```yaml
# [c2, n, num_heads, top_k, window_size, n_points, mlp_ratio, temperature, balance_loss_coeff, e]
- [-1, 2, C2fMoT, [512, 2, 8, 2, 7, 4, 2.0, 1.0, 0.01, 0.5]]
```

---

## 3. 与原网络结构的融合方式

### 3.1 MoT-only：替换 Neck 高层 C3k2

落地配置：

```text
ultralytics/cfg/models/master/v0_8/det/yolo-master-mot-n.yaml
```

融合方式：

```text
P5 -> Upsample -> Concat(P4) -> C2fMoT
P4 -> Upsample -> Concat(P3) -> C3k2
P3 -> Downsample -> Concat(P4) -> C2fMoT
P4 -> Downsample -> Concat(P5) -> C2fMoT
Detect(P3, P4, P5)
```

理由：

- P4/P5 分辨率较低，Transformer 代价可控；
- P3 保留 `C3k2`，避免最高分辨率上引入过重注意力；
- Neck 融合阶段更接近检测头，便于多尺度上下文直接影响分类/回归；
- 不侵入 backbone MoE，降低路由分布突变风险。

### 3.2 MoA+MoT：感受野多样性 + 架构多样性

落地配置：

```text
ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-mot-n.yaml
```

融合方式：

- Backbone P3 使用轻量 `C2fMoA`；
- Neck P3 使用 `C2fMoA` 保留 local/regional/global 感受野混合；
- Neck P4/P5 使用 `C2fMoT` 做架构级专家混合；
- Detect 头保持原 YOLO-Master 三尺度输出。

该变体面向高资源场景，不建议作为默认移动端配置。

---

## 4. 训练策略与参数配置

| 项 | 推荐值 | 说明 |
|---|---:|---|
| `top_k` | 2 | 默认兼顾表达力和稀疏性；极限延迟场景可设 1 |
| `window_size` | 7 | P4/P5 默认 7；若放到 P3 可改 4 |
| `n_points` | 4 | Deformable expert 每 head 采样点数 |
| `balance_loss_coeff` | 0.01 | MoT router z-loss 的模块内权重 |
| `hyp.moe` | 0.3 | 复用现有 mixture aux 全局权重，MoE + MoT 共用 |
| `lr0` | 0.006-0.008 | MoT/联合模型建议略低于 baseline |
| `warmup_epochs` | 5-7 | 给 Router 与 Deformable offsets 更长启动期 |
| `temperature anneal` | `0.97/epoch -> 0.3` | 训练后期让路由更确定 |
| `gradient_clip` | `max_norm=10` | 长训建议开启，防 Deformable offset 尖峰 |

训练 callback 已在 `scripts/compare_mot_ablation.py` 中接入：

```python
anneal_moa_temperature(trainer.model, factor=0.97, min_temp=0.3)
anneal_mot_temperature(trainer.model, factor=0.97, min_temp=0.3)
```

loss 接入已在 `ultralytics/utils/loss.py` 中完成：

```text
_collect_mixture_aux_loss = _collect_moe_aux_loss + _collect_mot_aux_loss
```

检测、分割、姿态、OBB loss 的原 `moe_loss` 槽位现在承载 mixture aux loss。这样保持现有日志字段兼容，不新增配置项。

---

## 5. 已完成代码落地

| 文件 | 变更 |
|---|---|
| `ultralytics/nn/tasks.py` | 导入 `C2fMoT` 并加入 `base_modules`，YAML 可解析 |
| `ultralytics/utils/loss.py` | 新增 `_collect_mot_aux_loss` 和 `_collect_mixture_aux_loss` |
| `ultralytics/nn/modules/mot/mot.py` | 修正 z-loss 轴向；增加训练期 exploration floor |
| `tests/test_mot.py` | 覆盖 forward/backward、专家梯度、z-loss、退火、YAML 解析 |
| `scripts/compare_mot_ablation.py` | 支持 build、latency、train、summary 四类实验 |

---

## 6. 本机实验验证

环境：

```text
macOS / Apple M1 Pro
Python 3.11.2
PyTorch 2.9.1
CUDA: unavailable
MPS: available
Ultralytics: 8.3.240
```

### 6.1 自动化测试

命令：

```bash
python3 -m pytest tests/test_moa.py tests/test_mot.py -q
```

结果：

```text
9 passed in 5.77s
```

覆盖范围：

- `MoTBlock` forward/backward；
- `C2fMoT` shape 与 aux loss 汇总；
- 三个 Transformer experts 均能收到梯度；
- z-loss 按专家维度计算；
- `anneal_mot_temperature` 生效；
- `yolo-master-mot-n.yaml` 与 `yolo-master-moa-mot-n.yaml` 能被 `DetectionModel` 解析。

### 6.2 Build 与参数量

命令：

```bash
python3 scripts/compare_mot_ablation.py \
  --check-build \
  --models v08 v08_moa v08_mot v08_moa_mot \
  --device cpu \
  --project runs/mot_ablation
```

输出文件：

```text
runs/mot_ablation/build_summary.csv
```

| 模型 | 参数量 | MoABlock | MoTBlock | 参数增量 |
|---|---:|---:|---:|---:|
| v0.8 baseline | 3.137637M | 0 | 0 | - |
| v0.8 MoA | 3.233538M | 6 | 0 | +3.06% |
| v0.8 MoT | 3.712284M | 0 | 6 | +18.31% |
| v0.8 MoA+MoT | 3.760829M | 2 | 6 | +19.86% |

结论：MoT 参数增量显著高于 MoA，但 N 号模型仍保持在 4M 参数以内，适合服务器端或精度优先场景；MoA+MoT 的额外参数相对 MoT-only 仅增加约 48.5K。

### 6.3 CPU 推理速度 smoke

命令：

```bash
python3 scripts/compare_mot_ablation.py \
  --benchmark \
  --models v08 v08_moa v08_mot v08_moa_mot \
  --device cpu \
  --imgsz 256 \
  --warmup 3 \
  --reps 10 \
  --project runs/mot_ablation
```

输出文件：

```text
runs/mot_ablation/latency_cpu_256.csv
```

| 模型 | mean latency | min | max | 相对 baseline |
|---|---:|---:|---:|---:|
| v0.8 baseline | 45.358 ms | 40.776 | 50.104 | - |
| v0.8 MoA | 47.200 ms | 37.619 | 54.418 | +4.06% |
| v0.8 MoT | 79.409 ms | 54.511 | 115.481 | +75.08% |
| v0.8 MoA+MoT | 112.594 ms | 89.203 | 228.554 | +148.24% |

说明：

- 该结果是 CPU 256 smoke，不代表 CUDA 部署性能；
- MoT 的 Deformable + Window + Local 三专家在 CPU 上开销明显；
- 正式速度结论应在 CUDA GPU 上使用 `imgsz=640, batch=1, warmup>=20, reps>=100` 重测；
- 若部署目标是 CPU，建议使用 `top_k=1`、减少 `C2fMoT` 数量，或只保留 P5 位置的 MoT。

### 6.4 COCO8 标准小数据集训练 smoke

命令：

```bash
python3 scripts/compare_mot_ablation.py \
  --train \
  --models v08 v08_moa v08_mot v08_moa_mot \
  --data ultralytics/cfg/datasets/coco8.yaml \
  --project runs/mot_smoke_coco8 \
  --epochs 1 \
  --imgsz 128 \
  --batch 2 \
  --device cpu \
  --workers 0 \
  --no-amp \
  --exist-ok \
  --patience 0
```

输出文件：

```text
runs/mot_smoke_coco8/summary.csv
```

| 模型 | train box | train cls | train dfl | val box | val cls | val dfl | mAP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.8 baseline | 6.07155 | 5.80654 | 4.27910 | 4.38737 | 5.92199 | 4.15894 | 0 |
| v0.8 MoA | 5.95791 | 5.84218 | 4.24882 | 4.38640 | 5.92056 | 4.16098 | 0 |
| v0.8 MoT | 6.03362 | 5.89880 | 4.21344 | 4.38644 | 5.92084 | 4.15954 | 0 |
| v0.8 MoA+MoT | 6.05760 | 5.69853 | 4.16258 | 4.38742 | 5.92080 | 4.15830 | 0 |

观察：

- 四个模型均完成训练、验证、best/last 权重保存和 CSV 汇总；
- MoT / MoA+MoT 的训练日志中 `moe_loss` 槽位约为 `1.76`，高于 baseline 的 `0.69`，说明 MoT z-loss 已被合并到 mixture aux loss；
- 从随机初始化在 COCO8 上训练 1 epoch，mAP=0 是预期现象，不能用于证明最终精度提升；
- 该实验的意义是验证工程闭环，而不是作为论文级精度结论。

---

## 7. 正式精度验证方案

### 7.1 COCO128 快速消融

建议先跑 50 epoch，至少 3 个 seed：

```bash
python3 scripts/compare_mot_ablation.py \
  --train \
  --models v08 v08_moa v08_mot v08_moa_mot \
  --data ultralytics/cfg/datasets/coco128.yaml \
  --project runs/mot_ablation_coco128_50e \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --exist-ok
```

验收表：

| 实验 | 配置 | 目的 | 通过标准 |
|---|---|---|---|
| A0 | v0.8 baseline | 当前基线 | 记录 mAP50/mAP50-95、速度、参数 |
| A1 | v0.8 MoA | 感受野多样性 | mAP50 不低于 A0，速度增量可控 |
| A2 | v0.8 MoT | 架构多样性 | mAP50 >= A0 + 0.5，或遮挡/密集子集有明显收益 |
| A3 | v0.8 MoA+MoT | 联合收益 | mAP50 >= max(A1,A2)，且延迟符合资源目标 |
| A4 | MoT top_k=1 | 高效路由 | 延迟下降，mAP 不明显退化 |
| A5 | MoT balance=0 | z-loss 消融 | 观察路由坍缩与 mAP 变化 |

### 7.2 完整 COCO 验证

只有 COCO128 显示正收益后再启动完整 COCO：

```bash
python3 scripts/compare_mot_ablation.py \
  --train \
  --models v08 v08_mot v08_moa_mot \
  --data ultralytics/cfg/datasets/coco.yaml \
  --project runs/mot_ablation_coco_300e_seed42 \
  --epochs 300 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --seed 42 \
  --exist-ok
```

正式报告应记录：

| 模型 | params | GFLOPs | mAP50 | mAP50-95 | AP_s | AP_m | AP_l | GPU latency 640 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.8 baseline | 已测 | 待测 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待测 |
| v0.8 MoT | 已测 | 待测 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待测 |
| v0.8 MoA+MoT | 已测 | 待测 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待测 |

---

## 8. 当前风险与建议

| 风险 | 当前观察 | 建议 |
|---|---|---|
| CPU 延迟高 | 256 CPU 下 MoT +75%，MoA+MoT +148% | CPU 部署用 top_k=1 或只放 P5 |
| 精度尚未证明 | 仅完成 COCO8 1e smoke | 必须跑 COCO128 50e / COCO 300e |
| Top-K 初始专家死亡 | 零初始化会固定选 expert 0/1 | 已加训练期 exploration floor |
| z-loss 量级错误 | 原实现沿空间维度计算 | 已改为沿专家维度计算 |
| `grid_sample` 导出 | Deformable expert 可能影响 ONNX/TensorRT | 部署前单独做 export/benchmark |
| mixture aux 命名 | 日志仍叫 `moe_loss` | 为兼容保留字段，文档中解释为 mixture aux |

---

## 9. 结论

MoT 已从概念报告推进到可构建、可解析、可反传、可训练 smoke 的工程状态。相对 v0.8 baseline，MoT-only 参数从 3.14M 增至 3.71M（+18.31%），MoA+MoT 增至 3.76M（+19.86%）；在 N 号模型上仍属于可实验的轻量区间。

实用性方面，MoT 在 CPU 上推理成本明显高于 MoA，适合优先在 CUDA GPU/服务器端验证；若目标是边缘部署，应先做 `top_k=1`、减少 MoT 层数、仅 P5 替换等消融。

有效性方面，当前 COCO8 1 epoch 结果只能证明训练链路，不足以证明 mAP 增益。要支撑“提升检测精度”的结论，需要按本文第 7 节完成 COCO128 50 epoch 和完整 COCO 300 epoch，并报告多 seed 均值、方差、GPU latency 与参数/FLOPs。
