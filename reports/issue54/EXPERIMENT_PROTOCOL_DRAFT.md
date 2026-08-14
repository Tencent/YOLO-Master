# Issue #54 实验协议草案：跨 seed MoT 路由稳定性

> 协议状态：**Draft，未注册、未执行**
> 版本：0.1 / 2026-07-30
> 任何数据下载或训练必须等待用户批准。

## 1. 研究问题与证据单位

### 1.1 主要问题

在相同训练协议下，从不同随机种子独立训练得到的 MoT checkpoint：

1. 是否在相同图像、相同层、相同空间位置选择相同语义专家；
2. 是否产生相近的完整路由概率分布和专家利用率；
3. 场景级路由结论是否能跨 seed 复现；
4. 路由不稳定是否与检测性能、loss、目标尺度和遮挡相关。

### 1.2 实验单位

- 最高层独立重复：**训练 seed / checkpoint**；
- VisDrone 第二层 cluster：**视频 sequence**；
- 第三层：image；
- 图像内重复：MoT layer、spatial token、expert。

image、token、layer 都不能替代独立训练 seed。seed-pair 之间也共享 checkpoint，
所以所有 pairwise seed 指标只作描述；正式推断优先使用“每个 seed 相对留一共识”的
seed-level summary。

## 2. 预注册模型与训练矩阵

### 2.1 模型

| 角色 | 配置 | 用途 |
|---|---|---|
| MoE 基线 | `ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml` | 官方合规；多 seed 路由控制 |
| MoT 主模型 | `ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml` | 主要研究对象 |
| MoA 对比 | `ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml` | 官方合规 |

不纳入 MoA+MoT、scene-aware、P5-only、utility router 或 adaptive K。

### 2.2 训练矩阵

首选推荐矩阵：

- MoT：seeds `17, 42, 73, 101, 137`；
- MoE：seeds `17, 42, 73`；
- MoA：seed `42`；
- 总计 9 次独立训练。

预算不足时 MVP：

- MoT：seeds `17, 42, 73`；
- MoE：seed `42`；
- MoA：seed `42`；
- 总计 5 次独立训练。

MVP 的三 seed 只支持描述性和探索性结论；五 seed 仍然有限，但比三 seed 更适合估计
between-seed variance、leave-one-seed-out 稳健性和 bootstrap 区间。

### 2.3 MVP 扩展门槛

扩展不能根据单个有利结果临时决定，必须在完整 MVP 审计后按以下门槛评估：

1. 如果 MoT 3 seeds 的 mAP、路由利用率或场景结论存在明显差异，则把 MoT 扩展到 5 seeds；
2. 如果需要判断差异是否为 MoT 特有现象，则把 MoE 扩展到 3 seeds；
3. 3 seeds 只报告探索性分布、原始结果和效应方向，不包装成充分验证的总体统计结论；
4. 任何扩展必须在 5-run MVP 的训练、路由、失败记录和协议偏离审计全部完成后决定。

“明显差异”的定量门槛必须在正式训练前、经过 smoke 了解测量分辨率后冻结，不能看过
MVP 正式结果再定义。

## 3. Smoke 后才冻结的统一训练协议

正式运行前在 Phase 2 生成一份 machine-readable protocol snapshot。下表只是 smoke
候选和待决项，不是已锁定参数：

| 项目 | Phase 1 候选/决策规则 |
|---|---|
| 数据集 | 候选 VisDrone2019-DET train/val；真实数据 smoke 前再确认版本和 manifest |
| 初始化 | 候选 YAML from scratch；smoke 后冻结是否 `pretrained=False` |
| image size | 候选 640；需结合显存和吞吐 smoke |
| epochs | 未冻结；根据统一短 smoke、已有收敛曲线和预算确定，所有正式组一致 |
| patience | 未冻结；不得采用 seed-specific early stopping |
| optimizer | 未冻结；候选 AdamW，正式前显式记录全部超参数 |
| requested batch | 未冻结；候选值必须在三模型上统一尝试 |
| actual batch | 由运行时事实登记，不得从 requested batch 推断或覆盖 |
| effective batch | 单独登记；若使用梯度累积，三模型保持可比 |
| precision | AMP 与 FP32 均做短 smoke；依据稳定性、吞吐和路由重复性后冻结一种主协议 |
| deterministic | 候选 `True`；记录 PyTorch/CUDA/cuDNN 的实际生效状态 |
| workers | smoke 后固定并记录；正式 run 不得按 seed 改变 |
| validation | 每 epoch 同一 val split；最终统一重跑 |
| checkpoint | 路由主分析统一使用每个 run 的 `best.pt`，选择规则预先固定为 val mAP50-95 |

必须保存：

- repo commit、dirty flag；
- 完整 args/default snapshot；
- Python、PyTorch、CUDA、driver、GPU；
- 数据 YAML、train/val image manifest 与 SHA-256；
- seed、模型配置 SHA-256、checkpoint SHA-256；
- epoch 状态、异常、resume 事件和失败原因。

如果发生中断，resume 只能恢复同一 model/seed/run ID。不得用新 seed 冒充恢复，也不得
静默用成功 seed 替换失败 seed。正式协议冻结时必须同时写入 requested 和实际生效参数。

## 4. 官方合规测量

对 EsMoE、MoT、MoA 的统一 seed 42 run 独立测量：

- `metrics/mAP50-95(B)`、`metrics/mAP50(B)`；
- P50/P95/P99 latency：同一 GPU、batch=1、同一 imgsz、固定预热时长、
  至少三轮、轮换模型顺序，保留每轮原始样本；
- torch profiler 实际 FLOPs；
- trainable/total Params；
- box/cls/dfl/total loss 曲线；
- NaN、Inf、发散、OOM、resume 事件；
- best 和 last checkpoint 的身份。

已有 PR 结果放在“外部参考”列，不与本项目数据合并计算均值或显著性。

## 5. 同图跨 checkpoint 路由采集

### 5.1 固定验证清单

- 按稳定相对路径排序，生成不可变 `val_manifest`；
- 记录文件 hash、sequence ID、图像尺寸、目标数量、尺度分组和遮挡信息；
- 所有 checkpoint 使用相同图像、顺序、letterbox、imgsz、batch、device 和 dtype；
- 不启用随机增强、TTA 或 shuffle。

### 5.2 schema 对齐

每个 checkpoint 先导出：

- model config hash；
- MoT layer 的稳定 module path；
- feature map H/W；
- expert count 和 `top_k`；
- expert index 到语义名称的映射；
- router temperature。

当前三个 MoT expert 的结构身份固定，因此按
LocalConv / Window / Deformable 语义索引对齐，不使用事后 Hungarian matching。
若 module path、expert identity 或空间 shape 不一致，停止比较，不做猜测式匹配。

### 5.3 路由输出

对每个 `seed × image × layer` 保存或流式汇总：

- post-top-k route weights；
- top-k expert indices；
- router logits；
- 由 `softmax(logits / temperature)` 得到的 full dense probability；
- expert utilization counts；
- normalized route entropy；
- checkpoint、image、layer、expert schema IDs。

JSD 和 entropy 必须使用完整 dense probability，不能使用已被 top-k 置零并重新归一化的
weights。当前模型 router hook 的第三个输出已经包含 logits，计划不修改核心模型。

## 6. 指标定义

### 6.1 checkpoint 重复推理确定性

在跨 seed 比较前，对每个 checkpoint 抽取固定 manifest 子集并连续重复三次：

- top-k indices exact agreement；
- full probability 的 max absolute/relative error；
- model output tensor 的 max error；
- 最终 detection 数量、坐标、置信度差异；
- batch=1 重复为主；跨 batch/device/dtype 另列，不混入主确定性判定。

预定义判定：

- indices 期望 100% 一致；
- 同一 precision mode 下使用明确 `atol/rtol`；synthetic test 只验证工具逻辑，最终阈值
  需在 AMP/FP32 smoke 后冻结；
- 超阈值先归因并修复推理噪声，不能直接解释为训练 seed 差异。

### 6.2 跨 seed 路由稳定性

主要指标：

1. top-1 token agreement；
2. top-k Jaccard（`top_k > 1` 时）；
3. per-expert probability absolute difference；
4. full probability Jensen-Shannon divergence；
5. normalized entropy：`H(p) / log(E)`；
6. expert utilization 的 between-seed variance、SD 和 CV；
7. 每个 seed 相对其他 seeds 概率 barycenter 的 deviation；
8. per-image、per-layer、per-expert 的 stability table。

辅助指标：

- pairwise seed heatmap；
- leave-one-seed-out stability；
- rank agreement；
- 路由 margin 与 effective expert count。

pairwise 数量为 3-seed 的 3 对或 5-seed 的 10 对，但这些 pair 不是 3 或 10 个新的独立
训练实验单位。

### 6.3 场景级重现率

预注册场景：

- dense vs sparse；
- small vs large objects；
- low vs high occlusion；
- 必要时 irregular proxy，仅作辅助。

对每个 seed 独立计算场景 contrast 的原始差、方向和区间。报告：

- effect sign agreement；
- 超过预定义最小效应阈值的 seed 比例；
- leave-one-seed-out 结论是否改变；
- effect size 与 95% CI；
- BH-FDR 后的结果。

不能用“某 seed 显著、另一个不显著”来声称二者效果不同；比较应基于 effect difference
及其不确定性。

### 6.4 与检测结果的关系

分为两个层级：

1. seed-level：checkpoint 的整体 mAP/loss 与其相对跨 seed 共识的稳定性；n=3/5，
   只做 Spearman/散点和效应方向，不作强因果或显著性结论；
2. image/sequence-level：route instability 与同图预测分歧、per-image loss proxy、
   目标尺度和遮挡的关联；使用 seed/sequence 嵌套或 cluster bootstrap。

检测 mAP 是数据集级指标，不能伪造“per-image mAP”。逐图分析应使用清晰定义的
prediction agreement、matched detection error 或 per-image loss proxy。

## 7. 统计分析计划

### 7.1 主 estimand

- MoT 的 between-seed route deviation；
- MoT 与 MoE 控制的 between-seed deviation 差；
- 场景 contrast 的跨 seed 重现率。

### 7.2 分层重采样

推荐采用：

1. 以 seed/checkpoint 为顶层 resampling unit；
2. VisDrone 内以 sequence 为 cluster；
3. sequence 内再抽 image；
4. 对每个重采样样本重新计算聚合指标。

为避免 seed-pair 依赖，主要推断使用每个 seed 相对 leave-one-out barycenter 的 deviation。
pairwise seed 结果只作为可视化和描述性补充。

3-seed MVP 的顶层 bootstrap 支持很弱，必须标注 exploratory。5-seed 方案报告 bootstrap
区间、leave-one-seed-out 范围和原始每-seed 点，不只给均值。

### 7.3 效应量与多重比较

- 连续 contrast：原始差 + 标准化 effect size + CI；
- agreement：比例及 cluster-aware CI；
- 方差：between-seed SD/variance，必要时报告 CV；
- layer × scene × expert 的多重检验使用 Benjamini-Hochberg FDR；
- 主指标和辅助指标分开，禁止事后挑显著项作为主结果。

## 8. 有效性与审计规则

### 8.1 有效 run

有效 run 必须：

- 完成冻结 epoch 或符合预注册的统一终止规则；
- 无未解释 NaN/Inf；
- config、data、checkpoint hash 齐全；
- 最终验证和路由采集成功；
- 无 seed-specific 超参数修改。

失败 run 保留在审计表中，标记 invalid/failed 及原因。是否重跑必须使用同一 seed 和同一
协议；不能换 seed 来改善结果。

### 8.2 结果边界

- 不把相关性写成因果；
- 不把单数据集结果外推到所有检测场景；
- 不把 argmax agreement 单独解释为概率稳定；
- 不把近均匀概率下的微小 argmax 波动夸大为专家语义变化；
- 不隐藏负结果、无效 run、OOM 或协议偏离；
- 不使用同 seed 内大量图像人为扩大训练重复数。

## 9. 输出物

未来代码输出应包括：

- `experiment_manifest.json`；
- `checkpoint_inventory.csv`；
- `route_schema.json`；
- `routing_records` 的可分片、可校验格式；
- `determinism_report.json`；
- `seed_stability_summary.csv`；
- `scene_reproducibility.csv`；
- `model_comparison.csv`；
- effect-size/CI 表和有限数量的审计图。

大体积逐图结果、checkpoint、原始日志和结论图只进入实验归档/报告，不进入代码 PR。

## 10. 执行前必须确认

进入下一阶段前需要用户确认：

1. 采用 MVP 5 runs 还是推荐 9 runs；
2. 是否批准 VisDrone 作为主数据集；
3. AMP 与 FP32 的 smoke 矩阵、候选 epochs 和 requested batch 范围；
4. smoke 结果出来后冻结 epochs、actual/effective batch、precision 和 optimizer；
5. 开跑当日重新审计 #137/#189/#190 并据此调整复用策略。

在上述确认前，本协议保持 Draft，不下载数据、不启动训练。
