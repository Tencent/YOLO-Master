# Issue #54 跨 Seed 路由指标定义

> 版本：Phase 1 / Schema v1
>
> 适用范围：多个独立训练 MoT checkpoint 的描述性稳定性分析

## 1. 身份与对齐

协议组由 `model_variant`、`dataset`、`dataset_version` 和 `split` 定义。组内记录只在以下字段完全一致时对齐：

- `image_id`；
- `layer_name`；
- `layer_index`；
- 专家名称集合。

专家数组的顺序可以不同，但名称集合必须一致。分析器先按专家名称排序，再同时重排概率和 token expert index。
缺失 image 或 layer 只在交集上计算指标，同时写入 `validation_issues`；不会补零或虚构匹配。
per-image、per-layer、global、专家利用率和场景重现率汇总都保留完整协议组字段，不允许跨
`dataset`、`dataset_version` 或 `split` 合并同名 image/layer。

## 2. 概率与离散路由

设某 image、layer 和 seed 的完整专家概率为 `p = (p_1, ..., p_E)`。`p_e` 必须有限、非负且总和为 1。

### 2.1 Dominant expert

`selected_expert = argmax_e p_e`。两个 seed 的 dominant route agreement 为 0 或 1。

### 2.2 Token top-1 agreement

对同一空间位置的完整概率取 top-1 expert。两个 seed 的 token agreement 是相同位置选择一致的比例。
空间 shape 不一致时结果为 `null`，不会截断较长数组。

### 2.3 Jensen--Shannon divergence

对专家语义已经对齐的两个概率向量 `p`、`q`：

`JSD(p, q) = 0.5 * KL_2(p || m) + 0.5 * KL_2(q || m)`，其中 `m = (p + q) / 2`。

使用以 2 为底的对数，因此理论范围为 `[0, 1]`。0 表示分布相同；值越高表示分布差异越大。
实现保留每个 seed-pair、image 和 layer 的原始 JSD。

### 2.4 Route entropy

`H(p) = -sum_e p_e * ln(p_e)`，单位为 nat。

归一化 entropy 为 `H(p) / ln(E)`，范围 `[0, 1]`。它描述单个 checkpoint 的路由分散程度，
不能单独证明跨 seed 稳定。

## 3. 专家利用率

对某 seed、layer 和 expert，利用率为：

`该 expert 被 token top-1 选择的次数 / 该层全部导出 token 数`。

between-seed variance 使用各独立 seed 利用率的样本方差，分母为 `n - 1`。只有一个 seed 时，
variance 和 standard deviation 为 `null`；两个 seed 时可以描述，但仍不足以形成稳定总体推断。

## 4. 场景结论重现率

对每个 `scene_dimension`、两个场景 level、layer 和 expert，先在每个 seed 内计算：

`effect_b_minus_a = mean_probability(level_b) - mean_probability(level_a)`。

随后保存每个 seed 的原始 effect，并报告多数效应方向的比例。该比例是方向重现率，不是显著性概率。
同一 seed 内的图像数量只能改善该 checkpoint 的描述精度，不能增加跨 seed 样本量。

## 5. 重复推理确定性

同一 `experiment_id`、checkpoint、image 和 layer 的 repeat 0 是基准。后续 repeat 必须满足：

- token top-1 agreement 等于 1；
- 专家概率最大绝对差不超过预注册容差；
- 专家语义和空间 shape 一致。

重复推理只检查 checkpoint 的确定性，不是新的训练重复，也不会增加 seed 数。
如果 repeat 0 缺失，结果标记为 `invalid_missing_base_repeat`，不得用其他 repeat 自动替代基准。

## 6. 汇总层级

分析器生成：

- `pairwise_seed_comparisons`：不可省略的原始 seed-pair、image、layer 行；
- `per_image_summary`：跨 layer 和 seed-pair 的描述汇总；
- `per_layer_summary`：跨 image 和 seed-pair 的描述汇总；
- `global_summary`：便于审计的总体描述；
- `expert_utilization_by_seed`：seed 内利用率；
- `expert_utilization_between_seed`：seed 级离散程度；
- `scene_conclusion_reproducibility`：场景效应原始值和方向重现率。

汇总均为描述性结果。seed-pair 共享 checkpoint，也不相互独立，因此不得把 pairwise 行数当成有效样本量。
空 routing 输入不是零样本分析，而是契约错误；CLI 必须返回非零且不得生成成功 analysis。

## 7. 证据强度

| 独立正式 seed 数 | 允许的表述 |
|---:|---|
| 0--2 | 不足以进行跨 seed 推断；只报告诊断或个案 |
| 3--4 | 探索性分布、原始数据和效应方向 |
| 至少 5 | 更强跨 seed 结论的最低门槛，仍需报告全部原始 seed 结果 |

本项目不把 548 张图片当成 548 次训练重复，不在 3 seeds 上生成看似精确的显著性结论。

## 8. 与单 Checkpoint 指标的关系

JSD 和 entropy 是通用数学量。PR #190 已在未合并代码中用于单 checkpoint perturbation 解释。
本项目仅在跨独立 checkpoint 的对齐框架中使用薄层实现。PR #190 若合并，将评估复用其底层原语，
但不会移除本项目的 registry、seed 身份、语义对齐、pairwise 保存和有效性管理。
