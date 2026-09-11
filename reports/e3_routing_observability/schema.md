# `e3.routing_snapshot.v1` 字段字典

## 顶层

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | string | 固定为 `e3.routing_snapshot.v1` |
| `scope` | string | 本次采集用途 |
| `dataset` | string | 数据配置标识，不记录私有绝对路径 |
| `sample` | object | 数据集内相对路径与 SHA-256 |
| `imgsz` | integer | 正方形推理输入尺寸 |
| `device` | string | 实际执行设备 |
| `weights` | string | 权重来源；本 smoke 为 YAML 随机初始化 |
| `families` | object | `moe`、`mot`、`latent` 三族结果 |
| `limitations` | string[] | 禁止外推的边界 |

## 路由族与层

每个 `families.<family>` 包含模型配置及 checksum、捕获层数、层记录、静态图路径和开销记录。

| 层字段 | 类型 | 说明 |
|---|---|---|
| `layer_name` | string | `named_modules()` 中的稳定层名 |
| `module_type` | string | Python 模块类型 |
| `family` | string | 统一路由族标识 |
| `num_experts` | integer | 专家总数 |
| `top_k` | integer | 配置或 snapshot 中的 Top-K |
| `routing_axis` | string | `spatial`、`image`、`expert` 或生产者声明的轴 |
| `probability_shape` | integer[] | 统一为 expert 轴在维度 1 的概率张量形状 |
| `expert_usage` | number[] | 生产者 snapshot 的平均专家负载 |
| `mean_router_probs` | number[] | 本次 hook 捕获并归一化的平均混合权重 |
| `entropy` | number | 平均混合权重的自然对数熵 |
| `normalized_entropy` | number | 以 `log(num_experts)` 归一化并限制在 `[0,1]` |
| `dispatch_policy` | string | dense/sparse/unknown 等生产者状态 |
| `aux_loss` | object | 明确的配置、观测与生命周期状态 |

训练 telemetry 使用相同的层字段，并在顶层增加 `issues`、`invalid_layers`、`unsupported_layers`、
`mean_normalized_entropy`、`configured_aux_layers` 和 `active_aux_layers`。非法证据不会被补零；训练不中断，
但会在 `issues` 中记录层名、模块类型、状态和原因。准入 smoke 对非法层保持失败退出。

## `aux_loss.status`

| 值 | 含义 |
|---|---|
| `not_configured` | 当前模块未暴露非零 balance/z-loss 配置，即 `N/A` |
| `configured_inactive_eval` | 已配置，但该损失仅训练态生效；推理 smoke 中观测零值是预期行为 |
| `active_training` | 训练态观测到非零 aux |
| `configured_zero_observed` | 训练态已配置但本次观测为零，需要继续排查或证明 |
| `available` | 非训练专用实现提供了可解释观测值 |

`status_code` 仅用于 TensorBoard 标量：`not_configured=0`、`configured_inactive_eval=1`、
`configured_zero_observed=2`、`active_training=3`、`available=4`。JSON 中的字符串 `status` 仍是审计依据，
不得只看数值编码判断损失是否进入训练总损失。

不得仅凭配置键存在断言 aux 已进入训练总损失；后续训练验证必须同时提供非零日志或单测证据。

## TensorBoard 键

- 全局：`routing/global/{routed_layers,invalid_layers,unsupported_layers,mean_normalized_entropy}`。
- 逐层：`routing/<family>/<layer>/normalized_entropy`。
- 专家：`routing/<family>/<layer>/expert_<n>_{usage,mean_probability}`。
- 辅助损失：`routing/<family>/<layer>/{aux_observed,aux_status_code}`。

层名只保留字母、数字、点、下划线和连字符，确保不同运行使用稳定路径。

训练目录中的 `telemetry.json` 是跨 rank 汇总，`telemetry_rank_<rank>.json` 保留本地记录。TensorBoard 只保存
数值标量；字符串状态、异常原因、采样间隔和最后一次完整 snapshot 以 JSON 为准。
