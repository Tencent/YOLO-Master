# Issue #54 实验登记与路由证据规范

> 规范版本：1
>
> 机器契约：`scripts/issue54/schema.py`

## 1. 设计原则

1. 所有训练和导出参数来自显式 manifest，不从实验目录名称猜测；
2. 私人绝对路径不进入证据文件，只保存脱敏相对路径和稳定 ID；
3. checkpoint、config、数据清单和图像使用 SHA-256 固定身份；
4. requested、actual 和 effective batch 分开保存；
5. `passed`、`failed`、`diagnostic` 和 `not_executed` 的证据意义不同；
6. 只有独立 checkpoint 对应的 `status=passed` 记录可计入正式 seed 数；
7. JSON 和 JSONL 使用确定性排序、UTF-8 与 checksum，便于重复生成和审计；
8. 已有输出默认不可覆盖，只有显式 `--overwrite` 才允许替换。

## 2. Experiment manifest

每次计划或执行的实验对应一个 JSON object。

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | integer | 当前固定为 1 |
| `experiment_id` | string | 便携且唯一的实验 ID |
| `model_variant` | string | `mot`、`moe`、`moa` 或明确诊断 variant |
| `seed` | integer | 独立训练随机种子 |
| `dataset` | string | 数据集名称 |
| `dataset_version` | string | 版本或快照标识 |
| `dataset_manifest_sha256` | SHA-256/null | 脱敏图像清单的固定身份 |
| `split` | string | 用于导出的 split |
| `requested_epochs` | integer/null | 启动前请求的 epoch |
| `epochs` | integer/null | 实际完成的 epoch |
| `requested_batch` | integer/`auto`/null | 用户或协议请求的 batch |
| `batch` | integer/null | 运行时实际 per-step batch |
| `effective_batch` | integer/null | 梯度累积和并行后的有效 batch |
| `imgsz` | integer/null | 实际输入尺寸 |
| `optimizer` | string/null | 实际 optimizer |
| `precision_mode` | string/null | 例如 `amp`、`fp32`，smoke 后冻结 |
| `checkpoint_path` | relative path/null | 相对 artifact root 的脱敏路径 |
| `checkpoint_sha256` | SHA-256/null | checkpoint 内容身份 |
| `config_path` | relative path/null | 相对 artifact root 的配置路径 |
| `config_sha256` | SHA-256/null | 配置内容身份 |
| `git_commit` | Git SHA | 产生实验的代码版本 |
| `timestamp` | ISO-8601 | 带时区的事件时间 |
| `status` | enum | 见状态规则 |
| `failure_reason` | string/null | `failed` 时必填 |
| `manifest_sha256` | SHA-256 | 规范化、脱敏 manifest 的 checksum |

### 2.1 状态规则

| 状态 | 正式证据 | 规则 |
|---|---|---|
| `passed` | 是 | checkpoint/config、实际 epochs/batch 和哈希必须存在 |
| `failed` | 否 | 必须保存 `failure_reason`，不能伪装为完成 |
| `diagnostic` | 否 | 可有 checkpoint 和路由证据，但只用于 smoke/诊断 |
| `not_executed` | 否 | 不得填写 checkpoint、实际 epochs 或实际 batch |

`not_executed` 允许登记未来计划，但它不等于已启动或已失败。任何未通过训练不能通过目录存在性自动升级为 `passed`。

## 3. Routing record

路由证据为 JSONL，每一行对应一个 experiment、image、layer 和 inference repeat。

| 字段组 | 字段 |
|---|---|
| 实验身份 | `schema_version`, `experiment_id`, `model_variant`, `seed`, `dataset`, `dataset_version`, `split` |
| artifact 身份 | `checkpoint_sha256` |
| 图像身份 | `image_id`, `image_path`, `image_sha256`, `scene_groups` |
| 层身份 | `layer_name`, `layer_index` |
| 专家证据 | `expert_names`, `expert_probabilities`, `selected_expert`, `top_k` |
| token 证据 | `token_top1_indices`, `spatial_shape` |
| 推理身份 | `inference_repeat`, `inference_batch_actual`, `timestamp` |
| 结果状态 | `status`, `failure_reason` |

`image_path` 只能是脱敏相对标识，`image_id` 是跨 seed 对齐主键，`image_sha256` 防止同名样本漂移。
`expert_probabilities` 是按图像和层聚合的完整 dense probability；`token_top1_indices` 保留空间路由用于
token agreement 和利用率。两者都来自同一次 forward。
导出器要求 router 明确暴露 `(weights, indices, logits)`，使用 temperature 从 raw logits 重建 dense
probability，并用 sparse weights 与 top-k indices 交叉核对语义。若第三项已经是归一化概率、存在非有限值、
负值或专家轴不归一化，导出必须失败，不能再次 softmax 后继续。

## 4. Registry

`build_experiment_registry.py` 读取一个或多个 manifest，验证后输出：

- 规范化并排序的 `experiments`；
- 全局 `status_counts`；
- 每个 variant 的正式 seed、正式训练数和证据强度；
- 防伪重复的 `counting_rule`；
- `registry_sha256`。

以下身份冲突是硬错误：

1. 两个 manifest 使用相同 `experiment_id`；
2. 相同 variant、dataset、version、split 和 seed 被登记两次；
3. 相同 `checkpoint_sha256` 被登记为另一个独立实验或 seed；
4. registry checksum 与规范化内容不一致。

## 5. 机器可读契约

`scripts/issue54/schema.py` 暴露：

- `EXPERIMENT_MANIFEST_JSON_SCHEMA`；
- `ROUTING_RECORD_JSON_SCHEMA`；
- `validate_experiment_manifest()`；
- `validate_routing_record()`；
- 版本常量和确定性 JSON/JSONL 读写函数。

JSON Schema 使用 draft 2020-12，`additionalProperties=false`。Python validator 还执行 JSON Schema
难以表达的跨字段规则，例如状态与 artifact 的一致性、概率归一化、argmax、token 数与 shape、
checksum、路径脱敏和实际训练值约束。

## 6. 文件与目录约定

正式运行时建议每个 experiment 将以下文件放入独立 artifact 子目录，但分析器不依赖该目录名称：

```text
experiment-manifest.json
routing-records.jsonl
training-summary.json
validation-summary.json
```

Registry 和跨 seed analysis 应放在共同审计目录中。manifest 内只保存相对于显式 root 的路径；
本地用户名、盘符、AutoDL 实例路径、Token 和密钥不得写入。
三个 CLI 的成功日志只显示输出文件名，已知失败使用不含私人绝对路径的单行错误。输出父目录可以自动创建，
但已有文件默认拒绝覆盖。

## 7. MVP 与扩展登记

MVP registry 预期包含 MoT 3 个 `passed` seeds、MoE 1 个和 MoA 1 个，共 5 次正式训练。
推荐扩展上限是 MoT 5、MoE 3、MoA 1，共 9 次。

扩展 manifest 可以先用 `not_executed` 登记，但只有 MVP 审计后批准并实际通过的 checkpoint 才改为 `passed`。
3 seeds 的 MoT registry 必须标记 `exploratory_only`，不能因图像、layer、repeat 或 pairwise 行数而升级。
