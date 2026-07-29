# Issue #54 最终实验结果

本页汇总 EsMoE-N、MoT-N、MoA-N、MoT-P5-N 的受控消融，以及同一个 MoT checkpoint 的
路由审计。结论以修正后的序列级统计和三轮 benchmark 为准；初版结果保留用于说明方法如何被
问题推动迭代，不再作为最终证据。

完整的“现象 → 原因 → 修正 → 复验 → 新问题”见
[`../experiment_journey_zh.md`](../experiment_journey_zh.md)。

## 1. 三层证据

| 层次 | 目的 | 固定条件 |
|---|---|---|
| 训练实验 | 比较四种结构 | VisDrone、30 epoch、640、batch 16、seed 42、FP32 |
| 当前代码复验 | 检查合并上游后 checkpoint 是否兼容 | 同一批 `best.pt`、VisDrone 548 val |
| benchmark v2 | 比较当前推理成本 | 同卡、固定输入、3 轮、每轮 200 次、顺序轮换 |

训练提交为 `58cb439`，方法修正提交为 `9eb5076`，合入的上游提交为 `d5afc4b`。MoT
checkpoint SHA-256 为
`a1857c81b7aebd0efb5a56f9d5b37405ef83edcc68890add15c9c480e9fee629`。

公开协议见 [`reproducibility_v2.json`](reproducibility_v2.json)，统一表见
[`model_comparison_v2.csv`](model_comparison_v2.csv)。

## 2. 消融与兼容性复验

### 2.1 原训练提交

| 模型 | mAP50-95 | mAP50 | 训练时间 | NaN/发散/恢复 |
|---|---:|---:|---:|---|
| EsMoE-N | 0.09014 | 0.17446 | 0.945 h | 0 |
| MoT-N | 0.08912 | 0.16984 | 1.381 h | 0 |
| MoA-N | 0.08710 | 0.16837 | 1.274 h | 0 |
| **MoT-P5-N** | **0.09164** | 0.17309 | 1.066 h | 0 |

四组最佳观测值均出现在第 30 轮。逐 epoch CSV 和曲线位于 [`training/`](training/)。
这是同预算架构比较，不是充分收敛的 SOTA 复现。

### 2.2 合并当前上游后的复验

| 模型 | mAP50-95 | mAP50 |
|---|---:|---:|
| EsMoE-N | 0.086933 | 0.168935 |
| MoT-N | 0.086633 | 0.167121 |
| MoA-N | 0.084307 | 0.163315 |
| **MoT-P5-N** | **0.087923** | **0.167885** |

绝对值相对训练日志下降，但四个 checkpoint 均能加载，MoT-P5 的排序仍保持第一。原始值见
[`benchmark_v2/post_merge_validation.csv`](benchmark_v2/post_merge_validation.csv)。
报告不把训练提交的 mAP 与当前代码的延迟伪装成同一次运行，而是同时保留两层身份。

## 3. benchmark 如何被修正

第一次 benchmark 使用单轮、固定 50 次预热，并以未初始化张量测实际 FLOPs。代码复查还发现
benchmark 在函数内部修改了全局梯度模式。这些问题不会改变模型参数，但会让输入、执行顺序和调用者
状态成为隐藏变量。

修正后：

1. 使用局部随机生成器和固定 seed 0 创建输入；
2. 不修改全局 RNG 与梯度模式；
3. 每轮至少预热 50 次且不少于 2 秒；
4. 运行 3 轮，每轮旋转模型顺序；
5. 汇总各轮 percentile 的中位数，同时保留 run min/max；
6. profiler 与 latency 使用同一确定性输入。

| 模型 | P50 / P95 / P99 (ms) | P50 run min-max | 实际 FLOPs (G) | Params (M) |
|---|---:|---:|---:|---:|
| EsMoE-N | 13.284 / 13.338 / 13.795 | 11.084-13.418 | 8.016 | 3.420 |
| MoT-N | 30.928 / 31.262 / 32.177 | 30.879-31.106 | 12.014 | 4.025 |
| MoA-N | 26.382 / 26.514 / 27.796 | 26.268-26.457 | 8.613 | 3.546 |
| **MoT-P5-N** | **17.008 / 17.065 / 17.180** | **16.989-17.058** | **8.143** | **3.494** |

EsMoE 首轮 P50 为 11.084 ms，后两轮为 13.418/13.284 ms，证明单轮测速不足以描述波动。
原始 12 行数据见 [`benchmark_v2/latency_rounds.csv`](benchmark_v2/latency_rounds.csv)。

### 3.1 混合架构判定

按当前代码复验，MoT-P5 相对完整 MoT：

- mAP50-95 增加 0.129 个百分点；
- P50/P95/P99 降低 45.01%/45.41%/46.61%；
- FLOPs 降低 32.22%，Params 降低 13.20%。

相对 EsMoE，MoT-P5 的 mAP50-95 增加 0.099 个百分点，但 P50 增加 28.03%，FLOPs 增加
1.58%。因此它是完整 MoT 的明确低预算替代，不是比 EsMoE 更快的协同结构。Issue 的
“mAP 提升 > 1%”若指绝对百分点也未达到，本报告不声明协同增益。

## 4. 场景统计为何被推翻

### 4.1 初版图像级结果

初版以图像为独立单位，得到 sparse 相对 dense 的 Deformable mean probability
`+0.000590 (q=0.00105)`，small 相对 large 的 Window top-1 share
`+0.004537 (q=0.03474)`。

进一步检查文件名发现，VisDrone 验证图来自连续视频。不同图像即使没有精确重叠，也可能属于同一
视频序列；把相邻帧当 128 个独立样本会夸大有效样本量，这是伪重复。

### 4.2 序列级配对复验

脚本现在先按视频序列聚合，再对两组共有序列做 paired bootstrap 和 paired sign-flip
permutation，并同时要求 `q <= 0.05` 且 CI 不跨 0。

| 比较 | 配对序列 | 代表性变化 | 95% CI | q | 最终判定 |
|---|---:|---:|---:|---:|---|
| dense → sparse | 10 | Deformable top-1 `+0.004057` | `[-0.003558, 0.011606]` | 0.683 | 不显著 |
| dense → sparse | 10 | Window mean prob `+0.000261` | `[0.000030, 0.000495]` | 0.359 | 不显著 |
| large → small | 21 | Window mean prob `-0.000203` | `[-0.000330, -0.000074]` | 0.126 | 不显著 |

修正后没有任何场景路由指标同时通过 FDR 和 CI 条件。初版的“稀疏偏好 Deformable”和“小目标
偏好 Window”均降级为探索性观察，不能作为场景推荐。证据见
[`visdrone_scenes_cluster/`](visdrone_scenes_cluster/)。

## 5. 用真实遮挡标注重做假设

初版 `irregular_occluded` 只是密度、尺度、长宽比代理，并与 dense 重叠 104/128 张图。为直接
检验遮挡假设，第二轮读取 VisDrone 原始 8 列标注中的 occlusion 字段：

1. 取遮挡比例上下四分位候选；
2. 每个视频序列各选一张低遮挡和一张高遮挡图；
3. 以 `log1p(目标数)` 和 `log(中位框面积)` 匹配；
4. 用全局平衡目标再次优化配对。

第一次最近邻匹配后，目标数标准化差异 SMD 为 0.301，说明遮挡仍与密度混杂。全局平衡后保留
25 对序列，目标数 SMD 降至 0.027，目标面积 SMD 为 -0.017；平均遮挡比例由 0.284 增至
0.753。

复验中 Deformable top-1 仅增加 `0.001575`，95% CI
`[-0.005933, 0.009746]`，`q=0.703`；Deformable mean probability 增加
`0.000040`，CI 也跨 0。当前 checkpoint 不支持“复杂遮挡显著提高 Deformable 激活”的假设。
配对与平衡证据见 [`visdrone_occlusion/`](visdrone_occlusion/)。

## 6. 跨域与路由器新问题

同一 MoT checkpoint 从 VisDrone 切换到 brain-tumor 后，Deformable mean probability 增加
`0.000674`，95% CI `[0.000520, 0.000834]`，`q=0.000233`。绝对变化小，且模型未在
brain-tumor 上训练；它只说明 OOD 输入改变了路由分布，不代表医疗检测或临床价值。

重跑时 Deformable top-1 差值由初版 `+0.009824` 变为 `+0.043936`，而 dense probability
变化仅从 `+0.000620` 变为 `+0.000674`。结合深层各专家概率接近 1/3，这说明 argmax 会被微小
数值差异放大，top-1 share 不应单独承担解释。

梯度探针也发现 `top_k=1` 时唯一权重归一化为 1，router 主任务梯度很弱：

| 配置 | router 梯度范数和 |
|---|---:|
| top-k=1, exploration=0 | `8.9e-9` |
| top-k=1, exploration=0.02 | `1.53e-4` |
| top-k=2, exploration=0.02 | `9.90e-3` |

这解释了弱分化，但没有证明改为 top-k=2 会提高检测；需要重训消融，不能直接修改现有 checkpoint
的推理行为。

## 7. 数据支持的建议

1. **低预算保留 MoT：选 MoT-P5。** 相比完整 MoT，当前 P50 降低 45.01%、FLOPs 降低
   32.22%，mAP50-95 增加 0.129 个百分点。
2. **速度优先：仍选 EsMoE。** MoT-P5 相比 EsMoE 的 P50 高 28.03%，当前没有速度协同。
3. **不要按密集/稀疏或大小目标切换专家。** 序列级复验没有指标通过复合显著性标准。
4. **不要用 top-1 热力图单独解释路由。** 应同时报告 dense probability、entropy、margin 和
   运行间稳定性。
5. **遮挡假设当前为负结果。** 25 对真实标注、密度与尺度平衡后，Deformable 激活未显著上升。

## 8. 限制与证据索引

- 30 epoch、单 seed，不能代表充分收敛上限；
- 场景比较只有 10/21 个配对序列，统计功效有限；
- 遮挡配对有 25 个序列，仍需更多数据和 route-aware 重训；
- 单张 RTX 5090 的 PyTorch 延迟不能外推到 TensorRT、ncnn 或其他硬件；
- 水平翻转和亮度扰动下主导专家一致率为 97.79%-99.61%，但概率近均匀使“稳定”不等于“专业化”；
- 公开包不含权重、原始图像、私人线粒体数据或本地绝对路径。

| 内容 | 文件 |
|---|---|
| 训练指标与初版单轮测速（历史） | [`model_comparison.csv`](model_comparison.csv) |
| 当前兼容性验证与三轮测速 | [`model_comparison_v2.csv`](model_comparison_v2.csv) |
| 当前复现协议 | [`reproducibility_v2.json`](reproducibility_v2.json) |
| 当前跨域审计 | [`cross_domain_v2/`](cross_domain_v2/) |
| 序列级场景审计 | [`visdrone_scenes_cluster/`](visdrone_scenes_cluster/) |
| 真实遮挡配对审计 | [`visdrone_occlusion/`](visdrone_occlusion/) |
| 初版图像级结果（审计轨迹） | [`cross_domain/`](cross_domain/)、[`visdrone_scenes/`](visdrone_scenes/) |
| 逐 epoch 指标与曲线 | [`training/`](training/) |

## 9. 检测效用路由扩展

后续实验不再用“专家激活率”代替检测价值，而是对一个 MoT 层逐专家强制干预，建立检测损失效用
矩阵，再训练冻结检测器的 utility router。

核心结果：

- 修正帧号、面积 caliper 和 truncation 后，目标级遮挡审计保留 12,296 对、76 个序列；
- `model.14.m.0` 的 2,048 图矩阵中，原路由平均 regret 为 0.03380；
- 场景残差 router 在 calibration val 将 regret 从 0.02186 降至 0.01737，但在 test-dev
  升至 0.05476；
- KL guard 在 test-dev 回退原路由，避免退化，但没有产生增益；
- adaptive K 阈值 0.35 将目标层实际专家调用从 3.000 降至 1.484，mAP50-95 从
  0.08743 降至 0.08695，三轮 P50 中位数从 26.877 ms 升至 28.244 ms。

脱敏统计与三轮原始测速见 [`utility_routing/`](utility_routing/)，完整迭代链见
[`../utility_router_adaptive_k_zh.md`](../utility_router_adaptive_k_zh.md)。
