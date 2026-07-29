# Issue #54 实验与研发链路

本文只记录真实发生的设计、失败、修正和复验。第一次结论即使后来被推翻也予以保留，因为本项目的
主要价值不仅是一个新 YAML，还包括如何避免把路由热力图解释成不存在的专家专业化。

## 1. 把 Issue 转成两个可证伪问题

Issue #54 要求比较 EsMoE、MoT、MoA，解释不同场景的路由，并探索混合结构。仓库已经能构建三类
模块，直接重复“模型能否运行”价值有限。代码检查后确定两个问题：

1. 固定一个 MoT checkpoint，只改变输入场景，专家路由是否出现稳定差异？
2. 完整 MoT 有 6 个 Transformer block，只在低分辨率 P5 放 1 个 MoTBlock，能否改善
   精度/延迟折中？

固定 checkpoint 是关键控制：如果分别训练航拍和医疗模型，再比较路由，输入域和模型参数会同时变化，
无法归因。

## 2. 第一版受控设计

四个模型统一使用 VisDrone、30 epoch、`imgsz=640`、batch 16、seed 42、FP32：

| key | 结构 | 作用 |
|---|---|---|
| `v10` | EsMoE backbone + 普通 neck | 基线 |
| `v10_mot` | EsMoE backbone + 6 个 MoTBlock | 完整 MoT |
| `v10_moa` | EsMoE backbone + MoA neck | 对照 |
| `v10_mot_p5` | EsMoE backbone + P5 的 1 个 MoTBlock | 新混合结构 |

最初的控制措施包括同型 4 张 RTX 5090 并行训练、同卡串行测速、checkpoint SHA-256、固定 seed
等量抽样，以及图像级 bootstrap/permutation/BH-FDR。brain-tumor 只作 OOD 路由审计，不报告
跨类别体系的医疗 mAP。

## 3. 先跑 smoke，暴露工程问题

### 3.1 四进程争用首次下载

**现象**：四个训练进程同时做 AMP 能力检查，共享参考权重尚未准备好，出现下载/读取竞争。

**设计**：父进程串行完成 preflight，再启动四个子进程。

**复验**：四模型均能并发进入训练，不再争用公共依赖。

### 3.2 autocast 下稀疏回写 dtype 不一致

**现象**：P5-MoT 在 sparse expert 路径将 FP16/BF16 输出写回 FP32 累加器时失败。

**原因**：高级索引原地赋值不会自动做普通表达式中的类型提升。

**设计**：回写前将 `expert_out * weight` 显式转为累加器 dtype，并增加 CPU bfloat16 autocast
回归测试。

**复验**：四模型 smoke 训练、benchmark 和跨域审计均跑通。

## 4. 全尺寸校准继续暴露隐藏状态

### 4.1 自动恢复把失败伪装成正常训练

**现象**：640 输入的首次 AMP epoch 出现非有限梯度，训练器自动恢复、关闭 AMP 后重跑；最终
CSV 有限，但耗时已经包含额外计算。

**风险**：只看最终曲线会漏报恢复，四模型成本比较失真。

**设计**：恢复时输出 warning，写入 `recovery_events.jsonl`，汇总恢复次数/原因/AMP fallback；
正式对比统一显式 `--no-amp`。

**复验**：30 epoch 四组均未出现 NaN、发散或恢复事件。

### 4.2 benchmark 测到了错误模型

**现象**：初版 benchmark 从 YAML 重建 80 类未训练模型；训练 checkpoint 实际为 VisDrone
10 类。

**设计**：增加 `--trained-weights`，直接加载各自 `best.pt`，同时记录 `nc`、SHA-256、Params
和实际 FLOPs。

**复验**：测速对象、检测类别和训练 checkpoint 身份一致。

### 4.3 best checkpoint 与末轮指标语义混用

**现象**：汇总行同时写 `best.pt` 身份和末轮 CSV 指标，最佳轮不是末轮时会产生歧义。

**设计**：分开记录 `final_*`、`best_observed_*` 和 best epoch，并固定以 mAP50-95 选择整行，
不拼接不同 epoch 的单指标峰值。四组被选行都在第 30 轮；MoT-P5 的 mAP50 单指标峰值则在
第 28 轮，故表中使用与第 30 轮 mAP50-95 配对的 0.17309。

## 5. 第一轮路由统计：从 token 修正到图像

直接把 token 当独立样本会严重夸大样本量。第一轮修正先按“图像 × 层 × 专家”聚合，再做
5,000 次 bootstrap、5,000 次双侧 permutation、Hedges' g 和 BH-FDR，并规定只有
`q <= 0.05` 且 CI 不跨 0 才称为稳定差异。

样本指纹审计随后发现 `dense` 与代理 `irregular_occluded` 重叠 104/128 张图。脚本因此增加
`sample_overlap.csv`，共享图像的独立样本检验自动失效。此时 dense/sparse、large/small 没有
精确图像重叠，看似可以继续。

## 6. 第一版结果

训练提交 `58cb439` 的结果：

| 模型 | mAP50-95 | mAP50 |
|---|---:|---:|
| EsMoE-N | 0.09014 | 0.17446 |
| MoT-N | 0.08912 | 0.16984 |
| MoA-N | 0.08710 | 0.16837 |
| MoT-P5-N | **0.09164** | 0.17309 |

按当时的单轮测速，MoT-P5 相比完整 MoT 的 P50 降低 58.65%，mAP50-95 增加 0.252 个百分点；
相比 EsMoE 则 P50 增加 5.74%。第一版结论是“P5 是完整 MoT 的低成本替代，但没有证明相对
EsMoE 的协同”。

图像级场景统计还得到：

- sparse 相对 dense 的 Deformable mean probability `+0.000590, q=0.00105`；
- small 相对 large 的 Window top-1 share `+0.004537, q=0.03474`。

这些结论当时满足预设检验规则，但后续复审发现统计单位仍不正确。

## 7. 完成后复审一：同一视频不是独立图像

**新问题**：VisDrone 文件名带视频序列前缀。dense 与 sparse 即使没有同一张图，也可能包含同一
序列的相邻帧；图像级检验仍存在伪重复。

**设计**：

1. 增加 `--cluster-regex`，从文件名恢复视频序列；
2. 先在序列内求均值；
3. 对两组共有序列做 paired cluster bootstrap；
4. permutation 改为 paired sign-flip；
5. Hedges' g 改为 paired effect size；
6. 显著性同时要求 FDR 和 CI，而不是只检查 q-value。

**复验**：

- dense/sparse 只剩 10 个配对序列；
- large/small 有 21 个配对序列；
- 没有任何指标同时通过 `q <= 0.05` 和 CI 条件；
- large → small 的 Window mean probability 虽有原始 `p=0.0060`，全局 FDR 后
  `q=0.126`，仍不能判为显著。

**新结论**：第一版的“稀疏偏好 Deformable”和“小目标偏好 Window”被推翻，只能保留为探索性
现象。这不是实验失败，而是避免了一个会被连续视频帧放大的伪结论。

## 8. 完成后复审二：代理遮挡不是遮挡标注

**新问题**：`irregular_occluded` 由密度、尺度和长宽比构造，既与 dense 高度重叠，也没有直接使用
遮挡标签，无法检验 Issue 中的 Deformable 遮挡假设。

**设计**：读取 VisDrone 原始 8 列标注的 occlusion 字段，取遮挡比例上下四分位；每个视频序列只
选择一对低/高遮挡图，并匹配目标数与框面积。

**第一次匹配**：局部最近邻保留 25 对，但 `log1p(目标数)` SMD 为 0.301，说明高遮挡组仍可能只是
更密集。

**再次设计**：加入全局 coordinate-descent 平衡目标，在保持“一序列一对”的前提下联合最小化
个体匹配距离和总体协变量偏差。

**再次实验**：

- 仍保留 25 对序列；
- 目标数 SMD 从 0.301 降至 0.027；
- 中位框面积 SMD 为 -0.017；
- 平均遮挡比例从 0.284 增至 0.753；
- Deformable top-1 `+0.001575`，CI `[-0.005933, 0.009746]`，`q=0.703`；
- Deformable mean probability `+0.000040`，CI 跨 0。

**结论**：控制密度和尺度后，当前 checkpoint 没有表现出遮挡场景下显著增强的 Deformable 路由。

## 9. 完成后复审三：单轮 benchmark 不稳定

**代码审计发现**：

- 用 `torch.empty` 的未初始化内容测实际 FLOPs；
- benchmark 修改了进程级梯度模式；
- 固定 50 次预热没有保证 GPU 达到稳定状态；
- 只跑一轮，模型执行顺序与数值无法分离。

**第一轮修正**：固定 seed 的 `torch.rand`、局部 generator、保留调用者梯度状态、至少 2 秒预热。

**再实验的新问题**：EsMoE 单轮 P50 为 11.084 ms，而后续轮次为 13.418/13.284 ms，说明即使
有时长预热，单轮仍不足。

**第二轮修正**：3 轮重复，每轮旋转模型顺序；报告每轮 percentile 的中位数及 run min/max。

**最终结果**：

| 模型 | P50/P95/P99 (ms) | FLOPs (G) |
|---|---:|---:|
| EsMoE-N | 13.284/13.338/13.795 | 8.016 |
| MoT-N | 30.928/31.262/32.177 | 12.014 |
| MoA-N | 26.382/26.514/27.796 | 8.613 |
| MoT-P5-N | 17.008/17.065/17.180 | 8.143 |

MoT-P5 相比完整 MoT 的 P50 降低 45.01%、FLOPs 降低 32.22%；相比 EsMoE 的 P50 增加
28.03%。方向与第一版一致，但效应量不再引用脆弱的单轮数字。

## 10. 合并上游后再验证

分支从旧基线同步到上游 `d5afc4b`。上游新增了通用 `RoutingInterpreter` 和路由恢复相关改动，
合并无冲突。当前专用脚本不重复替代通用解释器，而是补充其未覆盖的实验层能力：固定 checkpoint
跨域清单、文件哈希、TIFF、序列级统计、真实遮挡配对和复现报告。

同步后还复核了已合并 PR：#96 已覆盖 MoT/MoA+MoT 配置、基础消融和边界测试；#146 已覆盖
VisDrone 消融与跨域路由，但明确记录其两个域使用不同训练模型。由此重新收敛贡献边界：不重复认领
基础测试或首次消融，只提交 P5-only 结构、同 checkpoint 纠偏、序列级统计、真实遮挡复验与稳定
benchmark。

四个旧 checkpoint 在当前代码上重新跑完 VisDrone 548 val，mAP50-95 分别为
0.086933/0.086633/0.084307/0.087923；绝对值下降，但 MoT-P5 排名仍第一。

跨域重跑中，VisDrone → brain-tumor 的 Deformable mean probability 从初版 `+0.000620`
变为 `+0.000674`，方向和幅度接近；top-1 差值却从 `+0.009824` 变为 `+0.043936`。深层概率
接近 1/3 时，微小数值差异会放大为 argmax 跳变，因此最终解释以 dense probability、entropy
和 margin 为主，不单独依赖 top-1。

## 11. 路由器为何分化弱

梯度探针发现 `top_k=1` 时唯一选中权重重归一化为 1，主任务传给 router 的梯度极弱：

| 配置 | router 梯度范数和 |
|---|---:|
| top-k=1, exploration=0 | `8.9e-9` |
| top-k=1, exploration=0.02 | `1.53e-4` |
| top-k=2, exploration=0.02 | `9.90e-3` |

这能解释概率近均匀和 top-1 敏感，但不能证明 `top_k=2` 会提高检测。正确下一步是重训
`top_k / exploration_eps / straight-through` 多 seed 消融，而不是直接更改当前 checkpoint。

## 12. 最终判定

1. **MoT-P5 作为完整 MoT 的低预算替代：支持。** 当前 P50 降低 45.01%，FLOPs 降低
   32.22%，mAP50-95 增加 0.129 个百分点。
2. **MoT-P5 相对 EsMoE 的协同：不支持。** mAP50-95 仅增加 0.099 个百分点，P50 增加
   28.03%。
3. **密集/稀疏、大小目标存在稳定专家偏好：证据不足。** 序列级复验无显著项。
4. **遮挡提高 Deformable 激活：当前为负结果。** 25 对真实标注配对无显著变化。
5. **医疗输入：仅作 OOD 审计。** 不等于医疗检测性能或临床价值。

## 13. 证据与测试边界

| 证据 | 路径 |
|---|---|
| 当前协议 | `results/reproducibility_v2.json` |
| 训练、当前验证与三轮测速 | `results/model_comparison_v2.csv` |
| 三轮原始 latency | `results/benchmark_v2/latency_rounds.csv` |
| 序列级场景统计 | `results/visdrone_scenes_cluster/` |
| 真实遮挡配对、平衡与统计 | `results/visdrone_occlusion/` |
| 当前跨域审计 | `results/cross_domain_v2/` |
| 初版结果审计轨迹 | `results/cross_domain/`、`results/visdrone_scenes/` |

Issue 指定的 `window_size > feature map`、奇数尺寸 shift、eval 禁用 exploration 三项边界测试已存在
于当前上游 `tests/test_mot.py`；本分支复用并验证它们，没有将其误写为新增贡献。本分支新增的是
TIFF、dtype、序列配对、FDR+CI、真实遮挡匹配、确定性输入、梯度状态与多轮 benchmark 等测试。
相关回归共 `88 passed, 4 warnings`，warning 均来自既有 MoA head 数量自动调整。

公开结果包不包含模型权重、原始图像、私人线粒体数据、本地绝对路径或凭据。

## 14. 后续：从激活解释转向检测效用

图像级审计结束后继续追问“激活偏好是否真的有检测价值”，形成了：

```text
目标框内匹配审计
→ 单层强制专家检测效用矩阵
→ 冻结检测器的 utility router
→ 独立 split 失败与 KL 漂移保护
→ 保持最大 K 原语义的 adaptive K
→ 同图 mAP、三轮延迟、实际调用联合复验
```

过程中再次修正了帧号解析、面积 caliper、truncation 混杂、最大 K 改变基线路由语义以及
benchmark 中间结果不落盘、观测统计触发 CUDA 同步等问题。最终目标层实际调用下降 50.52%，
但 mAP50-95 下降 0.00048，P50 未改善；utility router 也未通过独立 test-dev。完整数据、
失败版本和 GRPO 前置条件见
[`utility_router_adaptive_k_zh.md`](utility_router_adaptive_k_zh.md)。

## 15. 交付前跨版本复验

**现象**：在项目声明支持的 Python 3.9 环境中，新增 utility 部署模块因
`from typing import Self` 在测试收集阶段失败；Python 3.11 以上不会暴露该问题。

**原因与修复**：实现误把开发环境版本当成项目最低版本。改用
`typing_extensions.Self`，不改变运行逻辑；同时新增结果包一致性测试，自动从逐 epoch、
逐轮 latency、检测指标与 utility report 复算公开汇总，并扫描本地路径和凭据。

**复验**：2026-07-29 在 Python 3.9.25、PyTorch 2.8.0+cu128 上为
`134 passed, 18 warnings`。14 条 warning 来自 Matplotlib/pyparsing 弃用提示，4 条来自既有
MoA head 自动调整，无失败。结果一致性测试还纠正了一个文字歧义：四组按 mAP50-95 选中的行均
在第 30 轮，但 MoT-P5 的 mAP50 单指标峰值在第 28 轮，不能表述为“所有单指标峰值都在第 30
轮”。RTX 5090 CUDA smoke 进一步确认同一 checkpoint 与 utility bundle 可加载，目标层实际
选择 `K=1`，退出部署上下文后恢复原固定-K 配置；该 smoke 不替代 128 图正式评测。
