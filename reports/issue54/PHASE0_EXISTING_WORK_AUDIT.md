# Issue #54 Phase 0：已有成果与重合风险审计

> 审计日期：2026-07-30
> 审计基线：`upstream/main@d5afc4b442ae`
> 工作分支：`issue54-mot-routing-stability`
> 研究方向：MoT 多随机种子路由稳定性、可重复性与结果有效性
> 状态：仅审计与规划；未下载数据、未训练、未修改核心模型代码

## 1. 审计边界

- Issue #54 当前为 **Open**；官方正文要求在 COCO 或 VisDrone 上至少训练
  EsMoE-N、v0.10-MoT-N、v0.10-MoA-N 三种变体，并统一报告精度、延迟、
  实际 FLOPs、参数量、训练稳定性和路由解释结果。
- Issue #54 的认领状态已由用户确认；公开文档不保留认领账号等非必要身份信息。
- 本审计只覆盖 Issue #54。不得恢复、读取或创建任何 Issue #36 内容。
- `reports/issue50`、`cloud_backup`、Issue #50 的分支、worktree、stash、实验与提交均为只读保护对象。
- 以下方向已明确排除：普通三模型消融、基础边界测试、已有跨域路由解释、
  P5-only MoT、utility router、adaptive K。

官方 Issue：[Tencent/YOLO-Master#54](https://github.com/Tencent/YOLO-Master/issues/54)

## 2. 指定 PR 的实时状态与实际落点

| PR | 2026-07-30 状态 | 经文件清单与当前代码核验的主要内容 | 对本项目的约束 |
|---|---|---|---|
| [#95](https://github.com/Tencent/YOLO-Master/pull/95) | Merged | `diagnose_mot_routing.py`、MoT 路由诊断测试、window/shift/exploration 边界覆盖；当前 main 可见 | 不再做基础路由诊断或重复边界测试 |
| [#96](https://github.com/Tencent/YOLO-Master/pull/96) | Merged | v0.10 MoT、MoA+MoT YAML；`compare_mot_ablation.py`；场景准备、诊断和 50 epoch VisDrone 结果 | 普通 MoE/MoT/MoA/MoA+MoT 消融不是创新点 |
| [#107](https://github.com/Tencent/YOLO-Master/pull/107) | Merged | VisDrone 准备/运行脚本、`analyze_mot_routing.py`、K8s runner 和结果报告 | 不重复数据下载器、基础逐图路由热力图和已有报告 |
| [#146](https://github.com/Tencent/YOLO-Master/pull/146) | Merged | 三模型 30 epoch 消融、跨域路由解释、一键 pipeline、更多边界测试 | 不重复普通三模型消融或跨域结论 |
| [#137](https://github.com/Tencent/YOLO-Master/pull/137) | Open / Draft | parser repeat 修复、显式 shift 修复、letterbox、图像级聚合、bootstrap/permutation、Hedges' g、FDR、checkpoint benchmark | 若合并应直接复用；不宣称这些通用修复为本项目贡献 |
| [#189](https://github.com/Tencent/YOLO-Master/pull/189) | Open | 仅改 `tests/test_mot.py`，增加训练态 window/shift/exploration 边界与梯度测试 | 完全避开该测试方向 |
| [#190](https://github.com/Tencent/YOLO-Master/pull/190) | Open | P5-only、序列/遮挡审计、单 checkpoint 跨域和扰动稳健性、JSD/entropy、utility router、adaptive K；单 seed 42 | 不重复这些实验；若合并则扩展其指标原语到跨训练 seed，而非另写同类单模型分析 |

注意：GitHub API 对已合并 PR 返回的 `state` 是 `closed`，上表以 `merged=true`
区分 Merged 与普通 Closed。

## 3. 当前 `upstream/main` 中已有能力

### 3.1 统一消融与模型配置

`scripts/compare_mot_ablation.py` 当前具备：

- v0.10 EsMoE、MoT、MoA、MoA+MoT 等模型登记；
- build、训练、summary、P50/P95/P99 latency 和实际 FLOPs 入口；
- mAP50-95、mAP50、loss NaN/发散摘要；
- resume 与 deterministic 参数；
- **仅一个** `--seed` 整数参数，默认 42；run 目录不包含 seed 维度。

当前 v0.10 配置：

- `yolo-master-n.yaml`：VisualEnhancedAdaptiveGateMoE backbone；
- `yolo-master-mot-n.yaml`：同一 backbone，neck 中三个 C2fMoT stage；
- `yolo-master-moa-n.yaml`：同一 backbone，neck 中三个 C2fMoA stage；
- `yolo-master-moa-mot-n.yaml`：MoA/MoT 交叉 neck；
- `yolo-master-mot-scene-n.yaml`：已有的 opt-in scene-aware MoT 实验配置。

因此，三模型协议的模型入口已经存在；本项目不需要新建基础模型 YAML。

### 3.2 MoT、MoE、MoA 模块

- `ultralytics/nn/modules/mot/` 已实现三个具固定语义身份的专家：
  `LocalConvTransformer`、`WindowTransformer`、`DeformableTransformer`。
- `_MoTRouter.forward(..., return_logits=True)` 能同时返回：
  post-top-k weights、top-k indices 和 pre-softmax logits。
- `MoTBlock` 在正常模型前向中已经请求 `return_logits=True`。因此后续分析脚本可由
  hook 的第三个输出恢复完整 dense probability，无需为了 JSD 修改路由核心。
- `ultralytics/nn/modules/moe/` 已有路由快照、usage、hooks、diagnostics 和可视化基础设施。
- `ultralytics/nn/modules/moa/` 已有 MoA block/router/wrapper；本项目只把它作为官方比较组，
  不重新研究其基础实现。

### 3.3 路由诊断和 benchmark

| 文件 | 当前能力 | 当前缺口 |
|---|---|---|
| `scripts/analyze_mot_routing.py` | 单模型逐图、逐 MoTBlock 的专家平均权重、top-1 token fraction、场景标签和热力图 | 直接 resize；只保存 post-top-k weights；无多个 checkpoint 对齐 |
| `scripts/diagnose_mot_routing.py` | 单模型逐图/逐层/逐专家记录；图像级场景聚合；单 seed 的 bootstrap/permutation | bootstrap 的 `seed=0` 是统计重采样 RNG，不是独立训练 seed |
| `scripts/prepare_mot_routing_scenes.py` | 生成密集/稀疏、大小目标、遮挡等场景分组 | 不提供跨 seed 对齐或重现率 |
| `scripts/run_issue54_pipeline.py` | 已有 Issue #54 基础 pipeline 编排 | 不提供多训练 seed 矩阵 |
| `benchmarks/benchmark_mot_dispatch.py` | MoT dispatch 性能 micro-benchmark | 不检查 checkpoint 路由可重复性 |
| `tests/test_mot.py` | forward/backward、温度、YAML、window、shift、exploration、稀疏推理等边界 | 不应重复；不是训练级可重复性测试 |

仓库其他 PEFT 脚本存在通用的 multi-seed 结果聚合范式，但没有一个工具将多个
MoT 独立训练 checkpoint 在相同图像、相同层、相同专家上对齐。

## 4. 十项候选研究缺口核验

| # | 候选能力 | 当前 main | Open PR | 审计结论 |
|---:|---|---|---|---|
| 1 | 多随机种子独立训练 | 单个 `--seed`，目录无 seed 维度 | #137 单 seed 42；#190 单 seed 42 | **未实现** |
| 2 | 相同验证图像上的跨 seed 路由对齐 | 仅单 checkpoint | #190 明确“one immutable checkpoint” | **未实现** |
| 3 | per-image / per-layer / per-expert route agreement | 有单 checkpoint 原始摘要 | #190 有扰动前后 agreement，不是训练 seed 间 agreement | **跨 seed 未实现** |
| 4 | 路由概率 Jensen-Shannon divergence | 当前 main 无 | #190 有单 checkpoint 图像扰动 JSD | **指标原语在开发，跨 seed 未实现** |
| 5 | route entropy | 当前 main 无正式逐图输出 | #190 有单 checkpoint normalized entropy | **指标原语在开发，跨 seed 未实现** |
| 6 | 专家利用率在 seed 间的方差 | 无 | 无 | **未实现** |
| 7 | 场景级结论的跨 seed 重现率 | 仅单 seed 场景结论 | #190 改善序列级分析，但仍单 seed | **未实现** |
| 8 | 稳定性与 mAP/loss/尺度/遮挡的相关性 | 分散报告，无跨 seed 联结 | #190 有单 checkpoint 尺度/遮挡审计 | **跨 seed 联结未实现** |
| 9 | checkpoint 重复推理确定性 | 有 deterministic benchmark input 和重复 latency round | 未比较重复前向的 route indices/probabilities/output | **未完整实现** |
| 10 | seed 级 bootstrap/effect size | 当前 bootstrap 以图像为单位 | #190 能以 sequence cluster 为单位，但没有训练 seed 顶层 | **未实现** |

结论：十项均未在“独立训练 seed”层面形成完整闭环。第 4、5 项以及部分第 3、7、8、
10 项与 PR #190 的单 checkpoint 分析存在方法原语重合，必须采用“复用后扩展到跨 seed”
策略，不能把 JSD、entropy 或 cluster bootstrap 本身包装成新贡献。

## 5. 真实且独立的研究问题

本项目的创新问题应表述为：

> 在固定数据、模型拓扑、训练预算和验证清单下，MoT 的语义专家路由是否能在多个
> 独立训练随机种子间复现；路由不稳定性是否超出普通训练方差，并是否与检测质量、
> 目标尺度、遮挡和场景结论的不稳定共同变化？

独立性来自以下组合，而不是来自单个常见指标：

1. 多个从头独立训练的 MoT checkpoint；
2. 同一验证图像清单、同一预处理、同一层/专家 schema 的严格对齐；
3. 完整概率分布、离散路由和专家利用率的联合度量；
4. MoE 多 seed 控制，用于区分“MoT 特有不稳定”与“一般训练随机性”；
5. seed 为最高层实验单位、sequence/image 为嵌套单位的统计设计；
6. null/negative result 也保留并解释，不按结果选择 seed。

## 6. 明确禁止重复

- 不把 EsMoE/MoT/MoA 三模型单 seed 排名作为创新结论；
- 不新增已有 window、odd shift、exploration_eps 基础边界测试；
- 不重新制作现有跨域路由热力图或重复 Deformable 场景论证；
- 不做 P5-only MoT；
- 不做 utility router；
- 不做 adaptive K；
- 不把“统计重采样使用了 seed”误称为“多随机种子训练”；
- 不把同一 seed 内数百张图像当作数百个独立训练重复；
- 不复刻 PR #137/#190 的通用统计代码；其合并后优先扩展现有实现。

## 7. Phase 0 审计结论

该方向仍有真实研究空间，且与指定 PR 的核心差异明确。可进入下一阶段的前提是：

- 用户确认实验规模和预算；
- 开跑前再次检查 PR #137/#189/#190 状态及 upstream/main；
- 将分析 schema、seed 列表、epoch 预算和停止规则冻结；
- 不以现有他人 checkpoint 替代官方要求的独立三模型训练证据。
