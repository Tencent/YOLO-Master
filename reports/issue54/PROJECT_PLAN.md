# Issue #54：MoT 多随机种子路由稳定性项目计划

> 计划版本：Phase 0 / Draft 0.1
> 基线：`upstream/main@d5afc4b442ae`
> 当前阶段禁止：数据下载、训练、核心模型修改、commit、push、PR

## 1. 项目目标

完成 Issue #54 官方最低交付，同时把研究增量聚焦在：

- MoT 独立训练 seed 间的路由稳定性和结果可重复性；
- 路由概率、离散路由、专家利用率和检测结果的联合有效性；
- seed 级统计推断，避免图像级伪重复；
- 可审计的实验清单、checkpoint 身份、失败记录和负结果管理。

本项目不会“只训练 MoT”。官方三模型统一协议是合规层；MoT 多 seed 是创新层。

## 2. 官方最低合规方案

### 2.1 必须独立运行

在同一数据集、split、图像尺寸、epoch、optimizer、batch/effective batch、增强、精度、
硬件、验证和 latency 协议下，至少独立训练：

1. `yolo-master-n.yaml`：EsMoE-N；
2. `yolo-master-mot-n.yaml`：MoT-N；
3. `yolo-master-moa-n.yaml`：MoA-N。

最低合规矩阵为同一个预注册 seed 的 3 次训练。三者均需独立生成：

- mAP50-95、mAP50；
- P50/P95/P99 latency；
- 实际执行 FLOPs；
- Params；
- loss 曲线、NaN/发散记录；
- 配置快照、git SHA、数据 manifest hash、checkpoint SHA-256。

### 2.2 已有结果如何使用

- PR #96、#146、#137、#190 的 30/50/300 epoch 结果只能作为外部参考、
  sanity range 和差异解释。
- 这些结果来自不同 GPU、epoch、batch、AMP、代码 SHA 或数据处理，不能拼接成我们的
  统一比较表，不能替代独立三模型训练。
- 已有 checkpoint 可用于 Phase 1 的 schema smoke test 或分析器兼容性检查；
  不作为本项目官方比较的主要证据。
- 如以后使用共同预训练初始化，必须三个模型都遵循可比的、固定 hash 的初始化政策；
  当前建议按现有参考脚本从 YAML 初始化并设 `pretrained=False`。

### 2.3 避免与已有 PR 重复

官方三模型 run 被明确标记为“合规复验”，不声称是算法创新。PR 贡献聚焦在：

- 多 checkpoint manifest 与严格对齐；
- 跨 seed 路由度量；
- 重复推理确定性检查；
- seed/sequence/image 分层统计；
- 合成数据单元测试和复现文档。

## 3. 创新研究规模

### 3.1 方案矩阵

| 方案 | MoT seeds | MoE seeds | MoA seeds | 唯一训练次数 | 目的 |
|---|---:|---:|---:|---:|---|
| 官方最低 | 1 | 1 | 1 | 3 | 合规验证，不能支持强多 seed 结论 |
| MVP | 3 | 1 | 1 | 5 | 首次跨 seed 路由对齐；结论以描述性/探索性为主 |
| **推荐** | **5** | **3** | **1** | **9** | MoT 稳定性主研究 + MoE 路由控制 + MoA 合规 |
| 可选完整对照 | 5 | 3 | 3 | 11 | 预算充足时估计 MoA 的一般训练方差 |

MVP 预注册 seeds：`17, 42, 73`。

推荐 seeds：MoT 为 `17, 42, 73, 101, 137`；MoE 为 `17, 42, 73`。

seed 列表必须在首个正式 run 前冻结。不得看到结果后删除“不好”的 seed，也不得只给
表现好的 seed 增加 epoch。

### 3.2 为什么推荐 MoE 多 seed、暂不推荐 MoA 多 seed

- MoE 同样是路由模型。MoE 3 seeds 能估计一般 mixture router 的训练随机性，
  从而判断 MoT 不稳定是否特殊。
- MoA 是官方必要比较组，但它不是本项目主要的语义 Transformer 专家路由对象；
  单 seed 足够完成最低合规。
- 如果 MoT 与 MoE 差异很小，或评审要求完整训练方差，再把 MoA 扩展到 3 seeds。

## 4. 里程碑与停止点

| 阶段 | 工作 | 预计日历时间 | 结束条件 |
|---|---|---:|---|
| Phase 0 | 同步、分支、审计、协议草案 | 1 天 | 本次三份文档完成 |
| Phase 1 | 再同步；冻结 manifest/schema；实现 dry-run 分析骨架和测试 | 2–4 天 | 不下载/训练前由用户批准 |
| Phase 2 | 数据校验与 3 模型协议 smoke；校准显存/吞吐/GPU 预算 | 1–2 天 | 协议冻结，不用 smoke 作研究结论 |
| Phase 3A | 官方三模型正式训练与统一 benchmark | 3–5 天 | 三个有效 run 完成 |
| Phase 3B | MoT 3 或 5 seeds；推荐方案加入 MoE 3 seeds | 5–12 天 | 所有预注册 run 有成功/失败状态 |
| Phase 4 | 同图路由提取、确定性检查、统计与图表 | 2–4 天 | 审计表与结论边界完成 |
| Phase 5 | 代码测试、报告/Discussion、PR 范围整理 | 2–3 天 | 用户批准后才 commit/push/PR |

按单张 RTX 4090 顺序执行的规划情景：

- MVP：约 8–12 天；
- 推荐方案：约 14–21 天；
- 可选完整对照：约 18–28 天。

## 5. RTX 4090 GPU 预算

以下是 Phase 0 的区间规划，不是本机实测，也不代表训练协议已经冻结。估算以
VisDrone、640 px、Nano 模型和单张 RTX 4090 为参考情景；epochs、requested/actual
batch、AMP/FP32、optimizer 等必须在 Phase 2 的统一 CPU/CUDA smoke 后共同冻结。
如果 smoke 改变任一假设，应重新计算预算，不得沿用本表作为正式资源记录。

| 方案 | 训练 GPU 小时 | 路由提取/验证/benchmark | 总计 |
|---|---:|---:|---:|
| 官方最低 3 runs | 15–27 h | 2–4 h | 17–31 h |
| MVP 5 runs | 27–45 h | 3–6 h | 30–51 h |
| **推荐 9 runs** | **55–90 h** | **5–10 h** | **60–100 h** |
| 完整 11 runs | 65–110 h | 6–12 h | 71–122 h |

如果 smoke 后冻结的 epoch 预算高于估算情景，训练预算将相应增加。不能只延长某个架构
或某些表现好的 seeds；任何正式预算调整必须对预注册比较组一致，或另列为明确的探索性阶段。

## 6. 计划提交到 PR 的代码

后续经用户批准可形成小而独立的代码 PR：

- 扩展或新增 multi-seed runner，run path 强制包含 model/seed；
- checkpoint inventory 和 SHA-256 manifest；
- 验证图像 manifest 与跨 checkpoint schema 对齐器；
- full dense probability、top-k indices、per-image/per-layer/per-expert 导出；
- route agreement、JSD、normalized entropy、utilization variance；
- checkpoint 重复推理确定性检查器；
- seed/sequence/image 分层统计实现；
- 合成 tensor/CSV 的单元测试；
- CLI、输出 schema 和复现文档。

优先扩展已合并的上游实现。若 PR #137 或 #190 在开发期间合并，先 rebase 并复用其
letterbox、JSD、entropy、cluster bootstrap 和 FDR 原语，不并行维护重复版本。

## 7. 只放实验报告或 Discussion 的内容

- 模型权重、原始数据、完整训练日志和大体积逐图矩阵；
- 具体 GPU 小时、显存峰值、精度/延迟数值和 Pareto 图；
- “某专家在某场景更稳定”等经验结论；
- null/negative result、失败 run 原因和外推限制；
- 未达到稳定增益的新 YAML 或未验证训练策略；
- 与已有 PR 结果的外部比较。

只有稳定、通用、有测试覆盖且不改变默认行为的工具代码适合 PR。实验结论应放
`reports/issue54` 和最终 GitHub Discussion。

## 8. 主要风险与缓解

| 风险 | 后果 | 缓解 |
|---|---|---|
| #137/#190 合并造成代码重合 | PR 冲突或创新性下降 | 每阶段开跑前审计；复用其原语，只保留跨 seed 层 |
| 3 seeds 样本太少 | CI 和方差估计不稳定 | MVP 明确为探索性；正式推荐 5 MoT seeds |
| 图像/视频帧伪重复 | 过窄 CI、虚假显著 | seed 为顶层单位；VisDrone sequence 为第二层 cluster |
| post-top-k 权重被误当完整概率 | JSD/entropy 失真 | 由 hook 返回的 logits 和 checkpoint temperature 重建 dense probability |
| expert/schema 不一致 | 错误对齐 | 固定同一 YAML/commit；核验模块名、专家身份、tensor shape；不一致即停止 |
| 非确定 CUDA/预处理 | 把推理噪声误当 seed 差异 | 先做同 checkpoint 重复推理；固定环境、order、batch、dtype |
| 训练失败或 OOM | 选择性缺失 | 统一降低所有组 batch 或匹配 effective batch；保留 invalid run，不静默替换 |
| 场景标签混杂 | 结论无法解释 | 使用原始尺度/遮挡标注；sequence 配对；报告原始 effect 与 CI |
| 结论选择偏差 | 夸大稳定性/收益 | 预注册主指标、seed、停止规则；保留 null/negative result |

## 9. 强制停止条件

出现任一情况时暂停并向用户汇报，不自行扩大范围：

1. 同名分支、输出目录或受保护 Issue #50 资产发生冲突；
2. upstream 或新 PR 已完整实现同一跨训练 seed pipeline；
3. checkpoint 的模型拓扑、专家身份或层名无法一一对应；
4. 同 checkpoint 重复推理在控制环境后仍不稳定；
5. 数据来源、split 或 manifest 无法验证；
6. GPU 预算超过用户批准范围；
7. 正式 run 与冻结协议不一致；
8. 需要修改 MoT 核心默认行为才能继续；
9. 用户尚未批准进入 Phase 1 或启动任何数据/训练操作。

## 10. Phase 0 完成标准

- 干净分支建立在最新 `upstream/main`；
- 指定 PR 与当前代码均完成审计；
- 真实跨 seed 缺口和重合边界明确；
- 官方最低、MVP、推荐方案及 GPU 预算成文；
- 未下载数据、未训练、未 commit、未 push、未创建 PR。
