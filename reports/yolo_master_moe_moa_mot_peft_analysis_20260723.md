# YOLO-Master 深度技术分析报告：MoE / MoA / MoT / PEFT（含 Adapter Planner）

> 分析对象：`/Users/gatilin/PycharmProjects/YOLO-Master-v260720`（v260720 工作区）
> 报告日期：2026-07-23
> 分析方法：核心源码精读（`ultralytics/nn/modules/{moe,moa,mot}`、`ultralytics/nn/peft`、`ultralytics/vpeft`、`ultralytics/utils/lora`）+ 官方文档（README、wiki、wiki-content）+ 实验报告（`reports/`、`docs/plans/`、ablation 结果）交叉验证。

---

## 0. 项目概览

**YOLO-Master**（Tencent Youtu Lab，CVPR 2026，arXiv 2512.23273）是基于 ultralytics 8.4.101 深度改造的实时目标检测（RTOD）框架，核心命题是 **"compute-on-demand"**：用实例条件自适应计算取代静态稠密计算。

- **主结果**：YOLO-Master-N 在 MS COCO 取得 **42.4% AP @ 1.62ms**，比 YOLOv13-N **+0.8 mAP 且快 17.8%**；在 COCO/VOC/VisDrone/KITTI/SKU-110K 五个基准全面领先。
- **四大动态化子系统**（本报告重点）：
  | 子系统 | 回答的问题 | 路由对象 | 稀疏性 |
  |:---|:---|:---|:---|
  | **ES-MoE** | "谁来处理"（FFN/卷积专家） | 卷积专家组 | Top-K 稀疏（eval eager-sparse） |
  | **MoA** | "用什么感受野看" | 3 组注意力头 | dense 软混合（不省 FLOPs） |
  | **MoT** | "用什么结构范式理解" | 3 个完整 Transformer 架构 | 软 Top-K（训练 dense / eval sample-sparse） |
  | **PEFT/MoLoRA** | "如何低成本适配下游" | LoRA 专家 + 层放置规划 | Top-K 稀疏 LoRA 专家 |
- **配置版本线**：`ultralytics/cfg/models/master/` 下 v0 → v0_15 共 16 个版本目录 + `exp/` 实验线，是一部完整的 MoE 架构演进史。
- **测试资产**：tests/ 共 99 个测试文件，其中 29 个专测 MoE/MoA/MoT/MoLoRA（MoA/MoT/MoE 146 passed；MoLoRA/PEFT 156 passed / 1 failed；Planner 106 passed）。
- **Agent 运行层**：`agent/` 为 yolo-master-agent Skill Bundle（9 个 skill runner，50+ 用例 8 套件，2026-06-29 全量通过）。

---

## 1. ES-MoE（Efficient Sparse Mixture-of-Experts）—— 核心创新

代码：`ultralytics/nn/modules/moe/`（26 个文件，按 STABLE / EXPERIMENTAL / LEGACY 三级 API 管理）

### 1.1 架构设计

- **经典块 `ES_MOE`**（`modules.py`）：`DynamicRoutingLayer` + N 个 `EfficientExpertGroup`，专家为 **kernel 3/5/7 的深度可分离卷积组**（多尺度感受野分工）。训练期 dense forward，eval 时 eager sparse Top-K。
- **路由器 `UltraEfficientRouter`**（`routers.py`）：8× avgpool 下采样 + DW 卷积 + reduction=16 通道压缩 + 1×1 投影，**FLOPs 比局部路由器低约 95%**。训练注入 noise、logits clamp ±30、z-loss = `logsumexp(logits/T)²`。`BaseRouter` 支持 `capacity_factor` 防 token 溢出。
- **专家库**（`experts.py`）：`SimpleExpert` / `GhostExpert` / `FusedGhostExpert` / `InvertedResidualExpert` / `SharedInvertedExpertGroup` / `EfficientExpertGroup`。**全部使用 GroupNorm**——Top-K 后 BN 在 n=1 时不稳定，这是踩坑后的工程决策。

### 1.2 版本演进主线（`gated.py`，112KB，项目的"活历史"）

| 版本 | 类名 | 关键变化 | 命运 |
|:---|:---|:---|:---|
| v0.6 | `HybridAdaptiveGateMoE` | 混合门控核心 | **模块消融 mAP50-95=0.61017，确立基线** |
| v0.7–v0.10 | low-rank / refine / detail / context 增强 | 均为**负收益，已弃用** | ❌ |
| v0.11 | `HybridAdaptiveGateMoEv2` | +`DualStreamGateRouterV2` | 过渡 |
| v0.12 | `OptimalHybridGateMoE` | **生产核心**（见下） | ✅ |
| v0.13/v0.14 | `MultiHeadRouterMoE` / `DiversifiedExpertMoE` | 消融变体 | 实验 |
| v0.15 | `GatedFusionMoE` | **当前最新** | ✅ |

**v0.12 设计要点**（docstring 明确记载）：SE-gated 静态/动态 split（`split_ratio` 按插入点 0.5/0.5/0.375，浅层多动态、P5 多静态）+ `DualStreamGateRouterV2`（LayerNorm 归一化通道统计 + **可学习 per-expert prior bias 实现无辅助损失均衡**，DDP 安全）+ hybrid 专家（≥8 专家用 `FusedGhostExpert`，否则 `SharedInvertedExpertGroup`）+ channel shuffle + complexity gate + 轻量 DW 3×3 refine。

**v0.15 `GatedFusionMoE`**：把 concat+1×1 融合换成 **CrossPathGate 内容感知门控**（同时看 static/dynamic 输出），并加 `drop_prob=0.05` stochastic depth。

### 1.3 YAML 用法（v0_15/det/yolo-master-n.yaml）

```yaml
# backbone 三处插入 GatedFusionMoE
# 参数: [out_ch, num_experts, top_k, split_ratio]
- [-1, 1, GatedFusionMoE, [512, 4, 2, 0.5]]    # P3/P4
- [-1, 1, GatedFusionMoE, [512, 8, 2, 0.5]]    # P4/P5
- [-1, 1, GatedFusionMoE, [1024, 16, 2, 0.375]] # P5
```

训练参数：`moe_num_experts=8, moe_top_k=2, moe_balance_loss=0.01`。

### 1.4 损失与训练集成

- **`MoELoss`**（`loss.py`）：GShard balance loss `N·Σ(importance·usage)`（支持 DDP all_reduce 全局 usage、可微 soft/hard 两种模式）+ z-loss + 可选 entropy / diversity（专家输出余弦正交化）/ variance loss。系数由 `MoEDynamicScheduler` 和 **`MapSaturationScheduler`（mAP 驱动）** 动态调节。
- **统一路由协议**（`nn/modules/routing_protocol.py`，404 行）：项目历史上有三条 aux loss 通道（MoE registry、模块属性、wrapper 收集器），现已收敛为**单一弱引用状态通道**：`publish_aux_loss / routing_snapshot / export_capabilities`，即 `RoutedModule` 协议。MoE/MoA/MoT/MoLoRA/latent 五类 aux loss 统一由 `nn/mixture_loss.py` 的 `CompositeCriterion` 汇总，按类做 **EMA 归一化**（persistent buffer 随 checkpoint 保存）+ 全局 `aux_budget=3.0` 预算缩放 + NaN 隔离。
- **trainer 集成**（`engine/trainer.py`）：追加 `mixture_aux_loss` 日志项、`MoERouterError` 自动恢复、`moe_router_lr_scale=0.5`（路由器低学习率）、`MixtureRuntimeController` 负责温度退火与 mAP 饱和调度。

### 1.5 推理侧工具链

- **`MoEPruner`**（`moe/pruning.py`）：按 usage 阈值 0.15 或 `keep_top_m` 物理裁掉低利用率专家并重构路由器，**免重训提速 20–30%**（`scripts/moe_pruning_sweep.py` + 绘图）。
- **诊断**（`moe/analysis.py`）：`diagnose_model` / `ExpertUsageTracker` / `RoutingCollapseDetector`（路由塌缩检测）。
- **版本自测**：`verify_moe_v0_11.py`（正确性/DDP/EMA）、`compare_moe_v0_12_voc.py`、`compare_moe_v0_13_15_voc.py`、`check_moe_ssot.py`、`audit_moe_usage.py`。

### 1.6 实验基线

VisDrone 50e：EsMoE-N **mAP50-95 = 0.12023**，P50 13.5ms，7.85 GFLOPs，3.45M 参数。Model Zoo（4090 TRT）：N 2.68M/8.7G/0.427/640FPS；S 9.69M/29.1G/0.489/424FPS；M 34.88M/97.4G/0.530/244FPS；L/X 训练中。

**评价：MoE 是项目中最成熟的子系统**——有论文级结果、完整版本演进、剪枝/诊断/恢复全套工程闭环。

---

## 2. MoA（Mixture-of-Attention）—— 多尺度注意力软路由

代码：`ultralytics/nn/modules/moa/`（block / heads / router / wrappers）

### 2.1 架构

- **3 组注意力头**（`heads.py`）：
  - `_LocalAttnHead`：DW-biased 窗口 7，抓细粒度纹理；
  - `_RegionalAttnHead`：stride-2 pooled KV，抓中程上下文；
  - `_GlobalAttnHead`：线性注意力，抓场景语义（`rf_seed` 按 block_index 播种）。
- **路由**（`router.py`）：每个空间 token 经 `1×1conv+GN+SiLU+1×1conv` 轻量软路由（**末层零初始化**）得到 `[B,3,H,W]` 权重，三组输出加权混合 + 1×1 fusion + **LayerScale(0.1)** + FFN（mlp_ratio=2）。
- **与 MoE 路由的本质区别**：**不做 Top-K 稀疏，三组始终全部激活**（dense soft routing），复杂度 O(H·W·C/8)。CNN 原生 `[B,C,H,W]→[B,C,H,W]`，零序列维 reshape，Flash-Attention 兼容。
- **集成方式**：`C2fMoA`（直接替换 C2f/C3k2）与 `NeckMoAFusion`（跨尺度 FPN/PAN 融合，yolo26-master-moa-n.yaml 中 neck 四处融合层全部替换）。

### 2.2 损失与训练

`aux = coeff·(GShard balance + 0.1·z_loss + 0.01·entropy_deficit)`，`aux_loss_coeff=0.01`；温度 0.97/epoch 退火至 0.3（trainer 回调 `_anneal_moa_mot_temperature`）。

### 2.3 实验结论

- v0.8 MoA：参数 **+3.06%**（3.23M），CPU 256px 延迟仅 **+4.06%** —— 开销很小。
- VisDrone 50e：MoA-N mAP50-95 = 0.12019，**与 EsMoE 基线持平**（P50 20.5ms）。
- issue-53 对照（VisDrone 320×320/50ep/RTX5070Ti）：**MoA vs MoE 打平**（mAP50 均 0.133），MoA 训练耗时 +34% 但无 NaN，moa.py 测试覆盖率提升至 95%（62 tests）。

**评价：MoA 工程质量很好、开销低，但精度收益尚未被证明**——它的价值更多是作为"注意力多尺度先验"的即插即用件，而非 mAP 增长点。

---

## 3. MoT（Mixture-of-Transformers）—— 架构级路由

代码：`ultralytics/nn/modules/mot/`（block / experts / router / wrappers）

### 3.1 架构

- **3 个完整 Transformer 专家**（`experts.py`）：
  - `_LocalConvTransformerExpert`：DW3×3 QKV 预混合 + DW7×7 V 位置编码 + gated FFN，O(N²)，面向小目标纹理；
  - `_WindowTransformerExpert`：Swin 窗口/移位窗口注意力，O(N·win²)，面向规则结构；
  - `_DeformableTransformerExpert`：每 query/head K=n_points=4 采样点 `grid_sample`，O(N·K)，面向遮挡/不规则形状。
- **路由**：top_k=2，token 级 1×1 软路由 + Top-K；训练期 `exploration_eps=0.02` dense floor **防零初始化导致 expert 2 死亡**；eval 用 **sample-sparse dispatch**（每专家只跑被选中样本，`torch.nonzero` 数据依赖控制流，ONNX/trace 自动回退 dense）。温度注册为 persistent buffer 随 checkpoint 保存。
- **与 MoA 的区别**：在完整架构范式间 Top-K 稀疏路由，而非注意力头组全量软混合。

### 3.2 Scene-Aware Router（2026-07-17 设计，已实现于 `mot/router.py`，标记 experimental）

零初始化**场景残差分支**：从特征图提取 3 维可微场景统计（高频能量 / 空间异质性 / 多尺度方差）→ 小 MLP → 专家 logits 残差广播到所有 token，与 legacy router 输出严格初始一致（零风险插入）。`scene_consistency_loss` 用 KL 将 mean 路由概率对齐到目标分布（Local←高频、Window←多尺度方差、Deformable←异质性）。CLI：`mot_scene_aware_router / mot_scene_hidden_dim / mot_scene_consistency`；配置 `v0_10/det/yolo-master-mot-scene-n.yaml`。

### 3.3 实验结论（关键诚实证据）

- 成本：v0.8 MoT 参数 **+18.31%**；CPU 256px 延迟 MoT **+75%**、MoA+MoT **+148%**。
- VisDrone 50e（8×H200，issue54 流水线）：MoT-N 0.12011（**无增益**，P50 22ms）；MoA+MoT-N **0.12158（相对 +1.12%）**，但 P50 延迟 **+67.9%**（13.5→22.7ms）。
- 路由分析（`analyze_mot_routing.py` / `diagnose_mot_routing.py`）：专家激活呈**"层位置相关"**（早中层偏 Deformable、末端偏 Window），**尚不支持"密集/小目标场景 Deformable 激活上升"的假设**——这正是 scene-aware router 要解决的问题。
- 报告结论：MoT 工程闭环（解析/训练/反传/导出）已完成；精度收益需 COCO128 50e / COCO 300e 多 seed 验证；CPU 部署建议 top_k=1 或仅 P5 替换。

**评价：MoT 是概念最性感（架构级路由）但当前性价比最弱的模块**——延迟 +52%~75% 换 ≤1.1% 相对增益。scene-aware router 是正确的下一步，但需要路由行为的实证改善。

---

## 4. PEFT 体系 —— LoRA / MoLoRA / V-PEFT / Adapter Planner

这是项目中**体量最大、层次最多**的子系统，分四层：

```
┌─────────────────────────────────────────────────────────┐
│ 运行时工具层  agent/runtime/cli/{lora_tools, peft_compare,│
│              moe_tools}.py + examples/lora_examples/*.yaml│
├─────────────────────────────────────────────────────────┤
│ 规划层       ultralytics/vpeft/（研究原型，AAAI 2026 目标） │
│             ultralytics/utils/lora/planner.py（2686 行）  │
├─────────────────────────────────────────────────────────┤
│ 内核层       ultralytics/utils/lora/（7743 行：api/config/ │
│             fallback/training/io/backend/sensitivity）    │
│             ultralytics/nn/peft/molora/（MoLoRA 内核）    │
├─────────────────────────────────────────────────────────┤
│ 模型层       YOLO 主干 + MoE 专家（注入点）               │
└─────────────────────────────────────────────────────────┘
```

### 4.1 LoRA 内核（`ultralytics/utils/lora/`，7743 行）

- **纯配置激活、零架构手术**：覆盖 YOLOv3–v12、RT-DETR、YOLO-World、YOLO-Master 全家族。
- 官方数据：**~10% 可训练参数达全量微调 95–98% 性能**，训练提速 40–60%、显存降 70%；YOLO11x 适配器仅 14.1MB（vs 全量 114.6MB）。
- 文件分工：`api.py`（1349 行，训练/注入主接口）、`config.py`（1050 行）、`fallback.py`（1001 行，降级策略）、`training.py`（888 行）、`io.py` / `backend.py` / `sensitivity.py`。
- 下游矩阵测试（RTX 5070 Ti）：简单场景（Brain Tumor）rank=4 最优；极端密集场景（VisDrone）rank≥16，mAP50 +25.4%（显存代价 <0.1GB）。

### 4.2 MoLoRA（`ultralytics/nn/peft/molora/`）—— MoE 思想进 PEFT

Mixture-of-LoRA：多专家 LoRA + 可学习路由。

- **三种路由器**：`LinearRouter` / `SpatialRouter` / `HybridRouter`（`router.py`）。
- **核心件**：`MoLoRAExpert` / `MoLoRALayer`（`layer.py`）、`MoLoRALoss` + `compute_expert_usage`（负载均衡/多样性辅助损失）。
- **MoE 感知扩展**（`moe_aware.py`）：`PerExpertRankAllocator`（按专家分 rank）、`RouterCalibration`（路由器校准）、`build_moe_aware_layer`。
- **Routing-Aware Merging**：依据路由统计和专家使用频率选择性融合专家权重，避免破坏路由能力（`tests/test_molora_routing_aware_merge.py`、`test_molora_merge_semantics.py` 守护语义正确性）。
- 用法：`get_peft_molora_model(model, MoLoRAConfig)` + `mark_only_molora_as_trainable`；示例 `examples/molora/{basic_finetune, continual_learning}.py`（含持续学习 + 知识蒸馏抗灾难性遗忘路径）。
- 消融（COCO 500 图子集 MPS，E1–E3）：E1 frequency rank 分配 [14,7,7,4] 生效（可训练参数 1,270,760，32.05%）；E2 router calibration r4/r8 分别 +36k/+72k 参数；**E3 frequency 分配使专家负载 Gini 系数降 27.7%**（0.0679→0.0491）。

### 4.3 V-PEFT 编译器（`ultralytics/vpeft/`，约 4100 行）—— Adapter Planner 的核心

> 用户提到的 "planer" 即 **Adapter Planner**。注意：代码里实际有**两个规划器**——研究级的 `vpeft` 包和工程级的 `utils/lora/planner.py`（2686 行）。

V-PEFT 定位为 **"constraint-aware optimization solver framework"**（docstring 注明 *Target venue: AAAI 2026*），把"LoRA 插在哪、每层 rank 给多少"形式化为组合优化问题：

**四模块流水线**：模型 → 图构建 → 约束校验 → 策略/求解 → 放置方案（`PlacementDecision`）。

1. **`graph.py`（1042 行）—— 架构感知**
   - `ComputationGraph` / `ComputationGraphBuilder`：把模型层抽象为节点（`NodeAttributes`）、数据流为边；
   - **`GATv2ArchitectureEncoder`**：用 GATv2 图注意力网络编码架构——这是它区别于普通启发式规划器的核心，让策略网络"看懂"网络结构。

2. **`constraints.py`（950 行）—— 8 种约束类型**
   `BudgetConstraint`（预算）、`OperatorCompatibilityConstraint`（算子兼容）、`SemanticProtectionConstraint`（语义保护）、`DeploymentCompatibilityConstraint`（部署兼容）、`VariantModuleCompatibilityConstraint`、`MoEConsistencyConstraint`（**MoE 一致性——PEFT×MoE 集成的关键**）、`DivisibilityConstraint`（整除）、`CandidateTargetConstraint`。经 `ConstraintRegistry` 注册管理，支持硬/软约束。

3. **`policy.py`（1172 行）—— 放置策略 + rank 分配**
   - `PlacementPolicy`：神经逐模块放置，带硬/软约束；
   - rank 分配器三件套：`SoftRankAllocator`（连续松弛 + 高斯软投影）、`GreedyRankAllocator`（效用贪心）、**`RLRankAllocator`（PPO 序列分配）**；
   - `HybridTrainingProtocol`：**SL 预热 + RL 微调**的两阶段训练协议；
   - 语义效用先验 `SEMANTIC_UTILITY = {backbone:0.5, neck:0.8, head:1.0, attention:1.2}`——越靠近任务头/注意力，单位 rank 价值越高。

4. **`solver.py`（984 行）—— 三种求解器**
   - `AlternatingOptimizationSolver`（AO）：块坐标上升 + 贪鱼子例程；
   - `DifferentiableOptimizationSolver`（DCO）：端到端连续松弛 + 对偶上升（softplus 平滑罚）；
   - `MIPRelaxationSolver`（MIPR）：OR-Tools 精确 MIP + 迭代取整回退。
   - 边际效用函数 `f(r) = log2(r)/log2(r_max)`（rank 的对数收益递减假设）。

**验证资产**：`scripts/validate_planner.py`、`planner_mps_coco128_calibration.py`、`verify_planner_YOLO11s/12s Planner+training validation.py`、`tests/test_planner*.py`（106 passed）。

### 4.4 PEFT × MoE 集成

- `MoEConsistencyConstraint` 保证 adapter 放置不破坏 MoE 路由结构；
- Routing-Aware Merging 按路由统计选择性融合专家权重；
- `scripts/eval_moe_peft.py`、`scripts/ablation_moe_peft_e1_molora_rank.py`、`ablation_moe_peft_e2_router_calibration.py` 构成 MoE-aware PEFT 消融线。

### 4.5 已知问题（来自 2026-07-16 深度分析报告，诚实记录）

1. **MoLoRA 是最需收敛的子系统**：`.half()` 后 Conv2d forward 失败（fp16 bug）；训练保存未接 `molora_enabled`；merge 用均匀平均与训练路由**不等价**。
2. **V-PEFT 未接入主训练链路**，仍属研究原型（`__init__.py` 明确说 "dynamic MoE adapter intentionally not exported"）。
3. `yolo26-master-n.yaml` 因 SPPF 参数不兼容无法构建；`default.yaml` 有 4 个重复 key（lora_tinit 等）。

---

## 5. 横向对比与统一基础设施

### 5.1 三"M"模块对比

| 维度 | ES-MoE | MoA | MoT |
|:---|:---|:---|:---|
| 路由对象 | 卷积专家组（k3/5/7） | 3 注意力头组 | 3 完整 Transformer 范式 |
| 稀疏性 | Top-K（eval 真稀疏，省 FLOPs） | dense 软混合（不省） | 软 Top-K（eval sample-sparse） |
| 参数开销 | 基线（可剪枝 -20~30% 延迟） | +3.06% | +18.31% |
| 延迟开销 | 基线（P50 13.5ms） | +4%（CPU） | +75%（CPU） |
| VisDrone 增益 | 基线 0.12023 | 持平 | 单独无增益；+MoA +1.12%（延迟 +67.9%） |
| 成熟度 | ★★★★★（论文核心） | ★★★☆（工程好，增益未证） | ★★★（闭环成，性价比弱） |
| ONNX 导出 | eager sparse → dense 回退 | dense 原生稳定 | sample-sparse → dense 回退 |

### 5.2 统一路由运行时（项目真正的"隐形骨架"）

- `routing_protocol.py`：单一弱引用 aux loss 通道 + `RoutedModule` 协议（`publish_aux_loss` / `routing_snapshot` / `export_capabilities`），替代了历史上三条并行通道（registry/属性/wrapper 收集器）。
- `nn/mixture_registry.py`：把 MoE/MoA/MoT/MoLoRA 注册进 `parse_model`，YAML 可直接声明。
- `nn/mixture_loss.py`：`CompositeCriterion` 汇总五类 aux loss，EMA 归一化 + `aux_budget=3.0` + NaN 隔离。
- `MixtureRuntimeController`（trainer 内）：温度退火（MoA/MoT 0.97/epoch → 0.3）、mAP 饱和调度、`MoERouterError` 自动恢复、`moe_router_lr_scale=0.5`。
- `latent_mixture.py`（503 行）：稠密潜空间路由模块（另一条 routing 研究线）。
- `export_capabilities` 统一标记 MoA=dense soft、MoE/MoT=eager sparse（ONNX 均回退 dense），保证导出路径一致。

---

## 6. 实验证据汇总

| 实验 | 配置 | 结果 | 来源 |
|:---|:---|:---|:---|
| COCO 主结果 | Master-N | **42.4 AP @ 1.62ms**（+0.8 / -17.8% vs v13-N） | README |
| VisDrone 基线 | EsMoE-N 50e | mAP50-95 0.12023, P50 13.5ms | issue54 |
| VisDrone MoA | MoA-N 50e | 0.12019（持平），P50 20.5ms | issue54 |
| VisDrone MoT | MoT-N 50e | 0.12011（无增益），P50 22ms | issue54 |
| VisDrone MoA+MoT | 50e, 8×H200 | **0.12158（+1.12%）**，P50 **+67.9%** | 20260702 报告 |
| MoA vs MoE 对照 | 320²/50e/5070Ti | 打平（mAP50 0.133），MoA 训练 +34% | issue-53 |
| MoE 模块消融 | v0.6 | mAP50-95 0.61017（v0.7–v0.10 负收益弃用） | gated.py 记录 |
| MoE-aware PEFT E3 | COCO 子集 MPS | 专家负载 Gini **-27.7%** | ablation REPORT |
| LoRA 下游矩阵 | VisDrone rank16 | mAP50 **+25.4%**（vs rank4） | README |

---

## 7. 已知问题与风险清单

| 严重度 | 问题 | 位置 |
|:---|:---|:---|
| 🔴 高 | MoLoRA fp16 forward 失败；merge 均匀平均 ≠ 训练路由语义 | `nn/peft/molora` |
| 🔴 高 | MoT/MoA 延迟 +52%~148% 而精度增益 ≤1.1%，性价比未闭环 | `mot/`, `moa/` |
| 🟡 中 | V-PEFT 未接主训练链路（研究原型，dynamic MoE adapter 未导出） | `vpeft/` |
| 🟡 中 | MoT 路由"层位置相关"而非"场景相关"，scene-aware 假设待证 | `mot/router.py` |
| 🟡 中 | `yolo26-master-n.yaml` SPPF 参数不兼容无法构建 | cfg/models/26 |
| 🟢 低 | `default.yaml` 4 个重复 key；EsMoE-L/X 权重仍在训练 | cfg |

---

## 8. 分析与建议

1. **MoE 是基本盘，应继续吃透**：v0.15 `GatedFusionMoE` 的 CrossPathGate 刚落地，建议优先补 COCO 300e 多 seed 结果；剪枝工具（20–30% 提速）是部署侧最实际的卖点，值得做成一键 pipeline。
2. **MoT 的出路在 scene-aware 实证的**：当前路由行为与"场景自适应"假设不符，建议先用 `analyze_mot_routing.py` 在密集/稀疏场景分组统计上验证 scene residual 分支是否真的改变了路由分布，再决定是否扩大投入；CPU 部署直接 top_k=1 或仅 P5。
3. **MoA 定位应改为"低成本即插即用件"**：+3% 参数 +4% 延迟换持平精度，作为可选增强件合理，不要再期待它独立涨点。
4. **MoLoRA 需要一次"收敛冲刺"**：fp16 bug、merge 语义不等价、保存链路未接通是三个硬问题，建议按 tests 中 routing-aware merge 用例为验收标准逐个收口——这是 PEFT 故事能否讲圆的关键。
5. **V-PEFT（Adapter Planner）是最有研究品位的资产**：GATv2 架构编码 + PPO rank 分配 + MIP 精确求解的组合在 PEFT 自动化方向很完整（AAAI 2026 目标）。建议：(a) 先接 `eval_moe_peft.py` 做端到端"规划→训练→评测"闭环；(b) 与 `utils/lora/planner.py`（2686 行工程版）做职责切分——研究版探索策略空间，工程版服务生产配置；(c) `SEMANTIC_UTILITY` 先验（attention=1.2 > head=1.0 > neck=0.8 > backbone=0.5）本身值得一组消融验证。
6. **统一路由基础设施是隐藏的技术债务清偿亮点**：`routing_protocol.py` + `mixture_loss.py` 把五类 aux loss 收口，这套机制本身（EMA 归一化 + aux 预算 + NaN 隔离 + DDP 安全）值得写进论文附录或单独工程博客。

---

## 附录 A：关键文件索引

| 主题 | 路径 |
|:---|:---|
| MoE 核心 | `ultralytics/nn/modules/moe/{modules,gated,routers,experts,loss,pruning,analysis}.py` |
| MoA 核心 | `ultralytics/nn/modules/moa/{block,heads,router,wrappers}.py` |
| MoT 核心 | `ultralytics/nn/modules/mot/{block,experts,router,wrappers}.py` |
| 统一路由 | `ultralytics/nn/modules/routing_protocol.py`、`ultralytics/nn/mixture_loss.py`、`mixture_registry.py` |
| MoLoRA | `ultralytics/nn/peft/molora/{config,layer,router,loss,model,moe_aware,utils}.py` |
| LoRA 内核 | `ultralytics/utils/lora/{api,config,planner,fallback,training,io,backend,sensitivity}.py` |
| V-PEFT | `ultralytics/vpeft/{graph,constraints,policy,solver}.py` |
| 模型配置 | `ultralytics/cfg/models/master/{v0..v0_15,exp}/`、`cfg/models/26/` |
| 实验报告 | `reports/yolo_master_deep_analysis_20260716.md`、`reports/yolo_master_mot_moa_visdrone_report_20260702.md`、`reports/issue-53-*.md`、`docs/mot_integration_experiment_report_2026-06-25.md` |
| 设计文档 | `docs/plans/2026-07-17-mot-scene-aware-router.md`、`2026-05-15-lora-package-spec.md`、`moe_stable_version_analysis.md` 等 14 篇 |

## 附录 B：一句话总结

> YOLO-Master 用 MoE 解决了"算力按需分配"并已拿到论文级结果；MoA/MoT 是两条"路由粒度升级"的探索线，工程闭环已成但性价比未证；PEFT/MoLoRA/V-PEFT 构成从"低成本微调"到"自动化适配器规划"的完整研究栈——其中 MoLoRA 的工程收敛和 V-PEFT 的主链路接入是下一阶段最值得投入的两件事。
