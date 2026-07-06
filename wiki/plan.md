# YOLO-Master Wiki 编写计划

## 目标
将 MoE 与 PEFT 相关的文档和代码内容整理为结构化的 wiki 文档。

## wiki 目录结构

```
wiki/
├── Home.md                          # 总览首页
├── _Sidebar.md                      # 侧边栏导航
├── MoE/
│   ├── Home.md                      # MoE 专题首页
│   ├── Core_Modules.md              # 核心 MoE 模块详解
│   ├── Routers_and_Experts.md       # 路由与专家模块
│   ├── Training_Loss_Pruning.md     # 训练、损失、剪枝与调度
│   ├── Diagnostics_and_Analysis.md  # 诊断与分析工具
│   ├── Mixture_of_Attention.md      # MOA 模块
│   └── Version_Evolution.md         # 版本演进与稳定版本
└── PEFT/
    ├── Home.md                      # PEFT 专题首页
    ├── LoRA_Core.md                 # LoRA 核心实现
    ├── MoLoRA.md                    # Mixture-of-LoRA
    ├── Training_and_IO.md           # 训练策略与 IO
    └── Planner_and_AutoConfig.md    # PEFT Planner
```

## 各页面内容来源

### MoE 系列
1. **Core_Modules.md**: `ultralytics/nn/modules/moe/modules.py` (~4384 行，40+ 类)
   - UltraOptimizedMoE, AdaptiveCapacityMoE, ES_MOE, OptimizedMOE, OptimizedMOEImproved
   - AdaptiveGateMoE, HyperSplitMoE, HyperFusedMoE, HyperUltimateMoE, UltimateOptimizedMoE
   - FusedAdaptiveGateMoE, HybridAdaptiveGateMoE 系列 (v0.4~v0.10)
   - MultiHeadRouterMoE, DiversifiedExpertMoE, GatedFusionMoE 等
2. **Routers_and_Experts.md**: `routers.py` + `experts.py`
   - UltraEfficientRouter, BaseRouter, EfficientSpatialRouter, AdaptiveRoutingLayer, LocalRoutingLayer, AdvancedRoutingLayer, DynamicRoutingLayer
   - OptimizedSimpleExpert, FusedGhostExpert, SimpleExpert, GhostExpert, InvertedResidualExpert, SharedInvertedExpertGroup
3. **Training_Loss_Pruning.md**: `loss.py` + `pruning.py` + `scheduler.py` + `utils.py`
   - MoELoss, MoEPruner, MoEDynamicScheduler, AdaptiveBalanceController
   - BatchedExpertComputation, FlopsUtils
4. **Diagnostics_and_Analysis.md**: `analysis.py` + `diagnostics.py` + `history.py` + `docs/moe_pruning_dynamic_schedule.md`
   - ExpertUsageTracker, RoutingCollapseDetector, MoELayerDiagnostic, MoEDiagnosticsRecorder
5. **Mixture_of_Attention.md**: `modules/moa/moa.py` (~716 行)
6. **Version_Evolution.md**: `docs/plans/moe_stable_version_analysis.md` + 现有 wiki/MoE_Modules_Explanation.md

### PEFT 系列
1. **LoRA_Core.md**: `utils/lora/api.py` + `config.py` + `fallback.py`
   - LoRAConfig, LoRAConfigBuilder, apply_lora, LoRADetectionModel, PeftProxy
   - FewShotLoRAConv, ManualLoRAConv
2. **MoLoRA.md**: `nn/peft/molora/` 全部文件 + `docs/molora_guide.md`
   - MoLoRAConfig, MoLoRAExpert, MoLoRALayer, MoLoRAModel, MoLoRALoss
   - LinearRouter, SpatialRouter, HybridRouter
   - Merge/Unmerge, 持续学习, 域分配
3. **Training_and_IO.md**: `utils/lora/training.py` + `io.py`
   - LoraTrainingStrategy, save/load/merge adapters
4. **Planner_and_AutoConfig.md**: `utils/lora/planner.py`
   - PEFTPlanner, ArchitectureFingerprint, LOVOValidator, PlacementDecision

## 编写要求
- 技术准确，基于代码实际实现
- 包含类/函数签名、参数说明、使用示例
- 中文为主，关键术语保留英文
- 结构清晰，使用 Markdown 标题层级
- 适当包含配置示例和代码片段
