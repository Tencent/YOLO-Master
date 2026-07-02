# MoLoRA 集成计划

日期: 2026-07-01
目标: 将 MoLoRA (Mixture-of-LoRA) 集成到 YOLO-Master 项目

## 状态概览

| 阶段 | 状态 | 说明 |
|------|------|------|
| Stage 1: 核心模块 | ✅ 完成 | 7 个文件，~2062 行 |
| Stage 2: 训练脚本集成 | ✅ 完成 | trainer.py, tasks.py, default.yaml |
| Stage 3: 动态路由增强 | ✅ 完成 | capacity_factor, expert_dropout, top_k warmup |
| Stage 4: 持续学习工作流 | ✅ 完成 | freeze_experts, domain pre-alloc, replay |
| Stage 5: 性能优化 | ✅ 完成 | batched expert indexing, frozen expert no-grad |
| Stage 6: 文档与示例 | ✅ 完成 | 文档 + 2 个示例脚本 |
| 测试 | ✅ 71/71 通过 | 含 13 个新测试 |

---

## Stage 1: 核心模块实现 ✅

创建 `ultralytics/nn/peft/molora/` 目录及以下文件：

| 文件 | 行数 | 职责 |
|------|------|------|
| `config.py` | 178 | MoLoRAConfig + MoLoRAConfigBuilder + 4 种预设 |
| `router.py` | 137 | LinearRouter / SpatialRouter / HybridRouter + 工厂 |
| `layer.py` | 455 | MoLoRAExpert + MoLoRALayer (含动态路由/域/冻结) |
| `loss.py` | 154 | MoLoRALoss: GShard balance + z-loss + diversity |
| `model.py` | 268 | get_peft_molora_model + MoLoRAModel (含 replay) |
| `utils.py` | 232 | 参数统计、merge/unmerge、初始化、域分配 |
| `__init__.py` | 53 | 公共 API 导出 |

**设计约束**: 继承 LoRAConfig、复用 auto_detect_targets、CNN-Native 整图路由、支持 merge/unmerge、共享 MOE_LOSS_REGISTRY。

---

## Stage 2: 训练脚本集成 ✅

| 文件 | 修改点 |
|------|--------|
| `ultralytics/nn/tasks.py` | `_has_moe_aux_registry_module` 识别 MoLoRA 模块 |
| `ultralytics/engine/trainer.py` | MoLoRA 初始化 + 参数注入（balance/z/diversity loss） |
| `ultralytics/cfg/default.yaml` | 新增 10 项 `molora_*` 参数 |

---

## Stage 3: 动态路由增强 ✅

- `capacity_factor`: 限制每专家处理的 token 数，防止负载不均
- `expert_dropout`: 训练时以概率 p 禁用专家，提升鲁棒性
- `top_k_warmup` + `warmup_steps`: 从 K=1 逐渐增加到目标 K，稳定训练初期

---

## Stage 4: 持续学习工作流 ✅

- `domain_experts` + `set_domain/clear_domain`: 域预分配，推理时只使用子集专家
- `freeze_experts` / `unfreeze_experts`: 专家冻结，防止旧域遗忘
- `save_expert_replay_buffer` / `load_expert_replay_buffer`: 专家回放，保存/恢复专家权重

---

## Stage 5: 性能优化 ✅

- `_compute_sparse_experts`: 按专家分组 gather，避免 per-sample 循环
- Frozen expert no-grad: 冻结专家在推理时跳过梯度计算

---

## Stage 6: 文档与示例 ✅

- `docs/molora_guide.md`: 完整使用指南（命令行、程序化、merge、持续学习、动态路由）
- `examples/molora/basic_finetune.py`: COCO 单域微调示例
- `examples/molora/continual_learning.py`: 白天→黑夜→雾天持续学习示例

---

## 测试覆盖

`tests/test_molora.py` — 71 测试全部通过：

| 测试类 | 测试数 | 覆盖内容 |
|--------|--------|----------|
| TestMoLoRAConfig | 10 | 默认值、from_lora_config、4 种预设、参数验证 |
| TestRouters | 8 | 三种路由前向形状、工厂、参数初始化 |
| TestMoLoRAExpert | 7 | Conv/Linear 前向、delta_weight、rsLoRA、初始化、dropout |
| TestMoLoRALayer | 13 | 前向、冻结、可训练性、top-k 路由、merge/unmerge、stats、backward、eval |
| TestMoLoRALoss | 7 | balance_loss、z_loss、diversity_loss、compute_expert_usage、loss container、零系数、数值稳定性 |
| TestMoLoRAModelWrapper | 5 | get_peft_molora_model、参数冻结、aux_loss、merge/unmerge、save/load |
| TestUtils | 9 | 形状工具、域分配、参数统计、初始化、冻结 |
| TestRegistryIntegration | 3 | MOE_LOSS_REGISTRY 读写、清除、eval |
| TestDynamicRouting | 5 | warmup、expert_dropout、capacity_factor、domain_prealloc、domain_clear |
| TestContinualLearning | 4 | freeze_experts、unfreeze_experts、unfreeze_all、expert_replay |

---

## 验证命令

```bash
# 完整测试
python -m pytest tests/test_molora.py -v --tb=short

# 现有 MoE 兼容性
python -m pytest tests/test_moe.py -v --tb=short

# 导入验证
python -c "from ultralytics.nn.peft.molora import *; print('All imports OK')"
```

---

## 文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `ultralytics/nn/peft/molora/__init__.py` | 53 | 模块入口 |
| `ultralytics/nn/peft/molora/config.py` | 178 | 配置与预设 |
| `ultralytics/nn/peft/molora/router.py` | 137 | 三种路由器 |
| `ultralytics/nn/peft/molora/layer.py` | 455 | 专家+层+动态路由+域+冻结 |
| `ultralytics/nn/peft/molora/loss.py` | 154 | 辅助损失函数 |
| `ultralytics/nn/peft/molora/model.py` | 268 | PEFT 包装器+回放 |
| `ultralytics/nn/peft/molora/utils.py` | 232 | 工具函数 |
| `tests/test_molora.py` | ~660 | 71 个单元测试 |
| `ultralytics/engine/trainer.py` | ~+20 | MoLoRA 初始化+注入 |
| `ultralytics/nn/tasks.py` | ~+2 | 识别 MoLoRA 模块 |
| `ultralytics/cfg/default.yaml` | ~+12 | MoLoRA 参数 |
| `docs/molora_guide.md` | ~180 | 使用指南 |
| `examples/molora/basic_finetune.py` | ~55 | 微调示例 |
| `examples/molora/continual_learning.py` | ~120 | 持续学习示例 |
| **总计** | **~2500+** | **核心+测试+文档+示例** |
