# MoLoRA (Mixture-of-LoRA) 使用指南

MoLoRA 是 YOLO-Master 的 PEFT 扩展，将标准 LoRA 升级为多专家稀疏适配架构。

## 1. 快速开始

### 1.1 命令行训练

在训练配置 YAML 或命令行中添加 MoLoRA 参数：

```bash
yolo detect train model=yolov8n.pt data=coco128.yaml \
  molora_num_experts=4 molora_top_k=2 molora_r=8 molora_alpha=16 \
  epochs=100
```

关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `molora_num_experts` | 0 | 专家数量（0=禁用） |
| `molora_top_k` | 2 | 每步激活的专家数 |
| `molora_router_type` | linear | 路由类型: linear / spatial / hybrid |
| `molora_r` | 8 | LoRA 秩 |
| `molora_alpha` | 16 | LoRA alpha |
| `molora_balance_loss` | 0.01 | balance loss 系数 |
| `molora_router_z_loss` | 0.001 | z-loss 系数 |
| `molora_diversity_loss` | 0.0 | diversity loss 系数 |
| `molora_expert_init` | default | 专家初始化: default / orthogonal / gaussian |
| `molora_use_rslora` | true | 使用 rsLoRA scaling |
| `molora_share_moe_registry` | true | 共享 MOE aux loss registry |
| `molora_capacity_factor` | 1.0 | 容量限制因子（1.0=无限制） |
| `molora_expert_dropout` | 0.0 | 专家 dropout 概率 |
| `molora_top_k_warmup` | null | 启用 top_k warmup（>0 表示启用） |
| `molora_warmup_steps` | 0 | top_k warmup 从 1 到目标值的总步数 |
| `molora_router_hidden_dim` | null | 路由隐藏层维度（null=自动=C/4） |

### 1.2 程序化使用

```python
from ultralytics import YOLO
from ultralytics.nn.peft.molora import MoLoRAConfig, get_peft_molora_model

# 加载基础模型
model = YOLO("yolov8n.pt").model

# 创建 MoLoRA 配置
cfg = MoLoRAConfig(
    r=8, alpha=16,
    num_experts=4, top_k=2,
    router_type="linear",
    balance_loss_coef=0.01,
    use_rslora=True,
)

# 包装模型
model = get_peft_molora_model(model, cfg)

# 训练
yolo = YOLO()
yolo.model = model
yolo.train(data="coco128.yaml", epochs=100)
```

## 2. 从 LoRA 升级

```python
from ultralytics.utils.lora.config import LoRAConfig
from ultralytics.nn.peft.molora import MoLoRAConfig

lora_cfg = LoRAConfig(r=8, alpha=16)
molora_cfg = MoLoRAConfig.from_lora_config(lora_cfg, num_experts=4, top_k=2)
```

## 3. 预设配置

```python
from ultralytics.nn.peft.molora import get_molora_preset

preset = get_molora_preset("preset_standard")  # 或 preset_small / preset_large / preset_continual
cfg = MoLoRAConfig(**preset)
```

## 4. 推理优化：Merge / Unmerge

```python
from ultralytics.nn.peft.molora import MoLoRAModel

wrapper = MoLoRAModel(model, cfg)

# 训练后合并为单一权重，零推理开销
wrapper.merge()
# 现在 forward 等价于标准 Conv/Linear，可用于 ONNX 导出

# 如需恢复 MoLoRA 行为
wrapper.unmerge()
```

## 5. 持续学习：多域顺序训练

### 5.1 域分配

```python
from ultralytics.nn.peft.molora import allocate_domain_experts, MoLoRAModel

# 将 8 个专家均分给 4 个域
alloc = allocate_domain_experts(8, ["day", "night", "fog", "rain"])
# {"day": [0,1], "night": [2,3], "fog": [4,5], "rain": [6,7]}

cfg = MoLoRAConfig(
    num_experts=8, top_k=2,
    domain_experts=alloc
)
wrapper = MoLoRAModel(model, cfg)

# 训练 day 域时，只使用专家 0,1
wrapper.set_domain("day")
wrapper.model.train()
# ... train on day dataset ...

# 切换到 night 域
wrapper.set_domain("night")
# ... train on night dataset ...
```

### 5.2 专家冻结

```python
# 完成 day 域训练后，冻结 day 专家防止遗忘
wrapper.freeze_experts([0, 1])

# 继续训练 night 域（只更新专家 2,3）
wrapper.set_domain("night")
# ... train ...
```

### 5.3 专家回放

```python
# 保存旧域专家权重
buffer = wrapper.save_expert_replay_buffer("day")

# 新域训练后，回放旧域专家防止遗忘
wrapper.load_expert_replay_buffer(buffer, domain="day")
```

## 6. 动态路由

### 6.1 top-k Warmup

`top_k_warmup` 是启用开关（>0 表示启用），`warmup_steps` 控制实际步数。

```python
# 前 1000 步从 top_k=1 逐渐增加到 2，稳定训练初期
cfg = MoLoRAConfig(
    num_experts=4, top_k=2,
    top_k_warmup=1,      # 启用 warmup（>0 即启用）
    warmup_steps=1000,  # 1000 步内从 K=1 逐渐增加到 K=2
)
```

### 6.2 Expert Dropout

```python
# 训练时以 10% 概率禁用专家，提升鲁棒性
cfg = MoLoRAConfig(
    num_experts=4, top_k=2,
    expert_dropout=0.1,
)
```

## 7. 诊断与监控

```python
# 获取参数统计
stats = wrapper.param_stats()
print(f"Total: {stats['total']}, Trainable: {stats['trainable']}, MoLoRA: {stats['molora']}")

# 查看每层路由统计
for name, m in wrapper.model.named_modules():
    if hasattr(m, "_last_routing_stats") and m._last_routing_stats:
        stats = m._last_routing_stats
        print(f"{name}: expert_usage={stats['expert_usage'].tolist()}")
```

## 8. 与 MoE 协同

MoLoRA 与 MoE 层共享 `MOE_LOSS_REGISTRY`，当 `share_moe_registry=True` 时：
- MoLoRA 的 balance loss 自动被 `_collect_moe_aux_loss` 收集
- 无需修改 trainer 的 loss 计算逻辑

```yaml
# 同时使用 MoE 和 MoLoRA
moe: true
moe_num_experts: 4
molora_num_experts: 4
```

## 9. 预期性能

| 场景 | mAP50 增益 | 参数量增加 |
|------|-----------|-----------|
| 单域微调（vs LoRA） | +0.5~+1.2 | ~2× LoRA |
| 多域持续学习 | +2.0~+5.0 | 相同 |
| 与 MoE 协同 | +0.5~+1.0（叠加） | 与 MoE 正交 |

域间差异越大（白天/黑夜、晴天/雾天），MoLoRA 收益越显著。
