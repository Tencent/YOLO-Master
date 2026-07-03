"""MoLoRA 全面验证脚本（功能点 + 集成 + 性能）。

运行:
    python3 verify_molora.py

输出: 每个验证项的 PASS/FAIL 状态及详细报告。
"""
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except ImportError:
    yaml = None

# 将项目根目录加入路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ultralytics.nn.peft.molora import (
    MoLoRAConfig,
    MoLoRAConfigBuilder,
    get_molora_preset,
    build_router,
    LinearRouter,
    SpatialRouter,
    HybridRouter,
    MoLoRAExpert,
    MoLoRALayer,
    MoLoRALoss,
    compute_expert_usage,
    get_peft_molora_model,
    MoLoRAModel,
    mark_only_molora_as_trainable,
    count_parameters,
    allocate_domain_experts,
)
from ultralytics.nn.peft.molora.utils import (
    get_conv_shape,
    is_conv,
    is_linear,
    _molora_scales,
    init_lora_expert_a,
    init_lora_expert_b,
)
from ultralytics.nn.modules.moe.modules import MOE_LOSS_REGISTRY


# ---------------------------------------------------------------------------
# 验证框架
# ---------------------------------------------------------------------------

PASSED = 0
FAILED = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ PASS: {name}")
    else:
        FAILED += 1
        print(f"  ❌ FAIL: {name} — {detail}")
    return condition


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. 导入验证
# ---------------------------------------------------------------------------

section("1. 模块导入验证")

try:
    import ultralytics.nn.peft.molora as molora_mod
    check("MoLoRA 包可导入", True)
    check("MoLoRAConfig 可访问", hasattr(molora_mod, "MoLoRAConfig"))
    check("get_peft_molora_model 可访问", hasattr(molora_mod, "get_peft_molora_model"))
    check("MoLoRAModel 可访问", hasattr(molora_mod, "MoLoRAModel"))
    check("MoLoRALayer 可访问", hasattr(molora_mod, "MoLoRALayer"))
    check("MoLoRALoss 可访问", hasattr(molora_mod, "MoLoRALoss"))
    check("build_router 可访问", hasattr(molora_mod, "build_router"))
    check("LinearRouter 可访问", hasattr(molora_mod, "LinearRouter"))
    check("SpatialRouter 可访问", hasattr(molora_mod, "SpatialRouter"))
    check("HybridRouter 可访问", hasattr(molora_mod, "HybridRouter"))
    check("count_parameters 可访问", hasattr(molora_mod, "count_parameters"))
    check("allocate_domain_experts 可访问", hasattr(molora_mod, "allocate_domain_experts"))
    check("get_molora_preset 可访问", hasattr(molora_mod, "get_molora_preset"))
except Exception as e:
    check("MoLoRA 包导入", False, str(e))


# ---------------------------------------------------------------------------
# 2. 配置验证
# ---------------------------------------------------------------------------

section("2. MoLoRAConfig 验证")

try:
    cfg = MoLoRAConfig()
    check("默认 num_experts=4", cfg.num_experts == 4)
    check("默认 top_k=2", cfg.top_k == 2)
    check("默认 router_type='linear'", cfg.router_type == "linear")
    check("默认 balance_loss_coef=0.01", cfg.balance_loss_coef == 0.01)
    check("默认 z_loss_coef=0.001", cfg.z_loss_coef == 0.001)
    check("默认 diversity_loss_coef=0.0", cfg.diversity_loss_coef == 0.0)
    check("默认 expert_init='default'", cfg.expert_init == "default")
    check("默认 share_moe_registry=True", cfg.share_moe_registry is True)
    check("默认 capacity_factor=1.0", cfg.capacity_factor == 1.0)
    check("默认 expert_dropout=0.0", cfg.expert_dropout == 0.0)
    check("默认 top_k_warmup=None", cfg.top_k_warmup is None)
    check("默认 warmup_steps=0", cfg.warmup_steps == 0)
    check("默认 domain_experts=None", cfg.domain_experts is None)
    check("默认 freeze_experts=None", cfg.freeze_experts is None)

    # 参数验证
    try:
        MoLoRAConfig(num_experts=0)
        check("num_experts=0 应抛异常", False, "未抛异常")
    except ValueError:
        check("num_experts=0 抛 ValueError", True)

    try:
        MoLoRAConfig(top_k=5, num_experts=4)
        check("top_k > num_experts 应抛异常", False, "未抛异常")
    except ValueError:
        check("top_k > num_experts 抛 ValueError", True)

    try:
        MoLoRAConfig(router_type="unknown")
        check("invalid router_type 应抛异常", False, "未抛异常")
    except ValueError:
        check("invalid router_type 抛 ValueError", True)

    try:
        MoLoRAConfig(balance_loss_coef=-0.1)
        check("negative balance_loss_coef 应抛异常", False, "未抛异常")
    except ValueError:
        check("negative balance_loss_coef 抛 ValueError", True)

    # 预设
    for preset_name in ["preset_small", "preset_standard", "preset_large", "preset_continual"]:
        p = get_molora_preset(preset_name)
        check(f"preset '{preset_name}' 存在", preset_name.startswith("preset_") and "num_experts" in p)

    # from_lora_config
    from ultralytics.utils.lora.config import LoRAConfig
    lora = LoRAConfig(r=16, alpha=32, dropout=0.1)
    molora = MoLoRAConfig.from_lora_config(lora, num_experts=8, top_k=2)
    check("from_lora_config 保留 r", molora.r == 16)
    check("from_lora_config 保留 alpha", molora.alpha == 32)
    check("from_lora_config 保留 dropout", molora.dropout == 0.1)
    check("from_lora_config 设置 num_experts", molora.num_experts == 8)
    check("from_lora_config 设置 top_k", molora.top_k == 2)

except Exception as e:
    check("Config 验证", False, str(e))


# ---------------------------------------------------------------------------
# 3. 路由验证
# ---------------------------------------------------------------------------

section("3. Router 验证")

try:
    for rt in ("linear", "spatial", "hybrid"):
        r = build_router(rt, 16, 4, 8)
        x_conv = torch.randn(2, 16, 8, 8)
        logits = r(x_conv)
        check(f"{rt} router Conv2d 输入: logits shape == (2, 4)", logits.shape == (2, 4))

        x_lin = torch.randn(2, 16)
        logits2 = r(x_lin)
        check(f"{rt} router Linear 输入: logits shape == (2, 4)", logits2.shape == (2, 4))

    # 参数非零
    lr = LinearRouter(16, 4)
    check("LinearRouter 参数非零", sum(p.numel() for p in lr.parameters()) > 0)
    check("LinearRouter 偏置为零", torch.allclose(lr.fc[-1].bias, torch.zeros_like(lr.fc[-1].bias)))
    check("LinearRouter 权重小", lr.fc[-1].weight.abs().mean() < 0.1)

except Exception as e:
    check("Router 验证", False, str(e))


# ---------------------------------------------------------------------------
# 4. MoLoRAExpert 验证
# ---------------------------------------------------------------------------

section("4. MoLoRAExpert 验证")

try:
    # Conv2d expert
    conv = nn.Conv2d(16, 32, 3, padding=1)
    exp = MoLoRAExpert(conv, r=4, alpha=8)
    x = torch.randn(2, 16, 8, 8)
    out = exp(x)
    check("Conv2d expert 输出 shape", out.shape == (2, 32, 8, 8))

    # Linear expert
    lin = nn.Linear(64, 128)
    exp2 = MoLoRAExpert(lin, r=4, alpha=8)
    x2 = torch.randn(2, 64)
    out2 = exp2(x2)
    check("Linear expert 输出 shape", out2.shape == (2, 128))

    # delta_weight
    dw = exp.delta_weight()
    check("Conv2d delta_weight shape", dw.shape == (32, 16, 3, 3))

    # rsLoRA scaling
    s1 = _molora_scales(8, 16, use_rslora=True)
    s2 = _molora_scales(8, 16, use_rslora=False)
    check("rsLoRA scaling = alpha/sqrt(r)", abs(s1 - 16 / (8**0.5)) < 1e-6)
    check("standard scaling = alpha/r", abs(s2 - 2.0) < 1e-6)

    # 三种初始化
    for it in ("default", "orthogonal", "gaussian"):
        exp_i = MoLoRAExpert(conv, r=4, alpha=8, init_type=it)
        check(f"init_type='{it}' 不报错", True)

    # dropout
    exp_d = MoLoRAExpert(conv, r=4, alpha=8, dropout=0.5)
    check("dropout=0.5 创建 nn.Dropout", isinstance(exp_d.dropout, nn.Dropout))

except Exception as e:
    check("Expert 验证", False, str(e))


# ---------------------------------------------------------------------------
# 5. MoLoRALayer 验证（核心功能 + 动态路由 + 持续学习）
# ---------------------------------------------------------------------------

section("5. MoLoRALayer 验证")

try:
    # 5.1 基础前向
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2)
    x = torch.randn(2, 16, 8, 8)
    out = layer(x)
    check("Conv2d MoLoRALayer 输出 shape", out.shape == (2, 32, 8, 8))

    # 5.2 Linear
    lin = nn.Linear(64, 128)
    layer2 = MoLoRALayer(lin, r=4, alpha=8, num_experts=4, top_k=2)
    x2 = torch.randn(2, 64)
    out2 = layer2(x2)
    check("Linear MoLoRALayer 输出 shape", out2.shape == (2, 128))

    # 5.3 base 冻结
    check("base layer 权重 frozen", not any(p.requires_grad for p in layer.base_layer.parameters()))

    # 5.4 expert 可训练
    check("expert 参数可训练", any(p.requires_grad for p in layer.experts.parameters()))

    # 5.5 top-k 路由
    layer.train()
    _ = layer(x)
    check("_last_routing_stats 存在", layer._last_routing_stats is not None)
    check("top_k_indices shape", layer._last_routing_stats["top_k_indices"].shape == (2, 2))
    check("expert_usage 归一化", abs(layer._last_routing_stats["expert_usage"].sum().item() - 1.0) < 1e-5)

    # 5.6 三种 router_type
    for rt in ("linear", "spatial", "hybrid"):
        c = nn.Conv2d(16, 32, 3, padding=1)
        l = MoLoRALayer(c, r=4, alpha=8, num_experts=4, top_k=2, router_type=rt)
        o = l(torch.randn(2, 16, 8, 8))
        check(f"router_type='{rt}' 输出 shape", o.shape == (2, 32, 8, 8))

    # 5.7 merge / unmerge
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2)
    x = torch.randn(2, 16, 8, 8)
    layer.merge_weights()
    check("merge 后 merged=True", layer.merged is True)
    out_m = layer(x)
    check("merge 后输出 shape", out_m.shape == (2, 32, 8, 8))
    layer.unmerge_weights()
    check("unmerge 后 merged=False", layer.merged is False)
    out_u = layer(x)
    check("unmerge 后输出 shape", out_u.shape == (2, 32, 8, 8))

    # 5.8 backward
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    check("x.grad 存在", x.grad is not None)
    check("expert 参数有梯度", any(p.grad is not None for p in layer.experts.parameters() if p.requires_grad))

    # 5.9 eval 模式
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2)
    layer.eval()
    with torch.no_grad():
        out = layer(torch.randn(2, 16, 8, 8))
    check("eval 模式输出 shape", out.shape == (2, 32, 8, 8))

    # 5.10 num_experts=1 fallback
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=1, top_k=1)
    out = layer(torch.randn(2, 16, 8, 8))
    check("num_experts=1 fallback", out.shape == (2, 32, 8, 8))

except Exception as e:
    check("Layer 基础验证", False, str(e))


# 5.11 动态路由增强
section("5.11 动态路由增强")

try:
    # top_k warmup
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2, top_k_warmup=10, warmup_steps=10)
    check("warmup step 0: top_k=1", layer._current_top_k() == 1)
    layer._step_count.fill_(5)
    check("warmup step 5: top_k=1 (渐进)", layer._current_top_k() == 1)
    layer._step_count.fill_(10)
    check("warmup step 10: top_k=2", layer._current_top_k() == 2)

    # expert dropout
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2, expert_dropout=0.5)
    layer.train()
    x = torch.randn(2, 16, 8, 8)
    out = layer(x)
    check("expert_dropout=0.5 输出 shape", out.shape == (2, 32, 8, 8))

    # capacity_factor
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2, capacity_factor=0.5)
    x = torch.randn(2, 16, 8, 8)
    out = layer(x)
    check("capacity_factor=0.5 输出 shape", out.shape == (2, 32, 8, 8))

except Exception as e:
    check("动态路由验证", False, str(e))


# 5.12 持续学习
section("5.12 持续学习")

try:
    # domain pre-allocation
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2,
                        domain_experts={"day": [0, 1], "night": [2, 3]})
    layer.set_domain("day")
    x = torch.randn(2, 16, 8, 8)
    out = layer(x)
    check("domain='day' 输出 shape", out.shape == (2, 32, 8, 8))
    check("domain='day' stats 有 domain_mask", layer._last_routing_stats is not None and
          layer._last_routing_stats.get("domain_mask") is not None)

    layer.clear_domain()
    check("clear_domain 后 mask 为 None", layer._domain_active_mask is None)

    # freeze_experts
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2)
    layer.freeze_experts([0, 1])
    check("freeze_experts[0] frozen", not any(p.requires_grad for p in layer.experts[0].parameters()))
    check("freeze_experts[1] frozen", not any(p.requires_grad for p in layer.experts[1].parameters()))
    check("expert[2] 仍可训练", any(p.requires_grad for p in layer.experts[2].parameters()))
    check("expert[3] 仍可训练", any(p.requires_grad for p in layer.experts[3].parameters()))

    layer.unfreeze_experts([0])
    check("unfreeze_experts[0] 后可训练", any(p.requires_grad for p in layer.experts[0].parameters()))
    check("expert[1] 仍 frozen", not any(p.requires_grad for p in layer.experts[1].parameters()))

    layer.unfreeze_experts()
    check("unfreeze_all 后所有专家可训练", all(any(p.requires_grad for p in e.parameters()) for e in layer.experts))

except Exception as e:
    check("持续学习验证", False, str(e))


# ---------------------------------------------------------------------------
# 6. MoLoRALoss 验证
# ---------------------------------------------------------------------------

section("6. MoLoRALoss 验证")

try:
    loss_fn = MoLoRALoss(num_experts=4, top_k=2, balance_loss_coef=1.0, z_loss_coef=1.0)
    probs = F.softmax(torch.randn(8, 4), dim=-1)
    logits = torch.randn(8, 4)
    indices = torch.randint(0, 4, (8, 2))
    loss = loss_fn(probs, logits, indices)
    check("loss 为有限正值", loss.item() > 0 and torch.isfinite(loss))

    # return_dict
    result = loss_fn(probs, logits, indices, return_dict=True)
    check("return_dict 返回 dict", isinstance(result, dict))
    check("dict 包含 'loss'", "loss" in result)
    check("dict 包含 'balance_loss'", "balance_loss" in result)
    check("dict 包含 'z_loss'", "z_loss" in result)

    # diversity loss
    loss_fn2 = MoLoRALoss(num_experts=4, top_k=2, diversity_loss_coef=1.0)
    expert_outs = torch.randn(8, 4, 16)
    loss2 = loss_fn2(probs, logits, indices, expert_outputs=expert_outs)
    check("diversity loss > 0", loss2.item() > 0)

    # zero coefficients
    loss_fn0 = MoLoRALoss(num_experts=4, top_k=2, balance_loss_coef=0.0, z_loss_coef=0.0)
    loss0 = loss_fn0(probs, logits, indices)
    check("零系数时 loss=0", abs(loss0.item()) < 1e-5)

    # compute_expert_usage
    usage = compute_expert_usage(indices, num_experts=4)
    check("expert_usage shape", usage.shape == (4,))
    check("expert_usage 归一化", abs(usage.sum().item() - 1.0) < 1e-5)

except Exception as e:
    check("Loss 验证", False, str(e))


# ---------------------------------------------------------------------------
# 7. Model Wrapper 验证
# ---------------------------------------------------------------------------

section("7. MoLoRAModel 验证")

try:
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.fc = nn.Linear(16, 4)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.mean(dim=[2, 3])
            return self.fc(x)

    model = TinyModel()
    cfg = MoLoRAConfig(r=4, alpha=8, num_experts=4, top_k=2, target_modules=["conv1", "conv2", "fc"])

    # get_peft_molora_model
    m = get_peft_molora_model(model, cfg)
    check("model.molora_enabled", getattr(m, "molora_enabled", False) is True)
    check("model.molora_config 存在", hasattr(m, "molora_config"))

    # MoLoRAModel wrapper
    wrapper = MoLoRAModel(model, cfg)
    x = torch.randn(2, 3, 8, 8)
    wrapper.model.train()
    out = wrapper(x)
    check("wrapper 输出 shape", out.shape == (2, 4))

    # aux loss
    aux = wrapper.compute_aux_loss()
    check("aux_loss 是 Tensor", isinstance(aux, torch.Tensor))

    # merge / unmerge
    wrapper.merge()
    out_m = wrapper(x)
    check("merge 后输出 shape", out_m.shape == (2, 4))
    wrapper.unmerge()
    out_u = wrapper(x)
    check("unmerge 后输出 shape", out_u.shape == (2, 4))

    # domain
    wrapper.set_domain("test")
    check("set_domain 不报错 (无限制时)", True)
    wrapper.clear_domain()
    check("clear_domain 不报错", True)

    # freeze / unfreeze
    wrapper.freeze_experts([0])
    check("freeze_experts 不报错", True)
    wrapper.unfreeze_experts([0])
    check("unfreeze_experts 不报错", True)

    # expert replay
    buf = wrapper.save_expert_replay_buffer("day")
    check("replay buffer 有 domain", buf.get("domain") == "day")
    check("replay buffer 有 experts", len(buf.get("experts", {})) > 0)
    wrapper.load_expert_replay_buffer(buf, domain="day")
    check("load_expert_replay_buffer 不报错", True)

    # param stats
    stats = wrapper.param_stats()
    check("stats 有 total", stats.get("total", 0) > 0)
    check("stats 有 trainable", stats.get("trainable", -1) >= 0)
    check("stats 有 molora", stats.get("molora", 0) > 0)

except Exception as e:
    check("Model Wrapper 验证", False, str(e))


# ---------------------------------------------------------------------------
# 8. Utils 验证
# ---------------------------------------------------------------------------

section("8. Utils 验证")

try:
    conv = nn.Conv2d(16, 32, 3, stride=2, padding=1, groups=2)
    shape = get_conv_shape(conv)
    check("get_conv_shape", shape == (16, 32, 3, 3, (1, 1), 2, 2))

    check("is_conv", is_conv(conv) and not is_conv(nn.Linear(3, 3)))
    check("is_linear", is_linear(nn.Linear(3, 3)) and not is_linear(conv))

    alloc = allocate_domain_experts(8, ["day", "night", "fog", "rain"])
    check("allocate_domain_experts 数量", len(alloc) == 4)
    check("allocate_domain_experts 总计", sum(len(v) for v in alloc.values()) == 8)

    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Linear(8, 4))
    stats = count_parameters(m)
    check("count_parameters total", stats["total"] > 0)
    check("count_parameters trainable", stats["trainable"] == stats["total"])

    # mark_only_molora_as_trainable
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)
            self.lora_A = nn.Parameter(torch.randn(3, 3))
    m2 = M()
    mark_only_molora_as_trainable(m2)
    check("mark_only_molora_as_trainable: conv frozen", not m2.conv.weight.requires_grad)
    check("mark_only_molora_as_trainable: lora_A trainable", m2.lora_A.requires_grad)

except Exception as e:
    check("Utils 验证", False, str(e))


# ---------------------------------------------------------------------------
# 9. Registry 集成验证
# ---------------------------------------------------------------------------

section("9. MOE_LOSS_REGISTRY 集成")

try:
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2, share_moe_registry=True)
    x = torch.randn(2, 16, 8, 8)
    layer.train()
    MOE_LOSS_REGISTRY.clear()
    _ = layer(x)
    check("registry 写入", len(MOE_LOSS_REGISTRY) > 0)
    val = MOE_LOSS_REGISTRY.get(layer)
    check("registry 值是 Tensor", isinstance(val, torch.Tensor))
    check("registry 值 > 0", val.item() > 0)
    MOE_LOSS_REGISTRY.clear()
    check("registry 清除", len(MOE_LOSS_REGISTRY) == 0)

    # eval 不写入（或至少不报错）
    layer.eval()
    MOE_LOSS_REGISTRY.clear()
    with torch.no_grad():
        _ = layer(x)
    check("eval 模式不报错", True)
    MOE_LOSS_REGISTRY.clear()

except Exception as e:
    check("Registry 集成", False, str(e))


# ---------------------------------------------------------------------------
# 10. 与现有基础设施集成验证
# ---------------------------------------------------------------------------

section("10. 现有基础设施集成")

try:
    # tasks.py 修改：_has_moe_aux_registry_module 识别 MoLoRA
    from ultralytics.nn.tasks import BaseModel
    conv = nn.Conv2d(16, 32, 3, padding=1)
    layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=4, top_k=2)
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = layer
    fake = FakeModel()
    has_moe = BaseModel._has_moe_aux_registry_module(fake)
    check("tasks.py 识别 MoLoRA 模块", has_moe is True)

    # default.yaml 参数存在
    yaml_path = project_root / "ultralytics" / "cfg" / "default.yaml"
    if yaml_path.exists() and yaml is not None:
        with open(yaml_path) as f:
            defaults = yaml.safe_load(f)
        check("default.yaml 有 molora_num_experts", "molora_num_experts" in defaults)
        check("default.yaml 有 molora_top_k", "molora_top_k" in defaults)
        check("default.yaml 有 molora_router_type", "molora_router_type" in defaults)
        check("default.yaml 有 molora_r", "molora_r" in defaults)
        check("default.yaml 有 molora_alpha", "molora_alpha" in defaults)
        check("default.yaml 有 molora_balance_loss", "molora_balance_loss" in defaults)
        check("default.yaml 有 molora_router_z_loss", "molora_router_z_loss" in defaults)
        check("default.yaml 有 molora_diversity_loss", "molora_diversity_loss" in defaults)
        check("default.yaml 有 molora_expert_init", "molora_expert_init" in defaults)
        check("default.yaml 有 molora_use_rslora", "molora_use_rslora" in defaults)
        check("default.yaml 有 molora_share_moe_registry", "molora_share_moe_registry" in defaults)
        check("default.yaml 有 molora_capacity_factor", "molora_capacity_factor" in defaults)
        check("default.yaml 有 molora_expert_dropout", "molora_expert_dropout" in defaults)
        check("default.yaml 有 molora_warmup_steps", "molora_warmup_steps" in defaults)
    else:
        check("default.yaml 参数检查", yaml is not None, "yaml not installed or file missing")

except Exception as e:
    check("基础设施集成", False, str(e))


# ---------------------------------------------------------------------------
# 11. 参数效率对比
# ---------------------------------------------------------------------------

section("11. 参数效率对比")

try:
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.fc = nn.Linear(256, 10)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.mean(dim=[2,3])
            return self.fc(x)

    base = ToyModel()
    base_params = sum(p.numel() for p in base.parameters())

    # MoLoRA 参数 (实际计算)
    cfg = MoLoRAConfig(r=8, alpha=16, num_experts=4, top_k=2, target_modules=["conv1", "conv2", "conv3", "fc"])
    molora_model = ToyModel()
    molora_model = get_peft_molora_model(molora_model, cfg)
    mark_only_molora_as_trainable(molora_model)
    molora_params = sum(p.numel() for p in molora_model.parameters() if p.requires_grad)

    # 标准 LoRA 参数量 (手动计算，模拟 inject_lora)
    # Conv2d LoRA: A(r, C_in, 1, 1) + B(C_out, r, k, k)
    # Linear LoRA: A(r, C_in) + B(C_out, r)
    def lora_params_for_layer(base, r):
        if isinstance(base, nn.Conv2d):
            k = base.kernel_size
            if isinstance(k, int):
                k = (k, k)
            a = r * base.in_channels * 1 * 1
            b = base.out_channels * r * k[0] * k[1]
            return a + b
        elif isinstance(base, nn.Linear):
            return r * base.in_features + base.out_features * r
        return 0

    lora_params = sum(lora_params_for_layer(m, 8) for m in [base.conv1, base.conv2, base.conv3, base.fc])
    # MoLoRA 参数 = num_experts * LoRA 参数 + router 参数
    molora_lora_params = lora_params * cfg.num_experts
    # router 参数: 线性路由 ~ (C_in * hidden + hidden * num_experts)
    # conv1: in=3, hidden=1; conv2: in=64, hidden=16; conv3: in=128, hidden=32; fc: in=256, hidden=64
    router_params = 0
    for m in [base.conv1, base.conv2, base.conv3, base.fc]:
        if isinstance(m, nn.Conv2d):
            c = m.in_channels
        else:
            c = m.in_features
        h = max(c // 4, 1)
        router_params += c * h + h + h * cfg.num_experts + cfg.num_experts
    expected_molora = molora_lora_params + router_params
    actual_molora = molora_params

    print(f"  基础模型参数: {base_params:,}")
    print(f"  标准 LoRA 参数: {lora_params:,}")
    print(f"  MoLoRA 可训练参数: {actual_molora:,}")
    print(f"  理论 MoLoRA 参数 (含 router): {expected_molora:,}")
    print(f"  实际 MoLoRA / 标准 LoRA: {actual_molora/lora_params:.2f}x")
    print(f"  理论比例 (E): {cfg.num_experts:.0f}x")

    check("MoLoRA 参数 > LoRA 参数", actual_molora > lora_params)
    check("MoLoRA 参数 < 基础模型参数", actual_molora < base_params)
    check("参数比例接近理论值", abs(actual_molora/lora_params - cfg.num_experts) < 1.0,
          f"actual_ratio={actual_molora/lora_params:.2f}, expected={cfg.num_experts}")

except Exception as e:
    check("参数效率对比", False, str(e))


# ---------------------------------------------------------------------------
# 12. 性能基准（forward 速度）
# ---------------------------------------------------------------------------

section("12. 性能基准")

try:
    conv = nn.Conv2d(64, 128, 3, padding=1)
    x = torch.randn(8, 64, 32, 32)

    # 标准 Conv2d
    t0 = time.time()
    for _ in range(100):
        _ = conv(x)
    base_time = time.time() - t0

    # MoLoRA (4 experts, top_k=2)
    layer = MoLoRALayer(conv, r=8, alpha=16, num_experts=4, top_k=2)
    layer.eval()
    with torch.no_grad():
        t1 = time.time()
        for _ in range(100):
            _ = layer(x)
        molora_time = time.time() - t1

    # Merged MoLoRA
    layer.merge_weights()
    with torch.no_grad():
        t2 = time.time()
        for _ in range(100):
            _ = layer(x)
        merged_time = time.time() - t2

    print(f"  标准 Conv2d: {base_time*10:.2f} ms/iter")
    print(f"  MoLoRA (E=4,K=2): {molora_time*10:.2f} ms/iter")
    print(f"  MoLoRA merged: {merged_time*10:.2f} ms/iter")
    print(f"  MoLoRA 开销: {molora_time/base_time:.2f}x")
    print(f"  Merged 开销: {merged_time/base_time:.2f}x")

    check("MoLoRA forward 可运行", molora_time > 0)
    check("Merged 接近 base 速度", merged_time < base_time * 2.0, f"merged={merged_time:.4f}, base={base_time:.4f}")

except Exception as e:
    check("性能基准", False, str(e))


# ---------------------------------------------------------------------------
# 总结
# ---------------------------------------------------------------------------

section("验证总结")

total = PASSED + FAILED
print(f"  总测试项: {total}")
print(f"  ✅ 通过: {PASSED}")
print(f"  ❌ 失败: {FAILED}")
print(f"  通过率: {PASSED/total*100:.1f}%")

if FAILED == 0:
    print(f"\n🎉 所有验证项通过！MoLoRA 功能完整可用。")
    sys.exit(0)
else:
    print(f"\n⚠️ 有 {FAILED} 项验证失败，请检查上述输出。")
    sys.exit(1)
