#!/usr/bin/env python3
"""
YOLO-Master Full Ablation Experiment — COCO2017 / COCO128
===========================================================
基于 full_ablation_spec.py 统一接口，运行全部 11 个 PEFT 变体：
  Full / LoRA / DoRA / LoHa / IA3 / HRA + 5 种 MoLoRA

支持：
  • COCO2017 全量实验（默认，50 epochs）
  • COCO128 快速模式（--quick，3 epochs）
  • 可选推理延迟测量（--measure-latency）
  • 多尺度 mAP 分解（小 / 中 / 大目标）
  • 实时 JSON 落盘 + 控制台进度与汇总表

用法：
    python scripts/full_ablation.py
    python scripts/full_ablation.py --quick --measure-latency
    python scripts/full_ablation.py --dataset coco --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 环境初始化 —— 必须在 import ultralytics 之前完成
# ═══════════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ablation_suite"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ["WANDB_MODE"] = "disabled"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

import numpy as np
import torch

# This experiment script uses one process per run. A stale torchrun
# environment would make the trainer enter its CUDA-only DDP path.
if not torch.cuda.is_available() and any(
    os.environ.get(name) not in (None, "", "-1") for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
):
    raise RuntimeError(
        "Distributed environment detected without CUDA. "
        "Run this script with plain python3 and unset RANK, LOCAL_RANK, WORLD_SIZE."
    )

import torch.nn as nn

from ultralytics.utils import SETTINGS
SETTINGS["wandb"] = False

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

# 标准 LoRA 基础设施
from ultralytics.utils.lora import apply_lora
from ultralytics.utils.lora.config import LoRAConfig

# MoLoRA 基础设施
from ultralytics.nn.peft.molora import (
    MoLoRAConfig,
    MoLoRAMoEAwareConfig,
    MoLoRAConfigBuilder,
    get_peft_molora_model,
    build_moe_aware_layer,
    mark_only_molora_as_trainable,
)
from ultralytics.nn.peft.molora.model import _parent_child_name, _get_submodule

import ultralytics
assert str(REPO_ROOT) in ultralytics.__file__, (
    f"加载的不是当前仓库的 ultralytics！got {ultralytics.__file__}"
)

# 统一接口规范
from full_ablation_spec import (
    DatasetConfig,
    VariantConfig,
    ExperimentResult,
    DATASET_REGISTRY,
    FULL_ABLATION_VARIANTS,
    QUICK_ABLATION_VARIANTS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 全局配置
# ═══════════════════════════════════════════════════════════════════════════════
HERE = Path(__file__).parent
MODEL_PATH = str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt")
RESULTS_JSON = HERE / "full_ablation_results.json"
PROJECT_DIR = HERE / "runs_full_ablation"

# 设备选择: MPS优先 → CUDA → CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

# 强制行缓冲，便于实时观察后台输出
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# MoLoRA 公共默认超参
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def count_params(m: nn.Module) -> Tuple[int, int]:
    """统计模型总参数量与可训练参数量。"""
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def detect_adapter_signature(m: nn.Module) -> Dict[str, Any]:
    """
    检测模型中各类型 PEFT adapter 的存在性与数量。
    """
    names = [n for n, _ in m.named_parameters()]
    return {
        "has_lora_A": any("lora_A" in n for n in names),
        "has_lora_B": any("lora_B" in n for n in names),
        "has_dora_magnitude": any("magnitude_vector" in n for n in names),
        "has_loha": any("hada" in n.lower() for n in names),
        "has_ia3": any("ia3" in n.lower() for n in names),
        "has_hra": any("hra" in n.lower() or "householder" in n.lower() for n in names),
        "has_adalora": any("adalora" in n.lower() or "s_vector" in n.lower() for n in names),
        "has_molora": any("molora" in n.lower() or "router" in n.lower() for n in names),
        "has_router_calibration": any("router_calibration" in n.lower() for n in names),
        "n_lora_params": sum(1 for n in names if "lora_" in n.lower()),
        "n_router_params": sum(1 for n in names if "router" in n.lower()),
    }


def collect_molora_diagnostics(model: nn.Module) -> Dict[str, Any]:
    """
    收集 MoLoRA 特有的诊断信息。
    """
    diag: Dict[str, Any] = {
        "molora_enabled": getattr(model, "molora_enabled", False),
        "molora_config": str(getattr(model, "molora_config", "N/A")),
        "per_expert_ranks": None,
        "router_calib_present": False,
        "router_types": {},
        "expert_counts": {},
        "top_k_values": {},
    }

    for name, module in model.named_modules():
        if hasattr(module, "_expert_ranks") and module._expert_ranks is not None:
            if diag["per_expert_ranks"] is None:
                diag["per_expert_ranks"] = module._expert_ranks

        if hasattr(module, "router_calibration") and module.router_calibration is not None:
            diag["router_calib_present"] = True

        if hasattr(module, "router"):
            router_type = type(module.router).__name__
            diag["router_types"][router_type] = diag["router_types"].get(router_type, 0) + 1

        if hasattr(module, "num_experts"):
            key = f"{name}:E{module.num_experts}"
            diag["expert_counts"][key] = diag["expert_counts"].get(key, 0) + 1
        if hasattr(module, "top_k"):
            key = f"{name}:K{module.top_k}"
            diag["top_k_values"][key] = diag["top_k_values"].get(key, 0) + 1

    return diag


def extract_final_metrics(results) -> Dict[str, float]:
    """从 ultralytics Results 对象中提取最终评估指标。"""
    final_metrics: Dict[str, float] = {}
    if results is None:
        return final_metrics

    if hasattr(results, "results_dict") and results.results_dict:
        final_metrics = {
            k: float(v) for k, v in results.results_dict.items()
            if isinstance(v, (int, float))
        }
    elif hasattr(results, "metrics") and results.metrics:
        final_metrics = {
            k: float(v) for k, v in results.metrics.items()
            if isinstance(v, (int, float))
        }

    return final_metrics


def extract_multiscale_map(val_results) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    最佳尝试：从验证结果中提取小/中/大目标的 mAP。
    COCOeval stats 数组索引: 0=AP, 1=AP50, 2=AP75, 3=APs, 4=APm, 5=APl
    """
    map_s = map_m = map_l = None
    try:
        if hasattr(val_results, "box") and val_results.box is not None:
            box = val_results.box
            # 方式1: 通过 eval stats
            if hasattr(box, "eval") and box.eval is not None:
                stats = getattr(box.eval, "stats", None)
                if stats is not None and len(stats) >= 6:
                    map_s = float(stats[3])
                    map_m = float(stats[4])
                    map_l = float(stats[5])
            # 方式2: 通过 maps 属性
            elif hasattr(box, "maps") and box.maps is not None:
                maps = box.maps
                if len(maps) >= 3:
                    map_s = float(maps[0])
                    map_m = float(maps[1])
                    map_l = float(maps[2])

        # 方式3: 通过 results_dict 中的显式键
        rd = getattr(val_results, "results_dict", {})
        if rd:
            if "metrics/mAP_small" in rd:
                map_s = float(rd["metrics/mAP_small"])
            if "metrics/mAP_medium" in rd:
                map_m = float(rd["metrics/mAP_medium"])
            if "metrics/mAP_large" in rd:
                map_l = float(rd["metrics/mAP_large"])
    except Exception:
        pass

    return map_s, map_m, map_l


def measure_latency(model: nn.Module, imgsz: int = 640, device: str = DEVICE, runs: int = 50) -> float:
    """
    测量 PyTorch eager 模式下单张图像的推理延迟（ms）。
    """
    try:
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        model.eval()

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        # 同步设备
        if device == "mps":
            torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()

        # 正式测量
        times = []
        with torch.no_grad():
            for _ in range(runs):
                if device == "mps":
                    torch.mps.synchronize()
                elif device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(x)
                if device == "mps":
                    torch.mps.synchronize()
                elif device.startswith("cuda"):
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

        return round(float(np.mean(times)), 3)
    except Exception as exc:
        warnings.warn(f"Latency measurement failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PEFT 应用器
# ═══════════════════════════════════════════════════════════════════════════════

def apply_standard_peft(model: YOLO, train_kwargs: Dict[str, Any]) -> YOLO:
    """
    显式应用标准 PEFT (LoRA / DoRA / LoHa / IA3 / HRA)。
    ultralytics 的 model.train() 会根据 lora_type 等参数自动注入，
    但这里显式调用 apply_lora 以确保 adapter 在训练前已正确挂载。
    """
    lora_type = train_kwargs.get("lora_type", "lora")
    lora_r = train_kwargs.get("lora_r", LORA_R)
    lora_alpha = train_kwargs.get("lora_alpha", LORA_ALPHA)
    lora_dropout = train_kwargs.get("lora_dropout", LORA_DROPOUT)
    lora_backend = train_kwargs.get("lora_backend", "peft")
    use_dora = train_kwargs.get("lora_use_dora", False)

    cfg = LoRAConfig(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        backend=lora_backend,
        variant=lora_type,
        use_dora=use_dora,
        peft_type=lora_type,
    )
    model = apply_lora(model.model, cfg)

    # apply_lora 返回 DetectionModel，需要重新包装为 YOLO
    if not isinstance(model, YOLO):
        yolo_model = YOLO(MODEL_PATH)
        yolo_model.model = model
        model = yolo_model

    return model


def apply_molora_standard(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """
    应用标准 MoLoRA (Mixture-of-LoRA)。
    """
    cfg = MoLoRAConfig(
        r=config.get("r", LORA_R),
        alpha=config.get("alpha", LORA_ALPHA),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        router_type=config.get("router_type", "linear"),
        dropout=config.get("dropout", LORA_DROPOUT),
        use_rslora=config.get("use_rslora", True),
        balance_loss_coef=config.get("balance_loss_coef", 0.01),
        z_loss_coef=config.get("z_loss_coef", 0.001),
    )
    get_peft_molora_model(model.model, cfg)
    return model


def apply_molora_moe_aware(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """
    应用 MoLoRA + MoE-aware (per-expert rank, frequency-based)。
    """
    cfg = MoLoRAMoEAwareConfig(
        r=config.get("r", LORA_R),
        alpha=config.get("alpha", LORA_ALPHA),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        router_type=config.get("router_type", "linear"),
        dropout=config.get("dropout", LORA_DROPOUT),
        use_rslora=config.get("use_rslora", True),
        balance_loss_coef=config.get("balance_loss_coef", 0.01),
        z_loss_coef=config.get("z_loss_coef", 0.001),
        per_expert_rank=True,
        rank_allocator_mode="frequency",
        rank_budget_total=config.get("rank_budget_total", 32),
        rank_min=config.get("rank_min", 2),
    )

    target_modules = MoLoRAConfigBuilder.auto_detect_targets(
        model.model, r=cfg.r, include_moe=True, only_backbone=False
    )

    modules_dict = dict(model.model.named_modules())
    for name in target_modules:
        if name not in modules_dict:
            continue
        base_layer = modules_dict[name]
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            continue

        parent_name, child_name = _parent_child_name(name)
        parent = _get_submodule(model.model, parent_name) if parent_name else model.model
        if parent is None or not hasattr(parent, child_name):
            continue

        num_experts = cfg.num_experts
        if num_experts == 4:
            usage_history = torch.tensor([0.5, 0.2, 0.2, 0.1])
        else:
            x = torch.linspace(0, 1, num_experts)
            usage_history = torch.exp(-3 * x)
            usage_history = usage_history / usage_history.sum()

        layer = build_moe_aware_layer(base_layer, cfg, usage_history=usage_history)
        setattr(parent, child_name, layer)

    model.model.molora_config = cfg
    model.model.molora_enabled = True
    mark_only_molora_as_trainable(model.model)
    return model


def apply_molora_router_calibration(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """
    应用 MoLoRA + Router Calibration (ΔW_r)。
    """
    cfg = MoLoRAMoEAwareConfig(
        r=config.get("r", LORA_R),
        alpha=config.get("alpha", LORA_ALPHA),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        router_type=config.get("router_type", "linear"),
        dropout=config.get("dropout", LORA_DROPOUT),
        use_rslora=config.get("use_rslora", True),
        balance_loss_coef=config.get("balance_loss_coef", 0.01),
        z_loss_coef=config.get("z_loss_coef", 0.001),
        router_calibration=True,
        router_calib_rank=config.get("router_calib_rank", 4),
    )

    target_modules = MoLoRAConfigBuilder.auto_detect_targets(
        model.model, r=cfg.r, include_moe=True, only_backbone=False
    )

    modules_dict = dict(model.model.named_modules())
    for name in target_modules:
        if name not in modules_dict:
            continue
        base_layer = modules_dict[name]
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            continue

        parent_name, child_name = _parent_child_name(name)
        parent = _get_submodule(model.model, parent_name) if parent_name else model.model
        if parent is None or not hasattr(parent, child_name):
            continue

        layer = build_moe_aware_layer(base_layer, cfg, usage_history=None)
        setattr(parent, child_name, layer)

    model.model.molora_config = cfg
    model.model.molora_enabled = True
    mark_only_molora_as_trainable(model.model)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 单变体执行逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def run_variant(
    variant: VariantConfig,
    dataset: DatasetConfig,
    seed: int = 42,
    measure_latency_flag: bool = False,
) -> ExperimentResult:
    """
    运行单个消融实验变体，返回 ExperimentResult 结构。
    所有错误被 try/except 隔离，失败时 ok=False 继续后续实验。
    """
    print(f"\n{'='*78}")
    print(f"=== Variant: {variant.name.upper()} {'='*55}")
    print(f"{'='*78}")
    print(f"Dataset    : {dataset.name} ({dataset.description})")
    print(f"Description: {variant.description}")
    print(f"PEFT type  : {variant.peft_type}")
    print(f"Epochs     : {variant.epochs} | Batch: {variant.batch} | Imgsz: {variant.imgsz}")
    if variant.train_kwargs:
        print(f"train kwargs: {variant.train_kwargs}")
    if variant.molora_config:
        print(f"MoLoRA config: {variant.molora_config}")

    t0 = time.time()
    result = ExperimentResult(
        dataset=dataset.name,
        variant=variant.name,
        seed=seed,
        ok=False,
    )
    model = None

    try:
        # ── 4.1 加载模型 ──
        model = YOLO(MODEL_PATH)
        base_total, base_train = count_params(model.model)
        print(f"[Pre-train]  total={base_total:,}  trainable={base_train:,}  ({base_train/base_total*100:.2f}%)")

        # ── 4.2 应用 PEFT ──
        if variant.peft_type == "full":
            # 全量微调：不做任何 PEFT 修改，但需显式解冻
            # （checkpoint 可能是在 PEFT/冻结状态下保存的）
            for p in model.model.parameters():
                p.requires_grad = True
        elif variant.peft_type == "peft":
            # 标准 PEFT (LoRA/DoRA/LoHa/IA3/HRA)：显式注入 adapter
            model = apply_standard_peft(model, variant.train_kwargs)
        elif variant.peft_type == "molora":
            model = apply_molora_standard(model, variant.molora_config)
        elif variant.peft_type == "molora_aware":
            model = apply_molora_moe_aware(model, variant.molora_config)
        elif variant.peft_type == "molora_calib":
            model = apply_molora_router_calibration(model, variant.molora_config)
        else:
            raise ValueError(f"Unknown peft_type: {variant.peft_type}")

        # ── 4.3 训练前统计 ──
        post_total, post_train = count_params(model.model)
        sig = detect_adapter_signature(model.model)
        molora_diag = collect_molora_diagnostics(model.model)

        print(f"[Post-wrap]  total={post_total:,}  trainable={post_train:,}  ({post_train/post_total*100:.2f}%)")
        print(f"[Adapter]    {sig}")
        if molora_diag.get("molora_enabled"):
            print(f"[MoLoRA]     enabled={molora_diag['molora_enabled']}, "
                  f"router_calib={molora_diag['router_calib_present']}, "
                  f"ranks={molora_diag['per_expert_ranks']}")
            if molora_diag.get("router_types"):
                print(f"[Router]     types={molora_diag['router_types']}")

        # ── 4.4 训练 ──
        train_kwargs = {
            "data": dataset.yaml,
            "epochs": variant.epochs,
            "batch": variant.batch,
            "imgsz": variant.imgsz,
            "device": DEVICE,
            "project": str(PROJECT_DIR),
            "exist_ok": True,
            "verbose": False,
            "workers": 2,
            "patience": 0,
            "plots": False,
            "save": False,
            "seed": seed,
        }
        # 仅标准 PEFT 变体将 train_kwargs 中的 lora_type 等参数传给 train()
        if variant.peft_type == "peft" and variant.train_kwargs:
            train_kwargs.update(variant.train_kwargs)

        train_results = model.train(**train_kwargs)

        # ── 4.5 提取训练指标 ──
        final_metrics = extract_final_metrics(train_results)
        result.final_metrics = final_metrics
        result.ok = True
        print(f"[Final metrics] {json.dumps(final_metrics, indent=2, ensure_ascii=False)}")

        # ── 4.6 多尺度 mAP 分解（训练后独立验证）──
        try:
            val_results = model.val(
                data=dataset.yaml,
                imgsz=variant.imgsz,
                device=DEVICE,
                verbose=False,
                plots=False,
            )
            map_s, map_m, map_l = extract_multiscale_map(val_results)
            result.map_small = map_s
            result.map_medium = map_m
            result.map_large = map_l
            if any(v is not None for v in (map_s, map_m, map_l)):
                print(f"[Multi-scale mAP] small={map_s}, medium={map_m}, large={map_l}")
        except Exception as e:
            print(f"[WARN] 多尺度 mAP 验证失败: {e}")

        # ── 4.7 可选推理延迟测量 ──
        if measure_latency_flag:
            try:
                lat_ms = measure_latency(model.model, imgsz=variant.imgsz, device=DEVICE)
                if lat_ms is not None:
                    result.latency_ms = lat_ms
                    result.latency_backend = "pytorch_eager"
                    print(f"[Latency] {lat_ms:.2f} ms (PyTorch eager, {variant.imgsz}px)")
            except Exception as e:
                print(f"[WARN] 延迟测量失败: {e}")

        # ── 4.8 参数与 adapter 签名落盘 ──
        result.params_total = post_total
        result.params_trainable = post_train
        result.trainable_pct = round(post_train / post_total * 100, 4) if post_total else 0.0
        result.adapter_sig = sig
        result.molora_diagnostics = molora_diag

    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {result.error}")
        traceback.print_exc()

        # 失败时仍尝试兜底统计参数
        try:
            if model is not None:
                post_total, post_train = count_params(model.model)
                result.params_total = post_total
                result.params_trainable = post_train
                result.trainable_pct = round(post_train / post_total * 100, 4) if post_total else 0.0
                result.adapter_sig = detect_adapter_signature(model.model)
                result.molora_diagnostics = collect_molora_diagnostics(model.model)
        except Exception:
            pass

    result.elapsed_sec = round(time.time() - t0, 1)
    print(f"[Done] elapsed={result.elapsed_sec}s  ok={result.ok}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 控制台输出与汇总
# ═══════════════════════════════════════════════════════════════════════════════

def print_header(variants: List[VariantConfig], dataset: DatasetConfig, measure_latency_flag: bool):
    """打印实验环境信息。"""
    print("\n" + "=" * 78)
    print("  YOLO-Master Full Ablation Experiment")
    print("=" * 78)
    print(f"  Model        : {MODEL_PATH}")
    print(f"  Dataset      : {dataset.name} ({dataset.description})")
    print(f"  Device       : {DEVICE}")
    print(f"  Variants     : {len(variants)}")
    print(f"  Latency meas.: {'YES' if measure_latency_flag else 'NO'}")
    print(f"  Output dir   : {PROJECT_DIR}")
    print(f"  Results JSON : {RESULTS_JSON}")
    print("=" * 78)
    print(f"\n[Boot] ultralytics: {ultralytics.__file__}")
    print(f"[Boot] version     : {ultralytics.__version__}")
    print(f"[Boot] torch       : {torch.__version__}  device={DEVICE}")
    print()


def print_summary_table(records: List[ExperimentResult]):
    """打印控制台汇总表。"""
    header = (
        f"{'Variant':<16} {'OK':<3} {'Total':>12} {'Trainable':>12} {'%':>7} "
        f"{'mAP50-95':>10} {'mAPs':>8} {'mAPm':>8} {'mAPl':>8} {'Lat(ms)':>8} {'Time(s)':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in records:
        m = r.final_metrics.get("metrics/mAP50-95(B)", float("nan"))
        m_str = f"{m:.4f}" if isinstance(m, float) and m == m else "N/A"
        ms = f"{r.map_small:.4f}" if r.map_small is not None else "N/A"
        mm = f"{r.map_medium:.4f}" if r.map_medium is not None else "N/A"
        ml = f"{r.map_large:.4f}" if r.map_large is not None else "N/A"
        lat = f"{r.latency_ms:.2f}" if r.latency_ms is not None else "N/A"

        print(
            f"{r.variant:<16} "
            f"{'Y' if r.ok else 'N':<3} "
            f"{r.params_total:>12,} "
            f"{r.params_trainable:>12,} "
            f"{r.trainable_pct:>7.3f} "
            f"{m_str:>10} "
            f"{ms:>8} "
            f"{mm:>8} "
            f"{ml:>8} "
            f"{lat:>8} "
            f"{r.elapsed_sec:>8.1f}"
        )

    print("=" * len(header))

    # 额外打印 adapter 签名明细表
    sig_header = (
        f"{'Variant':<16} {'lora_A':>6} {'dora':>5} {'loha':>5} {'ia3':>4} "
        f"{'hra':>4} {'molora':>7} {'calib':>5} {'router_params':>13}"
    )
    print("\n" + "-" * len(sig_header))
    print(sig_header)
    print("-" * len(sig_header))
    for r in records:
        sig = r.adapter_sig
        print(
            f"{r.variant:<16} "
            f"{'Y' if sig.get('has_lora_A') else 'N':>6} "
            f"{'Y' if sig.get('has_dora_magnitude') else 'N':>5} "
            f"{'Y' if sig.get('has_loha') else 'N':>5} "
            f"{'Y' if sig.get('has_ia3') else 'N':>4} "
            f"{'Y' if sig.get('has_hra') else 'N':>4} "
            f"{'Y' if sig.get('has_molora') or r.molora_diagnostics.get('molora_enabled') else 'N':>7} "
            f"{'Y' if sig.get('has_router_calibration') else 'N':>5} "
            f"{sig.get('n_router_params', 0):>13,}"
        )
    print("=" * len(sig_header))


def save_results(records: List[ExperimentResult]) -> bool:
    """将结果实时写入 JSON 文件。"""
    try:
        RESULTS_JSON.write_text(
            json.dumps(
                [r.to_dict() for r in records],
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        print(f"[WARN] 结果写入失败: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="YOLO-Master Full Ablation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="使用 COCO128 快速模式与 QUICK_ABLATION_VARIANTS（3 epochs）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称（默认 quick 模式用 coco128，全量用 coco）",
    )
    parser.add_argument(
        "--measure-latency",
        action="store_true",
        help="训练后额外测量 PyTorch eager 推理延迟",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="仅运行指定变体，逗号分隔（如 'full,lora_r16,molora_4e2k'）",
    )
    args = parser.parse_args()

    # 选择数据集与变体列表
    if args.quick:
        dataset_name = args.dataset or "coco128"
        variant_list = QUICK_ABLATION_VARIANTS
    else:
        dataset_name = args.dataset or "coco"
        variant_list = FULL_ABLATION_VARIANTS

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}")
    dataset = DATASET_REGISTRY[dataset_name]

    # 变体过滤
    if args.variants:
        allowed = {v.strip() for v in args.variants.split(",")}
        variant_list = [v for v in variant_list if v.name in allowed]
        if not variant_list:
            raise ValueError(f"无匹配的变体。可用: {[v.name for v in FULL_ABLATION_VARIANTS]}")

    print_header(variant_list, dataset, args.measure_latency)
    PROJECT_DIR.mkdir(exist_ok=True, parents=True)

    all_records: List[ExperimentResult] = []
    total = len(variant_list)

    for idx, variant in enumerate(variant_list, start=1):
        print(f"\n[Progress] {idx}/{total} — 开始运行变体 '{variant.name}'")
        rec = run_variant(
            variant=variant,
            dataset=dataset,
            seed=args.seed,
            measure_latency_flag=args.measure_latency,
        )
        all_records.append(rec)

        # 实时落盘：单个失败不丢之前结果
        save_results(all_records)

        # 清理 GPU / MPS 缓存，避免跨实验内存泄漏
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── 最终汇总 ──
    print_summary_table(all_records)

    print(f"\n{'='*78}")
    print(f"全部 {total} 个变体运行完毕。")
    print(f"  - 数据集    : {dataset.name}")
    print(f"  - 成功      : {sum(1 for r in all_records if r.ok)}")
    print(f"  - 失败      : {sum(1 for r in all_records if not r.ok)}")
    print(f"  - 结果 JSON : {RESULTS_JSON}")
    print(f"  - 训练日志  : {PROJECT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
