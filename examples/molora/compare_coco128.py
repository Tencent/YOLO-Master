"""MoLoRA vs LoRA 在 COCO128 上的真实对比实验。

使用 YOLOv8n 在 COCO128 上分别训练标准 LoRA 和 MoLoRA，
对比 mAP50 指标和训练参数效率。

Usage:
    python examples/molora/compare_coco128.py

注意：COCO128 会自动下载，首次运行需要网络。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import torch

from ultralytics import YOLO
from ultralytics.nn.peft.molora import (
    MoLoRAConfig, get_peft_molora_model, MoLoRAModel,
    mark_only_molora_as_trainable, count_parameters,
)
from ultralytics.utils.lora.api import apply_lora
from ultralytics.utils.lora.config import LoRAConfig


# COCO128 配置
DATA = "coco128.yaml"
MODEL = "yolov8n.pt"
EPOCHS = 3           # 快速验证（完整实验建议 50-100）
IMGSZ = 640
BATCH = 8
LR = 0.01
SEED = 42
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def train_baseline():
    """全参数微调基线。"""
    print("\n" + "="*60)
    print("训练 1: Baseline (全参数微调 yolov8n)")
    print("="*60)
    # 加载上次训练好的 baseline 权重（避免重复训练）
    baseline_best = Path.home() / "Downloads" / ".session_tmps" / "bed05d0f-229c-45d5-a6c5-2cc4abe4350e" / "YOLO-Master" / "runs" / "detect" / "train5" / "weights" / "best.pt"
    if baseline_best.exists():
        yolo = YOLO(str(baseline_best))
        print(f"  复用已训练的 baseline 权重: {baseline_best}")
    else:
        yolo = YOLO(MODEL)
        results = yolo.train(
            data=DATA,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            lr0=LR,
            lrf=0.01,
            seed=SEED,
            device=DEVICE,
            verbose=False,
            val=False,
            exist_ok=True,
        )
    val_results = yolo.val(data=DATA, imgsz=IMGSZ, device=DEVICE, verbose=False)
    mAP50 = val_results.results_dict.get("metrics/mAP50", 0.0)
    print(f"  Baseline mAP50: {mAP50:.4f}")
    elapsed = 0
    return {"mAP50": mAP50, "time": elapsed, "params": sum(p.numel() for p in yolo.model.parameters())}


def train_lora():
    """标准 LoRA 微调。"""
    print("\n" + "="*60)
    print(f"训练 2: LoRA (r=8, alpha=16)")
    print("="*60)
    yolo = YOLO(MODEL)
    # 应用 LoRA
    yolo.model = apply_lora(yolo.model, LoRAConfig(r=8, alpha=16, dropout=0.05))
    
    t0 = time.time()
    results = yolo.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        lr0=LR,
        lrf=0.01,
        seed=SEED,
        device=DEVICE,
        verbose=False,
        val=False,  # 跳过自动验证，最后手动验证
    )
    elapsed = time.time() - t0
    # 手动验证获取 mAP50
    val_results = yolo.val(data=DATA, imgsz=IMGSZ, device=DEVICE, verbose=False)
    mAP50 = val_results.results_dict.get("metrics/mAP50", 0.0)
    
    # 统计 LoRA 参数
    lora_params = sum(p.numel() for p in yolo.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in yolo.model.parameters())
    print(f"  LoRA mAP50: {mAP50:.4f}")
    print(f"  LoRA 可训练参数: {lora_params:,} / {total_params:,} ({100*lora_params/total_params:.2f}%)")
    print(f"  训练时间: {elapsed:.0f}s")
    return {"mAP50": mAP50, "time": elapsed, "params": lora_params, "total": total_params}


def train_molora():
    """MoLoRA 微调。"""
    print("\n" + "="*60)
    print(f"训练 3: MoLoRA (E=4, K=2, r=8, alpha=16)")
    print("="*60)
    yolo = YOLO(MODEL)
    
    # 应用 MoLoRA
    cfg = MoLoRAConfig(
        r=8, alpha=16,
        num_experts=4, top_k=2,
        router_type="linear",
        balance_loss_coef=0.01,
        z_loss_coef=0.001,
        use_rslora=True,
    )
    yolo.model = get_peft_molora_model(yolo.model, cfg)
    mark_only_molora_as_trainable(yolo.model)
    
    t0 = time.time()
    results = yolo.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        lr0=LR,
        lrf=0.01,
        seed=SEED,
        device=DEVICE,
        verbose=False,
        val=False,  # 跳过自动验证，最后手动验证
    )
    elapsed = time.time() - t0
    # 手动验证获取 mAP50
    val_results = yolo.val(data=DATA, imgsz=IMGSZ, device=DEVICE, verbose=False)
    mAP50 = val_results.results_dict.get("metrics/mAP50", 0.0)
    
    molora_params = sum(p.numel() for p in yolo.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in yolo.model.parameters())
    print(f"  MoLoRA mAP50: {mAP50:.4f}")
    print(f"  MoLoRA 可训练参数: {molora_params:,} / {total_params:,} ({100*molora_params/total_params:.2f}%)")
    print(f"  训练时间: {elapsed:.0f}s")
    return {"mAP50": mAP50, "time": elapsed, "params": molora_params, "total": total_params}


def main():
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS} (快速验证，建议完整实验用 50-100)")
    print(f"Dataset: {DATA} (自动下载)")
    
    # 三个实验顺序运行
    r1 = train_baseline()
    r2 = train_lora()
    r3 = train_molora()
    
    # 汇总
    print("\n" + "="*60)
    print("COCO128 对比结果")
    print("="*60)
    print(f"{'方法':<15} {'mAP50':>8} {'参数':>12} {'时间':>8}")
    print("-" * 50)
    print(f"{'Baseline':<15} {r1['mAP50']:>8.4f} {r1['params']:>12,} {r1['time']:>6.0f}s")
    print(f"{'LoRA':<15} {r2['mAP50']:>8.4f} {r2['params']:>12,} {r2['time']:>6.0f}s")
    print(f"{'MoLoRA':<15} {r3['mAP50']:>8.4f} {r3['params']:>12,} {r3['time']:>6.0f}s")
    print("-" * 50)
    print(f"LoRA vs Baseline: {r2['mAP50'] - r1['mAP50']:+.4f}")
    print(f"MoLoRA vs LoRA:   {r3['mAP50'] - r2['mAP50']:+.4f}")
    print(f"MoLoRA 参数 / LoRA 参数: {r3['params']/r2['params']:.2f}x")
    
    if r3['mAP50'] > r2['mAP50']:
        print("\n✅ MoLoRA 在 COCO128 上优于 LoRA")
    else:
        print("\n⚠️ MoLoRA 在 COCO128 上未超过 LoRA（可能需要更多 epoch 或调参）")
    
    # 保存结果
    import json
    report = {
        "baseline": r1,
        "lora": r2,
        "molora": r3,
    }
    out_path = Path(__file__).resolve().parent / "coco128_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
