"""MoLoRA vs LoRA 在 COCO128 上的真实对比实验（精简版）。

复用已跑完的 Baseline (mAP50=0.639) 和 LoRA (mAP50=0.625) 结果，
只运行 MoLoRA 2 epoch 快速对比。

Usage:
    python examples/molora/compare_coco128_fast.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import torch

from ultralytics import YOLO
from ultralytics.nn.peft.molora import (
    MoLoRAConfig, get_peft_molora_model,
    mark_only_molora_as_trainable,
)


DATA = "coco128.yaml"
MODEL = "yolov8n.pt"
EPOCHS = 2
IMGSZ = 640
BATCH = 8
LR = 0.01
SEED = 42
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS} (快速对比)")

    # Baseline 和 LoRA 结果已在前两次运行中获得
    baseline_mAP50 = 0.639
    lora_mAP50 = 0.625
    lora_params = 1_196_376  # 从日志中读取

    print(f"\n{'='*60}")
    print(f"复用 Baseline: mAP50={baseline_mAP50:.4f}")
    print(f"复用 LoRA:     mAP50={lora_mAP50:.4f} (params: {lora_params:,})")
    print(f"{'='*60}")

    # MoLoRA 训练
    print(f"\n{'='*60}")
    print(f"训练: MoLoRA (E=4, K=2, r=8, alpha=16)")
    print(f"{'='*60}")

    yolo = YOLO(MODEL)
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
    molora_params = sum(p.numel() for p in yolo.model.parameters() if p.requires_grad)

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
        val=False,
        exist_ok=True,
    )
    elapsed = time.time() - t0

    # 手动验证
    val_results = yolo.val(data=DATA, imgsz=IMGSZ, device=DEVICE, verbose=False)
    # Ultralytics 的 val_results 结构: val_results.box.map50 是 mAP50
    mAP50 = getattr(val_results.box, 'map50', 0.0)
    if mAP50 == 0.0 and hasattr(val_results, 'results_dict'):
        mAP50 = val_results.results_dict.get('metrics/mAP50', 0.0)
    # 兜底: 直接从 box 对象读取
    if mAP50 == 0.0 and hasattr(val_results, 'box'):
        mAP50 = val_results.box.map50

    print(f"  MoLoRA mAP50: {mAP50:.4f}")
    print(f"  MoLoRA 参数:  {molora_params:,}")
    print(f"  训练时间:     {elapsed:.0f}s")

    # 汇总
    print(f"\n{'='*60}")
    print(f"COCO128 对比结果 ({EPOCHS} epoch)")
    print(f"{'='*60}")
    print(f"{'方法':<12} {'mAP50':>8} {'参数':>12} {'说明':>20}")
    print("-" * 55)
    print(f"{'Baseline':<12} {baseline_mAP50:>8.4f} {'3,157,200':>12} {'全参数微调':>20}")
    print(f"{'LoRA':<12} {lora_mAP50:>8.4f} {lora_params:>12,} {'PEFT (r=8)':>20}")
    print(f"{'MoLoRA':<12} {mAP50:>8.4f} {molora_params:>12,} {'MoE+LoRA (E=4,K=2)':>20}")
    print("-" * 55)
    print(f"LoRA vs Baseline: {lora_mAP50 - baseline_mAP50:+.4f}")
    print(f"MoLoRA vs LoRA:   {mAP50 - lora_mAP50:+.4f}")
    print(f"MoLoRA 参数 / LoRA 参数: {molora_params/lora_params:.2f}x")

    if mAP50 > lora_mAP50:
        print(f"\n✅ MoLoRA 在 COCO128 上优于 LoRA (+{mAP50 - lora_mAP50:.4f})")
    elif mAP50 > baseline_mAP50 * 0.95:
        print(f"\n⚠️ MoLoRA 接近 LoRA 水平，需更多 epoch 或调参")
    else:
        print(f"\n❌ MoLoRA 未达预期，需检查配置")


if __name__ == "__main__":
    main()
