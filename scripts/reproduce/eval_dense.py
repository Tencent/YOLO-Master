#!/usr/bin/env python3
"""Evaluate EsMoE-N with dense inference (--no-sparse-eval equivalent)."""
from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE


def eval_dense(model_path, data_yaml):
    """Evaluate model with dense inference."""
    model = YOLO(model_path)
    
    # Apply dense eval fix
    count = 0
    for module in model.model.modules():
        if isinstance(module, ES_MOE):
            module.use_sparse_inference = False
            count += 1
    print(f"Set {count} ES_MOE modules to dense inference")
    
    results = model.val(data=data_yaml, workers=0)
    mAP50 = results.results_dict.get("metrics/mAP50(B)")
    mAP50_95 = results.results_dict.get("metrics/mAP50-95(B)")
    print(f"mAP50: {mAP50}")
    print(f"mAP50-95: {mAP50_95}")
    return mAP50, mAP50_95


if __name__ == "__main__":
    print("=== VisDrone EsMoE-N (dense eval) ===")
    eval_dense("runs/reproduce/visdrone/VisDrone_EsMoE-N/weights/best.pt", "VisDrone.yaml")
    
    print("\n=== SKU-110K EsMoE-N (dense eval) ===")
    eval_dense("runs/reproduce/sku110k/SKU-110K_EsMoE-N/weights/best.pt", "SKU-110K.yaml")
