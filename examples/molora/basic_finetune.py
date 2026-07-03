"""MoLoRA 基础微调示例：COCO 数据集单域微调。

Usage:
    python examples/molora/basic_finetune.py

展示如何用 MoLoRA 替代标准 LoRA 进行目标检测微调。
"""
import sys
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ultralytics import YOLO
from ultralytics.nn.peft.molora import MoLoRAConfig, get_peft_molora_model


def main():
    # 1. 加载预训练模型
    yolo = YOLO("yolov8n.pt")
    model = yolo.model

    # 2. 创建 MoLoRA 配置（标准预设）
    cfg = MoLoRAConfig(
        r=8,
        alpha=16,
        num_experts=4,
        top_k=2,
        router_type="linear",
        balance_loss_coef=0.01,
        z_loss_coef=0.001,
        use_rslora=True,
        expert_init="default",
    )

    # 3. 包装模型
    model = get_peft_molora_model(model, cfg)
    print(f"[MoLoRA] Wrapped model with {cfg.num_experts} experts, top_k={cfg.top_k}")

    # 4. 设置回 YOLO 对象
    yolo.model = model

    # 5. 训练（只训练 MoLoRA 参数，基础权重自动冻结）
    # 由于基础层已冻结，effective batch 和 learning rate 可能需要调整
    results = yolo.train(
        data="coco128.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=0.01,  # MoLoRA 参数量小，可用稍大 lr
        lrf=0.01,
        freeze=0,  # 不冻结 backbone，让 MoLoRA 自己控制
        augment=True,
    )

    # 6. 推理前合并权重（零开销）
    # yolo.model.merge()  # 如果 yolo.model 是 MoLoRAModel 实例
    # val_results = yolo.val(data="coco128.yaml")

    print("[Done] 训练完成，最佳 mAP:", results.results_dict.get("metrics/mAP50", "N/A"))


if __name__ == "__main__":
    main()
