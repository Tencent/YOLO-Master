"""MoLoRA 持续学习示例：多域顺序训练（白天 -> 黑夜 -> 雾天）。

展示如何用 MoLoRA 的域隔离和专家回放机制防止灾难性遗忘。

Usage:
    python examples/molora/continual_learning.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from ultralytics import YOLO
from ultralytics.nn.peft.molora import (
    MoLoRAConfig,
    get_peft_molora_model,
    MoLoRAModel,
    allocate_domain_experts,
)


# 模拟域数据集（实际使用时替换为真实数据集路径）
DOMAIN_DATASETS = {
    "day": "day_coco.yaml",
    "night": "night_coco.yaml",
    "fog": "fog_coco.yaml",
}


def train_domain(wrapper, domain, epochs=30):
    """在指定域上训练，使用 domain-specific 专家。"""
    print(f"\n========== Training domain: {domain} ==========")
    wrapper.set_domain(domain)
    wrapper.model.train()

    # 创建临时 YOLO 对象用于训练
    yolo = YOLO()
    yolo.model = wrapper.model

    data = DOMAIN_DATASETS.get(domain, "coco128.yaml")
    results = yolo.train(
        data=data,
        epochs=epochs,
        imgsz=640,
        batch=16,
        lr0=0.005,
        lrf=0.01,
        freeze=0,
        augment=True,
    )
    return results


def evaluate_domain(wrapper, domain):
    """在指定域上评估。"""
    print(f"\n---------- Evaluating domain: {domain} ----------")
    wrapper.set_domain(domain)
    wrapper.model.eval()

    yolo = YOLO()
    yolo.model = wrapper.model
    data = DOMAIN_DATASETS.get(domain, "coco128.yaml")
    results = yolo.val(data=data, imgsz=640)
    return results


def main():
    # 1. 加载模型
    yolo = YOLO("yolov8n.pt")
    model = yolo.model

    # 2. 分配专家：8 专家分给 3 个域
    num_experts = 8
    domains = ["day", "night", "fog"]
    domain_experts = allocate_domain_experts(num_experts, domains)
    print(f"[Domain Allocation] {domain_experts}")

    # 3. 创建 MoLoRA 配置
    cfg = MoLoRAConfig(
        r=8,
        alpha=16,
        num_experts=num_experts,
        top_k=2,
        router_type="linear",
        domain_experts=domain_experts,
        balance_loss_coef=0.01,
        z_loss_coef=0.001,
        use_rslora=True,
    )

    # 4. 包装并冻结非 MoLoRA 参数
    model = get_peft_molora_model(model, cfg)
    wrapper = MoLoRAModel(model, cfg)
    print(f"[MoLoRA] Initialized continual learning model")

    # 5. 阶段 1：训练 day 域
    train_domain(wrapper, "day", epochs=30)
    day_buffer = wrapper.save_expert_replay_buffer("day")
    day_results = evaluate_domain(wrapper, "day")
    print(f"Day mAP50: {day_results.results_dict.get('metrics/mAP50', 'N/A')}")

    # 6. 阶段 2：训练 night 域（冻结 day 专家）
    wrapper.freeze_experts(domain_experts["day"])
    train_domain(wrapper, "night", epochs=30)
    night_buffer = wrapper.save_expert_replay_buffer("night")

    # 评估 day 域：回放 day 专家防止遗忘
    wrapper.load_expert_replay_buffer(day_buffer, domain="day")
    day_after_night = evaluate_domain(wrapper, "day")
    print(f"Day mAP50 after night training (with replay): {day_after_night.results_dict.get('metrics/mAP50', 'N/A')}")

    # 评估 night 域
    night_results = evaluate_domain(wrapper, "night")
    print(f"Night mAP50: {night_results.results_dict.get('metrics/mAP50', 'N/A')}")

    # 7. 阶段 3：训练 fog 域
    wrapper.unfreeze_experts(domain_experts["day"])  # 解冻以便联合训练
    wrapper.freeze_experts(domain_experts["day"] + domain_experts["night"])
    train_domain(wrapper, "fog", epochs=30)
    fog_buffer = wrapper.save_expert_replay_buffer("fog")

    # 8. 最终评估：使用所有域专家
    wrapper.unfreeze_experts()  # 全部解冻
    for domain in domains:
        # 设置域并回放
        wrapper.set_domain(domain)
        wrapper.load_expert_replay_buffer(
            {"day": day_buffer, "night": night_buffer, "fog": fog_buffer}[domain],
            domain=domain,
        )
        results = evaluate_domain(wrapper, domain)
        print(f"Final {domain} mAP50: {results.results_dict.get('metrics/mAP50', 'N/A')}")

    # 9. 保存最终模型
    wrapper.save_checkpoint("molora_continual_final.pt")
    print("\n[Done] Continual learning complete. Checkpoint saved.")


if __name__ == "__main__":
    main()
