# YOLO-Master Wiki 导航

## 🏠 [首页](Home.md)

## 🔷 MoE — Mixture-of-Experts

- [MoE 总览](MoE/Home.md)
- [核心模块详解](MoE/MoE_Core_Modules.md)
- [路由与专家模块](MoE/MoE_Routers_Experts.md)
- [训练、损失与剪枝](MoE/MoE_Training_Loss_Pruning.md)
- [诊断与分析工具](MoE/MoE_Diagnostics_Analysis.md)
- [Mixture of Attention](MoE/Mixture_of_Attention.md)
- [版本演进指南 (CN)](MoE/MoE_Modules_Explanation.md)
- [Version Evolution (EN)](MoE/MoE_Modules_Explanation_EN.md)

## 🔶 PEFT — Parameter-Efficient Fine-Tuning

- [PEFT 总览](PEFT/Home.md)
- [LoRA 核心实现](PEFT/PEFT_LoRA_Core.md)
- [Mixture-of-LoRA](PEFT/PEFT_MoLoRA.md)
- [训练策略与 IO](PEFT/PEFT_Training_IO.md)
- [PEFT Planner](PEFT/PEFT_Planner.md)

---

## 📁 源码映射

| Wiki 文档 | 源码路径 |
|-----------|----------|
| MoE 核心模块 | `ultralytics/nn/modules/moe/` |
| Mixture of Attention | `ultralytics/nn/modules/moa/` |
| Mixture-of-LoRA | `ultralytics/nn/peft/molora/` |
| LoRA 核心与 Planner | `ultralytics/utils/lora/` |
