选择两个差异显著的垂类场景
- 建议 VisDrone 密集航拍 
- brain-tumor 稀疏医疗

为 YOLO-Master-EsMoE-N 在 examples/lora_examples/ 下新增 
- yolo_master_visdrone_lora.yaml 
- yolo_master_brain_tumor_lora.yaml

> 配置需覆盖：lora_r、lora_alpha、lora_use_rslora、lora_target_modules、lora_include_attention、lora_gradient_checkpointing

针对 MoE 模型明确路由层（routing）是否纳入 LoRA 目标模块，并在配置文件中注释说明理由
在每个场景上对比至少 3 组 rank（r=4, 8, 16）的微调效果，记录 mAP50-95、可训练参数量、训练时间、峰值显存
训练限制在 20~50 epoch 内完成，模拟少样本快速迭代场景

提供 README.md 说明各场景最佳 rank 推荐、目标模块选择建议、常见陷阱（如医疗灰度通道处理、航拍尺度变化）
提交 Pull Request，包含 LoRA 配置文件、训练脚本、对比表格、适配指南

https://github.com/Tencent/YOLO-Master
https://github.com/Tencent/YOLO-Master/tree/main/examples/lora_examples
https://github.com/Tencent/YOLO-Master/blob/main/examples/lora_examples/README.md
https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/datasets/VisDrone.yaml
https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml