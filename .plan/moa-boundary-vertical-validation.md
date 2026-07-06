在现有 tests/test_moa.py 基础上，系统补全 MoA 模块的边界测试与训练验证，提升测试覆盖率
测试补全（至少覆盖以下场景）：
- NeckMoAFusion 在跨尺度输入尺寸不匹配（如 hi 为 15×15、lo 为 7×7，非严格 2× 下采样）时的前向稳定性与形状保持
- MoABlock 的 temperature 退火到极小值（如 temperature < 1e-4）时，softmax 路由概率的数值稳定性（是否出现 NaN 或均匀分布）
- _LocalAttnHead 与 _GlobalAttnHead 在 num_heads 不能被 dim 整除时的降级处理（_safe_groups 的边界）
- C2fMoA 的 aux_loss 在多 MoABlock 嵌套时是否存在重复计数（类似 MoE 的 MOE_LOSS_REGISTRY 双计数问题）

缺陷修复：在补充测试过程中，若发现任何边界缺陷（如 IndexError、NaN 传播、形状不匹配），需一并定位并修复
覆盖率报告：
- 提供 pytest --cov 前后的覆盖率对比，至少覆盖 ultralytics/nn/modules/moa/ 目录
- 垂类训练验证：在 VisDrone 或 SKU-110K 上，使用 YOLO-Master-v0.10-MoA-N 训练 50~100 epoch，验证 MoA 模块在真实数据集上的收敛性，记录 mAP50-95 与 loss 曲线，与同配置的 MoE 基线对比
- 提交 Pull Request，包含测试代码、修复代码、覆盖率报告、训练日志

https://github.com/Tencent/YOLO-Master
https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moa/moa.py
https://github.com/Tencent/YOLO-Master/blob/main/tests/test_moa.py
https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml
https://github.com/Tencent/YOLO-Master/blob/main/scripts/compare_moa_ablation.py