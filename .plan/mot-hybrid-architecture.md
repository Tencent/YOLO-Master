- 在 COCO 或 VisDrone 数据集上，使用 scripts/compare_mot_ablation.py 作为参考脚本，训练至少 3 种模型变体：YOLO-Master-EsMoE-N（MoE 基线）、YOLO-Master-v0.10-MoT-N（MoT 实验模块）、YOLO-Master-v0.10-MoA-N（MoA 对比组）
- 对比测量每种变体的：mAP50-95、mAP50、Latency（ms，P50/P95/P99）、FLOPs（实际）、Params（M）、训练稳定性（loss 曲线是否发散、是否出现 NaN）

路由行为可解释性分析：
- 对 MoT：使用 diagnose_model 或自定义 hook 分析 MoTBlock 中各 Transformer expert（LocalConvTransformer / WindowTransformer / DeformableTransformer）的 token 路由分布，绘制专家激活热力图
- 对比不同场景（密集 vs 稀疏、小目标 vs 大目标）下的专家激活模式，验证 DeformableTransformer 是否在遮挡/不规则目标场景激活率显著上升

混合架构探索：
- 尝试将 MoT 与 MoE 进行层级组合（如 backbone 用 MoE、neck 用 MoT），或与 MoA 进行交叉组合，评估是否产生协同增益（mAP 提升 > 1% 或延迟降低 > 10% 视为有意义）

边界测试与稳定性修复：
- 补全 tests/test_mot.py 的边界测试（至少覆盖：MoTBlock 在 window_size 大于 feature map 时的降级处理、_WindowTransformerExpert 的 shift 操作在奇数尺寸输入时的边界、MoT 的 exploration_eps 在 eval 模式下是否被正确禁用）
- 若发现边界缺陷（如 IndexError、NaN、shape mismatch），需定位并修复
场景化洞察产出：基于对比数据，提出至少 3 条场景化推荐（如「密集小目标场景 MoT 的 WindowTransformer 激活率最高」；「复杂遮挡场景 DeformableTransformer 专家被优先路由」；「MoE + MoT 组合在 backbone 层带来 +X% mAP 但延迟增加 Y%」），每条需附数据支撑

- 完成后，在 GitHub Discussion 发表技术总结文章并提供实验脚本仓库链接
- 边界测试修复代码可提交 Pull Request；混合架构实验若产生稳定增益，可提交 Pull Request 补充新 YAML 配置

https://github.com/Tencent/YOLO-Master
https://github.com/Tencent/YOLO-Master/blob/main/scripts/compare_mot_ablation.py
https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/mot/mot.py
https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moa/moa.py
https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moe/diagnostics.py
https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/nn/modules/moe/pruning.py
https://github.com/Tencent/YOLO-Master/blob/main/tests/test_mot.py
https://github.com/Tencent/YOLO-Master/blob/main/tests/test_moa.py
https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml