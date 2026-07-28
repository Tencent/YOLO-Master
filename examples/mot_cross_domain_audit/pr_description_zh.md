# PR：MoT 同检查点路由审计与 P5 低预算混合架构

关联 Issue #54。

## 改动摘要

- 新增 MoT-P5 YAML：保留 EsMoE backbone，仅在 P5 neck 放置一个 MoTBlock；
- 扩展消融脚本，支持精确加载训练 checkpoint、实际 FLOPs、P50/P95/P99、SHA-256 和稳定性事件；
- 新增四卡训练、统一测速、跨域与 VisDrone 场景审计的一键编排；
- 新增图像级 bootstrap/permutation/BH-FDR、样本指纹重叠审计和扰动稳健性分析；
- 支持 16-bit/float TIFF、科研灰度图归一化和 letterbox；
- 修复 MoT 稀疏专家在 autocast 下的 dtype 回写错误；
- 为 AMP 恢复增加显式 warning 与结构化事件记录；
- 补充边界、统计、TIFF、混合配置及样本重叠测试；
- 提交原始 CSV、训练曲线、路由热力图、复现协议和中文实验链路。

## 受控实验

VisDrone，四模型均从 YAML 初始化，30 epoch，640，batch 16，seed 42，FP32。

| 模型 | mAP50-95 | mAP50 | P50/P95/P99 (ms) | FLOPs (G) | Params (M) |
|---|---:|---:|---:|---:|---:|
| EsMoE-N | 0.09014 | 0.17446 | 18.563/18.671/18.959 | 8.505 | 3.420 |
| MoT-N | 0.08912 | 0.16984 | 47.467/47.714/51.414 | 12.503 | 4.025 |
| MoA-N | 0.08710 | 0.16837 | 40.650/40.870/43.381 | 9.102 | 3.546 |
| MoT-P5-N | **0.09164** | 0.17309 | **19.628/19.707/20.539** | **8.632** | **3.494** |

MoT-P5 相比完整 MoT 的 P50 降低 58.65%、FLOPs 降低 30.96%，mAP50-95 增加 0.252
个百分点；相比 EsMoE 仅增加 0.150 个百分点且 P50 增加 5.74%。“mAP 提升 > 1%”按相对比例
是 1.66%，按绝对百分点则未达到，需维护者确认口径；本 PR 保守地不声明已产生协同增益。

## 路由结论

- dense 到 sparse：Deformable mean probability `+0.000590`
 （95% CI `[0.000451, 0.000734]`, `q=0.00105`）；
- large 到 small：Window top-1 share `+0.004537`
 （95% CI `[0.000850, 0.008337]`, `q=0.03474`），mean probability 不显著；
- VisDrone 到 brain-tumor：Deformable top-1 share `+0.009824`，仅作为 OOD 路由信号；
- dense 与 irregular/occluded 共享 104/128 张图，修复后禁用无效的独立样本推断，原遮挡假设未验证；
- 深层路由概率接近均匀，后续需重训 `top_k/exploration/straight-through` 消融。

## 验证

```bash
pytest \
  tests/test_ddp_lifecycle_ema_nan.py \
  tests/test_mot_cross_domain_analysis.py \
  tests/test_mot.py \
  tests/test_mot_routing_diagnostics.py -q
```

结果：`82 passed, 4 warnings`。

```bash
ruff check \
  scripts/analyze_mot_cross_domain.py \
  scripts/run_mot_cross_domain_experiment.py \
  scripts/compare_mot_ablation.py \
  tests/test_mot_cross_domain_analysis.py
```

结果：通过。

## 范围与限制

- 这是 30 epoch 单 seed 架构对比，不作为充分收敛 SOTA 声明；
- brain-tumor 只做固定 checkpoint 的 OOD 路由审计，不报告医疗性能；
- 公开结果不包含权重、原始数据、本地路径或私人线粒体数据；
- 未达到阈值的 MoT-P5 YAML 可作为实验配置评审，不建议直接设为默认模型。
