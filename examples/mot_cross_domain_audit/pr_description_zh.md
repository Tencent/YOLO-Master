# PR：MoT 序列级路由审计与 P5 低预算混合架构

关联 Issue #54。

## 改动摘要

- 新增 MoT-P5 YAML：保留 EsMoE backbone，仅在低分辨率 P5 neck 放置一个 MoTBlock；
- 扩展消融脚本：精确加载 checkpoint、实际 FLOPs、SHA-256、确定性输入、时长预热和多轮
  P50/P95/P99；
- 新增四卡训练、统一 benchmark、跨域/场景/真实遮挡审计的一键编排；
- 新增视频序列级 paired bootstrap、sign-flip permutation、paired Hedges' g 和 BH-FDR；
- 从 VisDrone 原始 occlusion 标注构建序列内低/高遮挡配对，并平衡目标数和框面积；
- 支持 16-bit/float TIFF、科研灰度图稳健归一化和 letterbox；
- 修复 MoT sparse expert 在 autocast 下的 dtype 回写错误；
- 为 AMP 自动恢复增加 warning 与结构化事件记录；
- 提交脱敏 CSV、曲线、热力图、复现协议及完整中文实验链路。

上游已经包含 Issue 指定的三项 `tests/test_mot.py` 边界测试，本分支负责复验，不将其声明为新增。
新增测试聚焦 TIFF、dtype、统计、序列配对、遮挡匹配和 benchmark 状态隔离。

## 与已合并 PR 的差异

- [#96](https://github.com/Tencent/YOLO-Master/pull/96) 已完成基础 MoT/MoA+MoT 配置、消融
  和边界测试；本 PR 的结构增量是 P5-only MoT；
- [#146](https://github.com/Tencent/YOLO-Master/pull/146) 已完成 VisDrone 消融与跨域路由，
  并在限制中指出 COCO/VisDrone 使用不同训练域模型；
- 本 PR 固定同一个 MoT checkpoint，直接消除上述参数混杂；
- 本 PR 进一步修正连续视频帧伪重复，加入原始遮挡标注配对和三轮稳定 benchmark。

因此建议将本 PR 作为 #54 的方法学修正与增量结果评审，而不是重复合入已有基础实现。

## 受控实验

四模型均从 YAML 初始化，在 VisDrone 上训练 30 epoch，640，batch 16，seed 42，FP32。

| 模型 | 训练 mAP50-95 | 当前代码 mAP50-95 | 当前 P50/P95/P99 (ms) | FLOPs (G) | Params (M) |
|---|---:|---:|---:|---:|---:|
| EsMoE-N | 0.09014 | 0.086933 | 13.284/13.338/13.795 | 8.016 | 3.420 |
| MoT-N | 0.08912 | 0.086633 | 30.928/31.262/32.177 | 12.014 | 4.025 |
| MoA-N | 0.08710 | 0.084307 | 26.382/26.514/27.796 | 8.613 | 3.546 |
| MoT-P5-N | **0.09164** | **0.087923** | **17.008/17.065/17.180** | **8.143** | **3.494** |

训练 mAP 来自训练提交 `58cb439`；当前 mAP 与延迟来自合并上游 `d5afc4b` 后的兼容性复验。
benchmark 使用同一 RTX 5090、固定输入、每模型至少 2 秒预热、3 轮执行和顺序轮换。

MoT-P5 相比完整 MoT：当前 mAP50-95 增加 0.129 个百分点，P50 降低 45.01%，FLOPs 降低
32.22%。相比 EsMoE：mAP50-95 仅增加 0.099 个百分点，P50 增加 28.03%。因此只声明它是
完整 MoT 的低预算替代，不声明相对 EsMoE 已产生协同。

## 路由结论

- 初版图像级 dense/sparse 与 large/small 显著项，在视频序列配对后均未通过 FDR+CI；
- 真实遮挡审计包含 25 对视频序列，目标数/框面积 SMD 为 0.027/-0.017；
- 低遮挡 → 高遮挡的 Deformable top-1 为 `+0.001575`，CI 跨 0，`q=0.703`；
- VisDrone → brain-tumor 的 Deformable mean probability 为 `+0.000674`，仅作 OOD 信号；
- 深层概率接近均匀且 top-1 对微小数值变化敏感，后续需重训
  `top_k/exploration/straight-through`，不在本 PR 中无实验修改默认行为。

## 验证

```bash
pytest \
  tests/test_ddp_lifecycle_ema_nan.py \
  tests/test_mot_cross_domain_analysis.py \
  tests/test_mot.py \
  tests/test_mot_routing_diagnostics.py -q

ruff check \
  scripts/analyze_mot_cross_domain.py \
  scripts/run_mot_cross_domain_experiment.py \
  scripts/compare_mot_ablation.py \
  scripts/prepare_mot_routing_scenes.py \
  tests/test_mot_cross_domain_analysis.py
```

结果：`88 passed, 4 warnings`。4 个 warning 均来自既有 MoA head 数量自动调整。

## 范围与限制

- 30 epoch、单 seed，不作为充分收敛 SOTA 声明；
- brain-tumor 只做固定 checkpoint 的 OOD 路由审计，不报告医疗检测性能；
- 单 GPU PyTorch latency 不外推到 TensorRT、ncnn 或其他硬件；
- 公开结果不包含权重、原始数据、本地路径、凭据或私人线粒体数据；
- MoT-P5 未达到相对 EsMoE 的预设增益阈值，不建议直接设为默认配置。
