# [Issue #54] MoT 消融、序列级路由解释与 P5 混合架构

本文是 GitHub Discussion 发布草稿。所有数值均有公开 CSV；初版结论被复验推翻的过程也予以保留。

## 摘要

本实验在 VisDrone 上以相同预算训练 EsMoE-N、MoT-N、MoA-N 和新增 MoT-P5-N，并固定同一个
MoT checkpoint 分析自然图像、航拍场景和 brain-tumor OOD 输入。

- 分支：`blues-kun/YOLO-Master:rhino-2026/issue-54-medical-routing`
- 训练提交：`58cb439407e2f5f7a6e1c4b6a3a9382499713e88`
- 方法修正提交：`9eb5076b0c7ed77a976b2efb0ff3d0e988e6be8a`
- 合入上游：`d5afc4b442aed815506914505064c5ef946de5ef`
- 训练协议：`30 epoch / imgsz=640 / batch=16 / seed=42 / FP32`
- MoT 权重 SHA-256：`a1857c81b7aebd0efb5a56f9d5b37405ef83edcc68890add15c9c480e9fee629`

上游 #96 已完成基础 MoT/MoA+MoT 配置和边界测试，#146 已完成 VisDrone 消融，但其跨域比较
使用不同训练域模型。本实验的增量是固定 checkpoint、按视频序列配对、使用真实遮挡标注，并量化
多轮 benchmark 波动。

## 消融结果

训练提交的最佳结果：

| 模型 | mAP50-95 | mAP50 | NaN/发散/恢复 |
|---|---:|---:|---|
| EsMoE-N | 0.09014 | 0.17446 | 0 |
| MoT-N | 0.08912 | 0.16984 | 0 |
| MoA-N | 0.08710 | 0.16837 | 0 |
| MoT-P5-N | **0.09164** | 0.17309 | 0 |

合并当前上游后，四个 checkpoint 在 VisDrone 548 val 上的 mAP50-95 为
0.086933/0.086633/0.084307/0.087923，MoT-P5 排名保持第一。

当前 benchmark 使用固定输入、至少 2 秒预热、3 轮执行和轮间顺序旋转：

| 模型 | P50/P95/P99 (ms) | P50 run min-max | FLOPs (G) | Params (M) |
|---|---:|---:|---:|---:|
| EsMoE-N | 13.284/13.338/13.795 | 11.084-13.418 | 8.016 | 3.420 |
| MoT-N | 30.928/31.262/32.177 | 30.879-31.106 | 12.014 | 4.025 |
| MoA-N | 26.382/26.514/27.796 | 26.268-26.457 | 8.613 | 3.546 |
| MoT-P5-N | **17.008/17.065/17.180** | **16.989-17.058** | **8.143** | **3.494** |

MoT-P5 相比完整 MoT 的 P50 降低 45.01%、FLOPs 降低 32.22%，mAP50-95 增加
0.129 个百分点；相比 EsMoE，P50 增加 28.03%，mAP50-95 只增加 0.099 个百分点。因此它是
完整 MoT 的低预算替代，不是已证实的 MoE 协同结构。

## 路由解释：一次被数据推翻的结论

第一版以图像为独立样本，得到 sparse 相对 dense 的 Deformable mean probability
`+0.000590 (q=0.00105)`，以及 small 相对 large 的 Window top-1
`+0.004537 (q=0.03474)`。

复查发现 VisDrone 图像来自连续视频，相邻帧并非独立重复。改为序列内聚合、共有序列配对
bootstrap 和 sign-flip permutation 后：

- dense/sparse 只有 10 个配对序列；
- large/small 有 21 个配对序列；
- 没有任何路由指标同时满足 `q <= 0.05` 且 bootstrap CI 不跨 0。

因此不再发布“稀疏场景偏好 Deformable”或“小目标偏好 Window”的场景推荐。

## 真实遮挡复验

代理 `irregular_occluded` 与 dense 重叠 104/128 张图，也没有直接使用遮挡标签。第二轮读取
VisDrone 原始 occlusion 字段，每个视频序列匹配一对低/高遮挡图，并平衡目标数和框面积。

- 25 对视频序列；
- 平均遮挡比例 0.284 → 0.753；
- 目标数 SMD 0.027，框面积 SMD -0.017；
- Deformable top-1 `+0.001575`，95% CI `[-0.005933, 0.009746]`，`q=0.703`；
- Deformable mean probability `+0.000040`，CI 跨 0。

当前 checkpoint 不支持“遮挡显著提高 Deformable 激活”的假设。

## 从目标激活到检测效用

进一步把 38,759 个目标按序列、类别、truncation 精确匹配，并限定面积比不超过 2，得到
12,296 对。浅层 `model.14.m.0` 中，LocalConv/Deformable 框内概率分别增加
0.00693/0.00259，Window 降低 0.00952。遮挡对应的是层相关组合重分配，不是单专家切换。

随后对同一图逐专家强制 `model.14.m.0`，用 `box+cls+dfl` 构造检测效用矩阵。2,048 图中，
原路由平均 regret 为 0.03380，而 oracle 的 Local/Window/Deformable 占比为
36.9%/35.7%/27.4%。

冻结检测器和专家训练的 utility router 在 calibration val 将 regret 从 0.02186 降至
0.01737，但在 test-dev 恶化到 0.05476。预先由 val 固定的 KL guard 在 test-dev 触发，恢复
基线 0.04556，说明当前只能安全拒绝，不能声明泛化提升。

adaptive K 阈值 0.35 将目标层实际调用从 3.000 降至 1.484，mAP50-95 从 0.08743 降至
0.08695；三轮 P50 中位数从 26.877 ms 升至 28.244 ms。局部稀疏成立，端到端加速未成立。

## OOD 与路由器诊断

VisDrone → brain-tumor 的 Deformable mean probability 增加 `0.000674`，95% CI
`[0.000520, 0.000834]`。这是小幅 OOD 路由变化，不是医疗检测或临床效果。

深层三专家概率接近 1/3；同一 checkpoint 两次代码版本复验中，Deformable top-1 差值从
`+0.009824` 变为 `+0.043936`，而 mean probability 基本稳定。`top_k=1` 梯度探针也显示
router 主任务梯度接近零。后续应重训 `top_k/exploration/straight-through` 消融，并同时报告
dense probability、entropy 和 margin，不能只看 argmax 热力图。

## 三条数据支撑的建议

1. 需要保留 MoT 且控制成本时，选择 MoT-P5；相比完整 MoT，P50 降低 45.01%。
2. 速度优先仍选择 EsMoE；MoT-P5 相比它的 P50 高 28.03%。
3. 当前数据不支持按密度、尺度或遮挡硬切单一专家；目标级结果显示的是组合重分配。

## 复现与限制

```bash
python scripts/run_mot_cross_domain_experiment.py \
  --devices 0 1 2 3 \
  --epochs 30 \
  --no-amp \
  --project runs/mot_cross_domain
```

这是 30 epoch、单 seed、单类 GPU 的受控实验，不代表充分收敛 SOTA，也不外推到 TensorRT/ncnn。
2026-07-29 在 Python 3.9.25、PyTorch 2.8.0+cu128 上复验为
`134 passed, 18 warnings`：14 条为 Matplotlib/pyparsing 弃用提示，4 条为既有 MoA head
自动调整，无失败。公开结果不含权重、原始数据、私人线粒体图像或本地路径。
