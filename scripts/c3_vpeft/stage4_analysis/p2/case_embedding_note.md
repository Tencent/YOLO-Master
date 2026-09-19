# [P2 调研笔记·草稿] case-embedding 与 V-PEFT 小样本的位置关系

> 状态: 调研方向笔记, 非本 C3 实验结论。供后续扩展(如 k-shot 档位实验)参考。

## 概念速写

- **case embedding**(案例嵌入/支持集编码): 将少量标注样本(shots)经编码器压成嵌入,
  在推理时以检索/原型/上下文形式注入模型, 让模型"照样本判类"。典型形态:
  原型网络(Prototypical Net)、kNN/简单向量机分类、基于支持集的相似度融合、
  以及大模型侧把 case 编进 prompt/前缀的 in-context 检索增强。
- 与**参数高效微调**的连接: 支持集嵌入也可以作为"放置先验"或"每个 case 的轻量适配"
  与 LoRA 等适配器结合(如按检索到的 case 选择 adapter / 动态 rank)。

## 与 V-PEFT(本实验) 的映射

- 本 C3 采用同预算"全量数据×三策略"口径(vpeft 的"小样本实战"体现在模型与数据规模小 +
  planner 把稀疏预算自动放到敏感层), 尚未跑 k=5/10/20 支持集档位;
- V-PEFT 的 planner 在"容量/敏感度先验"上的作用 ≈ case-embedding 路线中"哪些层适合
  携带少样本信息"的先验; 两者可组合为 *case-embedding-informed placement*:
  检索出与 query 最像的 shot 子集 → 让 planner 在该子集损失上放置 LoRA。

## 参考方向(待正式调研补充, 勿作引用结论)

1. few-shot detection 经典: 新类 support-set 微调检测头/两阶段 proposal;
2. 原型/相似度检索分支 + 冻结主干;
3. adapter 工厂: 每类一个轻量 adapter, 推理按类路由(= 类别粒度 case embedding);
4. 大规模 VLM/分割模型 in-context(如 SAM 的提示), 与本 n-scale YOLO 路线不同, 仅作对标。

## 后续实验建议(若做 k 档)

- 在 stage3 的 k-shot 数据划分(prepare_*.py 已生成 k 档子集)上补 vpeft 的 3-5 个 k 档,
  并记录每档 planner 决策(预算/目标层)是否随 shot 数迁移 —— 作为
  "case-embedding 先验"的证据。资源估计: 每档 1 单元 ~1h, 受 GPU 排队约束, P2 可选。
