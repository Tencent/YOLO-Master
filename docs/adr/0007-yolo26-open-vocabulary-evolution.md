# ADR-0007: YOLO-Master-OV —— 基于 YOLO26 底座的开放词汇检测演进

## Status

Proposed

## Context

YOLO-Master 当前是闭集实时检测器，核心资产为 ES-MoE 动态路由带来的 instance-conditional 自适应计算（COCO 42.4 AP @ 1.62ms）。项目的既定演进目标是 Open-Vocabulary Detection 与 Open-Set Segmentation。

外部环境的关键变化：

- **YOLO26（Ultralytics, 2026-01）** 引入端到端 NMS-Free 双头（one-to-many 训练监督 + one-to-one 推理输出）、去除 DFL、STAL 小目标标签分配、ProgLoss + MuSGD 训练管线。大词汇表下 NMS 开销与延迟抖动显著放大，NMS-Free 对开放词汇场景收益更大；DFL-free 的轻量头也更适合替换为 region-text 相似度头。
- **YOLOE-26 已由 Ultralytics 官方发布**（YOLO26 + YOLOE 范式，YOLOE-26x 文本提示 LVIS minival AP 40.6）。因此"YOLO26 + 开放词汇 head"本身不构成差异化。
- **VLM grounding 模型集群**（Rex-Omni-3B、WeVisOne-VL-2B/4B、GroundingPI-4B、GroundAnything-4B、LocateAnything-3B）把检测统一为坐标 token 生成，精度上限高但为 2B~4B 参数、单图百毫秒至秒级延迟，且均已开源批量标注管线。趋势性能力包括：离散坐标 token 协议、并行/扩散解码、以及"目标不存在返回 None"的可靠拒识（WeVisOne 在 WeRef-Bench 拒绝率 96.4%）。

差异化机会落在三点：ES-MoE 条件化稀疏计算（开放词汇检测中的效率学派）、VLM 教师蒸馏数据引擎（对标 WeVisOne 的 100M 图像数据引擎）、可靠拒识机制。

## Decision

将 YOLO-Master 底座迁移至 YOLO26，引入 YOLOE 式开放词汇 head，并以 VLM 教师蒸馏管线供给训练数据。整体架构：

```
                        ┌──────────────────────────────────────────┐
  Text prompts ───────► │ MobileCLIP text encoder (离线/训练期)      │
  (offline vocabulary)  └──────────────┬───────────────────────────┘
                                       │ prompt embeddings P ∈ R^{C×D}
                                       ▼
  Image ──► YOLO26 Backbone (ES-MoE) ──► PAN Neck (ES-MoE + 可选 RepVL-PAN 式文本注入)
                                       │
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                         ▼
   one-to-many head (训练)    one-to-one head (推理)      Abstention head
   box reg (DFL-free)         box reg (DFL-free)          objectness 校准 +
   + object embedding O       + object embedding O        query-conditional 拒识
              │                        │
              └──── Label = O · P^T (region-text 对比) ────┘
```

### 1. 开放词汇 Head：主选 RepRTA（YOLOE 式），备选 RepVL-PAN（YOLO-World 式）

**主选 RepRTA**：

- 分类层输出通道从闭集类别数 `nc` 改为 embedding 维度 `D`，推理时以 prompt 嵌入作为分类权重与 anchor 的 object embedding 对比：`Label = O · P^T`；
- 训练期文本嵌入精炼辅助网络在推理时重参数化进分类头，**推理开销与闭集 YOLO26 完全一致**；
- 与 YOLO26 的 DFL-free 轻量头天然兼容：head 不再随类别数膨胀，词汇表大小只影响一次矩阵乘宽度；
- Ultralytics 已内置 YOLOE-26 实现，可 fork 其 head 代码做 MoE 化改造，工程成本最低。

**备选/消融 RepVL-PAN**：在 Neck 的 CSPLayer 注入文本引导（T-CSPLayer + I-Pooling Attention），跨模态融合更深、上限可能更高，但换词表需重新重参数化并导出模型，部署灵活性差。作为论文消融项保留。

### 2. ES-MoE 迁移与开放词汇化改造（核心创新）

**Step 1（保守迁移）**：ES-MoE block 平移至 YOLO26 backbone 高层 stage（C3k2/PSABlock 替换位）与 Neck，保持 image-level 动态路由，先验证 MoE 在新底座上的精度-延迟收益。

**Step 2（Prompt 条件化路由，主创新）**：路由器输入扩展为视觉特征 + prompt 嵌入：

```
router_logits = W_v · gap(x_visual) + W_t · P_mean   # P_mean: 当前词表嵌入聚合
```

- 语义自适应计算：query 为"远处的信号灯"时激活小目标专家，query 为"卡车"时激活大目标/纹理专家；
- 部署兼容：prompt 固定时路由可按词表预固化（route-once-per-vocabulary），不破坏 RepRTA 的零开销特性；prompt 动态变化时退回逐 token 路由。两种模式均需支持并做消融。

**风险对策**：YOLO26 的 one-to-one head + STAL 分配可能与 MoE load-balancing loss 相互干扰（路由倾向影响正样本分布）。对策为 balance loss 权重随 ProgLoss 调度同步退火，前 50% epoch 冻结路由为均匀分布 warmup。

### 3. 拒识（Abstention）机制

- one-to-one head 增加轻量校准分支：对 top-K 候选的 region-text 相似度做温度标定（验证集拟合）；
- 相似度全部低于阈值 τ 时输出空集，τ 可按场景配置；
- 训练数据加入负样本 query（图像中不存在的类别），由 VLM 教师管线自动产出；
- 评测：自建拒识 benchmark + WeRef-Bench 风格 absent-target 测试。

## 数据管线：VLM 教师蒸馏

```
无标注图像池 (Objects365 / OpenImages / SA-1B 子集 / 自有数据)
        │
        ▼
┌─ Stage 1: 概念发现 ─────────────────────────────────┐
│  用 RAM/Tag2Text 或 VLM caption 提取候选类别/短语       │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌─ Stage 2: 多教师并行标注（互补专家）────────────────────┐
│  Teacher A: Rex-Omni-3B      → detection/pointing     │
│  Teacher B: WeVisOne-VL-4B   → box + mask + 拒识判定   │
│  Teacher C: GroundingPI-4B   → referring/point (难样本) │
│  均支持 vLLM/SGLang serving + JSONL 批量标注            │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌─ Stage 3: 交叉验证与清洗 ──────────────────────────────┐
│  · 框级：多教师 IoU 投票，≥2 教师 IoU>0.5 才保留          │
│  · 语义级：CLIP/MobileCLIP 对框内 crop 与文本算相似度过滤  │
│  · 拒识标签：教师输出 None 且其余教师也无检出 → 负样本     │
│  · 置信度分桶：高置信直接入训练集，中置信入蒸馏软标签池      │
└──────────────────────┬───────────────────────────────┘
                       ▼
        region-text pairs 训练集（YOLO-World 格式兼容）
        + 负样本 query 集（拒识训练用）
```

蒸馏策略：

1. **硬标签预训练**：高置信伪标签与 Objects365/GoldG/LVIS 人工标注混合，region-text 对比预训练（在线词表，batch 内正负名词采样，默认 M=80 可上调至 256）；
2. **软标签蒸馏（可选）**：教师 similarity logits 分布作为 soft target，对学生的 `O·P^T` 分布做 KL 蒸馏；
3. **拒识蒸馏**：教师输出 None 的 query-image 对作为负样本，配合 abstention 分支训练。

工程量估算：教师部署 3~5 天（2~3 卡 H 系列）；100 万张图像 × 平均 5 query 的批量标注与清洗约 1~2 周；数据格式转换 2~3 天。

## 训练路线图（四阶段）

| 阶段 | 目标 | 数据 | 关键点 |
|---|---|---|---|
| **S1 底座复现** | YOLO26 + ES-MoE 闭集复现，验证 MoE 在新底座收益 | COCO | 只换 backbone/neck block，不动 head；对比 YOLO26-N baseline |
| **S2 开放词汇预训练** | 接 RepRTA head，region-text 对比预训练 | Objects365 + GoldG + LVIS + 伪标签 | 冻结 text encoder；MoE 路由先 image-only |
| **S3 Prompt 条件化路由** | 路由器升级为 prompt-conditioned | 同 S2 | 与 S2 消融；验证 route-once 固化的零开销部署 |
| **S4 拒识 + 下游微调** | 加 abstention 分支；LoRA 下游适配 | 负样本 + 下游数据 | 复用现有 LoRA/PEFT 体系；VisDrone/SKU-110K 验证 |

## 评测方案

| 维度 | Benchmark | 指标 | 对标 |
|---|---|---|---|
| 开放词汇检测 | LVIS minival / LVIS val | AP, APr/APc/APf | YOLOE-26 (40.6@x)、YOLO-Worldv2、GroundingDINO-T |
| 密集场景 | Dense200 / VisDrone | F1 Mean | LocateAnything-3B（当前最强 58.7/39.9） |
| 闭集保持 | COCO | mAP50-95 + T4/4090 延迟 | YOLO26-N/S、YOLOE-26 |
| 拒识能力 | WeRef-Bench 风格自建集 | Rejection Rate | WeVisOne-VL（96.4%） |
| 零样本迁移 | 下游 LoRA 微调 | mAP | 现有 YOLO-PEFT 基线 |

## Consequences

### 收益

- 开放词汇能力以零推理开销进入 YOLO-Master，守住实时性护城河，与 2B~4B VLM grounding 模型形成代差级延迟优势；
- Prompt 条件化路由形成区别于 YOLOE-26 的独立学术叙事（MoE × 开放词汇）；
- VLM 教师管线可持续产出数据资产，并天然支持拒识负样本。

### 风险与对策

1. **YOLOE-26 先发撞车** → 不打"YOLO26+开放词汇"叙事，打"MoE 条件化计算 × 开放词汇"与数据引擎；精度以 N/S 档效率 Pareto 立论；
2. **MoE 与 one-to-one 分配冲突** → S1 充分验证；balance loss 退火 + 路由 warmup；必要时 MoE 只放 backbone；
3. **伪标签噪声上限** → 多教师投票 + CLIP 过滤 + 软标签兜底；先 100 万张验证收益再扩千万级；
4. **重参数化与动态路由矛盾** → route-once-per-vocabulary 固化模式，导出 ONNX/TensorRT 前完成路由固化 + RepRTA 折叠，保证部署图为纯静态。

## 里程碑

- **M1（2 周）**：教师 serving + 标注管线跑通，首批 10 万张伪标签人工抽检合格率 ≥90%；
  - 进度（2026-11-08）：管线四步（concepts/annotate/clean/convert）+ `smoke` 全部落地，
    coco128 双教师（Rex-Omni-3B + Qwen2.5-VL-3B）真实验证通过，见 `docs/ov_distill_guide.md` §6；
- **M2（4 周）**：S1 完成，YOLO26+ES-MoE 闭集 COCO 复现达标；
  - 进度（2026-11-08）：底座盘点发现 YOLO26 迁移已大半完成，组合配置
    `yoloe-26-master-n.yaml` 前向 + 1-epoch grounding 训练冒烟通过，剩余闭集基线复现
    与 S2 解耦可并行，见 `docs/ov_s1_base_analysis.md`；
- **M3（8 周）**：S2 完成，LVIS minival 文本提示 AP 达 YOLOE-26 同档位 ±1 AP；
  - 进度（2026-11-09）：数据池放大链路验证完成——ModelScope `PAI/COCO2017` 解决图像来源
    （~16 MB/s），val2017 千图四教师标注（19672 记录、0 失败），投票融合 F1 0.712 /
    拒识 94.3%（对 GT 复评）；S2 训练配方在 MPS 打通（AdamW + 恒定 lr0=0.002，
    修复 warmup/lrf/optimizer=auto 三个 LR 坑），yolo26n 预训练权重部分迁移（587/1278）
    显著加速收敛；千图冒烟 mAP 仍为千分位，精度达标需 GPU + 全量数据池长训。
    见 `docs/ov_distill_guide.md` §7；
  - 进度（2026-11-10）：**val2017 全量 4952 图三教师标注完成**（78538 记录、0 失败、
    0 重复；弃用 F1 0.47 的 QwenVL，WeVisOne/GroundingPI/Rex-Omni 单进程轮转 ~40 块）；
    全量清洗 33462 框（high 19056 + medium 14406）+ 9983 负样本，对 GT 复评
    **F1 0.720 / 拒识 96.6%**（优于千图口径）；S2 再训练 4 epoch（q1-q16 分块接力）
    val1k mAP50 **0.0026→0.0134**（千图版同轮次的 5.6 倍，曲线仍在爬升）。
    新坑：timeout 杀块=整块白训（仅 epoch 末落盘）、`last_healthy.pt` 不含中途进度。
    见 `docs/ov_distill_guide.md` §8；
  - 进度（2026-11-10 晚）：链路三处完善——①train_s2 `--save-steps` 步级断点
    （修复"timeout 杀块=整块白训"，实测可接力）；②CLIP 复核步骤落地
    （`clip-verify`，W-CLIP-Base ONNX + GT 自动阈值校准；实测 W-CLIP 区分度
    太弱不宜硬过滤，clip_score 改作元数据产出）；③**OV 真·留出评估工具**
    （seen60/unseen20 划分 + unseen-only 验证集自动生成）——关键发现：
    当前规模下 OV 能力几乎全部来自伪标签覆盖，文本迁移未涌现，
    真·留出 mAP 定为后续放大的北极星指标。seen60 链续训至 4 epoch
    与全量链对齐后：真·留出 unseen mAP50 0.0006→0.0031 单调上行
    （4 epoch 翻 5 倍，嵌入空间已带少量可迁移信号），但仍仅为同轮次
    伪留出 0.0190 的 1/6，远未跨过可用阈值。见 `docs/ov_distill_guide.md` §9；
  - 进度（2026-11-11）：收敛性门控实验——①s5k 链续训至 12 epoch：
    恒定 lr0=0.002 在 epoch 5-7 出现平台（~0.0145），阶梯衰减
    （0.0005→0.0002）后 val1k mAP50 翻倍至 **0.0289**，伪留出 unseen
    mAP50 同步翻倍至 **0.0385**——平台是 LR 所致而非数据上限；
    ②GT oracle 对照链（同图 GT，4 epoch）mAP50 仅 0.0023，落后伪标签
    链 6 倍（伪标签是文本条件化的干净易学目标，起飞更快；GT 长尾难框
    在 384px 下学习慢但在爬升）。③imgsz 640 接力 1 epoch 实测**负收益**
    （闭集 0.0264 / 伪留出同子集 0.0304，均低于 384 的 0.0289/0.0445，
    切换适应成本 1 epoch 收不回，且本机吞吐降至 1/4）——**结论：当前
    瓶颈是训练配方而非伪标签质量；数据池放大阶段维持 384，LR 阶梯衰减
    已固化（+100%），640 留待长 schedule 或 GPU 机器**。高负载期改
    半分片/32 分片分块保证 300s 窗口内完块。见指南 §9.4；
  - 进度（2026-11-11 深夜）：**数据池放大 5K→20K 启动**——train2017
    15K 新图下载就位（与 val1k/留出集图像级不相交），manifest
    （43678 正 + 30000 负 query）+ 15 千图片段 + 编排驱动
    `drive_20k.py`（片段×教师轮转选块、done-key 续跑、单教师子进程），
    实测 WeVisOne ~1.9 记录/秒，全量四教师预估 90+ 小时机时，
    拟以定时任务无人值守驱动。见指南 §10；
- **M4（12 周）**：S3+S4 完成，prompt 条件化路由消融 + 拒识指标 + 下游 LoRA 验证，形成完整论文实验矩阵。
