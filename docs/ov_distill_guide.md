# OV-Distill 数据管线使用指南

> 对应设计文档：ADR-0007（`docs/adr/0007-yolo26-open-vocabulary-evolution.md`）
> 代码位置：`scripts/ov_distill/` ｜ 测试：`tests/test_ov_distill.py`

用 VLM grounding 教师（Rex-Omni / WeVisOne-VL / GroundingPI）对无标注图像做开放词汇
自动标注，经多教师交叉验证清洗后，产出 YOLO grounding 训练集与拒识负样本集。

## 管线总览

```
图像目录 ──► concepts（概念发现，生成 manifest）
         ──► annotate（多教师批量标注，断点续跑）
         ──► clean（IoU 投票清洗 + 置信度分桶 + 负样本提取）
         ──► convert（YOLO grounding 训练格式）
```

四段均为独立子命令，中间产物全部是 JSONL，可逐段检查、断点续跑。

## 1. 环境准备

### 1.1 管线本机（CPU 即可）

```bash
pip install -e .   # 仓库根目录，管线仅依赖 Pillow + 标准库
python scripts/ov_distill/run_pipeline.py smoke   # 端到端自检（无需 GPU）
```

### 1.2 教师部署（GPU 机器，H 系列卡）

三种教师任选其二以上（投票需要 ≥2 个独立教师）：

**GroundingPI（GAM 协议，vLLM serving）**

```bash
git clone https://github.com/groundingpi/GroundingPI.git && cd GroundingPI
python3 -m pip install -r requirements.txt huggingface_hub
hf download GroundingPI/GroundingPI --local-dir weights/vlm
python3 run.py setup serve --platform gpu
python3 run.py serve                     # 服务地址 http://127.0.0.1:8000/v1
```

**Rex-Omni（本地权重）**

```bash
git clone https://github.com/IDEA-Research/Rex-Omni.git && cd Rex-Omni
conda create -n rexomni python=3.10 -y && conda activate rexomni
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && pip install -v -e .
```

**WeVisOne-VL（本地权重）**

```bash
git clone https://github.com/Tencent/WeVisOne-VL.git && cd WeVisOne-VL
conda create -n wevisone-vl python=3.10 -y && conda activate wevisone-vl
pip install -r requirements.txt && pip install -v -e .
```

**Qwen2.5-VL-Instruct（通用 VLM，备选教师）**

本仓库 `qwenvl:` 规格支持本地直连（Apple Silicon MPS 可用，ModelScope 下载快）：

```bash
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct
# 教师规格：qwenvl:<模型路径>
```

## 2. 四步操作

### Step 1 · 概念发现：图像目录 → manifest

```bash
# 方式 A：已知目标域，用固定词表（每行一个短语）
python scripts/ov_distill/run_pipeline.py concepts \
    --images data/ov_pool/images \
    --tagger file:data/ov_pool/vocab.txt \
    --global-vocab data/ov_pool/vocab.txt \
    --neg-per-image 2 \
    --out data/ov_pool/manifest.jsonl

# 方式 B：让 VLM 逐图发现概念（需先起 GroundingPI 服务）
python scripts/ov_distill/run_pipeline.py concepts \
    --images data/ov_pool/images \
    --tagger vlm:http://127.0.0.1:8000/v1:groundingpi \
    --out data/ov_pool/manifest.jsonl
```

`--neg-per-image N` 会从全局词表采样该图未命中的 query 作为负样本（拒识训练数据）；
`--max-queries`（默认 32）限制每图 query 数。

### Step 2 · 多教师批量标注

```bash
python scripts/ov_distill/run_pipeline.py annotate \
    --manifest data/ov_pool/manifest.jsonl \
    --teachers gam:http://127.0.0.1:8000/v1:groundingpi,rexomni:IDEA-Research/Rex-Omni \
    --out runs/ov_distill/raw.jsonl
```

- 断点续跑：中断后重复执行同一命令，已完成条目自动跳过，失败条目自动重试；
- 单教师失败不中断整批，失败记录带 `error` 字段；
- 多卡扩容：把 manifest 切成 N 份，每卡一个服务副本 + 一个输出文件，清洗前 `cat` 合并。

### Step 3 · 交叉验证清洗

```bash
python scripts/ov_distill/run_pipeline.py clean \
    --raw runs/ov_distill/raw.jsonl \
    --out runs/ov_distill/cleaned.jsonl \
    --iou 0.5 --min-votes 2
```

清洗规则：跨教师框 IoU ≥ 0.5 聚类 → 簇内教师数 ≥ 2 才保留 → 均值融合；
全票且簇内平均 IoU ≥ 0.7 记 `high` 桶，其余保留项记 `medium` 桶；
全体教师均判不存在的 (image, query) 写入 `negatives.jsonl`。

### Step 4 · 转训练格式

```bash
python scripts/ov_distill/run_pipeline.py convert \
    --cleaned runs/ov_distill/cleaned.jsonl \
    --out data/ov_distill_v1 --copy-images
```

产出：

```
data/ov_distill_v1/
├── images/            # --copy-images 时复制
├── labels/*.txt       # class_id x_center y_center w h（归一化）
├── categories.json    # 动态词表 [{"id": 0, "name": "person"}, ...]
└── grounding.jsonl    # 每行 {image, phrases: [{text, box_xywh_norm, bucket}]}
```

**训练 YOLOE/GroundingDataset 直读格式**（`--format grounding`）：

```bash
python scripts/ov_distill/run_pipeline.py convert \
    --cleaned runs/ov_distill/cleaned.jsonl \
    --out data/ov_distill_v1_grounding \
    --format grounding \
    --image-root /path/to/images   # file_name 相对此根书写，应与训练 img_path 一致
```

产出 `grounding_train.json`（GoldG 风格单 JSON：`images[].caption` +
`annotations[].tokens_positive` 字符区间，bbox 为绝对像素 xywh）。
负样本无需写入——训练端 `RandomLoadText` 自动按词表采样负样本。
可用 `--buckets high` 只保留高置信框（硬标签），medium 桶留给软标签蒸馏。

## 2.5 训练链路冒烟（S2 接线验证）

```bash
# 真实编码器（推荐）：mobileclip2_b.ts 放 weights/，clip-anytorch + setuptools<81 已装
python scripts/ov_distill/smoke_train_yoloe_master.py --epochs 3
# 离线 CI：确定性伪编码器
python scripts/ov_distill/smoke_train_yoloe_master.py --stub-text
```

`YOLOETrainerFromScratch` × `yoloe-26-master-n.yaml`（A2C2fMoE backbone + YOLOEDetect）
× coco128 grounding 伪标签（默认 `dataset_grounding4`，`--json-file` 可换）。
已验证（2026-11-08，真实 mobileclip2:b 编码器）：模型构建 7.60M 参数、
GroundingDataset 实载、文本嵌入缓存（`text_embeddings_mobileclip2_b.pt`）、
MoE router 参数组（lr=0.5x）、3 epoch 训练 + 文本 prompt val + checkpoint 落盘
（best.pt 16MB）、`set_classes + get_text_pe + predict` 开放词汇推理链路通畅。
注：玩具预算（imgsz=64、十几 epoch、126 图）下检测框质量无意义，
真实 OV 能力需按 S2 正式配方（imgsz 640、足量数据与训练量）验证。

## 3. 首批 1 万张验收清单（M1 出口标准）

| 检查项 | 方法 | 通过标准 |
|---|---|---|
| 标注成功率 | `annotate` 输出的 `failed / records` | < 2% |
| 抽检合格率 | 从 `cleaned.jsonl` 随机抽 200 框人工目检 | ≥ 90% |
| 负样本有效率 | 抽 50 条 `negatives.jsonl` 目检图中确实无该物体 | ≥ 95% |
| 教师一致性分布 | 统计 `high` 桶占比 | 50%~90%（过低说明教师分歧大，过高说明教师同源） |
| 词表覆盖 | `categories.json` 类别数 vs 预期目标域 | 符合场景预期 |

## 4. 常见问题

- **Q：能用单教师跑吗？** 可以，`--min-votes 1`，但失去交叉验证意义，只建议调通流程时用。
- **Q：manifest 里的相对路径相对谁？** 优先相对 manifest 所在目录解析，不存在则回退到当前工作目录；`concepts` 生成的 manifest 统一写绝对路径。
- **Q：mock 教师是什么？** 确定性伪标注（哈希生成伪框），用于 CI 和无 GPU 环境验证管线逻辑；`zz_absent` 前缀的 query 会被 mock 教师模拟为拒识。
- **Q：如何加新教师？** 继承 `scripts/ov_distill/teachers.py` 的 `TeacherClient`，实现 `detect()` 并在 `build_teacher()` 注册规格前缀。

## 5. 本地验证

```bash
python scripts/ov_distill/run_pipeline.py smoke --out runs/ov_distill_smoke
pytest tests/test_ov_distill.py -v
```

## 6. coco128 真实教师验证记录（2026-11-08，Rex-Omni-3B @ M3 Max MPS）

**环境**：`.venv-ov`（Python 3.12 + torch 2.9 + transformers 4.51.3 + rex-omni 1.0.0，
bf16/sdpa/MPS）；权重经 ModelScope 下载（HF 镜像过慢）。单图约 10~23 s。

**流程**：`make_coco128_manifest.py` 从 GT 生成 126 张图的 manifest（362 个正样本 query
+ 252 个负样本 query）→ `annotate --teachers rexomni:<path>`（614 条记录，0 失败，
断点续跑分 5 批完成）→ `clean --min-votes 1` → `convert` → `eval_coco128.py` 对 GT 评估。

**结果（IoU ≥ 0.5，贪心匹配）**：

| 指标 | 数值 | 说明 |
|---|---|---|
| 正样本 Precision | 0.768 | 部分 FP 源于 COCO GT 本身标注不全（如热狗/水果误标实际合理） |
| 正样本 Recall | 0.632 | FN 集中在小目标/密集类（book、fork、spoon、traffic light） |
| 正样本 F1 | 0.693 | 3B 教师 + 裸类别 query 的正常水位 |
| **负样本拒识率** | **0.964** | 252 条不存在类别 query，243 条正确返回空框 |
| 失败率 | 0% | 614 条记录全部成功 |

**结论**：管线在真实教师下端到端可用；拒识能力优秀（96.4%，与 WeVisOne-VL 官方
WeRef-Bench 口径同水位）；recall 偏低主要受单教师 + 小目标影响，正是 ADR-0007
设计多教师投票与 CLIP 过滤要解决的问题——引入第二个教师（WeVisOne-2B）后可复评。

### 双教师投票复评（同日补测，Rex-Omni-3B + Qwen2.5-VL-3B-Instruct）

> 注：WeVisOne-VL-2B 权重当时仅 HF 有源，本机网络下载实测不可行（镜像 ~320KB/s），
> 故以 ModelScope 可高速下载的 Qwen2.5-VL-3B-Instruct 作为第二教师替代，
> 管线侧新增 `qwenvl:` 教师规格（含缩放坐标系回原图映射 + 截断容错解析）。
> **当日晚间 WeVisOne-VL-2B 已上架 ModelScope（isLinXu/WeVisOne-VL-2B 镜像），
> 第三教师补测见下节。**

三方案对比（IoU ≥ 0.5，贪心匹配，同一 manifest / 同一 GT）：

| 方案 | Precision | Recall | F1 | 负样本拒识率 |
|---|---|---|---|---|
| Rex-Omni-3B 单教师 | 0.768 | 0.632 | 0.693 | 96.4% |
| Qwen2.5-VL-3B 单教师 | 0.528 | 0.435 | 0.477 | 90.5% |
| **双教师投票（min-votes 2）** | **0.929** | 0.393 | 0.552 | **99.2%** |

**结论**：
1. 投票机制按设计意图生效——**精度 0.77→0.93，拒识率 96.4%→99.2%**，伪标签噪声被大幅压制；
2. Recall 下降是交集投票的固有代价，验证了 ADR-0007 的分桶设计必要性：双教师一致框（high 桶）
   做硬标签，单教师独有框（medium 桶）做软标签/待检池，而不是直接丢弃；
3. 专业 grounding 教师（Rex-Omni）显著强于通用 VLM（Qwen2.5-VL-Instruct），教师选型比数量更关键。

**产物位置**：`runs/ov_distill_coco128/`（raw.jsonl 双教师 1228 条记录 / cleaned_voted.jsonl /
negatives_voted.jsonl / dataset_voted/ / viz/ 可视化抽检）。
复评命令：`eval_coco128.py --raw <raw或cleaned> --teacher <名称>`（可选教师过滤）。

### 三教师复评（同日晚补测，+ WeVisOne-VL-2B @ ModelScope 镜像）

**环境差异**：WeVisOne-VL-2B 基于 Qwen3-VL，需 transformers ≥ 4.57，与 Rex-Omni
所需的 4.51.3 冲突——WeVisOne 在 dev 环境（transformers 5.21 + torch 2.14，MPS）
单独跑 `annotate`（断点续跑按 image+teacher+query 键去重，可安全向同一 raw 追加；
注意 `device_map="mps"` 直载会段错误，客户端已改为 CPU 加载后 `.to("mps")`）。
`wevisone:` 客户端不依赖官方包，transformers 直载权重 + 官方 prompt/解析协议
（`<ref>label</ref><box><x1><y1><x2><y2></box>`，0-1000 归一化坐标 token，
逐 query `none` 拒识）。单图约 1.2~2.4 s（明显快于 Rex-Omni 的 10~23 s）。

四方案对比（IoU ≥ 0.5，贪心匹配，同一 manifest / 同一 GT）：

| 方案 | Precision | Recall | F1 | 负样本拒识率 |
|---|---|---|---|---|
| Rex-Omni-3B 单教师 | 0.768 | 0.632 | 0.693 | 96.4% |
| **WeVisOne-VL-2B 单教师** | **0.759** | **0.684** | **0.719** | 94.1% |
| 双教师投票（Rex+QwenVL，min-votes 2） | 0.929 | 0.393 | 0.552 | 99.2% |
| **三教师投票（min-votes 2）** | 0.850 | **0.620** | **0.717** | 97.6% |
| 三教师全票（high 桶，votes=3） | **0.973** | 0.345 | 0.509 | **99.6%** |

**结论**：
1. WeVisOne-VL-2B 是最强单教师（F1 0.719），且推理速度比 Rex-Omni 快约 8 倍；
2. 三教师投票取得最佳精度/召回平衡（P 0.85 / R 0.62）：两个强教师互证即可保留，
   不再被最弱教师拖累；
3. **high 桶（三教师全票）P=0.973、拒识 99.6%**——可直接作硬标签；medium 桶
   （2/3 票）354 框作软标签，分桶策略在真实三教师下完全成立。

**产物位置**：`runs/ov_distill_coco128/raw3.jsonl`（三教师 1842 条）/
`cleaned3_voted.jsonl`（684 框：high 330 + medium 354）/ `negatives3.jsonl`（218 条）/
`dataset_grounding3/`（grounding 训练格式，126 图 / 684 框 / 70 类）。

### 四教师复评（同日晚终测，+ GroundingPI-4B @ ModelScope）

**接入方式**：GroundingPI 是自定义架构（`GroundingPIForConditionalGeneration`，
MoonViT3d 视觉塔，Kimi K3 技术血统），权重快照自带完整远程代码——`groundingpi:`
客户端用 `trust_remote_code=True` 直载（dev 环境 transformers 5.21 ≥ 要求的 5.7），
无需官方仓库的 bundled transformers fork。协议即 GAM：dense prompt 多标签以
`</c>` 分隔，输出 `<|object_ref_start|>...<|box_start|><0-999 网格><|box_end|>`，
新增 `parse_gam_multi` 多标签解析。单图约 12 s（MPS）。**注意**：官方协议虽有
`None` 拒识载荷，但实测 GroundingPI 在 coco128 负样本 query 上**拒识率 0%**
（全部幻觉出框）——为此 clean 的负样本规则从"全教师 absent"放宽为
"无框通过投票且 absent 教师数 ≥ n-1"（`--neg-min-absent` 可调），幻觉孤票由
min-votes ≥ 2 天然淘汰。

五方案对比（IoU ≥ 0.5，贪心匹配，同一 manifest / 同一 GT）：

| 方案 | Precision | Recall | F1 | 负样本拒识率 |
|---|---|---|---|---|
| Rex-Omni-3B 单教师 | 0.768 | 0.632 | 0.693 | 96.4% |
| WeVisOne-VL-2B 单教师 | 0.759 | 0.684 | 0.719 | 94.1% |
| Qwen2.5-VL-3B 单教师 | 0.528 | 0.435 | 0.477 | 90.5% |
| GroundingPI-4B 单教师 | 0.683 | **0.701** | 0.692 | **0%** |
| **四教师投票（min-votes 2）** | 0.794 | **0.699** | **0.743** | 92.9% |
| **四教师全票（high 桶，votes=4）** | **0.984** | 0.335 | 0.500 | **99.6%** |

**结论**：
1. 教师性格画像齐全了：WeVisOne 最均衡最快（F1 0.719，2 s/图）、Rex-Omni 拒识最稳、
   GroundingPI 召回最高但零拒识（幻觉型教师）、QwenVL 通用兜底；
2. 四教师投票拿到全场最佳 F1（0.743）与召回（0.699）——幻觉型教师在投票框架下
   是召回增益而非噪声源；
3. **high 桶（四票全一致）P=0.984 / 拒识 99.6%**，837 框中 317 框硬标签、520 框软标签，
   分桶-蒸馏策略的数据侧证据链完整闭合；
4. 工程教训：`timeout` 杀进程可能留下同主键重复记录（写缓冲与续跑窗口竞争），
   评估前先按 (image, teacher, query) 去重（保留首个 ok 记录）。

**产物位置**：`runs/ov_distill_coco128/raw4.jsonl`（四教师 2456 条，已去重）/
`cleaned4_voted.jsonl`（837 框：high 317 + medium 520）/ `negatives4.jsonl`（218 条）/
`dataset_grounding4/`（grounding 训练格式，126 图 / 837 框 / 71 类）。


## 7. S2 数据池放大：val2017 千图级四教师标注与预训练冒烟（2026-11-09）

### 7.1 图像池来源（网络约束下的选型）

COCO 官网 `images.cocodataset.org` 本机直连仅 ~8 KB/s（SSL 疑似被中间人干预，需 `-k`），
1 GB 的 val2017.zip 直连不可行；HF 镜像 ~320 KB/s 也太慢。最终方案：

- **ModelScope 数据集 `PAI/COCO2017`**（~16 MB/s）：
  `modelscope download --dataset PAI/COCO2017 val2017.zip annotations_trainval2017.zip`
  （CLI 在 dev 环境 `~/miniconda3/envs/dev/bin/modelscope`；815 MB 约 52 s）
- val2017.zip 的 sha256 与官方一致（`4f7e2ccb…82f05`），5000 图 + instances_val2017.json 完整
- 解压布局：`datasets/coco/val2017/` + `datasets/coco/annotations/`

### 7.2 manifest 与标注规模

`make_val2017_manifest.py` 从 instances_val2017.json 直读 GT（跳过 iscrowd），
每图 query = 正样本类别 + 2 个 COCO 词表负样本，`--max-images 1000 --seed 0` 采样子集：

- **1000 图 / 4918 query（2918 正 + 2000 负）**
- 四教师标注（分块续跑，每块 ≤275 s，约 100 块、累计 ~7.5 h MPS）：
  - 批次A（dev 环境）：WeVisOne-VL-2B + GroundingPI-4B，~20 图/块
  - 批次B（.venv-ov，transformers 4.51.3）：Rex-Omni-3B + Qwen2.5-VL-3B，~90 记录/块/教师
- 产出 `runs/ov_s2_val1k/raw_dedup.jsonl`：**19672 条全 ok，0 失败**（含负样本拒识记录）

### 7.3 千图级标注质量复评（IoU ≥ 0.5，对 val2017 GT）

GT 由 instances_val2017.json 转出 YOLO 格式（`runs/ov_s2_val1k/gt_labels/`，
类别顺序已断言与 COCO_NAMES 一致）：

| 方案 | Precision | Recall | F1 | 负样本拒识率 |
|---|---|---|---|---|
| WeVisOne-VL-2B | **0.647** | 0.673 | **0.660** | 96.9% |
| GroundingPI-4B | 0.606 | **0.715** | 0.656 | 0.2% |
| Rex-Omni-3B | 0.609 | 0.631 | 0.620 | **97.1%** |
| Qwen2.5-VL-3B | 0.469 | 0.470 | 0.470 | 90.1% |
| **四教师投票（min-votes 2）** | **0.729** | **0.696** | **0.712** | 94.3% |

与 coco128 小样本结论一致：投票融合 F1 最优；GroundingPI 仍是幻觉型教师（拒识 ~0），
WeVisOne 取代 Rex-Omni 成为单教师 F1 最佳。千图级精度整体低于 coco128（场景更复杂、
单图类别更多），属预期。

### 7.4 S2 训练集产物

- `cleaned.jsonl`：**7176 框（high 2881 + medium 4295）**，负样本 1800 条，dropped 11970
- `dataset_grounding/grounding_train.json`：1000 图 / 7176 框 / 80 类（GoldG 格式，
  `tokens_positive` 字符区间；caption 只保留存活类别）
- `val1k.yaml` + `val/`（images 软链 + labels）：闭集验证用

### 7.5 S2 训练冒烟（yoloe-26-master-n @ MPS）

配方脚本 `train_s2.py`（分块接力，`--init-from` 权重传递；`resume=True` 对 dict 形式
data 会崩，故不用官方 resume）。imgsz 384 / batch 8 / MPS ≈ **1.15 it/s，~2.5 min/epoch**
（MoE `index_add_` 在 MPS 可跑，仅有非确定性警告）。

**踩到的三个 LR 坑（均已修）**：
1. 每块 epochs=1 时默认 warmup_epochs=3 → 整块困在 warmup 低 LR（~4e-5），7 epoch 白练；
2. warmup=0 后 lrf 默认 0.01 → 单块内线性衰减到接近 0，等效 LR 依然过低；
3. `optimizer=auto` 按 `lr_fit = 0.002*5/(4+nc)` 重拟合（nc=80 → **1.19e-4**），
   直接忽略 `--lr0`——必须显式 `optimizer=AdamW`。

修正后配方：`optimizer=AdamW, lr0=0.002, lrf=1.0（恒定）, warmup_epochs=0`。

**收敛证据**（每条链为逐块接力）：

| 链 | 初始化 | epoch | box | cls | dfl | val1k mAP50 |
|---|---|---|---|---|---|---|
| s2d | 从零 | 1→8 | 3.58→2.93 | 5.59→4.56 | 0.045→0.031 | 0.0011 @e11 |
| s2p | yolo26n.pt 部分迁移（587/1278 张量） | 1→4 | 2.83→2.28 | 5.47→4.49 | 0.031→0.022 | 0.0024 @e4 |

- 预训练初始化收敛显著更快（同轮次 box 低 ~0.7），`model.load()` 部分匹配可用；
- mAP 仍为千分位：1000 图 × 个位数 epoch 对从零 OV 检测头只是冒烟量级
  （YOLOE 官方 from-scratch 配方为 ~1.7M 图 × 300 epoch）。**链路正确性已验证，
  精度需 GPU + 全量数据池 + 长训**。OV 推理链路（`set_classes/get_text_pe/predict`
  自定义类别）在训练后权重上机制正常。

**产物位置**：`runs/ov_s2_val1k/`（manifest/raw_dedup/cleaned/negatives/dataset_grounding/
gt_labels/val1k.yaml/val/）、`runs/ov_s2_train/`（s2d_*、s2p_* 各块与 val）、
`weights/yolo26n.pt`（gh-proxy.com 代理下载，5.5 MB）。

## 8. S2 数据池全量放大：val2017 全量 4952 图三教师标注与再训练（2026-11-10）

### 8.1 标注规模与轮转策略

`make_val2017_manifest.py --max-images 0 --seed 0` 生成全量 manifest：
**4952 图（48 张无正样本被过滤）/ 24535 query（14631 正 + 9904 负）**。
val1k 的 1000 图 query 与全量 manifest 前段完全一致（已逐条校验），故以 val1k 的
`raw_dedup.jsonl`（19672 条）为底续跑，零浪费。

- **教师缩编为三家**：Qwen2.5-VL-3B 在 val1k 复评中 F1 仅 0.470（其他家 0.62+），
  5K 阶段弃跑；其历史记录保留但不参与投票（清洗时显式 `--neg-min-absent 2`
  固定负样本口径，不用 auto，避免前 1000 图四教师记录改变阈值语义）。
- **单教师单进程轮转**（Rex-Omni → GroundingPI → WeVisOne，落后者优先）：
  双模型同进程在内存压力下会互相挤兑（128G 机器一度 121G 已用），必须串行。
  速率随机器负载波动大：WeVisOne 50-200 图/块、GroundingPI 20-90、Rex-Omni 25-90。
- 产出 `runs/ov_s2_5k/raw_dedup.jsonl`：**78538 条全 ok、0 失败、0 重复**
  （per-query 断点续跑键值去重验证通过），三教师均满 4952 图。

### 8.2 全量清洗与质量复评（IoU ≥ 0.5，对 val2017 GT）

GT 标签由 instances_val2017.json 转出 YOLO 格式至 `runs/ov_s2_5k/gt_labels/`
（4952 文件，类别顺序断言 == COCO_NAMES）。

清洗：`--iou 0.5 --min-votes 2 --neg-min-absent 2` →
**cleaned.jsonl：33462 框（high 19056 + medium 14406）+ negatives 9983 条，dropped 45593**。

| 口径 | Precision | Recall | F1 | 负样本拒识率 |
|---|---|---|---|---|
| val1k 千图（四教师投票） | 0.729 | 0.696 | 0.712 | 94.3% |
| **5K 全量（三教师投票）** | **0.756** | 0.687 | **0.720** | **96.6%** |

全量口径与千图口径一致甚至略优（精度 +3pt、拒识 +2.3pt），
说明 val1k 的 1000 图采样有代表性，且砍掉 QwenVL 未伤融合质量。
`missing_records=10606` 为投票后整类被过滤的 query 数，按空预测计入 FN，属预期行为。

### 8.3 训练集产物与再训练

- `dataset_grounding/grounding_train.json`：**4945 图 / 33462 框 / 80 类**
  （7 张图框全被投票过滤）；为适配 Bash 单块 280 s 上限切成 4 份
  （q1-q4，各 ~1237 图 / ~155 step，~3.5 min/块）。
- **训练链 s5k_q1→q16（4 个完整 epoch）**：yolo26n.pt 初始化，
  AdamW + lr0=0.002 + lrf=1.0 + warmup=0，imgsz 384 / batch 8 / MPS。

| epoch | 块 | val1k mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| 1 | q1-q4 | 0.0026 | 0.0011 | 0.007 | 0.173 |
| 2 | q5-q8 | 0.0065 | 0.0031 | 0.161 | 0.032 |
| 3 | q9-q12 | 0.0110 | 0.0050 | 0.291 | 0.039 |
| 4 | q13-q16 | **0.0134** | **0.0066** | 0.193 | 0.050 |

对比千图版 s2p 链（4 epoch → mAP50 0.0024）：**5K 数据 1 个 epoch 即追平，
4 epoch 达到 5.6 倍**，且曲线仍在爬升——数据池放大收益明确。

### 8.4 新踩到的坑

- **timeout 杀块 = 整块白训**：ultralytics 只在 epoch 末保存权重，块被 timeout
  打断时不落盘。q7 曾在 92%（143/155 步）被杀。对策：分块要留足负载余量
  （155 步 @ 波动期 1.3-1.9 s/it 已贴线），机器高负载时段宁可再切小。
- `last_healthy.pt` 只是训练启动时 bootstrap 的初始化权重副本，
  **不含块内中途进度**，不能当作断点续训依据。

**产物位置**：`runs/ov_s2_5k/`（manifest/raw.jsonl/raw_dedup/cleaned/negatives/
gt_labels/dataset_grounding）、`runs/ov_s2_train/s5k_q*`（16 块）与
`val_s5k_q4/q8/q12/q16`（各轮验证）。

## 9. 链路完善：步级断点 + CLIP 复核 + OV 真·留出评估（2026-11-10）

### 9.1 步级断点保存（train_s2.py `--save-steps N`）

每 N 个训练 step 经 `on_train_batch_end` 回调把当前权重序列化到
`weights/last_step.pt`（完整 ultralytics checkpoint，序列化失败不阻断训练）。
块被 timeout 打断时用 `--init-from <块目录>/weights/last_step.pt` 接力，
损失从"整块"降为"最近 N 步"。已实测：150 s 杀块后 last_step.pt 落盘、
`model.load()` 全量 1278/1278 张量迁移成功。对比：`last_healthy.pt`
只是启动时的初始化副本（§8.4），不含中途进度。

### 9.2 CLIP 复核步骤（clip-verify，ADR Stage 3.5 落地）

新管线步骤 `run_pipeline.py clip-verify`（实现 `clip_verify.py`）：
裁框 → W-CLIP-Base（本地 ONNX 双塔，onnxruntime CPU）图像-文本余弦相似度
（cos×100，CLIP logits 口径），低分丢弃；分数缓存续跑
（`cache.jsonl`，框级键容忍浮点抖动）；阈值用 GT 自动校准
（保留 95% 真框的 5% 分位数），不用拍脑袋。

**5K 实测（medium 桶 14406 框）**：
- 校准阈值 62.392；真框均分 67.2 vs 假框 66.8 —— **W-CLIP 区分度仅 +0.43**
- 硬过滤版 `cleaned_verified.jsonl`：丢 735 框（5.1%），
  P 0.756→0.759、拒识 96.6%→97.2%，但 R 0.687→0.677，**F1 0.720→0.716 略降**
- 结论：W-CLIP-Base 对 COCO 裁剪框判别力太弱，硬过滤得不偿失；
  已产出全保留带分版 `cleaned_scored.jsonl`（clip_score 作元数据），
  供训练端质量加权/事后分析用。后续可换 openai ViT-L/14（Azure 直连
  ~10 KB/s 不可行，需镜像或 GPU 机器侧下载），编码器接口已插件化。

### 9.3 OV 真·留出评估（eval_ov_holdout.py）

闭集 mAP 只测"训过的类"，OV 的核心是"文本提示泛化到未见类"。
新工具把 COCO 80 类确定性划分 seen/unseen（默认 60/20，seed 可复现），
自动生成 unseen-only 验证集（标签重映射 + 图像软链 + yaml），
用 `set_classes(unseen)` 文本提示跑 val，输出 unseen mAP。

**关键实验（2026-11-10，两条链同配方分块接力至 4 epoch 对齐）**：

| epoch | s5k 伪留出 mAP50 | seen60 真·留出 mAP50 | s5k mAP50-95 | seen60 mAP50-95 |
|---|---|---|---|---|
| 1 | 0.0057 | 0.0006 | 0.0021 | 0.0001 |
| 2 | 0.0125 | 0.0010 | 0.0052 | 0.0005 |
| 3 | 0.0146 | 0.0024 | 0.0059 | 0.0009 |
| 4 | **0.0190** | **0.0031** | 0.0086 | 0.0013 |

- 伪留出（s5k 链）：unseen 类其实在训练标签里（全 80 类伪标签）；
- 真·留出（seen60 链）：训练标签剔除 unseen 20 类共 15260 框（余 18202 框），
  4 epoch 后 P≈0.004 / R≈0.28；
- 完整曲线图：`runs/ov_s2_5k/ov_holdout_curve.png`。

**结论：当前数据量级下，模型的"OV 能力"几乎全部来自伪标签覆盖，
文本迁移能力尚未涌现**。两条曲线同 epoch 对比：伪留出 4 epoch 达 0.0190
且仍在上升，真·留出仅 0.0031（约为伪留出的 1/6）——但真·留出曲线
单调上行（0.0006→0.0031，4 epoch 翻 5 倍），说明文本嵌入空间已开始
携带少量可迁移信号，只是远未跨过可用阈值。这是 S2 阶段最重要的标尺：
后续每轮数据池/epoch 放大，都应以真·留出 mAP 为北极星指标。
seen60 训练集与 unseen 验证集已就位：
`runs/ov_s2_5k/dataset_grounding_seen60/`、`runs/ov_s2_5k/ov_holdout*/`。

### 9.4 收敛性门控实验：LR 平台 + GT oracle（2026-11-11）

放大标注规模前先回答"瓶颈是数据还是配方"。两条对照链同图同配方
（4945 图、分块接力、AdamW、warmup=0、imgsz=384、batch=8、from scratch）：

**① s5k 伪标签链续训至 12 epoch（闭集 val1k mAP50）**：

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mAP50 | .0026 | .0065 | .0110 | .0134 | .0156 | .0142 | .0145 | **.0221** | **.0274** | .0277 | .0287 | **.0289** |

- epoch 5-7 在恒定 lr0=0.002 下出现平台（~0.0145）——**平台是 LR 造成的，
  不是数据上限**：epoch 8 起阶梯衰减（0.0005→0.0002）后 mAP 翻倍至 0.0289；
- 伪留出 unseen mAP50 同步翻倍：epoch 4 的 0.0190 → epoch 12 的 **0.0385**
  （`ov_holdout_s5k_q48/`），闭集改善直接转化为 OV 指标改善；
- 高负载期分块两次超时被杀：训练速度从 1.1 it/s 跌至 0.6 it/s，
  靠 `--save-steps` 保命后改用**半分片**（8 块/epoch，~77 步/块）跑完，
  分片文件 `grounding_train_q{1-4}{a,b}.json` 三个数据集均已预生成。

**② GT oracle 链（同图 GT 标签 36.3K 框，4 epoch）**：
mAP50 0.0001 → 0.0006 → 0.0013 → 0.0023，同轮次**落后伪标签链 6 倍**。
数据完整性已逐项排除（spans 校验 0 错误、crowd 仅 1.3% 且已剔除、
框面积分布相近）。机理：伪标签是"文本条件化"的干净易学习目标
（教师按 query 字符串产出、投票融合偏向大框），起飞更快；
GT 含全部小/难/遮挡框（<0.1% 相对面积占 12.9% vs 伪标签 4.9%），
在 384px 下学习慢但曲线仍在爬升。

**决策结论**：当前瓶颈是**训练配方**（恒定 LR 平台 + 分辨率 + 轮次），
不是伪标签质量。放大标注规模前应先固化配方：LR 阶梯衰减（已验证
+100%）、imgsz 384→640（46% GT 框相对面积 <1%，384px 下不可学）、
更长 schedule。配方固化后数据池放大才是高杠杆动作。

**③ imgsz 384→640 接力验证（2026-11-11，epoch 13，s5k_q48 接力 1 epoch）**：

| 指标（同口径） | 384 @ epoch12 | 640 接力 1 epoch |
|---|---|---|
| 闭集 val1k mAP50 | 0.0289 | 0.0264 |
| 伪留出 mAP50（同 1000 图子集） | 0.0445 | 0.0304 |

**分辨率切换在小预算下是负收益**：384 收敛模型切到 640 有适应成本
（P 0.23→0.15、R 上升，提案框增多但未校准），1 个 epoch 不足以收回；
且 640 在本机训练吞吐降至 1/4（batch 8 内存冲 19.9G 降速，需 batch 4 +
32 分片才能稳跑，1 epoch ≈ 1.6h）。**结论修正：数据池放大阶段保持
384 训练 + 384 评估；640 只在有完整长 schedule（或 GPU 机器）时再上**。
640 分块配方留档：batch 4 + `grounding_train_t{01..32}.json`（32 分片，
~39 步/块）+ `--save-steps 20`。

对比图：`runs/ov_s2_5k/convergence_comparison.png`；
oracle 数据集 `runs/ov_s2_5k/dataset_grounding_gt/`；
闭集评估工具 `scripts/ov_distill/eval_closed.py`。

### 9.5 产物与回归

- 新增：`clip_verify.py`、`eval_ov_holdout.py`、`eval_closed.py`；
  `train_s2.py` +`--save-steps`；管线新子命令 `clip-verify`
- 测试 32 → **38 项全绿**（缓存键/阈值校准/假编码器端到端/类划分/标签重映射）；
  ruff 通过

## 10. 数据池放大 5K → 20K：train2017 15K 扩展（2026-11-11 启动）

配方门控（§9.4）闭环后启动数据池放大。新池 = 既有 val2017 5K（已标注）
+ **train2017 15K 新图**（与评估集 val1k/留出集图像级不相交，协议更干净）。

### 10.1 数据就位

- 选图：`instances_train2017.json` 中有非 crowd 框的图像，seed 0 随机选
  15000 张，逐张从 images.cocodataset.org 下载（64 路并行 ~5.2 张/秒，
  共 2.7G，抽检 30 张全部有效、零空文件）→ `datasets/coco/train2017/`
- manifest：`make_val2017_manifest.py --instances instances_train2017.json`
  复用（同构脚本直接支持 train2017），15000 图 / 43678 正 query +
  30000 负 query → `runs/ov_s2_20k/manifest.jsonl`
- 切片：15 个千图片段 `manifest_s{01..15}.jsonl`——按片段轮转四教师，
  每个片段四教师齐后即可清洗出增量训练数据，不必等全量标完

### 10.2 标注编排（drive_20k.py）

- `scripts/ov_distill/drive_20k.py`：每次调用选出下一个未完成的
  （片段, 教师）对，子进程单教师跑一个 ~4 分钟块并退出；断点续跑由
  raw.jsonl 的 done-key 索引保证（键 = (image, 教师全名, query)）。
  教师顺序固定 wevisone → rexomni → qwenvl → groundingpi
- 实测吞吐（M3 Max MPS）：WeVisOne ~1.9 记录/秒（~2.3 s/图，含模型加载
  摊销）；预估全量四教师 ≈ 90+ 小时机时（GroundingPI 12 s/图为大头），
  适合无人值守定时驱动
- 测试：38 → **41 项全绿**（新增 pick_next 轮转/推进/完成三态）；ruff 通过

### 10.3 状态快照（启动日）

- slice 1 / wevisone：1829/4966（37%），其余教师待轮转
- 产出：`runs/ov_s2_20k/raw.jsonl`（持续增长中）
