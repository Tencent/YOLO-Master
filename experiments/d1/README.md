# D1：冻结 DINOv3 的缓存特征检测

负责人：冯焱琦（[@Frank95zz](https://github.com/Frank95zz)）。冻结 DINOv3，只训练多尺度 Adapter、LatentMixture 和检测头，研究检测精度保留率与训练成本。

阅读顺序：先看完成状态与方法，再看数据/缓存/训练复现，随后阅读已完成消融及最终对照，最后核对测试、版本归属与局限。本 README 是唯一的 D1 文档入口；四个 manifest 提供数据、权重、合同与官方评分工具校验，数据、Teacher 权重、缓存、checkpoint 和完整日志均放在仓库外。

## 课题目标与完成状态

本项目已完成 P0 冻结特征检测闭环和 P2 latent aux 消融。P1 已完成两数据集、两架构各三 seed 的精度/显存/时长对照，12/12 组固定末轮主结果及两数据集三 seed 统计已归档。

**结果更新（北京时间 2026-09-13）：** 12/12 组固定末轮结果、两数据集三 seed 统计、150 项 VisDrone 周期/内部 best 补充评分与成本分项已完成并归档。

| 任务 | 完成状态 | 已完成内容与证据 | 结果与交付 |
|---|---|---|---|
| P0：冻结特征检测闭环 | **已完成** | 多层 Teacher、可复现缓存、九分支 Adapter、LatentMixture、Detect、训练、评测、保存与严格重载 | COCO 检测结果、VisDrone 官方评分，以及最终入口四组真实六卡五轮验收 |
| P1：精度 / 显存 / 时长三维对照 | **两数据集三 seed 精度/显存/时长对照已完成** | 两数据集、两架构、三个 seed，共 12 次正式运行与官方评测已完成 | 按统一口径报告 AP、显存、GPU-hours、精度保留率及成本降低率，对照至少 50% 降本目标 |
| P2：latent aux 消融 | **已完成（消融路径）** | balance/z 的 27 次运行、gain 的 9 次新增运行，共 36 次独立训练 | 官方 MATLAB 评分、逐 seed 分析、关闭项验证和配置选择结论 |

P2 采用任务书中的消融路线，在 DINOv3 ViT-S/16 底座上完成 balance、z-loss 和 aux gain 扫描，给出各配置的精度、种子波动与选型依据。

正式配对实验在 PR 正文固定的代码版本上启动，执行 SHA、配方摘要和数据身份写入外部 plan.json 与 run.json。统一公共验收基线为 [acce839](https://github.com/Tencent/YOLO-Master/commit/acce839c7e895d6b179de7f7093fa879e237cc7b)；历史筛选和本轮对照分别记录预算与代码。

## 架构与实现

~~~text
RGB -> 固定 640 LetterBox -> 冻结 DINOv3 ViT-S/16
    -> block4 / block8 / block12：均为 [B,384,40,40]
    -> 每层分别适配到 P3/P4/P5，共九条独立分支
    -> 每个尺度的三路候选经 LatentMixture 融合
    -> YOLO26 Detect
~~~

- 三个 DINO block 均为 stride 16；Adapter 将其转换为 stride 8/16/32，输出通道为 64/128/256，网格为 80/40/20。
- 正式候选使用 BN64：P5 为 384->64->256 的 64 通道瓶颈分支，归一化采用 GroupNorm。
- Teacher 多层 API 为 DINOv3Teacher(output_layers=(4,8,12))；默认调用继续返回 dense["p4"]，兼容现有 Foundation 蒸馏消费者。
- Teacher 独立冻结并离线提取特征；student optimizer、DDP、EMA 和 checkpoint 仅管理下游模型。缓存与原图、标签和预处理合同一一对应。
- 复用既有 CompositeCriterion 和显式包含 latent 的 include_kinds，标量目标为 detection.sum()+aux；新增 D1 检测损失、raw/effective aux 报告及有限值验证，不改变收集器默认集合。
- weighted_sum 融合、可分离双线性 P3 和 foreach EMA 通过 D1 配方或入口显式启用，保留原实现供数值对照。
- 通用改动集中在 Trainer 扩展点、checkpoint 运行状态清理及辅助损失报告；Attention 的 FP32 分支默认关闭，旧对象未设置该属性时仍走原路径。

### 三种 P5 架构的区别

BASE、DW、BN64 是同一个冻结特征检测器的三种 P5 Adapter 变体，不是三个不同的 DINOv3，也不是三个不同的检测头。三者共享 Teacher、block4/8/12 缓存、P3/P4 分支、各尺度 LatentMixture 和 Detect；架构消融只替换三个来源各自的 P5 分支。

每个 block 都分别产生 P3、P4、P5 三个候选，而不是 block4 只接 P3、block8 只接 P4、block12 只接 P5。候选顺序始终为 block4、block8、block12。

| 共同分支 | 每个来源的计算 | 每个来源的输出 |
|---|---|---|
| P3 | 1×1 Conv(384→64) → GroupNorm → SiLU → bilinear 2× | [B,64,80,80] |
| P4 | 1×1 Conv(384→128) → GroupNorm → SiLU | [B,128,40,40] |
| P5 | 使用下述 BASE / DW / BN64 的一种 | [B,256,20,20] |

所有卷积均 bias=False，每个卷积后分别接 GroupNorm 和 SiLU；3×3 卷积 padding=1，1×1 卷积 padding=0。GroupNorm 使用 get_safe_groups(channels,8)，各来源、各尺度使用独立参数。正式提速配方中三者的 P3 统一采用 separable_bilinear2x，并保留原 bilinear 供数值对照。

#### BASE：普通卷积直接降采样

~~~text
[B,384,40,40]
  -> Conv 3×3, stride=2, groups=1, 384→256
  -> GroupNorm(256) -> SiLU
  -> [B,256,20,20]
~~~

一次普通卷积同时完成空间降采样和跨通道混合，每个输出通道连接全部 384 个输入通道。结构直接、表示能力不受中间瓶颈约束，但 P5 参数和卷积计算量最大。配置 p5_mode=conv，不设置 bottleneck_channels。

#### DW：先逐通道降采样，再混合通道

~~~text
[B,384,40,40]
  -> Depthwise Conv 3×3, stride=2, groups=384, 384→384
  -> GroupNorm(384) -> SiLU
  -> [B,384,20,20]
  -> Pointwise Conv 1×1, stride=1, groups=1, 384→256
  -> GroupNorm(256) -> SiLU
  -> [B,256,20,20]
~~~

第一步每个输入通道独立提取空间信息，不进行跨通道混合；第二步在已经缩小的 20×20 网格上用 1×1 卷积混合通道。参数和卷积计算量最少，但空间滤波与通道交互被拆开，表达方式不同于普通卷积。配置 p5_mode=depthwise。

#### BN64：先压缩通道，再用普通卷积降采样

~~~text
[B,384,40,40]
  -> Conv 1×1, stride=1, groups=1, 384→64
  -> GroupNorm(64) -> SiLU
  -> [B,64,40,40]
  -> Conv 3×3, stride=2, groups=1, 64→256
  -> GroupNorm(256) -> SiLU
  -> [B,256,20,20]
~~~

先在原 40×40 网格上将 384 通道压到 64，再用普通 3×3 卷积完成空间与跨通道联合处理。它保留了普通空间卷积的跨通道连接，同时通过 64 通道瓶颈限制成本；瓶颈也可能损失信息，需要实验判断。BN64 的 BN 指本项目的瓶颈变体命名，不是 BatchNorm。配置 p5_mode=bottleneck、p5_bottleneck_channels=64。

#### 参数与计算量

以下参数量由当前代码直接构造模块核对，统计卷积与 GroupNorm 仿射参数。单支指一个来源的 P5，三支对应 block4/8/12；冻结 Teacher 参数另列于最终配对实验。

| 项目 | BASE | DW | BN64 |
|---|---:|---:|---:|
| 单个 P5 分支参数 | 885,248 | 103,040 | 172,672 |
| 三个 P5 分支参数 | 2,655,744 | 309,120 | 518,016 |
| 整个九分支 Adapter 参数 | 2,878,080 | 531,456 | 740,352 |
| COCO 下游总可训练参数 | 3,542,567 | 1,195,943 | 1,404,839 |
| 单个 P5 分支卷积 MAC / 图片 | 353,894,400 | 40,704,000 | 98,304,000 |

P5 卷积 MAC 按输出网格×输出通道×每个输出的卷积乘加数计算。该列用于比较 P5 卷积成本；端到端训练时间另外计入其余模块、反向、优化器和 I/O，并以实测为准。

三种结构输出形状相同，均可接同一 LatentMixture/Detect，并复用同一份 Teacher 缓存。切换结构时重新初始化对应下游参数。正式对照采用 BN64，选择依据是结构成本与 COCO 筛选中的参数/精度折中。

对应配置：[BASE](../../ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml)、[DW](../../ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-dw-n.yaml)、[BN64](../../ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-bottleneck64-n.yaml)；实现见 [foundation_adapter.py](../../ultralytics/nn/modules/foundation_adapter.py)。

### 代码入口

| 功能 | 入口 |
|---|---|
| Teacher 多层输出 | [dinov3.py](../../ultralytics/nn/foundation/teachers/dinov3.py) |
| safetensors / NPY 缓存 | [cache.py](../../ultralytics/nn/foundation/cache.py)、[npy_cache.py](../../ultralytics/nn/foundation/npy_cache.py) |
| 多尺度适配与检测模型 | [foundation_adapter.py](../../ultralytics/nn/modules/foundation_adapter.py)、[foundation_detection_model.py](../../ultralytics/nn/foundation_detection_model.py) |
| Dataset / Trainer / Validator | [d1_cache.py](../../ultralytics/data/d1_cache.py)、[foundation_train.py](../../ultralytics/models/yolo/detect/foundation_train.py)、[foundation_val.py](../../ultralytics/models/yolo/detect/foundation_val.py) |
| 单次训练、评测及 scratch | [train.py](../../scripts/d1/train.py)、[rgb.py](../../scripts/d1/rgb.py) |
| 配对实验、计时与精确续训 | [compare.py](../../scripts/d1/compare.py)、[runtime.py](../../scripts/d1/runtime.py) |

## 安装与数据准备

D1 使用 Linux 和 Python >=3.10。已验证环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 5.15.1、Ultralytics 8.4.101、safetensors 0.8.0；真实多卡验收使用六张 A40。该记录是实测环境，不保证任意依赖组合均等价。

~~~bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[dev,foundation]"
pip install faster-coco-eval==1.8.0
export D1_WORK=/path/to/external/d1-work
export COCO_ROOT="$D1_WORK/datasets/coco"
export TEACHER_DIR="$D1_WORK/weights/dinov3-vits16"
~~~

Teacher 使用 [ModelScope ViT-S/16](https://www.modelscope.cn/models/facebook/dinov3-vits16-pretrain-lvd1689m)，固定来源版本 2e601320d0545509ab03374e2f8707f303e1de7a。将 config.json、model.safetensors、LICENSE.md、README.md 放入 TEACHER_DIR。文件大小、SHA256 与架构由 [Teacher manifest](manifests/dinov3-vits16.json) 校验。

COCO 下载 train2017.zip、val2017.zip、annotations_trainval2017.zip 和 coco2017labels.zip；URL、镜像来源、大小与 SHA256 见 [数据 manifest](manifests/coco2017-splits.json)。图片解压到 COCO_ROOT/images，官方标注到 COCO_ROOT/annotations；labels 压缩包内含 coco/labels，应解压到 D1_WORK/datasets。

~~~bash
python -m scripts.d1.prepare_coco --coco-root "$COCO_ROOT" \
  --weights-dir "$TEACHER_DIR" --output "$D1_WORK/inputs"
~~~

prepare_coco 验证 118,287/5,000 图片、稳定列表摘要、train/val 互斥、标签归属、官方标注文件存在性与 Teacher 权重，并从本地权重构造 Teacher。提供 --archives-dir 可额外校验源压缩包；--lists-only 用于恢复图片列表。重复执行保持官方划分、输出和证据一致。

正式预处理见 [合同](manifests/experiment-contract.json) 和 [COCO 配方](../../ultralytics/cfg/experiments/d1/dinov3-vits16-coco2017.yaml)：固定 640 方形 LetterBox、RGB CHW、[0,1]、DINOv3 mean/std；Teacher 不再缩放或裁剪，block 4/8/12 均输出 40x40 网格。缓存采用 d1-cache-v1 和 FP16。

VisDrone 使用官方 DET train/val，共 6,471/548 张，原始目录包含 VisDrone2019-DET-train 与 VisDrone2019-DET-val，各自有 images/annotations：

~~~bash
python -m scripts.d1.prepare_visdrone \
  --source "$D1_WORK/visdrone-original" --output "$D1_WORK/visdrone-prepared"
~~~

输出 dataset.yaml、train.txt、val.txt 与保留 ignore 信息的 sidecar；原始数据不删除。当前图片准备采用硬链接，source 与 output 须位于同一文件系统。--include-test-dev 是可选保留集，不混入 train/val。

## 特征缓存

~~~bash
python -m scripts.d1.cache_features build --data-root "$COCO_ROOT" --weights-dir "$TEACHER_DIR" \
  --samples-file "$D1_WORK/inputs/coco2017-train2017.txt" \
  --cache-dir "$D1_WORK/cache/train2017" --split train2017 --batch-size 16 --device 0
python -m scripts.d1.cache_features build --data-root "$COCO_ROOT" --weights-dir "$TEACHER_DIR" \
  --samples-file "$D1_WORK/inputs/coco2017-val2017.txt" \
  --cache-dir "$D1_WORK/cache/val2017" --split val2017 --batch-size 16 --device 0
python -m scripts.d1.cache_features verify --cache-dir "$D1_WORK/cache/train2017"
python -m scripts.d1.cache_features verify --cache-dir "$D1_WORK/cache/val2017"

# 可选：一图一个 NPY，无损转换并保留源分片。
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/train2017" --output "$D1_WORK/coco-npy"
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/val2017" --output "$D1_WORK/coco-npy"
~~~

VisDrone 使用相同入口：

~~~bash
python -m scripts.d1.cache_features build --data-root "$D1_WORK/visdrone-prepared" \
  --weights-dir "$TEACHER_DIR" --samples-file "$D1_WORK/visdrone-prepared/train.txt" \
  --split visdrone-train --cache-dir "$D1_WORK/cache/visdrone-train" --batch-size 16 --device 0
python -m scripts.d1.cache_features build --data-root "$D1_WORK/visdrone-prepared" \
  --weights-dir "$TEACHER_DIR" --samples-file "$D1_WORK/visdrone-prepared/val.txt" \
  --split visdrone-val --cache-dir "$D1_WORK/cache/visdrone-val" --batch-size 16 --device 0
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/visdrone-train" --output "$D1_WORK/visdrone-npy"
python -m scripts.d1.cache_features to-npy \
  --cache-dir "$D1_WORK/cache/visdrone-val" --output "$D1_WORK/visdrone-npy"
~~~

图片列表必须排序、无重复，每行形如 images/SPLIT/ID.jpg。build 使用 seed 0、确定性执行和关闭 TF32，记录图片、代码、batch、设备与依赖身份。工程小样本可使用 --limit，但训练 YAML 必须同步限制到已缓存图片。

单进程独占一个输出目录；并发写入、身份不一致或残留临时文件报错。中断后用相同代码和命令重放原 batch，已提交成员必须逐元素一致；不能编辑 build.json 绕过检查。旧代码的未完成缓存应在原提交续写，已完成缓存可继续校验、读取和转换。六卡抽取调度器不在本 PR 中。

safetensors 与 NPY 保存相同 FP16 特征，不引入额外有损量化。数据、缓存与输出建议放高速本地盘。--benchmark-read 测得的缓存读取吞吐不是训练吞吐。

## 训练、恢复与评测

默认功能配方为 [cached-detection.yaml](../../ultralytics/cfg/experiments/d1/cached-detection.yaml)，CLI 显式覆盖 batch、epochs、workers 和 seed。它不依赖个人研究队列。

~~~bash
python -m scripts.d1.train inspect --variant BN64
python -m scripts.d1.train train --approved --variant BN64 --dataset coco \
  --data /path/to/coco-local.yaml \
  --train-cache "$D1_WORK/coco-npy/train2017" --val-cache "$D1_WORK/coco-npy/val2017" \
  --output "$D1_WORK/runs/example" --device 0 --batch 16 --epochs 50 --workers 4 --seed 0

python -m scripts.d1.train evaluate --variant BN64 --dataset coco \
  --data /path/to/coco-local.yaml --val-cache "$D1_WORK/coco-npy/val2017" \
  --checkpoint "$D1_WORK/runs/example/weights/best.pt" \
  --output "$D1_WORK/evaluations/example-best" --device 0 --batch 16 \
  --annotations "$COCO_ROOT/annotations/instances_val2017.json"
~~~

coco-local.yaml 使用仓库常规检测 YAML，指定 path、train、val 与 80 类 names。VisDrone 使用 prepare_visdrone 生成的 10 类 dataset.yaml 和 --dataset visdrone。SCRATCH 使用 --variant SCRATCH，不传缓存参数。新运行输出目录必须不存在，避免覆盖。

多卡由外部 torchrun 启动，使用 --standalone --nproc_per_node=6 -m scripts.d1.train train，并传 --device 0,1,2,3,4,5 与可整除的全局 batch。--ema scalar-v1/foreach-v1、--p3-upsample bilinear/separable_bilinear2x 保留实现选择；--fp32 用于数值对照。只有提前完成全量缓存校验才使用 --trusted-cache。

精确恢复用于启用 --telemetry 的运行：重复原训练命令并追加 --resume-snapshot，使用同一执行提交、数据、模型、精度策略、batch 和 seed。--window 仅限制已执行轮数，保持完整学习率调度；普通 last.pt 用于评测，resume.pt 用于完整状态恢复。调整总预算后在新目录从头训练，旧运行记录独立保留。

评测严格重载 checkpoint，输出 evaluation.json、predictions.json 和 checkpoint SHA256。COCO 只有提供 --annotations 才报告标准 AP，使用 faster-coco-eval 1.8.0、完整 5,000 张验证图、maxDets=[1,10,100]；最多导出 300 个预测框不改变标准 AP 的 maxDets=100。

VisDrone 评测完整 548 张验证图并导出官方 TXT，每图最多 500 框；坐标还原到原图，序列化后宽或高不大于零的框被剔除并计数，非有限值报错。官方 ignore 语义和评分使用固定 MATLAB DET toolkit。公开的 evaluate_visdrone_official.m 复用本次实验的评分包装器，调用原版匹配/AP 函数；随附工具文件 SHA256 清单和缺少 Image Processing Toolbox 时使用的 mean2 兼容函数。

### VisDrone 官方评分复现

先运行 train evaluate 导出预测，再调用 MATLAB。以下命令从仓库根目录执行；VISDRONE_VAL 必须指向包含原始 images/annotations 的 VisDrone2019-DET-val，不能使用转换后的 YOLO 标签。D1_OUTPUT 为该次训练目录，D1_EVAL 必须为新的评测输出目录。

~~~bash
export VISDRONE_VAL="$D1_WORK/visdrone-original/VisDrone2019-DET-val"
export D1_OUTPUT="$D1_WORK/runs/visdrone-BN64"
export D1_EVAL="$D1_WORK/evaluations/visdrone-BN64-last"
python -m scripts.d1.train evaluate --variant BN64 --dataset visdrone \
  --data "$D1_WORK/visdrone-prepared/dataset.yaml" \
  --val-cache "$D1_WORK/visdrone-npy/visdrone-val" \
  --checkpoint "$D1_OUTPUT/weights/last.pt" \
  --output "$D1_EVAL" --device 0 --batch 16 --workers 4

git -c core.autocrlf=false clone https://github.com/VisDrone/VisDrone2018-DET-toolkit.git "$D1_WORK/visdrone-toolkit"
git -C "$D1_WORK/visdrone-toolkit" checkout --detach 005445782213e20cb91bc50a597db3dd949e749a
export VISDRONE_TOOLKIT="$D1_WORK/visdrone-toolkit"
export VISDRONE_PREDICTIONS="$D1_EVAL/visdrone-txt"
export VISDRONE_REPORT="$D1_EVAL/official-matlab.json"
matlab -batch "addpath('scripts/d1'); evaluate_visdrone_official(getenv('VISDRONE_TOOLKIT'), getenv('VISDRONE_VAL'), getenv('VISDRONE_PREDICTIONS'), getenv('VISDRONE_REPORT'), 'experiments/d1/manifests/visdrone-toolkit.json');"
python -m scripts.d1.evaluate_visdrone check --report "$VISDRONE_REPORT"
~~~

验证环境为 MATLAB R2026a，需启用 JVM；MATLAB 可在另一台机器上运行，只需复制原始 val、包含 export.json 的完整 visdrone-txt、固定 toolkit 与本仓库三个评分文件，并设置相应路径。Windows PowerShell 使用 $env:变量名 设置相同变量，再执行 & "$env:MATLAB_ROOT/bin/matlab.exe" -batch "...相同 MATLAB 表达式..."。非零退出视为失败。请使用新的报告路径，保留原始评分。

包装器校验固定工具文件 SHA256、GT/预测文件名完整一致、类别 1..10、排序后的每图 top500 与有限值，将空 TXT 规范化为 0×8 数组；按原图尺寸应用官方 ignore 处理。输出官方 AP/AP50/AP75/AR1/10/100/500、MATLAB 版本、toolkit 与 export 摘要。正式 val 应为 548 张，image_count 需与该样本数一致；小型 fixture 仅用于测试。


## 已完成的消融与研究结果

### COCO Adapter 筛选

完整 COCO、seed 0、固定第 50 轮 checkpoint，独立标准 COCO 评测：

| P5 结构 | 下游参数 | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE：3x3 stride2 Conv | 3,542,567 | 28.870 | 49.197 | 30.113 | 12.921 | 33.463 | 40.228 |
| DW：深度可分离分支 | 1,195,943 | 28.593 | 48.542 | 29.723 | 13.033 | 32.510 | 41.559 |
| BN64：64 通道瓶颈 | 1,404,839 | 29.745 | 49.284 | 31.203 | 12.965 | 33.086 | 42.992 |

BN64 被选作后续基座。单 seed 结果仅支持候选筛选，不构成统计等效或稳定提升证明。固定归档包含[完整报告](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/P5_FAST_RUN_20260909.md)与[机器可读汇总](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/p5-screen-20260909/suite-summary.json)，记录配方、计时与 checkpoint 身份。

### VisDrone Latent Aux 消融：P2 已完成

本节是所选 P2 消融路线的交付结果，包含真实运行、官方评分和统计分析；不以更换 Teacher 为本轮交付的前置条件。

完整 VisDrone train/val、BN64、weighted_sum、seeds 0/1/2，固定 300 轮调度的前 60 轮。统一评测第 60 轮，采用固定官方 MATLAB DET toolkit，保留 ignore 语义。

先扫描 balance={0,0.01,0.1} 与 z={0,0.001,0.01}、gain=0.1，共 27 次；再固定 balance=0.1、z=0 扫描 gain，新增 9 次并复用 3 次，共 36 个独立运行。

| latent_aux_gain | 官方 AP 均值（3 seeds） |
|---|---:|
| 0 | 8.022406397 |
| 0.03 | 8.115370260 |
| 0.1 | 8.131928488 |
| 0.3 | 8.081231746 |

按预定均值优先规则选择 balance=0.1、z=0、gain=0.1、budget=3.0。gain=0.1 相对关闭 aux 的平均 AP 增加 0.109522 点，三 seed 配对差为 -0.245386/+0.339334/+0.234618 点，探索性 95% 区间跨 0；这组三 seed 同时用于选参，结果体现小幅均值收益与较大的种子差异。gain=0.03 为波动较小的备选。

上述结果取自 300 轮调度的前 60 轮筛选窗口，统一评测第 60 轮 checkpoint。固定归档：[消融报告](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/E3.md)、[第一阶段证据](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/e3-stage1-official-20260911.json)、[第二阶段证据](https://github.com/Frank95zz/YOLO-Master/blob/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1/manifests/e3-stage2-official-20260911.json)。

## P1 最终配对实验

P1 已完成两数据集、两架构各三 seed 的正式训练、独立评测与成本审计。以下先固定复现合同与命令，再报告最终三维对照、周期最佳及成本结论。

冻结组固定 ViT-S/16 + BN64 + weighted_sum：balance=0.1、z=0、gain=0.1、budget=3.0。Scratch 使用随机初始化的标准 YOLO26-L 拓扑宽度匹配版：depth=1.0、width=0.9375、max_channels=512。两组重新训练，不续训筛选权重。

| 数据集 | train / val | 每组预算 | seeds | 每卡 / 全局 batch | 运行数 |
|---|---:|---:|---|---:|---:|
| COCO 2017 | 118,287 / 5,000 | 50 epochs | 0/1/2 | 64 / 384 | 6 |
| VisDrone2019-DET | 6,471 / 548 | 120 epochs | 0/1/2 | 16 / 96 | 6 |

| 参数口径 | COCO | VisDrone |
|---|---:|---:|
| 冻结 Teacher | 21,596,544 | 21,596,544 |
| BN64 下游可训练参数 | 1,404,839 | 1,340,259 |
| 冻结组总参数 | 23,001,383 | 22,936,803 |
| scratch 总参数 | 23,133,560 | 23,032,340 |
| scratch 相对总参数差 | +0.57% | +0.42% |

总参数匹配包含冻结 Teacher，项目采用 1% 的工程匹配容差。这是冻结特征路径与 RGB 从零训练路径的系统对照。配置为 [paired-comparison.yaml](../../ultralytics/cfg/experiments/d1/paired-comparison.yaml) 和 [scratch-total-l](../../ultralytics/cfg/models/26/yolo26-d1-scratch-total-l.yaml)。

每次运行独占六张 A40，各组串行。固定 640 方形单次 LetterBox、无颜色/几何/翻转/Mosaic/MixUp/Copy-Paste/多尺度增强，两组 RGB/NPY 均使用 NVMe。AdamW 使用共享参数分组，lr0=0.001、lrf=0.01、momentum=0.9、weight_decay=0.0005、cosine、warmup=3；Router 沿用半学习率分组。nbs 等于全局 batch，每 batch 一次有效更新。

两组统一采用 BF16 混合精度训练、FP32 主参数/优化器/EMA、FP32 检测损失和 FP32 验证/独立评测。BF16 路径关闭 GradScaler，记录 scale=1 与 gradient_scaling=false。Scratch 保留三个 Attention 的 FP32 敏感计算，workers=4/rank、prefetch=1；Frozen 使用 foreach EMA 和可分离 P3，Scratch 使用原生 EMA，实际开销全部计时。

~~~bash
python -m scripts.d1.compare --approved --output "$D1_WORK/final-comparison" \
  --coco-root "$COCO_ROOT" --coco-cache "$D1_WORK/coco-npy" \
  --visdrone-root "$D1_WORK/visdrone-prepared" --visdrone-cache "$D1_WORK/visdrone-npy" \
  --device 0,1,2,3,4,5
~~~

入口要求干净代码提交、新的外部输出目录及明确批准，不下载或删除数据。外部目录记录 plan.json、status.json、日志、运行身份和 ETA。正式主结果固定 COCO 第 50 轮、VisDrone 第 120 轮；每 5 轮保留周期 checkpoint，单组训练结束后依次独立评测周期快照、last.pt 与内部 best.pt。COCO 从周期快照选标准 AP 最优者并生成 standard-best.json，平局取更早轮；VisDrone 导出预测后标记等待 MATLAB 评分。

### 数值策略与执行版本

正式合同为 d1-paired-comparison-v2：COCO 50 轮、VisDrone 120 轮，均从第 1 轮采用对应的余弦调度，保持三 seed、总参数匹配及原 batch 配置。早期 100/300 轮配方的记录独立归档。

正式执行提交固定为 ee23d3edf04b3741e1a6441c57a9cc2ff4d2516c；VisDrone seed 0 两组保留 1f940445d4833794aa224fb626d99a8e2ed9703f。两版本模型、训练实现与精度策略一致，差异为 COCO 预算合同。学习率从第 1 轮按各自最终 50/120 轮预算计算，执行身份与配置摘要保存在实验记录中。

已用同一真实验证批次核对精度：Scratch P5 下采样卷积的 FP32 输出最大绝对值为 80,257.84；BF16 输出为 79,872，预测与 loss 有限。对应在线模型和 EMA 的对照表明，验证模式下的 BN 运行统计会显著影响激活范围；保留原 EMA/BN 算法，使用 BF16 的动态范围承载训练，并以 FP32 计算检测损失和正式验证。该批次还完成一次 BF16 反向及参数更新，558 组非零梯度均有限。

上述 compare 命令是通用复现入口：依次完成四组恢复门禁、各五轮完整短测与 ETA 估计，再执行配对训练。本轮实际执行使用外部调度器复用 compare.py 的数据校验、训练命令、恢复比较与评测函数；额外安排 VisDrone Scratch 的 40 轮稳定性检查，完整训练合同保持一致。

本轮启动顺序为四组模型/数据组合的六卡恢复一致性门禁，随后 Scratch 在 VisDrone 新 120 轮调度下连续运行至第 40 轮。通过有限值、更新次数、精度身份和完整快照检查后，同一运行从第 41 轮继续，之后执行其余正式组合。seed 0/1 先 Scratch 后 BN64，seed 2 先 BN64 后 Scratch，实际顺序纳入时间记录。该 40 轮窗口可通过 train 入口的 --epochs 120 --window 40 复现，后续保留 --epochs 120、去掉 --window 并指定 --resume-snapshot；完整数据、模型、batch、seed、精度与执行提交必须保持一致。外部调度器与日志保留在实验归档，不增加 PR 的公共入口。

本轮精度与预算改动的相关回归结果为 **687 passed、56 skipped、2 deselected**，耗时 70.35 秒；两个排除项沿用下文记录的已有问题。真实异常批次另通过 FP32 验证、BF16 前向、FP32 原生损失、有限梯度和优化器更新验收。

### 工程验收

执行提交 9004eac 已完成四种数据集/模型组合的真实六卡恢复一致性门禁：连续两轮与一轮后恢复到两轮的模型、EMA、优化器、scaler、scheduler、criterion 和各 rank 状态一致。每种组合还完成五个完整 epoch、周期 checkpoint 保存、完整验证集独立推理及严格重载。

DDP 在首次训练和恢复时先进行三次无 optimizer 更新的前反向以建立相同分桶，再恢复模型、损失调度和随机状态；一次性准备耗时单独记录。加载器按 epoch 重建并确定性设种子，避免无限预取跨轮改变恢复后的取样顺序。缺失样本、非有限值、漏更新、恢复不一致或 OOM 均停止，不自动缩小某一组的 batch。

以下时间为同次五轮短测第 4/5 轮的中位数，包含轮内验证与保存，不含训练后独立评测、Teacher 抽取和 MATLAB：

| 数据集 | BN64 秒/epoch | scratch 秒/epoch |
|---|---:|---:|
| COCO | 123.23 | 372.91 |
| VisDrone | 18.32 | 40.53 |

上述五轮短测用于预算与工程验收；正式结果按完整周期的标准评测与资源记录汇总。

### P1 精度 / 显存 / 时长三维对照

两数据集、两架构、seeds 0/1/2 共 **12 次正式训练及独立评测全部完成**。下表统一报告三 seed **均值 ± 样本标准差**，AP 使用 0-100 点；逐 seed 数值、checkpoint epoch/SHA256、配置及产物摘要保存在复现包实验记录中。

#### 固定末轮：最终精度与资源成本

精度主表固定 COCO 第 50 轮、VisDrone 第 120 轮。COCO 使用 faster-coco-eval 1.8.0、完整 val2017 5,000 张、bbox AP@[0.50:0.95]、maxDets=100；VisDrone 使用官方 MATLAB DET 工具 005445782213e20cb91bc50a597db3dd949e749a、完整 val 548 张、原始 ignore/类别协议。两组按同一数据集的同一口径评测。

| 数据集 / 固定末轮 | 架构 | AP | AP50 | AP75 |
|---|---|---:|---:|---:|
| COCO / 50 | BN64（冻结） | 28.666 ± 0.576 | 47.781 ± 0.710 | 30.195 ± 0.747 |
| COCO / 50 | Scratch | 27.087 ± 0.161 | 40.211 ± 0.273 | 28.791 ± 0.188 |
| VisDrone / 120 | BN64（冻结） | 7.312 ± 0.192 | 17.783 ± 0.277 | 5.162 ± 0.262 |
| VisDrone / 120 | Scratch | 11.026 ± 0.382 | 21.441 ± 0.731 | 10.056 ± 0.385 |

| 数据集 | 架构 | 单卡峰值 allocated（GiB） | 单卡峰值 reserved（GiB） | 训练作业（小时） | 训练 GPU-hours |
|---|---|---:|---:|---:|---:|
| COCO | BN64（冻结） | 9.040 ± 0.013 | 15.910 ± 0.432 | 1.766 ± 0.007 | 10.595 ± 0.042 |
| COCO | Scratch | 40.834 ± 0.010 | 43.540 ± 0.001 | 5.472 ± 0.011 | 32.830 ± 0.063 |
| VisDrone | BN64（冻结） | 5.276 ± 0.006 | 10.688 ± 0.338 | 0.604 ± 0.002 | 3.624 ± 0.012 |
| VisDrone | Scratch | 13.702 ± 0.051 | 18.076 ± 0.194 | 1.383 ± 0.012 | 8.301 ± 0.073 |

**显存统计口径：** allocated（张量显存）表示 PyTorch 张量实际占用的显存，包括模型参数、梯度、优化器状态、EMA、输入和中间特征；reserved（显存池）表示 PyTorch 缓存分配器管理的显存总量，包括正在使用和暂未使用的空间。两项峰值分别对应 torch.cuda.max_memory_allocated() 和 torch.cuda.max_memory_reserved()。本文以 allocated 为显存对照主指标，reserved 为补充指标。

每次运行先取全部训练 epoch、六张 GPU 中的单卡最大值，再报告三个 seed 的均值与样本标准差；不是六张卡显存之和。GiB = bytes / 2^30。两项峰值可能出现在不同时间，差值不能直接解释为同一时刻的空闲显存。reserved 较大本身不等于显存泄漏；CUDA 上下文等分配器外占用也可能使 nvidia-smi 的数值与上述指标不同。

训练作业时间包含启动、训练、数据等待、轮内验证及保存，GPU-hours = 六张卡 × 作业小时。VisDrone Scratch seed 0 的 40+80 轮合并计时且只计一次；BN64 seed 0 的接管结束时间保留 10 秒轮询误差。Teacher 抽取和训练后独立评测分列。

辅助指标的三 seed 均值：COCO BN64 / Scratch 的 APs 为 **12.542 / 11.202**、APm 为 **31.417 / 28.005**、APl 为 **42.060 / 39.724**；VisDrone AR500 为 **21.404 / 22.998**。完整辅助指标与标准差见机器可读汇总。

#### 配对结论与验收

降低率和保留率按两组均值计算：AP 保留率 = AP_BN64 / AP_Scratch；成本降低率 = 1 - 成本_BN64 / 成本_Scratch。配对区间先按相同 seed 求差，再用 n=3、df=2 的探索性 95% Student t 区间。

| 数据集 | AP 保留率 | allocated 降低 | reserved 降低 | 训练 GPU-hours 降低 | 固定末轮配对 AP 差及 95% 区间 |
|---|---:|---:|---:|---:|---|
| COCO | 105.83% | 77.86% | 63.46% | **67.73%** | +1.579，[-0.191, 3.349] |
| VisDrone | 66.31% | 61.50% | 40.87% | **56.34%** | -3.714，[-5.118, -2.311] |

COCO 固定末轮 AP 均值高于 Scratch，配对区间跨 0；VisDrone 体现精度与成本的折中。逐 seed 训练 GPU-hours 降低率的 95% 区间分别为 COCO **[67.53%, 67.92%]**、VisDrone **[55.75%, 56.93%]**。两数据集训练作业口径均达到 ≥50% 降低；COCO 加入一次 Teacher 抽取后仍达到该目标，详见成本分项。

#### 周期最佳：独立补充结果

两架构使用相同的每 5 轮快照频率，以独立标准 AP 最大者选周期最佳，并列取较早轮。COCO 每组评测 10 个周期快照，VisDrone 每组 24 个；训练成本始终计完整 50/120 轮。

| 数据集 | 架构 | 最佳 epoch（seed 0/1/2） | 周期最佳 AP（均值 ± 标准差） | BN64 AP 保留率 |
|---|---|---|---:|---:|
| COCO | BN64 | 35 / 40 / 35 | 28.800 ± 0.547 | 91.21% |
| COCO | Scratch | 20 / 20 / 20 | 31.577 ± 0.033 | 参照组 |
| VisDrone | BN64 | 40 / 50 / 35 | 8.067 ± 0.154 | 55.69% |
| VisDrone | Scratch | 30 / 25 / 25 | 14.485 ± 0.101 | 参照组 |

COCO 周期最佳排序与固定末轮不同，Scratch 的最佳 AP 更高；VisDrone 两种口径均为 Scratch 更高。VisDrone 周期最佳配对差为 **-6.418 AP 点**，95% 区间 **[-7.053, -5.783]**。因此最终报告同时保留固定末轮与周期最佳，便于判断训练后期变化。

VisDrone 的 144 个周期快照及 6 个内部 best 共 **150 项官方 MATLAB 补充评分全部完成**。内部 best 按训练内置指标选择，其官方评分单独归档；六组第 120 轮全部七项 AP/AR 与固定末轮报告一致。

#### 成本分项与一次抽取

| 成本口径（三 seed 均值） | COCO BN64 | COCO Scratch | VisDrone BN64 | VisDrone Scratch |
|---|---:|---:|---:|---:|
| 完整训练作业 GPU-hours | 10.5952 | 32.8304 | 3.6239 | 8.3007 |
| 全部独立预测/评测窗口秒数 | 500.369 | 821.552 | 265.656 | 325.088 |
| 上项单卡有效窗口 GPU-hours | 0.1390 | 0.2282 | 0.0738 | 0.0903 |
| 独立评测完整单卡作业 GPU-hours（含启动） | 0.2000 | 0.2852 | 0.1445 | 0.1815 |

独立窗口来自 evaluation.json 的 seconds，每次使用一张 GPU；COCO 每组计 10 个周期快照、last 和内部 best，VisDrone 每组计 24 个周期快照、last 和内部 best。它是进程内记录的评测窗口，不包含进程启动、数据传输与本机 MATLAB。VisDrone 的 150 项补充 MATLAB 评分累计 **4,136.251 秒**，六项固定末轮累计 **157.297 秒**，合计 **4,293.549 秒**；这是 CPU 评分墙钟累加，独立于 GPU-hours。

COCO 首次完整缓存的来源为研究归档 f4d2bc268bb6339f6545fc3ebe6a247c238cd883 中 experiments/d1/manifests/wp8-full-cache.json，SHA256 为 e3a2d0eae5c5433f33a3fa37be4bdfd7d5e24da8329f65c3d6cde4fa06fa12f3，实际抽取代码为 5b5e4affa056a8e4c10510a037df503b24f53e57。train/val 内容摘要与正式缓存身份一致。六卡 worker 阶段合计 **2,258.818 秒 = 3.7647 GPU-hours**，包含加载、抽取和写入；另有 CPU 最终校验 **1,355.489 秒**。据此：

- COCO 训练加一次抽取：**14.3599 GPU-hours**，相对 Scratch 完整训练降低 **56.26%**。
- 同一缓存复用三次、每次分摊 1/3 抽取：**11.8501 GPU-hours**，降低 **63.90%**。
- 以上 GPU 口径将 CPU 校验、NPY 转换、复制及独立评测另列；原始预训练不在本次下游比较范围内。

VisDrone 历史准备记录为 430.581 秒，但对应脚本可跳过已有缓存，且未保留各次已抽取/跳过数量，因此首次完整抽取 GPU-hours 记为 null 并附来源说明。VisDrone 的 **56.34%** 为已验证的训练作业降本。全部分项、源文件 SHA256 与计量边界见复现包 results/cost-breakdown.json。

#### 结果与证据索引

| 复现包文件 | 内容 |
|---|---|
| results/coco-three-seed-summary.json | COCO 最终精度、辅助指标、显存、时长及配对统计 |
| results/visdrone-three-seed-summary.json | VisDrone 固定末轮官方指标与三维配对统计 |
| results/visdrone-periodic-official-summary.json | 周期最佳、内部 best、150 项评分及原始报告 SHA256 |
| results/cost-breakdown.json | Teacher、训练外评测与 MATLAB 成本的来源、摘要和计量边界 |
| results/{dataset}-{variant}-seed{0,1,2}.json | 12 份完整实验记录：执行版本、配置摘要、数据、硬件、预算、seed、指标、产物、状态和局限 |

全部记录与上述训练合同绑定；源码提交、配置 hash、逐 seed 结果和大产物引用保留原始身份。汇总重排不改变实验数值或 checkpoint。

### 实验记录与复现包

每次正式运行按附录 B 汇总以下 11 个字段，原始训练记录保留在外部工作区，阶段归档保存独立快照：

| 字段 | 记录内容 |
|---|---|
| experiment_id | D1、数据集、架构、整数 seed 与首次启动 UTC 时间组成唯一 ID |
| git_ref | 完整执行 commit、工作区状态；采用 commit 固定版本 |
| config | 配方、模型、数据 YAML 路径与 hash；区分文件 SHA256 和规范化模型配置摘要 |
| dataset | COCO 2017 / VisDrone2019-DET、train/val、样本数与列表摘要 |
| hardware | GPU 型号/显存、CPU、驱动、CUDA、cuDNN、PyTorch/Python；注明采集时刻，TensorRT 标记为本流程不适用 |
| budget | 总 epochs、batch、imgsz、已完成 epoch、分段耗时与最终 GPU-hours |
| seed | 明确整数 0/1/2 |
| metrics | 官方主指标、辅助指标、训练显存与耗时；三 seed 均值及配对差采用探索性 95% t 区间（n=3、df=2） |
| artifact | checkpoint/export/log/report 路径与 SHA256；动态文件注明采样时间，结束后封存完整清单 |
| status | success / failed / inconclusive 为结论状态，execution_status 单列运行进度 |
| limitation | seed 数、筛选窗口、预处理、硬件采集时刻、评测与产物归档状态 |

阶段训练和恢复窗口使用 inconclusive 表示结论尚在生成，窗口完成与整项成功分别记录。最终 success 在完整预算、独立评分与产物校验完成后确认；执行错误使用 failed；结果完整但未达到量化目标时，执行状态与验收结论分别报告。

复现包采用 configs/、scripts/、results/、env/、README.md、limitations.md。configs 保存执行配方，scripts 提供从零复现与记录归档入口，results 保存 12 份最终实验记录、两数据集统计、补充评分、成本分项及文件摘要，env 保存依赖和硬件；大数据、缓存和权重按外部输入与产物引用管理。包与 PR 源码分别交付，最终状态为 complete。

### 提交前七项核对

| 验收项 | 本 PR 对应做法 |
|---|---|
| 命令与配置齐全 | 固定执行版本的模块入口、模型/配方 YAML、数据与缓存准备命令；复现包提供安装与启动脚本 |
| 固定基线 | BASE_REF、UPSTREAM_REF、RUN_REF、FINAL_REF 均用完整 commit；main 仅表示 PR 合并目标 |
| seed 与统计 | 正式对照和 aux 消融使用 seeds 0/1/2；P5 单 seed 筛选明确注明范围；同时报告逐 seed 值与统计区间 |
| 对照口径一致 | 每个数据集内部固定预算、数据划分、640 输入、增强、batch、精度和官方评测；训练路径、EMA 与预训练差异显式列出 |
| 负结果与证据链 | 沿用已锁定的有限值/恢复门禁、固定末轮、选参规则和 ≥50% 成本目标；由运行身份、日志、checkpoint 和官方评分支撑结论 |
| PR 四节 | 改动摘要、测试证据、消融数据、已知局限，分别覆盖动机、命令/环境/结果、统计和支持范围/开销/后续工作 |
| 文档与源码成熟度 | 当前源码已实现的接口以固定 commit 和测试核对；历史筛选、当前工程验证、最终对照结果分开列示，源码引用采用文件/符号及固定版本 |

负结果判读沿用既定实验合同：非有限值、漏更新、数据身份或精确恢复检查失败时停止并保留证据；GPU 时间降低率低于 50% 时据实报告差距及精度保留率；均值差的探索性区间跨 0 时报告均值、区间与种子波动。性能诊断参考逐轮等待、单步时间、显存、存储和共享资源记录，机制解释与直接观测区分呈现。最终使用固定末轮评分，标准 best 单列。

本次补充将既有合同与判读规则集中列出，保留规则和执行版本的时间顺序；最终三 seed 结果按已锁定口径汇总。新增探索性分析另行标注。

## 测试与兼容性

最终提交前在同一代码基础上重新运行上述完整回归：687 passed、56 skipped、2 deselected、1 warning，179.99 秒；六个复现 CLI 的 --help 均通过。新增 MATLAB 包装器的完整 548 图复评验证见官方评分说明。

公开 MATLAB 评分入口已在完整 VisDrone val 548 张上复评 BN64 seed 0 第 40 轮：AP=7.923179115437879，全部七项 AP/AR 与原报告的绝对差 <1e-10，MATLAB R2026a 正常退出。该复核使用公开包装器、固定 toolkit 摘要及原始预测，不改匹配/AP 算法。

测试按功能组织：test_d1_contracts/cache/cache_cli/adapter/model/pipeline/training/visdrone，公共 Teacher 测试并入已有的 test_foundation_dinov3.py。覆盖非法输入、缓存校验与续写、九条分支梯度、aux 标量组合、Teacher 隔离、checkpoint 严格重载、默认 Attention 行为、AMP 有限值及恢复协议。

普通 CI 使用合成数据并跳过未配置的真实 Teacher/CUDA 测试，不自动下载模型。真实输入通过 D1_DINOV3_WEIGHTS、D1_WP2_CACHE、D1_COCO_ROOT、D1_NPY_CACHE 显式指定；保留这些环境变量以兼容已有使用方式，启用 CUDA 验收时不要清空 CUDA_VISIBLE_DEVICES。

~~~bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest -q \
  tests/test_checkpoint_compat.py tests/test_d1_adapter.py \
  tests/test_d1_cache.py tests/test_d1_cache_cli.py \
  tests/test_d1_contracts.py tests/test_d1_model.py \
  tests/test_d1_pipeline.py tests/test_d1_training.py \
  tests/test_d1_visdrone.py tests/test_ddp_checkpoint_coordination.py \
  tests/test_ddp_lifecycle_ema_nan.py tests/test_foundation_cache_training.py \
  tests/test_foundation_checkpoint.py tests/test_foundation_config.py \
  tests/test_foundation_dinov3.py tests/test_foundation_distill_model.py \
  tests/test_foundation_f08_effect_gate.py tests/test_foundation_f09_foreground_effect_gate.py \
  tests/test_foundation_f15_benchmark.py tests/test_foundation_f15_effect_matrix.py \
  tests/test_foundation_f15_real_effect_gate.py tests/test_foundation_f15_release_audit.py \
  tests/test_foundation_losses.py tests/test_foundation_metrics_logging.py \
  tests/test_foundation_mixture_interaction.py tests/test_foundation_multirouter.py \
  tests/test_foundation_multitask.py tests/test_foundation_offline.py \
  tests/test_foundation_projectors.py tests/test_foundation_recipe_integrity.py \
  tests/test_foundation_routing_contract.py tests/test_foundation_sam3.py \
  tests/test_foundation_semantic.py tests/test_foundation_siglip2.py \
  tests/test_foundation_taps.py tests/test_foundation_teacher_protocol.py \
  tests/test_foundation_weight_schedule.py tests/test_latent_mixture.py \
  tests/test_mixture_loss_composition.py tests/test_yoloe_released_checkpoint_compat.py \
  tests/test_prevalidation_recovery.py tests/test_optimizer_group_audit.py \
  tests/test_default_config_integrity.py tests/test_master_model_configs.py \
  tests/test_engine.py::test_load_checkpoint_state_dict_rejected --deselect=tests/test_foundation_cache_training.py::test_response_kd_builds_pseudo_batch_from_cached_responses \
  --deselect=tests/test_foundation_config.py::test_enabled_without_teacher_is_rejected
~~~

50/120 轮执行代码 ee23d3edf04b3741e1a6441c57a9cc2ff4d2516c 的相关回归为 **687 passed、56 skipped、2 deselected**，pytest 耗时 70.35 秒，范围为上述相关模块。历史执行提交 9004eac 为 637 passed、56 skipped、2 deselected；历史整理版本 5af743f 为 675 passed、56 skipped、2 deselected。两个 deselected 用例在 UPSTREAM_REF=af961b9 和 PR 整理版本上均复现，节点及原因随命令列出。

仓库质量入口以下使用 UPSTREAM_REF 检查 D1 相对整合上游的质量增量，不替代后文 BASE_REF 的成果审计：

~~~bash
python scripts/check_changed_quality.py --base af961b99b8ef80491e58cb5fd16e25ebaf3741eb --no-untracked
git diff --check
~~~

提交前使用 Ruff 0.11.13 复核 45 个相关 Python 文件，ruff check、ruff format --check 以及支持文件的 codespell 检查均通过；以下完整变更质量命令退出码为 0。

此前整理版本已验证 BN64/Scratch 在 10/80 类、固定初始化和 640 输入下的 CPU 状态及前向摘要一致。本轮新增显式 BF16 训练上下文、FP32 损失与验证策略；普通 Trainer 保持原精度行为。恢复身份包含精度策略、总预算和代码提交，完整测试与运行记录保存在外部工作区。

## 基线与新增贡献

### 为什么保留上游同步点

唯一的官方验收起点是 BASE_REF=acce839；UPSTREAM_REF=af961b9 不是第二套官方基线，也不是第三个实验模型。它是整合 D1 代码前的上游快照，保留它有三个具体用途：

1. **划清归属**：把公共基线之后其他作者已经合入的代码，与本 PR 自有增量分开，避免把上游 172 个文件的变化计为 D1。
2. **固定质量比较对象**：记录 D1 加入前已有的行为、测试问题和 lint 告警；例如后续上游 b69acc4 已修复 aux 标量组合，本 PR 只认领在其上的新增。
3. **解释 PR 集成来源**：说明这 61 个交付文件接在哪个较新上游版本上，并给出 merge-base 与两段 diff，便于维护者复核兼容性和来源。

复现训练使用 RUN_REF/FINAL_REF；UPSTREAM_REF 用于区分后续上游同步与 D1 增量，固定质量检查对象并追溯集成来源，因此保留在文末的版本审计中。

### 固定引用与核验方式

补充细则将全部课题的代码验收基线锁定为 2026-08-21 23:59:59（UTC+8）的 Tencent/YOLO-Master main，用于划分历史成果与本轮新增 diff。Scratch 实验基线则按前述总参数匹配合同构建。

| 引用 | 完整 SHA / 定位方式 | 用途 |
|---|---|---|
| BASE_REF：统一公共验收基线 | acce839c7e895d6b179de7f7093fa879e237cc7b | 所有新增成果按此固定起点审计，不随 main 移动 |
| 发布来源：YOLO-Master-v26.08 | 43d40117c30811204fb9347efeabddce15f11a62 | 仅说明发布版本来源，不代替 BASE_REF |
| UPSTREAM_REF：本 PR 整合采用的上游快照 | af961b99b8ef80491e58cb5fd16e25ebaf3741eb | 分离后续上游同步与 D1 自有改动；不是新的验收基线 |
| RUN_REF：50/120 轮配对执行提交 | ee23d3edf04b3741e1a6441c57a9cc2ff4d2516c | 后续任务使用本提交；已启动的 VisDrone seed 0 两组保留原始执行身份 |
| FINAL_REF：提交给评审的代码快照 | 在 PR 正文锁定完整 40 位 SHA；检出该版本后用 git rev-parse HEAD 核对 | 后续结果补充若形成新提交，保留旧引用并重新锁定，不用可移动分支名代替 |

本 PR 面向 Tencent/YOLO-Master 的 main，基于上述上游快照整合。已验证 BASE_REF 是 UPSTREAM_REF 和当前 PR 提交的祖先，当前 PR 提交与 UPSTREAM_REF 的 merge-base 为 af961b9；正式实验继续固定 RUN_REF。

在检出 PR 正文指定的 FINAL_REF 后执行：

~~~bash
BASE_REF=acce839c7e895d6b179de7f7093fa879e237cc7b
UPSTREAM_REF=af961b99b8ef80491e58cb5fd16e25ebaf3741eb
FINAL_REF=$(git rev-parse HEAD)
RUN_REF=ee23d3edf04b3741e1a6441c57a9cc2ff4d2516c
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "$BASE_REF" "$FINAL_REF"
git merge-base "$UPSTREAM_REF" "$FINAL_REF"
git diff --stat "$BASE_REF" "$FINAL_REF"
git log --reverse --format=fuller "$BASE_REF..$FINAL_REF"

# 后续上游同步：保留原作者，不计为本 PR 的个人新增。
git diff "$BASE_REF" "$UPSTREAM_REF"
git log --reverse --format=fuller "$BASE_REF..$UPSTREAM_REF"

# D1 交付范围：与统一基线总 diff 一同提供给评审。
git diff --name-status "$UPSTREAM_REF" "$FINAL_REF"
git diff "$UPSTREAM_REF" "$FINAL_REF"
git log --reverse --format=fuller "$UPSTREAM_REF..$FINAL_REF"
~~~

公共基线到上游快照包含 104 个可达提交、172 个变更文件；D1 交付相对上游快照为 61 个文件（41 新增、20 修改）。两段有 8 个重叠文件，合并后的公共基线到 PR 总 diff 为 225 个文件，其中 D1 交付范围按 UPSTREAM_REF → FINAL_REF 单独列示。重叠文件是 .gitignore、tests/test_ddp_lifecycle_ema_nan.py、tests/test_mixture_loss_composition.py、ultralytics/engine/extensions/recovery.py、ultralytics/engine/trainer.py、ultralytics/nn/foundation/__init__.py、ultralytics/nn/mixture_loss.py、ultralytics/nn/tasks.py；这些文件通过两段 diff 核对归属。固定 BASE_REF 审计与 GitHub 基于目标 main 的合并差异一并提供。

### 已有能力与本轮增量

| 能力 | BASE_REF 已有能力 | 本轮交付与证据 |
|---|---|---|
| Foundation / Teacher | FoundationFeatures、DINOv3Teacher、预处理与冻结推理、默认 dense["p4"]，见[原始 Teacher](https://github.com/Tencent/YOLO-Master/blob/acce839c7e895d6b179de7f7093fa879e237cc7b/ultralytics/nn/foundation/teachers/dinov3.py) | 新增 output_layers 一基编号 API，公开 stage 选择、三层输出与严格形状/冻结验证；默认接口保持兼容 |
| LatentMixture | router_only、weighted_sum、value_fusion_weights、Router 与 aux 已存在，见[原始模块](https://github.com/Tencent/YOLO-Master/blob/acce839c7e895d6b179de7f7093fa879e237cc7b/ultralytics/nn/modules/latent_mixture.py) | 复用这些机制适配三个 DINO 来源；新增 BASE/DW/BN64 架构与同协议实验 |
| latent aux 收集 | collect_aux_loss 默认集合不含 latent，但 CompositeCriterion 的调用已经显式 include_kinds 包含 latent，见[原始损失组合](https://github.com/Tencent/YOLO-Master/blob/acce839c7e895d6b179de7f7093fa879e237cc7b/ultralytics/nn/mixture_loss.py) | D1 接入现有收集通道，增加 raw/effective 指标、标量/有限值/梯度验证和三 seed 扫描 |
| 检测与缓存 | 现有 YOLO 检测头、Trainer、Dataset、Foundation 蒸馏及其缓存能力 | 新增作为检测器输入的 D1 多层缓存合同、分片/NPY、九分支 Adapter、模型、Dataset/Trainer/Validator 和严格重载链路 |
| 运行与性能 | 现有 DDP、EMA、checkpoint、优化器与恢复基础设施 | 新增 D1 预取边界、精确恢复与计时门禁、可分离 P3、foreach EMA、显式 Scratch FP32 Attention；存储收益与模型收益分开报告 |
| 研究结论 | 基线已有实验与报告 | 锁定后完成的 P5 架构筛选、36 次独立 aux 运行、性能/恢复诊断与最终配对实验 |

后续上游已经在 [b69acc4](https://github.com/Tencent/YOLO-Master/commit/b69acc4f63e48742460a2d02c391e0491464b44d) 修复 native_loss 先求和、标量 aux 只加一次；该提交由 onion-hong 贡献并包含于 UPSTREAM_REF。本 PR 在此基础上增加显式输入验证、D1 指标报告与防回归测试。

### 来源、归属与证据状态

D1 交付负责人为冯焱琦（[@Frank95zz](https://github.com/Frank95zz)），负责本 PR 中的多层 Teacher 扩展、缓存检测实现、运行验收及实验分析；关联进展为 [Issue #266](https://github.com/Tencent/YOLO-Master/issues/266)。公共模块、Teacher 模型、数据集及后续上游提交保留各自原作者署名。

本 PR 首次整合提交为 [f5bf7bc](https://github.com/Frank95zz/YOLO-Master/commit/f5bf7bc56d128e02d3485fe8df1e15301bee7556)，它将研究分支中需要交付的实现移植、精简到后续上游，以功能移植和精简方式交付。原研究历史和证据保留在 [f4d2bc2 固定归档](https://github.com/Frank95zz/YOLO-Master/tree/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1)，原始工作提交与最终交付的对应关系如下。

| 工作包 | 可追溯原始提交 | 最终交付定位 |
|---|---|---|
| Teacher 多层输出 | [3b05836](https://github.com/Frank95zz/YOLO-Master/commit/3b05836aca025bcb9dbacc596c1c049de214a87a) | teachers/dinov3.py 与 test_foundation_dinov3.py |
| 分片与 NPY 特征缓存 | [5cd566d](https://github.com/Frank95zz/YOLO-Master/commit/5cd566dcdb1e47a9227a81852296ade76d9ba20f)、[e4297e7](https://github.com/Frank95zz/YOLO-Master/commit/e4297e742f472c443ce55914e31144afcc67705d) | foundation/cache.py、npy_cache.py 与 scripts/d1/cache_features.py |
| 多尺度 Adapter 与检测闭环 | [e1c60c2](https://github.com/Frank95zz/YOLO-Master/commit/e1c60c27c2a57285a384048ce9e379035f2a046a)、[a1255c0](https://github.com/Frank95zz/YOLO-Master/commit/a1255c09b989d20f4f130ddf55e29614811f5fc2)、[81c98b3](https://github.com/Frank95zz/YOLO-Master/commit/81c98b334eb7a696561753164695fe30eb38531f) | foundation_adapter.py、foundation_detection_model.py 与缓存训练/验证实现 |
| 轻量 P5 与性能实现 | [3dfe0f6](https://github.com/Frank95zz/YOLO-Master/commit/3dfe0f6e72022b62a2f169e693b4996ac0361f7e)、[6f88a42](https://github.com/Frank95zz/YOLO-Master/commit/6f88a422f426b5fe9bd013aa25a7ef6a95570231) | 三种 P5 配置、Adapter 与性能测试；数值结论见前文固定证据 |
| 正式配对与稳定性 | [2920727](https://github.com/Frank95zz/YOLO-Master/commit/2920727209a5e1a6b7cd53080d37a1c5d83e3517) 至 [9004eac](https://github.com/Frank95zz/YOLO-Master/commit/9004eac438acd7de0023702e26029b14276069a6) | compare.py、runtime.py、scratch 数值策略和真实门禁 |

上述固定链接将原始工作提交、实验报告与最终交付文件关联，供评审核验实现来源和实验依据。

## 局限与许可

固定离线特征训练采用与抽取一致的预处理；调整颜色、几何、多尺度策略或更换 Teacher、输入尺寸和层选择时须重新设计并构建匹配缓存。速度报告同时记录存储、CPU、缓存热度及共享任务条件，以区分模型和运行环境的影响。

公共入口仅加载可信本地 checkpoint；精确恢复使用对应执行提交及一致的运行身份。研究历史和运行证据固定在 [f4d2bc2](https://github.com/Frank95zz/YOLO-Master/tree/f4d2bc268bb6339f6545fc3ebe6a247c238cd883/experiments/d1)。

许可与来源：

- 代码沿用仓库 AGPL-3.0 许可，见根目录 [LICENSE](../../LICENSE)。
- COCO：[Terms of Use](https://cocodataset.org/#termsofuse)、[下载源](http://images.cocodataset.org/)。图片仍受各自原始 Flickr 许可约束；YOLO labels 仅是官方标注的表示形式。
- DINOv3：[ModelScope 来源](https://www.modelscope.cn/models/facebook/dinov3-vits16-pretrain-lvd1689m)、[上游项目](https://github.com/facebookresearch/dinov3)、[License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)。许可副本随权重保存，SHA256 见 Teacher manifest。
- VisDrone 使用官方 DET 数据与评测工具；原始标注和 ignore 信息保留，数据及工具使用以各自上游条款为准。
