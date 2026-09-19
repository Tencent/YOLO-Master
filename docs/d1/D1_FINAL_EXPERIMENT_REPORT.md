# D1 冻结 DINOv3 × LatentMixture 最终实验报告

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 课题编号 | D1 |
| 课题名称 | 冻结基础模型 × LatentMixture 适配头 |
| GitHub ID | xizaoge-shuai |
| 公共基线 | `acce839c7e895d6b179de7f7093fa879e237cc7b` |
| 报告依据代码版本 | `8127956a83fe7196b08f2941d335705a1603bffa` |
| 硬件 | NVIDIA GeForce RTX 4090 |
| 软件环境 | Python 3.10、PyTorch 2.13.0+cu130、Ultralytics 8.4.101 |
| 辅助依赖 | safetensors 0.8.0、pytest 9.1.1 |
| 实验日期 | 2026-08-25 |

报告依据代码版本是实验冻结时的证据节点。纳入本报告后的最终集成提交 SHA 以最终 PR 页面和结项登记表为准。

## 2. 摘要

本课题研究在不微调基础视觉模型的条件下，能否利用离线缓存的 DINOv3 多层特征训练轻量目标检测器，从而降低重复训练成本。实现方案首先冻结 DINOv3，通过确定性预处理离线抽取多层特征并保存为 FP16 safetensors；训练阶段不加载 DINOv3，只训练三个尺度的 LatentMixture 适配模块和 Detect 检测头。

P0 验证表明，离线缓存、真实 YOLO 标签、真实检测损失、验证指标、训练参数审计和 student-only checkpoint 均已形成闭环。最终回归测试为 36 passed、1 个非阻断 warning。

P1 在 COCO128 和 VisDrone500 两个确定性子集上比较离线特征方案与 YOLO26n 从零训练。COCO128 上，离线方案的 mAP50-95 为 0.009034，高于从零训练的 0.007290，同时训练时间下降 78.75%，计入缓存构建后的冷启动成本仍下降 76.01%。VisDrone500 上，离线方案保留了从零训练约 97.14% 的 mAP50-95，训练时间下降 73.69%，计入缓存构建和预加载后的冷启动成本下降 67.07%。因此，至少一个数据集满足训练成本降低 50% 的 P1 要求。

P2 完成了缓存驻留方式、LatentMixture auxiliary loss 权重和 DINOv3 底座尺寸三类消融。aux 权重 0.01 在单 seed 筛选中最优，但三 seed 复核未确认稳定收益。ViT-L/16 的三 seed 平均 mAP50-95 比 ViT-S/16 高 26.53%，但配对差值的 95% 置信区间跨越 0，同时训练时间、显存和缓存分别增加约 34.10%、94.84% 和 166.67%。因此，本报告不将其表述为统计显著提升，而将其判定为“正向均值趋势尚未确认”。

## 3. 研究问题与验收目标

本课题回答以下三个问题：

1. 冻结 DINOv3 并离线缓存特征后，能否只训练 LatentMixture 和检测头完成真实目标检测训练？
2. 与同设置下的 YOLO26n 从零训练相比，该方案能否降低训练时间和 GPU 成本，同时保留可接受的检测精度？
3. LatentMixture aux 权重和 DINOv3 底座尺寸如何影响精度、训练时间、显存与缓存空间？

| 等级 | 任务要求 | 本项目状态 |
|---|---|---|
| P0 | 离线抽取基础模型多层特征，接入 LatentMixture 和检测头，完成训练评测 | 完成 |
| P1 | 至少两个数据集完成精度、显存、时长三维对照，至少一个数据集成本降低 50% | 完成 |
| P2 | 验证 aux 注册并扫描权重和底座尺寸 | 完成实验与分析 |

P2 的“完成”表示消融、重复实验和证据链完整，不表示所有改动均取得统计显著的正向收益。

## 4. 方法

### 4.1 离线特征缓存

输入图像首先经过固定方形 letterbox 预处理，然后由冻结的 DINOv3 抽取多层特征。每张图像的特征单独保存为 safetensors 文件，manifest 同时记录数据集根目录、样本相对路径、模型 ID、固定 revision、输入尺寸、抽取层、特征形状、letterbox 几何信息、预处理 fingerprint、缓存路径和字节数。

ViT-S/16 主实验使用第 3、7、11 层特征；ViT-L/16 底座消融使用第 7、15、23 层特征。缓存统一保存为 FP16。ViT-L 前向采用 BF16，以规避 FP16 抽取时出现的 NaN/Inf，最终缓存仍转换为 FP16。

### 4.2 检测模型

训练模型由 LatentMixture-P3、LatentMixture-P4、LatentMixture-P5 和 Detect 组成。LatentMixture 将基础模型 token 特征投影并重构为检测金字塔，训练过程使用真实 YOLO 标签以及真实 box loss、classification loss 和 DFL loss。

### 4.3 教师边界

DINOv3 只参与离线缓存构建，不在检测训练阶段加载。训练脚本审计所有可训练参数，只允许 `p3`、`p4`、`p5` 和 `detect` 前缀。checkpoint 审计拒绝出现 `teacher_manager`、`_route_teachers`、`teacher_model`、`dinov3` 和 `siglip` 等教师状态项。ViT-S、ViT-L 以及多 seed 实验生成的 best checkpoint 均通过 student-only 审计。

### 4.4 缓存驻留策略

系统支持 `stream`、`cpu` 和 `gpu` 三种驻留方式。`stream` 每次按样本从磁盘读取；`cpu` 在训练前将 FP16 特征加载到主机内存；`gpu` 将 FP16 缓存全部加载到 GPU，并在进入模型计算前转换为 FP32。

## 5. 实验设置

### 5.1 数据集

| 数据集 | 训练样本 | 验证样本 | 划分 |
|---|---:|---:|---|
| COCO128 deterministic subset | 80 | 20 | 固定 split seed 0 |
| VisDrone500 deterministic subset | 400 | 100 | 固定 split seed 0 |

训练 seed 与 split seed 显式解耦。多 seed 实验固定数据划分，只改变训练 seed 0、1、2。

### 5.2 统一训练配置

| 参数 | 设置 |
|---|---|
| epochs | 100；aux 初筛为 30 |
| imgsz | 640 |
| batch | 8 |
| optimizer | AdamW |
| initial learning rate | 0.001 |
| weight decay | 0.0005 |
| learning-rate schedule | cosine |
| AMP | 关闭 |
| 数据增强 | 关闭 |
| workers | 0 |
| device | CUDA 0 |
| 验证置信度 | 0.001 |

从零训练基线与离线特征方法保持相同的数据划分、训练轮数、图像尺寸、batch、优化器、学习率、随机种子和评测方式。

### 5.3 评价指标

精度指标包括 Precision、Recall、mAP50 和 mAP50-95。成本指标包括纯训练时间、预加载时间、缓存抽取时间、warm cost、cold cost、peak VRAM 和缓存空间。warm cost 只计算检测训练；cold cost 计算缓存抽取、预加载和检测训练。

## 6. P0：最小训练闭环

### 6.1 100 张图缓存准入

| 指标 | 数值 |
|---|---:|
| 样本数 | 100 |
| 每样本缓存 | 3,686,648 bytes |
| 总缓存 | 368,664,800 bytes，约 0.343 GiB |
| 总抽取时间 | 10.512 s |
| 单样本抽取时间 | 0.105 s |
| 抽取峰值显存 | 88,309,248 bytes |

缓存验证重新读取 manifest 和所有缓存文件，检查层名、形状、dtype、样本身份与文件完整性，结果为 PASS。

### 6.2 接口维度

ViT-S/16 缓存三个 384×40×40 的特征层，生成的检测金字塔为 P3 64×80×80、P4 128×40×40、P5 256×20×20。ViT-L/16 对应三个缓存层均为 1024×40×40。

### 6.3 训练和参数审计

P0 训练满足以下契约：离线缓存为真实 DINOv3 特征；训练阶段不加载 teacher；只训练 LatentMixture-P3/P4/P5 和 Detect；使用真实 YOLO 标签与真实检测损失；输出验证指标；checkpoint 不包含 teacher 参数。

COCO128 模型包含 1,356,199 个可训练参数，VisDrone500 ViT-S 模型包含 1,291,619 个可训练参数，ViT-L 模型包含 2,438,499 个可训练参数。所有审计均未发现 unexpected trainable parameters 或 teacher parameters。

### 6.4 回归测试

最终测试覆盖特征缓存、缓存构建器、foundation wrapper、checkpoint 边界、routing contract 和离线缓存检测器，结果为 36 passed、1 warning、总耗时 2.03 秒。唯一 warning 来自测试代码将需要梯度的张量转换为 Python 标量，不影响训练逻辑和验收结论。

## 7. P1：两个数据集的成本—精度对照

### 7.1 COCO128

| 方法 | best epoch | mAP50 | mAP50-95 | 训练时间 | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| YOLO26n 从零训练 | 43 | 0.024280 | 0.007290 | 384.505 s | 2.352 GiB |
| 离线 DINOv3 + LatentMixture | 53 | 0.020744 | 0.009034 | 81.713 s | 1.895 GiB |

离线方案的 mAP50-95 相对基线提高 0.001744。warm training cost 下降 78.75%，cold cost 下降 76.01%，训练加预加载 wall time 下降 77.29%，peak VRAM 下降 19.44%。COCO128 单 seed 结果同时通过精度不劣于基线和训练成本降低 50% 两项门槛。

### 7.2 VisDrone500

| 方法 | best epoch | mAP50 | mAP50-95 | 训练时间 | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| YOLO26n 从零训练 | 44 | 0.028330 | 0.009780 | 1523.700 s | 2.672 GiB |
| 离线 DINOv3 + LatentMixture，aux=0.1 | 19 | 0.029074 | 0.009501 | 400.951 s | 3.580 GiB |

离线方案的 mAP50 高 0.000744，mAP50-95 低 0.000279，mAP50-95 保留比例为 97.14%。warm training cost 下降 73.69%，训练加预加载成本下降 71.83%，cold cost 下降 67.07%，但 peak VRAM 增加 33.98%。VisDrone500 证明了该方法能在基本保留检测精度的同时显著降低训练时间，但 GPU 全驻留策略用显存换取了吞吐。

### 7.3 P1 判定

COCO128 和 VisDrone500 均完成精度、显存和时长对照，两个数据集的 warm/cold 成本下降均超过 50%，因此 P1 实验要求已经满足。

## 8. P2 消融实验

### 8.1 缓存驻留方式消融

| 驻留方式 | 每轮中位时间 | 预加载特征 | 峰值显存 |
|---|---:|---:|---:|
| stream 基准 | 3.096 s | 0 | 约 1.52 GiB |
| CPU preload | 2.845 s | 0.343 GiB | 1.519 GiB |
| GPU preload | 0.533 s | 0.343 GiB | 1.894 GiB |

CPU 预加载只减少约 8.12% 的每轮时间，未达到降低 50% 的目标。GPU 全驻留将每轮中位时间降低约 82.77%，达到性能门槛，因此后续正式实验统一使用 GPU residency。

### 8.2 LatentMixture aux 权重消融

VisDrone500、seed 0、30 epoch 的初筛结果如下。

| aux 权重 | best epoch | best mAP50 | best mAP50-95 |
|---:|---:|---:|---:|
| 0.01 | 17 | 0.029526 | 0.010213 |
| 0.10 | 19 | 0.029799 | 0.009489 |
| 0.00 | 16 | 0.028845 | 0.009455 |
| 0.05 | 22 | 0.028090 | 0.009418 |
| 0.20 | 16 | 0.026702 | 0.009112 |

30 epoch 初筛选择 aux=0.01 进入 100 epoch 复核。seed 0 下，从零训练、aux=0.10 和 aux=0.01 的 mAP50-95 分别为 0.009780、0.009501 和 0.009951。aux=0.01 相对 aux=0.10 提高 0.000451，相对从零训练提高 0.000171。

三 seed 配对复核结果如下。

| seed | 从零训练 mAP50-95 | aux=0.01 mAP50-95 | 差值 |
|---:|---:|---:|---:|
| 0 | 0.009780 | 0.009951 | +0.000171 |
| 1 | 0.010300 | 0.008387 | -0.001913 |
| 2 | 0.009740 | 0.008811 | -0.000929 |

baseline mean±std 为 0.009940±0.000312；aux=0.01 为 0.009050±0.000809；paired delta mean 为 -0.000890，95% CI 为 [-0.003481, 0.001700]，method wins 为 1/3。因此，aux=0.01 的单 seed 正向结果没有在三 seed 中得到确认。

### 8.3 aux 注册和梯度验证

单测通过统一 mixture auxiliary loss 收集函数读取 LatentMixture auxiliary loss，并对总损失反向传播。LatentMixture token projection 和 Detect 分类分支均获得非空梯度，训练日志中的 `train/mixture_aux_loss` 为有限非零值。因此，aux 已进入统一损失收集和反向传播通道，而不是只存在于配置文件中的无效参数。

### 8.4 DINOv3 底座尺寸消融

在 VisDrone500 上比较 ViT-S/16 和 ViT-L/16。两组均使用 100 epoch、aux weight 0.0、固定 split seed 0，并运行训练 seed 0、1、2。

| seed | ViT-S mAP50-95 | ViT-L mAP50-95 | 配对差值 |
|---:|---:|---:|---:|
| 0 | 0.009877 | 0.010248 | +0.000371 |
| 1 | 0.009714 | 0.014782 | +0.005068 |
| 2 | 0.009471 | 0.011744 | +0.002272 |

| 指标 | ViT-S/16 | ViT-L/16 | 相对变化 |
|---|---:|---:|---:|
| mAP50-95 mean | 0.009687 | 0.012258 | +26.53% |
| mAP50-95 std | 0.000204 | 0.002310 | — |
| 训练时间均值 | 365.573 s | 490.227 s | +34.10% |
| cold time 均值 | 396.359 s | 516.150 s | +30.22% |
| 峰值显存均值 | 3.581 GiB | 6.978 GiB | +94.84% |
| GPU 预加载缓存 | 1.717 GiB | 4.578 GiB | +166.67% |

配对 mAP50-95 差值均值为 0.002570，标准差为 0.002363，95% CI 为 [-0.003299, 0.008439]，ViT-L wins 为 3/3。ViT-L 在三个 seed 上均取得更高 mAP50-95，但由于只有三个重复，95% 置信区间仍跨越 0。因此正式结论是：ViT-L 显示正向均值趋势，但尚不能确认统计显著收益。

## 9. 综合分析

离线方案的主要收益是将固定基础模型的特征抽取成本从每个 epoch 移到一次性预处理。相同缓存可被多个训练 seed、aux 权重和检测头配置复用，实验次数越多，一次性缓存成本越容易被摊薄。

GPU residency 是取得训练时间优势的关键。如果按样本从磁盘读取，缓存 I/O 会抵消冻结基础模型带来的收益；CPU preload 仍受主机到设备传输限制。GPU residency 能显著提高吞吐，但会增加显存占用，因此不能只报告训练时间而忽略显存成本。

aux=0.01 在单 seed 实验中表现最好，但三 seed 复核没有重现该优势。这是有效的负结果，说明当前数据规模下 aux 权重对训练 seed 敏感，不能将 0.01 表述为普遍最优值。

ViT-L 在三个 seed 上均高于 ViT-S，表明更大的基础特征可能具有更强的检测迁移能力。但 ViT-L 同时带来接近两倍的峰值显存和约 2.67 倍的缓存空间。考虑置信区间仍跨 0，当前证据支持将其作为候选升级方向，不支持直接设为默认方案。

## 10. 已知局限与风险

1. COCO128 和 VisDrone500 都是小规模确定性子集，绝对 mAP 较低，结论不能直接外推到完整 COCO 或完整 VisDrone。
2. COCO128 P1 主对照为单 seed；VisDrone 的主要机制结论补充了三 seed，但仍只有三个重复。
3. aux=0.01 的单 seed 最优没有通过三 seed 稳定性验证。
4. ViT-L 的配对差值均值为正，但 95% CI 跨 0，不能宣称统计显著。
5. GPU residency 通过增加显存占用换取训练速度；ViT-S 在 VisDrone 上的峰值显存高于从零训练基线。
6. cold cost 以缓存只构建一次为前提。若多个实验共享缓存，收益会进一步扩大；若每次都重建缓存，收益会降低。
7. ViT-L 实验使用固定 revision 的公开社区镜像完成。本报告记录模型身份和校验信息，但不在 PR 中重新分发基础模型权重或派生特征缓存。正式公开发布前应再次核对 DINOv3 许可证和模型来源授权。
8. 大型缓存、基础模型权重和完整 checkpoint 未提交到 Git。仓库只提供构建命令、manifest、机器可读结果和 checksum。

## 11. 结论

本项目完成了冻结 DINOv3、离线多层特征缓存、LatentMixture 三尺度适配和 Detect 检测头训练的完整闭环。训练阶段不加载基础模型，只更新 LatentMixture 和检测头，checkpoint 中不包含教师参数。

在 COCO128 和 VisDrone500 上，离线特征方法均显著减少训练时间。COCO128 同时获得更高 mAP50-95 和更低峰值显存；VisDrone500 保留约 97.14% 的 mAP50-95，并将冷启动成本降低约 67.07%，但 GPU residency 增加了显存占用。因此，离线冻结基础模型是一条有效的训练成本压缩路线，其资源收益取决于缓存驻留策略。

P2 消融表明，aux 权重和基础模型尺寸均存在明显的 seed 与资源权衡。aux=0.01 的单 seed 优势没有在三 seed 中得到确认；ViT-L 在三个 seed 上均取得更高精度，但置信区间跨 0，并付出更高训练、显存和缓存成本。最终建议保留 ViT-S/16 + GPU residency 作为成本优先方案，将 ViT-L/16 作为精度优先候选，而不是直接替换默认底座。

## 12. 验收映射

| 验收项 | 证据 | 结论 |
|---|---|---|
| 100 张图可复现缓存 | 本报告中的缓存准入结果、接口维度与资源统计 | PASS |
| 接口维度和资源报告 | cache manifest、interface dimensions、resource report | PASS |
| 只训练 LatentMixture 和 Detect | trainability audit | PASS |
| teacher 不进入训练和 checkpoint | checkpoint tests 和 6 个 best checkpoint 审计 | PASS |
| 两个数据集三维对照 | COCO128、VisDrone500 P1 JSON | PASS |
| 至少一个数据集成本下降 50% | 两个数据集 warm/cold cost 均超过 50% | PASS |
| aux 权重扫描 | 五档 30 epoch + 选中配置 100 epoch | PASS |
| aux 进入统一损失通道 | 单测、梯度和训练遥测 | PASS |
| 底座尺寸扫描 | ViT-S/ViT-L × 三 seed | PASS |
| 统计与局限声明 | 配对差值、均值、标准差、95% CI | PASS |
| 最终回归测试 | 36 passed、1 warning | PASS |

## 13. 证据文件

P0、P1、P2 的主要实验设置、结果、消融与统计结论已统一汇总在本报告中。原始运行 JSON、SHA256 清单、commit/diff snapshot、准入记录和机器本地实验产物属于项目内部过程证据，由参与者单独归档，不作为 YOLO-Master 上游功能的一部分合并。

## 14. 复现入口

缓存构建入口为 `python scripts/d1_build_feature_cache.py --help`，离线检测训练入口为 `python scripts/d1_train_cached_detector.py --help`，正式训练推荐使用 `--cache-residency gpu`。内部准入记录、原始运行日志、机器本地路径及一次性实验产物不随上游 PR 合并；公开评审材料以本报告、功能代码、复现脚本和单元测试为准。
