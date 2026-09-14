# 主题 D2｜Foundation 蒸馏：教师特征 → YOLO-Master

用最小可行的蒸馏，回答基础模型特征能否改善小型 YOLO backbone。**探索型课题，允许负结果**——交付的是可信结论与证据链，不是必须涨点。

| | |
|---|---|
| **基线 commit** | `e9ac08b`（= `upstream/main`） |
| **教师** | `facebook/dinov3-vits16-pretrain-lvd1689m`（冻结，仅训练期） |
| **学生** | `yolo26-master-n`，蒸馏 P4（第 19 层） |
| **P0 状态** | ✅ 已闭环（2026-08-25，CUDA）；证据 [`results/p0_train_ok/`](results/p0_train_ok/) |
| **权重标定** | ✅ 已闭环。最终 VOC 权重为 A `3.34277`、B `3.00161`、C `3.11066`；完整精度见 [`findings.md`](docs/p1/findings.md) |
| **可学性验证** | ✅ 探针 B 通过：KD 目标留出集降 **59.4%**，backbone 贡献 53%，不是投影器把戏（[`results/probe_b_learnability.json`](results/probe_b_learnability.json)） |
| **P1 状态** | ✅ **已完成**：VOC A/B/C/off × 3 seeds，共 12 次 400-epoch 运行；结论见 [`findings.md`](docs/p1/findings.md) |
| **Owner** | *待定（尚未正式组队）* |

## 目录结构

```
experiments/d2/
├── README.md              本文件：总览、环境安装、实验流程
├── docs/
│   ├── kd_explained.md    背景：蒸馏机制白话讲解
│   ├── design.md          跨阶段设计文档、判读线、归因顺序
│   ├── limitations.md     已知局限与降级方案
│   ├── p0/                P0：跑通链路（✅ 已闭环）
│   ├── p1/                P1：2×2 对照矩阵 + 权重标定（✅ 已完成）
│   └── p2/                P2：消融（未开始）
├── scripts/               五个脚本，见下方「脚本」表
├── configs/p1_voc/        P1 VOC 五份配置
├── env/                   训练依赖与硬件快照
├── p1_voc_matrix.csv      P1 VOC 矩阵
└── results/               归档证据
```

## 按阶段

| 阶段 | 状态 | 入口 |
|---|---|---|
| P0 | ✅ 已闭环 | [`docs/p0/`](docs/p0/README.md) |
| P1 | ✅ 已完成 | [`docs/p1/findings.md`](docs/p1/findings.md) |
| P2 | ⬜ 未开始 | [`docs/p2/`](docs/p2/README.md) |

## 文档

| 文件 | 内容 |
|---|---|
| [`kd_explained.md`](docs/kd_explained.md) | **先读这份**：蒸馏机制的白话讲解，不假设 YOLO / DINOv3 / KD 背景 |
| [`design.md`](docs/design.md) | HEAD 能力地图、设计选择与依据、P0 定义、**判读线**、负结果归因顺序 |
| [`p1_voc_matrix.csv`](p1_voc_matrix.csv) | P1 VOC 完整 2×2 矩阵；本轮完成 A/B/C + 共享基线 × 3 seed = 12 次运行 |
| [`configs/p1_voc/`](configs/p1_voc/) | P1 VOC 五份配置；由 `validate_pair.py` 机械校验预算与对照轴 |
| [`kd_gradient_analysis.md`](docs/p1/kd_gradient_analysis.md) | **权重标定为什么失败**：预注册准则的判读结果、KD 项未被优化的证据、`task_ratio` 作为影响力代理无效的机制，以及改用梯度比的标定方案 |
| [`limitations.md`](docs/limitations.md) | 已知局限、环境限制、风险触发与降级方案 |
| [`results/`](results/) | 归档证据。仓库根 `.gitignore` 忽略 `results.csv` / `args.yaml` / `*.log`，故归档时改名为 `metrics.csv` / `resolved_args.yaml` |

## 脚本

| 脚本 | 回答什么问题 | 什么时候跑 |
|---|---|---|
| [`record_environment.py`](scripts/record_environment.py) | **这次实验是谁跑的**——commit、dirty 状态、依赖版本、教师 revision 与权重 hash、输入文件 hash | 每批实验**开跑前** |
| [`validate_pair.py`](scripts/validate_pair.py) | **配置有没有混杂**——五份配置除对照轴外是否逐字相同 | 改完配置后 |
| [`collect_runs.py`](scripts/collect_runs.py) | **跑出了什么**——归档 + 跨 run 指标对比 + **事后**混杂核查 | 每批实验**跑完后** |
| [`kd_gradient_probe.py`](scripts/kd_gradient_probe.py) | **KD 到底推动了学生多少**——各损失形式的梯度尺度（`--mode kd`，无需 GPU）；`w` 该取多大（`--mode ratio`，需教师权重） | 锁定 `foundation_loss_weight` 之前 |
| [`plot_p1_findings.py`](scripts/plot_p1_findings.py) | **结果趋势是什么**——从归档 CSV 重建学习曲线、配对差值和最终 seed 图 | 更新 P1 归档后 |

两个检查的分工是刻意的：`validate_pair.py` 看的是配置文件，`collect_runs.py` 看的是跑完的
`args.yaml`——后者才记录了 trainer 真正解析出的值（`optimizer: auto` 展开成什么、
命令行覆盖了什么、没写的字段取了哪个默认值）。**配置一致不等于实际跑的一致**，
所以两道检查都要过。

## 环境安装

```bash
git clone -b d2 https://github.com/and-yliu/YOLO-Master.git
cd YOLO-Master

pip install -e .
pip install "transformers>=5"     # 4.x 不导出 DINOv3ViTBackbone，见 limitations.md §2.1
hf auth login                     # DINOv3 权重受控，需已接受许可的账号 token
```

DINOv3 许可需本人在 [模型页](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) 登录并同意条款后才会放行，`hf auth login` 只是把 token 写到本地。

验证环境就绪：

```bash
python -c "from transformers import DINOv3ViTBackbone; print('ok')"
python -c "from ultralytics.nn.foundation import DINOv3Teacher; print('ok')"
```

## 实验流程

任何一批实验都走同样的四步。**跳过任何一步的结果不作为证据采信**——
理由见每步下方。

```bash
# 1. 跑之前：确认配置层无混杂
python experiments/d2/scripts/validate_pair.py

# 2. 开跑前：记录本次实验的身份
python experiments/d2/scripts/record_environment.py \
  --out experiments/d2/results/environment_<批次名>.json

# 3. 跑训练（见下方各批次的命令）
yolo train ... project=d2/<批次> name=<run名>

# 4. 跑完后：归档 + 汇总 + 事后混杂核查
python experiments/d2/scripts/collect_runs.py runs/detect/d2/<批次>/* --label <批次名>
```

| 步 | 不做会怎样 |
|---|---|
| 1 | 某一格的 `epochs`/`batch`/`lr0` 悄悄不同，训练照跑、数字照出，但差异无法归因到蒸馏 |
| 2 | 运行无法追溯到代码版本、依赖版本与教师权重 |
| 4 | 「无混杂」只停留在配置层面，未在**实际解析出的参数**上验证 |

**第 1 步只适用于从配置文件跑的批次**（P1 的 15 次）。命令行驱动的探路实验
（如权重扫描）不读 `configs/`，第 1 步跳过，改由第 4 步的事后核查兜底。

**混杂为什么要查两次**：[`validate_pair.py`](scripts/validate_pair.py) 看的是配置文件，
`collect_runs.py` 看的是跑完的 `args.yaml`——后者才记录 trainer 真正解析出的值
（`optimizer: auto` 展开成什么、命令行覆盖了什么、未写字段取了哪个默认值）。
**配置一致不等于实际跑的一致。**

第 4 步会把每个 run 的 `results.csv` / `args.yaml` 归档进 `results/` 并改名为
`metrics.csv` / `resolved_args.yaml`（仓库根 `.gitignore:205-206` 按文件名忽略了原名），
同时生成一张跨 run 对比表和一份混杂核查结论。

控制台输出不落盘（Ultralytics 不写日志文件），需要逐 step 行为时自行 `2>&1 | tee`。

## 复现 P0

P0 = **用配置驱动仓库自身的训练路径跑通一次真实训练**，并核对 teacher / tap / projector / loss 与日志。

```bash
yolo train model=ultralytics/cfg/models/26/yolo26-master-n.yaml \
  data=coco128.yaml epochs=3 imgsz=256 batch=4 workers=0 device=0 \
  seed=17 deterministic=True pretrained=False amp=False plots=False \
  foundation_enabled=True foundation_teacher=dinov3 \
  foundation_model=facebook/dinov3-vits16-pretrain-lvd1689m \
  foundation_loss_weight=0.05 \
  project=d2/p0 name=train_ok
```

未显式给出的 foundation 参数取 `default.yaml` 默认值：`relational` 损失、`align_dim 256`、
`target_levels [p4]`、`constant` 权重调度。

证据已归档到 [`results/p0_train_ok/`](results/p0_train_ok/)。

> 该批次跑在 `collect_runs.py` 之前，因此归档是手工完成的，且**没有 `environment.json`**——
> torch 版本与 CUDA driver 是从控制台输出人工读取的（见 `design.md §5.3`）。
> 这正是上述流程要解决的问题；P1 起不再手工归档。

### 怎么读这份证据

```bash
column -s, -t experiments/d2/results/p0_train_ok/metrics.csv
```

四项核对全部由 CSV 直接证成：**存在 `train/foundation` 等 11 个 foundation 列**
（只有 wrapper 会产生）、**表头含 `foundation`**（KD 进了 loss 向量）、
**三个 epoch 数值有限非零**、**CSV 本身即指标落盘**。

最具判别力的是权重恒等式：

```
foundation_relational_raw × loss_weight × batch_size
0.314517 × 0.05 × 4 = 0.0629034 = train/foundation_loss    ✓ 三个 epoch 全部精确相等
```

它证明配置里的 `0.05` **真的作用到了被优化的目标上**，而不是被算出来打印在旁边。

两项看着像故障但符合设计：`foundation_cosine_raw = 0`（relational 分支下 cosine 分量按设计为零）、
`val/foundation = 0`（eval 模式补零，`foundation_distill_model.py:943`）。

> **不构成任何精度主张。** 本次 3 epoch、`pretrained=False`，mAP50-95 全程为 0，
> 模型尚未开始收敛。跑通只证明链路可优化，不证明泛化或 mAP 改善。
> 是否涨点必须由 P1 的同预算多 seed 配对回答。

### P0 暴露的问题

| 问题 | 数字 | 影响 |
|---|---|---|
| KD 权重过小 | `foundation_task_ratio ≈ 0.4%` | P1 若判 no-go 将无法归因，需先做权重标定（[`design.md §5.5`](docs/design.md)） |
| MoE 辅助损失压倒 KD | `mixture_aux` 是 `foundation` 的 26 倍 | 为负结果归因清单第 5 条提供了实测依据 |
| optimizer 与 P1 配置不一致 | P0 用 `auto`，P1 配置写 `SGD` | P0 数值不可与 P1 直接比较 |

## 判读线

```
|Δ mAP50-95| < 0.003  且  95% 置信区间包含 0   →   no-go
```

**`0.003` = 0.3 个百分点**，因为 Ultralytics 的 `metrics/mAP50-95(B)` 是 0–1 刻度。
按字面取 `0.3` 会让判读线宽达 30 个百分点、失去全部判别力。

本判读线在产生任何 mAP 数字之前提交。结果模糊时**只增加 seed，不移动判读线**。详见 [`design.md §6`](docs/design.md#6-判读线提前锁定)。
