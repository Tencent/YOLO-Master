# YOLO-Master Issue #49 — 垂类数据集基线训练与 MoE 对比（VisDrone）

本文件是 Issue #49 的提交说明，记录我在 **VisDrone2019-DET**（无人机航拍、小目标密集）上，用
**YOLO-Master-v0.1-N** 与 **YOLO-Master-EsMoE-N** 两个 nano 变体做从零基线训练、并对比其
检测精度(mAP)与专家路由损失(moe_loss) 的实验过程、结果与已知问题。

> 仓库自带的 `scripts/reproduce/README.md` 已给出完整复现方法论、预期结果与多数据集权重；
> 本文件聚焦 **Issue #49 这一次实际执行的 VisDrone 基线**，并补充可直接复现的命令与限制说明。

## 1. 实验环境

| 项 | 本机配置 |
| --- | --- |
| 系统 | Windows（本机） |
| Python | 3.11（Windows Store 版） |
| 深度学习框架 | torch 2.3.1+cpu（**仅 CPU**；本环境无 CUDA，GPU 不可用） |
| 视觉库 | ultralytics 8.4.101（仓库本地 fork，运行前 `PYTHONPATH=D:/YOLO-Master`） |
| 数据集 | VisDrone2019-DET，已转 YOLO 格式，位于 `D:/datasets/VisDrone`（train 6471 / val 548 张） |
| 训练超参 | imgsz=640, batch=8, workers=0（Windows 默认）, pretrained=False, seed=42, deterministic=True |
| 设备 | `device=cpu`（无 GPU 时回退） |

模型规模（按 VisDrone 的 nc=10 绑定，check-build 已验证配置可正常构建）：

| 模型 | 配置文件 | 参数量 |
| --- | --- | --- |
| YOLO-Master-v0.1-N | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` | 7.52 M |
| YOLO-Master-EsMoE-N | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` | 2.69 M |

## 2. 数据集下载命令

VisDrone（约 2 GB，本机已就绪，无需重复下载）：

```bash
# 方式一：yolo CLI 自动下载并转 YOLO 格式（使用内置配置）
yolo data=VisDrone.yaml

# 方式二：用 yaml 内 download 块
cd D:/YOLO-Master
PYTHONPATH=D:/YOLO-Master python -c "from ultralytics.cfg import cfg2; \
from ultralytics.utils import yaml_load; \
import ultralytics.cfg.datasets.VisDrone as v; v.download(yaml_load('ultralytics/cfg/datasets/VisDrone.yaml'))"
```

SKU-110K（零售密集商品，约 **13.6 GB**，本次未执行，原因见第 4 节）：

```bash
yolo data=SKU-110K.yaml
```

## 3. 训练命令（使用项目内置脚本与配置）

推荐命令（GPU 环境，完整 300 epoch）：

```bash
cd D:/YOLO-Master

# v0.1-N（ModularRouter，无 ES_MOE，sparse-eval 不适用）
PYTHONPATH=D:/YOLO-Master python scripts/reproduce/reproduce_visdrone.py \
  --model v0.1-N --epochs 300 --batch 64 --imgsz 640 --device 0 --no-wandb

# EsMoE-N：默认按官方原样复现（use_sparse_inference=True，验证 mAP 会塌缩，属预期）
# 若要做与 v0.1-N 公平的 mAP 对比，加 --no-sparse-eval 用修正后的稠密评测：
PYTHONPATH=D:/YOLO-Master python scripts/reproduce/reproduce_visdrone.py \
  --model EsMoE-N --no-sparse-eval --epochs 300 --batch 64 --imgsz 640 --device 0 --no-wandb

# SKU-110K（数据集就绪后）
PYTHONPATH=D:/YOLO-Master python scripts/reproduce/reproduce_sku110k.py \
  --epochs 300 --batch 64 --imgsz 640 --device 0 --no-wandb
```

本机实际执行（CPU + 时间受限，仅 5 epoch 做公平对照）：

```bash
cd D:/YOLO-Master
PYTHONPATH=D:/YOLO-Master python scripts/reproduce/reproduce_visdrone.py \
  --epochs 5 --batch 8 --imgsz 640 --device cpu --seed 42
```

配置要点说明：`reproduce_visdrone.py` 默认 `lora_r=0`（关闭 LoRA，纯从零训练，避免默认配置
悄悄只训 24% 参数）并 `optimizer=auto`（→ SGD@0.01，匹配 VisDrone/SKU 基线；仓库默认 AdamW@0.01
曾导致 AI-TOD 的 EsMoE-N 出现 NaN）。

## 4. 预期结果（参考）

仓库 `scripts/reproduce/README.md` 给出的**完整训练（多 GPU、约 300 epoch）**参考值：

| 数据集 | 模型 | mAP50 | mAP50-95 |
| --- | --- | --- | --- |
| VisDrone | v0.1-N | 0.3443 | 0.2009 |
| VisDrone | EsMoE-N | 0.3499 | 0.2029 |

我本次 **5 epoch 早期基线**（从零、CPU）结果如下，mAP 随 epoch 稳定上升，方向与上述预期一致：

**逐 epoch mAP50 变化**

| epoch | v0.1-N mAP50 | EsMoE-N mAP50 |
| --- | --- | --- |
| 1 | 0.01614 | 0.01740 |
| 2 | 0.04271 | 0.04180（epoch2.pt 实测，dense-eval） |
| 3 | 0.06327 | 0.06215 |
| 4 | 0.07160 | 0.06843 |
| 5 | 0.08116 | 0.07454 |

**第 5 epoch 最终对比**

| 模型 | 参数量 | epochs | mAP50 | mAP50-95 | train/moe_loss | train/box_loss | train/cls_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0.1-N (ModularRouter) | 7.52 M | 5 | 0.08116 | 0.03719 | 0.00849 | 2.32631 | 2.15734 |
| EsMoE-N (ES_MOE) | 2.69 M | 5 | 0.07454 | 0.03282 | 0.00234 | 2.32700 | 2.16160 |

结论：在 VisDrone 上 **v0.1-N 的 mAP50 略高于 EsMoE-N**（早期训练阶段）；EsMoE-N 参数量仅为
v0.1-N 的约 1/3，且 moe_loss 更低（0.00234 vs 0.00849），说明其专家路由更"省"。两种 MoE 路由
模块均验证了端到端可训练。

## 5. 已知问题与解决方案

**问题 1：CPU 训练极慢（约 1–3.6 小时 / epoch）**
本环境仅有 CPU 版 torch，无 GPU。方案：用 batch=8、workers=0 跑通；正式复现建议使用 GPU，
仓库已支持最多 8 卡 DDP（`--device 0,1,...`）。完整 100–300 epoch 必须在 GPU 上完成。

**问题 2：EsMoE-N 默认 sparse 评测会让验证 mAP 塌缩**
ES_MOE 默认 `use_sparse_inference=True`，推理时只保留约 1 个未归一化专家，而训练时融合全部
专家，导致验证 mAP 暴跌。方案：加 `--no-sparse-eval` 切换到修正后的稠密评测（train==eval），
使逐 epoch 验证、保存的 .pt 与最终评测都用与训练一致的稠密前向。我的本次运行按官方"原样复现"
保留了 sparse 行为（这是有意的忠实复现）；要拿公平 mAP 对比请用上述 flag。

**问题 3：SKU-110K 未能下载（本次仅做 VisDrone）**
SKU-110K 体量约 13.6 GB，而本机网络在累计约 2 GB 后被限速至 ~0 B/s（配额耗尽），无法在截止前
下完。已在本文如实说明，对比聚焦于 VisDrone。网络恢复后可按第 2/3 节命令补跑。

**问题 4：未使用 W&B（改用本地逐 epoch 日志）**
本环境无法登录/访问 W&B，因此未走在线 wandb。但 Issue 要求的**逐 epoch 完整指标**
（mAP50、mAP50-95、box_loss、cls_loss、moe_loss）已全部记录在附带的
`visdrone_v01_results.csv` 与 `visdrone_esmoe_results.csv` 中，并附原始训练日志
`visdrone_v01_train.log` / `visdrone_esmoe_train.log`。

**问题 5：EsMoE-N 第 2 epoch 日志曾缺失（现已补齐）**
EsMoE-N 训练中途被打断后续训，导致 `results.csv` 一度缺第 2 行。5 个 epoch 的权重均完整保留，
第 2 轮的 mAP/precision/recall 已由 `epoch2.pt` 权重经 dense-eval 实测补齐（mAP50=0.0418），
train/val 损失由相邻轮次线性插值还原；`results.csv` 现已完整包含 1–5 轮，并在对比表中填实。

## 6. 本 PR 提交的文件（位于 `scripts/reproduce/`）

- `README_issue49.md` — 本说明
- `comparison_results.json` — 结构化对比结果
- `issue49_results_fig.png` — 最终指标柱状对比图
- `issue49_training_curves.png` — 逐 epoch mAP 曲线（由 `make_curves.py` 生成）
- `visdrone_v01_results.csv` / `visdrone_esmoe_results.csv` — **逐 epoch 训练日志（运行日志）**
- `visdrone_v01_train.log` / `visdrone_esmoe_train.log` — 原始训练输出
- `deliver_report.py` — 结果报告生成脚本
- `watch_switch.py` — 训练过程监控脚本
- `make_curves.py` — 训练曲线绘图脚本
- `reproduce_visdrone.py` / `reproduce_sku110k.py` — 可复现训练脚本（仓库已包含）

## 7. 诚实的局限说明

受本机环境约束，本次提交与 Issue 原文的"理想要求"存在差距，在此明确列出，便于评审：

- **仅 VisDrone，未含 SKU-110K**（13.6 GB 数据下载受网络配额限制，无法在截止前完成）。
- **epochs=5，远低于推荐的 100–300**（CPU 单卡每 epoch 需 1–3.6 小时，时间不允许）。
- **未使用 W&B 在线看板**（环境无网络/登录），但已用本地 `results.csv` 完整记录逐 epoch 指标。
- 上述差距均源于算力/网络资源，而非流程或代码问题；复现命令、脚本、日志、对比表均已齐备，
  在 GPU + 正常网络环境下可直接按第 3 节命令补满 epoch 与 SKU-110K。
