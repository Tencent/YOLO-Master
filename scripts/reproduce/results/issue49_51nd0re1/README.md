# Issue #49: VisDrone 300-Epoch Reproduction

Related to [#49](https://github.com/Tencent/YOLO-Master/issues/49).

本目录记录在 VisDrone2019-DET 上从零训练 YOLO-Master-v0.1-N 与
YOLO-Master-EsMoE-N 的个人复现实验。三次运行均完成 300 epochs，并通过公开 W&B Report
和原始终端日志保留逐 epoch 指标。

> SKU-110K 数据已经下载，但受本次可用 GPU 时长、存储和实验截止时间限制，未执行训练与验证。
> 因此本文的实测结论仅覆盖 VisDrone，不包含任何推测或补造的 SKU-110K 指标。

## 交付文件

| 文件 | 内容 |
| --- | --- |
| `README.md` | 配置、命令、结果对比、已知问题 |
| `visdrone_epochs.csv` | 三次运行各 300 epoch，共 900 行逐轮指标 |
| `visdrone_training_logs_300e_b8.zip` | 三次运行的完整 `nohup` 原始日志 |

公开 W&B Report：<https://api.wandb.ai/links/51nd0re1-/hep6f7n1>

直接运行链接：

- v0.1-N：<https://wandb.ai/51nd0re1-/yolo-master-reproduce/runs/ebm62hkt>
- EsMoE-N（默认评估请求）：<https://wandb.ai/51nd0re1-/yolo-master-reproduce/runs/dr9ddccj>
- EsMoE-N（dense 评估请求）：<https://wandb.ai/51nd0re1-/yolo-master-reproduce/runs/eyutrjs9>

## 实验配置

| 项目 | 配置 |
| --- | --- |
| 数据集 | VisDrone2019-DET，train 6,471 / val 548 |
| 验证集实例数 | 38,759 |
| 输入尺寸 | 640 |
| Epochs | 300 |
| Batch size | 8 |
| Workers | 8 |
| 初始化 | `pretrained=False`，`lora_r=0` |
| 随机种子 | 42，`deterministic=True` |
| 设备 | 单卡 NVIDIA A40，45,403 MiB |
| 软件 | Python 3.10.18，PyTorch 2.4.0+cu118，Ultralytics 8.4.101 |
| AMP | 开启且预检查通过 |
| 优化器 | `optimizer=auto`，实际选择 MuSGD（lr=0.01，momentum=0.9） |
| W&B project | `yolo-master-reproduce` |

数据扫描报告 0 张损坏图像，并从 4 张训练图像中各移除 1 个重复标签。三份日志均未发现
`Traceback` 或 `ERROR`。

实际服务器已提前解压数据，因此运行时通过 `--data` 使用未提交的
`dataset/VisDrone.local.yaml`。该 YAML 只改变本地路径，类别和 train/val 划分与内置
[`VisDrone.yaml`](../../../../ultralytics/cfg/datasets/VisDrone.yaml) 一致。标准脚本默认仍使用
内置配置。

## 数据集准备

安装当前仓库，避免误用环境中另一份 `ultralytics`：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

使用内置 YAML 显式触发数据检查和自动下载：

```bash
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('SKU-110K.yaml', autodownload=True)"
```

训练脚本直接传入 `VisDrone.yaml` 或 `SKU-110K.yaml` 时，若数据缺失也会执行各 YAML 中的
`download` 配置。已手动解压的数据可使用等价的本地 YAML，并通过 `--data /abs/path/to/file.yaml`
覆盖默认值。

## 复现命令

v0.1-N：

```bash
python -u scripts/reproduce/reproduce_visdrone.py \
  --data VisDrone.yaml --model v0.1-N \
  --epochs 300 --imgsz 640 --batch 8 \
  --device 0 --workers 8 \
  --wandb --wandb-mode online \
  --wandb-project yolo-master-reproduce
```

EsMoE-N，按发布配置请求默认评估：

```bash
python -u scripts/reproduce/reproduce_visdrone.py \
  --data VisDrone.yaml --model EsMoE-N \
  --epochs 300 --imgsz 640 --batch 8 \
  --device 0 --workers 8 --sparse-eval \
  --wandb --wandb-mode online \
  --wandb-project yolo-master-reproduce
```

EsMoE-N，dense evaluation 对照：

```bash
python -u scripts/reproduce/reproduce_visdrone.py \
  --data VisDrone.yaml --model EsMoE-N \
  --epochs 300 --imgsz 640 --batch 8 \
  --device 0 --workers 8 --no-sparse-eval \
  --wandb --wandb-mode online \
  --wandb-project yolo-master-reproduce
```

SKU-110K 后续补测命令如下，但本次未执行：

```bash
python -u scripts/reproduce/reproduce_sku110k.py \
  --data SKU-110K.yaml --model v0.1-N \
  --epochs 300 --imgsz 640 --batch 8 --device 0 --workers 8 \
  --wandb --wandb-mode online --wandb-project yolo-master-reproduce

python -u scripts/reproduce/reproduce_sku110k.py \
  --data SKU-110K.yaml --model EsMoE-N \
  --epochs 300 --imgsz 640 --batch 8 --device 0 --workers 8 \
  --no-sparse-eval \
  --wandb --wandb-mode online --wandb-project yolo-master-reproduce
```

## VisDrone 结果

以下为 W&B 中第 300 epoch 的完整精度指标。损失越低越好，mAP 越高越好。

| 模型 | 评估请求 | 参数量 | 总耗时 | mAP50 | mAP50-95 | train/box_loss | train/cls_loss | train/moe_loss | val/box_loss | val/cls_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.1-N | 默认 | 7,516,742 | 38.582 h | 0.33565 | 0.19192 | 1.26900 | 0.77415 | 0.99992 | 1.37975 | 0.97624 |
| EsMoE-N | `--sparse-eval` | 2,664,122 | 34.621 h | **0.34123** | **0.19570** | 1.27215 | 0.77684 | 1.00000 | **1.37240** | **0.95565** |
| EsMoE-N | `--no-sparse-eval` | 2,664,122 | 34.911 h | **0.34123** | **0.19570** | 1.27215 | 0.77684 | 1.00000 | **1.37240** | **0.95565** |

以默认评估请求的 EsMoE-N 与 v0.1-N 对比：

| 指标 | v0.1-N | EsMoE-N | 变化 |
| --- | ---: | ---: | ---: |
| mAP50 | 0.33565 | 0.34123 | +0.00558（+1.66%） |
| mAP50-95 | 0.19192 | 0.19570 | +0.00378（+1.97%） |
| 参数量 | 7,516,742 | 2,664,122 | -64.56% |
| 总耗时 | 38.582 h | 34.621 h | -10.27% |

本次单种子实验中，EsMoE-N 以 v0.1-N 约 35.44% 的参数量取得略高的最终 mAP。
差值只有 0.378 至 0.558 个百分点，不能据单次运行判断统计显著性。总耗时是端到端训练耗时，
不是经过预热和重复测量的推理吞吐基准。

两条 EsMoE-N 运行的逐轮指标与最终结果完全一致。`dense_val` 记录的是命令请求的模式；
在本次所用发布配置中，两条命令未形成可观测的有效 sparse/dense 数值差异，因此这里不据此声称
Top-K sparse inference 的精度或速度收益。

`train/moe_loss` 是训练器中 mixture auxiliary loss 的 W&B 规范化别名。`val/moe_loss=0`
表示当前验证日志未发布独立辅助损失项，不表示验证前向没有使用 MoE。

## 逐 Epoch CSV

[`visdrone_epochs.csv`](visdrone_epochs.csv) 包含以下列：

`run_id, model, eval_request, epoch, map50, map50_95, precision, recall, box_loss, cls_loss, dfl_loss, moe_loss`

CSV 从原始控制台日志提取，因此数值保留终端显示精度；更高精度值和交互曲线以 W&B 为准。
每份原始日志都包含一个额外的重复 epoch 1 控制台块。CSV 按 epoch 编号保留最后一条完整记录，
最终得到每次运行严格 300 行，对应 epoch 1 至 300。

## 原始日志与校验

[`visdrone_training_logs_300e_b8.zip`](visdrone_training_logs_300e_b8.zip) 大小为
12,948,258 bytes，SHA-256：

`72212AE2765C87C767882F7AB00C11A96AB23A327BFEE81AFBE94DE19039D91A`

| ZIP 内文件 | 原始大小 | SHA-256 |
| --- | ---: | --- |
| `visdrone_v01n_300.log` | 40,761,811 bytes | `0649D25174B8D569861924EDF8CD035CB9F8B3E335C3BD95389C0CB7ACC6BF55` |
| `visdrone_esmoe_sparse_300.log` | 40,756,346 bytes | `8BE0A48484A28D15993847614A991309C588AE77D4F25A5937D3CFA7A1D86CF4` |
| `visdrone_esmoe_dense_300.log` | 40,756,164 bytes | `075B765B06AA917B201F91095B0C2C93E8DFAE1E1D7EBA831A41C6060B760BDD` |

日志保留 `nohup` 提示、ANSI 控制符和 tqdm 进度输出，以维持原始性；日常审阅建议优先查看 CSV
和 W&B Report。

## 已知限制与解决方案

1. **SKU-110K 未训练/验证**：数据虽已下载，但资源不足以在本次提交前完成公平的 300-epoch
   双模型对照。解决方案是使用上方同一组 `imgsz=640`、`batch=8`、seed 42 命令补跑，结果未完成前
   不应对 SKU-110K 下结论。
2. **实际运行使用本地 YAML**：服务器数据已手工解压，故日志中的 `data` 是绝对路径
   `VisDrone.local.yaml`。解决方案是直接使用脚本默认的 `VisDrone.yaml` 自动下载，或确保本地 YAML
   的类别与划分和内置配置一致。
3. **与论文总 batch/训练预算不同**：本次是单卡 A40、batch 8、300 epochs 的资源约束复现，
   不代表完全相同硬件和总 batch 条件下的严格论文重跑。
4. **单随机种子**：当前只运行 seed 42。严谨比较应补充多个种子并报告均值与标准差。
5. **W&B 版本较旧**：运行使用 W&B 0.16.0，日志提示可升级到 0.28.1。现有同步已完成；新环境
   建议更新 W&B，但不要在同一对比中混用会改变记录语义的回调版本。

## Checklist

- [x] VisDrone v0.1-N 完成 300 epochs
- [x] VisDrone EsMoE-N 默认评估请求完成 300 epochs
- [x] VisDrone EsMoE-N dense evaluation 对照完成 300 epochs
- [x] 每 epoch 记录 mAP50、mAP50-95、box_loss、cls_loss、moe_loss
- [x] 提供公开 W&B Report、逐轮 CSV 和完整原始日志
- [x] 使用仓库标准 `reproduce_visdrone.py` / `reproduce_sku110k.py` 接口给出复现命令
- [ ] SKU-110K 双模型训练与验证（资源限制，未执行）
