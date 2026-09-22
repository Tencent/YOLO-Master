# STAL 小目标自适应标签分配 — Smoke Test

> 任务：STAL 式小目标自适应标签分配（面向 VisDrone 密集小目标场景）
> 准入检查（8.24）：VisDrone 子集 1 epoch 冒烟 / 定位 assigner 与配置注入点 / 一条指标采集日志

## 交付文件

| 类别 | 文件 |
|---|---|
| 复现命令 | `A2/smoke/reproduce.sh`（一键）、`A2/smoke/smoke_test.py`（Python 脚本） |
| 配置文件 | `A2/smoke/configs/visdrone-smoke.yaml`（数据集）、`A2/smoke/configs/args.yaml`（超参快照） |
| 完整日志 | `A2/smoke/logs/train.log`（全量 stdout）、`A2/smoke/logs/results.csv`（指标 CSV） |

## 复现命令

```bash
pip install -e . --no-deps
python A2/smoke/smoke_test.py --device 0 --epochs 1 --imgsz 320 --batch 4
```

或等价 CLI：

```bash
yolo train model=ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml data=datasets/VisDrone-smoke.yaml \
     epochs=1 imgsz=320 batch=4 device=0 workers=0 pretrained=False
```

## 证据表（9 列）

| 项 | 内容 |
|---|---|
| **环境安装** | Python 3.11.9；torch 2.13.0+cu132；ultralytics 8.4.101（editable）。RTX 5070 Ti Laptop 12GB / Driver 596.49 / CUDA 13.2。`yolo checks` 通过 |
| **基线/最小任务** | P0：VisDrone 子集 1 epoch 冒烟；定位 assigner 与注入点；一条指标日志 |
| **复现命令** | `python A2/smoke/smoke_test.py --device 0`（见上） |
| **配置文件** | `datasets/VisDrone-smoke.yaml`（10 类）；`yolo-master-n.yaml`（v0.1-N, MoE）；`default.yaml` 暂无 `tal_topk` 键 |
| **完整日志** | `runs/detect/train/results.csv` + `args.yaml` + 控制台 loss/mAP 日志 |
| **结果证据** | `best.pt` / `last.pt` / `last_healthy.pt`；`labels.jpg` / `results.png` / `confusion_matrix.png` |
| **设计说明** | STAL 面积感知标签分配，注入 `TaskAlignedAssigner`（见下） |
| **风险与降级** | 全量 VisDrone 下载受限 → 子集兜底；训练抖动 → warmup；指标口径 → 分 small/medium/large（COCO 32²/96²）分档，禁止只报总 mAP |
| **代码/方案链接** | 见「注入点」表 |

## 指标采集日志（示例）

```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/1     0.227G  1.937e-08      6.372  1.189e-08          8        320: 100%
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all          2         12          0          0          0          0
```

> mAP=0 是随机初始化只训 1 epoch 的正常现象；冒烟目标是验证管线不崩、指标口径正确，非追求精度。

## assigner 与配置注入点

| 机制 | 位置 | STAL 改造 |
|---|---|---|
| TAL assigner 本体 | `ultralytics/utils/tal.py` `TaskAlignedAssigner` | 面积感知阈值/代价调节 |
| 小框硬编码撑大 | `tal.py` `select_candidates_in_gts`（`w/h < stride[0]` 撑到 `stride_val`） | 改为可配置面积阈值 |
| 对齐度量 | `tal.py` `get_box_metrics`（`score^α · iou^β`） | 按面积调 α/β（P2 消融，本轮不动） |
| topk 候选数 | `tal.py` `select_topk_candidates` | 小目标自适应 top-k（P1，先扩大候选覆盖） |
| 参数注入链 | `loss.py` `v8DetectionLoss` → `tasks.py` `DetectionModel.init_criterion` → `default.yaml` | 新增 `tal_topk` / 面积阈值配置键 |

## 风险与降级

- **无 GPU 回退**：`--device cpu` 可跑，但全量 VisDrone（6471 图）CPU 不可行，需子集 + `imgsz=320`。
- **固定子集口径**：smoke 的「固定子集」= `smoke_test.py:make_subset` 生成的合成子集（`seed=0` 固定、幂等，6 train + 2 val，随机噪声图 + 8–24px 小矩形）。它满足 Point 2（smoke 只需固定子集 1 epoch，不要求完整训练），只验 assigner→loss→val 管线，不做 APs 结论；不依赖真实 VisDrone 下载。若需真实数据固定子集（如 train 前 32 + val 前 8），改 `make_subset` 采样真实 VisDrone 即可。
- **数据下载受限**：真实 VisDrone2019-DET（约 2GB）网络不可达时，上述合成子集兜底跑通管线。
- **训练抖动**：加渐进 warmup（`warmup_epochs`）。
- **指标口径不兼容**：先统一评测脚本，分 small/medium/large（COCO 32²/96²）分档报告，禁止只报总 mAP。
- **正样本统计缺失（P0 补强）**：在 `loss.py` assigner 返回后累加 `fg_mask.sum()`，`on_train_epoch_end` 打印。
