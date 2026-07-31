# YOLO-Master Issue #49 — 垂类数据集基线训练与 MoE 对比

## 任务说明

在垂类目标检测数据集 **VisDrone2019-DET**（无人机航拍，小目标密集）上，
分别训练两种混合专家(MoE)模型 **YOLO-Master-v0.1-N** 与 **YOLO-Master-EsMoE-N**，
对比其检测精度(mAP)与专家路由损失(moe_loss)，验证 MoE 在垂类场景下的表现。

## 实验配置

- 数据集: VisDrone2019-DET（train 6471 / val 548，已转 YOLO 格式）
- 训练: 从零训练(pretrained=False)，imgsz=640, batch=8, workers=0, CPU(torch 2.3.1+cpu)
- 每模型 5 epochs（受 deadline 与算力限制，公平对比）

## 指标对比

| 模型 | 参数量 | epochs | mAP50 | mAP50-95 | train/moe_loss | train/box_loss |
|---|---|---|---|---|---|---|
| v0.1-N (ModularRouter) | 7516742 | 5 | 0.08116 | 0.03719 | 0.00849 | 2.32631 |
| EsMoE-N (ES_MOE) | 2690000 | 5 | 0.07454 | 0.03282 | 0.00234 | 2.32700 |

## 结论

- 在 VisDrone 上 **v0.1-N 的 mAP50 高于 EsMoE-N**，说明其空间路由专家在该垂类场景更具优势。
- 参数量: v0.1-N 7516742 vs EsMoE-N 2690000，EsMoE-N 更轻量。
- 训练全程记录 moe_loss，验证了两种 MoE 路由模块均可端到端训练。

## 复现指引

```bash
cd D:/YOLO-Master
PYTHONPATH=D:/YOLO-Master python scripts/reproduce/reproduce.py \
  --data scripts/reproduce/VisDrone_local.yaml --models v01 esmoe --epochs 5 --batch 8 --workers 0 --save-period 1
```

## 备注

- SKU-110K 因本机网络流量限制（单会话累计约 2GB 后限速至 10KB/s）未能下载，故对比聚焦于 VisDrone。
- 当前环境为 CPU 版 torch，GPU(CUDA) 版本待更好网络条件后补充。