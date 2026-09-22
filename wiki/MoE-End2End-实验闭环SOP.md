# MoE × End2End 实验闭环 SOP(可训练 → 可评测 → 可部署)

> 定位:MoE × End-to-End 消融实验(免 NMS / 骨干 MoE 等 2×2 类实验)的完整操作手册
> 来源:免 NMS 课题 A1(S1+S2)实战沉淀;每环节给出工具、命令、口径与检查点
> 工具目录:`scripts/a1/`(评测/分析/数据准备脚本)、`ultralytics/cfg/models/26/`(模型 YAML)、`ultralytics/cfg/datasets/`(数据集 YAML)

---

## 环节一 可训练

### 1.1 数据准备

| 项 | 工具/方法 | 检查点 |
|---|---|---|
| 获取 | COCO 官方/镜像源(val2017);train2017 可用 HuggingFace parquet 镜像 | 图片数与官方一致(val 5,000 / train 117,266) |
| 提取 | `scripts/a1/extract_coco2017_parquet.py`(纯 pyarrow,边提取边删) | 类别 0-79、0 corrupt |
| 拆分 | `scripts/a1/prepare_coco_train_split.py` / `prepare_coco_val_split.py`(固定 seed 抽样 + 清单存档) | manifest JSON 存档,可复现 |
| 分发 | 数据盘真实文件(软链解引用)+ `rsync -a` | 训练机数量校验 |

### 1.2 任务书机制(多机调度)

- 模板:`docs/discuss/多任务调度设计.md`(含评测约定内置 + 中断/续训预案)
- 必含字段:代码 SHA 锁定、数据、模型 YAML、训练命令、产物约定、**评测约定(指标清单)**、执行记录
- 训练前 5 项检查:数据集 / 代码 SHA / 模型参数 / 产物目录 / wandb
- 消融纪律:格子间唯一差异 = `model` 参数,其余逐字一致

### 1.3 训练

```bash
export YOLO_AUTOINSTALL=false   # 防图片校验误判 corrupt
nohup yolo train model=<模型yaml> data=<数据集yaml> \
  epochs=100 imgsz=640 batch=8 device=0 seed=0 save_period=10 \
  project=<runs目录> name=<run名> > <logs>/train.log 2>&1 &
```

- 口径:训练中 val 用固定拆分(跨实验可比);训后 best.pt 补跑完整评测集(官方口径)
- 显存注意:MoE 模型(A2C2fMoE)batch=8 时约 19-20GB,batch 上调会 OOM
- **中断/续训预案(实战)**:外部中断 → 依赖 `last.pt`/`last_healthy.pt` 自动断点;resume 前若 `ckpt["scaler"]` 为空 dict(MuSGD 未启用 AMP),备份后置 `None` 再续训;每次续训用独立日志

### 1.4 产物归档

- `scripts/a1/upload_run_to_share.sh <run目录> <日志>` → 共享盘归档目录
- 任务书回填:机器/起止时间/每 epoch 耗时/指标清单/异常记录

---

## 环节二 可评测

### 2.1 固定评测集(多格共享同一批图)

- 固定清单:`scripts/a1/eval_imgs_val2017_seed0_n100.txt`(val2017 seed=0 抽 100)
- 全量逐批:`eval_batch_log.py` 内部 seed=0 打乱(多格同 batch 序列,`shuffled_order.json` 落盘)

### 2.2 标准评测(精度 + 延迟 + 逐图数量)

```bash
python scripts/a1/eval_pipeline.py \
  --model <A.pt> <B.pt> <C.pt> <D.pt> \
  --data ultralytics/cfg/datasets/coco-train-2017.yaml \
  --img-list scripts/a1/eval_imgs_val2017_seed0_n100.txt \
  --device 0 --batch 8 --rounds 5
```

- 输出:val 精度(mAP/P/R)+ 单图三段 latency + 批处理吞吐 + 逐图(检测数,latency)明细
- 口径:显式 fuse 到标准部署形态;torchvision NMS 主口径(`--nms-impl`);GPU 逐段 sync

### 2.3 逐批原始日志(全量,分析用)

```bash
python scripts/a1/eval_batch_log.py \
  --model <A.pt> <B.pt> <C.pt> <D.pt> \
  --data ultralytics/cfg/datasets/coco-train-2017.yaml \
  --device 0 --batch 8 --seed 0 [--conf 0.001|0.25] [--no-val] \
  --out <输出目录>
```

- 逐批记录:批级 gt_n / cand_n(NMS 前)/ det_n(NMS 后)+ 逐图明细 + TP/FP/FN/逐批 P/R(IoU≥0.5 贪婪匹配)+ 四段耗时
- 双口径:conf=0.001(完整候选,val 同口径)/ conf=0.25(部署口径)

### 2.4 MoE 专属分析

```bash
python scripts/a1/analyze_moe_latency.py --model-b <MoE格.pt> [--model-a <dense对照.pt>]  # 延迟归因(profiler + 消融)
python scripts/a1/analyze_routing.py --model <MoE格.pt>                                  # 专家使用分布
python scripts/a1/analyze_expert_diversity.py --model <MoE格.pt>                          # 专家行为差异(绕过路由)
python scripts/a1/analyze_routing_evolution.py --weights-dir <weights目录>                # 集中度-epoch 演化
```

---

## 环节三 可部署

### 3.1 导出与 parity 验证

```bash
python scripts/a1/verify_export_parity_20k.py --model <A.pt> <B.pt> <C.pt> <D.pt>
```

- 标准:FP32 ONNX(imgsz 640,opset 13),PT vs ORT 最大绝对误差 <1e-2;行级匹配 ≥99%(topk 并列边界效应为已知容忍项)
- 实测(免 NMS 课题四格):A/B/D <1e-3 完美,C 99.3% 行匹配 ✅

### 3.2 部署形态约定(评测已锁定的口径)

| 约定 | 内容 | 依据 |
|---|---|---|
| 加载形态 | 标准 `YOLO(pt).predict()` 路径(默认 fuse:Conv-BN 融合 + 剥 one2many 头) | 不 fuse 白算双头,检测头计算近翻倍 |
| NMS 实现 | torchvision CUDA kernel 为主口径 | TorchNMS 纯 torch fallback 实测 ~0.12ms/保留框,裸 torch 部署踩坑 |
| 部署阈值 | conf=0.25 口径(最终输出框数/GT≈0.5,P≈0.68) | conf=0.001 是完整候选集口径(检测/GT≈20),非展示框 |
| 选型数据 | 延迟账本(单图/批处理/波动 CV) | 实测:免 NMS 批处理 -23.9%、CV 0.139→0.066;MoE 延迟 +193% |

### 3.3 部署选型决策表(20k 数据下的实测结论示例)

| 场景 | 推荐格 | 理由 |
|---|---|---|
| 实时单图(GPU) | 免 NMS(one2one) | mAP 代价仅 -3%,延迟 -5.2%,波动减半 |
| 批服务(GPU) | 免 NMS | 批处理收益 -23.9%,远超精度代价 |
| 需要最高精度、可接受延迟 | NMS(o2m) | mAP 最高,NMS 成本可控(torchvision 路径) |
| 当前不建议 | MoE 格 | 精度打平,延迟 +193%(CPU 驱动瓶颈,待批量专家实现升级) |

---

## 附:闭环全景

```
数据(1.1) → 训练(1.2/1.3) → 归档(1.4)
                  ↓
   评测:val 双口径 + eval_pipeline + 逐批日志 + MoE 分析(2.x)
                  ↓
   部署:ONNX parity + 部署形态口径 + 选型决策(3.x)
```
