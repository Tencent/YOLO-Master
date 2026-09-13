# 云端实验原始记录附录

本附录保留实验配置、评估结果与故障记录的原文，用于来源核验。命令示例表示计划配置，实际执行以运行参数及日志为准。原文仅规范化Markdown转义与HTML空格，未作措辞修订。结构化指标见同目录 cloud_history_evidence.json。

## rollout-2026-09-04T21-56-54-01a06cb5-453e-7360-9b58-be283ff5c20e.jsonl：第9行

消息角色：user；记录时间：2026-09-04T13:57:26.478Z

```text
metric       baseline       DTK       delta

AP           20.075     19.826     -0.249

AP50         34.684     34.318     -0.367

APs          11.806     11.774     -0.033

APm          29.384     28.898     -0.485

APl          38.672     37.218     -1.454我尝试了让8-16的扩展到24，结果120e实验是这样的，但是在30e试验下表现是更好的，为什么会这样
```

## rollout-2026-09-04T21-56-54-01a06cb5-453e-7360-9b58-be283ff5c20e.jsonl：第122行

消息角色：user；记录时间：2026-09-04T14:03:10.325Z

```text
后续应该是交叉微差，因为last.pt的APs是实验组领先0.007
```

## rollout-2026-09-04T22-07-38-01a06cbf-18cd-76b3-b2cd-959774747c41.jsonl：第48行

消息角色：user；记录时间：2026-09-04T14:12:12.580Z

```text
# Files pasted by the user:

## "===== Comparing epoch 101 ===== Evaluating baseline: /home/ubuntu/yuanlei/YOLO-…": C:\Users\yqy\.codex/attachments/2fa37909-f7cd-478f-b851-acb0d08b8989/pasted-text.txt

## My request:
刚刚磁盘容量满了剩下的没跑完，从这里可以看出：我的策略其实在最后和原方案基本收敛到一个地方了
```

## rollout-2026-09-05T09-42-15-01a06f3b-0a1b-7391-a7fd-d87746d83a3d.jsonl：第122行

消息角色：user；记录时间：2026-09-05T01:51:42.956Z

```text
不对，我试验的时候  "tal_candidate_expand_linear_decay": false,也被覆盖了
```

## rollout-2026-09-05T09-42-15-01a06f3b-0a1b-7391-a7fd-d87746d83a3d.jsonl：第312行

消息角色：user；记录时间：2026-09-05T01:58:38.052Z

```text
python train_explicit.py --name /infinite/common/yqy/YOLO-Master/A2OR/runs/axis_decay_cloud_120e_b16 --resume  /infinite/common/yqy/YOLO-Master/A2OR/runs/axis_decay_cloud_120e_b16/weights/last_healthy.pt --print-config

\=== Explicit YOLO-Master training configuration ===

model=/infinite/common/yqy/YOLO-Master/ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml (pretrained=False; resume=True)

data_yaml=/infinite/common/yqy/YOLO-Master/A2OR/.runtime_data/visdrone_full_0764528ce5ce.yaml

data_root=/infinite/datasets/yqy (dataset=/infinite/datasets/yqy/VisDrone)

dataset=VisDrone

run_dir=/infinite/common/yqy/YOLO-Master/A2OR/runs/axis_decay_cloud_120e_b16

{

  "data": "/infinite/common/yqy/YOLO-Master/A2OR/.runtime_data/visdrone_full_0764528ce5ce.yaml",

  "epochs": 120,

  "patience": 0,

  "batch": 16,

  "nbs": 64,

  "workers": 0,

  "imgsz": 800,

  "device": "0",

  "seed": 0,

  "deterministic": true,

  "optimizer": "auto",

  "amp": true,

  "verbose": true,

  "single_cls": false,

  "rect": false,

  "cos_lr": false,

  "multi_scale": 0.0,

  "compile": false,

  "cache": false,

  "val": true,

  "split": "val",

  "fraction": 1.0,

  "close_mosaic": 10,

  "tal_topk": 10,

  "lr0": 0.01,

  "lrf": 0.01,

  "momentum": 0.937,

  "weight_decay": 0.0005,

  "warmup_epochs": 3.0,

  "warmup_momentum": 0.8,

  "warmup_bias_lr": 0.1,

  "box": 7.5,

  "cls": 0.5,

  "cls_pw": 0.0,

  "dfl": 1.5,

  "hsv_h": 0.015,

  "hsv_s": 0.7,

  "hsv_v": 0.4,

  "degrees": 0.0,

  "translate": 0.1,

  "scale": 0.5,

  "shear": 0.0,

  "perspective": 0.0,

  "flipud": 0.0,

  "fliplr": 0.5,

  "bgr": 0.0,

  "mosaic": 1.0,

  "mixup": 0.0,

  "cutmix": 0.0,

  "tal_alpha": 0.5,

  "tal_beta": 6.0,

  "tal_dynamic_topk_small": false,

  "tal_dynamic_topk_lambda": 0.0,

  "tal_dynamic_topk_cap": false,

  "tal_dynamic_topk_min": 0,

  "tal_dynamic_topk_max": 0,

  "assignment_stats": true,

  "assignment_small_area": 1024.0,

  "assignment_medium_area": 9216.0,

  "tal_candidate_expand_0_8": 16.0,

  "tal_candidate_expand_8_16": -1.0,

  "tal_candidate_expand_linear_decay": false,

  "tal_candidate_expand_full_epochs": 0,

  "tal_candidate_expand_decay_epochs": 0,

  "save": true,

  "save_period": 1,

  "plots": true,

  "project": "/infinite/common/yqy/YOLO-Master/A2OR/runs",

  "name": "/infinite/common/yqy/YOLO-Master/A2OR/runs/axis_decay_cloud_120e_b16",

  "exist_ok": false,

  "pretrained": false,

  "resume": "/infinite/common/yqy/YOLO-Master/A2OR/runs/axis_decay_cloud_120e_b16/weights/last_healthy.pt"

}参数是错的，你自己看，更新的也不对
```

## rollout-2026-09-08T19-01-20-01a080ad-f72e-7403-a7cf-6c3a30935e03.jsonl：第9行

消息角色：user；记录时间：2026-09-08T11:02:01.733Z

```text
给我个使用compareaps比较这两个预实验结果的命令c1_coverage_b16_s0_30e  full_baseline_b16_auto_30e_s0
```

## rollout-2026-09-08T19-01-20-01a080ad-f72e-7403-a7cf-6c3a30935e03.jsonl：第134行

消息角色：user；记录时间：2026-09-08T15:36:27.654Z

```text
跑完120e实验了，给一个比较指令
```

## rollout-2026-09-09T00-25-55-01a081d7-21c1-7b41-993e-d2e1ad111a97.jsonl：第9行

消息角色：user；记录时间：2026-09-08T16:26:55.906Z

```text
又到了我们最喜欢的分析环节。请你调用之前的baseline，分析C1实验和baseline的结果，shape                                  GT    APs(pt)  AP50s(pt)  ARs(pt)  hit@0.5(%)

8_16px_x_0_8px                          240    0.0761     0.2499    3.7872     31.6667

16_32px_x_0_8px                          54    0.0791     0.3739    5.8918     22.2222

0_8px_x_0_8px                          1025    0.0835     0.2697    2.4668     21.0732

0_8px_x_32_64px                          16    0.4813     3.2256   11.2500     56.2500

0_8px_x_8_16px                         3053    0.5148     1.6294   11.6008     35.0147

16_32px_x_8_16px                       1788    1.6089     2.3558   12.0720     59.2282

8_16px_x_8_16px                        2489    1.6857     2.9830   14.8540     56.0064

0_8px_x_16_32px                        1994    1.8116     3.3745   18.4051     46.2889

32_64px_x_8_16px                        365    2.4561     2.4785   17.1146     69.0411

8_16px_x_32_64px                       1203    3.5403     7.1593   25.1199     76.0599

32_64px_x_16_32px                      1471    5.4992     5.9752   23.5423     74.0313

8_16px_x_16_32px                       5116    6.6467     9.6123   27.8696     64.4449

16_32px_x_16_32px                      5526   11.4854    14.1074   30.2785     71.4622

16_32px_x_32_64px                      2245   16.7543    16.5630   42.2830     78.7973

64_infpx_x_8_16px                         1   20.0000    25.0000   20.0000    100.0000

0_8px_x_64_infpx                          0       nan        nan       nan         nan

8_16px_x_64_infpx                         0       nan        nan       nan         nan

16_32px_x_64_infpx                        0       nan        nan       nan         nan

32_64px_x_0_8px                           0       nan        nan       nan         nan

32_64px_x_32_64px                         0       nan        nan       nan         nan

32_64px_x_64_infpx                        0       nan        nan       nan         nan

64_infpx_x_0_8px                          0       nan        nan       nan         nan

64_infpx_x_16_32px                        0       nan        nan       nan         nan

64_infpx_x_32_64px                        0       nan        nan       nan         nan

64_infpx_x_64_infpx                       0       nan        nan       nan         nan这是C1实验的数据
shape                                  GT    APs(pt)  AP50s(pt)  ARs(pt)  hit@0.5(%)

16_32px_x_0_8px                          54    0.0530     0.2573    4.2441     29.6296

8_16px_x_0_8px                          240    0.0632     0.2052    3.4487     29.1667

0_8px_x_0_8px                          1025    0.0795     0.2473    2.7084     21.0732

0_8px_x_32_64px                          16    0.4436     2.7301   13.1250     62.5000

0_8px_x_8_16px                         3053    0.4538     1.4193   11.0300     34.9492

16_32px_x_8_16px                       1788    1.6398     2.3811   11.2731     59.1723

8_16px_x_8_16px                        2489    1.6409     2.7879   14.4048     55.5645

32_64px_x_8_16px                        365    1.7888     1.9707   16.2539     65.7534

8_16px_x_32_64px                       1203    2.7915     6.1472   13.4339     76.4755

0_8px_x_16_32px                        1994    3.6569     3.6645   19.2406     46.5898

32_64px_x_16_32px                      1471    5.7926     6.1485   23.5576     73.6914

8_16px_x_16_32px                       5116    6.4614     9.6019   29.4400     64.9726

16_32px_x_16_32px                      5526   11.9703    14.2709   30.3859     71.1907

16_32px_x_32_64px                      2245   16.5092    16.1615   41.2146     79.2873

64_infpx_x_8_16px                         1   70.0000    20.0000   70.0000    100.0000

0_8px_x_64_infpx                          0       nan        nan       nan         nan

8_16px_x_64_infpx                         0       nan        nan       nan         nan

16_32px_x_64_infpx                        0       nan        nan       nan         nan

32_64px_x_0_8px                           0       nan        nan       nan         nan

32_64px_x_32_64px                         0       nan        nan       nan         nan

32_64px_x_64_infpx                        0       nan        nan       nan         nan

64_infpx_x_0_8px                          0       nan        nan       nan         nan

64_infpx_x_16_32px                        0       nan        nan       nan         nan

64_infpx_x_32_64px                        0       nan        nan       nan         nan

64_infpx_x_64_infpx                       0       nan        nan       nan         nan

这是baseline的数据
```

## rollout-2026-09-09T00-25-55-01a081d7-21c1-7b41-993e-d2e1ad111a97.jsonl：第105行

消息角色：user；记录时间：2026-09-08T16:34:05.140Z

```text
为什么c1在小样本进步很多，总体APs还是不如baseline
```

## rollout-2026-09-09T11-59-34-01a08452-2fef-7f01-a37e-1043eefa445d.jsonl：第224行

消息角色：user；记录时间：2026-09-09T04:13:13.805Z

```text
\--- runs/c2_tiered_vd100pct_s0_120e_b4_w0/args.yaml     2026-09-09 01:05:25.638594537 +0800

+++ runs/baseline_b16_adamw_120e_s0_w6/args.yaml        2026-09-04 01:47:48.925778234 +0800

@@ -1,7 +1,7 @@

 task: detect

 mode: train

-model: /infinite/common/yqy/YOLO-Master/ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml

-data: /infinite/common/yqy/YOLO-Master/A2OR/.runtime_data/visdrone_full_0764528ce5ce.yaml

+model: /home/ubuntu/yuanlei/YOLO-Master/A2OR/runs/baseline_b16_adamw_120e_s0_w6-2/weights/last_healthy.pt

+data: /home/ubuntu/yuanlei/YOLO-Master/A2OR/visdrone_full.yaml

 epochs: 120

 time: null

 patience: 0

@@ -11,13 +11,13 @@

 save_period: 1

 cache: false

 device: '0'

-workers: 8

-project: /infinite/common/yqy/YOLO-Master/A2OR/runs

-name: c2_tiered_vd100pct_s0_120e_b4_w0

+workers: 6

+project: /home/ubuntu/yuanlei/YOLO-Master/A2OR/runs

+name: baseline_b16_adamw_120e_s0_w6-2

 exist_ok: false

 pretrained: false

 cls_remap: true

-optimizer: auto

+optimizer: AdamW

 verbose: true

 seed: 0

 deterministic: true

@@ -25,7 +25,7 @@

 rect: false

 cos_lr: false

 close_mosaic: 10

-resume: false

+resume: /home/ubuntu/yuanlei/YOLO-Master/A2OR/runs/baseline_b16_adamw_120e_s0_w6-2/weights/last_healthy.pt

 amp: true

 fraction: 1.0

 profile: false

@@ -82,7 +82,7 @@

 weight_decay: 0.0005

 warmup_epochs: 3.0

 warmup_momentum: 0.8

-warmup_bias_lr: 0.0

+warmup_bias_lr: 0.1

 distill_model: null

 dis: 6.0

 foundation_enabled: false

@@ -141,27 +141,14 @@

 cls: 0.5

 cls_pw: 0.0

 dfl: 1.5

-assignment_stats: true

+assignment_stats: false

 assignment_small_area: 1024.0

 assignment_medium_area: 9216.0

-tal_candidate_expand_0_8: 16.0

-tal_candidate_expand_8_16: -1.0

-tal_candidate_expand_linear_decay: false

-tal_candidate_expand_full_epochs: 0

-tal_candidate_expand_decay_epochs: 0

-tal_candidate_expand_coverage_triggered: false

-tal_candidate_expand_coverage_tiered: true

-tal_candidate_expand_coverage_min: 3

-tal_candidate_expand_coverage_target: 20.0

-tal_candidate_expand_coverage_long_side: 32.0

 tal_topk: 10

 tal_alpha: 0.5

 tal_beta: 6.0

 tal_dynamic_topk_small: false

-tal_dynamic_topk_lambda: 0.0

-tal_dynamic_topk_cap: false

-tal_dynamic_topk_min: 0

-tal_dynamic_topk_max: 0

+tal_dynamic_topk_lambda: 0.8

 pose: 12.0

 kobj: 1.0

 rle: 1.0

@@ -350,18 +337,11 @@

 moe_map_saturation_threshold: 0.001

 moe_map_saturation_decay_factor: 0.8

 moe_map_saturation_min_scale: 0.1

-save_dir: /infinite/common/yqy/YOLO-Master/A2OR/runs/c2_tiered_vd100pct_s0_120e_b4_w0

-effective_optimizer: MuSGD

+save_dir: /home/ubuntu/yuanlei/YOLO-Master/A2OR/runs/baseline_b16_adamw_120e_s0_w6-2

+effective_optimizer: AdamW

 effective_optimizer_lrs:

\-- 0.03

 \- 0.01

\-- 0.03

 \- 0.01

\-- 0.03

 \- 0.01

\-- 0.03

\-- 0.01

\-- 0.015

 \- 0.005

\-- 0.06

 \- 0.02帮我看看
```

## rollout-2026-09-09T11-59-34-01a08452-2fef-7f01-a37e-1043eefa445d.jsonl：第364行

消息角色：user；记录时间：2026-09-09T08:50:23.908Z

```text
现在两个都跑完了，给我个比较指令
```

## rollout-2026-09-09T11-59-34-01a08452-2fef-7f01-a37e-1043eefa445d.jsonl：第443行

消息角色：user；记录时间：2026-09-09T09:02:23.608Z

```text
COCO maxDets=100 (percentage points)

checkpoint       metric       baseline       exp       delta

best.pt          AP             20.111     20.173     +0.061

best.pt          AP50           34.720     34.562     -0.158

best.pt          APs            11.802     11.672     -0.130

best.pt          APm            29.507     30.256     +0.749

best.pt          APl            38.063     36.368     -1.695挂了呀……
```

## rollout-2026-09-09T11-59-34-01a08452-2fef-7f01-a37e-1043eefa445d.jsonl：第457行

消息角色：user；记录时间：2026-09-09T09:05:35.607Z

```text
checkpoint       metric       baseline       exp       delta

last.pt          AP             19.982     20.102     +0.121

last.pt          AP50           34.445     34.601     +0.157

last.pt          APs            11.579     11.694     +0.116

last.pt          APm            29.410     30.171     +0.762

last.pt          APl            38.263     35.131     -3.132领先太少也没啥意义啊
```

## rollout-2026-09-09T11-59-34-01a08452-2fef-7f01-a37e-1043eefa445d.jsonl：第469行

消息角色：user；记录时间：2026-09-09T09:07:41.755Z

```text
有没有啥靠谱一点的方案，感觉这个方向也不太成立，因为已经收窄到了精细化的跳调参，但是其实更宽松的C1也基本上趋近于baseline。我们需要一个更大的变动
```

## rollout-2026-09-09T11-59-34-01a08452-2fef-7f01-a37e-1043eefa445d.jsonl：第718行

消息角色：user；记录时间：2026-09-09T14:11:20.012Z

```text
我现在觉得D1有些太复杂了，和这个问题本身需要的有所偏差，有没有简单点的方法，专注于框选
```

## rollout-2026-09-09T11-59-34-01a08452-2fef-7f01-a37e-1043eefa445d.jsonl：第792行

消息角色：user；记录时间：2026-09-10T01:22:44.357Z

```text
你确定给边长小的分配更少的k是对的吗，我之前的DTK尝试就没成功，你这个和那个有什么区别
```

## 固定扩张云端逐epoch APs日志摘取

附件：2fa37909-f7cd-478f-b851-acb0d08b8989；COCO maxDets=100，AP点。

| epoch | baseline APs | axis APs | 日志Δ |
|---:|---:|---:|---:|
| 101 | 11.576 | 11.687 | +0.111 |
| 102 | 11.629 | 11.741 | +0.112 |
| 103 | 11.697 | 11.748 | +0.050 |
| 104 | 11.752 | 11.719 | -0.033 |
| 105 | 11.796 | 11.784 | -0.012 |
| 106 | 11.818 | 11.730 | -0.088 |
| 107 | 11.795 | 11.757 | -0.039 |
| 108 | 11.780 | 11.759 | -0.021 |
| 109 | 11.806 | 11.774 | -0.033 |
| 110 | 11.775 | 11.774 | -0.001 |
