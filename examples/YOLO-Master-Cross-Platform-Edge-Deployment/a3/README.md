# A3：动态路由、Softmax、top-k 与异构专家在 ONNX / TensorRT / INT8 下的真实兼容性

**状态（2026-09-12）：P0 全部完成；P1 的 ES-MoE 族完成、MoT 族因上游没有发布训练权重而无法开展；P2 的混合精度敏感层回退与 QAT 已完成（QAT 为负结果），路由一致率工具与五族对比未完成。详见第 11 节。**

本文件与 `a3/`、`scripts/a3/` 是 `skywalker-lt/yolo-master-edge` 分支 `dev/a3-smoke` 的原样副本，放在跨平台边缘部署示例 `examples/YOLO-Master-Cross-Platform-Edge-Deployment/` 内（本文件位于 `a3/README.md`；下文所有路径均相对于该示例根目录）。

<img width="3386" height="1241" alt="50001 copy" src="https://github.com/user-attachments/assets/e2c83800-36d6-4ea2-be5c-af2abbc43524" />

<br>
<br>

Branch `dev/a3-smoke` rebased基于边缘仓库冻结基线`446f53ac`创建。本文档只覆盖**8.24 smoke test / 准入检查**的范围：
- 最小`yolo.export` smoke test
- 环境矩阵+后端版本
- 失败日志与成功日志，
- 准入条件表格

所有结果均按要求在统一公共基线**Tencent/YOLO-Master `acce839c7e895d6b179de7f7093fa879e237cc7b`**（2026-08-21 23:59，ultralytics 8.4.101）上测得；而此前在legacy版本`3ea98305`（8.3.240）上的一轮结果已作废，仅保留副本在 `a3/results-3ea98305/` 与 `a3/logs-3ea98305/` 供对照使用（`scripts/a3/compare_baselines.py`可print两版对比与结论是否不同）。

---

## 1. 课题定位与已有工作（已更新为基于`acce839c`）

**Upstream公共仓库（Tencent/YOLO-Master，即题目所指的"仓库"）已有的能力，本课题不作为任何新增的功能：** 基于Ultralytics的完整的“训练->部署的全流程（train、eval、infer和基础ONNX于TRT部署脚本）、`yolo.export` Skill runner、MoE / MoA / MoT 模块及其routing机制、专家利用率分析与pruning、LoRA，以及除`yolo-master-edge`之外的其它YOLO系列第三方部署案例（TensorRT-YOLO、ncnn-YOLO-Master-android）。在`acce839c`上，exporter还自带了`export_preflight`：按compatibility矩阵（`ultralytics/cfg/export-capability-matrix.yaml`）对每个路由模块给出dynamic / dense_fallback / merged / routing_preserved / refuse策略，并在strict模式下直接拒绝export。另有两项工作来自本仓库`yolo-master-edge`（目前也作为公共仓库）里的前期工作（跨平台多后端Runtime、Jetson TRT部署以及分支 `dev/project03` 在**剪枝后**v0.1-N上进行的INT8 quant研究、基于原来[犀牛鸟官网上](https://opensource.tencent.com/summer-of-code/project/108/practice)的课题三完成），本将直接借鉴/复用以上工作，并同样不作为新成果呈现。

**本课题补齐的缺口：**

| 目前缺口 | 本branch做法 |
|---|---|
| Upstream preflight的判定细粒度是"module-wise"：兼容性矩阵直接把所有的MoE族的ONNX导出一律标为`dense_fallback` (:"data-dependent routing is replaced by dense fallback"），而v0.1系的模型导出分支其实是精确 top-k（gather） | `scripts/a3/smoke_export.py` 按**模块的实例**判定 preserve / dense / reject，并用三层导出后验证（算子直方图、expert conv计数、ORT强对齐）；`--dynamic`在导出前直接明确拒绝（失败日志） |
| Upstream v26.08 Release权重在`acce839c`下的兼容性：旧pickle缺少能在新的eval forward能读取的attribute；报错：`AttributeError 'EfficientSpatialRouter' object has no attribute 'capacity factor' | Explicit quant修复（`scripts/a3/diagnose_moe.py`），逐项记录补了什么 |
| 默认模型的INT8 calibration覆盖 | 按导出计算图（dense等价路径）做校准，并检查未校准的quantizer |
| INT8结论此前只在branch `dev/project03`的剪枝模型上成立 | 在**Upstrem里的未剪枝**v0.1-N 与 EsMoE-N（COCO）上重跑Implicit INT8 与Explicit Q/DQ |
| 代码引用沿用旧行号 | 所有策略依据在运行时按锁定commit重新grep，连同命中行一起写入 `a3/results/smoke_*.json` |

锁定commit：YOLO-Master `acce839c7e895d6b179de7f7093fa879e237cc7b`（fork`skywalker-lt/YOLO-Master`的`main`已强制rebase到该 commit）；边缘repo基线`446f53ac4b91abfa193a56ab11f91a5d158611c8`。

---

## 2. 8.24 准入检查完成情况

| 序号 | 姓名 | 第一志愿 | 环境安装 | 基线/最小任务 | 复现命令 | 配置文件 | 完整日志 | 结果证据 | 设计说明 | 风险与降级 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 李汭恒 | A3 | `scripts/a3/setup_l4.sh` 生成 `a3/env/matrix.md`、`a3/env/pip-freeze.txt`，日志 `a3/logs/setup_l4.log` | P0 已完成：`yolo.export` ONNX smoke test（v0.1-N、EsMoE-N）及ORT对齐；ORT / TRT fp32 / fp16 与PyTorch的精度差异；附INT8校准策略（Implicit + Explicit QDQ） | 第四部份（`scripts/a3/run_all_l4.sh`） | `a3/config/coco-a3.yaml`、`a3/config/sensitive_sets.md` | 成功：`a3/logs/v01n_smoke.log`、`a3/logs/esmoen_smoke.log`；失败（拒绝）：`a3/logs/v01n_smoke_dynamic_reject.log`；失败（TensorRT构建）：`a3/results/failed_v01n_int8_headrouters.txt`；INT8：`a3/logs/*_trt_ladder.log`、`*_int8_pins_*.log`、`*_trt_stempair.log`、`*_int8_qdq.log`、`*_backend_val.log` | `a3/results/*.json`、`a3/results/ladder_*.csv`，第6章 | 第7章 | 第8章 |

---

## 3. 环境矩阵

L4 GPU容器：`torch 2.4.1+cu124` Docker Image；`scripts/a3/setup_l4.sh`按**固定版本**安装全部后端（TensorRT用`tensorrt-cu12==10.13.3.9`，与dev/project03在A100上实验所用版本一致；ONNX Runtime 1.20.2；modelopt 0.27.1； `ultralytics` 以editable安装锁定commit的YOLO-Master，依赖按其声明解析，但通过constraints文件把torch锁定在预装的2.4.1+cu124）。每装完新的依赖就重新检查torch版本，防止pip在解析modelopt、onnxruntime-gpu等依赖时把torch换掉。整条链路跑完后再次采集`pip freeze`并做 diff，以证明导出过程没有触发任何ultralytics的自动安装（`YOLO_AUTOINSTALL=False`）。

| item | value |
|---|---|
| captured_utc | 2026-08-23T07:14:33+00:00 |
| host | fea95d9954c8 |
| os | Ubuntu 22.04.5 LTS |
| kernel | 6.8.0-50-generic |
| cpu | AMD EPYC 7702 64-Core Processor |
| ram_gb | 503 |
| gpu | NVIDIA L4, 23034 MiB, 580.126.20 |
| cuda_driver_max | CUDA Version: 13.0 |
| cuda_toolkit_nvcc | Build cuda_12.4.r12.4/compiler.34097967_0 |
| python | 3.11.10 |
| torch | 2.4.1+cu124 |
| torch_cuda | 12.4 |
| cudnn | 90100 |
| torchvision | 0.19.1+cu124 |
| numpy | 2.4.6 |
| onnx | 1.17.0 |
| onnxslim | 0.1.96 |
| onnxruntime | 1.20.2 |
| onnxruntime_providers | TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider |
| tensorrt | 10.13.3.9 |
| modelopt | 0.27.1 |
| pycocotools | installed (no __version__) |
| opencv | 5.0.0 |
| ultralytics | 8.4.101 |
| ultralytics_file | /data/YOLO-Master/ultralytics/__init__.py |
| yolo_master_repo | `acce839c7e89` (2026-08-21, main) |
| edge_repo | dev/a3-smoke（见 `a3/env/matrix.json`） |

---

## 4. 复现命令

```bash
# 建议在Hopper / Ada-Lovelace架构GPU的Linux机器执行
cd /data/yolo-master-edge && git switch dev/a3-smoke
bash scripts/a3/setup_l4.sh # 1. 环境（可重复执行）
. /root/a3venv/bin/activate
nohup bash scripts/a3/run_all_l4.sh > a3/logs/run_all.log 2>&1 & # 2. 完整链路
# STAGES=env,smoke,ladder,qdq,backend（可用 STAGES=smoke 等只跑子集）
bash scripts/a3/run_v01n_pins_l4.sh # 3. v0.1-N Implicit INT8 钉层集合诊断
python scripts/a3/compare_baselines.py --old a3/results-3ea98305 --new a3/results   # 4. 两版基线对比
```

分阶段执行的独立命令：

```bash
python scripts/a3/env_matrix.py --out a3/env
python scripts/a3/smoke_export.py --model /data/YOLO-Master/YOLO-Master-v0.1-N.pt \
    --out runs/a3/v01n --data a3/config/coco-a3.yaml \
    --json a3/results/smoke_v01n.json --log a3/logs/v01n_smoke.log
python scripts/a3/smoke_export.py --model /data/YOLO-Master/YOLO-Master-EsMoE-N.pt \
    --out runs/a3/esmoen --data a3/config/coco-a3.yaml \
    --json a3/results/smoke_esmoen.json --log a3/logs/esmoen_smoke.log
python scripts/a3/smoke_export.py --model /data/YOLO-Master/YOLO-Master-v0.1-N.pt \
    --out runs/a3/v01n-dynamic --dynamic ... # Exit Code 2：即失败日志
python scripts/a3/quantize_trt.py --model <pt> --data a3/config/coco-a3.yaml --out runs/a3/<m>/trt \
    --calib-images /data/datasets/coco/images/train2017 --calib-n 1024 \
    --modes fp32,fp16,int8 --max-rounds 0 --tf32-baseline off # 构建失败的会被记录而不是直接停止
python scripts/a3/quantize_trt.py --model runs/a3/<m>/trt/<stem>.onnx ... --modes int8 --pins head|routers|none
python scripts/a3/quantize_trt.py --model runs/a3/<m>/trt/<stem>.onnx ... --modes int8 --pins head --pin-modules 0,1
python scripts/a3/qat_moe.py --model <pt> --data a3/config/coco-a3.yaml --out runs/a3/<m>/qdq \
    --calib-n 1024 --skip-train # 只校准的Explicit Q/DQ
python scripts/a3/backend_val.py --model <pt> --onnx runs/a3/<m>/<stem>.onnx \
    --data a3/config/coco-a3.yaml --json a3/results/backend_<m>.json
```

---

## 5. 配置文件

- `a3/config/coco-a3.yaml`：COCO路径，`train: images/train2017`（仅用于校准），`val: images/val2017`（用于评测）。
- `a3/config/sensitive_sets.md`：校准策略（不少于1024张训练集图片、Implicit quant用entropy、Explicit per-channel Q/DQ）以及保持FP16的敏感层set（stem前两层、路由器、DFL）。
- 全程统一评测：ultralytics `.val`（8.4.101），imgsz 640，batch 1，val2017全5000张；PyTorch、ONNX Runtime 与TensorRT engine走同一条路径，因此所有精度严格可对比。

---

## 6. 结果证据

所有精度均为ultralytics 8.4.101 `.val`的mAP50-95（COCO val2017，5000 张，imgsz 640，batch 1）。延迟为 TRT engine 在L4 上model-only 200次执行中取中位数。原始数据在 `a3/results/`，逐层精度audit在对应的日志里。

**关于PyTorch原始树脂的说明：** 同权重同机器、CPU fp32，在`8.3.240`的validator下 v0.1-N测得0.4292，而在`8.4.101`下则测得0.4250（EsMoE-N为0.4270和0.4228）；两个不同版本的模型原始输出逐元素一致（bbox坐标最大差 0.001 px，cls 5e-5），定位后差异全部来自8.4.101 的val/postprocess路径（`nms.py`、`ops.py`、`augment.py`、`val.py`均有较大幅度的改动）。因此**绝对mAP不跨ultralytics版本作比较，
本节所有差值都在同一公共基线`acce839c`里的`Ultralytics 8.4.101`内计算。**

### 6.1 P0：`yolo.export` smoke test与各后端相比PyTorch的精度差异

| 模型 | 后端 | mAP50-95 | ∆ vs PyTorch | 延迟（ms）| 大小（MB） | 原始来源 |
|---|---|---|---|---|---|---|
| v0.1-N（OptimizedMOEImproved x3，E=4/8/16，k=2） | PyTorch（已修复） | **0.4250** | 0 | - | 15.1（.pt） | `backend_v01n.json` |
| | ONNX Runtime CUDA EP（默认 TF32） | 0.4245 | -0.0005 | - | 30.5（.onnx） | `backend_v01n.json` |
| | TensorRT fp32（关闭 TF32） | 0.4246 | -0.0004 | 3.990 | 41.2 | `ladder_v01n.csv` |
| | TensorRT fp16 | 0.4244 | -0.0006 | 1.927（-51.7%） | 21.6 | `ladder_v01n.csv` |
| EsMoE-N（ES_MOE x4，E=3，k=3） | PyTorch（dense 与 sparse 等价） | **0.4228** | 0 | - | 5.7（.pt） | `backend_esmoen.json` |
| | ONNX Runtime CUDA EP（默认 TF32） | 0.4225 | -0.0003 | - | 11.1（.onnx） | `backend_esmoen.json` |
| | TensorRT fp32（关闭 TF32） | 0.4225 | -0.0003 | 2.822 | 20.7 | `ladder_esmoen.csv` |
| | TensorRT fp16 | 0.4225 | -0.0003 | 1.720（-39.1%） | 12.4 | `ladder_esmoen.csv` |

<br>

<img width="1389" height="698" alt="matplot_figure_0-5" src="https://github.com/user-attachments/assets/ca99dfab-6dde-4e27-9e91-3c5e03c9282b" />


---

Smoke Test（`smoke_*.json`）：

| 模型 | 本branch判定 | upstream preflight判定 | ONNX中的router算子 | expert conv 模块/图 | ORT fp32 对齐（归一化误差 / top-100 anchor一致率） | ORT默认TF32 |
|---|---|---|---|---|---|---|
| v0.1-N | 3 个块 `preserve`（精确 gather，无语义变化） | `dense_fallback` x3 | TopK 3、GatherElements 6、Softmax 12，无 If/Loop/NonZero | 56 / 56 | 6.3e-6 / 100% | 框坐标最大偏移 1.38 px，99.9% |
| EsMoE-N | 4 个块 `dense`，`semantic_change=false`（k==E，无 top-k / 阈值） | `dense_fallback` x4 | TopK 0、Softmax 13，无 If/Loop/NonZero | 24 / 24 | 5.4e-6 / 100% | 最大偏移 2.98 px，100% |
| v0.1-N `--dynamic` | **拒绝导出，Exit Code 2**（`v01n_smoke_dynamic_reject.log`） | - | - | - | - | - |

### 6.2 INT8量化（Implicit PTQ 作对照 + Explicit Q/DQ），均使用 1024 张 train2017 校准图

| 模型 | INT8校准策略 | mAP50-95 | 对 fp32 的差（AP） | 延迟 ms | 体积 MB | engine 逐层精度（Int8 / Half / Float） |
|---|---|---|---|---|---|---|
| v0.1-N | Implicit entropy，head与router为FP16 | **构建不可行**（`failed_v01n_int8 _headrouters.txt`） | - | - | - | TensorRT OBEY：`/model.7` Conv+SiLU 融合无可用实现 |
| v0.1-N | Implicit entropy，只保留head | 0.3706 | -5.40 | 1.969 | 17.4 | 199 / 179 / 56 |
| v0.1-N | Implicit entropy，只保留router | 0.3340 | -9.06 | 1.930 | 16.7 | `v01n_int8_pins_routers.log` |
| v0.1-N | Implicit entropy，全量化| 0.3341 | -9.05 | 1.902 | 17.2 | `v01n_int8_pins_none.log` |
| v0.1-N | Implicit entropy，head + stem前两层 | **build失败**（`failed_v01n _stempair.txt`） | - | - | - | 同上 |
| v0.1-N | **Explicit Q/DQ，仅校准（modelopt）** | **0.4129** | **-1.17** | 2.238 | 21.1 | 196 / 176 / 85 |
| EsMoE-N | Implicit entropy，head与router为FP16 | 0.3554 | -6.71 | 1.937 | 13.8 | 165 / 147 / 60 |
| EsMoE-N | Implicit entropy，head + stem 前两层 | 0.3705 | -5.20 | 1.912 | 13.2 | `esmoen_trt_stempair.log` |
| EsMoE-N | **Explicit Q/DQ，仅校准（modelopt）** | **0.4082** | **-1.43** | 2.195 | 12.6 | 174 / 168 / 112 |

<br>

<img width="1466" height="800" alt="matplot_figure_0-8" src="https://github.com/user-attachments/assets/192f14bf-a575-4be0-a4dd-32bbf58a7168" />

<br>
<br>

结果解释：
- N-scale尺度上，两个模型族的INT8量化都**比fp16慢**（v0.1-N：1.97 vs fp16 1.93 ms，EsMoE-N：1.94 vs fp16 1.72 ms），与dev/project03在剪枝模型上的结论一致：在TensorRT GPU上这种小模型，部署精度应选fp16。
- Explicit Q/DQ 是唯一能把精度损失做到-2AP以内的方法（剪枝版v0.1-N为-0.80，未剪枝版则为-1.17，因为 4/8/16 conv expert全部量化到INT8）；这一结论在`3ea98305`（之前的废弃结果）与`acce839c`两版基线上一致（-1.23 / -1.49 vs -1.17 / -1.43）。
- **Implicit INT8 的"失败算子"证据在新基线上的发现：** Upstream Ultralytics 8.4.101 的exporter生成的v0.1-N图上，TRT在OBEY精度约束下，但凡同时保留head+另外的任意一组（router或stem），`/model.7` 的Conv+SiLU融合就找不到可实现的kernel从而构建失败；只保留head则可以构建，精度损失与旧废弃基线上保留head+router的方法同级（-5.4 对 -5.3）；只保留router和全量化所有层的结果相同（-9.1），因此说明v0.1-N的router pin在Implicit quant里无作用，所有精度损失全部来自head。EsMoE-N的图不受影响（head+router -6.7，额外+stem -5.2）。
- **Routing的校准问题**（剪枝模型k==E时未发现）：未剪枝v0.1-N若在eager的稀疏路径上做校准，caliset里未被路由激活到的专家没有任何激活统计，而导出图会计算全部专家，modelopt在导出时直接assert "Quantizer has not been calibrated"。修复办法是校准时走与export相同的dense路径，并对未校准的quantizer做Explicit audit（本次 v0.1-N的360个和EsMoE-N的282个全部完成校准，0个禁用）。**校准覆盖必须以导出图为准，而并非eager的路由语义为准。**
- ORT的CUDA EP默认对conv启用TF32：框坐标最大偏移 1.4/3.0 px，但mAP只差-0.0005 / -0.0003。因此smoke test的对齐测试限制`use_tf32=0`以做fp32 vs fp32 的比较，TF32的结果则单独记录。

### 6.3 未剪枝 v0.1 N/S/M/L 与 EsMoE-N 的 INT8 缩放研究（L4，acce839c，COCO val2017 5000 张）

同一条链路推到 S/M/L（上游发布权重，未剪枝）。fp32 为关闭 TF32 的基线；QDQ 为 modelopt Explicit Q/DQ（max 校准，仅校准不训练）；Implicit 为 TensorRT entropy 校准加 `--pin-attn-core`（head、router 与 attention core 保持 fp16，qkv/proj 保持 INT8）。原始数据：`a3/results/qdq_v01{s,m,l}.json`、`qdq_v01m_pin{2,4}.json`、`a3/logs/v01{s,m,l}_trt_ladder.log`、`v01{m,l}_impl.log`、`v01m_qdq_sq.log`；图：`a3/plots/a3_int8_unpruned.png`。

| 规模 | fp32 ms / mAP50-95 | fp16 ms / mAP50-95 | Explicit QDQ ms / mAP50-95 / MB | Implicit entropy ms / mAP50-95 / MB |
|---|---|---|---|---|
| N | 3.990 / 0.4246 | 1.927 / 0.4244 | 2.238 / 0.4129 / 21.1 | 构建不可行（6.2 节） |
| S | 9.835 / 0.4833 | 3.320 / 0.4834 | 3.571 / 0.4789 / 48.8 | 未跑 |
| M | 20.509 / 0.5217 | 6.164 / 0.5215 | 5.405 / **0.0977（崩溃）** / 74.6 | 4.770 / 0.4979 / 65.7 |
| L | 28.142 / 0.5334 | 8.102 / 0.5333 | 8.133 / 0.3299 / 83.0 | 6.818 / 0.5053 / 74.9 |

M 尺度 QDQ 崩溃的定位与修复（`scripts/a3/act_range.py`、`act_range2.py`）：按块统计激活 max/p99.9 比值，S 全网均匀（1 至 4），M 出现系统性离群（5 至 17，最坏在 MoE expert 卷积 `experts.0.conv.3` 与宽 C3k2 主干，attention 块并不特殊）。per-tensor max 校准把 scale 撑到离群值上，主体被压扁。把离群最重的两个块 pin 到 fp16 即恢复：pin{4,5} 0.5061 / 6.139 ms，pin{4,5,11,14} 0.5095 / 6.346 ms，但恢复后延迟已不低于 fp16（6.164 ms），因为离群块正是计算最重的块。SmoothQuant（modelopt `INT8_SMOOTHQUANT_CFG`）在这个全卷积模型上匹配到 0 个模块，无效（`v01m_qdq_sq.log`）。

同一模型在 Implicit（entropy）校准下不崩溃：entropy 校准对离群值做 KL 最优裁剪，而 max 校准不裁剪，因此 M/L 的"离群崩溃"有一半是校准方法的产物。L 的 Implicit INT8 比 fp16 快 17%、小 40%，代价 -2.8 AP。

两个否定结果（N 尺度）：(1) Conv+BN 在量化前折叠（`qdq_*_fused.json`）没有减少 fp32 层或 reformat，TensorRT 构建时已经把 BN 折进 INT8 卷积；(2) 让整个 MoE 块保持 fp16、只量化主干与 head（`qdq_*_moefp16.json`）既不更快也不更小：v0.1-N 2.428 ms / 0.4143 / 25.1 MB，EsMoE-N 2.224 ms / 0.4087 / 13.5 MB，reformat 只是被搬到了主干里的 INT8 孤岛边界上。N 尺度卷积受访存与 launch 限制，任何 INT8 区域都只增加精度边界的 reformat；fp16 无边界，所以最快。

Pareto 结论（INT8 要和整条 fp16 模型尺寸前沿比，而不是和自己的 fp16 比）：S-fp16 0.4834 @ 3.32 ms / 65 MB，M-INT8（Implicit entropy）0.4979 @ 4.77 ms / 66 MB，M-fp16 0.5215 @ 6.16 ms / 112 MB，L-fp16 0.5333 @ 8.10 ms / 125 MB 构成前沿；L-INT8 0.5053 @ 6.82 ms 被 M-fp16 支配（更慢且更不准），N/S 的 INT8 与全部 Explicit QDQ 都不在前沿上。整个缩放范围内 INT8 只赢得一个 Pareto 点（M，Implicit entropy 加 attn-core 保护），其余场景以 fp16 部署。校准方法在每个尺度上都是决定性的：max 对离群值脆弱，entropy 裁剪离群值。

---

## 7. 设计说明

### 7.1 动态路由导出策略（Explicit，不允许静默改变语义）

在`acce839c`上，`yolo.export`会先运行`ultralytics/utils/export_preflight.py`：按建荣性矩阵与模块自定的`export_capabilities()`给每个路由模块一个策略。矩阵把所有MoE族的ONNX导出一律标为 `dense_fallback`，默认的runtime声明也一律为`exact_sparse_export=False`。这在EsMoE上没问题，而在处理v0.1系列时则过于保守了，`OptimizedMOEImproved`的导出分支先计算全部专家、再用`torch.gather`取 top-k，语义与eager一致。`smoke_export.py`在调用同一个`YOLO.export`之前，按**模块实例**做判定并把依据（文件、运行时重新grep得到的行号、commit）写进结果JSON：

| MoE族类 | 锁定commit下的导出分支行为 | 本branch策略 | 语义变化 |
|---|---|---|---|
| `OptimizedMOEImproved`（v0.1 系列） | 先计算全部专家，再用 `torch.gather` 取 top-k（`selected = torch.gather(all_outs, ...)`） | `preserve` | 无（精确 top-k）；上游 preflight 标为 `dense_fallback` 属于保守标注 |
| `ES_MOE`（EsMoE 系列） | 强制走 `_dense_forward`：对全部专家做softmax加权求和，跳过`_sparse_forward`的top-k / `dynamic_threshold` | `dense`；仅当`k==E` 且未启用top-k / 阈值时视为无语义变化（发布的 COCO 权重满足：E=3，k=3，`use_top_k=False`，无 `dynamic_threshold`） | 否则则须用`--routing dense` Explicit声明，并回报eager-sparse与eager-dense的AP差距 |
| 其他族 | 没有经过验证的export分支 | `reject` | - |
| 任何族 + `--dynamic` | gather分支把B/H/W以int写死在`view/expand`里 | **导出前拒绝**（Exit code 2）| 动态轴图在其他shape下会静默出错 |

前提条件：torch < 2.9（`ultralytics/utils/export/engine.py` 对 torch >= 2.4 强制 `dynamo=False`；dynamo 导出器会绕过上述guard，静默abort掉专家）。

### 7.2 导出后验证（三层）
1. ONNX算子直方图：统计 TopK / Gather / GatherElements / Softmax 的数量；一旦出现 `If / Loop / NonZero` 即判定为失败，说明guard未生效。
2. 专家卷积计数：图中`/experts`前缀下的 `Conv` 节点数对比模块中`*.experts.*`下的`nn.Conv2d`数。
3. ORT对齐：在真实val图片上，ONNX Runtime（fp32，`use_tf32=0`）对比**保留原生稀疏语义的eager PyTorch**，报告最大绝对误差、normalization误差以及top-100 anchor一致率。

### 7.3 发布权重在Upstream 8.4.101下必要的修复（若不动则崩溃或导致结论全部错误）
- v0.1-N：checkpoint早于`add_residual`属性诞生；8.3.x的兼容shim将其默认为True（cls conf坍缩至约0.04，mAP50-95从0.43直接暴降到0.007），8.4.x直接删掉了这个shim，同时router基类新增了forward里无条件读取的`capacity_factor`，旧的pickle直接报错`AttributeError`。修复层（`scripts/a3/diagnose_moe.py`）对缺失的属性补全了默认值并print：`add_residual=False`、`routing.capacity_factor=None`、progressive-sparsity等相关字段。
- 两个checkpoint的旧pickle还携带了新版里已经改成只读的字段（`aux_loss`），加载时也一并剥离（`_strip_property_shadows`）。
- EsMoE-N：eval与export统一走dense路径（`use_sparse_inference=False`），与sectipn 7.1的判定保持一致。

### 7.4 为何INT8方案用"cali-only explicit Q/DQ"
dev/project03的研究在prunned v0.1-N（COCO，A100，TRT 10.13.3.9）上的结论是Implicit PTQ 无法量化 MoE 块（路由的 gather/expand 会打断 INT8
链，TensorRT 静默回退到 FP16）；Explicit per-channel Q/DQ 可达 -0.80 AP；而 3 个 epoch 的 QAT 反而退化到 -3.2 AP。因此本分支只做
校准（`qat_moe.py --skip-train`），并把Implicit INT8 留在阶梯中作为"失败算子"的对照档（engine inspector 的审计会列出每一层的精度；
新基线上 v0.1-N 的Implicit档位本身就以"构建不可行"的形式成为失败证据）。

---

## 8. 风险与降级处理

| 风险 | 处理 |
|---|---|
| 动态轴导出 | 策略层直接拒绝；只导出静态 1x3x640x640。需要多分辨率时按分辨率分别导出。 |
| 上游 preflight 把精确 top-k 导出标为 `dense_fallback` | 本分支按模块实例判定并用三层验证证明；若上游矩阵在后续版本改为 `refuse`，需用 `--routing` Explicit声明或改矩阵。 |
| TensorRT Implicit INT8 在 OBEY 约束下构建不可行（v0.1-N，8.4.101 图） | 失败被记录为档位证据（`failed_*.txt`），阶梯继续；可行配方为只钉 head；需要真正 INT8 的 MoE 块只走Explicit Q/DQ。 |
| TensorRT Implicit INT8 把 MoE 块静默留在 FP16 | 用 engine inspector 做逐层精度审计并写入日志。 |
| 其他启用了 top-k / 阈值的 ES_MOE 权重 | 默认拒绝；可用 `--routing dense` 声明回退，但必须附上 eager-sparse 与 eager-dense 的 AP 差。 |
| 稀疏路由下的校准覆盖 | 校准走与导出等价的 dense 路径；未校准的量化器会被审计并报告（绝不让导出静默断言）。 |
| lineage 切换导致绝对 mAP 变化 | 已证明模型输出一致、差异在验证路径；只在同一 lineage 内比较差值，`compare_baselines.py` 用于对照。 |
| torch >= 2.9 / dynamo 导出器 | 运行前断言拒绝。 |
| ONNX Runtime 的 CUDA EP "可用"但加载失败 | 把 torch 自带的 cuDNN / cuBLAS 放入 `LD_LIBRARY_PATH`；`backend_val.py` 在 GPU 评测前先证明 EP 真正加载成功。 |
| modelopt / TensorRT / ORT 版本 | 由 `setup_l4.sh` 固定，`pip freeze` 前后 diff；L4 与 A100（project03）的数字不可直接比较，只比较同一台机器阶梯内的差值。 |
| MoT 族（P1 要求）没有训练权重 | 本阶段不做；MoT 在 TorchScript / CoreML trace 下会丢专家（只有 ONNX 的 guard 有效），P1 需先解决权重来源。 |
| `/data` 是网络卷（5000 张评测 + 1024 张校准的 I/O） | 可 rsync 到本地盘后修改 yaml 指向；校准列表由确定性的等距采样生成，可复现。 |

---

## 🎯 9. 后续目标（P1 / P2）

- P1：在同一条链路上加入 MoT（需要训练权重）；记录失败算子、体积与延迟。
- P2：路由一致率工具：把 TopK 索引张量作为额外的图输出导出，逐图对比 FP32 ORT 与 INT8 engine 的专家选择一致率
  （v0.1 系列有真实的 top-k；EsMoE 在 k == E 下没有"选择"可比，改比路由权重的漂移）；五族对比；混合精度的敏感层回退已经有
  `quantize_trt.py --bisect/--ablate` 工具链可用。

---

## 10. 配套运行时（我们的 `yolo-master-edge` main 分支，本分支未改动）

我们为上游模型编写的跨平台推理运行时（C++17；ONNX Runtime / NCNN / MNN / TensorRT / CoreML；Linux、Windows、Jetson、macOS；
CPU / CUDA / Metal）、Windows 图形界面、Jetson Orin 原生 TensorRT 部署脚本（`jetson/`）、CoreML 导出与 macOS 应用（`mac/`），
以及 VisDrone / SKU-110K / AI-TOD-v2 上的基准结果。详见 `TECHNICAL_REPORT.md`、`jetson/README.md`、`mac/README.md` 与
`main` 分支的 README。

---

## 11. 目标分级完成情况（2026-09-12）

| 分级 | 要求 | 状态 | 证据 |
|---|---|---|---|
| P0 保底 | detect 基线导出 ONNX，ONNX Runtime 或 TensorRT FP16 跑通，PyTorch 与部署后端精度差基线 | 完成 | 第 6.1 节：v0.1-N 与 EsMoE-N 的 ORT CUDA / TRT fp32 / TRT fp16，差值 0.0003 至 0.0006 AP；`a3/results/*.json`、`a3/results/ladder_*.csv` |
| P1 预期 | ES-MoE 与 MoT 两族 INT8 PTQ，记录精度、时延、体积、失败算子 | 一半 | ES-MoE：第 6.2 节（Implicit 失败算子记录、Explicit Q/DQ、体积、逐层精度）；MoT：上游模型库只发布了 EsMoE N/S/M 与 v0.1 N/S/M/L 权重，没有任何 MoT/MoA/MoLoRA 权重，随机初始化的图只能验证导出语义，不能给出量化精度差；训练脚本已就绪（`mot-pretrain`），未训练 |
| P2 理想 | 五族对比；FP32/INT8 专家选择一致率；QAT 或混合精度敏感层回退 | 部分 | 混合精度回退：`quantize_trt.py --pins/--bisect/--ablate` 与 6.2 的 pin 集；QAT：剪枝版 v0.1-N 上 3 epoch 冻结 amax 反而退化 -3.2 AP（负结果，见 `dev/project03`）；一致率工具已设计（第 9 节）未实现；五族中只有 v0.1 与 ES-MoE 有发布权重 |

补充的 MEASURED 数据点（NVIDIA L40S，TensorRT 10.16，COCO val2017 5000 张，同一份 ONNX，`yolo-master-edge` main 分支 `API_SERVER_RESULTS.md`）：TensorRT fp16 相对 fp32 的 mAP50-95 变化为 v0.1-N -0.0002、EsMoE-N -0.0001（评测器为 `eval_map.py`，绝对值不与本页的 ultralytics `.val` 比较）；fp16 ONNX 在 ORT CUDA EP 上没有速度收益，因为路由子图必须保持 fp32。
