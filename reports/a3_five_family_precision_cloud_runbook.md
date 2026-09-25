# A3 五族量化、真实 MoT 路由漂移与自动回退：Python 云端运行手册

## 1. 运行环境结论

这套流水线不是纯 CPU 实验。正式 COCO 结果采用“GPU 为主、CPU 辅助”的环境：

| 阶段 | 默认执行设备 | 说明 |
|---|---|---|
| 配置解析、SHA-256 锁定、报告生成 | CPU | 不需要 CUDA |
| FP32 ONNX 导出 | CPU | 默认 `A3_EXPORT_DEVICE=cpu`，减少导出设备差异 |
| ONNX 图转换与 INT8 图改写 | CPU | 图优化和 Q/DQ 插入本身在主机端执行 |
| INT8 校准推理 | CUDA 优先 | `CUDAExecutionProvider` 优先，CPU 只作为算子回退 |
| 真实训练 MoT 路由漂移 | CUDA | 正式实验使用 `A3_DEVICE=cuda:0` |
| FP32、FP16、INT8 mAP 验证 | CUDA | 五个方案使用相同验证契约 |
| FP32/INT8 逐层路由采集 | CUDA 优先 | ONNX Runtime 优先使用 CUDA EP |
| 敏感层逐组消融 | CUDA | 是整套实验最耗时的部分 |
| 时延测试 | CUDA 优先 | 同时记录实际 CUDA/CPU 节点分配，识别静默回退 |

CPU 可以用于小规模代码烟测，但不应把 CPU 时延、CPU 上的 FP16 结果作为最终部署结论。

## 2. 云端环境要求

### 2.1 推荐硬件

- NVIDIA GPU；建议至少 16 GB 显存。实际要求取决于五个 checkpoint。
- 建议至少 32 GB 系统内存。
- 若五族权重已经齐全，只需 COCO 2017 完整验证集与 COCO8 校准集；若需要在云端补训缺失
  权重，则还需要完整 train2017。
- 正式时延实验期间尽量独占 GPU，避免其他任务干扰。

流水线没有写死 GPU 型号。最终锁文件会记录 GPU 名称、CUDA、cuDNN、PyTorch、ONNX Runtime
版本和 Provider 列表。

### 2.2 推荐软件

- Linux x86_64 云实例。
- Python 3.10 或 3.11。
- 与云端驱动兼容的 CUDA 版 PyTorch。
- `onnxruntime-gpu`，且必须能看到 `CUDAExecutionProvider`。
- 项目依赖以及 `requirements-a3-precision.txt` 中的 ONNX、OpenCV、NumPy、SciPy、PyYAML。

不要在同一个环境中混装相互冲突的 CPU/GPU ONNX Runtime 包。正式运行前必须执行下面的 Python 门禁。

## 3. Python 初始化项目和依赖

以下代码可直接放入 Jupyter Notebook、云端 Python 控制台或一个 Python 脚本中。只需要修改
`REPO`。

```python
from pathlib import Path
import subprocess
import sys

REPO = Path("/mnt/workspace/YOLO-Master").resolve()
if not (REPO / "scripts/run_a3_five_family_precision.py").is_file():
    raise FileNotFoundError(f"不是完整的 YOLO-Master 仓库：{REPO}")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "."],
    cwd=REPO,
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements-a3-precision.txt"],
    cwd=REPO,
    check=True,
)
```

如果拿到的是本任务生成的覆盖包，可先用 Python 解压到同一版本的定制
`YOLO_Master` 仓库根目录：

```python
from pathlib import Path
import tarfile

REPO = Path("/mnt/workspace/YOLO-Master").resolve()
BUNDLE = Path("/mnt/workspace/a3_precision_harness_cloud_20260902.tar.gz").resolve()

with tarfile.open(BUNDLE, mode="r:gz") as archive:
    for member in archive.getmembers():
        destination = (REPO / member.name).resolve()
        if destination != REPO and REPO not in destination.parents:
            raise RuntimeError(f"压缩包包含越界路径：{member.name}")
    archive.extractall(REPO)
```

覆盖包不是原版 Ultralytics 的独立补丁，必须叠加到当前定制版 `YOLO_Master` 代码上。

## 4. GPU、CUDA 和依赖门禁

下面的检查必须全部通过，才能生成正式 GPU 报告：

```python
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN:", torch.backends.cudnn.version())
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("ONNX:", onnx.__version__)
print("ONNX Runtime:", ort.__version__)
print("ORT providers:", ort.get_available_providers())

if not torch.cuda.is_available():
    raise RuntimeError("正式实验要求 CUDA 版 PyTorch 和可用的 NVIDIA GPU")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("正式实验要求 onnxruntime-gpu 提供 CUDAExecutionProvider")

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {props.name}, VRAM={props.total_memory / 1024**3:.1f} GiB")
```

若这里失败，不要继续正式运行。需要先修正 PyTorch CUDA、NVIDIA 驱动或
`onnxruntime-gpu` 环境。

## 5. COCO 2017 数据和五族权重

推荐使用 COCO 2017 做新的五族统一实验。五个 checkpoint 必须全部是 COCO 80 类真实训练权重，
不能混入现有的 10 类 VisDrone 权重。按实验要求，正式 mAP 使用完整 5000 张 COCO val；INT8 校准
使用独立的 COCO8 train（4 张）。仅运行量化验证时不需要 train2017；本次缺失权重训练需要完整
train2017，训练完成后可参照 `reports/a3_missing_coco_weights_training_runbook.md` 验收五族权重。

COCO 根目录至少应包含：

```text
coco/
├── images/
│   └── val2017/
└── labels/
    └── val2017/

coco8/
└── images/
    └── train/
```

使用 Python 设置本进程及其子进程的环境变量：

```python
import os
from pathlib import Path

REPO = Path("/mnt/workspace/YOLO-Master").resolve()
COCO_ROOT = Path("/mnt/workspace/datasets/coco").resolve()
CALIB_IMAGES = Path("/mnt/workspace/datasets/coco8/images/train").resolve()
WEIGHT_ROOT = Path("/mnt/workspace/weights").resolve()
OUTPUT_DIR = Path("/mnt/workspace/a3_five_family_coco_results").resolve()

required_paths = {
    "COCO8 calibration images": CALIB_IMAGES,
    "COCO val images": COCO_ROOT / "images/val2017",
    "COCO val labels": COCO_ROOT / "labels/val2017",
    "MoE weight": Path("/mnt/workspace/yolo_master_n.pt"),
    "MoA weight": WEIGHT_ROOT / "moa_coco.pt",
    "MoT weight": WEIGHT_ROOT / "mot_coco.pt",
    "MoLoRA weight": WEIGHT_ROOT / "molora_coco.pt",
    "Latent weight": WEIGHT_ROOT / "latent_coco.pt",
    "MoE embedded config": Path("/mnt/workspace/a3_coco_training/moe_checkpoint_architecture.yaml"),
}
missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
if missing:
    raise FileNotFoundError("缺少正式实验输入：\n" + "\n".join(missing))

os.environ.update(
    {
        "A3_DEVICE": "cuda:0",
        "A3_EXPORT_DEVICE": "cpu",
        "A3_CONFIG": str(REPO / "configs/a3_five_family_precision_coco.yaml"),
        "A3_OUTPUT_DIR": str(OUTPUT_DIR),
        "A3_COCO_ROOT": str(COCO_ROOT),
        "A3_COCO_CALIB_IMAGES": str(CALIB_IMAGES),
        "A3_MOE_WEIGHT": str(required_paths["MoE weight"]),
        "A3_MOE_CONFIG": str(required_paths["MoE embedded config"]),
        "A3_MOA_WEIGHT": str(required_paths["MoA weight"]),
        "A3_MOT_WEIGHT": str(required_paths["MoT weight"]),
        "A3_MOLORA_WEIGHT": str(required_paths["MoLoRA weight"]),
        "A3_LATENT_WEIGHT": str(required_paths["Latent weight"]),
        "A3_VALIDATION_SAMPLES": "5000",
        "A3_STAGE0_SAMPLES": "500",
    }
)
```

如果 MoLoRA 使用“基础 checkpoint + adapter 目录”，再设置：

```python
os.environ["A3_MOLORA_ADAPTER"] = "/mnt/workspace/weights/molora_adapter"
```

如果 MoLoRA checkpoint 已经合并 adapter，则不要设置这个变量：

```python
os.environ.pop("A3_MOLORA_ADAPTER", None)
```

## 6. Python 运行器与日志

下面的函数不依赖 Bash 的 `tee`，会同时把输出显示在 Notebook 中并写入日志：

```python
from pathlib import Path
import os
import subprocess
import sys

def run_a3(stage: str, *, family: str | None = None, extra_args=()):
    config = Path(os.environ["A3_CONFIG"]).resolve()
    output_dir = Path(os.environ["A3_OUTPUT_DIR"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"cloud_run_{stage}.log"

    command = [
        sys.executable,
        "scripts/run_a3_five_family_precision.py",
        "--config",
        str(config),
        "--stage",
        stage,
    ]
    if family is not None:
        command.extend(["--family", family])
    command.extend(str(item) for item in extra_args)

    print("运行：", command)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"A3 stage={stage!r} 失败，退出码={return_code}，日志={log_path}"
        )
    return log_path
```

## 7. 第 0 阶段：真实训练 MoT 路由漂移

这一阶段强制位于五族锁定和量化之前：

```python
run_a3("stage0")
```

它同时检查：

- checkpoint 必须是存在的 `.pt` 文件；
- checkpoint 必须带有正数 `train_args.epochs`；
- 所有 MoT Router 末端投影必须有限、非零；
- 不允许注入随机 Router；
- 记录逐图、逐层 FP16/INT8 路由漂移。

验收文件位于 `A3_OUTPUT_DIR/stage0_trained_mot_drift/`：

- `trained_mot_router_gate.json`；
- `trained_mot_routing_drift.json`；
- `trained_mot_routing_per_sample.csv`。

## 8. 锁定五族、配置、数据和运行环境

```python
run_a3("lock")
```

`experiment.lock.json` 会记录：

- 五族配置、checkpoint 和可选 adapter 的路径与 SHA-256；
- checkpoint 类别数、真实模块类型、路由层和训练元数据；
- COCO 校准/验证图像的精确列表及内容摘要；
- 标签和数据 YAML 摘要；
- Git 状态、Python、PyTorch、CUDA、cuDNN、ONNX Runtime 和 Provider。

后续任何锁定内容发生变化都会停止运行。确实要创建一份新实验锁时才使用：

```python
run_a3("lock", extra_args=("--refresh-lock",))
```

## 9. 建议先执行 GPU 小矩阵烟测

烟测必须使用独立输出目录，不能和正式结果混用：

```python
SMOKE_DIR = Path("/mnt/workspace/a3_coco_smoke").resolve()
os.environ.update(
    {
        "A3_OUTPUT_DIR": str(SMOKE_DIR),
        "A3_VALIDATION_SAMPLES": "32",
        "A3_STAGE0_SAMPLES": "8",
        "A3_ROUTE_SAMPLES": "4",
        "A3_SENSITIVITY_SAMPLES": "16",
        "A3_SENSITIVITY_CALIBRATION_SAMPLES": "4",
        "A3_MAX_GROUPS": "4",
        "A3_LATENCY_REPEATS": "10",
        "A3_BENCHMARK_IMAGES": "4",
    }
)

run_a3("all")
```

烟测通过只证明代码、权重和 Provider 链路可运行，不代表完整 COCO 结论。

## 10. 正式 COCO 五族完整运行

先删除烟测缩减变量并切换到新的正式输出目录：

```python
for name in (
    "A3_ROUTE_SAMPLES",
    "A3_SENSITIVITY_SAMPLES",
    "A3_SENSITIVITY_CALIBRATION_SAMPLES",
    "A3_MAX_GROUPS",
    "A3_LATENCY_REPEATS",
    "A3_BENCHMARK_IMAGES",
):
    os.environ.pop(name, None)

os.environ.update(
    {
        "A3_OUTPUT_DIR": "/mnt/workspace/a3_five_family_coco_results",
        "A3_VALIDATION_SAMPLES": "5000",
        "A3_STAGE0_SAMPLES": "500",
    }
)

run_a3("all")
```

`all` 的固定顺序是：

1. 真实训练 MoT 路由漂移；
2. 五族资产与环境锁定；
3. FP32 导出、FP16 转换、全 INT8 和手工回退构建；
4. 统一精度、时延、体积和失败算子验证；
5. 逐层路由漂移和敏感组 mAP 消融；
6. 漂移与 mAP 损失相关性；
7. 自动敏感层选择和 FP32 回退；
8. FP32、FP16、全 INT8、手工回退、自动回退最终对比。

## 11. 云端中断后的 Python 分阶段恢复

已有锁文件且输入没有变化时，可按阶段恢复：

```python
for stage in ("export", "validate-pre", "sensitivity", "auto", "compare"):
    run_a3(stage)
```

只恢复一个族：

```python
run_a3("sensitivity", family="mot")
```

最终正式报告仍要求五族全部完成。分族恢复后，应再运行：

```python
run_a3("compare")
```

## 12. 逐层路由和回退语义

- 稀疏路由族直接采集 FP32/全 INT8 ONNX 的真实 TopK 值和索引。
- 没有 TopK 节点的稠密 MoA/Latent Router 采集真实 Softmax 概率，并使用统一 Top-2 排名探针。
- 每层计算有序/无序 Top-K 完全一致率、Jaccard、token/sample 翻转率、Top-1 翻转率、
  Top-1/Top-2 margin、概率 MAE、总变差和 Jensen-Shannon。
- 单 Router INT8 假量化提供因果漂移证据；真实全 INT8 ONNX 中间输出提供部署图证据。
- 敏感层评分结合路由漂移与逐组 mAP 损失。
- 手工回退和自动回退把选中的 Router、TopK 上游、Attention 或 MatMul 节点保留为 FP32；
  其他可量化节点保持 INT8。
- 独立 FP16 图是半精度基线，不冒充 INT8 图内的 FP16 混合回退。

所有 ONNX 精度图使用 masked-dense 语义，用于比较数值精度和路由稳定性，不宣称真正减少专家执行。
真正的条件专家运行时由另一套动态运行时任务独立验收。

## 13. 输出和验收

每族目录包含：

- `artifacts/fp32.onnx`；
- `artifacts/fp16.onnx`；
- `artifacts/full_int8.onnx`；
- `artifacts/manual_fallback.onnx`；
- `artifacts/auto_fallback.onnx`；
- `comparison.pre_auto.json`；
- `comparison.final.json`；
- `route_drift.int8_per_layer.json`；
- `route_drift.one_router_int8_per_layer.json`；
- `sensitivity.json`；
- `artifacts/*.quantization.json`。

全局结果：

- `experiment.lock.json`；
- `five_family_precision_summary.json`；
- `five_family_precision_table.csv`；
- `five_family_precision_report.md`；
- `pipeline_failures.json`。

正式验收至少应满足：

- `five_family_precision_summary.json` 的 `status` 为 `success`；
- 五族、五个方案均存在且验证成功；
- FP32、FP16、INT8 时延记录实际 Provider；
- CPU 回退节点数量和算子类型已记录；
- 全 INT8 图确实包含 Q/DQ 或量化算子，不接受“零节点量化”的假成功；
- 敏感层选择原因、mAP 损失和相关性均可追溯；
- 第 0 阶段真实训练门禁和 Router 非退化门禁均通过。

## 14. 可选：VisDrone 复现实验

现有 10 类 VisDrone 权重只能用于 VisDrone，不得用于 COCO mAP。使用 Python 切换配置和路径：

```python
os.environ.update(
    {
        "A3_CONFIG": str(REPO / "configs/a3_five_family_precision.yaml"),
        "A3_OUTPUT_DIR": "/mnt/workspace/a3_five_family_visdrone_results",
        "A3_DATA_YAML": "/mnt/workspace/visdrone/data.yaml",
        "A3_CALIB_IMAGES": "/mnt/workspace/visdrone/images/train",
        "A3_VAL_IMAGES": "/mnt/workspace/visdrone/images/val",
        "A3_VAL_LABELS": "/mnt/workspace/visdrone/labels/val",
        "A3_MOE_WEIGHT": "/mnt/workspace/weights/moe_visdrone.pt",
        "A3_MOA_WEIGHT": "/mnt/workspace/weights/moa_visdrone.pt",
        "A3_MOT_WEIGHT": "/mnt/workspace/weights/mot_v10_visdrone_50e.pt",
        "A3_MOLORA_WEIGHT": "/mnt/workspace/weights/molora_visdrone.pt",
        "A3_LATENT_WEIGHT": "/mnt/workspace/weights/latent_visdrone.pt",
    }
)

run_a3("all")
```

已知的真实训练 MoT VisDrone checkpoint SHA-256：
`5f6ff684f74c773de5cdf0a2e4a51773317829bbeccc48c5c13f0ed3a2f3d417`。
