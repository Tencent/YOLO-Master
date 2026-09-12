# A3 四族缺失权重训练手册（A10 / Python / 额度版）

## 1. 锁定方案

本方案用于五族 FP32/FP16/INT8 和路由漂移对照，不宣称完整 COCO 收敛精度。

- 单张 NVIDIA A10 22 GiB；
- `/mnt/workspace/yolo26n.pt` 初始化缺失四族；
- 每族使用同一个固定 COCO train2017 10% 子集训练；
- 完整 COCO val2017（5000 张）验证；
- 5 epoch，`imgsz=640`，`batch=4`，AMP，seed=42；
- 顺序为 MoT、MoA、Latent、MoLoRA，禁止并行；
- 已有 MoE 继续使用 `/mnt/workspace/yolo_master_n.pt`。

原来的完整 COCO、30 epoch 任务约需 78 小时，而且 batch 8 会 OOM。新任务使用
`/mnt/workspace/a3_coco_training_budget_v1`，不会续跑旧目录。

## 2. 上传并覆盖代码

把 `a3_precision_harness_cloud_20260903_v3.tar.gz` 上传到 `/mnt/workspace/` 后运行：

```python
from pathlib import Path
import hashlib
import tarfile

REPO = Path("/mnt/workspace/YOLO-Master").resolve()
BUNDLE = Path("/mnt/workspace/a3_precision_harness_cloud_20260903_v3.tar.gz").resolve()
print("SHA-256:", hashlib.sha256(BUNDLE.read_bytes()).hexdigest().upper())

with tarfile.open(BUNDLE, "r:gz") as archive:
    for member in archive.getmembers():
        target = (REPO / member.name).resolve()
        if target != REPO and REPO not in target.parents:
            raise RuntimeError(f"压缩包包含越界路径：{member.name}")
    archive.extractall(REPO)

for relative in (
    "scripts/train_a3_missing_coco_weights.py",
    "configs/a3_missing_coco_weights.yaml",
):
    path = REPO / relative
    print(path, path.is_file())
    assert path.is_file(), path
```

## 3. GPU 依赖门禁

每次更换云实例后运行一次。Ultralytics Settings 自动创建不是错误；新实例可能需要重装 `polars`。

```python
import importlib.util
import subprocess
import sys
import torch

if importlib.util.find_spec("polars") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "polars>=0.20.0"],
        check=True,
    )

assert torch.cuda.is_available(), "当前实例没有 CUDA GPU"
props = torch.cuda.get_device_properties(0)
print(torch.__version__, torch.version.cuda, props.name, props.total_memory / 1024**3)
```

## 4. 确认旧任务停止

```python
import psutil

active = []
for process in psutil.process_iter(["pid", "cmdline"]):
    try:
        command = " ".join(process.info["cmdline"] or [])
        if "train_a3_missing_coco_weights.py" in command and "--mode train" in command:
            active.append((process.info["pid"], command))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
print("训练进程:", active)
assert not active, "旧训练子进程仍在运行"
```

结果必须是 `[]`。

## 5. 完整训练控制代码

该代码实时显示并保存输出；中断 Notebook 时也会终止训练子进程。

```python
from pathlib import Path
import os
import subprocess
import sys

REPO = Path("/mnt/workspace/YOLO-Master").resolve()
OUTPUT_ROOT = Path("/mnt/workspace/a3_coco_training_budget_v1").resolve()
LOG_ROOT = OUTPUT_ROOT / "logs"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_ENV = os.environ.copy()
TRAIN_ENV.update({
    "A3_COCO_ROOT": "/mnt/workspace/datasets/coco",
    "A3_MOE_WEIGHT": "/mnt/workspace/yolo_master_n.pt",
    "A3_INITIALIZER_COCO_WEIGHT": "/mnt/workspace/yolo26n.pt",
    "A3_TRAIN_OUTPUT_ROOT": str(OUTPUT_ROOT),
    "A3_WEIGHT_ROOT": "/mnt/workspace/weights",
    "PYTHONUNBUFFERED": "1",
})

def train_family(family: str) -> None:
    if family not in {"mot", "moa", "latent", "molora"}:
        raise ValueError(f"未知模型族：{family}")
    command = [
        sys.executable, "-u", "scripts/train_a3_missing_coco_weights.py",
        "--mode", "train",
        "--family", family,
        "--epochs", "5",
        "--fraction", "0.10",
        "--batch", "4",
        "--workers", "8",
        "--device", "cuda:0",
        "--resume", "auto",
        "--skip-dataset-scan",
    ]
    log_path = LOG_ROOT / f"{family}_training.log"
    print("开始训练:", family, flush=True)
    print("日志:", log_path, flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=TRAIN_ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{family} 训练失败，返回码={return_code}，查看 {log_path}")
    print(f"{family} 训练完成。", flush=True)
```

## 6. 逐族运行

先只运行 MoT：

```python
train_family("mot")
```

完成并生成 `/mnt/workspace/weights/mot_coco.pt` 后，依次分别运行：

```python
train_family("moa")
```

```python
train_family("latent")
```

```python
train_family("molora")
```

不要一次启动多个族。每个 epoch 保存一次 `last.pt`；实例中断后用相同调用自动续训。

## 7. 最终验收

```python
command = [
    sys.executable, "-u", "scripts/train_a3_missing_coco_weights.py",
    "--mode", "verify",
]
subprocess.run(command, cwd=REPO, env=TRAIN_ENV, check=True)
```

最终必须输出 `"status": "passed"`。之后先运行真实训练 MoT 路由漂移，再进入五族量化验证。

## 8. 关键说明

- `--skip-dataset-scan` 只跳过重复遍历文件，Ultralytics 仍会读取现有标签缓存；
- 训练子集比例固定为 10%，四族输入预算一致；
- 每轮训练后仍使用完整 val2017 验证；
- batch 4 是根据本次 A10 上 batch 8 OOM 的实测结果锁定；
- 若 Notebook 中断，不要删除 budget 目录，重新调用同一族即可续训。
