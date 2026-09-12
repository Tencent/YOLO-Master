# A3 确定性 Top-K 云端 548 张复验手册（2026-09-12）

## 运行目的

本轮不是继续 INT8，而是用 PR 分支的新 `max_deadband_then_lowest_expert_id` 合同重新导出真实 MoT
checkpoint 的 6 个动态块，并在 VisDrone val 548 张上比较：

- eager checkpoint FP32；
- ORT CPU 路由器 + 确定性 host Top-K + GPU PyTorch checkpoint 专家；
- mAP50-95 差值、逐位置路由漂移、真实专家调用及缩减。

导出阶段使用 CPU；完整模型和 checkpoint 专家验证使用 GPU 0。当前任务不是 TensorRT 性能测试，日志中的
混合耗时仍只作诊断。

## 单个 Jupyter Python 单元

直接复制整个单元运行。它会实时输出导出和验证日志，科学门禁失败返回码 `2` 也会保留 JSON 和证据包。

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import zipfile


WORKSPACE = Path("/mnt/workspace")
REPO = WORKSPACE / "YOLO-Master"
REMOTE = "https://github.com/KennnMai/YOLO-Master.git"
BRANCH = "rhino-a3-dev/smoke/a3"
MINIMUM_COMMIT = "56325fe"
POLICY = "max_deadband_then_lowest_expert_id"

CKPT = REPO / "examples/artifacts/mot_v10_visdrone_50e/best_for_quantization.pt"
IMAGE = WORKSPACE / "visdrone/images/val/0000364_01765_d_0000782.jpg"
DATA = REPO / "examples/configs/visdrone_local_for_mot.yaml"

run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
BUNDLES = REPO / f"examples/artifacts/mot_v10_dynamic_topk_{run_tag}"
OUT = REPO / f"examples/results/mot_v10_dynamic_topk_full_val_{run_tag}"
RESULT = OUT / "DYNAMIC_MOT_DETERMINISTIC_TOPK_FULL_VAL.json"
EXPORT_LOG = OUT / "export_dynamic_blocks.log"
VAL_LOG = OUT / "full_548_validation.log"
METADATA = OUT / "run_metadata.json"
EVIDENCE = WORKSPACE / f"mot_dynamic_deterministic_topk_{run_tag}_evidence.zip"


def run_checked(command: list[str], *, cwd: Path = REPO) -> str:
    print("RUN:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    return result.stdout.strip()


def run_stream(command: list[str], log_path: Path) -> int:
    print("\nRUN:", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        return process.wait()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


assert (REPO / ".git").is_dir(), f"不是 Git 仓库：{REPO}"
tracked_dirty = run_checked(
    ["git", "status", "--porcelain", "--untracked-files=no"],
).strip()
assert not tracked_dirty, (
    "云端仓库存在 tracked 修改，脚本拒绝覆盖。请先备份/提交这些修改，再重跑：\n" + tracked_dirty
)
run_checked(["git", "fetch", REMOTE, BRANCH])
run_checked(["git", "checkout", "--detach", "FETCH_HEAD"])
source_commit = run_checked(["git", "rev-parse", "HEAD"])
run_checked(["git", "merge-base", "--is-ancestor", MINIMUM_COMMIT, source_commit])
print("source commit:", source_commit, flush=True)

os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo_dynamic_topk_settings"
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

for required in (CKPT, IMAGE, DATA):
    assert required.is_file(), f"缺少文件：{required}"

try:
    import onnxruntime as ort
except ImportError:
    print("Installing CPU ONNX Runtime into:", sys.executable, flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "onnxruntime==1.23.2"]
    )
    importlib.invalidate_caches()
    import onnxruntime as ort

import torch

providers = ort.get_available_providers()
print("Python:", sys.executable, flush=True)
print("PyTorch:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("ONNX Runtime:", ort.__version__, providers, flush=True)
assert torch.cuda.is_available(), "本轮 548 张完整验证需要 GPU"
assert "CPUExecutionProvider" in providers, providers
print("GPU:", torch.cuda.get_device_name(0), flush=True)

export_command = [
    sys.executable,
    "-u",
    str(REPO / "scripts/export_dynamic_blocks_from_checkpoint.py"),
    str(CKPT),
    str(BUNDLES),
    "--input-image",
    str(IMAGE),
    "--imgsz",
    "640",
    "--batch",
    "1",
    "--device",
    "cpu",
    "--family",
    "mot",
    "--validate",
    "--overwrite",
]
export_code = run_stream(export_command, EXPORT_LOG)
assert export_code == 0, f"导出失败，查看：{EXPORT_LOG}"

model_manifest_path = BUNDLES / "dynamic_model_blocks.json"
model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
assert model_manifest["dynamic_block_count"] == 6, model_manifest["dynamic_block_count"]
assert model_manifest["status"] != "FAILED", model_manifest["status"]
for block in model_manifest["blocks"]:
    bundle_path = BUNDLES / block["bundle_manifest"]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["host_topk_tie_break"] == POLICY, (block["module_name"], bundle)
print("6/6 bundles use:", POLICY, flush=True)

validation_command = [
    sys.executable,
    "-u",
    str(REPO / "scripts/validate_dynamic_mot_full_val.py"),
    str(CKPT),
    str(model_manifest_path),
    str(DATA),
    str(RESULT),
    "--device",
    "0",
    "--imgsz",
    "640",
    "--batch",
    "8",
    "--workers",
    "4",
    "--map-tolerance-pct-points",
    "0.5",
    "--require-exact-route",
]
validation_code = run_stream(validation_command, VAL_LOG)
assert validation_code in (0, 2), f"程序异常退出 {validation_code}；查看：{VAL_LOG}"
assert RESULT.is_file(), RESULT

result = json.loads(RESULT.read_text(encoding="utf-8"))
gate = result["accuracy_gate"]
route = result["route_gate"]
dynamic = result["dynamic_execution_gate"]
print("\n===== DETERMINISTIC TOP-K 548 RESULT =====", flush=True)
print("status:", result["status"], flush=True)
print("eager mAP50-95:", gate["eager_mAP50_95"], flush=True)
print("dynamic mAP50-95:", gate["dynamic_mAP50_95"], flush=True)
print("delta (percentage points):", gate["dynamic_minus_eager_percentage_points"], flush=True)
print("route mismatch:", route["mismatched_locations"], "/", route["total_locations"], flush=True)
print("accuracy gate:", gate["passed"], flush=True)
print("exact-route gate:", route["passed"], flush=True)
print("dynamic execution gate:", dynamic["passed"], flush=True)

metadata = {
    "schema_version": 1,
    "source_commit": source_commit,
    "branch": BRANCH,
    "checkpoint": {"path": str(CKPT), "sha256": sha256(CKPT)},
    "topk_policy": POLICY,
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
    "onnxruntime": ort.__version__,
    "providers": providers,
    "export_return_code": export_code,
    "validation_return_code": validation_code,
    "result_status": result["status"],
}
METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with zipfile.ZipFile(EVIDENCE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(RESULT, RESULT.name)
    archive.write(EXPORT_LOG, EXPORT_LOG.name)
    archive.write(VAL_LOG, VAL_LOG.name)
    archive.write(METADATA, METADATA.name)
    archive.write(model_manifest_path, "manifests/dynamic_model_blocks.json")
    for bundle_path in BUNDLES.rglob("dynamic_bundle.json"):
        archive.write(bundle_path, Path("block_manifests") / bundle_path.relative_to(BUNDLES))

print("\n结果 JSON：", RESULT, flush=True)
print("证据包：", EVIDENCE, flush=True)
print("证据包 SHA256：", sha256(EVIDENCE), flush=True)
if validation_code == 2:
    print("注意：程序完整运行，但至少一个科学门禁失败；请保留并下载证据包。", flush=True)
else:
    print("确定性 Top-K 的 548 张严格门禁通过。", flush=True)
```

## 结果判读

- `return code = 0`：精度、严格零漂移和动态执行门禁全部通过。
- `return code = 2`：程序没有崩溃，但至少一个科学门禁失败；必须下载证据包分析，不能删除失败结果。
- 其他返回码：环境或程序异常，先看实时输出和两个 `.log`。

最终只需从 `/mnt/workspace/` 下载打印出的
`mot_dynamic_deterministic_topk_<时间戳>_evidence.zip` 并发回来。不要下载全部 ONNX 文件，证据包已经包含
结果、日志、运行环境、模型清单和 6 个 bundle 清单。
