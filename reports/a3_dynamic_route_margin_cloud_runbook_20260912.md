# MoT 路由裕量云端复验

本轮只新增诊断字段，不改路由、专家执行、预测或门禁。需要在原有云端目录覆盖两个运行时文件后重跑 548 张。

## Jupyter Python 单元

先上传 `dynamic_route_margin_audit_cloud_20260912_v1.zip` 到 `/mnt/workspace/`，然后运行：

```python
from pathlib import Path
import os
import subprocess
import sys
import zipfile

repo = Path("/mnt/workspace/YOLO-Master")
archive = Path("/mnt/workspace/dynamic_route_margin_audit_cloud_20260912_v1.zip")
checkpoint = repo / "examples/artifacts/mot_v10_visdrone_50e/best_for_quantization.pt"
bundles = repo / "examples/artifacts/mot_v10_dynamic_blocks/dynamic_model_blocks.json"
data = repo / "examples/configs/visdrone_local_for_mot.yaml"
result = repo / "examples/results/mot_v10_dynamic_margin_20260912/DYNAMIC_MOT_ROUTE_MARGIN.json"
log = result.with_suffix(".log")

assert repo.is_dir(), repo
assert archive.is_file(), archive
assert checkpoint.is_file(), checkpoint
assert bundles.is_file(), bundles
assert data.is_file(), data

with zipfile.ZipFile(archive) as zf:
    bad = [name for name in zf.namelist() if Path(name).is_absolute() or ".." in Path(name).parts]
    assert not bad, bad
    zf.extractall(repo)

os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo_dynamic_settings"
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
result.parent.mkdir(parents=True, exist_ok=True)

command = [
    sys.executable,
    "-u",
    str(repo / "scripts/validate_dynamic_mot_full_val.py"),
    str(checkpoint),
    str(bundles),
    str(data),
    str(result),
    "--device", "0",
    "--imgsz", "640",
    "--batch", "8",
    "--workers", "4",
    "--map-tolerance-pct-points", "0.5",
]

print("RUN:", " ".join(command), flush=True)
with log.open("w", encoding="utf-8") as stream:
    process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        stream.write(line)
        stream.flush()
    return_code = process.wait()

print("return code:", return_code, flush=True)
print("result:", result, flush=True)
print("log:", log, flush=True)
assert return_code in (0, 2), f"abnormal exit {return_code}; inspect {log}"
assert result.is_file(), result
```

返回码 `0` 表示所有科学门禁通过；`2` 表示程序完整运行但至少一个门禁失败。两种情况都会保留 JSON，禁止只
看返回码而删除失败证据。输出 JSON 的每个 `blocks[*].route_margin_audit` 是本轮新增的核心结果。
