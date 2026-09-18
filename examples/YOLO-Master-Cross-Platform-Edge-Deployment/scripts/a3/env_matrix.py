#!/usr/bin/env python
"""A3 8.24 entry check: environment matrix + backend versions -> a3/env/matrix.{json,md}.

Captures everything a reviewer needs to reproduce a number: host/GPU/driver/CUDA,
python + torch stack, every backend (onnx, onnxruntime + providers, tensorrt, modelopt),
the ultralytics fork version, and the LOCKED commits of both repos (with dirty-tree flags).

Usage (inside the a3 venv):
  python scripts/a3/env_matrix.py --out a3/env
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import socket
import subprocess
from pathlib import Path


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def ver(mod: str, attr: str = "__version__") -> str:
    try:
        m = __import__(mod)
        for part in mod.split(".")[1:]:
            m = getattr(m, part)
        return str(getattr(m, attr, "?"))
    except Exception as e:  # noqa: BLE001
        return f"MISSING ({type(e).__name__})"


def git_info(repo: str) -> dict:
    head = sh(f"git -C {repo} rev-parse HEAD")
    date = sh(f"git -C {repo} log -1 --format=%cd --date=short")
    branch = sh(f"git -C {repo} rev-parse --abbrev-ref HEAD")
    dirty = sh(f"git -C {repo} status --porcelain --untracked-files=no")
    return {"path": repo, "commit": head, "date": date, "branch": branch,
            "dirty_tracked_files": [l.split(maxsplit=1)[-1] for l in dirty.splitlines() if l.strip()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="a3/env")
    ap.add_argument("--edge-repo", default=".")
    ap.add_argument("--yolo-repo", default="/data/YOLO-Master")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch  # noqa: E402  (first: preloads cuDNN for onnxruntime)
    providers = "MISSING"
    try:
        import onnxruntime as ort
        providers = ",".join(ort.get_available_providers())
    except Exception as e:  # noqa: BLE001
        providers = f"MISSING ({type(e).__name__})"

    m = {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "os": sh("lsb_release -ds 2>/dev/null || head -1 /etc/os-release"),
        "kernel": platform.release(),
        "cpu": sh("lscpu | awk -F: '/Model name/{print $2}' | xargs"),
        "ram_gb": sh("free -g | awk '/Mem:/{print $2}'"),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
        "cuda_driver_max": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1"),
        "cuda_toolkit_nvcc": sh("nvcc --version 2>/dev/null | tail -1 | xargs || true"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "torchvision": ver("torchvision"),
        "numpy": ver("numpy"),
        "onnx": ver("onnx"),
        "onnxslim": ver("onnxslim"),
        "onnxruntime": ver("onnxruntime"),
        "onnxruntime_providers": providers,
        "tensorrt": ver("tensorrt"),
        "modelopt": ver("modelopt"),
        "pycocotools": ver("pycocotools") if "MISSING" in ver("pycocotools") else "installed (no __version__)",
        "opencv": ver("cv2"),
        "ultralytics": ver("ultralytics"),
        "ultralytics_file": ver("ultralytics", "__file__"),
        "yolo_master_repo": git_info(args.yolo_repo),
        "edge_repo": git_info(args.edge_repo),
    }
    (out / "matrix.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))

    rows = []
    for k, v in m.items():
        if isinstance(v, dict):
            v = (f"`{v['commit'][:12]}` ({v['date']}, {v['branch']}"
                 + (f", dirty: {', '.join(v['dirty_tracked_files'])}" if v["dirty_tracked_files"] else "")
                 + ")")
        rows.append(f"| {k} | {v} |")
    md = "| item | value |\n|---|---|\n" + "\n".join(rows) + "\n"
    (out / "matrix.md").write_text(md)
    print(md)
    print(f"[env] wrote {out / 'matrix.json'} and matrix.md")


if __name__ == "__main__":
    main()
