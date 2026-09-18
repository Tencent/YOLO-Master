#!/usr/bin/env bash
# A3 8.24 entry check: L4 environment bootstrap (idempotent, re-run after a pod rebuild).
# The pod ships a bare torch 2.4.1+cu124; everything else is pinned here so the
# environment matrix (a3/env/) is reproducible. Never lets pip replace torch.
set -euo pipefail
VENV=${VENV:-/root/a3venv}
EDGE=${EDGE:-/data/yolo-master-edge}
YM=${YM:-/data/YOLO-Master}
export PATH=/usr/local/cuda-12.4/bin:$PATH
export YOLO_AUTOINSTALL=False
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
. "$VENV/bin/activate"
assert_torch() { python -c "import torch; v=torch.__version__; assert v.startswith('2.4.1'), v; print('[torch]', v, 'cuda', torch.version.cuda)"; }
assert_torch
pip install -q -U pip wheel setuptools
pip install -q tensorrt-cu12==10.13.3.9
pip install -q onnx==1.17.0 "onnxslim>=0.1.48" onnxruntime-gpu==1.20.2 pycocotools \
    opencv-python-headless pyyaml matplotlib pandas tqdm psutil scipy ultralytics-thop \
    py-cpuinfo requests pillow seaborn polars
assert_torch
pip install -q "nvidia-modelopt[torch]==0.27.1" || pip install -q "nvidia-modelopt[torch]==0.29.0"
assert_torch
# editable install of the locked YOLO-Master with its declared deps resolved, but torch
# pinned to the preinstalled wheel via a constraints file (never re-downloaded)
echo "torch==$(python -c 'import torch; print(torch.__version__)')" > /tmp/a3-constraints.txt
pip install -q -c /tmp/a3-constraints.txt -e "$YM"
git config --global --add safe.directory "$YM" >/dev/null 2>&1 || true
git config --global --add safe.directory "$EDGE" >/dev/null 2>&1 || true
python - <<'PY'
import torch  # first: preloads cuDNN 9 for ORT
import tensorrt, onnxruntime as ort, onnx, ultralytics
import modelopt.torch.quantization as mtq, modelopt
print("[trt]", tensorrt.__version__)
print("[ort]", ort.__version__, ort.get_available_providers())
print("[onnx]", onnx.__version__, "[modelopt]", modelopt.__version__)
print("[ultralytics]", ultralytics.__version__, ultralytics.__file__)
assert "CUDAExecutionProvider" in ort.get_available_providers(), "ORT CUDA EP missing"
assert "/data/YOLO-Master" in ultralytics.__file__, "ultralytics is not the locked YOLO-Master checkout"
PY
mkdir -p "$EDGE/a3/env"
pip freeze > "$EDGE/a3/env/pip-freeze.txt"
echo "[setup] done: $VENV"
