import sys
from pathlib import Path

# Pin the repository so the OFFICIAL 43d4011 (YOLO-Master-v26.08) ultralytics package is used.
ROOT = Path("D:/YOLO-Master")
sys.path.insert(0, str(ROOT))

import torch
print(f"[smoke] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(
        f"[smoke] device0={torch.cuda.get_device_name(0)} "
        f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
        flush=True,
    )

from ultralytics import YOLO

# EsMoE-N config AT locked commit 43d4011 (YOLO-Master-v26.08 tag)
CFG = "D:/YOLO-Master/ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml"
# Local VisDrone (already on disk, absolute path -> no download)
DATA = "D:/datasets/VisDrone/VisDrone_local.yaml"

print(f"[smoke] model_cfg={CFG}", flush=True)
print(f"[smoke] data={DATA}", flush=True)

model = YOLO(CFG)
results = model.train(
    data=DATA,
    epochs=5,
    imgsz=640,
    batch=8,
    device=0,  # RTX 4060 Laptop GPU
    workers=0,  # Windows: avoid multiprocessing I/O deadlock
    seed=42,
    optimizer="auto",
    pretrained=False,  # fully offline: no pretrained weight download
    amp=False,  # offline: skip AMP reference-weight download (yolo26n.pt)
    name="smoke_c1_esmoe_n",
    project="D:/YOLO-Master/runs/smoke_43d4011",
    exist_ok=True,
    verbose=True,
)
print("[smoke] DONE", flush=True)
