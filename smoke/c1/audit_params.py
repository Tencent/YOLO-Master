import sys
from pathlib import Path

ROOT = Path("D:/YOLO-Master")
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

CONFIGS = {
    "v0.1-N":      "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",
    "EsMoE-N":     "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml",
    "EsMoE-P2-N":  "ultralytics/cfg/models/master/v0/det/yolo-master-n-p2.yaml",
    "v0.1-P2-N":   "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2.yaml",
    "UoMoE-N":     "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe.yaml",
    "UoMoE-P2-N":  "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe-p2.yaml",
}

print(f"{'model':<12}{'params(M)':>12}{'trainable(M)':>14}")
for name, cfg in CONFIGS.items():
    try:
        m = YOLO(str(ROOT / cfg), task="detect")
        total = sum(p.numel() for p in m.model.parameters())
        trainable = sum(p.numel() for p in m.model.parameters() if p.requires_grad)
        print(f"{name:<12}{total/1e6:>12.3f}{trainable/1e6:>14.3f}")
    except Exception as e:
        print(f"{name:<12}  ERROR: {e}")
