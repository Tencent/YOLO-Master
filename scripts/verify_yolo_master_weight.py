"""Verify YOLO-Master-EsMoE-N.pt model loading and MoE-aware PEFT compatibility."""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO

MODEL_PATH = REPO_ROOT / "YOLO-Master-EsMoE-N.pt"
print(f"Loading {MODEL_PATH}...")
model = YOLO(str(MODEL_PATH))

print(f"\nModel info:")
print(f"  type: {type(model.model).__name__}")
print(f"  total params: {sum(p.numel() for p in model.model.parameters()):,}")
print(f"  named modules: {len(list(model.model.named_modules()))}")

# Check for MoE modules
moe_modules = []
for name, mod in model.model.named_modules():
    if "moe" in name.lower() or "expert" in name.lower() or "router" in name.lower():
        moe_modules.append((name, type(mod).__name__))

print(f"\nMoE-related modules ({len(moe_modules)}):")
for name, t in moe_modules[:20]:
    print(f"  {name}: {t}")
if len(moe_modules) > 20:
    print(f"  ... and {len(moe_modules)-20} more")

# Check Conv2d/Linear layers for PEFT wrapping targets
conv_linear = [(name, type(mod).__name__) for name, mod in model.model.named_modules()
               if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear))]
print(f"\nConv2d/Linear layers: {len(conv_linear)}")
print("  First 10:")
for name, t in conv_linear[:10]:
    print(f"    {name}: {t}")

# Try MoLoRA auto-detect
from ultralytics.nn.peft.molora.config import MoLoRAConfigBuilder
targets = MoLoRAConfigBuilder.auto_detect_targets(
    model.model, r=8, include_moe=True, only_backbone=False
)
print(f"\nMoLoRA auto-detect targets: {len(targets)}")
print("  First 10:")
for t in targets[:10]:
    print(f"    {t}")

print("\n✅ YOLO-Master-EsMoE-N.pt is compatible with MoE-aware PEFT")
