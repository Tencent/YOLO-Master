import torch, torch.nn as nn
from ultralytics import YOLO
from diagnose_moe import _fix_add_residual, _strip_property_shadows

W = {"S": "/data/yolo-master-edge/models/larger/weights/YOLO-Master-v0.1-S.pt",
     "M": "/data/yolo-master-edge/models/larger/weights/YOLO-Master-v0.1-M.pt"}

def block_stats(net):
    rows = {}
    for i, m in enumerate(net.model):
        params = sum(p.numel() for p in m.parameters())
        # first + last conv channels within the block
        convs = [c for c in m.modules() if isinstance(c, nn.Conv2d)]
        cin = convs[0].in_channels if convs else 0
        cout = convs[-1].out_channels if convs else 0
        # widest intermediate channel
        widest = max((c.out_channels for c in convs), default=0)
        rows[i] = (type(m).__name__, cin, cout, widest, len(convs), params)
    return rows

S = {}
for tag, w in W.items():
    y = YOLO(w); _fix_add_residual(y); _strip_property_shadows(y)
    S[tag] = block_stats(y.model.eval())

print("idx  type                 |     S: cin->cout (wide, nconv, params)  |     M: cin->cout (wide, nconv, params)  | M/S params")
for i in sorted(S["S"]):
    s = S["S"][i]; m = S["M"].get(i, ("-",0,0,0,0,0))
    ratio = m[5] / s[5] if s[5] else 0
    flag = "  <<<" if (ratio > 2.6 or ratio < 1.6) and s[5] > 0 else ""
    print(f"{i:3d}  {s[0]:20s} | {s[1]:4d}->{s[2]:4d} (w{s[3]:4d} n{s[4]} {s[5]:>9,}) "
          f"| {m[1]:4d}->{m[2]:4d} (w{m[3]:4d} n{m[4]} {m[5]:>9,}) | {ratio:.2f}x{flag}")

# expert channel widths in each MoE block
print("\n--- MoE expert internal widths (first expert, S vs M) ---")
for tag in ("S", "M"):
    y = YOLO(W[tag]); _fix_add_residual(y); _strip_property_shadows(y)
    for n, mod in y.model.named_modules():
        if hasattr(mod, "experts") and hasattr(mod, "routing"):
            e0 = mod.experts[0]
            cs = [c.in_channels for c in e0.modules() if isinstance(c, nn.Conv2d)]
            co = [c.out_channels for c in e0.modules() if isinstance(c, nn.Conv2d)]
            print(f"  {tag} {n}: E={len(mod.experts)} expert0 conv chans in={cs} out={co}")
