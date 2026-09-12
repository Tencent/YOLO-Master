import os, glob, numpy as np, torch, torch.nn as nn
from ultralytics import YOLO
from diagnose_moe import _fix_add_residual, _strip_property_shadows
import quantize_trt as qt  # letterbox_blob

W = {"S": "/data/yolo-master-edge/models/larger/weights/YOLO-Master-v0.1-S.pt",
     "M": "/data/yolo-master-edge/models/larger/weights/YOLO-Master-v0.1-M.pt"}
imgs = sorted(glob.glob("/data/datasets/coco/images/train2017/*.jpg"))[:24]

def analyze(tag, w):
    y = YOLO(w); _fix_add_residual(y); _strip_property_shadows(y)
    net = y.model.eval()
    # map each Conv2d to its top-level block index + whether it's inside an attention (A2C2f) block
    top = list(net.model)
    stats = {}   # conv_name -> [maxabs, p999 accumulators]
    name_of = {}
    for bi, blk in enumerate(top):
        is_attn = type(blk).__name__ == "A2C2f"
        is_head = type(blk).__name__ == "Detect"
        for cn, c in blk.named_modules():
            if isinstance(c, nn.Conv2d):
                name_of[c] = (bi, type(blk).__name__, is_attn, is_head)
    def hook(m, inp, out):
        x = inp[0].detach().float().abs()
        mx = x.max().item()
        p999 = torch.quantile(x.flatten()[:200000], 0.999).item() if x.numel() else 0
        s = stats.setdefault(id(m), [0.0, [], name_of.get(m, (-1, "?", False, False))])
        s[0] = max(s[0], mx); s[1].append(p999)
    hs = [c.register_forward_hook(hook) for c in name_of]
    with torch.no_grad():
        for p in imgs:
            x = torch.from_numpy(qt.letterbox_blob(p, 640))
            net(x)
    for h in hs: h.remove()
    # aggregate per top-level block: worst outlier ratio (maxabs / median-p999)
    per_block = {}
    for sid, (mx, p999s, (bi, bt, isa, ish)) in stats.items():
        p = float(np.median(p999s)) if p999s else 1e-9
        ratio = mx / max(p, 1e-9)
        b = per_block.setdefault(bi, [bt, isa, ish, 0.0, 0.0])
        b[3] = max(b[3], ratio)   # worst outlier ratio in block
        b[4] = max(b[4], mx)      # worst max activation
    return per_block

for tag in ("S", "M"):
    pb = analyze(tag, W[tag])
    print(f"\n== {tag}: per-block worst activation-outlier ratio (max|act| / p99.9) and max|act| ==")
    print("   idx  type            attn head   outlier_ratio   max|act|")
    for bi in sorted(pb):
        bt, isa, ish, ratio, mx = pb[bi]
        flag = " <<< ATTENTION" if isa else (" <<< HEAD" if ish else ("  <-- outlier" if ratio > 40 else ""))
        print(f"   {bi:3d}  {bt:14s}  {int(isa)}    {int(ish)}     {ratio:8.1f}       {mx:8.1f}{flag}")
