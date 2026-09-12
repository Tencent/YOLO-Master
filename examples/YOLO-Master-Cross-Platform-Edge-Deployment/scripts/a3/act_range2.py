import glob, numpy as np, torch, torch.nn as nn
from ultralytics import YOLO
from diagnose_moe import _fix_add_residual, _strip_property_shadows
import quantize_trt as qt

W = {"S": "/data/yolo-master-edge/models/larger/weights/YOLO-Master-v0.1-S.pt",
     "M": "/data/yolo-master-edge/models/larger/weights/YOLO-Master-v0.1-M.pt"}
imgs = sorted(glob.glob("/data/datasets/coco/images/train2017/*.jpg"))[:12]
torch.set_num_threads(8)

def analyze(tag, w):
    y = YOLO(w); _fix_add_residual(y); _strip_property_shadows(y)
    net = y.model.eval()
    top = list(net.model)
    name_of = {}
    for bi, blk in enumerate(top):
        bt = type(blk).__name__
        for cn, c in blk.named_modules():
            if isinstance(c, nn.Conv2d):
                # label the conv role inside the block by its submodule name
                role = "attn" if any(k in cn.lower() for k in ("attn","qkv","pe","proj")) else \
                       ("head" if bt == "Detect" else "conv")
                name_of[c] = (bi, bt, cn, role)
    acc = {}
    def hook(m, inp, out):
        x = inp[0].detach().float().abs().flatten()
        n = x.numel()
        if n > 50000:
            idx = torch.randint(0, n, (50000,)); x = x[idx]
        xv = x.numpy()
        mx = float(xv.max()) if xv.size else 0.0
        p = float(np.percentile(xv, 99.9)) if xv.size else 1e-9
        s = acc.setdefault(id(m), [0.0, [], name_of[m]])
        s[0] = max(s[0], mx); s[1].append(p)
    hs = [c.register_forward_hook(hook) for c in name_of]
    with torch.no_grad():
        for pth in imgs:
            net(torch.from_numpy(qt.letterbox_blob(pth, 640)))
    for h in hs: h.remove()
    per_block = {}
    for sid, (mx, ps, (bi, bt, cn, role)) in acc.items():
        p = float(np.median(ps)) if ps else 1e-9
        ratio = mx / max(p, 1e-9)
        b = per_block.setdefault(bi, [bt, 0.0, 0.0, ""])
        if ratio > b[1]:
            b[1] = ratio; b[2] = mx; b[3] = f"{role}:{cn}"
    return per_block

for tag in ("S", "M"):
    pb = analyze(tag, W[tag])
    print(f"\n== {tag}: per-block WORST conv-input outlier ratio (max/p99.9) ==", flush=True)
    print("  idx type            worst_ratio   max|act|   worst_conv", flush=True)
    for bi in sorted(pb):
        bt, ratio, mx, role = pb[bi]
        flag = "  <<<< KILLER" if ratio > 80 else ("  <<< high" if ratio > 40 else "")
        print(f"  {bi:3d} {bt:14s}  {ratio:9.1f}  {mx:9.1f}   {role[:40]}{flag}", flush=True)
print("\nDONE", flush=True)
