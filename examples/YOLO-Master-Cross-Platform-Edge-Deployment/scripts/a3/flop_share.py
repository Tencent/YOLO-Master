import torch, torch.nn as nn
from ultralytics import YOLO
from diagnose_moe import _fix_add_residual, _strip_property_shadows

# per-module conv FLOP accounting via forward hooks (MACs), CPU, batch 1, 640
def conv_macs(m, inp, out):
    x = inp[0]
    # standard conv MAC = Cout * Hout * Wout * (Cin/groups * kh * kw)
    cout = out.shape[1]; hw = out.shape[2] * out.shape[3]
    cin = m.in_channels // m.groups
    kh, kw = m.kernel_size
    m._macs = cout * hw * cin * kh * kw

for tag, w in [("v0.1-N", "/data/YOLO-Master/YOLO-Master-v0.1-N.pt"),
               ("v0.1-S", "/data/yolo-master-edge/models/larger/weights/YOLO-Master-v0.1-S.pt"),
               ("v0.1-L", "/data/yolo-master-edge/models/larger/weights/YOLO-Master-v0.1-L.pt")]:
    y = YOLO(w); _fix_add_residual(y); _strip_property_shadows(y)
    net = y.model.eval()
    hooks = [mod.register_forward_hook(conv_macs) for mod in net.modules() if isinstance(mod, nn.Conv2d)]
    for mod in net.modules():
        mod._macs = 0
    with torch.no_grad():
        net(torch.zeros(1, 3, 640, 640))
    for h in hooks: h.remove()
    # classify each conv: which MoE block + is it an expert?
    total = 0; expert_by_block = {}; block_k = {}
    moe_names = {n: m for n, m in net.named_modules() if hasattr(m, "experts") and hasattr(m, "routing")}
    for n, mod in net.named_modules():
        if isinstance(mod, nn.Conv2d):
            total += mod._macs
            for mn, mm in moe_names.items():
                if n.startswith(mn + ".experts."):
                    E = len(mm.experts); k = int(getattr(mm, "top_k", 2))
                    expert_by_block[mn] = expert_by_block.get(mn, 0) + mod._macs
                    block_k[mn] = (E, k)
    exp_total = sum(expert_by_block.values())
    # dense computes all E; sparse computes k. saved = sum over blocks of (E-k)/E * block_expert_macs
    saved = 0
    for mn, macs in expert_by_block.items():
        E, k = block_k[mn]
        saved += macs * (E - k) / E
    print(f"== {tag}: total {total/1e9:.2f} GMAC | experts {exp_total/1e9:.2f} GMAC "
          f"({100*exp_total/total:.1f}% of conv MACs)")
    print(f"   theoretical batch=1 sparse saving: {saved/1e9:.2f} GMAC = "
          f"{100*saved/total:.1f}% of TOTAL, {100*saved/exp_total:.1f}% of expert MACs")
    for mn in sorted(expert_by_block):
        E, k = block_k[mn]
        print(f"      {mn}: E={E} k={k}  expert MACs {expert_by_block[mn]/1e9:.3f}G  "
              f"-> compute {k}/{E}")
