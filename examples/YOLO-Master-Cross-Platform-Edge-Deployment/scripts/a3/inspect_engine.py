import sys, json
from collections import Counter
import tensorrt as trt

path = sys.argv[1]
logger = trt.Logger(trt.Logger.ERROR)
rt = trt.Runtime(logger)
raw = open(path, "rb").read()
meta_len = int.from_bytes(raw[:4], byteorder="little", signed=True)
eng = rt.deserialize_cuda_engine(raw[4 + meta_len:])
insp = eng.create_engine_inspector()
info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
layers = info.get("Layers", [])

def prec(l):
    outs = l.get("Outputs") or []
    fmt = outs[0].get("Format/Datatype", "?") if outs else "?"
    return fmt.split()[0] if isinstance(fmt, str) else "?"

def ltype(l):
    return l.get("LayerType", "?")

def kind(l):
    name = (l.get("Name") or "").lower()
    t = ltype(l).lower()
    if "reformat" in t or "reformat" in name or "copy" in name: return "reformat"
    if "constantoutput" in t or t == "constant": return "constant"
    return t or "other"

prec_ct = Counter(prec(l) for l in layers)
type_ct = Counter(ltype(l) for l in layers)
print(f"== {path.split('/')[-2]}")
print(f"   layers={len(layers)}  precision={dict(prec_ct)}")
print(f"   layer TYPES: {dict(type_ct.most_common())}")

# reformats: count + total (reformats are the INT8<->float transition cost)
reformats = [l for l in layers if kind(l) == "reformat"]
print(f"   REFORMAT layers: {len(reformats)}  precisions={dict(Counter(prec(l) for l in reformats))}")

# the Float (FP32) layers: what kind are they?
fp = [l for l in layers if prec(l) == "Float"]
print(f"   FLOAT layers: {len(fp)}  by TYPE: {dict(Counter(ltype(l) for l in fp).most_common())}")
for l in fp[:30]:
    print(f"      {ltype(l):16s} {l.get('Name','')[:80]}")
