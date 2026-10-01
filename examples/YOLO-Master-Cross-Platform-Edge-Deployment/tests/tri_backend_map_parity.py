import onnx
from pathlib import Path
from ultralytics import YOLO
W = Path("/data/YOLO-Master/scripts/reproduce/results/result-esmoen-visdrone/weights")
SIM = Path("/data/yolo-master-edge/models/esmoe_n_visdrone_sim.onnx")
DATA = "/data/yolo-master-edge/visdrone50/visdrone50.yaml"
raw = onnx.load(str(W/"best.onnx")); sim = onnx.load(str(SIM))
meta = {p.key: p.value for p in raw.metadata_props}
del sim.metadata_props[:]
for k, v in meta.items():
    e = sim.metadata_props.add(); e.key, e.value = k, v
onnx.save(sim, str(SIM))
print(f"[meta] restored {list(meta.keys())}", flush=True)
backends = {"PyTorch (.pt)": str(W/"best.pt"),
            "ONNX-sim (ORT)": str(SIM),
            "NCNN (pnnx)": str(W/"best_ncnn_model")}
res = {}
for name, mdl in backends.items():
    r = YOLO(mdl, task="detect").val(data=DATA, imgsz=640, batch=1, device="cpu", workers=0, plots=False, verbose=False)
    res[name] = (float(r.box.map50), float(r.box.map), r.speed.get("inference", 0.0))
    print(f"[done] {name:16s} mAP50={res[name][0]:.4f} mAP50-95={res[name][1]:.4f} inf={res[name][2]:.1f}ms", flush=True)
base = res["PyTorch (.pt)"][1]
print("\n=== 50-img parity vs PyTorch (mAP50-95) ===", flush=True)
for name,(m50,m5095,_) in res.items():
    print(f"  {name:16s} mAP50-95={m5095:.4f}  d_vs_pt={(m5095-base)*100:+.3f}pts", flush=True)
