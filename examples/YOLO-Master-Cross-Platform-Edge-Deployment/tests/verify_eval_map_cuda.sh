set -e
echo "### C++ CUDA binary (my letterbox+decode+per-class-NMS pipeline):"
python /data/yolo-master-edge/scripts/eval_map.py --preds /data/yolo-master-edge/preds_cuda
echo "### ultralytics val references (device=0):"
python - <<'PY'
from ultralytics import YOLO
PT="/data/YOLO-Master/scripts/reproduce/results/result-esmoen-visdrone/weights/best.pt"
ONNX="/data/yolo-master-edge/models/esmoe_n_visdrone_sim.onnx"
for tag,m in [("PyTorch .pt",PT),("ONNX-sim CUDA",ONNX)]:
    r=YOLO(m,task="detect").val(data="VisDrone.yaml", imgsz=640, batch=1, device=0, workers=0, verbose=False, plots=False)
    print(f"{tag}: mAP50={float(r.box.map50):.4f}  mAP50-95={float(r.box.map):.4f}")
PY
