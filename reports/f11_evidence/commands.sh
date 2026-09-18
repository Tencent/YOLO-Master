# 冒烟 0：基线 smoke
python -c "from ultralytics import YOLO; m=YOLO('ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml'); m.train(data='coco8.yaml', epochs=1, imgsz=640, batch=8, device=0, project='runs/smoke')"

# 冒烟 1：教师特征缓存
python scripts/cache_teacher_features.py --data coco8.yaml --num 100 --out runs/f11_teacher_cache

# 冒烟 2：q_teacher 非退化
python scripts/gen_q_teacher.py --cache runs/f11_teacher_cache --model yolo-master-n.yaml --device cpu

# 冒烟 3：Router KD 闭环
python scripts/smoke_router_kd.py --epochs 1 --layers 1 --experts 2 --device 0

# 冒烟 4：证据归档（本脚本）
python scripts/archive_evidence.py --output reports/f11_evidence/