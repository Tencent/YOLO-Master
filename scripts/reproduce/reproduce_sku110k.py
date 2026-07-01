from ultralytics import YOLO

model = YOLO("yolo_master_n.pt")
model.train(data="coco.yaml", epoch=300, batch=256, imgsz=640)
model.val(data="coco.yaml")
results = model("image.jpg")

model.export(format='engine', half=True)
