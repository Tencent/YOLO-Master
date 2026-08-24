# 03 - 复现命令

> 提交日期：2026-08-24
> 锁定版本：main @ e9ac08b
> 对应登记表列：复现命令

> **环境前置**：已完成 `01_environment_install.md` 中的安装步骤；虚拟环境已激活；当前在 `D:\YOLO_MASTER` 目录下。

## 0. 锁定版本校验

```powershell
git rev-parse HEAD
# 应输出: e9ac08b2bd135c379206b8f77df9714b8800cbe0
```

## 1. 拉取预训练权重

```powershell
# HuggingFace 镜像源（中国大陆稳定）
curl.exe -L -o yolo_master_n.pt "https://hf-mirror.com/gatilin/YOLO-Master-ckpts-v0/resolve/main/YOLO-Master-EsMoE-N/YOLO-Master-EsMoE-N.pt"

# 验证 SHA256
sha256sum yolo_master_n.pt
# 预期: 29e1b93f09b16c8cf7c402f36dcaafc19d4812155631ed45b769e941e4c88c32
```

## 2. 拉取 COCO8 数据集（4 train + 4 val）

```powershell
curl.exe -L -o coco8.zip "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
Expand-Archive -Path coco8.zip -DestinationPath C:\Users\86133\datasets -Force
# 验证 data.yaml 存在
Test-Path C:\Users\86133\datasets\coco8\data.yaml
# 预期: True
```

## 3. 导出预检（preflight 链路验证）

```powershell
python -m pytest tests/test_export_preflight.py -v
# 预期: 3 passed
```

## 4. ONNX 导出

```powershell
yolo export model=yolo_master_n.pt format=onnx imgsz=640 device=cpu
# 预期输出: ONNX: export success 19.1s, saved as 'yolo_master_n.onnx' (10.6 MB)

# 验证 SHA256
sha256sum yolo_master_n.onnx
# 预期: 586cf6feeee5bbadddc2cd7cb5852ed49e06f4008e024859904fffc4394c1055
```

## 5. ONNX Runtime 端到端推理

```powershell
yolo predict model=yolo_master_n.onnx source=ultralytics/assets/bus.jpg device=cpu
# 预期: image 1/1 bus.jpg: 640x640 4 persons, 1 bus
```

## 6. PyTorch 精度基线

```powershell
yolo val model=yolo_master_n.pt data=C:\Users\86133\datasets\coco8\data.yaml device=cpu imgsz=640
# 关键指标: mAP50-95 ≈ 0.743
```

## 7. ONNX 精度基线

```powershell
yolo val model=yolo_master_n.onnx data=C:\Users\86133\datasets\coco8\data.yaml device=cpu imgsz=640
# 关键指标: mAP50-95 ≈ 0.711
```

## 8. 一键复现（Python API 汇总）

```python
from ultralytics import YOLO
import json

DATA = r"C:\Users\86133\datasets\coco8\data.yaml"

m_pt = YOLO("yolo_master_n.pt")
r_pt = m_pt.val(data=DATA, device="cpu", imgsz=640, save_json=True, verbose=False, project="smoke/a3/06_result_evidence/_val_pt", name="pt")

m_ort = YOLO("yolo_master_n.onnx")
r_ort = m_ort.val(data=DATA, device="cpu", imgsz=640, save_json=True, verbose=False, project="smoke/a3/06_result_evidence/_val_onnx", name="onnx")

metrics = {
    "pt":   {"mAP50-95": r_pt.box.map,   "mAP50": r_pt.box.map50,   "P": r_pt.box.mp,   "R": r_pt.box.mr,   "inference_ms": r_pt.speed["inference"]},
    "onnx": {"mAP50-95": r_ort.box.map,  "mAP50": r_ort.box.map50,  "P": r_ort.box.mp,  "R": r_ort.box.mr,  "inference_ms": r_ort.speed["inference"]},
}
metrics["delta_mAP50-95"] = metrics["onnx"]["mAP50-95"] - metrics["pt"]["mAP50-95"]
metrics["delta_mAP50"]    = metrics["onnx"]["mAP50"]    - metrics["pt"]["mAP50"]
metrics["speedup_x"]      = metrics["pt"]["inference_ms"] / metrics["onnx"]["inference_ms"]
print(json.dumps(metrics, indent=2))
```

## 9. 验证 immovable tag

```powershell
git tag -l rhino-2026-0824-a3-baseline
git rev-parse rhino-2026-0824-a3-baseline
git cat-file -t rhino-2026-0824-a3-baseline
# 预期: tag (annotated, 不可 force-push 改写)
```

## 10. 验证增量（基线 → 最终）

```powershell
git log --reverse e9ac08b..HEAD            # 新增 commit 集合
git diff --stat e9ac08b..HEAD              # 变更文件统计
git merge-base --is-ancestor e9ac08b HEAD  # 基线可达性
```
