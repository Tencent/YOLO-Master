# A3 P1 云端实验方案：MoT 真实权重自训 + INT8 PTQ

> 日期：2026-08-29
> 目标：① 训出 MoT 真实权重，重跑路由漂移真实数字；② 打通真实 INT8 PTQ，出「体积/速度/精度」三件套
> 平台：魔搭 PAI-DSW A10 24GB（剩约 32h）

---

## 一、总览

| 阶段 | 产出 | 预估耗时（A10） |
|------|------|------|
| 0. 数据准备 | VisDrone YOLO 格式 labels（本机/云端） | 本机做，不耗额度 |
| 1. MoT 自训 | `mot_n_visdrone.pt`（router 收敛） | 10 epoch ≈ 1~1.5h |
| 2. 路由漂移重跑 | 真实 top-k 漂移数字 | 本机 CPU 即可 |
| 3. INT8 PTQ | FP32/INT8 的 ONNX + 三件套 | ≈ 0.5h |

---

## 二、阶段 0：数据准备（本机，不耗云端额度）

VisDrone 原始标注是逗号分隔的 `x,y,w,h,score,cls,...`，需转成 YOLO 格式。
本机 `data/visdrone/VisDrone2019-DET-val` 已解压，train 还是 zip。

```bash
# 1. 解压 train
cd /d/YOLO_Master/YOLO_Master
unzip -q data/visdrone/VisDrone2019-DET-train.zip -d data/visdrone/

# 2. 转换标签（VisDrone → YOLO），生成 images/{split} + labels/{split}
/d/Anaconda3/envs/yolo_env/python.exe - <<'PY'
from pathlib import Path
from PIL import Image

ROOT = Path("data/visdrone")

def visdrone2yolo(split):
    src = ROOT / f"VisDrone2019-DET-{split}"
    img_dir = ROOT / "images" / split
    lab_dir = ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for img in (src / "images").glob("*.jpg"):
        img.rename(img_dir / img.name)  # 若已存在则跳过
    for f in (src / "annotations").glob("*.txt"):
        img = img_dir / f.with_suffix(".jpg").name
        dw, dh = 1.0 / img.size[0], 1.0 / img.size[1]
        lines = []
        for row in (x.split(",") for x in f.read_text().strip().splitlines()):
            if row[4] != "0":  # 跳过 ignored region
                x, y, w, h = map(int, row[:4])
                cls = int(row[5]) - 1
                lines.append(f"{cls} {(x+w/2)*dw:.6f} {(y+h/2)*dh:.6f} {w*dw:.6f} {h*dh:.6f}")
        (lab_dir / f.name).write_text("\n".join(lines))

visdrone2yolo("train")
visdrone2yolo("val")
print("done")
PY

# 3. 打包上传云端（或云端重新下载）。打包用 Python zipfile，别用 GNU tar（漏大文件）
/d/Anaconda3/envs/yolo_env/python.exe -c "import shutil; shutil.make_archive('data/visdrone_yolo', 'zip', 'data/visdrone', 'images', 'labels')"
```

---

## 三、阶段 1：MoT 自训（云端 A10）

### 3.1 建 nc=10 的 MoT 配置

```bash
cp ultralytics/cfg/models/26/yolo26-master-mot-n.yaml \
   ultralytics/cfg/models/26/yolo26-master-mot-n-visdrone.yaml
# 把第 3 行 nc: 80 改成 nc: 10
```

### 3.2 数据 yaml（云端路径）

```yaml
# visdrone.yaml（放云端 /mnt/workspace/）
path: /mnt/workspace/visdrone
train: images/train   # 6471
val: images/val       # 548
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

### 3.3 训练命令

```bash
cd /mnt/workspace/YOLO-Master
yolo train \
  model=ultralytics/cfg/models/26/yolo26-master-mot-n-visdrone.yaml \
  data=/mnt/workspace/visdrone.yaml \
  epochs=10 imgsz=640 batch=16 device=0 \
  project=/mnt/workspace/a3_mot_train name=mot_n_visdrone \
  workers=8 cache=ram
```

**要点**：
- 目标只是让 **router 收敛**（专家分工明确），不是 SOTA 精度，10 epoch 足够。
- MoT 训练会自动算 balance + router z-loss（block.py 内建），router 会学到非均匀分布。
- 训完看 `runs/a3_mot_train/mot_n_visdrone/weights/best.pt` 的 mAP，只要 router 不再退化为均匀即可。

---

## 四、阶段 2：路由漂移真实数字（本机 CPU）

训出的权重拉回本机，复用已写好的脚本：

```bash
/d/Anaconda3/envs/yolo_env/python.exe scripts/a3_mot_drift_analysis.py \
  --model /path/to/mot_n_visdrone.pt \
  --source data/visdrone/VisDrone2019-DET-val/images --limit 548 \
  --out runs/a3_mot_drift_real
```

> 注意：脚本里 `inject_nondegenerate_routers` 只对「零初始化」退化的路由生效，
> 真实权重会自动跳过（非全零）。若脚本无条件注入会污染真实权重，需先判断
> router 是否退化再决定是否注入（后续我给脚本加 `--no-inject` 开关）。

---

## 五、阶段 3：真实 INT8 PTQ（云端，三件套）

### 5.1 导出 ONNX

```bash
yolo export model=/mnt/workspace/a3_mot_train/mot_n_visdrone/weights/best.pt \
  format=onnx imgsz=640 device=0 opset=17
```

### 5.2 INT8 量化（两条路对比）

**路线 A — 仓库 MoE 感知量化（推荐，路由层保 fp16）**：

```python
from ultralytics import YOLO
from ultralytics.nn.modules.moe.quantize import quantize_moe_model
model = YOLO("best.pt").model.eval()
# 动态量化（weight-only），路由节点排除在量化外
out = quantize_moe_model(model, backend="onnx", dynamic_quantize=True,
                         output_path="best_int8.onnx")
```

**路线 B — 标准 onnxruntime 静态量化（含校准）**：

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
# 用 100 张 val 图做校准，nodes_to_exclude 保路由层
```

**关键对比点（A3 核心）**：跑三个版本——`FP32 ONNX`、`全量 INT8`、`路由保 FP16 的 INT8`，
看「路由保护」是否比「全量 INT8」显著提升精度。这直接验证 `quantize.py` 的设计假设。

### 5.3 三件套评测

```bash
# 体积
ls -lh best.pt best.onnx best_int8.onnx

# 精度（FP32 vs INT8，同一 val 集）
yolo val model=best.onnx data=/mnt/workspace/visdrone.yaml imgsz=640 device=0
yolo val model=best_int8.onnx data=/mnt/workspace/visdrone.yaml imgsz=640 device=0

# 速度（inference ms / img，val 输出自带 speed 字段）
```

产出表格：

| 版本 | 体积 | mAP50-95 | 推理时延 | 备注 |
|------|------|----------|----------|------|
| FP32 PT | | | | 基线 |
| FP32 ONNX | | | | 导出无损检查 |
| INT8（全量） | | | | 路由不保护 |
| INT8（路由保 FP16） | | | | quantize.py 方案 |

---

## 六、额度与排期建议

| 项 | 预估 |
|------|------|
| MoT 自训 10 epoch | 1~1.5h |
| INT8 PTQ + 评测 | 0.5h |
| 合计 | ~2h，占剩余 32h 的 6% |

**省额度要点**：数据准备、标签转换、漂移分析全部本机做；云端只跑训练 + 导出 + 量化。
不用就 destroy 实例。

---

## 七、关键风险

1. **MoT-n 训练是否收敛**：若 10 epoch 后 router 仍接近均匀（z-loss/balance 未起效），
   排查 aux loss 系数，或加 epoch。
2. **INT8 精度崩塌**：若全量 INT8 掉点严重，正是 A3 要的「负结果」，如实记录，
   并用「路由保 FP16」版本对照——这本身就是结论。
3. **onnxruntime 版本**：静态量化需要 `onnxruntime>=1.17`，云端确认已装。
