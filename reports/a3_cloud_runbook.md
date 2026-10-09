# A3 P1 云端训练操作手册（魔搭 PAI-DSW A10）

> 目标：打开云端后照本执行，训出 MoT 真实权重 + 完成 INT8 PTQ。
> 本机已备好：数据 zip（`data/visdrone/visdrone_yolo.zip`）、nc=10 配置、全套脚本。

---

## 第 0 步：本机 push 代码（可选但推荐）

把 P1 新增文件推到 fork，云端 clone 即可拿到。当前在 `rhino-a3-dev/smoke/a3` 分支：

```bash
cd /d/YOLO_Master/YOLO_Master
git add scripts/a3_router_drift_analysis.py scripts/a3_router_margin_analysis.py \
        scripts/a3_router_bitwidth_sweep.py scripts/a3_mot_drift_analysis.py \
        scripts/convert_visdrone_to_yolo.py \
        ultralytics/cfg/models/26/yolo26-master-mot-n-visdrone.yaml \
        reports/a3_router_drift_20260829.md reports/a3_mot_drift_20260829.md reports/a3_p1_cloud_plan.md
git commit -m "A3 P1: 路由漂移分析脚本 + VisDrone MoT 配置"
git push origin rhino-a3-dev/smoke/a3
```

> 若不想 push，也可手动把这 8 个文件上传到云端对应目录。

---

## 第 1 步：上传数据

本机已打包 `data/visdrone/visdrone_yolo.zip`（约 1.4G，含 images + labels + data.yaml）。

上传方式（任选）：
- 魔搭实例「上传文件」组件（网页拖拽，1.4G 可能需要几分钟）
- 或 OSS / scp / 网盘中转

---

## 第 2 步：云端获取代码 + 装环境

```bash
# 1. clone fork（走你熟悉的镜像/代理）
git clone https://github.com/KennnMai/YOLO-Master.git
cd YOLO-Master
git checkout rhino-a3-dev/smoke/a3   # 含 P1 新脚本的分支

# 2. 装依赖（魔搭 A10 镜像通常预装 torch+CUDA）
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnx onnxruntime onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 解压数据
cd /mnt/workspace
unzip -q visdrone_yolo.zip -d visdrone    # 得到 visdrone/{images,labels,data.yaml}
```

---

## 第 3 步：训练 MoT（目标：router 收敛）

```bash
cd /mnt/workspace/YOLO-Master
yolo train \
  model=ultralytics/cfg/models/26/yolo26-master-mot-n-visdrone.yaml \
  data=/mnt/workspace/visdrone/data.yaml \
  epochs=10 imgsz=640 batch=16 device=0 \
  project=/mnt/workspace/a3_mot_train name=mot_n_visdrone \
  workers=8 cache=ram
```

**预期**：A10 上约 1~1.5h 跑完 10 epoch。产出
`/mnt/workspace/a3_mot_train/mot_n_visdrone/weights/best.pt`。

**验收**：看训练日志里 router 相关的 aux loss（balance/z-loss）在下降、最终 mAP 非零即可；
不需要 SOTA，只要 router 不再退化为均匀分布。

---

## 第 4 步：导出 ONNX + INT8 PTQ

```bash
# 导出
yolo export model=/mnt/workspace/a3_mot_train/mot_n_visdrone/weights/best.pt \
  format=onnx imgsz=640 device=0 opset=17

# INT8 量化（MoE 感知：路由层自动保 fp16）
python - <<'PY'
from ultralytics import YOLO
from ultralytics.nn.modules.moe.quantize import quantize_moe_model
model = YOLO("/mnt/workspace/a3_mot_train/mot_n_visdrone/weights/best.pt").model.eval()
quantize_moe_model(model, backend="onnx", dynamic_quantize=True,
                   output_path="/mnt/workspace/best_int8.onnx")
PY
```

---

## 第 5 步：三件套评测（体积/速度/精度）

```bash
ls -lh best.pt best.onnx best_int8.onnx        # 体积
yolo val model=best.onnx      data=/mnt/workspace/visdrone/data.yaml imgsz=640 device=0   # FP32 ONNX 精度
yolo val model=best_int8.onnx data=/mnt/workspace/visdrone/data.yaml imgsz=640 device=0   # INT8 精度
# 速度看 val 输出末尾的 Speed 字段（preprocess/inference/postprocess ms）
```

> 进阶（A3 核心对照）：再做一份「全量 INT8（不保路由）」对比，验证路由保护是否显著提精度。

---

## 第 6 步：拉回权重，本机跑真实漂移

```bash
# 下载 best.pt 到本机，然后：
cd /d/YOLO_Master/YOLO_Master
/d/Anaconda3/envs/yolo_env/python.exe scripts/a3_mot_drift_analysis.py \
  --model /path/to/best.pt \
  --source data/visdrone/VisDrone2019-DET-val/images --limit 548 \
  --out runs/a3_mot_drift_real
# 脚本会自动检测真实权重（非零）跳过随机注入，直接测真实 top-k 漂移
```

---

## 额度提醒

全程云端约 2h（训练 1.5h + 导出/量化/评测 0.5h），占剩余 32h 的 6%。
**不用就 destroy 实例**，否则空跑也烧额度。
