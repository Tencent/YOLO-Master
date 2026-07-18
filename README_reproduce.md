

# YOLO-Master 复现：VisDrone & SKU-110K

本目录包含在 VisDrone 和 SKU-110K 数据集上复现 YOLO-Master v0.1-N 和 EsMoE-N 的训练脚本与结果。

## 数据集下载
- *VisDrone2019-DET: [官网下载](http://aiskyeye.com/)，解压后按 `ultralytics/cfg/datasets/VisDrone.yaml` 组织目录。
- *SKU-110K: [GitHub 项目页](https://github.com/eg4000/SKU110K_CVPR19) 下载，解压后按 `ultralytics/cfg/datasets/SKU-110K.yaml` 组织目录。

## 训练命令
### VisDrone
```bash
# 训练 v0.1-N 基线
python scripts/reproduce/reproduce_visdrone.py --model v01

# 训练 EsMoE-N
python scripts/reproduce/reproduce_visdrone.py --model moe
```

### SKU-110K
```bash
# 训练 v0.1-N 基线
python scripts/reproduce/reproduce_sku110k.py --model v01

# 训练 EsMoE-N
python scripts/reproduce/reproduce_sku110k.py --model moe
```

## 复现结果对比

| 数据集 | 模型 | 输入分辨率 | 训练轮数 | mAP50 | mAP50-95 | 参数量 |
|--------|------|------------|----------|-------|----------|--------|
| VisDrone | YOLO-Master v0.1-N | 800 | 120 | 0.360 | 0.213 | 7.5M |
| VisDrone | YOLO-Master EsMoE-N | 800 | 120 | 0.360 | 0.212 | 3.4M |
| SKU-110K | YOLO-Master v0.1-N | 640 | 120 | 0.885 | 0.564 | 7.5M |
| SKU-110K | YOLO-Master EsMoE-N | 640 | 120 | 0.886 | 0.563 | 3.4M |

## 训练日志查看
### 离线日志下载（百度网盘）
由于 WandB 项目设置为私有，无法直接公开访问，我们将所有离线运行包上传至百度网盘：
- **下载链接**: [https://pan.baidu.com/s/1CG9I9rxe-Z2Pmhla06fnbg?pwd=tiw2](https://pan.baidu.com/s/1CG9I9rxe-Z2Pmhla06fnbg?pwd=tiw2)
  **提取码**: `tiw2`
- 文件夹内包含以下离线日志：
  - `wandb_visdrone_v01n_800.zip`
  - `wandb_visdrone_moe_800.zip`
  - `wandb_sku110k_v01n_640.zip`
  - `wandb_sku110k_moe_640.zip`

### 使用方法
1. 下载所需 zip 文件并解压。
2. 在命令行执行（以 VisDrone v0.1-N 为例，文件夹名请以实际解压结果为准）：
   ```bash
   wandb sync ./run-xxx.wandb
   ```
3. 浏览器会自动打开 WandB 页面，显示完整的损失曲线、mAP 曲线等所有指标。

## 已知问题
- WandB 为私有团队空间，无法提供可公开访问的在线链接，请使用离线包。
- Windows 训练时必须设置 `workers=0` 以避免多进程死锁。
- 若显存不足（<8GB），请减小脚本中的 `BATCH` 变量（例如 VisDrone 可改为 4，SKU-110K 改为 8）。
- VisDrone 小物体密集，`imgsz=800` 为推荐最低值，不宜进一步降低。

## 本地复现环境
- GPU: NVIDIA RTX 5060 Laptop (8GB)
- CUDA Driver: 13.1
- PyTorch: 2.11.0 (CUDA 12.8)
- 依赖安装命令：
  ```bash
  pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  pip install ultralytics wandb
  ```
```


