
# YOLO-Master 复现：VisDrone & SKU-110K

本目录包含在 VisDrone 和 SKU-110K 数据集上复现 YOLO-Master v0.1-N 和 EsMoE-N 的训练脚本与结果。

## 数据集下载
- **VisDrone2019-DET**: [官网下载](http://aiskyeye.com/)，解压后按 `ultralytics/cfg/datasets/VisDrone.yaml` 组织目录。
- **SKU-110K**: [GitHub 项目页](https://github.com/eg4000/SKU110K_CVPR19) 下载，解压后按 `ultralytics/cfg/datasets/SKU-110K.yaml` 组织目录。

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

## 训练日志说明
- **在线 Wandb（私有项目，无法公开访问）**
  wandb_visdrone_v01n_800.zip：VisDrone 数据集 YOLO-Master v0.1-N 800 分辨率基线实验离线日志，run 目录：run-51fa0ypd.wandb
  
  wandb_visdrone_esmoeen_800.zip：VisDrone 数据集 YOLO-Master EsMoE-N 800 分辨率实验离线日志
  
  wandb_sku110_v01n_640.zip：SKU-110K 商品检测 YOLO-Master v0.1-N 640 分辨率基线离线日志，run 目录：run-s27cwcfb.wandb
  
  wandb_sku110_esmoeen_640.zip：SKU-110K 商品检测 YOLO-Master EsMoE-N 640 分辨率离线日志，run 目录：run-f5ysopdn.wandb
- **离线日志**
  已将 Wandb 离线包打包为 zip 文件放在本目录下（`wandb_visdrone_v01n_800.zip` 等），下载后执行以下命令即可本地查看所有训练曲线：
  ```bash
  wandb sync ./run-xxx.wandb
  ```
- 也可直接从 `runs/` 文件夹中的 `results.csv` 读取每 epoch 详细指标。

## 已知问题
- WandB 为私有团队空间，无法直接提供可公开访问的链接，已用离线包替代。
- Windows 训练时必须设置 `workers=0` 以避免多进程死锁。
- 若显存不足，请减小脚本中的 `BATCH` 变量（例如 VisDrone 改为 4，SKU-110K 改为 8）。
- VisDrone 小目标密集，imgsz=800 为推荐基础设置，不宜进一步降低。

## 解决方案
- 如需提升小目标精度，可尝试增大 `imgsz` 至 1280，同时相应降低 batch size。
- 确保数据集路径与对应 yaml 文件完全一致。
- 若 Wandb 网络不稳定，可提前设置 `WANDB_MODE=offline` 先离线训练，再手动同步。
```

