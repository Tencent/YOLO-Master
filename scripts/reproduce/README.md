# VisDrone 复现报告（YOLO-Master-v0.1-N / EsMoE-N）

在 VisDrone（航拍密集小目标）上复现两个 nano 模型：

| 模型 | 配置 |
| --- | --- |
| YOLO-Master-v0.1-N | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` |
| YOLO-Master-EsMoE-N | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` |

数据配置：`ultralytics/cfg/datasets/VisDrone.yaml`  
训练设置：`imgsz=640`，`epochs=100`（按 GPU 调整 batch）

**W&B 公开项目：** https://wandb.ai/benifiveyouga-zhejiang-university/yolo-master-reproduce

| Run | URL |
| --- | --- |
| VisDrone_v0.1-N | https://wandb.ai/benifiveyouga-zhejiang-university/yolo-master-reproduce/runs/6pneidnl |
| VisDrone_EsMoE-N | https://wandb.ai/benifiveyouga-zhejiang-university/yolo-master-reproduce/runs/nw91ftxm |

---

## 1. 环境

| 项目 | 本机 |
| --- | --- |
| OS | Windows 10 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop，12 GB |
| Python | 3.11（conda: `yolo_master`） |
| PyTorch | `2.11.0+cu128` |

```powershell
conda create -n yolo_master python=3.11 -y
conda activate yolo_master
pip install -r requirements.txt
pip install -e .
pip install wandb
pip install torch torchvision torchaudio --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128
wandb login
yolo settings wandb=False
```

---

## 2. 数据集下载命令

### 自动下载

```powershell
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"
```

### 国内备用（推荐）

1. 下载 zip 到 `C:\workspace\datasets\downloads\VisDrone\`：
   - https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip
   - https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip
   - https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip（可选）
2. 转换：

```powershell
python scripts/download_visdrone.py `
  --root C:\workspace\datasets\VisDrone `
  --download-dir C:\workspace\datasets\downloads\VisDrone `
  --yaml-out C:\workspace\datasets\VisDrone.yaml `
  --skip-download
```

就绪后应有：`datasets/VisDrone/images/train`、`datasets/VisDrone/images/val`。

---

## 3. 训练命令

```powershell
# 构建检查
python scripts/reproduce/reproduce_visdrone.py --check-build

# v0.1-N（12GB 推荐 batch=8）
python scripts/reproduce/reproduce_visdrone.py `
  --model v0.1-N --epochs 100 --batch 8 --workers 0 `
  --wandb-project yolo-master-reproduce

# EsMoE-N（必须加 --no-sparse-eval；本机 batch=16 可跑通）
python scripts/reproduce/reproduce_visdrone.py `
  --model EsMoE-N --epochs 100 --batch 16 --workers 0 --no-sparse-eval `
  --wandb-project yolo-master-reproduce

# 一次训练两个基线（EsMoE 同样需要 --no-sparse-eval）
python scripts/reproduce/reproduce_visdrone.py `
  --epochs 100 --batch 8 --workers 0 --no-sparse-eval `
  --wandb-project yolo-master-reproduce
```

输出目录：

```text
runs/reproduce/visdrone/VisDrone_v0.1-N/
runs/reproduce/visdrone/VisDrone_EsMoE-N/
```

每 epoch 写入 `results.csv`（含 mAP50、mAP50-95、box/cls/dfl/mixture_aux_loss），并同步到 W&B。

---

## 4. 预期结果

### 4.1 论文 / 官方参考

| 来源 | Model | mAP50 | mAP50-95 |
| --- | --- | --- | --- |
| 论文 YOLO-Master-N | — | 0.337 | 0.196 |
| 官方 reproduce 权重 | v0.1-N | 0.3443 | 0.2009 |
| 官方 reproduce 权重 | EsMoE-N | 0.3499 | 0.2029 |

官方多为约 300 epoch、更大 batch；本机 100 epoch 指标略低属预期。

### 4.2 本机复现结果

| Model | epochs | batch | imgsz | mAP50 | mAP50-95 | 耗时 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0.1-N | 100 | 8 | 640 | **0.3083** | **0.1737** | 8.16 h | best.pt / W&B |
| EsMoE-N | 100 | 16 | 640 | **0.3155** | **0.1775** | 8.74 h | `--no-sparse-eval` |

`results.csv` 按 epoch 峰值：v0.1-N 约 epoch 87（mAP50=0.3098）；EsMoE-N 约 epoch 80（mAP50=0.3155）。

---

## 5. 结果对比

| Dataset | Model | 本机 mAP50 | 官方 mAP50 | Δ | 本机 mAP50-95 | 官方 mAP50-95 | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VisDrone | v0.1-N | 0.3083 | 0.3443 | -0.036 | 0.1737 | 0.2009 | -0.027 |
| VisDrone | EsMoE-N | 0.3155 | 0.3499 | -0.034 | 0.1775 | 0.2029 | -0.025 |

| 本机两模型 | mAP50 | mAP50-95 |
| --- | --- | --- |
| EsMoE-N − v0.1-N | **+0.0072** | **+0.0038** |

差距主因：epochs=100（参考约 300）、batch 较小。训练曲线正常：loss 下降、mAP 上升。

---

## 6. 运行日志摘要

完整逐 epoch 日志位于训练输出目录中的 `results.csv`；W&B 上可查看全程曲线（链接见文首）。

### 末 epoch（epoch=100）

| Model | box_loss | cls_loss | mixture_aux_loss | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| v0.1-N | 1.390 | 0.942 | 1.000 | 0.306 | 0.171 |
| EsMoE-N | 1.384 | 0.940 | 1.000 | 0.315 | 0.177 |

### 过程摘录（v0.1-N）

| Epoch | mAP50 | mAP50-95 | box_loss | cls_loss |
| --- | --- | --- | --- | --- |
| 1 | ~0.000 | ~0.000 | 5.511 | 5.943 |
| 4 | 0.072 | 0.033 | 2.359 | 2.049 |
| 50 | （见 results.csv） | | | |
| 100 | 0.306 | 0.171 | 1.390 | 0.942 |
| best.pt | **0.308** | **0.174** | — | — |

### 本地产物路径

```text
runs/reproduce/visdrone/VisDrone_v0.1-N/results.csv
runs/reproduce/visdrone/VisDrone_v0.1-N/weights/best.pt
runs/reproduce/visdrone/VisDrone_EsMoE-N/results.csv
runs/reproduce/visdrone/VisDrone_EsMoE-N/weights/best.pt
```

---

## 7. 已知问题与解决方案

### 7.1 CUDA OOM

- **现象**：默认 `batch=64`（或自动降到 32）OOM。  
- **原因**：VisDrone 单图实例极多，Assigner 显存占用高。  
- **解决**：v0.1-N 用 `--batch 8`；EsMoE-N 可用 `--batch 16`；仍不够则 `8/4` + `--workers 0`。

### 7.2 EsMoE-N 验证 mAP ≈ 0

- **现象**：不加修正时验证崩溃。  
- **原因**：ES_MOE 默认稀疏推理与训练 dense 不一致。  
- **解决**：加 `--no-sparse-eval`。

### 7.3 数据集下载失败

- **现象**：`ultralytics.com/assets/VisDrone*.zip` 连接中断。  
- **解决**：ghproxy / GitHub 手动下载 + `scripts/download_visdrone.py --skip-download`。

### 7.4 W&B Invalid project name（Windows）

- **现象**：`yolo settings wandb=True` 后，project 被设为本地路径报错。  
- **解决**：`yolo settings wandb=False`；使用命令行 `--wandb-project yolo-master-reproduce`（本脚本也会自动关闭 Ultralytics 内置 wandb）。

### 7.5 Team 无法 Public

- **现象**：Visibility 只有 Team。  
- **解决**：Team Settings → Privacy 允许公开后，再将项目设为 Public。
