# stage1_env_data · 环境与数据集就绪

**阶段目标**：在共享 8×A800 服务器上重建 YOLO-Master 运行环境，下载并准备 NEU-DET / DeepPCB 双数据集（含 k-few-shot 划分），锁定许可与 SHA-256，形成后续所有实验的可复现底座。

**执行时间**：2026-09-07

**产出（本目录，随 fork push）**

| 文件 | 用途 |
|---|---|
| `env_setup.sh` + `env_setup.log` | 独立 conda env(py3.11) + editable 安装 YOLO-Master + torch CUDA 验证 |
| `download_datasets.sh` + `download.log` | 走代理下载 NEU-DET(GitHub 镜像) / DeepPCB(GitHub 镜像) |
| `prepare_neu_det.py` + `prepare_neu.log` | NEU-DET → YOLO 布局 + few-shot(k5/10/50/100) 划分 |
| `prepare_deeppcb.py` + `prepare_deeppcb.log` | DeepPCB → YOLO 布局 + few-shot 划分 |
| `datasets/manifest.md` | 数据来源/许可/统计/SHA-256 留档（**权威说明，见它**） |

## 关键结论速览

1. **环境**：conda env = `/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master`（python 3.11，用户工作区内，不污染团队共享 env）。安装链路 `env_setup.sh`：conda create → `pip install torch==2.6.0 torchvision==0.21.0`（内网 pypi 镜像）→ `requirements.txt` → `pip install -e <repo>`。状态见检查单。
2. **NEU-DET**（✅ 完成）：1800 张灰度图 / 6 类；train 1620 / val 180；few-shot k5=30, k10=60, k50=300, k100=600（每类 5/10/50/100）；产物 `/mnt/pfs/zitao_team/baiyouheng/datasets/NEU-DET-yolo/neu_det_yolo_v3/`。许可：镜像**无显式 LICENSE**（学术用途，如实声明）。
3. **DeepPCB**（✅ 完成）：1500 test 图（本镜像 640×640 灰度 jpg）/ 6 类；train 1350 / val 150；标注 `x1 y1 x2 y2 cls(1..6)` 已抽验匹配图幅。few-shot **类别极不平衡**：k5=8, k10=13, k50=70, k100=142（稀有类 open/short 不足）→ 小样本曲线主线用 NEU-DET，DeepPCB 主实验走全量 1350。许可：**MIT**。产物 `/mnt/pfs/zitao_team/baiyouheng/datasets/DeepPCB-yolo/deeppcb_v1/`。

## 收尾检查单（Stage1）

- [x] NEU-DET 下载（1800 张 train 1620/test 180）
- [x] DeepPCB 下载（1500 test 图，MIT）
- [x] NEU-DET 转换 + few-shot 划分完成，`split_report.json` 就绪
- [x] DeepPCB 转换 + few-shot 划分完成（类不均衡已记录）
- [x] `datasets/manifest.md` 留档（来源/许可/统计/可复现说明）
- [x] conda env(python3.11) 创建完成，`yolo version` 输出 8.4.101（`env_setup.log` 末尾 `ENV_SETUP_DONE`）
- [x] `torch.cuda.is_available()=True`，可见 8×A800（`torch 2.6.0+cu124 | cuda_avail True | ndev 8`）
- [x] git commit 到 `c3-vpeft-smoke`（阶段 1 提交已入库）
