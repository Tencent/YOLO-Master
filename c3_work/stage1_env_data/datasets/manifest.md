# datasets/manifest.md · 数据来源 / 许可 / 统计 / 可复现性

> 由 `download_datasets.sh` + `prepare_neu_det.py` + `prepare_deeppcb.py` 生成（2026-09-07）。
> 大文件不入 git；本 manifest + `split_report.json`（各数据集 out 目录内）为结构化留档。

## 1. NEU-DET（钢表面 6 类缺陷）

| 项 | 值 |
|---|---|
| 下载源 | GitHub `Marfbin/NEU-DET-with-yolov8`（`git clone --depth 1`，经代理 `http://10.198.7.60:7890`） |
| 数据本质 | 官方 NEU Surface Defect Database 的 YOLOv8 格式镜像；1800 张 200×200 灰度 jpg，6 类 |
| 原始划分 | train 1620 / test 180（本仓库自带，test 含标注，作 val） |
| 许可 | **无显式 LICENSE**（镜像仓库与官方数据库页面均未附许可）→ 学术/评测用途，提交物中如实声明许可不确定性 |
| 类别 | crazing / inclusion / patches / pitted_surface / rolled-in_scale / scratches |
| 本机产物 | `/mnt/pfs/zitao_team/baiyouheng/datasets/NEU-DET-yolo/neu_det_yolo_v3/`（images/labels/train.txt/val.txt/shots/k{5,10,50,100}/neu_det.yaml/split_report.json） |
| few-shot 采样 | 每类 5/10/50/100 张分层采样：k5=30, k10=60, k50=300, k100=600（train），val=180 固定 |
| SHA-256 | `split_report.json` → `sample_images_sha256`（每集 5 张样本） |
| 可复现说明 | 产物为磁盘固化（train.txt/val.txt/shots 均已生成），**实验一律引用固化文件**。注意：脚本用 `set` 去重后迭代，跨进程 set 迭代顺序受 PYTHONHASHSEED 影响，重跑可能产生不同子集；因此不以重跑为准，以首次固化产物为准 |

## 2. DeepPCB（印刷电路板 6 类缺陷）

| 项 | 值 |
|---|---|
| 下载源 | GitHub `tangsanli5201/DeepPCB`（`git clone --depth 1`，经代理） |
| 数据本质 | PCB 缺陷测试图 1500 张（含标注），+1500 模板图（不参与训练）；本镜像为 **640×640 灰度 jpg**，标注 `x1 y1 x2 y2 cls`，cls=1..6（像素坐标，与 640 图幅匹配，已抽样验证 max_coord<640） |
| 标注类别 | 1..6 → open/short/mousebite/spur/pinhole/spurious_copper → YOLO cls 0..5 |
| 许可 | **MIT**（仓库 LICENSE）✅ |
| 本机产物 | `/mnt/pfs/zitao_team/baiyouheng/datasets/DeepPCB-yolo/deeppcb_v1/`（images/labels/train.txt/val.txt/shots/k{5,10,50,100}/deeppcb.yaml/split_report.json） |
| 划分 | seed=824 随机：train 1350 / val 150 |
| few-shot 采样 | **类别极不平衡**，按"每类至多 k 张"union 去重后：k5=8, k10=13, k50=70, k100=142（open/short 等稀有类严重不足）→ 小样本曲线主线建议用 NEU-DET；DeepPCB 主实验走全量 1350 |
| SHA-256 | `split_report.json` → `sample_images_sha256` |
| 可复现说明 | 同上：以固化产物为准 |

## 3. 环境（另见 `env_setup.sh` / `env_setup.log`）

- conda env：`/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master`（python 3.11，用户工作区，不污染团队共享 env）
- 依赖：`pip install torch==2.6.0 torchvision==0.21.0` + `requirements.txt` + `pip install -e <repo>`
- 验证：`yolo version` 输出 8.4.x；`torch.cuda.is_available()==True`
