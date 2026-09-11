# C1 VisDrone Smoke Test

8.24 准入最小冒烟测试：在官方稳定 commit `43d4011`（YOLO-Master-v26.08）上跑通 ES-MoE-N 5-epoch 训练。

## 仓库基线

- 官方 tag：`YOLO-Master-v26.08`
- Commit：`43d40117c30811204fb9347efeabddce15f11a62`
- 本地已 checkout 到该 commit：`D:/YOLO-Master`

## 运行环境

- OS：Windows 11
- Python：3.11（MS Store `python3.11.exe`）
- torch：2.3.1+cu118（安装于 `D:/torch_cuda_pkg`）
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（8GB 显存）
- 数据：VisDrone2019-DET 已就绪于 `D:/datasets/VisDrone`

## 复现命令

```bat
cd /d D:/YOLO-Master
set PYTHONPATH=D:/torch_cuda_pkg
python D:/YOLO-Master/smoke/c1/smoke_c1_esmoe_n.py
```

> 在 git bash 中请使用完整路径 `C:/Users/Administrator/AppData/Local/Microsoft/WindowsApps/python3.11.exe`。

## 关键超参

| 参数 | 值 | 说明 |
|---|---|---|
| model | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` | ES-MoE-N |
| data | `D:/datasets/VisDrone/VisDrone_local.yaml` | 本地 VisDrone，绝对路径免下载 |
| epochs | 5 | 最小 smoke |
| imgsz | 640 | 默认分辨率 |
| batch | 8 | RTX 4060 8GB 适配 |
| device | 0 | RTX 4060 |
| workers | 0 | Windows 避免多进程死锁 |
| seed | 42 | 固定随机 |
| pretrained | False | 离线：不下载预训练权重 |
| amp | False | 离线：跳过 AMP 参考权重下载 |

## 实际运行结果

- 完成时间：2026-08-24 00:09
- 日志：`D:/smoke/smoke_c1_esmoe_n_43d4011.log`
- 产物：`D:/YOLO-Master/runs/smoke_43d4011/smoke_c1_esmoe_n/`
- epoch 5 指标：precision=0.201, recall=0.110, **mAP50=0.0536, mAP50-95=0.0239**
- 全程无联网下载记录，可离线复现。

> 5 epoch 从零训练 mAP 偏低为预期，仅用于链路冒烟验证，非最终质量门槛。
