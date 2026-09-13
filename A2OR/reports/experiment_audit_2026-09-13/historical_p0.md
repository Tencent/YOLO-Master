> 来源：git HEAD (`cea362b`) 中的 `A2/runs/p0_y26n_vd640_s42_50e/P0_FINAL_REPORT.md`。仅复制历史文档，未恢复原实验目录。

# A2 P0 实验报告：YOLO26n + 当前 STAL-style baseline

## 结论

当前带内嵌 STAL-style 分配的 YOLO26n 即本项目 baseline；P0 不要求再证明其优于传统 TAL。
50 epoch 训练、逐 epoch checkpoint、逐 epoch 正样本统计和小/中/大目标分档评测均已完成。

## AP 单位

代码内部以 0–1 保存 AP，报告中乘以 100 写成 AP points。`APs +1.0` 等价于内部数值 `+0.01`，
例如从 7.31 AP 提升到 8.31 AP。

## 面积分档结果

面积使用原图像素：small `<32²`，medium `32²–96²`，large `≥96²`。

| 协议 | AP | AP50 | AP75 | APs | APm | APl | ARs | ARm | ARl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO maxDets=100 | 15.19 | 27.26 | 14.69 | 7.31 | 23.23 | 41.98 | 18.56 | 41.38 | 60.87 |
| Dense maxDets=300 | 15.25 | 27.40 | 14.75 | 7.36 | 23.31 | 42.03 | 18.81 | 41.64 | 61.02 |

标准口径最佳总体 AP 出现在 epoch 50，为 15.19；最佳 APs 出现在 epoch 47，为 7.36。
P1 必须按预先固定的 checkpoint 选择规则比较，不能事后只挑 APs 最高的 epoch。

## Epoch 50 正样本统计

| 分支/档位 | positives per GT | zero-GT rate |
|---|---:|---:|
| o2m/small | 3.7259 | 8.88% |
| o2m/medium | 9.9389 | 0.02% |
| o2m/large | 9.9760 | 0.10% |
| o2o/small | 0.9221 | 7.79% |
| o2o/medium | 0.9999 | 0.01% |
| o2o/large | 1.0000 | 0.00% |

小目标分配明显弱于中、大目标。注意 epoch 41 起关闭 Mosaic，原始 GT 数量和目标尺度分布发生改变，
因此原始计数不能跨 epoch 40/41 直接比较，应优先查看 positives-per-GT 和 zero-GT rate。

## 总体训练结果

- Epoch 50 Precision：0.4110
- Epoch 50 Recall：0.3005
- Epoch 50 Ultralytics mAP50：0.2703
- Epoch 50 Ultralytics mAP50-95：0.1496

## P0 验收状态

- [x] VisDrone2019-DET baseline 完成 50 epoch。
- [x] 保存 50 个 epoch checkpoint、best.pt 和 last.pt。
- [x] 每 epoch 记录 O2M/O2O 小、中、大目标正样本和零分配统计。
- [x] 输出标准 COCO 和密集场景补充口径的 APs/APm/APl、ARs/ARm/ARl。
- [x] 输出逐 checkpoint 面积分档演化记录和曲线。
- [x] 固定后续 P1/P2 评测口径。

## 限制

训练统计面积在 640 训练输入及增强后计算；正式 AP/AR 面积分档在原图像素计算，两者用途不同，不能逐项等同。
本次只有 seed=42，P1 的提升仍需多 seed 或置信区间证明不是噪声。
