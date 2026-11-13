# S1 底座差异分析：YOLO-Master × 开放词汇（OV）现状盘点

> 日期：2026-11-08 ｜ 关联：[ADR-0007](adr/0007-yolo26-open-vocabulary-evolution.md) ｜ 状态：已验证

## 结论（TL;DR）

**S1 的工作量远小于预期：YOLO26 底座迁移在仓库里已基本完成**，`yolo26-master-n.yaml` 把
ES-MoE（A2C2fMoE）架在了 YOLO26 骨干上，`yoloe-26.yaml` 提供了 YOLOEDetect + RepRTA
开放词汇头。本轮新增的 `yoloe-26-master-n.yaml` 组合配置已完成 **前向冒烟 + 1-epoch
真实训练冒烟（含 grounding 数据、文本嵌入缓存、MoE 路由优化器分组）**，S1 可视为收尾，
重心直接转入 S2（OV 预训练）的工程放大。

## 一、现状盘点：三个"已就位"

| 组件 | 位置 | 状态 |
|------|------|------|
| YOLO26 底座 + ES-MoE | `ultralytics/cfg/models/26/yolo26-master-n.yaml`（A2C2fMoE @ P3/P4/P5，layers 4/6/8） | ✅ 已存在 |
| 开放词汇 head（RepRTA） | `ultralytics/nn/modules/head.py:981 YOLOEDetect`，配置 `yoloe-26.yaml`（`text_model: mobileclip2:b`） | ✅ 已存在 |
| Grounding 训练链路 | `GroundingDataset`（GoldG 风格 JSON）+ `YOLOETrainerFromScratch` + `RandomLoadText` 负样本增强 | ✅ 已存在 |

**本轮补齐的三块拼图**：

1. **组合配置** `ultralytics/cfg/models/26/yoloe-26-master-n.yaml`：A2C2fMoE backbone
   （experts 4/8/16，top_k=2）+ PAN + `YOLOEDetect [nc, 512, True]`。7.60M 参数（n 档）。
2. **数据格式对接**：`scripts/ov_distill/convert.py --format grounding` 新增 GoldG 风格
   单 JSON 输出（caption + tokens_positive 字符区间），被 `GroundingDataset` 直接消费
   （118 图 / 395 框 / 66 类，实载验证通过）。
3. **训练冒烟** `scripts/ov_distill/smoke_train_yoloe_master.py`：`YOLOETrainerFromScratch`
   × 组合配置 × coco128 伪标签 grounding 数据，1 epoch / 30 steps 全链路通过，
   loss（box/cls/dfl）与 `mixture_aux_loss` 正常，优化器正确识别 36 个 router 参数组
   （lr=0.5x 分组生效）。

## 二、数据格式约定（已验证的事实）

`GroundingDataset`（`ultralytics/data/dataset.py:770`）期望的 JSON schema：

```json
{
  "images": [{"id": 0, "file_name": "xx.jpg", "height": 480, "width": 640, "caption": "cat. dog."}],
  "annotations": [{"image_id": 0, "bbox": [x, y, w, h], "iscrowd": 0, "tokens_positive": [[0, 3]]}]
}
```

- `bbox` 为**绝对像素 xywh（左上角原点）**；`tokens_positive` 是类别名在 `caption` 里的
  **Python 切片区间（end 不含）**，类别名 = `caption[start:end]` 小写化。
- **负样本不进标注文件**：训练端 `RandomLoadText`（`augment.py:2655`）从图内类别表 +
  高频 padding 词表自动采样负样本，`max_samples=min(nc, 80)`。
- 训练入口的数据 dict 形态：

```python
data = {
    "train": {"yolo_data": [], "grounding_data": [{"img_path": ..., "json_file": ...}]},
    "val": {"yolo_data": ["coco128.yaml"]},  # nc/names 取自 val 集
}
```

## 三、遗留缺口与改动点清单

| # | 缺口 | 影响 | 建议 |
|---|------|------|------|
| G1 | `mobileclip2_b.ts` 不在 ultralytics 官方资产清单，且本机到 github.com 不通；`mobileclip_blt.ts` 虽在清单内但同样下载失败 | S2 真实预训练缺文本编码器 | ~~内网/镜像渠道获取~~ **已解决（2026-11-08）**：经 ghfast.top 代理从 ultralytics/assets v8.4.0 release 下载（253MB），放 `weights/` 即被 `attempt_download_asset` 命中；另需 `pip install clip-anytorch "setuptools<81"` 提供 clip 分词器；512 维输出已实测 |
| G2 | `text_model.py` 顶层 `import clip` 缺失时触发 github 在线安装（本机会卡死） | 离线环境首次跑 YOLOE 训练会挂 | 离线环境预装 `clip`，或按冒烟脚本方式预注入 stub |
| G3 | RepRTA 重参数化（推理零开销）只在 `YOLOEDetect.fuse()` 后成立；MoE 侧 ES-MoE 的推理合并与 RepRTA 的交互未验证 | S2 导出部署 | S2 末段补 `model.fuse()` + 导出回归 |
| G4 | 组合配置仅 n 档；s/m/l/x 档 scale 未定义 | 规模化实验 | 参照 `yolo26-master` 系列补齐 scales |
| G5 | 拒识（abstention）分支与 prompt 条件化路由（ADR-0007 §架构）未动工 | S3/S4 差异化核心 | 按里程碑 M3/M4 推进 |

## 四、S1 → S2 的衔接建议

S1 复现项（闭集 COCO 上 yolo26-master-n 基线数字）不阻塞 S2 启动——闭集基线与 OV 链路
解耦，可并行。S2 的最小可行路径已经打通：

```
无标注图 → concepts → annotate(Rex-Omni + QwenVL) → clean(投票) → convert --format grounding
        → YOLOETrainerFromScratch(yoloe-26-master-n) → RepRTA 开放词汇检测器
```

下一步的工程瓶颈不在代码而在**资源**：文本编码器权重获取（G1）与 OV 预训练所需的
数据池规模（建议先 10k~50k 图验证伪标签 scaling law）。
