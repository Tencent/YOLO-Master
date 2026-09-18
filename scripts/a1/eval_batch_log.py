"""A1 四格逐 batch 原始 latency 日志。

固定评测集:val 5k 按固定 seed 打乱(四格共享同一顺序),按 batch 切分;
四格对**完全相同的 batch 序列**逐一推理,逐批记录原始数据:
- batch 索引与图片清单
- gt 目标数(标签文件行数)、候选框数(NMS 前,conf 过滤后)、检测框数(NMS 后最终输出,每图)
- 整批 pre / inference / postprocess / total 耗时(ms)
另跑 val 5k 全量精度(mAP50/50-95、P/R、F1、每类 AP)并落盘。

输出(全部写到 --out 目录):
- batch_latency_<模型名>.csv   逐批一行(model,batch_idx,gt_n,det_n,pre/inf/post/total_ms)
- batch_latency_<模型名>.json  完整明细(含每批图片清单)
- val_metrics_<模型名>.json    精度汇总 + 每类 AP
- 控制台汇总:总延迟/均摊/图/FPS、gt/det 总数

实现要点(与 eval_pipeline.py 同口径):
- 显式 fuse 到标准部署形态(剥 one2many 头 + Conv-BN 融合)
- GPU 每段 sync 计时;NMS 用 torchvision CUDA kernel(计时前 import,不走纯 torch fallback)

用法:
    python scripts/a1/eval_batch_log.py \
        --model <A.pt> <B.pt> <C.pt> <D.pt> \
        --data ultralytics/cfg/datasets/coco-train-2017.yaml \
        --device 0 --batch 8 --seed 0 \
        --out <输出目录>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]  # YOLO-Master 仓库根
sys.path.insert(0, str(REPO_ROOT))

os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("YOLO_VERBOSE", "false")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression

assert str(REPO_ROOT) in ultralytics.__file__, f"加载的不是当前仓库的 ultralytics!got {ultralytics.__file__}"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", required=True, help="权重路径(四格一次传入,共享同一 batch 序列)")
    ap.add_argument("--data", required=True, help="数据集 YAML(path/train/val/names)")
    ap.add_argument("--device", default=None, help="cpu 或 cuda:0(默认自动选择)")
    ap.add_argument("--batch", type=int, default=8, help="batch 大小(5000 应整除,如 8)")
    ap.add_argument("--seed", type=int, default=0, help="5k 打乱 seed(四格共享同一顺序)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--split", default="val")
    ap.add_argument("--no-val", action="store_true", help="跳过精度 val(只出逐批 latency 日志)")
    ap.add_argument("--out", required=True, help="原始日志输出目录")
    return ap.parse_args()


def parse_device(s: str | None) -> torch.device:
    """规范化设备串(兼容 ultralytics 风格 '0' = cuda:0)。"""
    if s is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if s.isdigit():
        return torch.device(f"cuda:{s}")
    return torch.device(s)


def sync_time(device: torch.device) -> float:
    """GPU 同步后取时,保证每段计时覆盖真实执行。"""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return time.perf_counter()


def resolve_images(data_yaml: str | Path, split: str) -> tuple[list[Path], Path | None]:
    """从数据集 YAML 解析 split 图片列表与标签目录。"""
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg["path"])
    imgs_dir = root / cfg[split]
    lbls_dir = root / cfg[split].replace("images", "labels", 1)
    images = sorted(imgs_dir.glob("*.jpg")) + sorted(imgs_dir.glob("*.png"))
    return images, (lbls_dir if lbls_dir.is_dir() else None)


def letterbox_tensor(
    img0: np.ndarray, imgsz: int, stride: int, device: torch.device
) -> tuple[torch.Tensor, float, float, float]:
    """LetterBox 正方形 + BGR→RGB + 归一化,输出 (tensor(1,3,H,W), ratio, pad_left, pad_top)。

    ratio/pad 与图像变换同源(get_params),供 gt 标签映射到 letterbox 空间(P/R 匹配用)。
    """
    lb = LetterBox((imgsz, imgsz), auto=False, stride=stride)
    params = lb.get_params({"img": img0})
    im = lb(image=img0)
    im = im[..., ::-1].transpose((2, 0, 1))  # BGR→RGB, HWC→CHW
    im = np.ascontiguousarray(im)
    return (
        torch.from_numpy(im).to(device).float().unsqueeze(0) / 255.0,
        float(params["ratio"][0]),
        float(params["left"]),
        float(params["top"]),
    )


def count_gt(label_path: Path | None) -> int:
    """统计标签文件行数(无标签目录返回 -1)。"""
    if label_path is None:
        return -1
    return sum(1 for _ in label_path.read_text().splitlines()) if label_path.exists() else 0


def read_gt_boxes(label_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """读取 YOLO 标签:返回 (cls 数组, xyxy 归一化数组)。空标签返回空数组。"""
    if label_path is None or not label_path.exists():
        return np.zeros((0,), dtype=int), np.zeros((0, 4))
    cls, boxes = [], []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, w, h = map(float, parts[:5])
        cls.append(int(c))
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.asarray(cls, dtype=int), np.asarray(boxes).reshape(-1, 4)


def gt_to_letterbox(
    boxes_xyxy_norm: np.ndarray, w0: int, h0: int, ratio: float, pad_left: float, pad_top: float
) -> np.ndarray:
    """gt 归一化 xyxy → letterbox 空间 xyxy 像素。"""
    boxes = boxes_xyxy_norm.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * w0 * ratio + pad_left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * h0 * ratio + pad_top
    return boxes


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """两个 xyxy 框的 IoU。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def match_batch(
    det_boxes: list[np.ndarray], det_cls: list[np.ndarray], gt_boxes: list[np.ndarray], gt_cls: list[np.ndarray]
) -> tuple[int, int, int]:
    """逐图 greedy IoU(≥0.5,同类别)匹配,返回批级 (tp, fp, fn)。"""
    tp = fp = fn = 0
    for d_boxes, d_cls, g_boxes, g_cls in zip(det_boxes, det_cls, gt_boxes, gt_cls):
        matched = set()
        for i in range(len(d_boxes)):
            best_j, best_iou = -1, 0.0
            for j in range(len(g_boxes)):
                if j in matched or g_cls[j] != d_cls[i]:
                    continue
                iou = iou_xyxy(d_boxes[i], g_boxes[j])
                if iou > best_iou:
                    best_j, best_iou = j, iou
            if best_j >= 0 and best_iou >= 0.5:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
        fn += len(g_boxes) - len(matched)
    return tp, fp, fn


def nms_with_count(preds: torch.Tensor, conf: float, iou: float, max_det: int, end2end: bool):
    """NMS 并统计每图候选框数(NMS 前)与最终检测框数(NMS 后)。"""
    if end2end:
        # preds (B, max_det, 6):候选 = topk 后的 300 框中 conf 过阈值的数量
        cand_ns = (preds[..., 4] > conf).sum(dim=-1).tolist()
    else:
        # 稠密 (B, 4+nc, N):候选与 nms 内部 xc 同口径
        nc = preds.shape[1] - 4
        cand_ns = (preds[:, 4 : 4 + nc].amax(1) > conf).sum(dim=-1).tolist()
    out = non_max_suppression(preds, conf, iou, max_det=max_det, end2end=end2end)
    det_ns = [len(o) if o is not None else 0 for o in out]
    return out, det_ns, cand_ns


def run_val(yolo: YOLO, args: argparse.Namespace, out_dir: Path, stem: str) -> dict:
    """val 5k 全量:精度汇总 + 每类 AP。"""
    metrics = yolo.val(
        data=args.data,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        split=args.split,
        project=str(out_dir / "val"),
        name=stem,
        verbose=False,
    )
    rec = {k: float(v) for k, v in metrics.results_dict.items()}
    box = getattr(metrics, "box", None)
    if box is not None:
        for attr in ("maps", "ap50", "p", "r"):
            arr = getattr(box, attr, None)
            if arr is not None:
                try:
                    rec[f"per_class_{attr}"] = [round(float(x), 4) for x in np.asarray(arr).reshape(-1)]
                except (TypeError, ValueError):
                    pass
    out_dir.joinpath(f"val_metrics_{stem}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return rec


def run_batch_log(
    det_model: torch.nn.Module,
    batches: list[list[Path]],
    label_dir: Path | None,
    args: argparse.Namespace,
    device: torch.device,
    end2end: bool,
    stride: int,
) -> list[dict]:
    """逐 batch 推理并记录原始数据。"""
    import torchvision  # noqa: F401  计时前 import,保证 NMS 走 CUDA kernel 快路径

    records = []
    with torch.no_grad():
        for bi, batch_imgs in enumerate(batches):
            t0 = sync_time(device)
            tensors = []
            gt_cls_list, gt_boxes_list = [], []
            for p in batch_imgs:
                img0 = cv2.imread(str(p))
                h0, w0 = img0.shape[:2]
                t, ratio, pad_left, pad_top = letterbox_tensor(img0, args.imgsz, stride, device)
                tensors.append(t)
                if label_dir:
                    g_cls, g_boxes = read_gt_boxes(label_dir / f"{p.stem}.txt")
                    gt_cls_list.append(g_cls)
                    gt_boxes_list.append(gt_to_letterbox(g_boxes, w0, h0, ratio, pad_left, pad_top))
                else:
                    gt_cls_list.append(np.zeros((0,), dtype=int))
                    gt_boxes_list.append(np.zeros((0, 4)))
            x = torch.cat(tensors, dim=0)
            t1 = sync_time(device)
            raw = det_model(x)
            preds = raw[0] if isinstance(raw, (tuple, list)) else raw
            t2 = sync_time(device)
            out, det_ns, cand_ns = nms_with_count(preds, args.conf, args.iou, args.max_det, end2end)
            t3 = sync_time(device)
            # 计时外:拆 det 框并做 P/R 匹配(逐图 greedy IoU≥0.5)
            det_boxes_list = [o[:, :4].cpu().numpy() if o is not None and len(o) else np.zeros((0, 4)) for o in out]
            det_cls_list = [
                o[:, 5].cpu().numpy().astype(int) if o is not None and len(o) else np.zeros((0,), dtype=int)
                for o in out
            ]
            tp, fp, fn = match_batch(det_boxes_list, det_cls_list, gt_boxes_list, gt_cls_list)
            gt_ns = [count_gt(label_dir / f"{p.stem}.txt" if label_dir else None) for p in batch_imgs]
            records.append(
                {
                    "batch_idx": bi,
                    "imgs": [p.name for p in batch_imgs],
                    # 主字段 = 批级总数;*_per_img = 逐图明细(原始数据)
                    "gt_n": sum(gt_ns),
                    "gt_n_per_img": gt_ns,
                    "cand_n": sum(cand_ns),  # NMS 前的候选框数(conf 过滤后)
                    "cand_n_per_img": cand_ns,
                    "det_n": sum(det_ns),  # NMS 后的最终输出框数
                    "det_n_per_img": det_ns,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "P": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                    "R": round(tp / (tp + fn), 4) if tp + fn else 0.0,
                    "pre_ms": round((t1 - t0) * 1e3, 4),
                    "inf_ms": round((t2 - t1) * 1e3, 4),
                    "post_ms": round((t3 - t2) * 1e3, 4),
                    "total_ms": round((t3 - t0) * 1e3, 4),
                }
            )
    return records


def main() -> None:
    """主流程:5k 打乱 → 四格共享 batch 序列 → 逐批日志 + val 精度。"""
    args = parse_args()
    device = parse_device(args.device)
    images, label_dir = resolve_images(args.data, args.split)

    # 5k 固定 seed 打乱(四格共享同一顺序)
    rng = random.Random(args.seed)
    order = list(range(len(images)))
    rng.shuffle(order)
    shuffled = [images[i] for i in order]
    batches = [shuffled[i : i + args.batch] for i in range(0, len(shuffled), args.batch)]
    print(f"评测集: {len(images)} 张,seed={args.seed} 打乱,batch={args.batch} → {len(batches)} 批")
    if len(images) % args.batch:
        print(f"⚠️ 末批 {len(batches[-1])} 张(不足 batch)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 打乱清单落盘(可审计)
    out_dir.joinpath("shuffled_order.json").write_text(
        json.dumps({"seed": args.seed, "images": [p.name for p in shuffled]}, indent=2, ensure_ascii=False)
    )

    import re

    def model_tag(pt: str, idx: int) -> str:
        """产物文件标识:优先从归档路径提取 a1-s2-<格> 段(权重同名 best.pt 时防覆盖),否则用序号。"""
        for part in Path(pt).parts:
            m = re.match(r"a1-s2-([a-d])-", part)
            if m:
                return f"a1-s2-{m.group(1)}"
        return f"model{idx}"

    for idx, pt in enumerate(args.model):
        stem = model_tag(pt, idx)
        print(f"\n[开始] {pt}")
        yolo = YOLO(pt)
        val_rec = {} if args.no_val else run_val(yolo, args, out_dir, stem)
        if val_rec:
            print(
                f"  val: mAP50={val_rec.get('metrics/mAP50(B)', 0):.4f} mAP50-95={val_rec.get('metrics/mAP50-95(B)', 0):.4f}"
            )

        # 延迟:显式 fuse 到标准部署形态
        det_model = yolo.model
        det_model.fuse()
        det_model.eval().to(device)
        end2end = bool(getattr(det_model.model[-1], "end2end", False))
        stride = int(max(det_model.stride.tolist()))
        # warmup(首批)
        with torch.no_grad():
            w = torch.cat(
                [letterbox_tensor(cv2.imread(str(p)), args.imgsz, stride, device)[0] for p in batches[0]], dim=0
            )
            det_model(w)
            torch.cuda.empty_cache()

        records = run_batch_log(det_model, batches, label_dir, args, device, end2end, stride)

        # 落盘
        out_dir.joinpath(f"batch_latency_{stem}.json").write_text(
            json.dumps(
                {"model": pt, "batch": args.batch, "seed": args.seed, "records": records}, indent=2, ensure_ascii=False
            )
        )
        csv = ["model,batch_idx,n_imgs,gt_n,cand_n,det_n,tp,fp,fn,P,R,pre_ms,inf_ms,post_ms,total_ms"]
        for r in records:
            csv.append(
                f"{stem},{r['batch_idx']},{len(r['imgs'])},{r['gt_n']},{r['cand_n']},{r['det_n']},"
                f"{r['tp']},{r['fp']},{r['fn']},{r['P']},{r['R']},"
                f"{r['pre_ms']},{r['inf_ms']},{r['post_ms']},{r['total_ms']}"
            )
        out_dir.joinpath(f"batch_latency_{stem}.csv").write_text("\n".join(csv))

        # 汇总
        tots = [r["total_ms"] for r in records]
        infs = [r["inf_ms"] for r in records]
        n_imgs = sum(len(r["imgs"]) for r in records)
        gt_total = sum(r["gt_n"] for r in records)
        det_total = sum(r["det_n"] for r in records)
        cand_total = sum(r["cand_n"] for r in records)
        tp_sum = sum(r["tp"] for r in records)
        fp_sum = sum(r["fp"] for r in records)
        fn_sum = sum(r["fn"] for r in records)
        print(
            f"  batch 日志 {len(records)} 批 / {n_imgs} 图: 总 {sum(tots):.1f} ms | "
            f"均摊/图 {sum(tots) / n_imgs:.3f} ms | FPS {1000 * n_imgs / sum(tots):.1f} | "
            f"inf 均摊 {sum(infs) / n_imgs:.3f} ms"
        )
        print(
            f"  gt 目标总数 {gt_total} | 候选框总数 {cand_total} | 检测框总数 {det_total} | 检测/GT 比 {det_total / gt_total:.2f}"
        )
        print(
            f"  逐批匹配汇总: TP={tp_sum} FP={fp_sum} FN={fn_sum} | "
            f"P={tp_sum / (tp_sum + fp_sum):.4f} R={tp_sum / (tp_sum + fn_sum):.4f} (IoU≥0.5 逐图 greedy)"
        )
        print(f"  [产物] {out_dir}/batch_latency_{stem}.csv / .json")
    print("\n完成 ✅")


if __name__ == "__main__":
    main()
