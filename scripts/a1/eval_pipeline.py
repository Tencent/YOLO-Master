"""A1 模型推理评测 pipeline:精度 + latency + 逐图检测数量。

对一个训练完成的模型权重做纯评测(不做训练),如实记录三类指标:
1. 精度(数据有 gt 时):mAP50/50-95、P/R/F1、mAP_s/m/l —— 复用 ultralytics val 链路
2. latency:单图(batch=1 三段计时)+ 批处理(batch=N 整批计时)双口径
3. 检测目标数量:每图检测框数、gt 目标数(有标签时)、NMS 前候选数,与该图 latency 逐图对应

输出:逐图明细 JSON/CSV + 汇总统计(mean±std/median/p95/FPS)。

用法:
    python scripts/a1/eval_pipeline.py \
        --model <权重.pt> --data <数据.yaml> \
        [--device cpu] [--batch 8] [--rounds 5] [--max-imgs 32] [--no-val]

实现要点(代码事实,2026-09-02 核实):
- 显式调用 DetectionModel.fuse()(nn/tasks.py:316):剥 end2end 模型的 one2many 头 + Conv-BN 融合,
  等价标准部署路径 YOLO(pt).predict() 的 AutoBackend(fuse=True)。
  注意:不 fuse 直接 torch.load 推理,端到端模型 forward 会白算 one2many 头(head.py:161-168),
  检测头计算近翻倍。
- 计时逐段手动插桩,GPU 上每段 torch.cuda.synchronize() 后取时。
- 预处理统一 LetterBox(auto=False) 正方形输入(与 val 链路一致),批处理可直接 stack。
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
    ap.add_argument("--model", nargs="+", required=True, help="权重路径(可传多个,一次评测多格)")
    ap.add_argument("--data", required=True, help="数据集 YAML(path/train/val/names)")
    ap.add_argument("--device", default=None, help="cpu 或 cuda:0(默认自动选择)")
    ap.add_argument("--batch", type=int, default=8, help="批处理口径的 batch 大小")
    ap.add_argument("--rounds", type=int, default=5, help="每图/每批计时重复轮数(≥3)")
    ap.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    ap.add_argument("--conf", type=float, default=0.001, help="置信度阈值(与训练默认一致)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    ap.add_argument("--max-det", type=int, default=300, help="单图最大检测数")
    ap.add_argument("--max-imgs", type=int, default=None, help="延迟评测用图上限(固定 seed 抽样;CPU 全量很慢)")
    ap.add_argument("--seed", type=int, default=0, help="抽样 seed")
    ap.add_argument("--split", default="val", help="数据 YAML 中的 split 键")
    ap.add_argument("--no-val", action="store_true", help="跳过精度评测")
    ap.add_argument(
        "--nms-impl",
        default="torchvision",
        choices=["torchvision", "torchnms"],
        help="NMS 实现:torchvision(CUDA kernel,标准部署快路径)/ torchnms(ultralytics 纯 torch fallback,慢)",
    )
    ap.add_argument("--out", default=None, help="输出目录(默认 runs/eval_pipeline/<模型名>_<时间戳>)")
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


def sample_images(images: list[Path], max_imgs: int | None, seed: int) -> list[Path]:
    """固定 seed 抽样(max_imgs=None 时返回全部)。"""
    if max_imgs is None or max_imgs >= len(images):
        return images
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(images)), max_imgs))
    return [images[i] for i in idx]


def count_gt(label_path: Path | None) -> int:
    """统计标签文件行数(无标签目录返回 None 语义的 0,由调用方区分)。"""
    if label_path is None:
        return -1
    return sum(1 for _ in label_path.read_text().splitlines()) if label_path.exists() else 0


def letterbox_tensor(img0: np.ndarray, imgsz: int, stride: int, device: torch.device) -> torch.Tensor:
    """LetterBox 正方形 + BGR→RGB + 归一化,输出 (1,3,H,W) float32。"""
    im = LetterBox(imgsz, auto=False, stride=stride)(image=img0)
    im = im[..., ::-1].transpose((2, 0, 1))  # BGR→RGB, HWC→CHW
    im = np.ascontiguousarray(im)
    return torch.from_numpy(im).to(device).float().unsqueeze(0) / 255.0


def nms_with_count(preds: torch.Tensor, conf: float, iou: float, max_det: int, end2end: bool):
    """NMS 前统计每图候选数(不计时),返回 (nms 输出, 候选数列表)。"""
    if end2end:
        # preds (B, max_det, 6):候选 = topk 后的 300 框中 conf 过阈值的数量
        n_cand = (preds[..., 4] > conf).sum(dim=-1).tolist()
    else:
        # 稠密 (B, 4+nc, N):候选与 nms 内部 xc 同口径
        nc = preds.shape[1] - 4
        n_cand = (preds[:, 4 : 4 + nc].amax(1) > conf).sum(dim=-1).tolist()
    out = non_max_suppression(preds, conf, iou, max_det=max_det, end2end=end2end)
    return out, n_cand


def summarize(times_ms: list[float]) -> dict[str, float]:
    """汇总一次计时的统计量。"""
    ts = np.asarray(times_ms)
    return {
        "mean_ms": round(float(ts.mean()), 4),
        "std_ms": round(float(ts.std()) if len(ts) >= 2 else 0.0, 4),
        "median_ms": round(float(np.median(ts)), 4),
        "p95_ms": round(float(np.percentile(ts, 95)), 4),
        "fps": round(1000.0 / float(ts.mean()), 2) if ts.mean() > 0 else 0.0,
        "n": len(times_ms),
    }


def measure_single(
    det_model: torch.nn.Module,
    img_path: Path,
    label_path: Path | None,
    args: argparse.Namespace,
    device: torch.device,
    end2end: bool,
    stride: int,
) -> dict:
    """单图口径:逐图 rounds 轮三段计时,记录检测框数/候选数。"""
    img0 = cv2.imread(str(img_path))
    rec = {
        "img": img_path.name,
        "gt_n": count_gt(label_path),
        "cand_n": None,
        "det_n": None,
        "rounds": [],
    }
    for _ in range(args.rounds):
        t0 = sync_time(device)
        im = letterbox_tensor(img0, args.imgsz, stride, device)
        t1 = sync_time(device)
        raw = det_model(im)
        preds = raw[0] if isinstance(raw, (tuple, list)) else raw  # 非导出 forward 返回 (y, preds)
        t2 = sync_time(device)
        out, n_cand = nms_with_count(preds, args.conf, args.iou, args.max_det, end2end)
        t3 = sync_time(device)
        rec["cand_n"] = n_cand[0]
        rec["det_n"] = len(out[0]) if out[0] is not None else 0
        rec["rounds"].append(
            {
                "pre_ms": round((t1 - t0) * 1e3, 4),
                "inf_ms": round((t2 - t1) * 1e3, 4),
                "post_ms": round((t3 - t2) * 1e3, 4),
                "total_ms": round((t3 - t0) * 1e3, 4),
            }
        )
    return rec


def measure_batch(
    det_model: torch.nn.Module,
    img_paths: list[Path],
    label_dir: Path | None,
    args: argparse.Namespace,
    device: torch.device,
    end2end: bool,
    stride: int,
) -> dict:
    """批处理口径:batch 张图 stack 后整批 rounds 轮计时,报整批与均摊/图。"""
    imgs = [cv2.imread(str(p)) for p in img_paths]
    x = torch.cat([letterbox_tensor(im, args.imgsz, stride, device) for im in imgs], dim=0)
    gt_ns = [count_gt(label_dir / f"{p.stem}.txt" if label_dir else None) for p in img_paths]
    rounds = []
    for _ in range(args.rounds):
        t0 = sync_time(device)
        raw = det_model(x)
        preds = raw[0] if isinstance(raw, (tuple, list)) else raw  # 非导出 forward 返回 (y, preds)
        t1 = sync_time(device)
        out, n_cand = nms_with_count(preds, args.conf, args.iou, args.max_det, end2end)
        t2 = sync_time(device)
        det_ns = [len(o) if o is not None else 0 for o in out]
        rounds.append(
            {
                "inf_ms": round((t1 - t0) * 1e3, 4),  # 整批 forward
                "post_ms": round((t2 - t1) * 1e3, 4),  # 整批后处理
                "total_ms": round((t2 - t0) * 1e3, 4),  # 整批合计
            }
        )
    return {
        "imgs": [p.name for p in img_paths],
        "gt_ns": gt_ns,
        "cand_ns": n_cand,
        "det_ns": det_ns,
        "batch": len(img_paths),
        "rounds": rounds,
    }


def run_val(yolo: YOLO, args: argparse.Namespace, out_dir: Path) -> dict:
    """精度评测:复用 ultralytics val 链路,返回 results_dict。"""
    metrics = yolo.val(
        data=args.data,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        split=args.split,
        project=str(out_dir / "val"),
        name="",
        verbose=False,
    )
    return {k: float(v) for k, v in metrics.results_dict.items()}


def eval_one_model(
    pt: str, images: list[Path], label_dir: Path | None, args: argparse.Namespace, device: torch.device, out_dir: Path
) -> dict:
    """对单个权重完成精度 + 单图/批处理 latency + 逐图数量评测。"""
    yolo = YOLO(pt)
    record: dict = {
        "model": pt,
        "device": str(device),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "batch": args.batch,
        "rounds": args.rounds,
        "seed": args.seed,
        "split": args.split,
        "n_imgs": len(images),
        "end2end": None,
    }
    # 1. 精度(未 fuse 状态,与训练内 val 同形态)
    record["val"] = {} if args.no_val else run_val(yolo, args, out_dir)

    # 2. 延迟:显式 fuse 到标准部署形态
    det_model = yolo.model
    det_model.fuse()
    det_model.eval().to(device)
    head = det_model.model[-1]
    end2end = bool(getattr(head, "end2end", False))
    record["end2end"] = end2end
    record["nms_impl"] = args.nms_impl
    stride = int(max(det_model.stride.tolist()))
    if args.nms_impl == "torchvision":
        # nms.py 按 sys.modules 选路:预先 import torchvision(慢,计时外)走 CUDA NMS 快路径
        import torchvision  # noqa: F401

    # warmup
    with torch.no_grad():
        for _ in range(3):
            im_warm = letterbox_tensor(cv2.imread(str(images[0])), args.imgsz, stride, device)
            det_model(im_warm)

    # 2a. 单图口径
    single_recs = []
    with torch.no_grad():
        for img in images:
            lbl = label_dir / f"{img.stem}.txt" if label_dir else None
            single_recs.append(measure_single(det_model, img, lbl, args, device, end2end, stride))
    record["single"] = single_recs
    for seg in ("pre_ms", "inf_ms", "post_ms", "total_ms"):
        record[f"single_{seg[:-3]}"] = summarize([r["rounds"][-1][seg] for r in single_recs])

    # 2b. 批处理口径(末批不足 batch 时按实际数量)
    batch_recs = []
    with torch.no_grad():
        for i in range(0, len(images), args.batch):
            chunk = images[i : i + args.batch]
            batch_recs.append(measure_batch(det_model, chunk, label_dir, args, device, end2end, stride))
    record["batch_runs"] = batch_recs
    for seg in ("inf_ms", "post_ms", "total_ms"):
        per_batch = [r["rounds"][-1][seg] for r in batch_recs]
        record[f"batch_{seg[:-3]}"] = summarize(per_batch)
    record["batch_per_img_ms"] = summarize([r["rounds"][-1]["total_ms"] / r["batch"] for r in batch_recs])
    return record


def write_outputs(record: dict, out_dir: Path) -> None:
    """落盘逐图明细 JSON/CSV 与汇总 JSON。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_detail.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    csv_lines = ["model,img,gt_n,cand_n,det_n,pre_ms,inf_ms,post_ms,total_ms"]
    for r in record["single"]:
        last = r["rounds"][-1]
        csv_lines.append(
            f"{Path(record['model']).stem},{r['img']},{r['gt_n']},{r['cand_n']},{r['det_n']},"
            f"{last['pre_ms']},{last['inf_ms']},{last['post_ms']},{last['total_ms']}"
        )
    (out_dir / "eval_detail.csv").write_text("\n".join(csv_lines))
    summary = {
        "model": record["model"],
        "device": record["device"],
        "end2end": record["end2end"],
        "val": record["val"],
        "single": {k: v for k, v in record.items() if k.startswith("single_")},
        "batch": {k: v for k, v in record.items() if k.startswith("batch_")},
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def print_summary(record: dict) -> None:
    """控制台汇总表。"""
    m = Path(record["model"]).stem
    s, b = record["single_total"], record["batch_total"]
    print(f"\n== {m} | device={record['device']} | end2end={record['end2end']} ==")
    if record["val"]:
        v = record["val"]
        print(f"  val: mAP50={v.get('metrics/mAP50(B)', 0):.4f} mAP50-95={v.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  单图: 合计 {s['mean_ms']}±{s['std_ms']} ms (median {s['median_ms']}, p95 {s['p95_ms']}, FPS {s['fps']})")
    for seg, lab in (("pre", "pre"), ("inf", "inference"), ("post", "postprocess")):
        seg_s = record[f"single_{seg}"]
        print(f"    {lab:<11} {seg_s['mean_ms']}±{seg_s['std_ms']} ms")
    print(
        f"  批处理(batch={record['batch']}): 整批 {b['mean_ms']}±{b['std_ms']} ms | "
        f"均摊/图 {record['batch_per_img_ms']['mean_ms']} ms (FPS {record['batch_per_img_ms']['fps']})"
    )


def main() -> None:
    """主流程:逐模型评测并落盘。"""
    args = parse_args()
    device = parse_device(args.device)
    images, label_dir = resolve_images(args.data, args.split)
    images = sample_images(images, args.max_imgs, args.seed)
    if not images:
        sys.exit(f"split={args.split} 下无图片,请检查数据集 YAML")
    print(f"评测图数: {len(images)} | device={device} | rounds={args.rounds} | batch={args.batch}")

    out_root = Path(args.out) if args.out else REPO_ROOT / "runs" / "eval_pipeline"
    out_root.mkdir(parents=True, exist_ok=True)
    for pt in args.model:
        print(f"\n[开始] {pt}")
        out_dir = out_root / f"{Path(pt).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        record = eval_one_model(pt, images, label_dir, args, device, out_dir)
        write_outputs(record, out_dir)
        print_summary(record)
        print(f"[产物] {out_dir}")
    print("\n完成 ✅")


if __name__ == "__main__":
    main()
