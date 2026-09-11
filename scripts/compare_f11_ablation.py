#!/usr/bin/env python3
"""F11 · T4 三方同预算对照脚本(A baseline / B 表征 KD / C Router KD)

依据 06 文档 §6(F11全部任务分析与设计):
  - A 组 baseline:   稀疏路由学生,无 KD
  - B 组 表征 KD:    backbone 中间层特征对齐 DINO 教师(cosine_kd_loss)
  - C 组 Router KD:  路由软目标蒸馏(FoundationTeacherRouter + routing_kd_loss,F11 核心)

三组共用同一模型 yolo-master-n.yaml、同一数据、epoch/batch/imgsz/seed,
确保"同预算"可对照。

用法:
  # 1) 冒烟: 3 组各 1 epoch(coco8, 8GB 安全参数)
  python scripts/compare_f11_ablation.py --train --epochs 1 --data coco8.yaml --imgsz 320 --batch 4 --device 0

  # 2) 完整: 3 组各 100 epoch(VisDrone, 按 8GB 方法论)
  python scripts/compare_f11_ablation.py --train --epochs 100 --data VisDrone.yaml --imgsz 640 --batch 8 --device 0

  # 3) 汇总: 只读 runs/f11_ablation 下的 results.csv 生成对照表
  python scripts/compare_f11_ablation.py --summary-only --project runs/f11_ablation

参考:
  - ultralytics/nn/foundation_distill_model.py: FoundationDistillationModel 包装器
  - ultralytics/nn/foundation/losses.py:        RouterKDLoss / cosine_kd_loss / relational_kd_loss
  - ultralytics/nn/foundation/routing.py:       FoundationTeacherRouter / routing_kd_loss
  - #54 scripts/compare_mot_ablation.py:        三变体消融框架(本脚本轻量版)

文档版本：v1.0（2026-08-24）｜Owner：张伟林（Zviolin）｜F11 T4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.foundation import (  # noqa: E402
    DEFAULT_DINOV2_MODEL,
    DEFAULT_DINOV3_MODEL,
    DEFAULT_SIGLIP2_MODEL,
)
from ultralytics.utils import YAML  # noqa: E402

CONFIG_DIR = ROOT / "configs"

# --teacher 切换教师时 foundation_model 的默认值(与 foundation_distill_model.py 白名单一致)
TEACHER_DEFAULT_MODELS = {
    "dinov2": DEFAULT_DINOV2_MODEL,
    "dinov3": DEFAULT_DINOV3_MODEL,
    "siglip2": DEFAULT_SIGLIP2_MODEL,
}


@dataclass(frozen=True)
class GroupSpec:
    """三方对照中的一组定义。"""

    key: str
    label: str
    cfg: Path


GROUPS = [
    GroupSpec(key="a", label="A-baseline", cfg=CONFIG_DIR / "f11_baseline.yaml"),
    GroupSpec(key="b", label="B-repr-kd", cfg=CONFIG_DIR / "f11_repr_kd.yaml"),
    GroupSpec(key="c", label="C-router-kd", cfg=CONFIG_DIR / "f11_router_kd.yaml"),
]


def normalize_device(device: str) -> str:
    """把数字设备字符串(如 '0')规范为完整设备字符串(如 'cuda:0')。"""
    if not device:
        return "cpu"
    if device.isdigit():
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    return device


def build_foundation_overrides(spec: GroupSpec, args: argparse.Namespace | None = None) -> dict:
    """读取 config yaml 的 foundation 段,构造训练 overrides。

    T6R.4 教师对比:args.teacher 非空时,覆盖 foundation_teacher。
    """
    if not spec.cfg.exists():
        raise SystemExit(f"缺少配置: {spec.cfg}")
    data = YAML.load(spec.cfg)
    if not isinstance(data, dict):
        raise SystemExit(f"配置格式错误: {spec.cfg}")
    foundation = data.get("foundation", {}) or {}
    enabled = bool(foundation.get("enabled", False))
    overrides = {"model": str(data.get("model"))}
    if not enabled:
        # A 组: 显式关闭 Foundation,无任何蒸馏副作用
        overrides.update(
            {
                "foundation_enabled": False,
                "foundation_loss_weight": 0.0,
                "foundation_router_distill": False,
                "foundation_router_loss_weight": 0.0,
            }
        )
        return overrides

    overrides["foundation_enabled"] = True
    overrides["foundation_backend"] = str(foundation.get("backend", "transformers"))
    # T6R.4: --teacher 命令行参数优先于 yaml 里的 teacher 字段
    teacher_default = str(foundation.get("teacher", "dinov3"))
    if args is not None and getattr(args, "teacher", None):
        teacher_default = args.teacher
    overrides["foundation_teacher"] = teacher_default
    overrides["foundation_loss_weight"] = float(foundation.get("loss_weight", 0.0) or 0.0)
    router_distill = bool(foundation.get("router_distill", False))
    router_weight = float(foundation.get("router_loss_weight", 0.0) or 0.0)
    overrides["foundation_router_distill"] = router_distill
    overrides["foundation_router_loss_weight"] = router_weight
    # F11 路由 KD 温度 τ: --router-temperature 命令行优先于 yaml 的 router_temperature 字段
    # (gen_q_teacher.py 熵校验确定 τ=0.5 合法;框架默认 1.0 过高导致教师目标近均匀)
    router_temperature = foundation.get("router_temperature", None)
    if args is not None and getattr(args, "router_temperature", None) is not None:
        router_temperature = args.router_temperature
    if router_temperature is not None:
        overrides["foundation_router_temperature"] = float(router_temperature)
    # 教师模型(HF model_id) → foundation_model(F11 8GB 友好)
    # 注意: yaml 里的 model 字段是为该 yaml 默认教师配置的;--teacher 切换教师时必须改用
    # 目标教师的默认 model_id——直接丢弃键会被 get_cfg 校验器拒绝(router KD 要求
    # foundation_model/foundation_weights 非空),照抄 yaml 的 model 又会把 SigLIP2 的
    # model_id 误传给 DINOv2/DINOv3 导致加载错乱
    yaml_teacher = str(foundation.get("teacher", "dinov3"))
    teacher_switched = args is not None and getattr(args, "teacher", None) and args.teacher != yaml_teacher
    if teacher_switched:
        default_model = TEACHER_DEFAULT_MODELS.get(teacher_default)
        if default_model is not None:
            overrides["foundation_model"] = default_model
    elif foundation.get("model"):
        overrides["foundation_model"] = str(foundation.get("model"))
    # 8GB 降级: 教师统一 cpu 推理,避免与学生抢显存
    overrides["foundation_teacher_device"] = "cpu"
    return overrides


def run_train(args) -> None:
    """按指定 group 顺序训练三组。"""
    project = Path(args.project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    device = normalize_device(args.device)

    keys = [g.key for g in GROUPS if g.key in args.groups]
    if not keys:
        raise SystemExit(f"--groups 无效: {args.groups}(可用: a,b,c)")

    for key in keys:
        spec = next(g for g in GROUPS if g.key == key)
        # T6R.2: --cfg 覆盖该组的 config yaml(如 --cfg configs/f11_mot.yaml 跑 MoT 学生)
        cfg_override = getattr(args, "cfg", None)
        if cfg_override:
            cfg_path = Path(cfg_override)
            if not cfg_path.exists():
                raise SystemExit(f"--cfg 文件不存在: {cfg_path}")
            spec = replace(spec, cfg=cfg_path)
            print(f"[F11][T4] --cfg 覆盖: {spec.label} → {cfg_path}")
        overrides = build_foundation_overrides(spec, args)
        print(f"\n[F11][T4] ==== 训练 {spec.label} ({spec.cfg.name}) ====")
        print(f"[F11][T4] foundation overrides: {json.dumps(overrides, ensure_ascii=False)}")

        model_yaml = overrides.pop("model")
        # T6R.1: --layers N 覆盖 model yaml 中的 layers 字段(给 MoEBlock)
        layers_override = getattr(args, "layers", None)
        model = YOLO(str(ROOT / model_yaml))
        if layers_override and layers_override > 0:
            try:
                # ultralytics 8.4:model.model.args.layers 可写
                if hasattr(model, "model") and hasattr(model.model, "args"):
                    old_layers = model.model.args.get("layers") if isinstance(model.model.args, dict) else None
                    model.model.args["layers"] = layers_override
                    print(f"[F11][T4] --layers 覆盖: {old_layers} → {layers_override}")
                else:
                    print(f"[F11][T4][WARN] --layers {layers_override} 无法注入(模型无 model.args),跳过")
            except Exception as exc:  # noqa: BLE001
                print(f"[F11][T4][WARN] --layers 注入失败: {exc}")
        # run 子目录: 用户传 --name 时优先,否则用 group 默认小写名
        run_name = args.name if args.name else f"{spec.label.lower()}"
        train_overrides = {
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": device,
            "project": str(project),
            "name": run_name,
            "seed": args.seed,
            "workers": args.workers if args.workers is not None else 4,
            "cache": args.cache if args.cache else False,
            "exist_ok": True,  # 同名 run 覆盖,不再自动加 -N 后缀
        }
        # 接训模式: --resume 时只训第一个 key,且用 YOLO(ckpt).train(resume=True)
        if args.resume:
            if key != keys[0]:
                print(f"[F11][T4] --resume 模式只训第一个 group ({keys[0]}),跳过 {key}")
                continue
            ckpt_path = project / run_name / "weights" / "last.pt"
            if not ckpt_path.exists():
                raise SystemExit(f"--resume 失败: 找不到 {ckpt_path}")
            print(f"[F11][T4] 接训 {ckpt_path}")
            model = YOLO(str(ckpt_path))
            # 接训时 resume=True,ckpt 里的 train_args 决定 epochs/batch/project/name,
            # 不传 data/device/workers(避免与 ckpt 不一致导致 state shape 冲突)
            model.train(resume=True)
            print(f"[F11][T4] ✅ {spec.label} 接训完成 → {project / run_name}")
            return
        train_overrides.update(overrides)
        model.train(**train_overrides)
        print(f"[F11][T4] ✅ {spec.label} 训练完成 → {project / run_name}")


def collect_metrics(project: Path) -> list[dict]:
    """收集每组每个 seed 的 results.csv 末行 mAP。"""
    rows = []
    if not project.exists():
        return rows
    for csv_path in sorted(project.rglob("results.csv")):
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                last = None
                for last in reader:
                    pass
                if last is None:
                    continue
            rows.append(
                {
                    "run": str(csv_path.parent.name),
                    "mAP50-95": float(last.get("metrics/mAP50-95(B)", "nan")),
                    "mAP50": float(last.get("metrics/mAP50(B)", "nan")),
                    "box_loss": float(last.get("train/box_loss", "nan")),
                    "cls_loss": float(last.get("train/cls_loss", "nan")),
                }
            )
        except (ValueError, KeyError, csv.Error) as exc:
            print(f"[F11][T4][WARN] 跳过 {csv_path}: {exc}")
    return rows


def run_summary(args) -> None:
    """汇总三组 mAP 生成对照表。"""
    project = Path(args.project).resolve()
    rows = collect_metrics(project)
    if not rows:
        print(f"[F11][T4] 未在 {project} 找到任何 results.csv,请先 --train")
        return

    print(f"\n[F11][T4] ==== 三方对照汇总 ({project}) ====")
    print(f"{'run':<20} {'mAP50-95':>10} {'mAP50':>8} {'box_loss':>10} {'cls_loss':>10}")
    print("-" * 62)
    for row in sorted(rows, key=lambda r: -r["mAP50-95"]):
        print(
            f"{row['run']:<20} {row['mAP50-95']:>10.3f} {row['mAP50']:>8.3f} "
            f"{row['box_loss']:>10.4f} {row['cls_loss']:>10.4f}"
        )

    # 保存 JSON 供文档引用
    out = project / "f11_ablation_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n[F11][T4] 汇总已保存: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F11 · T4 三方同预算对照(A/B/C)")
    parser.add_argument("--train", action="store_true", help="运行三组训练")
    parser.add_argument("--summary-only", action="store_true", help="只汇总不训练")
    parser.add_argument("--groups", default="a,b,c", help="逗号分隔的组 key(默认 a,b,c)")
    parser.add_argument("--epochs", type=int, default=1, help="每组建训 epoch(冒烟=1)")
    parser.add_argument("--data", default="coco8.yaml", help="数据集 yaml")
    parser.add_argument("--imgsz", type=int, default=320, help="输入分辨率(8GB 冒烟=320)")
    parser.add_argument("--batch", type=int, default=4, help="batch size(8GB=4)")
    parser.add_argument("--device", default="0", help="训练设备(0 / cpu / cuda:0)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--project", default="runs/f11_ablation", help="输出目录")
    # F11 扩展(2026-08-30 新增):接训/批量实验用 — 自定义 run 名 + workers + cache
    parser.add_argument("--name", default=None, help="run 子目录名(可选,不传走 group 默认名)")
    parser.add_argument("--workers", type=int, default=None, help="dataloader workers(可选,不传走 YOLO 默认)")
    parser.add_argument("--cache", default=None, choices=["ram", "disk", "false"], help="dataset cache 模式")
    # F11 扩展(2026-09-03 新增):接训支持
    parser.add_argument("--resume", action="store_true", help="从 project/name/weights/last.pt 接训(只对第一个 group 生效)")
    # F11 扩展(2026-09-06 新增):T6R.1 多层 / T6R.4 教师对比
    parser.add_argument("--layers", type=int, default=None, help="MoEBlock 层数(默认 None=model.yaml 原值,T6R.1 用 3)")
    parser.add_argument("--teacher", default=None, help="覆盖 config yaml 中的 foundation.teacher(T6R.4 用,合法值: dinov2 / dinov3 / siglip2 / multi)")
    parser.add_argument("--cfg", default=None, help="覆盖组的 config yaml(T6R.2 用,如 configs/f11_mot.yaml)")
    parser.add_argument("--router-temperature", type=float, default=None, help="路由 KD 温度 τ(覆盖 yaml 的 router_temperature,推荐 0.5)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train:
        run_train(args)
    if args.summary_only or not args.train:
        run_summary(args)


if __name__ == "__main__":
    main()
