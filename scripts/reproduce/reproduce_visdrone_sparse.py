#!/usr/bin/env python3
"""按照论文 ES-MoE 设计训练 VisDrone baseline：soft Top-2 训练 + hard Top-2 验证。

与 reproduce_visdrone.py 的区别：不使用 --no-sparse-eval，确保 top_k=2 < num_experts=4，
实现论文设计的真正稀疏路由。每 N 个 epoch 输出路由诊断指标到 W&B。
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import (
    DatasetSpec, ModelSpec, _read_last_metrics, _float_or_blank,
    write_summary, _completed_epoch, _make_wandb_callbacks,
)

ROOT = Path(__file__).resolve().parents[2]
SPARSE_MODEL = ModelSpec("EsMoE-N", "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml", uses_esmoe=True)
DATASET = DatasetSpec("VisDrone", "VisDrone.yaml", "")  # project 由命令行指定


def _make_sparse_topk_callback():
    """确保 ES_MOE 使用 top_k=2 的稀疏路由（论文设计）。"""
    from ultralytics.nn.modules.moe.modules import ES_MOE
    from ultralytics.utils import LOGGER

    state = {"applied": False}

    def _apply(trainer):
        if state["applied"]:
            return
        targets = [trainer.model]
        ema = getattr(trainer, "ema", None)
        if ema is not None and getattr(ema, "ema", None) is not None:
            targets.append(ema.ema)

        count = 0
        for target in targets:
            for m in target.modules():
                if isinstance(m, ES_MOE):
                    m.use_sparse_inference = True
                    m.use_top_k = True
                    m.top_k = 2
                    count += 1
        if count:
            LOGGER.info(f"[reproduce:sparse] top_k=2 enabled on {count} ES_MOE module(s)")
            state["applied"] = True

    return _apply


def _make_routing_diag_callback(every_n_epochs: int = 5):
    """每 N 个 epoch 在验证后收集路由诊断指标，记录到 W&B 及训练日志。"""
    from ultralytics.nn.modules.moe.modules import ES_MOE
    from ultralytics.utils import LOGGER

    def _on_val_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        if epoch % every_n_epochs != 0:
            return

        # 从 EMA 模型读取各层 ES_MOE 的 last_routing_snapshot
        ema = getattr(trainer, "ema", None)
        target = ema.ema if (ema is not None and getattr(ema, "ema", None) is not None) else trainer.model

        layers_data = {}
        for name, m in target.named_modules():
            snap = getattr(m, "last_routing_snapshot", None)
            if not isinstance(snap, dict) or not snap:
                continue
            usage = snap.get("expert_usage")
            if usage is None:
                continue
            usage_list = [float(x) for x in usage.detach().cpu().reshape(-1)]
            layers_data[name] = {
                "num_experts": len(usage_list),
                "usage": usage_list,
            }

        if not layers_data:
            return

        # 计算每层指标
        for name, d in layers_data.items():
            short = name.replace("model.", "L").replace(".routing", "")
            usage = d["usage"]
            gini = _compute_gini(usage)
            dominant = max(usage) / (sum(usage) + 1e-12)

            LOGGER.info(
                f"[routing] epoch={epoch:>4d}  {short}: "
                + "  ".join(f"E{i}:{u:.3f}" for i, u in enumerate(usage))
                + f"  | Gini={gini:.3f}  Dominant={dominant:.3f}"
            )

            # 写入 W&B
            try:
                wandb_run = getattr(trainer, "wandb", None)
                if wandb_run is not None:
                    wandb_run.log({
                        f"routing/{short}/gini": gini,
                        f"routing/{short}/dominant": dominant,
                        "routing/epoch": epoch,
                    }, step=epoch)
                    for i, u in enumerate(usage):
                        wandb_run.log({f"routing/{short}/E{i}_usage": u}, step=epoch)
            except Exception:
                pass

    return _on_val_end


def _compute_gini(values):
    import numpy as np
    arr = np.array(values, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return 0.0
    arr = arr / total
    n = len(arr)
    diff_sum = np.abs(arr[:, None] - arr[None, :]).sum()
    return float(diff_sum / (2 * n))


def train_sparse(args):
    from ultralytics import YOLO

    dataset = DatasetSpec("VisDrone", "VisDrone.yaml", args.project)
    spec = SPARSE_MODEL
    run_name = f"{dataset.name}_{spec.name}"
    run_dir = Path(args.project) / run_name

    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    done = _completed_epoch(run_dir)

    if best_pt.exists() and done is not None and done + 1 >= args.epochs:
        print(f"[skip] {run_name}: complete at epoch {done}", flush=True)
        return {"model": spec.name, "status": "skipped"}

    if last_pt.exists() and done is not None:
        print(f"[resume] {run_name}: {last_pt} epoch={done} -> {args.epochs}", flush=True)
        model = YOLO(str(last_pt))
        resume = True
    else:
        print(f"[train:sparse] {run_name}: cfg={spec.cfg} data={dataset.data}  epochs={args.epochs}", flush=True)
        model = YOLO(str(ROOT / spec.cfg))
        resume = False

    # 注入 top_k=2 回调
    cb = _make_sparse_topk_callback()
    model.add_callback("on_pretrain_routine_end", cb)

    # W&B 回调
    if args.wandb and args.wandb_mode != "disabled":
        wandb_callbacks = _make_wandb_callbacks(run_name, dataset, spec, args, dense_val=False)
        for event, fn in wandb_callbacks.items():
            model.add_callback(event, fn)

    # 路由诊断回调（每 N 个 epoch 在验证后记录路由指标）
    diag_cb = _make_routing_diag_callback(every_n_epochs=args.routing_diag_interval)
    model.add_callback("on_fit_epoch_end", diag_cb)

    start = time.time()
    model.train(
        data=dataset.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(args.project),
        name=run_name,
        exist_ok=True,
        pretrained=False,
        lora_r=0,
        optimizer="auto",
        val=True,
        plots=True,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
        verbose=args.verbose,
        moe_balance_loss=args.moe_balance_loss,
    )
    return {"model": spec.name, "status": "resumed" if resume else "ok",
            "duration_s": f"{time.time() - start:.1f}"}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Reproduce VisDrone EsMoE-N with sparse top-k=2 (paper design)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--project", default="/root/workspace/YOLO-Master-docs/issue2/VisDrone/pahse0-baseline-train")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check-build", action="store_true")
    # W&B
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", action="store_false", dest="wandb")
    p.add_argument("--wandb-project", default="yolo-master-reproduce")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--moe-balance-loss", type=float, default=0.6)
    p.add_argument("--routing-diag-interval", type=int, default=5, help="每 N 个 epoch 收集路由诊断")

    args = p.parse_args()

    if args.check_build:
        from ultralytics.nn.tasks import DetectionModel
        m = DetectionModel(str(ROOT / SPARSE_MODEL.cfg), ch=3, nc=80, verbose=False)
        print(f"[build-ok] {SPARSE_MODEL.name}: {sum(p.numel() for p in m.parameters()) / 1e6:.3f}M")
        sys.exit(0)

    if args.dry_run:
        print(f"[dry-run] Project: {args.project}  epochs={args.epochs}  batch={args.batch}  "
              f"device={args.device}  wandb={args.wandb}")
        sys.exit(0)

    Path(args.project).mkdir(parents=True, exist_ok=True)
    try:
        status = train_sparse(args)
        print(f"\n[train:sparse] DONE: {status}")
    except Exception as exc:
        print(f"[fail] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    # 收集产物到项目目录
    dataset = DatasetSpec("VisDrone", "VisDrone.yaml", args.project)
    run_dir = Path(args.project) / f"{dataset.name}_{spec.name}"
    out_dir = Path(args.project) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        import shutil
        shutil.copy2(best_src, out_dir / "best.pt")
        print(f"[collect] best.pt -> {out_dir / 'best.pt'}")

    for f in ["args.yaml", "results.csv"]:
        src = run_dir / f
        if src.exists():
            import shutil
            shutil.copy2(src, out_dir / f)
            print(f"[collect] {f} -> {out_dir / f}")

    sys.exit(0)
