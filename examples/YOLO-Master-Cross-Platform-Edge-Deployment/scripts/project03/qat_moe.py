#!/usr/bin/env python
"""QAT for pruned YOLO-Master checkpoints (project03 phase 2, INT8 path).

PTQ measured on this family: distributed trunk noise kills implicit-int8 accuracy
(COCO surgical PTQ -30.7% relative; stem-pair recipe still -7.1%). QAT with
fake-quant finetuning is the documented path to the <1% bar. Target per publisher:
v0.1 on COCO.

Pipeline:
  1. load pruned .pt (legacy-checkpoint repairs applied)
  2. modelopt INT8 fake-quant insertion; SENSITIVE layers stay unquantized
     (stem pair model.0/1, router paths, DFL) - mirrors the measured profile
  3. calibrate scales on >=512 spread-sampled TRAIN images (never val)
  4. QAT finetune on COCO train2017 (ultralytics trainer, amp off, small lr)
  5. export Q/DQ ONNX (head in export mode, no fuse - TRT folds BN itself)
  6. build TRT explicit-quantization engine (int8-qdq mode) + val + latency

Run on a GPU pod (PYTHONPATH=/data/YOLO-Master, /root/p03 venv):
  python scripts/project03/qat_moe.py \
    --model runs/project03/pod-prune/v01-n-coco/YOLO-Master-v0.1-N_pruned_t0.10.pt \
    --data /data/tmp/coco-qat.yaml --out runs/project03/qat/v01-n-coco \
    --calib-n 512 --epochs 3 --batch 64
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import quantize_trt as qt  # letterbox_blob, build_engine, eval/time helpers


SENSITIVE_SETS = {
    # implicit-PTQ-era full sensitive set (the original deliverable)
    "default": ("*model.0.*", "*model.1.*", "*routing*", "*router*", "*dfl*"),
    # explicit per-channel QDQ may not need most exclusions: fewer FP16 islands
    # = fewer reformats = faster engines. Sweep these for the latency target.
    "dfl": ("*dfl*",),
    "none": (),
}
DISABLE_PATTERNS = SENSITIVE_SETS["default"]


def spread_sample(dirpath: Path, n: int) -> list[str]:
    imgs = sorted(p for p in dirpath.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if len(imgs) <= n:
        return [str(p) for p in imgs]
    idx = np.linspace(0, len(imgs) - 1, n).astype(int)
    return [str(imgs[i]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="yaml with REAL train split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-n", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="train-set fraction (smoke tests)")
    ap.add_argument("--skip-train", action="store_true",
                    help="calibrate-only (PTQ-with-QDQ baseline)")
    ap.add_argument("--smoothquant", action="store_true",
                    help="use INT8_SMOOTHQUANT_CFG: migrate activation outliers into weights "
                         "(fixes systemic activation-outlier collapse at M/L scale)")
    ap.add_argument("--sensitive", choices=list(SENSITIVE_SETS), default="default",
                    help="base exclusion set (see SENSITIVE_SETS)")
    ap.add_argument("--disable", default="",
                    help="extra comma-separated quantizer-disable patterns, "
                         "e.g. '*experts*,*shared_expert*' for UoMoE")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import yaml as pyyaml
    from ultralytics import YOLO
    import modelopt.torch.quantization as mtq

    cfg_d = pyyaml.safe_load(Path(args.data).read_text())
    root = Path(cfg_d["path"])
    train_dir = root / cfg_d["train"]

    yolo = YOLO(args.model)
    # A3 rebase onto the RELEASED (non-pruned) weights (see quantize_trt.py)
    from diagnose_moe import _fix_add_residual, _force_dense_esmoe, _strip_property_shadows
    print(f"[repair] add_residual forced False on {_fix_add_residual(yolo)} blocks; "
          f"property shadows stripped {_strip_property_shadows(yolo)}; "
          f"ES_MOE dense={_force_dense_esmoe(yolo)}")
    net = yolo.model.cuda().eval()

    # legacy-pickle repair: pruned/old checkpoints predate several training-era
    # attributes on the MoE modules; the training forward path hard-requires them
    from ultralytics.nn.modules.moe.modules import MoELoss
    for _n, _m in net.named_modules():
        if hasattr(_m, "experts") and hasattr(_m, "routing"):
            if not hasattr(_m, "moe_loss_fn"):
                _m.moe_loss_fn = MoELoss(
                    balance_loss_coeff=1.0, z_loss_coeff=0.1,
                    num_experts=len(_m.experts),
                    top_k=int(getattr(_m, "_current_top_k", getattr(_m, "top_k", 2))))
                print(f"[repair] attached moe_loss_fn to {_n} "
                      f"(E={len(_m.experts)})")
            for _attr, _dflt in (("last_routing_snapshot", {}),
                                 ("expert_dropout_rate", 0.15),
                                 ("dropout_interval", 100)):
                if not hasattr(_m, _attr):
                    setattr(_m, _attr, _dflt)

    # ---- fake-quant insertion with the sensitive set disabled ----------------
    base_cfg = mtq.INT8_SMOOTHQUANT_CFG if args.smoothquant else mtq.INT8_DEFAULT_CFG
    print(f"[mtq] base config: {'INT8_SMOOTHQUANT' if args.smoothquant else 'INT8_DEFAULT'}")
    qcfg = copy.deepcopy(base_cfg)
    extra = tuple(p.strip() for p in args.disable.split(",") if p.strip())
    base = SENSITIVE_SETS[args.sensitive]
    print(f"[mtq] sensitive set '{args.sensitive}': {base}")
    for pat in base + extra:
        qcfg["quant_cfg"][pat] = {"enable": False}
    if extra:
        print(f"[mtq] extra disabled patterns: {extra}")

    calib = spread_sample(train_dir, args.calib_n)
    print(f"[calib] {len(calib)} train images from {train_dir}")

    def forward_loop(model):
        # Calibrate in EXPORT-EQUIVALENT mode. On the unpruned models (k < E) the eager
        # path routes sparsely, so experts never selected for any calibration image
        # get no activation statistics - but the exported graph computes ALL experts
        # (stack + gather), so every expert quantizer must be calibrated on every
        # input, exactly what the graph will see. Same mock the exporters use.
        from unittest import mock
        model.eval()
        with torch.no_grad(), mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
            for i in range(0, len(calib), 16):
                blobs = [qt.letterbox_blob(p, args.imgsz) for p in calib[i:i + 16]]
                x = torch.from_numpy(np.concatenate(blobs, 0)).cuda()
                model(x)

    def _quantize_now(model):
        t0 = time.time()
        mtq.quantize(model, qcfg, forward_loop)
        n_q = sum(1 for _, m in model.named_modules()
                  if type(m).__name__ == "TensorQuantizer" and m.is_enabled)
        # explicit coverage audit: a quantizer that saw no data cannot be exported;
        # never let that surface as a silent assert deep in the exporter
        uncal = [n for n, m in model.named_modules()
                 if type(m).__name__ == "TensorQuantizer" and m.is_enabled
                 and getattr(m, "amax", None) is None and not getattr(m, "_dynamic", False)]
        if uncal:
            print(f"[mtq] WARNING {len(uncal)} quantizers received NO calibration data "
                  f"(routed-expert coverage gap); disabling them (kept FP16): {uncal[:8]}...")
            for n, m in model.named_modules():
                if n in uncal:
                    m.disable()
            n_q -= len(uncal)
        print(f"[mtq] calibrated in {time.time() - t0:.0f}s, {n_q} active quantizers, "
              f"{len(uncal)} uncalibrated->disabled")
        return n_q

    # ---- QAT finetune --------------------------------------------------------
    if not args.skip_train:
        # TWO trainer traps. (1) get_model() rebuilds the net from its yaml and
        # intersect-loads weights: on a PRUNED checkpoint that resurrects the
        # unpruned architecture (7.55M params vs 3.33M) with random routers and
        # phantom experts. Patch it to trust the already-built module. (2) even
        # with (1), quantization must happen inside the trainer via callback so
        # nothing downstream swaps the module out from under the quantizers.
        from ultralytics.models.yolo.detect import DetectionTrainer
        _orig_get_model = DetectionTrainer.get_model

        def _get_model_trusting(self, cfg=None, weights=None, verbose=True):
            if weights is not None and isinstance(weights, torch.nn.Module):
                print(f"[qat] get_model: trusting loaded module as-is "
                      f"({sum(p.numel() for p in weights.parameters())/1e6:.2f}M params)")
                return weights
            return _orig_get_model(self, cfg=cfg, weights=weights, verbose=verbose)

        DetectionTrainer.get_model = _get_model_trusting
        # the FINAL epoch force-saves regardless of save=False, and modelopt's
        # dynamic classes cannot be pickled by torch.save - the save must not run
        DetectionTrainer.save_model = lambda self: None

        def _qat_cb(trainer):
            _quantize_now(trainer.model)
            from ultralytics.utils.torch_utils import ModelEMA
            trainer.ema = ModelEMA(trainer.model)   # EMA must wrap the quantized net
            print("[mtq] quantized inside trainer; EMA rebuilt")

        yolo.add_callback("on_pretrain_routine_end", _qat_cb)
        def _do_train():
            yolo.train(data=args.data, epochs=args.epochs, batch=args.batch,
                       imgsz=args.imgsz, lr0=args.lr0, lrf=0.1, optimizer="AdamW",
                       amp=False, workers=args.workers, fraction=args.fraction,
                       project=str(out), name="qat", exist_ok=True, plots=False,
                       val=True, save=False,
                       lora_r=0, lora_backend="fallback",
                       warmup_epochs=0.0, warmup_bias_lr=0.0,
                       moe_noise_std=0.0, moe_expert_warmup_epochs=0,
                       mosaic=0.0)
        try:
            _do_train()
        except Exception as _e:
            done = getattr(getattr(yolo, "trainer", None), "epoch", -1) + 1
            if done < args.epochs:
                raise
            # save_model is no-op'd (modelopt classes cannot pickle), so every
            # post-training step that touches best.pt/last.pt fails; all we need
            # is the in-memory trainer state
            print(f"[qat] ignoring trainer post-step failure after {done} epochs: {_e}")

                # light-touch augmentation for 3 epochs
        # export from in-memory weights (best.pt round-trips are untrustworthy).
        # Short QAT means the EMA may lag the raw weights: evaluate BOTH.
        candidates = {"ema": yolo.trainer.ema.ema, "raw": yolo.trainer.model}
        net = None   # chosen below after both are measured

    if args.skip_train:
        _quantize_now(net)
        mtq.print_quant_summary(net)
        candidates = {"model": net}

    n_final = sum(1 for _, m in list(candidates.values())[0].named_modules()
                  if type(m).__name__ == "TensorQuantizer" and m.is_enabled)
    if n_final == 0:
        raise RuntimeError("no active quantizers on the model about to be exported "
                           "- the trainer dropped them; export would be a plain fp32 graph")
    print(f"[mtq] exporting with {n_final} active quantizers "
          f"({len(candidates)} candidate weight sets)")

    # ---- Q/DQ ONNX export + engine + eval per candidate ----------------------
    # export on CPU: the TorchScript tracer segfaults tracing the fake-quant
    # graph on CUDA (torch 2.6 + modelopt 0.27); the CPU trace is clean
    import faulthandler
    faulthandler.enable()
    names = cfg_d["names"]
    qt.ULTRA_META.update({
        "description": "project03 QAT int8-qdq", "author": "project03",
        "version": "8.3.240", "task": "detect", "stride": 32, "batch": 1,
        "imgsz": [args.imgsz, args.imgsz],
        "names": {int(k): str(v) for k, v in names.items()},
    })
    results = {}
    for tag, cand in candidates.items():
        cand = cand.float().eval().cpu()
        head = cand.model[-1]
        head.export = True
        head.format = "onnx"
        dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
        onnx_path = out / f"qat_qdq_{tag}.onnx"
        torch.onnx.export(cand, dummy, str(onnx_path), opset_version=17,
                          input_names=["images"], output_names=["output0"],
                          do_constant_folding=True)
        print(f"[onnx] {onnx_path}")
        ep = out / f"model_int8-qdq_{tag}.engine"
        info = qt.build_engine(str(onnx_path), ep, "int8-qdq", args.imgsz)
        m50, mm = qt.eval_engine(ep, args.data, args.imgsz, 1, args.workers, 0, None)
        lat = qt.time_engine(ep, args.imgsz)
        print(f"[int8-qdq {tag}] {lat['lat_ms_median']}ms  mAP50 {m50:.4f}  "
              f"mAP50-95 {mm:.4f}  size {info['size_MB']}MB")
        results[tag] = {"mAP50": m50, "mAP50_95": mm, **lat,
                        "size_MB": info["size_MB"]}
    (out / "result.json").write_text(
        __import__("json").dumps({"epochs": args.epochs, "calib_n": len(calib),
                                  **results}, indent=1))


if __name__ == "__main__":
    main()
