#!/usr/bin/env python
"""Convert the v26.08 mixture targets (MoA / MoT / MoLoRA) to Core ML .mlpackage.

Runs in the dedicated py3.11 venv (/data/tmp/venv-coreml: torch 2.5.1 + coremltools 9.0
+ numpy<=2.3.5), NOT in venv-v2608: coremltools has no py3.14 wheels and its TorchScript
frontend breaks on torch 2.11+ (see coreml_export/README.md).

Weight identity: this script never re-seeds. It rebuilds each target's architecture from
the same cfg export_mixture.py used and loads the state_dict that export_mixture.py
--save-pt dumped from the ONNX-producing env, so the .mlpackage carries bit-identical
weights to the .onnx and the .ref.npz references.

Core ML blockers handled here, all edge-side (the fork worktree is never modified):
  1. MoA's window partition is rank 7 and MoT's is rank 6; Core ML tensors max out at
     rank 5. Both are replaced at trace time with bit-exact rank<=5 reshape/permute
     chains (self-tested against the originals before any tracing).
  2. Tracing runs under a mocked torch.onnx.is_in_onnx_export so the routed modules take
     their declared dense export branch, matching the semantics of the saved _raw
     references (same trick export_mixture.py uses for the reference forward).
  3. The traced wrapper returns only the detection tensor (raw[0] semantics), so MoT's
     internal (out, aux) tuples never surface as model outputs.

Known open risk (documented in ultralytics/cfg/export-capability-matrix.yaml): the MoT
deformable expert's F.grid_sample(align_corners=True). Conversion either maps it to MIL
resample or fails loudly per target; the batch continues either way.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# rank<=5 window partition rewrites (Core ML cannot represent rank-6/7 tensors)
# ---------------------------------------------------------------------------

def _moa_partition_r5(t: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B, nh, H, W, hd] -> [B*nh*nW, win*win, hd] without exceeding rank 5."""
    B, nh, height, width, hd = t.shape
    win = int(window_size)
    if win < 1 or height % win or width % win:
        raise ValueError(f"window_size={win} must evenly divide spatial shape {(height, width)}")
    t = t.reshape(B * nh, height // win, win, width, hd)
    t = t.permute(0, 1, 3, 2, 4)                                  # [BN, Hb, W, wh, hd]
    t = t.reshape(B * nh * (height // win), width // win, win, win, hd)
    t = t.permute(0, 1, 3, 2, 4)                                  # [G, Wb, wh, ww, hd]
    return t.reshape(-1, win * win, hd)


def _moa_unpartition_r5(windows, window_size, batch_size, num_heads, height, width):
    """Inverse of _moa_partition_r5 back to [B, nh, H, W, hd], rank<=5 throughout."""
    win = int(window_size)
    if win < 1 or height % win or width % win:
        raise ValueError(f"window_size={win} must evenly divide spatial shape {(height, width)}")
    hd = windows.shape[-1]
    g = batch_size * num_heads * (height // win)
    t = windows.reshape(g, width // win, win, win, hd)            # [G, Wb, wh, ww, hd]
    t = t.permute(0, 2, 1, 3, 4)                                  # [G, wh, Wb, ww, hd]
    t = t.reshape(batch_size * num_heads, height, width, hd)
    return t.reshape(batch_size, num_heads, height, width, hd)


def _mot_partition_r5(x: torch.Tensor, win: int) -> torch.Tensor:
    """[B, H, W, C] -> [B*nH*nW, win*win, C] without exceeding rank 5."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // win, win, W, C)
    x = x.permute(0, 1, 3, 2, 4)                                  # [B, Hb, W, wh, C]
    x = x.reshape(B * (H // win), W // win, win, win, C)
    x = x.permute(0, 1, 3, 2, 4)                                  # [G, Wb, wh, ww, C]
    return x.reshape(-1, win * win, C)


def _mot_reverse_r5(windows: torch.Tensor, win: int, H: int, W: int) -> torch.Tensor:
    """[B*nH*nW, win*win, C] -> [B, H, W, C], rank<=5 throughout."""
    windows_per_image = (H // win) * (W // win)
    B = windows.shape[0] // windows_per_image
    C = windows.shape[2]
    t = windows.reshape(B * (H // win), W // win, win, win, C)    # [G, Wb, wh, ww, C]
    t = t.permute(0, 2, 1, 3, 4)                                  # [G, wh, Wb, ww, C]
    return t.reshape(B, H, W, C)


def _roll_via_slice(x: torch.Tensor, shift: int, dims: tuple) -> torch.Tensor:
    """Cyclic shift via python slicing (aten::slice) instead of aten::narrow.

    coremltools 9.0's narrow handler dies on the frozen MoT graph ("inhomogeneous
    shape" while building the begin const); the slice handler converts the identical
    computation cleanly. Shapes are static under our trace, so int() is exact.
    """
    if shift == 0:
        return x
    for dim in dims:
        n = int(x.shape[dim])
        s = shift % n
        if s == 0:
            continue
        if dim == 1:
            x = torch.cat([x[:, n - s:], x[:, :n - s]], dim=1)
        elif dim == 2:
            x = torch.cat([x[:, :, n - s:], x[:, :, :n - s]], dim=2)
        else:
            raise ValueError(f"_roll_via_slice supports dims 1/2, got {dim}")
    return x


def _validate_router_input_shape_only(x, expected_channels: int, context: str = "") -> None:
    """Trace-safe replacement for moe.routers._validate_router_input.

    The original's isnan().any()/isinf().any() guards are eager runtime checks, but
    tracing records the tensor computations anyway: in ONNX they surface as the 9
    IsInf/IsNaN nodes the MNN lowering rewrites, and in CoreML they become rank-0
    non_zero ops that the macOS model compiler rejects ("invalid rank 0"). The python
    shape checks are kept; the tensor-valued checks are dropped for conversion only.
    """
    from ultralytics.nn.modules.moe.routers import MoERouterError, ShapeMismatchError
    if x.dim() != 4:
        raise MoERouterError(f"Router input must be 4-D (NCHW), got {x.dim()}-D shape {tuple(x.shape)}"
                             + (f" [{context}]" if context else ""))
    if expected_channels > 0 and x.shape[1] != expected_channels:
        raise ShapeMismatchError(expected=f"(N, {expected_channels}, H, W)", actual=tuple(x.shape),
                                 context=context or "router input")


def selftest_and_patch():
    """Prove the rank<=5 rewrites bit-exact against the originals, then install them."""
    from ultralytics.nn.modules.moa import heads as moa_heads
    from ultralytics.nn.modules.mot import experts as mot_experts

    torch.manual_seed(7)
    for B, nh, H, W, hd, win in ((1, 3, 14, 14, 16, 7), (2, 2, 28, 21, 8, 7),
                                 (1, 1, 8, 12, 4, 4), (3, 4, 21, 35, 32, 7)):
        t = torch.randn(B, nh, H, W, hd)
        ref = moa_heads._window_partition_2d(t, win)
        got = _moa_partition_r5(t, win)
        assert torch.equal(ref, got), f"moa partition mismatch B={B} nh={nh} H={H} W={W}"
        back_ref = moa_heads._window_unpartition_2d(ref, win, B, nh, H, W)
        back_got = _moa_unpartition_r5(got, win, B, nh, H, W)
        assert torch.equal(back_ref, back_got) and torch.equal(back_ref, t)
    for B, H, W, C, win in ((1, 14, 14, 24, 7), (2, 28, 21, 16, 7), (1, 12, 8, 8, 4)):
        x = torch.randn(B, H, W, C)
        ref = mot_experts._WindowTransformerExpert._window_partition(x, win)
        got = _mot_partition_r5(x, win)
        assert torch.equal(ref, got), f"mot partition mismatch B={B} H={H} W={W}"
        back_ref = mot_experts._WindowTransformerExpert._window_reverse(ref, win, H, W)
        back_got = _mot_reverse_r5(got, win, H, W)
        assert torch.equal(back_ref, back_got) and torch.equal(back_ref, x)

    torch.manual_seed(11)
    for shape, shift, dims in (((2, 14, 14, 8), -3, (1, 2)), ((1, 21, 28, 4), 3, (1, 2)),
                               ((1, 10, 10, 4), -7, (1, 2))):
        x = torch.randn(*shape)
        assert torch.equal(mot_experts._roll_via_cat(x, shift, dims), _roll_via_slice(x, shift, dims))

    moa_heads._window_partition_2d = _moa_partition_r5
    moa_heads._window_unpartition_2d = _moa_unpartition_r5
    mot_experts._WindowTransformerExpert._window_partition = staticmethod(_mot_partition_r5)
    mot_experts._WindowTransformerExpert._window_reverse = staticmethod(_mot_reverse_r5)
    mot_experts._roll_via_cat = _roll_via_slice

    from ultralytics.nn.modules.moe import routers as moe_routers
    moe_routers._validate_router_input = _validate_router_input_shape_only
    print("[patch] rank<=5 window rewrites + slice-roll verified bit-exact and installed; "
          "router NaN/Inf guards dropped (trace-dead, CoreML-incompatible)")


# ---------------------------------------------------------------------------
# per-target conversion
# ---------------------------------------------------------------------------

class _DetOnly(nn.Module):
    """Return only the detection tensor (matches the raw[0] reference semantics)."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        y = self.m(x)
        while not torch.is_tensor(y):
            y = y[0]
        return y


def mil_op_census(mlmodel) -> Counter:
    """Count MIL op types across all functions/blocks of a saved mlprogram spec."""
    ops = Counter()

    def walk_block(block):
        for op in block.operations:
            ops[op.type] += 1
            for blocks in op.blocks:
                walk_block(blocks)

    prog = mlmodel.get_spec().mlProgram
    for fn in prog.functions.values():
        for block in fn.block_specializations.values():
            walk_block(block)
    return ops


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="YOLO-Master v26.08 checkout (prepended to sys.path)")
    ap.add_argument("--out", default="models/mixture", help="dir holding <name>.pt inputs and .mlpackage outputs")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--only", default="", help="comma-separated subset of target names")
    ap.add_argument("--target", default="macos13", choices=["macos13", "macos14", "macos15"])
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import coremltools as ct
    from ultralytics import YOLO
    from ultralytics import __version__

    print(f"[env] ultralytics {__version__} torch {torch.__version__} coremltools {ct.__version__}")
    assert not __version__.startswith("8.3."), "old-lineage ultralytics resolved - wrong sys.path"

    selftest_and_patch()

    out = Path(args.out)
    V010 = "ultralytics/cfg/models/master/v0_10/det"
    Y26 = "ultralytics/cfg/models/26"

    def build(cfg):
        return YOLO(str(Path(args.repo) / cfg))

    def build_molora(merged):
        from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel
        y = build(f"{V010}/yolo-master-moa-n.yaml")
        mol = MoLoRAModel(y.model, MoLoRAConfig(r=8, alpha=16, num_experts=4, top_k=2))
        if merged:
            mol.merge(mode="uniform")     # sets the merged flags; load_state_dict then
        y.model = mol.model               # overwrites every tensor with the saved values
        return y

    # name -> (factory, end2end_export_flag or None meaning "leave as built")
    TARGETS = {
        "mot-n":                (lambda: build(f"{V010}/yolo-master-mot-n.yaml"), None),
        "moa-n":                (lambda: build(f"{V010}/yolo-master-moa-n.yaml"), None),
        "moa-mot-n":            (lambda: build(f"{V010}/yolo-master-moa-mot-n.yaml"), None),
        "yolo26-moa-n-e2e":     (lambda: build(f"{Y26}/yolo26-master-moa-n.yaml"), None),
        "yolo26-moa-n-anchors": (lambda: build(f"{Y26}/yolo26-master-moa-n.yaml"), False),
        "molora-merged":        (lambda: build_molora(True), None),
        "molora-routed":        (lambda: build_molora(False), None),
    }
    only = {s for s in args.only.split(",") if s} or set(TARGETS)
    deploy = {"macos13": ct.target.macOS13, "macos14": ct.target.macOS14,
              "macos15": ct.target.macOS15}[args.target]

    results = {}
    for name, (factory, e2e_flag) in TARGETS.items():
        if name not in only:
            continue
        print(f"\n=== {name} ===")
        try:
            y = factory()
            sd = torch.load(out / f"{name}.pt", map_location="cpu", weights_only=True)
            y.model.load_state_dict(sd, strict=True)
            model = y.model.eval()
            if e2e_flag is False:
                model.model[-1].end2end = False   # anchors variant: match the ONNX export
            for p in model.parameters():
                p.requires_grad_(False)

            wrapped = _DetOnly(model)
            ex = torch.zeros(1, 3, args.imgsz, args.imgsz)
            with torch.no_grad():
                raw = wrapped(ex)                 # eager warmup bakes static spatial dims
            print(f"  det output {list(raw.shape)}")

            # nan_to_num on the rank-0 complexity scalar (gated.py _apply_complexity_gate)
            # lowers to non_zero + scatter_nd chains that the macOS compiler rejects
            # ("non_zero: invalid rank 0"). Identity is numerically exact on finite
            # values, which is all inference ever produces here; the ONNX/MNN exports
            # keep their guards.
            with mock.patch("torch.onnx.is_in_onnx_export", return_value=True), \
                 mock.patch("torch.nan_to_num",
                            lambda x, nan=0.0, posinf=None, neginf=None, out=None: x):
                traced = torch.jit.trace(wrapped, ex, strict=False, check_trace=False)
            traced = torch.jit.freeze(traced.eval())
            try:
                torch.jit.run_frozen_optimizations(traced)
            except Exception:
                pass

            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="images", shape=(1, 3, args.imgsz, args.imgsz), dtype=float)],
                minimum_deployment_target=deploy,
                compute_units=ct.ComputeUnit.ALL,
                convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT32,   # parity-grade; mlprogram defaults to fp16
            )

            outs = [(o.name, list(o.type.multiArrayType.shape))
                    for o in mlmodel.get_spec().description.output]
            assert len(outs) == 1, f"expected a single det output, got {outs}"
            names = y.model.names if isinstance(y.model.names, dict) else dict(enumerate(y.model.names))
            e2e = bool(getattr(model.model[-1], "end2end", False)) and e2e_flag is not False
            meta = mlmodel.user_defined_metadata
            meta["task"] = "detect"
            meta["output"] = outs[0][0]
            meta["imgsz"] = str(args.imgsz)
            meta["names"] = ",".join(str(names[k]) for k in sorted(names))
            meta["end2end"] = str(e2e).lower()

            dst = out / f"{name}.mlpackage"
            mlmodel.save(str(dst))

            census = mil_op_census(ct.models.MLModel(str(dst), skip_model_load=True))
            top = ", ".join(f"{op}:{n}" for op, n in census.most_common(8))
            notable = {op: census[op] for op in ("resample", "topk", "one_hot", "grid_sample")
                       if census.get(op)}
            print(f"  saved {dst.name}  output {outs[0]}  end2end={e2e}")
            print(f"  MIL ops: {sum(census.values())} total, {len(census)} types | top: {top}")
            if notable:
                print(f"  notable ops: {notable}")
            results[name] = f"OK  {outs[0][1]}  ops={sum(census.values())}"
        except Exception as e:
            traceback.print_exc()
            results[name] = f"FAILED: {type(e).__name__}: {str(e)[:180]}"

    print("\n===== coreml export summary =====")
    for name, r in results.items():
        print(f"  {name:24s} {r}")
    if any(r.startswith("FAILED") for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
