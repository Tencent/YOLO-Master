# Core ML export

`export_coreml.py` converts an Ultralytics / YOLO-Master `.pt` checkpoint (detector **or**
segmenter) into a Core ML `.mlpackage` carrying the exact metadata the [macOS app](../mac/) reads
(`names`, `imgsz`, `output` tensor, `task`, and — for segmentation — `proto`/`nm`). It emits an
mlprogram and validates the class count against the output shape.

Conversion runs on **Linux** (coremltools; only *prediction* needs macOS).

## Environment

```bash
conda create -n cmlexport python=3.11 -y
pip install torch==2.5.1 torchvision==0.20.1 coremltools==9.0
```

Then install the `ultralytics` build that **matches the checkpoint** — mixing them fails at trace time:

| Checkpoint | ultralytics |
| :--------- | :---------- |
| YOLO-Master (v0.1 / EsMoE / UoMoE, incl. P2, seg) | the fork: `pip install -e /path/to/YOLO-Master --no-deps` |
| stock YOLO (yolo11, …) | `pip install ultralytics` |
| sunsmarterjie/yolov12 | stock ultralytics **+** `--yolov12-aattn` (see below) |

`torch 2.11` breaks coremltools' torch frontend (`aten::Int`) — pin **torch 2.5**.

## Usage

```bash
# detector / segmenter (task auto-detected from the output count)
python export_coreml.py --weights model.pt --imgsz 640 --out model.mlpackage

# yolov12 authors' checkpoints (split qk+v area-attention)
python export_coreml.py --weights yolov12x.pt --imgsz 640 --out yolov12x.mlpackage --yolov12-aattn

# a LoRA-fine-tuned model: merge the trained adapters, then export
python export_coreml.py --weights base.pt --merge-lora-dir lora_adapter/ --imgsz 640 --out ft.mlpackage
```

## What the script handles (and why)

- **YOLO-Master MoE** — forces the dense path (`is_in_onnx_export=True`), constant-folds shapes
  (`jit.freeze` + `run_frozen_optimizations`, fixing the dynamic `aten::Int`), and no-ops EsMoE's
  in-place aux-loss telemetry (fixing `aten::copy_` "No matching select or slice"). All no-ops for plain models.
- **Segmentation** — detects the 2-output signature and writes `task=segment` + `proto`/`nm`.
- **Area attention (YOLOv12)** — an eager warmup bakes each layer's concrete spatial dims so the
  area reshapes fold to static shapes; `--yolov12-aattn` swaps in the qk+v AAttn the authors' weights expect.
- **LoRA** — `--merge-lora-dir` merges trained adapters before export (a merged LoRA is a static graph;
  routed MoLoRA cannot be traced). Note: `apply_lora` may re-initialize the detection head — restore it
  from the base before merging if you apply LoRA fresh.

## Precision

`ct.convert(..., convert_to="mlprogram")` emits fp16 weights and fp16 activations (coremltools'
default `compute_precision` for ML programs); the packages here are fp16 already, no separate
step (v0.1-N: 7.55 M parameters, 15.5 MB on disk; yolov12x: 59.4 M, 119 MB).

`quantize_w8a16.py in.mlpackage out.mlpackage` turns an existing package into W8A16 (int8
per-channel weights, fp16 activations; the Project03 recipe). yolov12x goes 119 -> 60 MB.

## Neural Engine and the MoE router (`--no-ane-safe-moe` to opt out)

The stock export of `OptimizedMOEImproved` (the v0.1 family) and `OptimizedMOE` keeps the router's
`torch.topk` and a `torch.gather` over the stacked expert outputs in the graph: MIL `topk`,
`gather_along_axis` and int32 index tensors. The ANE has no kernels for those, so under
`cpuAndNeuralEngine` Core ML runs every router on the CPU with a sync and a copy on both sides
of each MoE block (v0.1-N: 9.2 ms on the ANE against 4.1 ms for the router-free v0.1-seg-N graph
on an M4 Max, while the GPU ran it in 4.3 ms).

By default the exporter now writes the same top-k mixture with arithmetic only:
`rank_i = sum_j step(p_j - p_i)`, `mask_i = step(k - 0.5 - rank_i)`, `w = p * mask / sum(p * mask)`,
`y = shared(x) + sum_i w_i * expert_i(x)`, with `step(z) = clamp(32768 z, 0, 1)`. The graph has no
integer op left (`moe_export: ane_safe_rank` in the metadata); eager fp32 outputs agree with the
stock path to 3e-4 over a 0..637 output range on the v0.1-N checkpoint, and two experts whose
pooled probabilities are within 3e-5 of each other (an fp16 tie) get a proportional blend instead
of an arbitrary winner.

Note what the export cannot change: a static Core ML graph computes EVERY expert and masks the
mixture, so v0.1-N runs its 4 + 8 + 16 experts on every frame. The sparse top-2 saving exists
only in the PyTorch eager path.

`compute_plan.py model.mlpackage --units ane` (macOS 14.4+, coremltools 8+) prints which device
Core ML gives each op and lists the ops the ANE cannot take, with their estimated cost share.
It is the text form of Xcode's performance report and the way to check whether an "ANE" number
from the Bench is the Neural Engine or a CPU fallback.
