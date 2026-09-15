# Project03 phase 2: TensorRT INT8 on the pruned checkpoints (MEASURED)

Derived from the working results log of yolo-master-edge branch `dev/project03`. Raw ladders, sensitivity reports, pin sets, bisect/ablation tables and QAT result files are in the sibling directories `trt/` and `qat/` (engines and calibration caches are not included; rebuild with `scripts/project03/quantize_trt.py`).

All numbers measured on A100-SXM4-80GB pods, TensorRT 10.13.3.9, batch 1, 640x640,
pruned t0.10 checkpoints (~56% param cut from phase 1), full val sets, ultralytics
val protocol. Latency = median of 200 engine executions, model-only.

Criterion readings (competition bar): INT8 mAP loss < 1.0 AP point; FP16 latency
cut >= 25% vs FP32; target model per publisher: v0.1 on COCO.

## 1. Headline: criterion status

| Criterion | Result | Status |
|---|---|---|
| Params -30%+ | -56.2% (v0.1-N), -56.9% (UoMoE-N), mAP delta ~0 | PASS (phase 1) |
| FP16 latency >= 25% cut | pure-FP32 3.331ms -> fp16 2.115ms = -36.5%, mAP unchanged | PASS |
| INT8 < 1.0 AP point | explicit Q/DQ calibrate-only: -0.80 AP (v0.1-N COCO), re-validated 0.4206 exactly | PASS |
| INT8 QAT (margin attempt) | regressed (-3.2/-3.3 AP); negative result, retry recipe documented | banked |
| Android < 100ms real-time | phase 3 | pending |

The FP16 criterion requires the TF32-disabled baseline: TRT silently enables TF32
on Ampere, which flatters "FP32" to 2.65ms; the honest unoptimized-FP32 baseline
is 3.331ms.

## 2. The INT8 investigation, in the order it actually happened

### 2.1 Implicit PTQ (TRT entropy calibrator) collapses

v0.1-N COCO, fp32 = 0.4286 mAP50-95. "Surgical" = head + router subgraphs +
attention cores pinned FP16 (OBEY_PRECISION_CONSTRAINTS), trunk INT8.

| Calibration | surgical | + stem-pair recipe |
|---|---|---|
| 256 val images (naive protocol) | 0.2995 (-12.9 pts) | - |
| 1024 train images | 0.3756 (-5.3 pts) | 0.3959 (-3.3 pts) |
| 2048 train images | 0.4044 (-2.4 pts) | 0.4055 (-2.3 pts) |
| 4096 train images | 0.4038 (-2.5 pts) | 0.4058 (-2.3 pts) |

Findings:
- Calibration coverage dominates up to ~2048 images (+7.6 AP from coverage alone).
  256 images cannot cover COCO's activation diversity. Val-set calibration is also
  leakage; all corrected runs calibrate on train2017.
- Coverage saturates at ~2048: the residual -2.3 pts is the implicit per-tensor
  method's floor, not a data problem.
- The stem-pair recipe (modules 0-1 FP16) is worth +2.0 AP at weak calibration but
  fades to +0.1 as calibration improves - good calibration subsumes the stem's
  range problem.

### 2.2 Explicit quantization (Q/DQ) steps under the floor

modelopt 0.27.1 fake-quant insertion -> calibrate -> ONNX with Q/DQ nodes -> TRT
explicit INT8 engine. Sensitive set excluded from quantization: stem pair
(model.0/1), router paths, DFL. Per-channel weight scales are the key advantage
over implicit per-tensor calibration.

| Model (COCO) | fp32 | explicit Q/DQ, calibrate-only | AP points lost |
|---|---|---|---|
| v0.1-N pruned | 0.4286 | 0.4206 (64 imgs!) | 0.80 PASS |
| UoMoE-N pruned | 0.4127 | 0.3534 (1024 imgs) | 5.93 fail |
| UoMoE-N pruned, experts excluded | 0.4127 | 0.3988 (1024 imgs) | 1.39 |

Coverage of the v0.1-N deliverable, verified from the quantizer summary and the
engine audit (176 Int8 layers, the widest coverage of any engine built): attention
QKV + output projections INT8 (32/32 entries enabled), head cv2/cv3 towers INT8
(48/48; only the DFL is excluded, and the final mixed-semantics concats are
functional ops that are never quantized), all expert convs INT8 (the disabled
expert entries are BatchNorms, which fold into convs at build). Excluded
(sensitive set): stem pair, router paths, DFL.

UoMoE divergence: implicit TRT could never actually quantize MoE expert convs
(the routing gather/expand chain breaks the int8 path - they silently ran fp16),
so implicit PTQ never exposed them. Explicit Q/DQ quantizes them for real, and
UoMoE's experts carry 200-336x channel range disparities (expert specialization).
Excluding experts + shared_expert recovers 4.5 of 5.9 pts. v0.1-N's spatial-MoE
experts tolerate quantization.

### 2.3 QAT (v0.1-N COCO, the criterion run)

3-epoch full-model finetune on train2017 through fake quant, 1024-image
calibration init, AdamW lr 1e-4, amp off. Fork-trainer traps that had to be
disabled for a short finetune of a pretrained model: unconditional LoRA injection
(lora_r=0 is the off switch), 3-epoch warmup with 0.1 bias-lr, MoE routing noise
(moe_noise_std=0.5), expert-warmup routing scramble.

| Engine | mAP50-95 | AP points vs fp32 | model-only latency |
|---|---|---|---|
| explicit Q/DQ calibrate-only (the deliverable) | 0.4206 | -0.80 | 2.359ms |
| QAT 3-epoch, EMA weights | 0.3969 | -3.17 | 2.346ms |
| QAT 3-epoch, raw weights | 0.3954 | -3.32 | 2.333ms |

NEGATIVE RESULT: 3 epochs at lr 1e-4 against amax scales frozen at initialization
walked the weights ~2.4 AP away from their own PTQ starting point (consistent
across EMA and raw weight sets, so not EMA lag). At short-finetune budgets,
calibrate-only explicit PTQ beats QAT. Documented retry recipe if margin is ever
needed: save quantized state_dicts per epoch (plain tensors pickle fine), lr
~1e-5, single epoch, and a post-training re-calibration pass to re-sync
activation scales to the final weights.

## 3. Where the AP goes: localization studies (AI-TOD-v2, UoMoE-N)

- Module bisection: damage is distributed (backbone-half pinned recovers to
  -1.06 pts, neck-half to -1.98; additive, no single culprit).
- Leave-one-out ablation vs all-pinned floor: damage decays monotonically with
  depth. Stem 0.0126, first downsample 0.0107 (together ~83% of backbone damage),
  first attention block 0.0038, everything else <=0.003, MoE blocks ~0
  (unquantizable under implicit TRT anyway).
- Recovery is sub-additive: pinning only the top modules recovers less than the
  ablation predicts - distributed noise compounds. PTQ cannot reach <1 pt on this
  architecture at meaningful INT8 coverage under IMPLICIT quantization; explicit
  Q/DQ is the method that gets there.
- Cross-dataset surgical PTQ (256-cal, superseded but instructive): UoMoE-N
  AI-TOD -14.6% rel / COCO -30.7% / VisDrone -48.5%; v0.1-N COCO -30.1% /
  VisDrone -33.6%. Neither UoMoE-specific nor tiny-object-driven; and AI-TOD-v2,
  the "extreme" dataset, actually has the SMALLEST absolute INT8 gap (~2.9 pts).

## 3.5 Model-only vs end-to-end latency (A100, 300 val images, conf 0.25)

| Engine | model-only | e2e/frame | pre | infer+copies | NMS | mAP50-95 (reval) |
|---|---|---|---|---|---|---|
| fp32 | 2.422ms | 15.20ms | 8.80 | 3.29 | 2.37 | 0.42859 |
| Q/DQ calibrate-only | 2.359ms | 12.16ms | 8.23 | 3.20 | 0.87 | 0.42060 |
| QAT ema | 2.346ms | 11.84ms | 8.03 | 3.17 | 0.79 | 0.39693 |
| QAT raw | 2.333ms | 12.25ms | 8.40 | 3.16 | 0.81 | 0.39543 |

End-to-end is PREPROCESS-BOUND on this host: letterboxing (~8ms, CPU) dwarfs
inference (~3.2ms). Precision changes barely move e2e (-20% comes mostly from
cheaper NMS: int8 engines emit fewer above-threshold candidates). Deployment
lesson: at these model speeds the pipeline is the bottleneck - GPU letterbox /
fused preprocessing matters more than further model quantization on server-class
hardware. All mAP numbers reproduced exactly on independent re-validation.

## 3.6 L4 ladder: the deployment-class verdict (on-device builds)

| Engine | model-only | cut vs honest fp32 | e2e/frame | mAP50-95 |
|---|---|---|---|---|
| fp32 (TF32 off) | 3.047ms | ref | 17.2ms | 0.42867 |
| fp32 + TF32 | 2.422ms | 20.5% | 11.9ms | 0.42856 |
| fp16 | 1.607ms | 47.3% | 11.1ms | 0.42857 |
| mixed-INT8 (Q/DQ deliverable) | 1.896ms | 37.8% | 11.4ms | 0.42102 |

Even on Ada (proper INT8 tensor cores), fp16 beats the mixed-INT8 engine: Q/DQ
reformats plus the FP16 sensitive-set islands bound INT8's payoff, exactly as
the earlier Orin finding predicted. The INT8 accuracy pass is architecture-
portable (0.42102 L4 vs 0.42060 A100 from the same ONNX, -0.77 pts, under the
1.0 bar on both). FP16 is the latency-optimal deployment precision across every
GPU measured; the mixed-INT8 engine is the accuracy-criterion artifact and the
path for INT8-only backends.

## 3.7 The INT8-vs-FP16 endgame: Orin, ceiling proof, and mechanism

Orin Nano (10W, on-device trtexec builds, GPU compute median):
| engine | latency | vs fp32 |
|---|---|---|
| fp32 | 37.62ms | ref |
| fp16 | 21.87ms | -41.9% (criterion PASS on-target, 45.7 FPS) |
| int8-qdq (deliverable) | 24.77ms | -34.2%, SLOWER than fp16 |
| int8 ceiling (uncalibrated implicit, max coverage) | 23.00ms | still slower than fp16 |

The ceiling row is the decisive experiment: with TRT free to quantize every
layer and zero Q/DQ constraints, INT8 still loses to FP16. No placement, QAT,
or coverage fix can beat a ceiling. Cross-checked on L4: fp16 1.607ms (opt5:
1.560) vs best int8 1.834; island-removal variants (stem/router re-included)
gain <=0.06ms and lose 0.7-1.2 AP; ONNX-level quantization of the attention
matmuls (attnq) is slower still (1.997ms) AND collapses accuracy to 0.004.

Mechanism (per-layer profiling, L4): reformats are only ~25% of the int8
deficit (0.184 vs 0.087ms). The dominant costs are (1) unfolded weight-Q/DQ:
TRT executes weight QuantizeLinear at RUNTIME in every conv from the modelopt
export, and (2) the float attention region (~20% of runtime) plus elementwise
traffic that no precision helps. Root cause of "no headroom": nano channel
widths (16-64, odd post-pruning counts) underfill int8 IMMA tiles, so int8
GEMMs are not faster than fp16 tensor-core GEMMs at these shapes.

VERDICT: FP16 is the deployment precision on every TRT GPU (A100, L4, Orin
Nano). INT8's latency wins live on int8-native backends: ARM CPU dot-product
(ncnn, phase 3) and NPUs. Future lever if GPU INT8 must pay: alignment-aware
pruning (keep channel counts %32 for IMMA tile fill).

## 3.8 The INT8 scaling game (v0.1 S/M/L, pruned, A100)

Pruning first (all scales converge to 2/2/2 experts, all free):
S 29.2->12.35M (-57.7%, +0.00002 mAP) | M 52.2->29.07M (-44.3%, -0.00015)
| L 58.5->35.32M (-39.6%, +0.00008). Expert redundancy is scale-invariant.

Precision ladder (fp32 = TF32-off honest baseline; QDQ rows use the
bisect-derived per-model protection sets):

| scale | fp32 | fp16 | int8-naive (vs fp16) | int8-QDQ | QDQ AP pts |
|---|---|---|---|---|---|
| N (ref, L4) | 3.047ms | 1.607ms | slower | 1.896ms / 0.4206 | -0.80 PASS |
| S | 5.128ms | 2.354ms | 2.224ms (-5.5%) | 2.747ms / 0.4840 | -0.46 PASS |
| M | 10.055ms | 3.093ms | 2.862ms (-7.5%) | 3.347ms / 0.5118 | -1.59 |
| L | 13.261ms | 4.699ms | 4.551ms (-3.1%) | 5.056ms / 0.5272 | -1.31 |

Findings:
1. The GPU INT8 speed crossover is REAL from S up: implicit int8 beats fp16 at
   every scale >= S, peaking at M. The gain tracks WIDTH, not parameter count
   (L = M-width + depth; the extra depth adds unquantizable elementwise or
   attention share and dilutes the win to -3.1%). This vindicates the IMMA
   tile-fill theory against the N-scale verdict.
2. Accurate INT8 still loses to fp16 at every scale: the QDQ toolchain
   overhead (unfolded runtime weight-quant + reformats, ~0.4-0.5ms) exceeds
   the naive win everywhere. Fold the weight-Q at graph level and S/M would
   flip; documented as the highest-value toolchain fix.
3. Sensitive sets are PER-MODEL, not per-architecture: N/S need the stem pair;
   M's epicenter is modules {4,5,6}; L's is {7,8,9} (mid-backbone attn/MoE
   region). Discovered via a torch-side fake-quant protection bisect
   (~8 rounds x 3min, no engine builds; note: AutoBackend fuse() corrupts
   fake-quant models during in-torch val and must be no-op'd).
4. QDQ accuracy passes the <1 pt bar at N and S; M (-1.59) and L (-1.31) need
   layer-granularity protection or QAT-with-recalibration to close - documented
   as the path, not run (diminishing returns for a side study).

## 3.9 Orin Nano scaling ladder: INT8's full win (on-device builds, 10W)

| scale | fp32 | fp16 | int8-naive | int8-QDQ | QDQ vs fp16 |
|---|---|---|---|---|---|
| N | 37.73ms | 21.81 | 23.11 | 24.74 | +13% |
| S | 75.32 | 39.12 | 39.39 | 37.03 | -5.4% (accuracy PASS -0.46) |
| M | 147.39 | 71.24 | 68.65 | 66.49 | -6.7% (-1.59 AP) |
| L | 208.91 | 104.13 | 110.89 | 96.62 | -7.2% (-1.31 AP) |

EXPLICIT INT8 BEATS FP16 AT EVERY SCALE ABOVE N ON THE ORIN, margin growing
with scale (5.4 -> 6.7 -> 7.2%). The QDQ toolchain overhead that drowned
2ms-class datacenter engines amortizes to nothing at 37-100ms latencies. At L,
QDQ even beats naive by 12.9%: the protected fp16 islands + per-channel scales
give TRT a better graph than quantize-everything (naive loses to fp16 at L).
N stays the exception - at 22ms the fixed overhead still bites. M/L QDQ
builds need the swapfile on a 4GB Nano (jetson/11_p03_trt_bench.sh provisions
/swapfile-p03 automatically).

Orin deployment menu: N-fp16 21.8ms/46FPS (fastest), or S-int8-QDQ
37.0ms/27FPS at +5.5 AP over N - the accuracy-per-latency option that exists
only because of the INT8 program. M/L QDQ are latency trophies pending the
layer-bisect/QAT accuracy path (-1.59/-1.31 vs the 1.0 bar).

## 4. Latency notes (A100, batch 1)

- Every INT8 engine (2.02-2.36ms) is slower than or equal to plain FP16 (2.107ms).
  INT8 pays on INT8-tensor-core-bound hardware at higher occupancy; the A100
  batch-1 regime is launch-bound. The INT8 case must be made on Orin/Android.
- Q/DQ explicit engines carry extra reformat cost (2.36ms vs 2.04ms implicit).
- Stem-pair FP16 pin costs <=40us (~2%, within noise) for ~0.9% of FLOPs - the
  cheapest accuracy insurance in the network.
- TF32 flatters FP32 baselines on Ampere; disable it for criterion-honest
  comparisons.

## 5. Toolchain traps (baked into scripts/project03/quantize_trt.py)

1. PREFER_PRECISION_CONSTRAINTS silently discards every FP16 pin when a
   calibrator provides scales; OBEY + per-layer output-type constraints
   (skipping Cast/Identity) is mandatory, and pins that split fused SiLU groups
   make OBEY builds infeasible (survive-and-revert logic needed).
2. The engine inspector (DETAILED profiling verbosity) is the only ground truth
   for what precision layers actually run in.
3. Ultralytics .val() needs the 4-byte-header engine container format.
4. TRT 11 removed the implicit-quant API entirely (downgraded to 10.13.3.9).
5. cuda-python cudaMalloc returns null on some pods; torch-backed device memory
   is the portable path.
6. torch.onnx.export segfaults tracing modelopt fake-quant graphs on CUDA;
   CPU-side tracing is clean.
7. Old-lineage trainer: unconditional apply_lora (disable via lora_r=0), polars
   dependency at first results-write, PEFT demand from lora_backend=auto.

## 6. Recommended deployment recipes

- v0.1-N (target): explicit Q/DQ INT8, sensitive set excluded (stem pair,
  routers, DFL), >=1024 train-image calibration; QAT finetune for margin.
  FP16 engine as the latency-optimal GPU variant.
- UoMoE-N: same, plus exclude experts + shared_expert (1.39 pts calibrate-only;
  QAT to close). FP16 remains the safe default.
- Never calibrate implicit PTQ with <1024 images or with val data.

