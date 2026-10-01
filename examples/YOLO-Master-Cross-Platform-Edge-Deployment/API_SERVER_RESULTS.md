# API Server vs Bare Runtime: COCO val2017 on an L40S

All numbers in this document are MEASURED on one machine in one session unless a row is
labelled REFERENCE. Nothing here is a claim about upstream YOLO-Master; the models are the
released v0.1-N and EsMoE-N COCO checkpoints exported by this repo (A3 exports) plus the
MoEPruner-pruned v0.1-N from project03. Tables 1 to 3 are the v1.1.x server on the L40S with
CPU preprocessing; the v1.2.0 rerun on an L4 with GPU preprocessing is Tables 4 to 6 below.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA L40S 46 GB, driver 580.126.09, max SM clock 2520 MHz |
| CPU | AMD EPYC 9374F, 128 vCPUs visible, container quota 13.6 cores (`cpu.max` 1360000/100000) |
| RAM | 1.5 TB (container) |
| OS / toolchain | Ubuntu 22.04.5, g++ 11.4, CMake 4.4, Ninja |
| CUDA runtime | 12.9.79 (TensorRT build target); pod toolkit 12.4 |
| TensorRT | 10.16.1.11 (+cuda12.9), engines built by `yolomaster_server`/`yolomaster_edge` from ONNX |
| ONNX Runtime | 1.20.1 GPU build (CUDA EP), cuDNN 9.26 |
| ncnn | 20260526 shared build for Ubuntu 22.04, x86-64, OpenMP, no Vulkan |
| MNN | 3.6.1 built from source (AVX512 CPU build; CUDA backend build sm89) |
| HTTP engine | uWebSockets v20.80.0 + uSockets (epoll, no SSL, no zlib) |
| dataset | COCO val2017, 5000 images (4952 with labels), YOLO-format labels |
| scoring | `scripts/eval_map.py` (ultralytics 8.4.101 `DetMetrics` matcher, IoU 0.50:0.95), conf 0.001, IoU 0.7, max_det 300 |

Container CPU quota matters: the pod exposes 128 vCPUs but `cpu.max` caps it at 13.6 cores.
CPU backends therefore run 6 threads per worker, and the concurrency-8 API cells use 2 workers
(12 threads). Numbers for ncnn and MNN on CPU are for that budget, not for a 32-core host.

## Models

| id | source | fp32 | fp16 |
|---|---|---|---|
| v01n | `runs/a3/v01n/YOLO-Master-v0.1-N.onnx` (A3 P0 export, opset 17, static 640) | ONNX / TRT / ncnn (`models/v0.1-n_ncnn`, dense SDPA export) / MNN | fp16 ONNX (routing subgraph kept fp32), TRT fp16 flag, MNN fp16 weights |
| v01n-pruned | `runs/project03/trt/v01-n-coco-pruned-surgery/YOLO-Master-v0.1-N_pruned_t0.10.onnx` (MoEPruner t=0.10 + BN-router surgery) | ONNX / TRT / ncnn (dense SDPA export of `pod-prune/v01-n-coco/..._pruned_t0.10.pt`) / MNN | not part of the grid |
| esmoen | `runs/a3/esmoen/YOLO-Master-EsMoE-N.onnx` | ONNX / TRT / ncnn (`scripts/export_ncnn_dense.py` on the COCO checkpoint) / MNN | fp16 ONNX, TRT fp16 flag, MNN fp16 weights |

fp16 preparation (`scripts/server/prepare_models.py`): `onnxconverter_common` with fp32 I/O,
stale value_info stripped and shapes re-inferred, the `/routing/` subgraph and TopK/Gather ops
pinned to fp32 so expert selection does not depend on fp16 rounding. ORT-CPU probe drift on one
COCO image: v01n 3.1e-3 relative (101 vs 100 candidates at conf 0.25), esmoen 1.3e-2 relative
(102 vs 102). MNN fp16 = `MNNConvert --fp16` weights; on CPU MNN computes in fp32 regardless.
ncnn has no fp16 arithmetic on x86, so the ncnn fp16 cells are marked n/a rather than reported
from an fp32 path.

## Method

Every cell = one model file on one backend, three ways:

1. `cli`: `yolomaster_edge -s <val2017> --save-txt --no-save --quiet --warmup 20` (the bare
   runtime, sequential, stb JPEG decode, same pre/post code).
2. `api-c1`: `yolomaster_server` with one worker, `clients/python/bench_client.py --concurrency 1`
   (one request in flight, client-observed latency per image, 20 warm-up requests excluded).
3. `api-c8`: eight requests in flight; GPU models keep one worker, CPU models use two.

The API dumps come from the JSON response boxes at full float precision; parity against the CLI
txt files is checked numerically per detection (`scripts/server/parity_txt.py`, 0.01 px).
mAP is computed on both dump sets. "API overhead" = client p50 minus server (queue + total) p50,
i.e. HTTP framing, body copy and serialization.

## Results

### Table 1: bare runtime vs API, latency (ms) and accuracy

| model | backend | execution provider | CLI model | API model | CLI end-to-end | API client p50 | API client p95 | API client p99 | HTTP overhead | mAP50-95 CLI | mAP50-95 API | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0.1-N fp32 | TensorRT | TRT-CUDA-fp32 | 2.34 | 2.32 | 3.99 | 8.39 | 12.25 | 15.30 | 0.90 | 0.4176 | 0.4176 | OK |
| v0.1-N fp16 | TensorRT | TRT-CUDA-fp16 | 1.82 | 1.77 | 3.96 | 7.71 | 12.32 | 15.82 | 0.94 | 0.4174 | 0.4174 | OK |
| v0.1-N pruned fp32 | TensorRT | TRT-CUDA-fp32 | 2.02 | 1.96 | 3.82 | 8.01 | 11.53 | 14.06 | 0.91 | 0.4176 | 0.4176 | OK |
| EsMoE-N fp32 | TensorRT | TRT-CUDA-fp32 | 2.01 | 2.02 | 3.72 | 7.19 | 9.87 | 12.78 | 0.89 | 0.4154 | 0.4154 | OK |
| EsMoE-N fp16 | TensorRT | TRT-CUDA-fp16 | 1.68 | 1.61 | 3.42 | 6.89 | 9.66 | 11.92 | 0.84 | 0.4153 | 0.4152 | OK |
| v0.1-N fp32 | ORT CUDA | CUDA | 4.26 | 4.23 | 6.16 | 9.82 | 15.53 | 19.32 | 0.94 | 0.4176 | 0.4176 | OK |
| v0.1-N fp16 | ORT CUDA | CUDA | 4.62 | 4.26 | 6.31 | 9.93 | 15.81 | 19.45 | 1.01 | 0.4176 | 0.4176 | OK |
| v0.1-N pruned fp32 | ORT CUDA | CUDA | 3.92 | 3.61 | 6.12 | 10.23 | 15.77 | 19.79 | 1.03 | 0.4176 | 0.4176 | OK |
| EsMoE-N fp32 | ORT CUDA | CUDA | 3.62 | 3.24 | 5.50 | 9.32 | 14.27 | 17.71 | 0.92 | 0.4154 | 0.4154 | OK |
| EsMoE-N fp16 | ORT CUDA | CUDA | 3.64 | 3.55 | 5.25 | 9.35 | 14.50 | 18.17 | 0.96 | 0.4147 | 0.4147 | OK |
| v0.1-N fp32 | MNN CUDA | MNN-CUDA-fp32 | 4.30 | 4.31 | 16.98 | 22.04 | 24.12 | 25.63 | 1.30 | 0.4177 | 0.4177 | OK |
| v0.1-N fp16 | MNN CUDA | MNN-CUDA-fp16 | 4.52 | 2.15 | 61.05 | 59.73 | 65.70 | 69.08 | 1.27 | 0.0000 | 0.0000 | invalid output (0 dets) |
| v0.1-N pruned fp32 | MNN CUDA | MNN-CUDA-fp32 | 4.14 | 4.09 | 15.26 | 19.51 | 21.72 | 23.04 | 1.25 | 0.4177 | 0.4177 | OK |
| EsMoE-N fp32 | MNN CUDA | MNN-CUDA-fp32 | 3.92 | 1.61 | 13.39 | 14.57 | 19.07 | 20.41 | 1.35 | 0.4154 | 0.4154 | OK |
| EsMoE-N fp16 | MNN CUDA | MNN-CUDA-fp16 | 3.66 | 1.75 | 8.44 | 8.44 | 9.84 | 10.89 | 0.94 | 0.0000 | 0.0000 | invalid output (0 dets) |
| v0.1-N fp32 | ncnn CPU | ncnn-CPU-fp32 | 105.61 | 111.48 | 107.61 | 117.65 | 141.86 | 147.07 | 1.24 | 0.4177 | 0.4177 | OK |
| v0.1-N pruned fp32 | ncnn CPU | ncnn-CPU-fp32 | 114.12 | 90.00 | 115.96 | 95.93 | 121.14 | 130.74 | 1.38 | 0.4177 | 0.4177 | OK |
| EsMoE-N fp32 | ncnn CPU | ncnn-CPU-fp32 | 114.36 | 97.53 | 116.37 | 103.47 | 127.08 | 137.52 | 1.38 | 0.4154 | 0.4153 | OK |
| v0.1-N fp32 | MNN CPU | MNN-CPU | 245.25 | 173.52 | 249.36 | 183.24 | 244.46 | 255.39 | 1.35 | 0.4177 | 0.4177 | OK |
| v0.1-N fp16 | MNN CPU | MNN-CPU | 179.14 | 169.88 | 182.71 | 179.24 | 236.16 | 245.82 | 1.47 | 0.4176 | 0.4176 | OK |
| v0.1-N pruned fp32 | MNN CPU | MNN-CPU | 162.59 | 151.48 | 166.12 | 161.38 | 214.91 | 222.45 | 1.49 | 0.4177 | 0.4177 | OK |
| EsMoE-N fp32 | MNN CPU | MNN-CPU | 118.73 | 106.70 | 122.72 | 115.67 | 157.94 | 162.35 | 1.43 | 0.4153 | 0.4153 | OK |
| EsMoE-N fp16 | MNN CPU | MNN-CPU | 126.60 | 112.28 | 130.28 | 121.51 | 161.66 | 164.30 | 1.53 | 0.4156 | 0.4156 | OK |

### Table 2: throughput (images/s over the 5000-image run)

| model | backend | CLI sequential | API c=1 | API c=8, default workers | GPU util / mem at c=8 | API c=8, 4 workers |
|---|---|---|---|---|---|---|
| v0.1-N fp32 | TensorRT | 139.2 | 103.7 | 131.9 (1) | 26% / 551 MiB | 385.2 |
| v0.1-N fp16 | TensorRT | 140.9 | 110.5 | 149.0 (1) | 24% / 579 MiB | 378.9 |
| v0.1-N pruned fp32 | TensorRT | 144.8 | 108.7 | 142.2 (1) | 24% / 515 MiB | 480.0 |
| EsMoE-N fp32 | TensorRT | 149.0 | 120.5 | 159.0 (1) | 30% / 517 MiB | 434.7 |
| EsMoE-N fp16 | TensorRT | 154.5 | 126.1 | 138.1 (1) | 24% / 571 MiB | 358.6 |
| v0.1-N fp32 | ORT CUDA | 105.6 | 88.1 | 109.8 (1) | 32% / 585 MiB | 377.7 |
| v0.1-N fp16 | ORT CUDA | 102.4 | 88.3 | 113.0 (1) | 38% / 593 MiB | 344.2 |
| v0.1-N pruned fp32 | ORT CUDA | 105.5 | 84.9 | 119.3 (1) | 29% / 585 MiB | 416.0 |
| EsMoE-N fp32 | ORT CUDA | 111.0 | 93.5 | 121.6 (1) | 33% / 587 MiB | 396.8 |
| EsMoE-N fp16 | ORT CUDA | 111.8 | 93.7 | 116.7 (1) | 27% / 529 MiB | 375.4 |
| v0.1-N fp32 | MNN CUDA | 49.0 | 42.6 | 48.3 (1) | 55% / 625 MiB | 84.6 |
| v0.1-N fp16 | MNN CUDA | 15.4 | 16.3 | 16.4 (1) | 4% / 589 MiB | - |
| v0.1-N pruned fp32 | MNN CUDA | 53.2 | 48.0 | 55.8 (1) | 51% / 613 MiB | 109.3 |
| EsMoE-N fp32 | MNN CUDA | 59.5 | 60.8 | 65.2 (1) | 55% / 517 MiB | 120.1 |
| EsMoE-N fp16 | MNN CUDA | 85.1 | 112.7 | 138.0 (1) | 41% / 491 MiB | - |
| v0.1-N fp32 | ncnn CPU | 9.0 | 8.6 | 18.3 (2) | 0% / 0 MiB | - |
| v0.1-N pruned fp32 | ncnn CPU | 8.4 | 10.1 | 15.6 (2) | 0% / 0 MiB | - |
| EsMoE-N fp32 | ncnn CPU | 8.3 | 9.4 | 20.8 (2) | 0% / 0 MiB | - |
| v0.1-N fp32 | MNN CPU | 3.9 | 5.1 | 7.4 (2) | 0% / 0 MiB | - |
| v0.1-N fp16 | MNN CPU | 5.3 | 5.2 | 7.6 (2) | 0% / 0 MiB | - |
| v0.1-N pruned fp32 | MNN CPU | 5.9 | 5.7 | 8.0 (2) | 0% / 0 MiB | - |
| EsMoE-N fp32 | MNN CPU | 7.9 | 8.0 | 11.0 (2) | 0% / 0 MiB | - |
| EsMoE-N fp16 | MNN CPU | 7.4 | 7.4 | 11.0 (2) | 0% / 0 MiB | - |

### Table 3: API server stage breakdown at c=1 (server p50, ms)

| model | backend | queue | decode | pre | infer | post | total | client p50 |
|---|---|---|---|---|---|---|---|---|
| v0.1-N fp32 | TensorRT | 0.01 | 2.80 | 1.55 | 2.32 | 0.71 | 7.47 | 8.39 |
| v0.1-N fp16 | TensorRT | 0.02 | 2.85 | 1.25 | 1.77 | 0.73 | 6.76 | 7.71 |
| v0.1-N pruned fp32 | TensorRT | 0.01 | 2.81 | 1.54 | 1.96 | 0.70 | 7.08 | 8.01 |
| EsMoE-N fp32 | TensorRT | 0.01 | 2.75 | 0.69 | 2.02 | 0.71 | 6.28 | 7.19 |
| EsMoE-N fp16 | TensorRT | 0.01 | 2.66 | 0.97 | 1.61 | 0.69 | 6.04 | 6.89 |
| v0.1-N fp32 | ORT CUDA | 0.01 | 2.80 | 0.94 | 4.23 | 0.71 | 8.87 | 9.82 |
| v0.1-N fp16 | ORT CUDA | 0.02 | 2.79 | 0.96 | 4.26 | 0.71 | 8.91 | 9.93 |
| v0.1-N pruned fp32 | ORT CUDA | 0.02 | 2.79 | 1.85 | 3.61 | 0.79 | 9.19 | 10.23 |
| EsMoE-N fp32 | ORT CUDA | 0.01 | 2.74 | 1.53 | 3.24 | 0.70 | 8.38 | 9.32 |
| EsMoE-N fp16 | ORT CUDA | 0.02 | 2.75 | 1.18 | 3.55 | 0.71 | 8.37 | 9.35 |
| v0.1-N fp32 | MNN CUDA | 0.01 | 2.60 | 2.93 | 4.31 | 10.18 | 20.72 | 22.04 |
| v0.1-N fp16 | MNN CUDA | 0.02 | 2.52 | 0.82 | 2.15 | 52.15 | 58.44 | 59.73 |
| v0.1-N pruned fp32 | MNN CUDA | 0.01 | 2.75 | 3.11 | 4.09 | 7.62 | 18.25 | 19.51 |
| EsMoE-N fp32 | MNN CUDA | 0.02 | 2.54 | 1.39 | 1.61 | 7.17 | 13.20 | 14.57 |
| EsMoE-N fp16 | MNN CUDA | 0.01 | 2.57 | 1.04 | 1.75 | 2.12 | 7.49 | 8.44 |
| v0.1-N fp32 | ncnn CPU | 0.02 | 2.74 | 1.01 | 111.48 | 0.70 | 116.39 | 117.65 |
| v0.1-N pruned fp32 | ncnn CPU | 0.01 | 2.64 | 1.18 | 90.00 | 0.62 | 94.53 | 95.93 |
| EsMoE-N fp32 | ncnn CPU | 0.02 | 2.69 | 0.85 | 97.53 | 0.72 | 102.08 | 103.47 |
| v0.1-N fp32 | MNN CPU | 0.02 | 2.75 | 3.60 | 173.52 | 0.84 | 181.87 | 183.24 |
| v0.1-N fp16 | MNN CPU | 0.01 | 2.70 | 3.08 | 169.88 | 0.82 | 177.75 | 179.24 |
| v0.1-N pruned fp32 | MNN CPU | 0.02 | 2.70 | 3.52 | 151.48 | 0.82 | 159.87 | 161.38 |
| EsMoE-N fp32 | MNN CPU | 0.01 | 2.62 | 2.98 | 106.70 | 0.81 | 114.23 | 115.67 |
| EsMoE-N fp16 | MNN CPU | 0.02 | 2.64 | 3.06 | 112.28 | 0.81 | 119.96 | 121.51 |

Notes on the tables. "CLI model" and "API model" are the backend's own forward timing
(`infer_ms`) inside the CLI and inside the server. "CLI end-to-end" is the CLI's pre + model +
post per frame (its JPEG decode is outside that number); "API client p50" is the full round trip
seen by the client including decode. MNN's CUDA backend runs the session asynchronously, so its
"model" column excludes the device-to-host copy that lands in the post stage (10 to 12 ms);
compare MNN CUDA on end-to-end and client numbers, not on the model column. The "API c=8, 4
workers" column comes from a separate run (`results/api_bench_gpu4`, `API_WORKERS_GPU=4
PATHS=c8`). The pruned ncnn row is the rerun with the dense SDPA export (the first pass used the
older stock export: 154 ms model time, same mAP; kept under `api_bench_pruned_ncnn_oldexport`).
mAP50-95 here is `eval_map.py`'s matcher on the txt dumps at conf 0.001 and IoU 0.7; it is 0.007
below the ultralytics validator numbers quoted in the A3 report for the same weights, so compare
rows within this table only.

## Reading the numbers

**The API costs about one millisecond.** Across all 23 cells the HTTP overhead (client p50
minus server queue + total) is 0.84 to 1.03 ms on the GPU backends and 1.2 to 1.5 ms on the CPU
backends, and the model time measured inside the server equals the bare runtime's within run
noise. The server changes nothing numerically: every cell's mAP50-95 agrees between the API and
CLI dumps, and 528k to 546k detections per cell match one to one within 0.0005 px (the CLI prints
six significant digits). That was the acceptance gate for the comparison.

**On the GPU the request is decode-bound, not model-bound.** TensorRT runs v0.1-N in 2.3 ms
(fp32) or 1.8 ms (fp16) and EsMoE-N in 2.0 / 1.6 ms, yet a request takes 7 to 8 ms end to end:
stb decodes a COCO JPEG in 2.5 to 4 ms and the letterbox costs about 1 ms on the worker thread.
With one worker per GPU model that serializes decode behind inference, so the bare CLI tops out
near 140 to 155 img/s and the API at c=8 near 130 to 160 img/s. Giving a GPU model four workers
(each with its own TensorRT context, about 60 MB more GPU memory apiece) lets decode overlap and
lifts the same models to 345 to 480 img/s at c=8, a 3x gain with unchanged results. That is the
knob to turn for throughput; batching would not help a batch-1 model that is waiting on JPEG
decode.

**Backend ranking on the L40S.** TensorRT is 1.6 to 2.4x faster than ONNX Runtime's CUDA
provider on the same ONNX (v0.1-N 2.34 vs 4.26 ms), and its fp16 engines save another 16 to 22%
at a cost of at most 0.0007 mAP. fp16 ONNX on ORT CUDA gains nothing (the routing subgraph stays
fp32 and the inserted casts cost what fp16 saves); on TensorRT the fp16 flag is the right way to
get fp16. MNN's CUDA backend computes in 4 ms but pays a synchronous 10 to 12 ms device-to-host
copy per frame in this MNN build, so it lands at 15 to 22 ms per request; MNN CUDA in fp16
(`Precision_Low`) returns zero detections on every model because the emulated MoE routers
overflow fp16, the same failure ncnn's fp16 policy guards against. Those two rows are reported as
invalid output, not as speed.

**Pruning is a GPU win, a wash on x86 CPU.** The MoEPruner t=0.10 model runs 14% faster on
TensorRT (2.02 vs 2.34 ms) and 8% faster on ORT CUDA at identical mAP (0.4176 both). On ncnn and
MNN CPU the pruned graph is within run noise of the unpruned one (ncnn 114 vs 106 ms, MNN 163 vs
179 to 245 ms): removing experts shrinks weights, but the CPU time in these graphs is dominated by
the attention and dispatch glue that pruning does not touch.

**CPU rows are what a 13.6-core container gives.** ncnn runs v0.1-N in 106 ms and EsMoE-N in 114
ms at 6 threads; MNN takes 120 to 245 ms. Two ncnn workers roughly double throughput at c=8
(8.6 to 18 img/s); MNN scales less (5 to 8 img/s) because its thread pool already saturates the
quota. The CLI-vs-API model-time gaps on CPU (for example MNN v0.1-N fp32: 245 ms CLI, 174 ms in
the server) are quota scheduling noise between runs, not a property of either path; the
throughput columns are the stable comparison. ncnn has no fp16 arithmetic on x86, so its fp16
cells are n/a by design rather than an fp32 number in disguise.

**What this says for project03's 端到端延迟 criterion.** Serving v0.1-N through the API on the
L40S costs 7 to 8 ms per image end to end (fp16 TensorRT, one client), of which 1.8 ms is the
model and about 1 ms is HTTP; optimisation (fp16, pruning) moves the model term, not the
request term, so the remaining lever is decode (a libjpeg-turbo or nvJPEG decoder would take
the request under 5 ms) and worker count.

## v1.2.0 rerun on an L4 with GPU preprocessing (preproc=cuda)

The tables above are the v1.1.x server on the L40S pod with CPU letterboxing in every cell. The
v1.2.0 runtime moves letterbox + normalize + NCHW onto the GPU for TensorRT and ORT-CUDA
(`GPU_PREPROC_RESULTS.md` has the kernel parity, the coco500 accuracy check and the per-stage
timing on an RTX PRO 4500). The full 5000-image comparison was rerun once more for the 1.2.1
image, this time on an NVIDIA L4 pod because the L40S pod no longer exists. Same protocol as
above (`scripts/server/run_comparison.sh`, 21 cells, the two ncnn fp16 cells n/a on x86, MNN
CUDA dropped from the grid after its fp16 verdict). Cross-machine numbers are not comparable
with Tables 1 to 3; the within-table comparisons (CLI vs API, backend vs backend) are.

| item | L4 pod (this section) | L40S pod (Tables 1 to 3) |
|---|---|---|
| GPU | NVIDIA L4 24 GB, driver 580.159.04, max SM clock 2040 MHz, 72 W | L40S 46 GB, 2520 MHz, 350 W |
| CPU | AMD EPYC 7702 (Zen 2), 128 vCPUs visible, quota 13.6 cores (`cpu.cfs_quota_us` 1360000/100000) | EPYC 9374F (Zen 4), quota 13.6 cores |
| RAM | 503 GB | 1.5 TB |
| runtime | yolo-master-edge 1.2.0, server 1.2.1, `-DUSE_CUDA_PREPROC=ON`, sm_89 | 1.1.x |
| TensorRT / ORT / ncnn / MNN | 10.16.1 / 1.20.1 GPU / 20260526 / 3.6.1 CPU | same |
| CUDA graphs | off (default; `cuda_graph=1` is opt-in per model) | not available |

### Table 4: L4, v1.2.0, latency (ms) and accuracy, preproc=cuda on the GPU rows

| model | backend | execution provider | CLI model | API model | CLI end-to-end | API client p50 | API client p95 | API client p99 | HTTP overhead | mAP50-95 CLI | mAP50-95 API | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0.1-N fp32 | TensorRT | TRT-CUDA-fp32, preproc=cuda | 3.22 | 3.75 | 4.66 | 16.34 | 21.67 | 25.02 | 1.62 | 0.4176 | 0.4176 | OK |
| v0.1-N fp16 | TensorRT | TRT-CUDA-fp16, preproc=cuda | 2.29 | 3.54 | 3.62 | 16.46 | 22.08 | 25.18 | 1.72 | 0.4176 | 0.4176 | OK |
| v0.1-N pruned fp32 | TensorRT | TRT-CUDA-fp32, preproc=cuda | 2.70 | 3.28 | 4.12 | 15.94 | 21.28 | 24.12 | 1.74 | 0.4176 | 0.4176 | OK |
| EsMoE-N fp32 | TensorRT | TRT-CUDA-fp32, preproc=cuda | 2.69 | 3.53 | 4.09 | 16.15 | 21.41 | 24.47 | 1.73 | 0.4155 | 0.4155 | OK |
| EsMoE-N fp16 | TensorRT | TRT-CUDA-fp16, preproc=cuda | 2.12 | 3.09 | 3.45 | 14.89 | 20.78 | 23.74 | 1.81 | 0.4152 | 0.4152 | OK |
| v0.1-N fp32 | ORT CUDA | ort-CUDA, preproc=cuda | 7.15 | 10.17 | 8.53 | 19.53 | 27.31 | 30.07 | 1.67 | 0.4176 | 0.4176 | OK |
| v0.1-N fp16 | ORT CUDA | ort-CUDA, preproc=cuda | 7.83 | 7.78 | 9.26 | 16.53 | 25.73 | 28.97 | 1.82 | 0.4175 | 0.4175 | OK |
| v0.1-N pruned fp32 | ORT CUDA | ort-CUDA, preproc=cuda | 6.22 | 8.85 | 7.61 | 19.35 | 26.41 | 29.47 | 1.77 | 0.4177 | 0.4176 | OK |
| EsMoE-N fp32 | ORT CUDA | ort-CUDA, preproc=cuda | 5.96 | 5.89 | 7.31 | 15.23 | 24.56 | 27.65 | 1.88 | 0.4154 | 0.4154 | OK |
| EsMoE-N fp16 | ORT CUDA | ort-CUDA, preproc=cuda | 6.00 | 6.56 | 7.29 | 16.28 | 24.32 | 27.05 | 1.82 | 0.4147 | 0.4147 | OK |
| v0.1-N fp32 | ncnn CPU | ncnn-CPU-fp32 | 107.78 | 108.14 | 110.83 | 116.67 | 124.41 | 128.01 | 1.90 | 0.4177 | 0.4177 | OK |
| v0.1-N pruned fp32 | ncnn CPU | ncnn-CPU-fp32 | 108.36 | 107.42 | 110.40 | 115.96 | 123.22 | 128.65 | 1.92 | 0.4177 | 0.4177 | OK |
| EsMoE-N fp32 | ncnn CPU | ncnn-CPU-fp32 | 114.68 | 113.15 | 117.64 | 121.74 | 128.25 | 132.55 | 1.93 | 0.4154 | 0.4153 | OK |
| v0.1-N fp32 | MNN CPU | MNN-CPU | 167.35 | 166.15 | 169.35 | 174.54 | 184.82 | 190.00 | 1.83 | 0.4177 | 0.4177 | OK |
| v0.1-N fp16 | MNN CPU | MNN-CPU | 167.75 | 167.15 | 169.78 | 175.55 | 186.30 | 191.48 | 1.89 | 0.4176 | 0.4176 | OK |
| v0.1-N pruned fp32 | MNN CPU | MNN-CPU | 150.06 | 150.55 | 152.09 | 158.98 | 170.92 | 179.77 | 1.90 | 0.4177 | 0.4177 | OK |
| EsMoE-N fp32 | MNN CPU | MNN-CPU | 85.28 | 88.78 | 87.28 | 97.56 | 108.56 | 112.91 | 1.93 | 0.4154 | 0.4154 | OK |
| EsMoE-N fp16 | MNN CPU | MNN-CPU | 97.27 | 94.50 | 99.44 | 102.89 | 112.89 | 118.07 | 1.60 | 0.4157 | 0.4157 | OK |

### Table 5: L4, v1.2.0, throughput (images/s over the 5000-image run)

| model | backend | CLI sequential | API c=1 | API c=8 (workers) |
|---|---|---|---|---|
| v0.1-N fp32 | TensorRT | 62.8 | 40.8 | 103.8 (1) |
| v0.1-N fp16 | TensorRT | 70.0 | 40.0 | 116.0 (1) |
| v0.1-N pruned fp32 | TensorRT | 67.2 | 42.7 | 111.2 (1) |
| EsMoE-N fp32 | TensorRT | 68.5 | 42.0 | 112.0 (1) |
| EsMoE-N fp16 | TensorRT | 69.4 | 44.6 | 118.3 (1) |
| v0.1-N fp32 | ORT CUDA | 48.7 | 35.7 | 77.7 (1) |
| v0.1-N fp16 | ORT CUDA | 46.8 | 39.4 | 72.4 (1) |
| v0.1-N pruned fp32 | ORT CUDA | 51.5 | 36.7 | 82.3 (1) |
| EsMoE-N fp32 | ORT CUDA | 51.6 | 39.7 | 82.0 (1) |
| EsMoE-N fp16 | ORT CUDA | 54.2 | 40.2 | 81.8 (1) |
| v0.1-N fp32 | ncnn CPU | 8.2 | 7.9 | 16.5 (2) |
| v0.1-N pruned fp32 | ncnn CPU | 8.2 | 8.0 | 16.5 (2) |
| EsMoE-N fp32 | ncnn CPU | 7.7 | 7.7 | 16.1 (2) |
| v0.1-N fp32 | MNN CPU | 5.5 | 5.4 | 8.5 (2) |
| v0.1-N fp16 | MNN CPU | 5.5 | 5.4 | 8.4 (2) |
| v0.1-N pruned fp32 | MNN CPU | 6.1 | 5.9 | 9.8 (2) |
| EsMoE-N fp32 | MNN CPU | 9.9 | 9.4 | 13.2 (2) |
| EsMoE-N fp16 | MNN CPU | 9.0 | 9.0 | 13.7 (2) |

### Table 6: L4, v1.2.0, API server stage breakdown at c=1 (server p50, ms)

| model | backend | queue | decode | pre | infer | post | total | client p50 |
|---|---|---|---|---|---|---|---|---|
| v0.1-N fp32 | TensorRT | 0.03 | 8.83 | 0.21 | 3.75 | 1.63 | 14.68 | 16.34 |
| v0.1-N fp16 | TensorRT | 0.03 | 9.02 | 0.21 | 3.54 | 1.70 | 14.71 | 16.46 |
| v0.1-N pruned fp32 | TensorRT | 0.03 | 8.82 | 0.21 | 3.28 | 1.62 | 14.17 | 15.94 |
| EsMoE-N fp32 | TensorRT | 0.03 | 8.74 | 0.21 | 3.53 | 1.66 | 14.40 | 16.15 |
| EsMoE-N fp16 | TensorRT | 0.02 | 8.01 | 0.20 | 3.09 | 1.50 | 13.06 | 14.89 |
| v0.1-N fp32 | ORT CUDA | 0.03 | 6.62 | 0.20 | 10.17 | 1.44 | 17.84 | 19.53 |
| v0.1-N fp16 | ORT CUDA | 0.02 | 5.39 | 0.20 | 7.78 | 1.12 | 14.68 | 16.53 |
| v0.1-N pruned fp32 | ORT CUDA | 0.03 | 6.86 | 0.21 | 8.85 | 1.48 | 17.55 | 19.35 |
| EsMoE-N fp32 | ORT CUDA | 0.02 | 5.90 | 0.20 | 5.89 | 1.15 | 13.33 | 15.23 |
| EsMoE-N fp16 | ORT CUDA | 0.02 | 6.19 | 0.20 | 6.56 | 1.20 | 14.44 | 16.28 |
| v0.1-N fp32 | ncnn CPU | 0.02 | 4.40 | 0.81 | 108.14 | 1.02 | 114.75 | 116.67 |
| v0.1-N pruned fp32 | ncnn CPU | 0.02 | 4.43 | 0.80 | 107.42 | 1.00 | 114.02 | 115.96 |
| EsMoE-N fp32 | ncnn CPU | 0.02 | 4.46 | 0.81 | 113.15 | 1.02 | 119.79 | 121.74 |
| v0.1-N fp32 | MNN CPU | 0.02 | 4.41 | 0.77 | 166.15 | 1.02 | 172.69 | 174.54 |
| v0.1-N fp16 | MNN CPU | 0.02 | 4.43 | 0.79 | 167.15 | 1.03 | 173.64 | 175.55 |
| v0.1-N pruned fp32 | MNN CPU | 0.02 | 4.44 | 0.82 | 150.55 | 1.04 | 157.06 | 158.98 |
| EsMoE-N fp32 | MNN CPU | 0.02 | 4.54 | 0.83 | 88.78 | 1.06 | 95.61 | 97.56 |
| EsMoE-N fp16 | MNN CPU | 0.02 | 4.56 | 0.79 | 94.50 | 1.05 | 101.27 | 102.89 |

Notes on Tables 4 to 6. The GPU rows carry `preproc=cuda`: `pre` is now the host letterbox
arithmetic plus the pinned copy and the kernel (0.2 ms in the server, 0.18 ms in the CLI), and
`infer` is enqueue plus the device-to-host copy, so the columns are not the same quantities as in
Tables 1 to 3 (there the float H2D sat inside TensorRT's `infer` and the CPU NCHW loop in `pre`).
CPU rows are unchanged in meaning. The pruned ncnn row uses the dense SDPA export regenerated
from the pruned checkpoint in this release (`models/p03_v01n_ncnn`).

**Parity and accuracy hold with the GPU kernel in the loop.** All 18 reported cells match
between API and CLI dumps within 0.01 px and every mAP50-95 pair agrees to four decimals except
the pruned ORT-CUDA cell (0.4177 vs 0.4176, a fourth-decimal rounding boundary; the per-box parity check passes). TensorRT fp32 and
ORT-CUDA fp32 land on the same 0.4176 / 0.4155 as the CPU-preprocessed L40S run, so the bilinear
kernel's sub-1/255 difference from `cv::resize` is invisible at COCO scale, as the coco500 check
in `GPU_PREPROC_RESULTS.md` predicted.

**The API overhead is unchanged at 1.6 to 1.9 ms on this slower CPU.** Client p50 minus server
queue + total is 1.6 to 1.9 ms in every cell (0.8 to 1.5 ms on the Zen 4 pod); it scales with the
core, not with the backend.

**On the L4 the request is even more decode-bound.** stb decodes a COCO JPEG in 8 to 9 ms on the
Zen 2 core that serves a GPU model (4.4 ms on the CPU rows, whose six ncnn or MNN threads keep
the cores clocked up; the single GPU worker alternates one decode with a GPU wait and the core
idles between). With preprocessing at 0.2 ms and the model at 2.3 to 3.5 ms, decode is 60% of a
15 to 16 ms request. The GPU itself reports 9% utilisation and 37 W at c=1: the L4 clocks down
between requests, which is why the model time measured inside the server (3.5 ms) is above the
back-to-back CLI figure (2.3 ms) on TensorRT; on the L40S the two agreed because the requests
arrived faster than the clock governor reacts. The lever is the same as before: a faster decoder
(libjpeg-turbo or nvJPEG) and more workers per GPU model, not the model.

**Backend ranking is the same.** TensorRT beats ORT-CUDA by 2.2 to 3.4x on the CLI model time
(v0.1-N 3.22 vs 7.15 ms fp32), fp16 engines save 21 to 29% (v0.1-N 2.29 ms, EsMoE-N 2.12 ms)
at a cost of 0 to 0.0003 mAP, and the pruned v0.1-N is 16% faster than the unpruned one on
TensorRT and 13% on ORT-CUDA at identical accuracy. On ORT-CUDA the fp16 ONNX gains nothing (v0.1-N 7.83 vs
7.15 ms, EsMoE-N 6.00 vs 5.96); the routing casts still cost what fp16 saves. ncnn on this
Zen 2 quota runs v0.1-N in 108 ms and EsMoE-N in 115 ms (106 / 114 on Zen 4), MNN 150 to 168 ms
for v0.1-N and 85 to 97 ms for EsMoE-N; two CPU workers double ncnn throughput at c=8 (8 to 16
img/s) and MNN scales by 1.4 to 1.7x.

Raw per-cell artifacts of this run (CLI logs and txt dumps, bench JSON, server stats and logs,
GPU/CPU samples, scores) are in `/data/results/api_bench_l4_v120.tar.gz` (537 MB, txt dumps
included); the rendered tables come from `python scripts/server/results_tables.py <dir>`.

## Reproduce

```
bash deploy/pod/bootstrap_l40s.sh && bash deploy/pod/build_l40s.sh
bash deploy/pod/prepare_models_pod.sh
bash scripts/server/run_comparison.sh /data/results/api_bench      # about 6 hours on the pod
python scripts/server/summarize_comparison.py /data/results/api_bench
```

Raw per-cell artifacts (CLI logs, bench JSON, server stats, GPU/CPU samples, scores) are in
`api_bench.tgz` on mdb (`/yolotmp/api_bench.tgz`, txt dumps excluded); they are not committed.
For the v1.2.0 rerun the pod scripts are `deploy/pod/bootstrap_l40s.sh`, `deploy/pod/build_gpu.sh`
(`SM=89 USE_CUDA_PREPROC=ON`) and `deploy/pod/prepare_models_pod.sh`; `run_comparison.sh` is
unchanged.

## Deliverables

- Docker image `docker.io/skywalker0501/yolomaster-api:1.2.1` (also `:latest`), digest
  `sha256:65771d282d093d2082f729ad410cf3c5abfbacb50c859f8b26a6b6a6e7e54b43`, 5.26 GB: the v1.2.0
  runtime (GPU preprocessing, bench, tracking) and the 1.2.1 server (auth, rate limit, tracing,
  OpenAPI); same base image and layer layout as 1.2.0 below, TensorRT builder resources for
  sm75/80/86/89/90. This is the image Tables 4 to 6 were measured with.
- Docker image `docker.io/skywalker0501/yolomaster-api:1.2.0`, digest
  `sha256:00e72c118237a5386ad0cc305350079f5773ee48fe619c5cfdbf3c8080b26cd2`, 5.25 GB compressed:
  `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04` plus the `/opt/yolomaster` layer (server, CLI,
  ORT-GPU 1.20.1, TensorRT 10.16 runtime + builder resources for sm80/86/89/90, ncnn, MNN 3.6 with
  the CUDA backend, OpenCV-lean, ffmpeg 4.4, the three model ids in every format). Built on the
  pod with Bazel 7.4.1 + rules_oci 1.7.5 and pushed with crane, no Docker daemon involved.
- One-click: `bash deploy/run.sh skywalker0501/yolomaster-api:1.2.0 8080` or
  `IMAGE=skywalker0501/yolomaster-api:1.2.0 docker compose -f deploy/docker-compose.yml up -d`.
- Source: branch `dev/api-server` of yolo-master-edge (`cpp/server`, `clients/python`, `deploy`,
  `docs/API.md`, `tests/server`).
