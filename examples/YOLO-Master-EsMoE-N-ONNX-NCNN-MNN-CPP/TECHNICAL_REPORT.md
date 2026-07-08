# Edge Deployment of YOLO-Master-EsMoE-N — A Technical Report

End-to-end deployment of **YOLO-Master-EsMoE-N** (VisDrone) to the edge: three export formats (ONNX / NCNN / MNN), mixed-precision INT8, a single universal C++ runtime, cross-platform builds, and accuracy/latency validation against the PyTorch original that isolates *format fidelity* from *pipeline noise*.

**Repo:** https://github.com/skywalker-lt/yolo-master-edge

The interesting parts of this work were not the happy paths (`model.export()` mostly works); they were the failure modes — a quantized model that silently emits **zero detections**, an mAP that reads **1.3 points low for the wrong reason**, and a 129 MB "portable" bundle that ships a **PostgreSQL client**. This report documents those, and how each was diagnosed and closed.

---

## 1. Why the model's internals dictate the deployment strategy

EsMoE-N is not a vanilla CNN. Three structural facts drove every downstream decision:

1. **Mixture-of-Experts (`ES_MOE`).** At training/inference the router sparsely selects experts. That path is export-hostile (data-dependent control flow) — but `ES_MOE.forward` switches to a **dense** unroll under `torch.onnx.is_in_onnx_export()`: a static loop over the full expert list, `Conv/Pool/Softmax/Mul/Add`, no dynamic dispatch. Crucially, the dense path is also the **numerically faithful** one — it sidesteps the sparse-inference collapse that the sparse path exhibits — so exporting *improves* determinism rather than approximating it.
2. **Area-attention (`A2C2f`).** The backbone carries transformer-style attention blocks. These reshape activations to `[1, 1600, 192]` internally, which — as we'll see — is exactly where the static-shape assumptions of downstream quantizers break.
3. **A stride-4/8/16/32 detection head** whose classification branch produces raw logits fed through a terminal sigmoid. This branch is the single most quantization-sensitive component in the network, for a concrete reason developed in §3.

The takeaway: the model exports cleanly precisely *because* the dense MoE path is standard ops — but the attention and the head are landmines for INT8 and for third-party converters.

## 2. Export pipeline

### 2.1 ONNX (opset 12, onnxsim)

Exported to a fully **static** graph — input `images [1,3,640,640]`, output `output0 [1,14,8400]` (4 box + 10 class over 8400 anchors), 628 nodes, IR 7 — and simplified with **onnxsim**. Opset 12 was chosen deliberately as the compatibility floor: it loads unchanged under ONNXRuntime 1.18 / 1.20 / 1.27 and converts cleanly to *both* NCNN and MNN, which is the operational definition of "opset-compatible" that matters here.

The export emits shape-inference warnings on the attention transposes (`.../attn/Transpose_output_0 source:{1,1600,192} target:{}`, resolved by ONNXRuntime's lenient merge). These are benign at inference — ORT resolves the shapes at runtime — but they are a **leading indicator**: any tool that requires fully static shape propagation (offline quantizers) will choke here. That prediction is borne out in §3.5.

Ultralytics metadata (class names, `imgsz`, `stride`, `task`) is embedded in the ONNX `metadata_props`, which the C++ runtime later reads to auto-configure itself — no hardcoded class tables.

### 2.2 NCNN via pnnx

Converted through **pnnx** (PyTorch/ONNX → pnnx IR → ncnn), not the legacy `onnx2ncnn`. pnnx preserves higher-level operator semantics and emits a cleaner graph. The param file was validated structurally: magic `7767517`, **561 layers / 665 blobs**, input blob `in0`, sigmoid-terminated head. A `metadata.yaml` sidecar carries the same names/imgsz so the ncnn path is self-describing like the ONNX one.

### 2.3 MNN

Converted with `mnnconvert` (ONNX → MNN, 10.8 MB) — the *same* graph as ONNX/ncnn, which lets us later prove MNN correctness by direct tensor comparison against ONNX rather than a separate mAP run (§5.3).

## 3. INT8 quantization — the substantive part

The requirement was ≤ 1.0% mAP error under INT8 with ≥ 300 calibration images. The naive route fails silently and instructively.

### 3.1 The collapse: full INT8 emits zero detections

Static per-channel INT8 over the whole graph produces a model that runs, returns the correct output tensor shape, contains **no NaNs** — and detects **nothing**. mAP = 0.0000.

Isolating the output tensor shows why. The box-regression channels are intact (`min 0, max 644, mean 210`, matching FP32 within noise); the **classification channels are uniformly zero** (`max = 0.0000`, zero scores above 0.001). The failure is entirely in the class head.

The mechanism: the classification branch emits wide-dynamic-range **logits** consumed by a sigmoid. Per-tensor/per-channel MinMax calibration maps that wide range onto 256 INT8 levels; the small positive logits that correspond to real detections fall *below one quantization step* and round to a value whose sigmoid is ~0. The nonlinearity turns a modest quantization error on the logits into a total loss of signal. Box regression, by contrast, is a smooth linear readout with no saturating nonlinearity downstream, so it tolerates INT8 comfortably. This asymmetry — **regression robust, classification catastrophic** — is the key diagnostic.

### 3.2 Localizing the sensitivity

Keeping the detection head (`/model.25/`, 85 nodes) in FP32 and quantizing everything else recovers the model to **mAP50-95 = 0.1924, −1.12%** vs PyTorch — functional, but over budget. The residual loss is not uniform; it concentrates in two more structures:

- **The MoE router.** Expert mixing is a softmax over routing logits. INT8 perturbs the routing weights, which re-weights the expert combination — a first-order change to the features, not a rounding error on them.
- **Area-attention.** Attention scores pass through a softmax whose output is sensitive to input scale; INT8 on the QK path shifts the attention distribution.

Both are the same failure class as the head: **a softmax/sigmoid amplifying a quantization perturbation.**

### 3.3 The mixed-precision recipe

The fix is node-level precision assignment: keep the three softmax/sigmoid-bearing blocks — head (`/model.25/`), attention (`/attn/`), router (`routing`), **289 nodes** — in FP32, INT8 everything else (the bulk of the convolutional compute). The progression is monotonic and diagnostic:

| Configuration | mAP50-95 | Δ vs PyTorch |
|---|---|---|
| Full INT8 | 0.0000 | collapse |
| head FP32 | 0.1924 | −1.12% |
| head + attention + router FP32 | **0.1952** | **−0.84% ✅** |

Final model: **10.9 → 5.4 MB (2.0×)**, mAP50-95 error **−0.84%**, inside the 1.0% budget. This is not a lucky threshold — it's the direct consequence of removing quantization from exactly the operators that violate the "smooth, non-saturating" assumption PTQ relies on.

### 3.4 Calibration engineering

Three non-obvious details mattered:

- **Letterbox-matched calibration.** Calibrators default to a plain resize; the model is trained on **letterboxed** input. Calibrating on the wrong preprocessing biases every activation range. We pre-letterbox 300+ VisDrone *train* images (no val leakage) to 640×640 and calibrate on those, so the calibration distribution matches inference exactly.
- **QOperator, and the opset floor.** Per-channel INT8 emits `DequantizeLinear` with an `axis` attribute, which is **only valid at opset ≥ 13**; the opset-12 export must be lifted (we upgrade to 17 in-line) or the quantized model is an invalid graph. QOperator (`QLinearConv`/`QLinearMatMul`) is chosen over QDQ for CPU execution.
- **MinMax over Percentile.** Percentile/entropy calibration builds a histogram per activation tensor; on a graph with hundreds of attention/MoE intermediates × hundreds of images, that is pathologically slow for no accuracy gain here — the exclusions already remove the outlier-heavy layers, so MinMax on the remaining well-behaved convolutions is both faster and sufficient.

### 3.5 Third-party INT8 toolchains hit the attention wall

MNN's offline quantizer (`mnnquant`) aborts immediately on this model — `std::length_error: cannot create std::vector larger than max_size()` — before any calibration runs. The cause is precisely the `[1,1600,192]` attention reshapes flagged in §2.1: the quantizer allocates buffers from statically-inferred tensor dimensions, and the dynamically-shaped attention intermediate reads back as a garbage size. MNN executes this graph fine at *inference* (it resolves shapes lazily); its *quantizer* assumes static shapes. This is a limitation of the tool's static-shape contract, not of the model, and it is not configurable. The ONNXRuntime quantizer, which tolerates dynamic intermediates, is the correct vehicle for this architecture.

### 3.6 Where INT8 actually pays off

On x86 CPU, INT8 is *slower* than FP32 — measured at **137 ms/frame vs 49 ms for FP32 on the same host, ~2.8× slower** (7.2 vs 19.5 FPS). The QDQ/QOperator kernels don't engage INT8 SIMD paths that beat the well-tuned FP32 convolutions, and the FP32↔INT8 boundaries around the excluded blocks add conversion overhead. This is expected, not a defect: INT8's throughput win is a property of **INT8 tensor-core hardware** (TensorRT on Orin, NPUs), not of desktop CPUs. We therefore treat the ONNX INT8 result as the **accuracy proof** (−0.84%, in budget) and locate the **performance** validation on the TensorRT path (§8), where the same mixed-precision assignment maps onto tensor-core execution.

## 4. The inference runtime

### 4.1 Universal binary

One executable (`yolomaster_edge`) with **ONNXRuntime** and **NCNN** backends behind a common interface. Backend, class names, and input size are **auto-detected** from the model (ONNX metadata / ncnn `metadata.yaml`), so the same binary serves any exported YOLO-Master variant with no recompilation. Source can be an image, a directory, a video, or a `dataset.yaml`. Sixteen robustness tests (corrupt images, missing files, imgsz mismatch, backend inference, etc.) pass on every platform.

### 4.2 Preprocessing

Aspect-ratio-preserving **letterbox** (min-side scale, 114 padding) → RGB `/255` NCHW, matching training. The letterbox metadata (scale, pad) is threaded through decode so boxes map back to original-image pixel coordinates in float, with no intermediate integer rounding.

### 4.3 Decode, NMS, and the mAP-parity subtlety

An early version of the C++ pipeline read **1.3 mAP points low** despite bit-accurate inference. The cause was in the decode, not the model: ultralytics `val` uses **`multi_label=True`** — one detection per class scoring above threshold per anchor, not a single argmax. Reproducing that (a `--multi-label` mode) recovered the gap exactly (0.3375 → 0.3494 mAP50). NMS is **per-class** (`agnostic=False`), implemented with a class-offset trick (shift each box by `class_id × 8192` so cross-class boxes never suppress each other), and capped at 300 detections. Default `conf` is low, appropriate to VisDrone's small/dense objects; `--conf`/`--iou` are tunable per deployment.

### 4.4 Dependency surgery for a real portable bundle

The first self-contained Linux bundle was **231 shared libraries, 129 MB** — because Ubuntu's `libopencv_imgcodecs` links **GDAL**, which transitively pulls in PostgreSQL (`libpq`), MySQL, `libpoppler` (PDF), HDF5, and the GIS stack, and `libopencv_dnn` pulls protobuf. An object detector does not need a Postgres client. We removed both by replacing `cv::imread`/`imwrite` with **stb_image** (single-header) and `cv::dnn::NMSBoxes`/`blobFromImage` with a hand-written NMS and a manual NCHW pack. That drops the OpenCV surface to **core + imgproc only**: **231 → 10 libraries, 129 → 35 MB**, at a cost of a **0.087%** detection-count difference (stb vs OpenCV JPEG decoders diverge by sub-LSB pixel values on a handful of borderline boxes) — well inside tolerance. On Linux the binary is `$ORIGIN`-rpath'd and verified to run with no `LD_LIBRARY_PATH`; on Windows the MSVC runtime is bundled so targets need no VC++ Redistributable.

## 5. Accuracy validation

### 5.1 Methodology — one metric harness for everything

Every model — PyTorch, ONNX, NCNN, MNN, INT8 — is scored through a single path: predictions at **conf 0.001, NMS iou 0.7, multi-label, cap 300** (ultralytics `val` settings), fed to ultralytics' own `DetMetrics` + `box_iou` + `match_predictions` (`eval_map.py`). This guarantees the numbers are comparable across formats and directly comparable to the ultralytics reference, rather than four subtly different mAP implementations. ONNX/ncnn predictions are produced by the C++ runtime (`--save-txt`); MNN by a Python runner replicating the identical decode.

### 5.2 Results — 548 val images (> 500 requirement)

| Model | mAP50 | mAP50-95 | Δ mAP50-95 vs PyTorch |
|---|---|---|---|
| **PyTorch (reference)** | 0.3504 | 0.2036 | — |
| ONNX | 0.3495 | 0.2034 | **−0.02%** |
| NCNN | 0.3495 | 0.2034 | **−0.02%** |
| MNN  | 0.3495 | 0.2034 | **−0.02%** |
| INT8 (mixed) | 0.3377 | 0.1952 | **−0.84%** |

All three FP32 export formats land on **identical** mAP (0.2034) — as they should, being the same graph — at **−0.02%** from PyTorch, 25× inside the 0.5% target. INT8 is **−0.84%**, inside the 1.0% target. (The INT8 mAP50 drop is larger, −1.27%, reflecting slightly softer classification confidences at INT8; the budget is defined on mAP50-95, which passes.)

### 5.3 Numerical parity — isolating format from pipeline

Because the FP32 formats share a graph, we verify fidelity directly rather than only through mAP. Feeding **identical letterboxed inputs** to MNN and the source ONNX across 100 val images yields **max|Δ| = 0.096, mean|Δ| = 9.7e-05** on the raw `[1,14,8400]` output. The max is a single box-coordinate least-significant bit (coordinates run to ~640; 0.096 px is nothing); the mean is negligible. Detection counts over the full set are effectively equal (ONNX 157,464 vs ncnn 157,465 at conf 0.001). This distinguishes *format equivalence* from *coincidentally similar mAP*.

The same discipline caught a **false alarm** on the CUDA path: a raw `max|Δ| = 2.31` looked alarming until it was traced to FP32 box-coordinate variance in a single anchor — functional mAP was identical. A naive "max-abs-diff < ε" gate would have failed a correct model; the box-vs-class decomposition is what makes the comparison meaningful.

## 6. Latency and throughput

Per-frame inference, VisDrone val:

| Platform | Backend | infer (ms) | FPS |
|---|---|---|---|
| Windows 11 CPU | ONNX (ORT) | 37.6 | **25.4** |
| Windows 11 CPU | NCNN | 80.1 | 12.2 |
| Linux CPU (4-thread) | ONNX (ORT) | 40.0 | 25.0 |
| Linux CPU (4-thread) | **MNN** | 74.0 | 13.5 |
| Linux CPU (4-thread) | NCNN | ~80 | ~12.5 |
| Linux CPU (4-thread) | ONNX INT8 (mixed) | 137 | 7.2 |
| Linux H200 | **ONNX CUDA (C++)** | 7.8 | **~128** |
| Jetson Orin Nano 4GB | **TensorRT FP16** | 27.8 | **35.7** |

The ordering is consistent and explicable: **ORT is ~2× faster than MNN and NCNN on x86** because it is heavily x86/AVX-tuned, while MNN and NCNN are mobile/ARM-first runtimes — which is exactly why both are carried forward for the Orin, where that ranking is expected to invert. CUDA delivers a ~5× step over CPU. **INT8 is the slowest CPU row (2.8× slower than FP32 ONNX), for the reasons in §3.6** — a reminder that INT8 is a *hardware*-dependent optimization, not a free win. On x86 CPU no format beats ORT, so the "best export format" is platform-dependent, not absolute — the reason we ship three.

## 7. Cross-platform builds and distribution

A single cross-platform **CMake** builds and runs on two platforms today: **Linux x86_64** (with the ONNXRuntime CUDA EP; C++ CUDA mAP50-95 = 0.2033, −0.03% vs PyTorch, at 7.8 ms/frame / ~128 FPS) and **Windows 11 x64** (VS 2026 / MSVC 19.5x). The Windows port surfaced three concrete portability issues, each fixed in the build system rather than worked around: `Ort::Session` takes `const wchar_t*` on Windows (a platform `ORTCHAR_T` shim); the prebuilt OpenCV config doesn't recognize the VS 2026 toolset and reports an empty runtime (point `OpenCV_DIR` at the concrete `vc16/lib` config); and the exe needs the MSVC runtime on clean targets (bundled via `InstallRequiredSystemLibraries`). Both platforms ship as **self-contained, relocatable bundles** — Linux 35 MB (`$ORIGIN`, 10 libs, verified isolated), Windows with its runtime bundled — installable by unzip.

## 8. Embedded GPU deployment: Jetson Orin

The runtime was taken to a **Jetson Orin Nano 4 GB** (JetPack 7: Ubuntu 24.04, CUDA 13.2, TensorRT 10.16.2, sm87). The same CMake produces a native aarch64 binary with a third backend — a `trt_backend` that deserializes a prebuilt engine and runs it via `enqueueV3`, joining the ONNXRuntime and NCNN backends behind the same interface. The engine is built on-device with `trtexec` from the exported ONNX.

**Result.** The FP16 engine runs at **27.8 ms/frame GPU compute (35.7 FPS)**, and on-device accuracy over the full 548 VisDrone val images is **mAP50 0.3488 / mAP50-95 0.2029 — −0.46% / −0.34% vs the PyTorch FP32 baseline** (0.3504 / 0.2036), matching the x86 ONNX result to within 0.2 mAP points. On-device mAP is scored with a dependency-free reimplementation of the same metric harness (`scripts/eval_map_standalone.py`).

**FP16, not INT8.** §3.6 reserved the INT8 *throughput* proof for this path, on the expectation that tensor-core INT8 would invert the CPU result. For this model it does not. The mixed-precision assignment from §3 keeps the attention, head, and router in higher precision, so INT8 leaves the compute-heavy area-attention on FP32/FP16 kernels; combined with TensorRT's INT8 being lossier than the ONNXRuntime path, the calibrated INT8 engine measures **0.3202 mAP50 at 21.7 FPS — slower *and* less accurate than FP16**. Where a network's dominant cost is attention that does not quantize, **FP16 is the correct embedded target**; INT8's tensor-core advantage applies to convolution-dominated models, not this one. This refines the expectation stated in §3.6.

**Build notes.** Two toolchain specifics are worth recording. On sm87 with TensorRT 10.16.2 a pure-FP16 build fails at low builder-optimization levels (the timing model references an sm80 shader that has no sm87 base); `--builderOptimizationLevel=3` selects tactics by on-device profiling instead and builds cleanly. And an ONNXRuntime-quantized QDQ model must use symmetric activations and non-quantized bias to be accepted by TensorRT's parser (`quantize_int8.py --symmetric`). The 4 GB module also needs swap for the engine *build* (inference itself uses ~20 MB). Full reproduction is in [`jetson/DEPLOYMENT_LOG.md`](jetson/DEPLOYMENT_LOG.md).

**Distribution.** A prebuilt aarch64 bundle (`jetson/30_package.sh`) ships the binary with OpenCV bundled and TensorRT/CUDA taken from JetPack; it runs on any Orin (Nano/NX/AGX) on JetPack 7, with the per-device engine built once by an included script.

## 9. Future work

- **Production drone platform — DJI Manifold 3.** VisDrone is aerial/drone imagery, so the natural production target is an onboard drone computer. [DJI Manifold 3](https://enterprise.dji.com/manifold-3) is an **NVIDIA Orin NX-based** enterprise edge computer purpose-built for drones — the exact aarch64 + TensorRT path above deploys onto it directly. Validating this pipeline on the Manifold 3 exercises **real-time on-drone inference in operational conditions** (aerial surveillance, infrastructure inspection, search-and-rescue), closing the loop from VisDrone training to production drone edge deployment.

---

*Reproducibility:* the C++ runtime, all scripts (`quantize_int8.py`, `eval_map.py`, `eval_map_standalone.py`, `mnn_val.py`, `mnn_parity.py`, `package_linux.sh`), and the Jetson kit (`jetson/`, incl. `DEPLOYMENT_LOG.md`) are in the repository above; the exported models and prebuilt bundles (Linux, Windows, Jetson Orin) are attached to the [Releases](https://github.com/skywalker-lt/yolo-master-edge/releases) page.
