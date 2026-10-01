# GPU preprocessing, CUDA graphs and MNN CUDA fp16: v1.2.0 measurements

MEASURED on an NVIDIA RTX PRO 4500 Blackwell (driver 580.159, sm_120), TensorRT 10.16.1 (cuda 12.9),
ONNX Runtime 1.20.1 GPU build, MNN 3.6.1 built with `MNN_CUDA` for sm_120, runtime commit 5f212b8
built with `-DUSE_CUDA_PREPROC=ON`. Images: `visdrone50` (50 VisDrone val images) for parity and the
per-stage table, `datasets/coco500` (every 10th COCO val2017 image, 500 images, 497 labelled) for
accuracy; the accuracy numbers are the runtime's in-process mAP at the val protocol (conf 0.001,
iou 0.7, multi-label, max_det 300), which equals `scripts/eval_map.py` on a txt dump to four decimals.
Raw logs and bench JSONs: `deploy/pod/validate_gpu.sh` and `validate_gpu_accuracy.sh` outputs
(`/root/validate_gpu*` on the pod; copies in the release assets of the v1.2.0 tag).

## 1. What changed in the timing contract

Until 1.1.1, `pre_ms` was the CPU letterbox plus the BGR to RGB / 255 / NCHW loop, and on TensorRT
`infer_ms` included the float H2D copy. With `preproc=cuda` (the default in a `USE_CUDA_PREPROC`
build) the raw uint8 frame is copied to the device (3x less than the float tensor) and one kernel
writes the input tensor; `pre_ms` = host staging + raw H2D + kernel, `infer_ms` = enqueue / Run + D2H
(events on the backend's stream). Every table row below states its preprocessing path; do not
compare a `preproc=cuda` row with a 1.1.1 row stage by stage, only end to end.

## 2. Kernel parity with the CPU reference

`preproc_parity` over visdrone50 at 640: worst |diff| 0.002911 (letterbox) and 0.002939 (stretch)
against the 1/255 = 0.003922 limit, 0 failures. The CUDA kernel samples bilinearly in float with
half-pixel centres; cv::resize uses fixed-point coefficients, so bit equality was never the contract.

## 3. Accuracy: GPU preprocessing is neutral (coco500, mAP50-95)

| model | path | cpu preproc | cuda preproc | cuda preproc + graph |
|---|---|---|---|---|
| EsMoE-N | TensorRT fp16 | 0.4354 | 0.4352 | 0.4352 |
| EsMoE-N | ORT-CUDA | 0.4351 | 0.4350 | |
| v0.1-N | TensorRT fp16 | 0.4309 | 0.4311 | 0.4311 |
| v0.1-N | ORT-CUDA | 0.4309 | 0.4310 | |

The 0.0001 to 0.0002 deltas go both ways: the noise floor of a 1/255 input perturbation through an
fp16 engine. Per-box txt parity is therefore NOT the right gate for this change: at conf 0.001 a few
hundred marginal boxes flip per 50 images, at conf 0.25 three to four boxes per 50 images
(`parity_txt.py --tol 1.0`: 617 / 619 and 639 / 640 matched), and mAP does not move.

## 4. Latency per stage (visdrone50, medians, ms, batch 1, 640)

| model | path | preproc | pre | infer | post | total | probe (bench) |
|---|---|---|---|---|---|---|---|
| EsMoE-N | TensorRT fp16 | cpu | 2.54 | 1.22 | 0.91 | 4.67 | 1.05 |
| EsMoE-N | TensorRT fp16 | cuda | 0.08 | 1.07 | 0.63 | 1.78 | 1.05 |
| EsMoE-N | TensorRT fp16 | cuda + graph | 0.07 | 0.83 | 0.61 | 1.51 | 0.82 |
| v0.1-N | TensorRT fp16 | cpu | 1.66 | 1.18 | 0.65 | 3.49 | 1.17 |
| v0.1-N | TensorRT fp16 | cuda | 0.08 | 1.20 | 0.64 | 1.91 | 1.18 |
| v0.1-N | TensorRT fp16 | cuda + graph | 0.08 | 0.97 | 0.63 | 1.67 | 0.96 |
| EsMoE-N | ORT-CUDA | cpu | 0.62 | 3.44 | 0.65 | 4.72 | 3.41 |
| EsMoE-N | ORT-CUDA | cuda (IoBinding) | 0.08 | 3.22 | 0.65 | 3.95 | 3.13 |
| v0.1-N | ORT-CUDA | cpu | 0.70 | 3.82 | 0.64 | 5.17 | 3.75 |
| v0.1-N | ORT-CUDA | cuda (IoBinding) | 0.08 | 3.60 | 0.64 | 4.33 | 3.57 |
| EsMoE-N | MNN CUDA fp32 | cpu | 1.25 | 1.87 | 7.07 | 10.20 | 1.80 |
| v0.1-N | MNN CUDA fp32 | cpu | 0.75 | 2.24 | 9.88 | 12.88 | 2.17 |

Readings. The preprocessing stage drops from 1.7 to 2.5 ms to 0.08 ms on TensorRT and from 0.6 to
0.7 ms to 0.08 ms on ORT (the CPU rows of ORT were already cheaper because ORT's own copy sits in
`infer`). End to end, TensorRT fp16 goes from 4.67 to 1.78 ms (EsMoE-N) and 3.49 to 1.91 ms
(v0.1-N); ORT-CUDA from 4.72 to 3.95 and 5.17 to 4.33. The CUDA graph removes a further 0.2 to
0.25 ms of launch overhead on TensorRT (probe 1.05 to 0.82 ms, 1.17 to 0.96 ms) at identical
detections (byte-identical txt dumps, graph vs no graph). In graph mode `infer_ms` includes the
preprocessing kernel (the events bracket the replay). MNN's post stage is the decode of its
channel-major output on the CPU and is the same in both paths; MNN CUDA does not use the kernel.

Compared with the 1.1.1 server table on the L40S (TensorRT fp16 v0.1-N 1.8 ms model time inside a
7.7 ms request), the model time is the same class of number on this card and the preprocessing that
used to dominate the request is gone from the runtime side; the server's remaining per-request cost
is JPEG decode (stb, CPU) and HTTP.

## 5. MNN CUDA fp16: the verdict is a documented downgrade

Three ways to run a routed YOLO-Master model in fp16 on MNN's CUDA backend were tried:

| variant | EsMoE-N coco500 mAP50-95 | v0.1-N | outcome |
|---|---|---|---|
| `model.mnn`, session `Precision_Low` (the 1.1.1 behaviour) | 0 detections (1.1.1 bug) | 0 | now refused: `fp16_safe: false` in the sidecar, runs fp32 with an `ep_note` |
| C2: `model-fp16-routed.mnn` converted from the routing-protected fp16 ONNX (Cast nodes keep the router fp32), `Precision_Low` | 0.0000 (0 detections) | hung, killed | the CUDA backend computes every float tensor in half regardless of the graph's Casts |
| C1: multi-path session, fp32 routing segments (`model.paths.json`, `YOLOMASTER_MNN_MULTIPATH=1`) | 0.0000 (0 detections, post 695 ms) | not run | MNN 3.6.1's Tensor-mode paths return wrong outputs even on CPU (299 vs 5581 dets) |
| fp32 (`Precision_High`) | 0.4351 | 0.4309 | correct; 1.87 / 2.24 ms model time |

Shipped behaviour: a `--precision fp16` request on a routed model under MNN CUDA runs fp32 and says
so in `ep_note` ("fp16 refused: routed model (metadata fp16_safe: false); running fp32"); the
`fp16_safe` stamp is written by `scripts/server/prepare_models.py` when the ONNX carries `/routing/`
nodes. Dense models (no routing) keep fp16. The multipath path stays in the tree behind the
environment variable as the experiment record.

## 6. Reproduce

```
bash deploy/pod/bootstrap_l40s.sh                       # TensorRT, cuDNN, CUDA 12.9 runtime + nvcc, FFmpeg dev
SM=120 BUILD_ROOT=/root/build bash deploy/pod/build_gpu.sh    # MNN CUDA, runtime + server, parity, both suites
SRC=models OUT=/root/models_api bash deploy/pod/prepare_models_pod.sh
MODELS=/root/models_api OUT=/root/validate_gpu  bash deploy/pod/validate_gpu.sh
MODELS=/root/models_api OUT=/root/validate_gpu2 bash deploy/pod/validate_gpu_accuracy.sh
```
