# Changelog

Two version numbers: the runtime, CLI and repository tag (`VERSION`) and the API server image
(`cpp/server/SERVER_VERSION`), because the server first shipped as image tag 1.2.0 on top of
runtime 1.1.1 and that image stays as published.

## Runtime 1.2.0 / server 1.2.1 (in progress: Linux phase complete, macOS phase on dev/v1.2.0-mac)

macOS Core ML Runner 1.2.0 (`mac/`, details in `mac/RELEASE_NOTES-1.2.0.md` and
`mac/CHANGELOG-1.2.0.md`)
- The OpenCV-free parts of the C++ core (in-process mAP, BoT-SORT / ByteTrack, bench statistics)
  now live in `mac/Sources/YOLOMasterCore` and are compiled by both CMake (`yolomaster_ccore`)
  and the Swift package; Linux output proven identical.
- Bench mode (cold / sustained, `yolomaster-bench/v1` with `tool: macos`), in-process accuracy
  at the val protocol (the first Core ML mAP number), `--save-txt` in the C++ format, tracking
  on video (Vision-based camera motion for BoT-SORT), Metal GPU preprocessing with a CPU
  fallback, `scripts/bench_schema_check.py` (shared with Linux T19), `scripts/preproc_compare.py`,
  `mac/tests/run_mac_tests.sh`.

Runtime and CLI
- One `preprocess_nchw` for every backend (letterbox, BGR to RGB, /255, NCHW); TensorRT and MNN
  keep their staging buffers; output byte-identical to 1.1.1.
- ncnn: fp32 is pinned per layer (the 1e30 mask and 1e-9 nudge chains of the emulated MoE
  routers, via `Layer::featmask`) instead of refusing fp16 for the whole model; metadata
  `fp16_mixed: false` vetoes.
- MNN: an fp16 request on a routed model (metadata `fp16_safe: false`) is downgraded to fp32 and
  explained in `ep_note` instead of returning zero detections; experimental multi-path session
  behind `YOLOMASTER_MNN_MULTIPATH` (not correct on MNN 3.6.1, see the source comment).
- Bench mode: `--bench cold|sustained`, `--bench-json`, one `yolomaster-bench/v1` schema (floor
  rank percentiles, the phones' convention); `--accuracy` scores the source with an in-process
  mAP that equals `scripts/eval_map*.py` to four decimals; `yolomaster_score` scores txt dumps.
- Multi-object tracking: `--track botsort|bytetrack` on video sources (Kalman, two-stage IoU
  association, BoT-SORT camera motion compensation behind `USE_GMC`), ids in the annotated
  video and as a 7th txt column.
- `scripts/make_coco_subset.py` (coco500) and `scripts/package_eval_sets.sh` for the labelled
  eval sets (release assets).
- `--version` on both binaries; bench JSON carries version, commit and build flags.
- GPU preprocessing (`USE_CUDA_PREPROC`): one CUDA kernel (letterbox, BGR to RGB, /255, NCHW) from
  the raw uint8 frame into the TensorRT input tensor or an ORT IoBinding on the CUDA EP; TensorRT
  CUDA-graph replay (`--cuda-graph`, spec `cuda_graph=1`). RTX PRO 4500: TensorRT fp16 end to end
  4.67 to 1.78 ms (EsMoE-N) and 3.49 to 1.91 ms (v0.1-N), graph a further 0.2 ms, coco500 mAP
  unchanged to 0.0002; ORT-CUDA 4.72 to 3.95 and 5.17 to 4.33 ms. `GPU_PREPROC_RESULTS.md`.
- MNN CUDA fp16 on routed models: every fp16 variant (session precision, routing-protected
  conversion, multi-path session) returns zero detections on MNN 3.6.1, so the request is
  downgraded to fp32 with an `ep_note` instead of failing silently (the 1.1.1 bug).
- Phone graph: the pruned v0.1-N ships as the dense ncnn export (fused SDPA, native gate
  broadcast; x86 parity with the ONNX path 261/261 detections, the stock export was not) with a
  re-quantized ACIQ INT8 sibling (139/145 layers, 2048 COCO train images): coco500 in-process
  mAP50-95 fp32 0.4309, INT8 0.4205 (-1.04 AP, x86 ncnn); the S26 re-timing is the Android phase.

API server 1.2.1
- `track=` on `/v1/video` and `/v1/stream` (`track_id` per detection), `POST /v1/bench`,
  per-request slicing documented and tested.
- API-key authentication (`api_keys`, `YM_API_KEYS`, `--api-key-file`), token-bucket rate
  limiting (`rate_limit`), inbound `X-Request-Id` and W3C `traceparent` propagation, JSON access
  log (`log_format: json`), `GET /openapi.json` rendered from the route table, counters
  `yolomaster_rate_limited_total` and `yolomaster_auth_failed_total`.
- Image tag from `SERVER_VERSION` (`deploy/docker/version.bzl` written by `stage_bundle.sh`);
  `YM_API_KEYS` passthrough in `deploy/run.sh` and `docker-compose.yml`.
- Tests: `tests/run_server_tests.sh` boots a second, hardened instance for
  `tests/server/test_hardened.py`; CPU-only GitHub Actions workflow.
- Image `skywalker0501/yolomaster-api:1.2.1` (digest `sha256:65771d28...`, 5.26 GB) and the
  full COCO val2017 comparison rerun with it on an NVIDIA L4 (21 cells, GPU preprocessing on
  the TensorRT and ORT-CUDA rows): `API_SERVER_RESULTS.md` Tables 4 to 6. API overhead 1.6 to
  1.9 ms, every cell parity-identical to the CLI, TensorRT fp16 v0.1-N 2.29 ms model time.
- Linux bundles `yolomaster-edge-linux-x64-1.2.0.tar.gz` (CPU) and
  `yolomaster-edge-linux-x64-gpu_cuda12-1.2.0.tar.gz` (CUDA 12, TensorRT, ORT-GPU, GPU
  preprocessing for sm75 to sm90) with `SHA256SUMS-linux-1.2.0.txt`.

## 1.1.1

Previous release (see the GitHub release notes).
