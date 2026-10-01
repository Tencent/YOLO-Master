# YOLO-Master Edge API Server

`yolomaster_server` is a C++ HTTP/1.1 + WebSocket inference service built on the same runtime
core as the `yolomaster_edge` CLI. One process serves any number of models, each on one of the
four backends (ONNX Runtime, TensorRT, ncnn, MNN), on CPU or GPU, in fp32, fp16 or int8, batch 1
per request. Every request returns the per-stage timings the runtime measures, and the server
exposes Prometheus metrics and rolling latency percentiles.

## Run

```
yolomaster_server --config deploy/server.json
yolomaster_server -p 8080 \
  -m v01n-trt=/opt/yolomaster/models/v01n/model.onnx,backend=trt,device=cuda,precision=fp16 \
  -m v01n-ncnn=/opt/yolomaster/models/v01n/ncnn,backend=ncnn,threads=8,workers=4
```

Model spec keys (CLI `-m id=path,key=value,...` or the `models[]` entries of the JSON config):

| key | values | default | notes |
|---|---|---|---|
| `path` | `.onnx`, `.engine`, `.mnn`, `<dir>_ncnn`, `model.ncnn.param` | required | `.onnx` with `backend=trt` builds and caches an engine |
| `backend` | `auto`, `onnx`, `trt`, `ncnn`, `mnn` | `auto` (from the extension) | |
| `device` | `cpu`, `cuda`, `vulkan`, `opencl` | `cpu` | onnx: CUDA EP; ncnn: Vulkan; mnn: CUDA or OpenCL |
| `precision` | `auto`, `fp32`, `fp16`, `int8` | `auto` | ncnn int8 loads the `-int8_ncnn` sibling; trt fp16 sets the builder flag |
| `threads` | int | 4 | intra-op threads per worker (CPU backends) |
| `workers` | int | 0 = auto | worker threads, one Backend instance each (GPU: 1, CPU: min(4, cores/threads)) |
| `imgsz` | int | model metadata | fixed-shape models override |
| `conf`, `iou`, `max_det` | float, float, int | 0.25, 0.5, 300 | per-request overrides allowed |
| `classes` | `auto`, `visdrone`, `sku` | `auto` | class names, else from model metadata |
| `preload` | bool | true | `false` = load on first request; `/readyz` only waits for preloaded models |
| `slicing`, `tile_size` | `off`/`dense`/`sparse`, int | `off`, 0 | Sparse SAHI tiled inference |
| `preproc` | `gpu`, `cpu` | `gpu` | TensorRT / ORT-CUDA: letterbox + normalize on the GPU (builds with `USE_CUDA_PREPROC`; `cpu` = the reference path) |
| `cuda_graph` | bool | false | TensorRT: capture the per-frame stream work into a CUDA graph and replay it |

Server options (JSON keys and CLI flags): `port`, `host`, `loop_threads` (event-loop threads,
default 2), `max_body_mb` (32), `max_pixels` (50 M), `max_queue` (64 pending jobs per model, then
503), `request_timeout_ms` (10000, queue plus inference deadline, then 504), `ws_max_payload_mb`
(16), `engine_cache_dir`, `cors`, `access_log`. `--check-config` validates and exits;
`--print-config` prints the effective config.

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/healthz` | process liveness (`503` while shutting down) |
| GET | `/readyz` | all preloaded models warmed up (`503` + reason otherwise) |
| GET | `/v1/models` | model cards: backend, execution provider, imgsz, class count, workers, queue depth |
| POST | `/v1/models/{id}/load`, `/unload` | start or stop a model's worker pool |
| POST | `/v1/infer` | one image, one result |
| POST | `/v1/infer/batch` | K files in one multipart body, K independent batch-1 jobs |
| POST | `/v1/video` | video file upload, NDJSON stream of per-frame results (`track=` for ids) |
| WS | `/v1/stream` | binary frames in, JSON per frame out, keep-latest backpressure |
| POST | `/v1/bench` | probe run on one worker of a loaded model, `yolomaster-bench/v1` JSON |
| GET | `/v1/stats` | rolling p50/p90/p95/p99 latency (1 min, 5 min windows) per model |
| GET | `/metrics` | Prometheus text exposition |
| GET | `/openapi.json` | OpenAPI 3.1 document generated from the route table |

### POST /v1/infer

Body: the image bytes (`Content-Type: image/*`, JPEG/PNG/BMP/GIF/TGA/PSD) or `multipart/form-data`
with a `file` field. Query parameters:

| parameter | meaning |
|---|---|
| `model` | model id (optional when only one model is configured) |
| `conf`, `iou`, `max_det`, `multi_label` | detection parameters; defaults from the model spec |
| `slicing`, `tile_size` | `dense` or `sparse` tiled inference for large images |
| `return` | `json` (default), `txt` (`class conf x1 y1 x2 y2` per line, the CLI's `--save-txt` format), `coco` (`[{image_id, category_id, bbox, score}]`), `annotated` (JPEG) |
| `image_id`, `coco91` | for `return=coco`: image id, map 80-class ids to COCO 91 ids |
| `masks=overlay` | segmentation models: composite masks into the annotated JPEG |
| `mask_coeffs=1` | include raw mask coefficients in JSON detections |
| `names=0` | omit class names in JSON |
| `quality` | JPEG quality for `annotated` (90) |

JSON response:

```json
{
  "request_id": "000000000000002a", "model": "v01n-trt", "backend": "TRT-CUDA-fp16",
  "image": {"width": 640, "height": 427},
  "params": {"conf": 0.25, "iou": 0.5, "max_det": 300, "imgsz": 640, "multi_label": false},
  "count": 3,
  "detections": [{"class_id": 0, "name": "person", "conf": 0.912, "box": [x1, y1, x2, y2]}],
  "timings_ms": {"queue": 0.02, "decode": 2.1, "pre": 0.4, "infer": 1.9, "post": 0.1, "encode": 0, "total": 4.6},
  "seg": false, "worker": 0
}
```

Boxes are pixel `x1 y1 x2 y2` in the original image. `timings_ms.total` is server time from
dequeue to serialization (decode + pre + infer + post + encode); `queue` is the wait before it.
Headers: `X-Request-Id`, `X-Infer-Ms`, `X-Detections` (annotated).

Errors are JSON `{"error", "code", "request_id"}`: `400` bad image or parameter, `404` unknown
model or route, `413` body over `max_body_mb`, `503` queue full or model loading (`Retry-After`),
`504` deadline exceeded, `500` inference error (the worker rebuilds its backend after three
consecutive failures).

### POST /v1/infer/batch

`multipart/form-data` with any number of file parts. Each file is an independent batch-1 job
spread across the model's workers; the response is `{"count", "results": [per-file JSON with
"file"]}` in the request order. This is deliberately not adaptive batching: real-time targets
are batch 1, and multi-image throughput comes from workers, not from padding a batch.

### POST /v1/video

Body: a video file (any container OpenCV can decode). Query: `model`, `every=N` (process every
Nth frame), `max_frames`, plus the detection parameters. Response: `application/x-ndjson`, one
JSON object per processed frame with `frame` (index), then a final `{"done": true, "frames",
"decoded"}` line. Frames are processed sequentially in order with one job in flight.

Tracking: `track=botsort|bytetrack` on `/v1/video` and `/v1/stream` runs a multi-object tracker
over the frames of that request or connection (one tracker per video, one per WebSocket
session), and every detection gains a `track_id`. `botsort` (default recommendation) adds
sparse-optical-flow camera motion compensation; `bytetrack` is the same association without it.
Unless `conf` is given, the detector threshold drops to the tracker's low threshold (0.1) so the
second association stage has candidates; boxes are the tracker's Kalman estimates. The NDJSON tail
of `/v1/video` echoes `"track"`; a stream connection switches with a text message
`{"track": "bytetrack"}` (new ids) or `{"track": "off"}`.

Slicing per request: `slicing=off|dense|sparse&tile_size=N` on every inference endpoint runs the
Sparse SAHI mode of the runtime (global pass plus tiles) for that request; the response carries
`slicing.tiles_run` / `slicing.tiles_total`. A model spec `slicing=` sets the default.

### WS /v1/stream

`ws://host/v1/stream?model=<id>[&conf=..&iou=..&return=json|annotated]`. Send binary messages
(encoded frames, JPEG recommended). Each frame answers with a JSON text message
(`seq`, `count`, `detections`, `timings_ms`, `dropped`); `return=annotated` sends the annotated
JPEG as a binary message before the JSON. While a frame is being processed, newer frames replace
the one pending frame (keep-latest), so a camera can push at any rate and the stream stays live;
`dropped` counts the frames skipped that way. A text message with JSON `{"conf": 0.3, "iou":
0.5, "max_det": 100, "return": "json"}` updates the parameters for that connection.

### POST /v1/bench

`POST /v1/bench?model=<id>&warmup=10&iters=50` (empty body). One worker of the model runs
`warmup` untimed and `iters` timed forwards of a gray 114 probe at the model's imgsz
(`forward_raw` without decoding on ONNX Runtime and ncnn, a full `infer()` on TensorRT and MNN,
reported as `cold.probe_mode`) and answers the `yolomaster-bench/v1` document the CLI's `--bench`
writes: `model`, `environment` (host, CPU, GPU, threads, commit, build flags, version), `protocol`,
`cold.infer_ms{n, mean, median, p90, p95, p99, min, max}` (floor-rank percentiles, the phone Bench
tabs' convention) plus `request_id` and `worker`. The run holds that worker for its duration;
other workers keep serving. Accuracy is not measured over HTTP: use the CLI's `--accuracy`.
`tool` names the producer: `cli`, `server`, `macos` (the Core ML runner's CLI and app, whose
`sustained` block adds a `thermal` array, one sample per second, in the bands Cool / Normal /
Hot / Critical: die-temperature bands from the SMC in the app, ProcessInfo pressure states
mapped onto the same names in the CLI), `android`, `ios`. Any document
can be checked with `scripts/bench_schema_check.py`.

Optional blocks a producer may add (absent elsewhere, never required): `model.moe_export`
(`ane_safe_rank` or `topk_gather`, how a MoE package was exported) and, from the macOS app,
`host_meters` (per-cell, one sample per second: `seconds`, `thermal_start` / `thermal_peak` in the
app's die-temperature bands Cool / Normal / Hot / Critical, `die_celsius_*` start / peak / mean
and the `die_celsius` track when the SMC sensor exists, `power_state` and `power_w_*` mean / min /
max plus the `power_w` track when a battery exists, positive = charging). The app's "Export run"
writes every cell of a run as one JSON array of such documents.

### GET /metrics

Prometheus exposition: `yolomaster_requests_total{model,code}`, `yolomaster_queue_depth{model}`,
`yolomaster_busy_workers{model}`, `yolomaster_workers{model}`, `yolomaster_stage_seconds`
histograms with `stage=queue|decode|pre|infer|post|encode|total` (buckets from 0.5 ms to 10 s),
`yolomaster_ws_dropped_frames_total{model}`, `yolomaster_gpu_memory_bytes{kind}` (CUDA builds),
`yolomaster_uptime_seconds`.

## Authentication

Off by default. Configure keys with `"api_keys": ["..."]` in the config file, `YM_API_KEYS=k1,k2`
in the environment, or `--api-key-file <file>` (one key per line; each source replaces the
previous one, keys shorter than 16 characters are refused at startup). With keys set, every
route except `/`, `/healthz`, `/readyz`, `/openapi.json` and CORS preflight needs
`X-API-Key: <key>` or `Authorization: Bearer <key>`; the WebSocket upgrade is checked the same
way. Failures answer `401` with `WWW-Authenticate` and count in `yolomaster_auth_failed_total`.
Keys are compared in constant time over the whole list. `auth_exempt` in the config extends the
open paths. `--print-config` reports only the number of keys.

## Rate limiting

`"rate_limit": {"rps": 5, "burst": 10}` (or `--rate-rps` / `--rate-burst`) enables a token
bucket per client: per API key when authenticated, else per peer address. One bucket map is
shared by all event-loop threads. Every limited response carries `X-RateLimit-Limit` (the
burst) and `X-RateLimit-Remaining`; a rejected request answers `429` with `Retry-After` and
counts in `yolomaster_rate_limited_total`. `/healthz`, `/readyz` and preflight are never
limited; a WebSocket upgrade costs one token.

## Request ids, tracing and logs

Every response carries `X-Request-Id` (an inbound value matching `[A-Za-z0-9._-]{1,64}` is
kept, anything else is replaced) and a W3C `traceparent`: an inbound header keeps its trace id
and gets a new span id, otherwise the server starts a trace. The access log (`access_log`,
stderr) is a plain line by default; `"log_format": "json"` (or `--log-format json`) writes one
object per request: `ts`, `rid`, `trace_id`, `span_id`, `parent_span`, `method`, `path`,
`status`, `bytes`, `ms`, `client`, and for inference `model`, `worker` and the `stages`
(queue, decode, pre, infer, post, encode, total, milliseconds). An OpenTelemetry collector
ingests it with the filelog receiver and a `json_parser` operator whose `timestamp` reads `ts`
(RFC 3339 with milliseconds) and whose `trace` block maps `trace_id` / `span_id`.

## OpenAPI

`GET /openapi.json` serves an OpenAPI 3.1 document rendered from the same route table the
server registers its handlers from, so it cannot drift: paths, query parameters, request
media types, response schemas (`InferResult`, `Detection`, `BatchResult`, `BenchResult`,
`ModelsList`, `Stats`, `Error`, ...), the security schemes, and the WebSocket entry marked
`x-websocket: true`. `GET /` lists the same endpoints.

## Clients

`clients/python/yolomaster_client.py` (stdlib only; `opencv-python` and `websocket-client`
for the video helpers):

```python
from yolomaster_client import Client
c = Client("http://localhost:8080")
c.wait_ready()
r = c.infer("img.jpg", model="v01n-trt", conf=0.3)          # dict
txt = c.infer("img.jpg", model="v01n-trt", ret="txt")      # str
jpg = c.infer("img.jpg", model="v01n-trt", ret="annotated") # bytes
rs = c.infer_many(["a.jpg", "b.jpg"], model="v01n-trt")
for fr in c.video("clip.mp4", model="v01n-trt", every=2): ...
for fr in c.stream_video("clip.mp4", model="v01n-trt"): ...  # WebSocket
```

`clients/python/bench_client.py` is the closed-loop benchmark used for the API-vs-CLI
comparison (`--concurrency`, `--dump-txt`, `--summary`).

## Operational notes

- One `Backend` instance per worker thread; workers never share a TensorRT context or an ORT
  session. GPU models default to one worker (the device serializes anyway); raise `workers` only
  after measuring.
- TensorRT engines built from `.onnx` are cached under `engine_cache_dir` keyed by ONNX content,
  GPU name, TensorRT version and precision. The first load of a model builds (seconds to minutes);
  `/readyz` stays `503` until warm-up completes. Mount the cache directory as a volume in Docker.
- `SIGTERM`/`SIGINT`: stop accepting, drain queued jobs up to `--drain-ms`, fail the rest with 503,
  join workers, exit 0.
- Decoding happens on worker threads, never on the event loops; body and pixel caps bound memory.
- TLS is not built in: terminate it at a reverse proxy.
