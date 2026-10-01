# Deploying the YOLO-Master Edge API

Three ways to run `yolomaster_server`, from most to least packaged.

## 1. Docker image (one click)

```
bash deploy/run.sh <dockerhub-user>/yolomaster-api:1.2.0 8080     # docker run + health wait
# or
IMAGE=<dockerhub-user>/yolomaster-api:1.2.0 docker compose -f deploy/docker-compose.yml up -d
```

Requirements on the host: Docker, an NVIDIA driver, the NVIDIA container toolkit (`--gpus all`).
The image is `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04` plus one layer at `/opt/yolomaster`:

```
/opt/yolomaster/bin/yolomaster_server      the API server (ONNX Runtime CUDA/CPU, TensorRT, ncnn, MNN)
/opt/yolomaster/bin/yolomaster_edge        the CLI runner (same core, for offline batch jobs)
/opt/yolomaster/lib/                       ORT-GPU 1.20.1, TensorRT 10.16, ncnn, MNN 3.6, OpenCV-lean, ffmpeg 4.4
/opt/yolomaster/models/<id>/               model.onnx, model-fp16.onnx, model.mnn, model-fp16.mnn, ncnn/, metadata.yaml
/opt/yolomaster/server.json                default config (v01n-trt preloaded, the rest lazy)
/opt/yolomaster/cache/trt                  TensorRT engines built on first start (mount it)
```

Entrypoint modes: default = server; `cli ...` runs `yolomaster_edge` with the given arguments;
`shell` opens bash. Environment: `YM_CONFIG` (config path), `YM_ENGINE_CACHE`.

First start builds the TensorRT engines for the preloaded models (about one to three minutes on
an L40S); `/readyz` returns 503 until then, `/healthz` 200 immediately. Keep `./cache` mounted so
later starts deserialize instead of rebuilding.

CPU-only hosts: run the same image without `--gpus` and point `server.json` at the ncnn or MNN
entries (`device: cpu`); ONNX Runtime falls back to its CPU provider automatically.

## 2. Building the image on a RunPod pod (no Docker daemon)

Follows https://docs.runpod.io/tutorials/pods/build-docker-images : Bazel + rules_oci, no dockerd.

```
bash deploy/pod/bootstrap_l40s.sh            # toolchain, TensorRT 10.16 (+cuda12.9), cuDNN 9, cmake, bazelisk
bash deploy/pod/build_l40s.sh                # MNN, core + CLI + server with all four backends, tests
bash deploy/pod/prepare_models_pod.sh        # /data/models_api/<id>/... incl. MNN conversions
bash deploy/docker/stage_bundle.sh           # deploy/docker/bundle.tar (the /opt/yolomaster layer)
# Docker Hub auth without dockerd: write ~/.docker/config.json
#   {"auths":{"https://index.docker.io/v1/":{"auth":"<base64 of user:token>"}}}   (chmod 600)
DOCKERHUB_USER=<user> bash deploy/docker/push.sh both      # tarball (yolomaster-api.tar) + push
```

`deploy/docker/WORKSPACE` pins rules_oci 1.7.5 and the base image by digest; `BUILD.bazel`
defines `//:image`, `//:tarball` (docker load-able tar) and `//:push`.

## 3. Bare Linux host (no container)

```
bash deploy/pod/bootstrap_l40s.sh && bash deploy/pod/build_l40s.sh
cpp/build_l40s/server/yolomaster_server --config deploy/server.json
```

or extract the bundle tar anywhere: `tar xf bundle.tar -C / && /opt/yolomaster/entrypoint.sh`.

## Authentication, rate limit, tracing

The image runs open by default. `YM_API_KEYS=key1,key2` in the container environment (`deploy/run.sh`
and `docker-compose.yml` pass it through) turns on API-key authentication; `rate_limit`, `log_format`
and `auth_exempt` are config keys (see docs/API.md). Every response carries `X-Request-Id` and a W3C
`traceparent`; `GET /openapi.json` describes the running server.

## Configuration

See `docs/API.md` for the model spec keys and the server options. The reference config
`deploy/server.json` preloads `v01n-trt` (v0.1-N, TensorRT fp16) and registers the ONNX Runtime,
ncnn, MNN, pruned and EsMoE variants as lazy models (`preload: false`, loaded on first request or
`POST /v1/models/{id}/load`).

Sizing rules of thumb (measured on the L40S pod, see `API_SERVER_RESULTS.md`):

- GPU models: `workers: 1`; the device serializes kernels anyway and a second context only adds
  memory. Throughput scales with client concurrency up to the point where the queue fills.
- CPU models: `threads` x `workers` about equal to the physical cores you want to give the model;
  `workers: 1, threads: 8` for latency, `workers: 8, threads: 8` for throughput on a 64-core box.
- `max_queue` bounds memory (each queued job holds one encoded image) and turns overload into
  fast 503s with `Retry-After` instead of growing latency.
- `request_timeout_ms` is the queue-plus-inference deadline; stale jobs are dropped with 504.

## Monitoring

`/metrics` (Prometheus) and `/v1/stats` (JSON percentiles) are always on. A minimal scrape config:

```
scrape_configs:
  - job_name: yolomaster
    static_configs: [{targets: ["yolomaster-api:8080"]}]
```

Useful series: `yolomaster_stage_seconds_bucket{stage="infer"}` (model time),
`yolomaster_stage_seconds_bucket{stage="queue"}` (saturation), `yolomaster_requests_total{code="503"}`
(overload), `yolomaster_gpu_memory_bytes`.
