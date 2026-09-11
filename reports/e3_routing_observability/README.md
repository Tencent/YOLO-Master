# E3 Routing Observability

This package completes the Rhino-Bird E3 P0/P1 scope: it normalizes the existing MoE, MoT, and Latent routing
snapshots, records them during training, writes JSON and TensorBoard scalars, and presents the same evidence in a
read-only local dashboard. It does not change routing algorithms or model `forward` methods.

## Dashboard

```bash
.venv/bin/python scripts/e3_routing_dashboard.py
```

Open `http://127.0.0.1:8000/`. The committed example combines 13 routed layers, live-training validation, three static
expert-load figures, and the formal CUDA overhead result. Refreshing the page discovers new telemetry under `runs/`.
The server binds to loopback by default and rejects path traversal, out-of-root symlinks, and oversized artifacts.

## Reproduce

Generate a fresh three-family snapshot and static figures under ignored `runs/`:

```bash
.venv/bin/python scripts/e3_routing_snapshot.py --data coco8.yaml --device auto
```

Verify real training-time JSON and TensorBoard logging on COCO8:

```bash
uv pip install --python .venv/bin/python tensorboard
.venv/bin/python scripts/e3_routing_validation.py --device 0
```

The training validation uses an isolated Ultralytics settings directory, trains all three families for two epochs, and runs
a telemetry-off control. It passes only when every routing tag contains steps 1 and 2, final TensorBoard values match
JSON, and disabling telemetry preserves deterministic training values while writing no routing data. Use
`--device cpu` when CUDA is unavailable.

Re-run the formal paired overhead protocol:

```bash
.venv/bin/python scripts/e3_overhead_benchmark.py \
  --data african-wildlife.yaml --device 0 --native-data \
  --rounds 5 --warmup 50 --steps 200 --routing-interval 100 \
  --imgsz 640 --batch 2 --workers 4 --output runs/e3_overhead_cuda
```

Generated checkpoints, events, telemetry, and temporary settings stay below ignored `runs/`; only the curated example
in `results/` is versioned.

## Training interface

Routing collection is opt-in and reuses `TrainingTelemetry`:

```bash
YOLO_TRAIN_TELEMETRY=1 \
YOLO_TRAIN_TELEMETRY_ROUTING_INTERVAL=100 \
.venv/bin/yolo train \
  model=ultralytics/cfg/models/26/yolo26-master-n.yaml \
  data=coco8.yaml epochs=2 imgsz=64 batch=2 device=cpu workers=0 \
  project=runs/e3_training name=moe_observability plots=False
```

The run directory receives `telemetry.json`, `telemetry_rank_<rank>.json`, and, when the existing TensorBoard
integration is enabled, `events.out.tfevents.*`. Numeric routing metrics use the `routing/` TensorBoard namespace;
JSON remains authoritative for status strings and error reasons.

Environment contract:

- `YOLO_TRAIN_TELEMETRY`: disabled by default; `1` registers telemetry callbacks.
- `YOLO_TRAIN_TELEMETRY_ROUTING_ENABLED`: enabled within telemetry by default; `0` keeps timing but disables routing.
- `YOLO_TRAIN_TELEMETRY_ROUTING_INTERVAL`: sampling interval, default and minimum `1`.
- `YOLO_TRAIN_TELEMETRY_LOSS_STEPS`: retained leading loss observations, default `20`.
- `YOLO_TRAIN_TELEMETRY_WARMUP_STEPS` and `YOLO_TRAIN_TELEMETRY_MAX_STEPS`: benchmark window, default `0`.
- `YOLO_TRAIN_TELEMETRY_RAW_STEPS`: disabled by default; enables per-step timings for formal benchmarks.

## Evidence and baselines

- `schema.md` defines `e3.routing_snapshot.v1`, aux states, and TensorBoard keys.
- `results/route_stats.json` and three PNGs provide the P0 three-family example.
- `results/live_training_summary.json` records the clean RTX 4060 online logging gate.
- `results/overhead_summary.json` retains 30 runs with mean, median, p95, throughput, memory, pair deltas, and 95% CI.
- `results/dataset_provenance.json` records the formal dataset source and split fingerprints.
- `results/checksums.sha256` covers every committed evidence file.

The admission baseline is `e9ac08b2`. The PR is based on Tencent `main` at `af961b9`. The formal overhead result is
attributed to its actual clean commit, `bad5b33`: mean overhead is -1.821% (MoE), 0.074% (MoT), and 2.671% (Latent),
with a highest paired-bootstrap 95% CI upper bound of 5.618%; all pass the predefined 10% gate. Negative overhead is
treated as measurement noise, not acceleration. The complete 50-epoch package remains on
`archive/e3-routing-evidence-20260909` at `6d8f000` and is intentionally excluded from this PR.

## Limitations

- Only MoE, MoT, and Latent are normalized; other routed families are reported as `unsupported`.
- COCO8 smoke results validate collection and logging, not accuracy or semantic expert specialization.
- The formal overhead conclusion applies only to its recorded RTX 4060/WSL2, FP32, batch-2 configuration.
- The committed P0 figures use the admission MPS snapshot; current CUDA training evidence is machine-readable.
- MoT CUDA kernels warn that a fixed seed does not guarantee bitwise determinism.
- The dashboard is a local evidence viewer, not a remote job-control service.
