# Studio FastAPI service

`main_engine.py` and `api/v1/jobs.py` expose Studio Runtime over HTTP. Application
startup creates one `studio.JobsManager` and restores history; shutdown waits
for its cleanup. See [architecture](architecture.md) for layer responsibilities
and [API](api.md) for endpoints, response fields and error codes.

## Start the service

With Python 3.10+ in the source checkout:

```bash
python -m pip install -e ".[studio]"
python main_engine.py
```

The script binds to `127.0.0.1:8000` and runs one process. It disables proxy-header
trust and access-query logging and returns a nonzero exit on handled startup or
shutdown failure. Invalid configuration produces a fixed error without echoing
its value. Raw `uvicorn main_engine:app` uses the same lifespan but lacks the
script's failure-exit and signal-replay handling. Use one process without reload;
see [deployment limits](security.md#trusted-local-deployment).

## Embedding

`create_app()` and `app.dependency_overrides[get_jobs_manager]` support trusted
embedding and tests. An override must return the same Studio-compatible manager
throughout the lifespan; `manager_factory` supplies a custom factory. Concurrent
lifespan startup is rejected, and `get_jobs_manager` does not construct an owner
outside lifespan. HTTP clients communicate through REST without state-file access.

## Trusted configuration and existing UI entry points

Service settings use the `STUDIO_` prefix. Matching `F1_` aliases are accepted for
these suffixes: `ENGINE_HOST`, `ENGINE_PORT`, `CORS_ORIGINS`, `JOBS_STATE_PATH`,
`MODEL_ROOTS`, `DATA_ROOTS`, `OUTPUT_ROOT`, `NETWORK_INPUT_HOSTS`,
`CPU_CONCURRENCY`, `GPU_CONCURRENCY`, `MAX_PENDING_JOBS`, `STOP_GRACE_SECONDS`,
`SHUTDOWN_GRACE_SECONDS` and `HTTP_DRAIN_SECONDS`.

Different trimmed values for both names fail startup. Empty legacy CORS/root/
state-path settings retain their defaults; an explicit empty
`STUDIO_CORS_ORIGINS` disables the allowlist. Root lists use the platform path
separator (`;` on Windows, `:` on POSIX). State defaults to `runs/jobs_state.json`.
Model and data roots default to the working directory; the output root is its
`runs/` subdirectory. Start from the repository root for the quick-start defaults.
Clients use `STUDIO_API_URL`, with `F1_STUDIO_API_URL` as an alias; see
[frontend settings](frontend.md#trusted-local-configuration).

Cross-origin resource sharing (CORS) controls browser access from other origins.
Default origins are localhost/127.0.0.1 on ports 8000, 5173, 3000 and 8080,
plus legacy `null` for the file console. Peers must be loopback and Host must
be literal local. Remote/wildcard/credential-bearing origins and remote bind
addresses are rejected. Disallowed preflights return 400; actual requests return
403. The `null` origin cannot identify its source; CORS is not authentication.
See [security](security.md#trusted-local-deployment).

The source `frontend/` console is mounted at `/` when present. A built `web/dist/`
is mounted at `/console`; API routes take precedence. Downloads stream from an
already-open authorized file using Starlette 0.49.3+ FileResponse range/header
semantics. An installed wheel includes `main_engine`, `api`, `studio` and F1
compatibility packages, but no source UI assets. The API starts without Gradio
or those assets. See [clients and launcher](frontend.md) for UI startup.

## History and lifecycle

Version-1 F1 snapshots retain terminal records, timestamps, errors and manifests.
Safe absolute artifact entries receive contained relative download IDs at read
time. Restored active jobs become `failed / SERVICE_RESTARTED`; startup neither
relaunches them nor signals historical PIDs. Validate and back up real history
before migration, and give each state store a single service owner.

An admission write failure rolls back the job and starts no worker. Later write
failures keep execution state in memory and set `persistence_error` on the owner;
the latest state may then be absent after restart. `/health` reports liveness
(`status=ok`, `service="YOLO-Master F1 Task Engine"`) without that internal warning.
See [Runtime persistence](runtime.md#persistence-and-recovery) for recovery rules.

Shutdown drains HTTP requests (default 5 seconds), then awaits Runtime's checkpoint
cooperation and cleanup. Runtime's caller budget is
`shutdown_grace_seconds + stop_grace_seconds + 16` (default 78 seconds). Active
jobs stopped by service shutdown become `failed / SERVICE_SHUTDOWN`. Cleanup
failure retains ownership and occupied slots and fails lifespan; persistent
cleanup failure can prevent a bounded process exit. The [launcher](frontend.md)
waits for the service's configured budgets.

Checkpoint publication and ordered log-tail persistence are described in
[Runtime](runtime.md#training-shutdown-and-recovery); output-capture coverage is in
[security](security.md#logs-errors-and-persisted-state). For automated and manual
verification, use [testing and reproduction](testing.md).
