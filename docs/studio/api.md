# Studio HTTP API

[Studio documentation](README.md) · [Service configuration](service.md)

Run `python main_engine.py` from the repository root. The default origin is
`http://127.0.0.1:8000`. `/docs` and `/openapi.json` describe the running HTTP
contract. Collection routes accept both `/api/v1/jobs` and `/api/v1/jobs/`.

## Submit a local diagnostic

With the API running in another terminal, this standard-library Python example
submits a diagnostic without model weights. Run it from the source checkout;
the example uses the default server output root `runs`:

```python
import json
from urllib.request import Request, urlopen

payload = {
    "job_id": "docs-diagnose-001",
    "task_type": "diagnose",
    "output": {"output_dir": "runs"},
}
request = Request(
    "http://127.0.0.1:8000/api/v1/jobs",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=10) as response:
    print(response.status, json.load(response)["job_id"])
```

Expect HTTP 201 and the submitted ID. Use a new ID for each run; duplicates
receive 409. Query `GET /api/v1/jobs/docs-diagnose-001` until terminal, then read
its `/logs` and `/artifacts`. Diagnostic reports describe the local environment.
For other tasks, `params` carries handler-specific values such as `model_path`,
`data_source`, `device`, `epochs`
and `imgsz`; paths must satisfy the server's independent trusted-root policy.
The handler validates task parameters during execution; query the job's final
status for the outcome.

## Status, errors and cancellation

Jobs have five states: `pending`, `running`, `completed`, `failed` and
`cancelled`. The last three are final states. Admission and cancellation
acknowledgements precede execution/cleanup; poll the job for its final outcome.

- `POST /api/v1/jobs`: 201 accepted; 409 duplicate ID; 429 `QUEUE_FULL`;
  422 invalid schema or input path; 503 `SERVICE_UNAVAILABLE`,
  `SERVICE_SHUTDOWN` or `PERSISTENCE_FAILED`.
- `GET /api/v1/jobs?limit=10&offset=0`: atomic recent-job page and total.
  `limit` is 1–100, `offset` is nonnegative.
- `GET /api/v1/jobs/{job_id}`: status, error code/message, metadata,
  artifact count and numeric duration. Duration excludes queue time; an absent
  start is null and an actual zero duration is `0.0`.
- `POST /api/v1/jobs/{job_id}/cancel`: 202 `cancel_requested` acknowledges
  cancellation; 200 replays an already terminal job; 409 means not cancellable;
  404 means unknown job. Continue polling after 202 until cleanup is confirmed.
- Unknown jobs and missing artifacts return 404. Unsupported local-boundary
  requests receive 403 `LOCAL_ONLY`; disallowed CORS preflights receive 400.

Structured transport errors use `detail.code` and `detail.message`. Existing
duplicate, path, unknown-job and missing-artifact details remain strings;
schema/query validation returns a sanitized error list. Handle these shapes
separately and use status/code for program decisions.
An admission persistence failure rolls back the unaccepted job and launches no
worker. A later write failure has a different [durability boundary](runtime.md#persistence-and-recovery):
execution truth can survive in memory without reaching disk. `/health` keeps
the compatibility liveness response and exposes no persistence-warning field.

## Log cursors and artifacts

`GET /api/v1/jobs/{job_id}/logs?offset=0&limit=200` pages sanitized flattened
lines. An explicit `limit` is 1–10000. Omitting it returns all currently
remaining lines from `offset`; there is no implicit 10000-line truncation.
Follow `next_offset` until null. Null means the current tail only; continue at
the prior `offset + len(logs)` for later polls, including terminal tail drains.
Changing job/API selection must invalidate older in-flight responses.

`GET /api/v1/jobs/{job_id}/artifacts` returns exact relative `artifact_id`
values, names, image flags and `download_url` values. Download from
`GET /static/artifacts/{job_id}/{artifact_id:path}`, preserving nested IDs and
URL-encoding their components. Downloads recheck manifest membership,
containment and the identity of the opened file. Use the returned download URL.
A confirmed `weights/shutdown.pt` may also be
authorized after cleanup on a `failed / SERVICE_SHUTDOWN` job. Only listed
artifacts are downloadable.

## Regenerate the domain schema

[job_schema.json](job_schema.json) is generated from `core.schema.JobRequest`
in Pydantic validation mode. It describes domain fields; handler-specific
`params` and HTTP response models are documented separately. The service applies
its own lifecycle/security policy to validated inputs. The older
`docs/api/job_schema.json` describes the legacy F1 registry.

From the repository root, run this Python block (or save it as a temporary script):

```python
import json
from pathlib import Path

from core.schema import JobRequest

schema = JobRequest.model_json_schema(mode="validation")
# Drop only the volatile import-time timestamp annotation; validation is unchanged.
schema["properties"]["metadata"]["default"].pop("created_at", None)
Path("docs/studio/job_schema.json").write_text(
    json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
```

Record the source revision, Python and Pydantic versions when regenerating.
Generate again in a fresh process and compare the parsed JSON and bytes; only
the timestamp annotation is intentionally omitted. Use the running OpenAPI for
complete endpoint/response contracts.
