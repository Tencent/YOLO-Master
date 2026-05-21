from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from runtime.cli.async_jobs import AsyncJobManager
from runtime.cli.contract import response


@dataclass(frozen=True)
class JobDeps:
    manager_factory: Callable[[], AsyncJobManager] = AsyncJobManager


def requested_job_id(request: dict[str, Any]) -> str:
    job_id = request["inputs"].get("job_id") or request["params"].get("job_id")
    if not job_id:
        raise ValueError("`inputs.job_id` or `params.job_id` is required.")
    return str(job_id)


def run_job_status(request: dict[str, Any], deps: JobDeps = JobDeps()) -> dict[str, Any]:
    status = deps.manager_factory().status(requested_job_id(request))
    return response(
        request["skill"],
        "ok" if status.get("status") != "missing" else "failed",
        "async job status collected" if status.get("status") != "missing" else "async job not found",
        job=status,
    )


def run_job_cancel(request: dict[str, Any], deps: JobDeps = JobDeps()) -> dict[str, Any]:
    status = deps.manager_factory().cancel(requested_job_id(request))
    return response(
        request["skill"],
        "ok" if status.get("cancelled") else "partial",
        "async job cancelled" if status.get("cancelled") else "async job was not running",
        job=status,
    )
