"""Exercise the real CI aggregate gate across event and job-result combinations."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
REQUIRED_PR_JOBS = {"Tests", "MixtureDDP", "MixtureP0Regression"}


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_summary_waits_for_every_ci_job():
    jobs = _workflow()["jobs"]
    assert set(jobs) - {"Summary"} == set(jobs["Summary"]["needs"])


@pytest.mark.parametrize(
    "event,override,expected_success",
    [
        ("pull_request", {}, True),
        ("push", {"MixtureP0Regression": "failure"}, False),
        ("pull_request", {"MixtureP0Regression": "skipped"}, False),
        ("pull_request", {"Tests": "cancelled"}, False),
        ("push", {"MixtureDDP": "skipped"}, False),
        ("workflow_dispatch", {name: "skipped" for name in REQUIRED_PR_JOBS}, True),
        ("schedule", {name: "skipped" for name in REQUIRED_PR_JOBS}, True),
        ("schedule", {"IsolatedExports": "failure"}, False),
        ("workflow_dispatch", {"GPU": "cancelled"}, False),
        ("pull_request", {"FutureRegression": "failure"}, False),
    ],
)
def test_summary_propagates_job_failures(event, override, expected_success, tmp_path):
    summary = _workflow()["jobs"]["Summary"]
    step = next(item for item in summary["steps"] if item["name"] == "Validate job results")
    # Execute the workflow's embedded Python with this interpreter on every platform.
    command = step["run"].strip().splitlines()
    assert command[0] == "python - <<'PY'" and command[-1] == "PY"
    script = "\n".join(command[1:-1])
    statuses = {name: "success" if name in REQUIRED_PR_JOBS else "skipped" for name in summary["needs"]}
    statuses.update(override)
    env = {
        **os.environ,
        "CI_EVENT": event,
        "CI_NEEDS": json.dumps({name: {"result": result} for name, result in statuses.items()}),
    }

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=tmp_path, capture_output=True, text=True, check=False
    )

    assert (result.returncode == 0) == expected_success, result.stdout + result.stderr
