#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DISPATCHER = SKILL_ROOT / "scripts" / "run_yolo_master_skill.py"
DEFAULT_CASES = SKILL_ROOT / "assets" / "autotrain_cases.json"
REPORT_DIR = SKILL_ROOT / "logs"
SUITE_ALIASES = {
    "quick": {"fast-smoke", "dry-run", "contract"},
    "smoke": {"fast-smoke", "cli-smoke", "deep-smoke"},
    "extended": {"extended-cli"},
}


def dotted_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, list):
            if not part.isdigit():
                return None
            idx = int(part)
            if idx >= len(current):
                return None
            current = current[idx]
            continue
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def load_dispatcher_module() -> Any:
    spec = importlib.util.spec_from_file_location("yolo_master_skill_dispatcher", DISPATCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load dispatcher module from {DISPATCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_result(
    case: dict[str, Any],
    request: dict[str, Any],
    payload: dict[str, Any],
    *,
    elapsed: float,
    returncode: int,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    result = {
        "name": case["name"],
        "suite": case.get("suite", "all"),
        "elapsed_sec": round(elapsed, 3),
        "returncode": returncode,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "payload": payload,
        "passed": True,
        "checks": [],
    }

    expect = case.get("expect", {})
    if "status" in expect:
        ok = payload.get("status") == expect["status"]
        result["checks"].append({"kind": "status", "ok": ok, "expected": expect["status"], "actual": payload.get("status")})
        result["passed"] &= ok
    for path in expect.get("paths", []):
        ok = dotted_get(payload, path) is not None
        result["checks"].append({"kind": "path", "ok": ok, "path": path})
        result["passed"] &= ok
    for path in expect.get("nonempty", []):
        value = dotted_get(payload, path)
        ok = bool(value)
        result["checks"].append({"kind": "nonempty", "ok": ok, "path": path})
        result["passed"] &= ok
    for path, expected in expect.get("equals", {}).items():
        actual = dotted_get(payload, path)
        ok = actual == expected
        result["checks"].append({"kind": "equals", "ok": ok, "path": path, "expected": expected, "actual": actual})
        result["passed"] &= ok
    for path, expected in expect.get("contains", {}).items():
        actual = dotted_get(payload, path)
        if isinstance(actual, str):
            ok = str(expected) in actual
        elif isinstance(actual, (list, tuple, set)):
            ok = any(str(expected) in str(item) for item in actual)
        else:
            ok = False
        result["checks"].append({"kind": "contains", "ok": ok, "path": path, "expected": expected, "actual": actual})
        result["passed"] &= ok
    for path in expect.get("path_exists", []):
        value = dotted_get(payload, path)
        ok = bool(value) and Path(str(value)).exists()
        result["checks"].append({"kind": "path_exists", "ok": ok, "path": path, "actual": value})
        result["passed"] &= ok
    if "max_elapsed_sec" in expect:
        ok = elapsed <= float(expect["max_elapsed_sec"])
        result["checks"].append(
            {
                "kind": "max_elapsed_sec",
                "ok": ok,
                "expected": expect["max_elapsed_sec"],
                "actual": round(elapsed, 3),
            }
        )
        result["passed"] &= ok
    expected_returncode = expect.get("returncode")
    if expected_returncode is not None:
        ok = returncode == int(expected_returncode)
        result["checks"].append(
            {"kind": "returncode", "ok": ok, "expected": int(expected_returncode), "actual": returncode}
        )
        result["passed"] &= ok
    elif returncode != 0:
        result["passed"] = False
    return result


def run_dispatcher_case(case: dict[str, Any]) -> dict[str, Any]:
    request = dict(case["request"])
    cmd = [sys.executable, str(DISPATCHER), "--json", json.dumps(request, ensure_ascii=False)]
    env = os.environ.copy()
    env["YOLO_MASTER_AGENT_RUNTIME_CACHE"] = "1"
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, env=env)
    elapsed = time.perf_counter() - start
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    payload: dict[str, Any]
    try:
        payload = json.loads(stdout.splitlines()[-1] if stdout else "{}")
    except Exception:
        payload = {
            "skill": request.get("skill"),
            "status": "failed",
            "summary": "failed to parse dispatcher output",
            "raw_stdout": stdout,
            "raw_stderr": stderr,
        }
    return build_result(case, request, payload, elapsed=elapsed, returncode=proc.returncode, stdout=stdout, stderr=stderr)


def run_probe_case(case: dict[str, Any]) -> dict[str, Any]:
    probe = case.get("probe", {})
    kind = probe.get("kind")
    request = dict(case.get("request", {}))
    start = time.perf_counter()
    stdout = ""
    stderr = ""
    if kind == "recovery_auto_retry":
        module = load_dispatcher_module()
        calls: list[list[str]] = []
        original_run_cli = module.run_cli

        def fake_run_cli(args, cwd=None, force_install=False):
            calls.append(list(args))
            if len(calls) == 1:
                return {
                    "cmd": ["yolo", *args],
                    "cwd": ".",
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "RuntimeError: not implemented for 'MPS'",
                    "install": {"status": "available", "path": "/tmp/yolo"},
                }
            return {
                "cmd": ["yolo", *args],
                "cwd": ".",
                "returncode": 0,
                "stdout": "done",
                "stderr": "",
                "install": {"status": "available", "path": "/tmp/yolo"},
            }

        module.run_cli = fake_run_cli
        try:
            outcome = module.run_cli_with_recovery(
                {
                    "skill": probe.get("skill", "yolo.train"),
                    "runtime": {"allow_device_fallback": True},
                    "params": {"device": probe.get("device", "mps")},
                },
                probe.get("mode", "train"),
                {"model": probe.get("model", "m.pt"), "device": probe.get("device", "mps")},
                failure_summary="training failed",
                selected_device=probe.get("device", "mps"),
                selection_source=probe.get("selection_source", "auto"),
            )
        finally:
            module.run_cli = original_run_cli
        payload = {
            "skill": probe.get("skill", "yolo.train"),
            "status": "ok" if outcome["failed"] is None else "failed",
            "summary": "recovery probe finished",
            "data": {
                "attempt_count": len(outcome["attempts"]),
                "final_device": outcome["device"],
                "recovery": outcome["recovery"] or {},
                "calls": calls,
            },
        }
        return build_result(case, request, payload, elapsed=time.perf_counter() - start, returncode=0 if outcome["failed"] is None else 1, stdout=stdout, stderr=stderr)
    if kind == "recovery_no_retry":
        module = load_dispatcher_module()
        classification = module.classify_cli_failure(
            {
                "cmd": ["yolo", probe.get("mode", "train"), f"device={probe.get('device', 'mps')}"],
                "stdout": "",
                "stderr": "RuntimeError: not implemented for 'MPS'",
                "returncode": 1,
            }
        )
        payload = {
            "skill": probe.get("skill", "yolo.train"),
            "status": "ok",
            "summary": "recovery guard probe finished",
            "data": {
                "should_retry": module.should_retry_with_cpu(
                    {
                        "runtime": {"allow_device_fallback": True},
                        "params": {"device": probe.get("device", "mps")},
                    },
                    {
                        "cmd": ["yolo", probe.get("mode", "train"), f"device={probe.get('device', 'mps')}"],
                        "stdout": "",
                        "stderr": "RuntimeError: not implemented for 'MPS'",
                        "returncode": 1,
                    },
                    selected_device=probe.get("device", "mps"),
                    selection_source=probe.get("selection_source", "runtime"),
                ),
                "classification": classification,
            },
        }
        return build_result(case, request, payload, elapsed=time.perf_counter() - start, returncode=0, stdout=stdout, stderr=stderr)
    payload = {
        "skill": request.get("skill", "unknown"),
        "status": "failed",
        "summary": f"unsupported probe kind: {kind}",
    }
    return build_result(case, request, payload, elapsed=time.perf_counter() - start, returncode=1, stdout=stdout, stderr=stderr)


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("executor") == "probe":
        return run_probe_case(case)
    return run_dispatcher_case(case)


def default_enabled(case: dict[str, Any]) -> bool:
    return not bool(case.get("manual_only", False))


def select_cases(cases: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    if suite == "all":
        return [case for case in cases if default_enabled(case)]
    allowed = SUITE_ALIASES.get(suite, {suite})
    return [case for case in cases if case.get("suite") in allowed]


def recommend(results: list[dict[str, Any]]) -> list[str]:
    failures = [r for r in results if not r["passed"]]
    slow_cases = [r for r in results if r["elapsed_sec"] >= 5]
    if not failures:
        recs = ["All cases passed. Keep the current skill shape and expand the case set before adding new handlers."]
        if slow_cases:
            recs.append("Keep heavyweight cases in deep-smoke and use fast-smoke for tight iteration loops.")
        return recs
    failed_names = {r["name"] for r in failures}
    recs = []
    if any(name.startswith("system_") for name in failed_names):
        recs.append("Tighten the system-action router and keep lazy imports for CLI-less actions.")
    if any(name.endswith("_dry_run") for name in failed_names):
        recs.append("Patch the dry-run plan output before touching real execution paths.")
    if any(name.startswith("inspect_") for name in failed_names):
        recs.append("Simplify inspect-time model construction and keep path normalization strict.")
    if any(name.startswith("pipeline_") for name in failed_names):
        recs.append("Refactor pipeline orchestration to surface stage-level errors and stage manifests.")
    if any(name.startswith("recovery_") for name in failed_names):
        recs.append("Keep the recovery probes green when adjusting auto device selection or CLI failure handling.")
    if any(check["kind"] == "max_elapsed_sec" and not check["ok"] for r in failures for check in r["checks"]):
        recs.append("Preserve fast-smoke latency budgets with lazy imports or by moving expensive checks into deep-smoke.")
    if not recs:
        recs.append("Review the failing cases and add more narrow assertions around returned artifacts and paths.")
    return recs


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite": report["suite"],
        "total": report["total"],
        "passed": report["passed"],
        "failed": report["failed"],
        "score": report["score"],
        "slowest": report["slowest"],
        "recommendations": report["recommendations"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoTrain-style validator for the YOLO-Master skill.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to autotrain case JSON.")
    parser.add_argument(
        "--suite",
        default="quick",
        help=(
            "Case suite to run: quick, all, smoke, extended, fast-smoke, cli-smoke, deep-smoke, dry-run, "
            "contract, or any suite present in the case file. `quick` is the default agent loop; "
            "`all` skips cases marked manual_only but can take minutes because it includes deep-smoke."
        ),
    )
    parser.add_argument("--out", default=str(REPORT_DIR / "autotrain-report.json"), help="Output report path.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the summary report.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the top-level summary to stdout while still writing the full report to --out.",
    )
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    selected = select_cases(cases, args.suite)
    results = [run_case(case) for case in selected]
    passed = sum(1 for r in results if r["passed"])
    report = {
        "suite": args.suite,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "score": round(passed / len(results), 3) if results else 0.0,
        "slowest": [
            {"name": item["name"], "elapsed_sec": item["elapsed_sec"]}
            for item in sorted(results, key=lambda item: item["elapsed_sec"], reverse=True)[:5]
        ],
        "results": results,
        "recommendations": recommend(results),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output = console_summary(report) if args.summary_only else report
    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
