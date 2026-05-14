#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

DISPATCHER = SKILL_ROOT / "scripts" / "run_yolo_master_skill.py"
DEFAULT_CASES = SKILL_ROOT / "assets" / "autotrain_cases.json"
REPORT_DIR = SKILL_ROOT / "logs"
SUITE_ALIASES = {
    "quick": {"fast-smoke", "dry-run", "contract"},
    "smoke": {"fast-smoke", "cli-smoke", "deep-smoke"},
    "extended": {"extended-cli"},
}
ENV_UNSET = object()


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
    return importlib.import_module("runtime.cli.dispatcher")


def apply_env_overrides(env: dict[str, str], overrides: dict[str, Any] | None) -> dict[str, str]:
    if not overrides:
        return env
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


@contextlib.contextmanager
def temporary_env(overrides: dict[str, Any] | None):
    if not overrides:
        yield
        return
    previous: dict[str, Any] = {}
    for key, value in overrides.items():
        previous[key] = os.environ[key] if key in os.environ else ENV_UNSET
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is ENV_UNSET:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    env = apply_env_overrides(os.environ.copy(), case.get("env"))
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
    if kind == "multimodal_stub":
        module = load_dispatcher_module()
        calls: list[dict[str, Any]] = []
        open_world_mode = str(request.get("params", {}).get("prompt_template") or "") == "vlm_open_world_detection"
        generic_taxonomy_probe = bool(request.get("params", {}).get("open_world_taxonomy_require_exact_for_generic"))
        if open_world_mode:
            if generic_taxonomy_probe:
                fake_outputs = iter(
                    [
                        json.dumps(
                            {
                                "answer": "open-world vlm answer",
                                "visual_evidence": ["a bus and visible grass are both present"],
                                "caption": {"short": "A bus near grass", "dense": "A bus occupies the main view while a grassy area appears nearby.", "tags": ["bus", "grass"]},
                                "global_classification": [{"open_label": "bus", "class_id": 5, "coco_label": "bus", "confidence": 0.96}],
                                "vlm_detections": [
                                    {
                                        "proposal_id": "ow1",
                                        "open_label": "grass",
                                        "confidence": 0.88,
                                        "bbox_xyxy": [702, 742, 760, 875],
                                        "bbox_quality": "estimated",
                                        "linked_yolo_indices": [],
                                        "open_world_action": "open_world_add",
                                        "visual_evidence": "green grassy patch near the scene edge",
                                        "rationale": "generic vegetation region visible in image",
                                    }
                                ],
                                "yolo_cross_check": {
                                    "confirmed": ["bus"],
                                    "false_positives": [],
                                    "possible_misses": ["grass"],
                                    "duplicate_or_fragmented": [],
                                    "notes": [],
                                },
                                "fusion_hints": {
                                    "add_open_world_detections": [{"proposal_id": "ow1", "confidence": 0.88, "evidence": "visible grassy patch"}],
                                    "add_vlm_detections": [],
                                    "suppress_yolo_indices": [],
                                    "relabel_yolo": [],
                                    "adjust_boxes": [],
                                },
                                "uncertainty": "low",
                                "recommended_next_actions": ["preserve the open-world region but keep taxonomy matching conservative"],
                            }
                        ),
                        json.dumps(
                            {
                                "answer": "open-world refined answer",
                                "visual_evidence": ["bus remains correct and the grass region is still visible"],
                                "caption": {"short": "Bus with grass", "dense": "The verification pass agrees that the bus is correct and a grassy area is visible.", "tags": ["bus", "grass"]},
                                "global_classification": [{"open_label": "bus", "class_id": 5, "coco_label": "bus", "confidence": 0.95}],
                                "vlm_detections": [
                                    {
                                        "proposal_id": "ow1",
                                        "open_label": "grass",
                                        "confidence": 0.9,
                                        "bbox_xyxy": [702, 742, 760, 875],
                                        "bbox_quality": "estimated",
                                        "linked_yolo_indices": [],
                                        "open_world_action": "open_world_add",
                                        "visual_evidence": "green patch near the lower side",
                                        "rationale": "still visible after verification",
                                    }
                                ],
                                "yolo_cross_check": {
                                    "confirmed": ["bus"],
                                    "false_positives": [],
                                    "possible_misses": ["grass"],
                                    "duplicate_or_fragmented": [],
                                    "notes": [],
                                },
                                "fusion_hints": {
                                    "add_open_world_detections": [{"proposal_id": "ow1", "confidence": 0.9, "evidence": "verified grassy patch"}],
                                    "add_vlm_detections": [],
                                    "suppress_yolo_indices": [],
                                    "relabel_yolo": [],
                                    "adjust_boxes": [],
                                },
                                "uncertainty": "low",
                                "recommended_next_actions": ["keep the generic label, but do not force a taxonomy anchor"],
                            }
                        ),
                    ]
                )
            else:
                fake_outputs = iter(
                    [
                        json.dumps(
                            {
                                "answer": "open-world vlm answer",
                                "visual_evidence": ["a bus and a traffic cone are both visible"],
                                "caption": {"short": "A bus on the road near a traffic cone", "dense": "A parked bus occupies most of the frame and a small traffic cone sits nearby.", "tags": ["bus", "traffic cone", "road"]},
                                "global_classification": [{"open_label": "bus", "class_id": 5, "coco_label": "bus", "confidence": 0.96}],
                                "vlm_detections": [
                                    {
                                        "proposal_id": "ow1",
                                        "open_label": "traffic cone",
                                        "confidence": 0.88,
                                        "bbox_xyxy": [702, 742, 760, 875],
                                        "bbox_quality": "estimated",
                                        "linked_yolo_indices": [],
                                        "open_world_action": "open_world_add",
                                        "visual_evidence": "orange cone on the right side of the bus",
                                        "rationale": "clearly visible novel roadside object",
                                    }
                                ],
                                "yolo_cross_check": {
                                    "confirmed": ["bus"],
                                    "false_positives": [],
                                    "possible_misses": ["traffic cone"],
                                    "duplicate_or_fragmented": [],
                                    "notes": [],
                                },
                                "fusion_hints": {
                                    "add_open_world_detections": [{"proposal_id": "ow1", "confidence": 0.88, "evidence": "visible roadside cone"}],
                                    "add_vlm_detections": [],
                                    "suppress_yolo_indices": [],
                                    "relabel_yolo": [],
                                    "adjust_boxes": [],
                                },
                                "uncertainty": "low",
                                "recommended_next_actions": ["preserve open-world object for downstream reasoning"],
                            }
                        ),
                        json.dumps(
                            {
                                "answer": "open-world refined answer",
                                "visual_evidence": ["bus remains correct and the cone is still visible"],
                                "caption": {"short": "Bus with roadside cone", "dense": "The verification pass agrees that the bus is correct and a cone-like object is visible.", "tags": ["bus", "cone"]},
                                "global_classification": [{"open_label": "bus", "class_id": 5, "coco_label": "bus", "confidence": 0.95}],
                                "vlm_detections": [
                                    {
                                        "proposal_id": "ow1",
                                        "open_label": "traffic cone",
                                        "confidence": 0.9,
                                        "bbox_xyxy": [702, 742, 760, 875],
                                        "bbox_quality": "estimated",
                                        "linked_yolo_indices": [],
                                        "open_world_action": "open_world_add",
                                        "visual_evidence": "small cone near the curb",
                                        "rationale": "still visible after verification",
                                    }
                                ],
                                "yolo_cross_check": {
                                    "confirmed": ["bus"],
                                    "false_positives": [],
                                    "possible_misses": ["traffic cone"],
                                    "duplicate_or_fragmented": [],
                                    "notes": [],
                                },
                                "fusion_hints": {
                                    "add_open_world_detections": [{"proposal_id": "ow1", "confidence": 0.9, "evidence": "verified roadside cone"}],
                                    "add_vlm_detections": [],
                                    "suppress_yolo_indices": [],
                                    "relabel_yolo": [],
                                    "adjust_boxes": [],
                                },
                                "uncertainty": "low",
                                "recommended_next_actions": ["keep the novel object in open-world preview"],
                            }
                        ),
                    ]
                )
        else:
            fake_outputs = iter(
                [
                    json.dumps(
                        {
                            "answer": "vlm answer",
                            "visual_evidence": ["image supports the bus detection"],
                            "yolo_cross_check": {
                                "confirmed": ["bus"],
                                "false_positives": [],
                                "possible_misses": [],
                                "duplicate_or_fragmented": [],
                                "notes": [],
                            },
                            "uncertainty": "low",
                            "recommended_next_actions": ["continue validation"],
                        }
                    ),
                    json.dumps(
                        {
                            "answer": "refined answer",
                            "visual_evidence": ["detector and VLM agree"],
                            "yolo_cross_check": {
                                "confirmed": ["bus"],
                                "false_positives": [],
                                "possible_misses": [],
                                "duplicate_or_fragmented": [],
                                "notes": [],
                            },
                            "uncertainty": "low",
                            "recommended_next_actions": ["continue validation"],
                        }
                    ),
                ]
            )
        original_urlopen = module.urllib.request.urlopen

        class FakeResponse:
            def __init__(self, payload: dict[str, Any]):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request_obj, timeout=120):
            body = json.loads(request_obj.data.decode("utf-8"))
            calls.append({"url": request_obj.full_url, "body": body, "timeout": timeout})
            text = next(fake_outputs, "fallback answer")
            if request_obj.full_url.endswith("/chat/completions"):
                payload = {
                    "id": f"chat-{len(calls)}",
                    "choices": [{"message": {"content": text}}],
                    "usage": {"input_tokens": 8, "output_tokens": 4},
                }
            else:
                payload = {
                    "id": f"resp-{len(calls)}",
                    "output_text": text,
                    "usage": {"input_tokens": 8, "output_tokens": 4},
                }
            return FakeResponse(payload)

        probe_env = dict(case.get("env") or {})
        probe_env.setdefault("OPENAI_API_KEY", "stub-key")
        with temporary_env(probe_env):
            module.urllib.request.urlopen = fake_urlopen
            try:
                payload = module.run_multimodal_infer(module.normalize_request(request))
            finally:
                module.urllib.request.urlopen = original_urlopen
        result_payload = {
            "skill": probe.get("skill", "yolo.multimodal.infer"),
            "status": payload.get("status", "failed"),
            "summary": "multimodal stub probe finished",
            "data": {
                "call_count": len(calls),
                "calls": calls,
                "vlm": payload.get("multimodal", {}).get("vlm", {}),
                "llm_refine": payload.get("multimodal", {}).get("llm_refine", {}),
                "image": payload.get("multimodal", {}).get("image", {}),
            },
            "payload": payload,
        }
        return build_result(
            case,
            request,
            result_payload,
            elapsed=time.perf_counter() - start,
            returncode=0 if payload.get("status") == "ok" else 1,
            stdout=stdout,
            stderr=stderr,
        )
    if kind == "multimodal_batch_stub":
        module = load_dispatcher_module()
        calls: list[dict[str, Any]] = []

        class FakeArray(list):
            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self)

        class FakeBoxes:
            def __init__(self):
                self.xyxy = FakeArray([[121.5, 459.0, 688.5, 945.0]])
                self.cls = FakeArray([0])
                self.conf = FakeArray([0.91])

            def __len__(self):
                return len(self.cls)

        class FakeResult:
            def __init__(self, path: str):
                self.path = path
                self.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}
                self.boxes = FakeBoxes()
                self.names = {0: "bus"}

        class FakeModel:
            def __init__(self):
                self.predictor = type("Predictor", (), {"save_dir": Path("/tmp")})()

            def predict(self, source=None, **kwargs):
                calls.append({"source": str(source), "kwargs": kwargs})
                return [FakeResult(str(source))]

        original_build_model = module.build_model
        original_call_openai = module.call_openai_compatible

        def fake_build_model(request):
            return FakeModel()

        def fake_call_openai_compatible(*, model, user_text, developer_text=None, image_url=None, image_detail="auto", base_url=None, api_mode="auto", max_output_tokens=800, temperature=None):
            calls.append(
                {
                    "model": model,
                    "api_mode": api_mode,
                    "image_url": image_url,
                    "user_text": user_text,
                }
            )
            payload = {
                "answer": "batch answer",
                "visual_evidence": ["stub evidence"],
                "yolo_cross_check": {
                    "confirmed": ["bus"],
                    "false_positives": [],
                    "possible_misses": [],
                    "duplicate_or_fragmented": [],
                    "notes": [],
                },
                "uncertainty": "low",
                "recommended_next_actions": ["continue validation"],
            }
            return {
                "status": "ok",
                "provider": "openai",
                "api_mode": api_mode,
                "model": model,
                "text": json.dumps(payload),
                "response_id": f"resp-{len(calls)}",
                "usage": {"input_tokens": 8, "output_tokens": 4},
            }

        probe_env = dict(case.get("env") or {})
        probe_env.setdefault("OPENAI_API_KEY", "stub-key")
        with temporary_env(probe_env):
            module.build_model = fake_build_model
            module.call_openai_compatible = fake_call_openai_compatible
            try:
                payload = module.run_multimodal_evaluate(module.normalize_request(request))
            finally:
                module.build_model = original_build_model
                module.call_openai_compatible = original_call_openai

        result_payload = {
            "skill": probe.get("skill", "yolo.multimodal.evaluate"),
            "status": payload.get("status", "failed"),
            "summary": "multimodal batch stub probe finished",
            "data": {
                "call_count": len(calls),
                "aggregate": payload.get("evaluation", {}),
                "baseline_status": payload.get("baseline", {}).get("status"),
                "first_item": payload.get("results", [{}])[0] if payload.get("results") else {},
            },
            "payload": payload,
        }
        return build_result(
            case,
            request,
            result_payload,
            elapsed=time.perf_counter() - start,
            returncode=0 if payload.get("status") in {"ok", "partial"} else 1,
            stdout=stdout,
            stderr=stderr,
        )
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
    if any(name.startswith("multimodal_") for name in failed_names):
        recs.append("Keep the multimodal blocked-path, fake OpenAI probe, and thinking-with-image contract aligned.")
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
