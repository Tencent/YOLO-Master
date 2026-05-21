#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
import urllib.request  # noqa: F401 - validator probes monkeypatch dispatcher.urllib.request
from pathlib import Path
from typing import Any

SKILL_ROOT = Path(__file__).resolve().parents[2]
for candidate in (SKILL_ROOT,):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from runtime.cli.contract import (
    finalize_payload,
    plan_response,
    response,
    write_manifest,
)
from runtime.cli.async_jobs import async_requested, submit_async_skill
from runtime.cli import executor as cli_executor
from runtime.cli.core_handlers import (
    CoreDeps,
    metrics_payload as metrics_payload_impl,
    run_benchmark as run_benchmark_impl,
    run_export as run_export_impl,
    run_lora_adapters as run_lora_adapters_impl,
    run_predict_like as run_predict_like_impl,
    run_train_like as run_train_like_impl,
    run_tune as run_tune_impl,
    run_val as run_val_impl,
    summarize_results as summarize_results_impl,
)
from runtime.cli.job_handlers import (
    JobDeps,
    run_job_cancel as run_job_cancel_impl,
    run_job_status as run_job_status_impl,
)
from runtime.cli.launcher_handlers import (
    LauncherDeps,
    format_solution_arg as format_solution_arg_impl,
    get_cfg_helpers as get_cfg_helpers_impl,
    run_solutions as run_solutions_impl,
    run_ui_launch as run_ui_launch_impl,
)
from runtime.cli.normalize import (
    is_dry_run,
    normalize_request,
)
from runtime.cli.lora_tools import LoraDiagnoseDeps, run_lora_diagnose as run_lora_diagnose_impl
from runtime.cli.multimodal_handlers import (
    MultimodalDeps,
    run_multimodal_evaluate as run_multimodal_evaluate_impl,
    run_multimodal_infer as run_multimodal_infer_impl,
)
from runtime.cli.model_handlers import ModelDeps, run_model_inspect as run_model_inspect_impl
from runtime.cli.moe_tools import run_moe_diagnose, run_moe_prune
from runtime.cli.peft_compare import PeftCompareDeps, run_peft_compare as run_peft_compare_impl
from runtime.cli.pipeline import PipelineDeps, run_experiment_pipeline
from runtime.cli.sahi_compare import run_sahi_compare
from runtime.cli.system_handlers import (
    SystemDeps,
    get_checks_helpers as get_checks_helpers_impl,
    run_system as run_system_impl,
)
from runtime.multimodal.runtime import (
    call_openai_compatible,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_CACHE: dict[str, Any] = {}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)

_executor_run_cli = cli_executor.run_cli
_executor_run_cli_with_recovery = cli_executor.run_cli_with_recovery


def run_cli(args: list[str], cwd: Path | None = None, force_install: bool = False) -> dict[str, Any]:
    return _executor_run_cli(args, cwd=cwd, force_install=force_install)


def run_cli_with_recovery(
    request: dict[str, Any],
    mode: str,
    values: dict[str, Any],
    *,
    failure_summary: str,
    selected_device: str | None,
    selection_source: str | None,
) -> dict[str, Any]:
    original_run_cli = cli_executor.run_cli
    cli_executor.run_cli = globals()["run_cli"]
    try:
        return _executor_run_cli_with_recovery(
            request,
            mode,
            values,
            failure_summary=failure_summary,
            selected_device=selected_device,
            selection_source=selection_source,
        )
    finally:
        cli_executor.run_cli = original_run_cli


def classify_cli_failure(cli_result: dict[str, Any]) -> dict[str, Any]:
    return cli_executor.classify_cli_failure(cli_result)


def should_retry_with_cpu(
    request: dict[str, Any],
    cli_result: dict[str, Any],
    *,
    selected_device: str | None,
    selection_source: str | None,
) -> bool:
    return cli_executor.should_retry_with_cpu(
        request,
        cli_result,
        selected_device=selected_device,
        selection_source=selection_source,
    )


def cached(name: str, loader):
    if name not in MODULE_CACHE:
        MODULE_CACHE[name] = loader()
    return MODULE_CACHE[name]


def get_ultralytics_core() -> dict[str, Any]:
    def _loader():
        ultralytics = importlib.import_module("ultralytics")
        utils = importlib.import_module("ultralytics.utils")
        return {
            "YOLO": ultralytics.YOLO,
            "version": ultralytics.__version__,
            "SETTINGS": utils.SETTINGS,
            "SETTINGS_FILE": utils.SETTINGS_FILE,
            "YAML": utils.YAML,
        }

    return cached("ultralytics_core", _loader)


def get_cfg_helpers() -> dict[str, Any]:
    return get_cfg_helpers_impl()


def get_checks_helpers() -> dict[str, Any]:
    return get_checks_helpers_impl()


def best_checkpoint(payload: dict[str, Any]) -> str | None:
    for artifact in payload.get("artifacts", []):
        if artifact.get("kind") == "checkpoint" and artifact.get("label") == "best":
            return artifact.get("path")
    for artifact in payload.get("artifacts", []):
        if artifact.get("kind") == "checkpoint":
            return artifact.get("path")
    return None


def metrics_payload(metrics: Any) -> dict[str, Any]:
    return metrics_payload_impl(metrics)


def summarize_results(results: Any, max_items: int = 10) -> list[dict[str, Any]]:
    return summarize_results_impl(results, max_items=max_items)


def build_model(request: dict[str, Any]) -> Any:
    inputs = request["inputs"]
    model_ref = inputs.get("model")
    if not model_ref:
        raise ValueError("`inputs.model` is required.")
    YOLO = get_ultralytics_core()["YOLO"]
    return YOLO(model_ref, task=inputs.get("task"))


def run_system(request: dict[str, Any]) -> dict[str, Any]:
    return run_system_impl(
        request,
        SystemDeps(
            run_cli=run_cli,
            get_ultralytics_core=get_ultralytics_core,
        ),
    )


def run_model_inspect(request: dict[str, Any]) -> dict[str, Any]:
    return run_model_inspect_impl(request, ModelDeps(build_model=build_model))


def core_deps() -> CoreDeps:
    return CoreDeps(
        build_model=build_model,
        run_cli=run_cli,
        run_cli_with_recovery=run_cli_with_recovery,
    )


def run_train_like(request: dict[str, Any], skill_name: str) -> dict[str, Any]:
    return run_train_like_impl(request, skill_name, core_deps())


def run_val(request: dict[str, Any]) -> dict[str, Any]:
    return run_val_impl(request, core_deps())


def run_predict_like(request: dict[str, Any], mode: str) -> dict[str, Any]:
    return run_predict_like_impl(request, mode, core_deps())


def run_multimodal_infer(request: dict[str, Any]) -> dict[str, Any]:
    return run_multimodal_infer_impl(
        request,
        MultimodalDeps(
            build_model=build_model,
            call_openai_compatible=call_openai_compatible,
            run_val=run_val,
        ),
    )


def run_multimodal_evaluate(request: dict[str, Any]) -> dict[str, Any]:
    return run_multimodal_evaluate_impl(
        request,
        MultimodalDeps(
            build_model=build_model,
            call_openai_compatible=call_openai_compatible,
            run_val=run_val,
        ),
    )


def run_export(request: dict[str, Any]) -> dict[str, Any]:
    return run_export_impl(request, core_deps())


def run_benchmark(request: dict[str, Any]) -> dict[str, Any]:
    return run_benchmark_impl(request, core_deps())


def run_tune(request: dict[str, Any]) -> dict[str, Any]:
    return run_tune_impl(request, core_deps())


def run_lora_adapters(request: dict[str, Any]) -> dict[str, Any]:
    return run_lora_adapters_impl(request, core_deps())


def run_lora_diagnose(request: dict[str, Any]) -> dict[str, Any]:
    return run_lora_diagnose_impl(
        request,
        LoraDiagnoseDeps(
            build_model=build_model,
            is_dry_run=is_dry_run,
            response=response,
            plan_response=plan_response,
            write_manifest=write_manifest,
        ),
    )


def run_peft_compare(request: dict[str, Any]) -> dict[str, Any]:
    return run_peft_compare_impl(
        request,
        PeftCompareDeps(
            normalize_request=normalize_request,
            is_dry_run=is_dry_run,
            response=response,
            plan_response=plan_response,
            write_manifest=write_manifest,
            best_checkpoint=best_checkpoint,
            run_train_like=run_train_like,
            run_val=run_val,
        ),
    )


def format_solution_arg(key: str, value: Any) -> str:
    return format_solution_arg_impl(key, value)


def run_solutions(request: dict[str, Any]) -> dict[str, Any]:
    return run_solutions_impl(
        request,
        LauncherDeps(
            run_cli=run_cli,
            get_cfg_helpers=get_cfg_helpers,
        ),
    )


def run_ui_launch(request: dict[str, Any]) -> dict[str, Any]:
    return run_ui_launch_impl(
        request,
        LauncherDeps(
            run_cli=run_cli,
            get_cfg_helpers=get_cfg_helpers,
        ),
    )


def run_pipeline(request: dict[str, Any]) -> dict[str, Any]:
    if async_requested(request) and not is_dry_run(request):
        return submit_async_skill(request)
    return run_experiment_pipeline(
        request,
        PipelineDeps(
            normalize_request=normalize_request,
            is_dry_run=is_dry_run,
            response=response,
            plan_response=plan_response,
            write_manifest=write_manifest,
            best_checkpoint=best_checkpoint,
            run_system=run_system,
            run_model_inspect=run_model_inspect,
            run_train_like=run_train_like,
            run_val=run_val,
            run_export=run_export,
            run_benchmark=run_benchmark,
            run_lora_diagnose=run_lora_diagnose,
            run_moe_diagnose=run_moe_diagnose,
            run_peft_compare=run_peft_compare,
        ),
    )


def run_job_status(request: dict[str, Any]) -> dict[str, Any]:
    return run_job_status_impl(request, JobDeps())


def run_job_cancel(request: dict[str, Any]) -> dict[str, Any]:
    return run_job_cancel_impl(request, JobDeps())


HANDLERS = {
    "yolo.system": run_system,
    "yolo.model.inspect": run_model_inspect,
    "yolo.train": lambda request: run_train_like(request, "yolo.train"),
    "yolo.lora.train": lambda request: run_train_like(request, "yolo.lora.train"),
    "yolo.val": run_val,
    "yolo.predict": lambda request: run_predict_like(request, "predict"),
    "yolo.track": lambda request: run_predict_like(request, "track"),
    "yolo.multimodal.infer": run_multimodal_infer,
    "yolo.multimodal.evaluate": run_multimodal_evaluate,
    "yolo.export": run_export,
    "yolo.benchmark": run_benchmark,
    "yolo.tune": run_tune,
    "yolo.lora.adapters": run_lora_adapters,
    "yolo.lora.diagnose": run_lora_diagnose,
    "yolo.eval.peft_compare": run_peft_compare,
    "yolo.eval.sparse_sahi_compare": run_sahi_compare,
    "yolo.moe.diagnose": run_moe_diagnose,
    "yolo.moe.prune": run_moe_prune,
    "yolo.solutions.run": run_solutions,
    "yolo.ui.launch": run_ui_launch,
    "yolo.pipeline.experiment": run_pipeline,
    "yolo.job.status": run_job_status,
    "yolo.job.cancel": run_job_cancel,
}


def load_request(args: argparse.Namespace) -> dict[str, Any]:
    if args.request:
        return json.loads(Path(args.request).read_text(encoding="utf-8"))
    if args.json:
        return json.loads(args.json)
    stdin = sys.stdin.read().strip()
    if stdin:
        return json.loads(stdin)
    raise ValueError("Provide --request, --json, or JSON on stdin.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured dispatcher for the YOLO-Master agent skill.")
    parser.add_argument("--request", help="Path to a JSON request file.")
    parser.add_argument("--json", help="Inline JSON request.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    request: dict[str, Any] | None = None
    try:
        request = normalize_request(load_request(args))
        skill = request.get("skill")
        if skill not in HANDLERS:
            raise ValueError(f"Unsupported skill: {skill}")
        payload = HANDLERS[skill](request)
        payload = finalize_payload(request, payload)
    except Exception as exc:
        payload = response(
            request.get("skill", "unknown") if request else "unknown",
            "failed",
            str(exc),
            error={"type": type(exc).__name__, "traceback": traceback.format_exc()},
        )
        if request:
            payload = finalize_payload(request, payload)

    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("status") in {"ok", "running", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
