"""Run one source-locked P2 bridge cell with frozen-state and resume checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


from scripts.a1.diagnostic_hooks import install_first_forward_check


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(request):
    for relative, expected in request["p2_bridge"]["source_hashes"].items():
        if sha256(REPO / relative) != expected:
            raise RuntimeError(f"source drift: {relative}")
    for path, expected in request["p2_bridge"]["input_hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"input drift: {path}")


def frozen_digest(model, keys=None):
    import torch

    if keys is None:
        keys = {n for n, p in model.named_parameters() if not p.requires_grad}
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                keys.update(f"{n}.{k}" for k, _ in m.named_buffers(recurse=False))
    state = model.state_dict()
    digest = hashlib.sha256()
    for name in sorted(keys):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest(), keys


def gain_summary(model):
    """Summarize scalar or per-channel gates without assuming their shape."""
    return [
        {
            "name": n,
            "shape": list(m.gain.shape),
            "abs_max": float(m.gain.detach().abs().max().cpu()),
            "l2": float(m.gain.detach().norm().cpu()),
        }
        for n, m in model.named_modules()
        if hasattr(m, "gain") and hasattr(m, "base")
    ]


def validate_batch_budget(trainer, request):
    """Reject trainer fallback that silently changes the paired microbatch budget."""
    expected = request["params"]["batch"]
    actual = (trainer.args.batch, trainer.batch_size, trainer.train_loader.batch_size)
    if actual != (expected, expected, expected):
        raise RuntimeError(f"batch budget drift: requested={expected}, actual={actual}")
    if len(trainer.train_loader.dataset) != request["p2_bridge"]["train_images"]:
        raise RuntimeError("training dataset size drift")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-checkpoint", action="store_true")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    verify_sources(request)
    bridge = request["p2_bridge"]
    alpha = bridge["alpha"]
    if alpha not in (0.0, 0.1):
        raise ValueError("pilot only permits predeclared alpha=0 or 0.1")
    os.environ.update(A1_E2E_O2O_TAL_TOPK="7", A1_E2E_O2O_TAL_TOPK2="1", A1_E2E_O2O_CLS_GAIN="1.0")
    import torch
    from run_p1_bn_frozen import (
        configure_r19_exploration,
        enforce_and_schedule_p1_policy,
        enforce_p1_freeze_policy,
        read_request,
        runtime_policy_payload,
        validate_candidate_runtime,
        validate_runtime_p1_policy,
        write_failure_report,
    )

    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect

    torch.set_num_threads(2)
    read_request(args.request)
    params = dict(request["params"])
    if params["device"] != "cpu" and not args.dry_run:
        device = int(params["device"])
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(bridge["cuda_allocator_limit_gib"] * 1024**3 / total, device)
    run_dir = Path(params["project"]) / params["name"]
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "alpha": alpha, "save_dir": str(run_dir)}))
        return
    if (run_dir / "completed.json").exists():
        raise FileExistsError("completed cell must not be resumed")
    if args.resume:
        checkpoint = run_dir / "weights/last.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved_model = saved.get("ema") or saved.get("model")
        if saved_model.model[-1].p2_o2o_gradient_alpha != alpha:
            raise RuntimeError("resume coefficient mismatch")
        if saved.get("optimizer") is None or saved.get("epoch", -1) < 0:
            raise RuntimeError("checkpoint lacks optimizer/epoch resume state")
        receipt = json.loads((run_dir / "p2_bridge_runtime.json").read_text())
        if receipt["source_hashes"] != bridge["source_hashes"] or receipt["request_sha256"] != sha256(args.request):
            raise RuntimeError("resume request/source drift")
        del saved, saved_model
        model = YOLO(str(checkpoint), task="detect")
        params["resume"] = str(checkpoint)
    else:
        if run_dir.exists():
            raise FileExistsError(f"refusing to reuse existing cell: {run_dir}")
        model = YOLO(request["inputs"]["model"], task="detect")

    def prepare(trainer):
        configure_r19_exploration(trainer, request)
        enforce_p1_freeze_policy(trainer)
        validate_runtime_p1_policy(trainer)
        validate_batch_budget(trainer, request)
        if trainer.p1_frozen_factor_base_parameters != 459232:
            raise RuntimeError("frozen base parameter count mismatch")
        head = trainer.model.model[-1]
        if type(head) is not Detect or not head.end2end:
            raise RuntimeError("pilot requires native end-to-end Detect")
        head.p2_o2o_gradient_alpha = alpha
        if trainer.ema:
            trainer.ema.ema.model[-1].p2_o2o_gradient_alpha = alpha
        trainer.p2_frozen_before, trainer.p2_frozen_keys = frozen_digest(trainer.model)
        payload = {
            **validate_candidate_runtime(trainer.model, request),
            "alpha": alpha,
            "source_hashes": bridge["source_hashes"],
            "request_sha256": sha256(args.request),
            "resume": args.resume,
            "start_epoch": trainer.start_epoch,
            "frozen_factor_base_parameters": trainer.p1_frozen_factor_base_parameters,
            "frozen_digest_start": trainer.p2_frozen_before,
            "batch": trainer.batch_size,
            "train_images": len(trainer.train_loader.dataset),
            "batches_per_epoch": len(trainer.train_loader),
            "automatic_batch_reduction": "disabled by per-batch OOM retry guard",
        }
        (Path(trainer.save_dir) / "p2_bridge_runtime.json").write_text(json.dumps(payload, indent=2) + "\n")
        (Path(trainer.save_dir) / "p1_runtime_policy_pretrain.json").write_text(
            json.dumps(runtime_policy_payload(trainer, request["request_id"]), indent=2) + "\n"
        )
        print("P2_BRIDGE_RUNTIME " + json.dumps(payload), flush=True)

        install_first_forward_check(head, alpha, Path(trainer.save_dir) / "p2_first_forward.json", hook_cleanup)

    def check_batch(trainer):
        validate_batch_budget(trainer, request)
        # The engine retries by halving batch only while this counter is below 3.
        # Normal epoch completion resets it to zero; an OOM must propagate instead.
        trainer._oom_retries = 3
        enforce_and_schedule_p1_policy(trainer)
        if trainer.model.model[-1].p2_o2o_gradient_alpha != alpha:
            raise RuntimeError("live bridge coefficient drift")

    def check_epoch(trainer):
        digest, _ = frozen_digest(trainer.model, trainer.p2_frozen_keys)
        payload = {
            "epoch": trainer.epoch + 1,
            "alpha": alpha,
            "frozen_unchanged": digest == trainer.p2_frozen_before,
            "frozen_digest": digest,
            "gains": gain_summary(trainer.model),
        }
        with (Path(trainer.save_dir) / "p2_epoch_audit.jsonl").open("a") as f:
            f.write(json.dumps(payload) + "\n")
        if not payload["frozen_unchanged"]:
            raise RuntimeError("frozen parameters or BatchNorm state changed")
        verify_sources(request)

    hook_cleanup = ExitStack()
    model.add_callback("on_pretrain_routine_end", prepare)
    model.add_callback("on_train_batch_start", check_batch)
    model.add_callback("on_train_epoch_end", check_epoch)
    if args.stop_after_checkpoint:

        def stop_after_checkpoint(trainer):
            if trainer.epoch == 0 and trainer.last.is_file():
                print("P2_PLANNED_RESUME_PROBE checkpoint saved", flush=True)
                raise SystemExit(75)

        model.add_callback("on_model_save", stop_after_checkpoint)
    try:
        result = model.train(data=request["inputs"]["data"], **params)
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code == 75:
            raise
        write_failure_report(model, request, error)
        raise
    payload = {
        "status": "completed",
        "alpha": alpha,
        "metrics": dict(result.results_dict),
        "last": str(model.trainer.last),
    }
    (run_dir / "completed.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
