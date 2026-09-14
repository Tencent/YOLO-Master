"""Explicit, scoped diagnostic hooks; importing this module installs no hooks."""

import json
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def capture_head_inputs(head):
    """Capture original input references and release them even when forward raises."""
    captured = []

    def capture(_module, inputs):
        captured[:] = inputs[0]

    handle = head.register_forward_pre_hook(capture)
    try:
        yield captured
    finally:
        handle.remove()
        captured.clear()


def install_first_forward_check(head, alpha, output, cleanup):
    """Install a one-shot check owned by the caller's ExitStack for failure cleanup."""

    def first_forward(module, inputs, outputs):
        try:
            flags = [t.requires_grad for t in outputs["one2one"]["feats"]]
            if any(flags) != bool(alpha):
                raise RuntimeError(f"first-batch intervention inactive: alpha={alpha}, flags={flags}")
            Path(output).write_text(
                json.dumps({"alpha": alpha, "one2one_feature_requires_grad": flags, "training": module.training})
                + "\n",
                encoding="utf-8",
            )
        finally:
            handle.remove()

    handle = head.register_forward_hook(first_forward)
    cleanup.callback(handle.remove)
    return handle
