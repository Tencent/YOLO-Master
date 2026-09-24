#!/usr/bin/env python3
"""Run an offline single-stage Foundation distillation smoke test.

This smoke deliberately avoids detector data and external teacher weights. It exercises the
minimum acceptance chain used by the real wrapper: frozen teacher feature -> live student P4
hook -> spatial/channel projector -> cosine loss -> optimizer step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_symbol(relative_path: str, symbol: str):
    """Load one symbol without importing the full optional YOLO runtime."""
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"_foundation_smoke_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load smoke dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


P4AlignmentProjector = _load_symbol("ultralytics/nn/foundation/projectors.py", "P4AlignmentProjector")
cosine_kd_loss = _load_symbol("ultralytics/nn/foundation/losses.py", "cosine_kd_loss")


class TinyStudent(nn.Module):
    """Tiny backbone with an explicit P4-like stage for the offline smoke."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.p4 = nn.Conv2d(8, 12, 3, stride=2, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.p4(torch.relu(self.stem(images)))


class FrozenTeacher(nn.Module):
    """Small deterministic teacher double with a different grid and channel count."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 20, 3, stride=4, padding=1, bias=False)
        self.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.encoder(images)


def run_smoke(*, steps: int = 16, seed: int = 20260824) -> dict[str, object]:
    """Run the deterministic smoke and return machine-readable acceptance evidence."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    torch.manual_seed(seed)
    student = TinyStudent().train()
    teacher = FrozenTeacher()
    projector = P4AlignmentProjector(student_channels=12, teacher_channels=20, align_dim=8, use_norm=False)
    captured: dict[str, torch.Tensor] = {}

    def capture_p4(_module: nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
        captured["p4"] = output

    handle = student.p4.register_forward_hook(capture_p4)
    optimizer = torch.optim.Adam((*student.parameters(), *projector.student_proj.parameters()), lr=0.03)
    images = torch.randn(2, 3, 32, 32)
    losses: list[float] = []
    student_shape: tuple[int, ...] | None = None
    teacher_shape: tuple[int, ...] | None = None
    aligned_shape: tuple[int, ...] | None = None
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            student(images)
            student_feature = captured.pop("p4")
            teacher_feature = teacher(images)
            student_aligned, teacher_aligned = projector(student_feature, teacher_feature)
            loss = cosine_kd_loss(student_aligned, teacher_aligned)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            student_shape = tuple(student_feature.shape)
            teacher_shape = tuple(teacher_feature.shape)
            aligned_shape = tuple(student_aligned.shape)
    finally:
        handle.remove()

    teacher_grad_free = all(parameter.grad is None for parameter in teacher.parameters())
    result = {
        "schema_version": 1,
        "stage": "p4",
        "steps": steps,
        "seed": seed,
        "student_shape": student_shape,
        "teacher_shape": teacher_shape,
        "aligned_shape": aligned_shape,
        "teacher_resized": bool(projector.alignment["teacher_resized"]),
        "teacher_frozen": all(not parameter.requires_grad for parameter in teacher.parameters()),
        "teacher_grad_free": teacher_grad_free,
        "projector_trainable": any(parameter.requires_grad for parameter in projector.student_proj.parameters()),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "finite_losses": all(torch.isfinite(torch.tensor(losses)).tolist()),
    }
    result["passed"] = all(
        (
            result["teacher_frozen"],
            result["teacher_grad_free"],
            result["projector_trainable"],
            result["loss_decreased"],
            result["finite_losses"],
            student_shape is not None,
            teacher_shape is not None,
            aligned_shape is not None,
            aligned_shape[-2:] == student_shape[-2:],
        )
    )
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the P0 smoke from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args(argv)
    result = run_smoke(steps=args.steps, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
