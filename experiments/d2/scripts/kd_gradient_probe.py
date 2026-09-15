"""Measure the gradient a Foundation KD loss actually delivers to the student.

Motivation: ``foundation_task_ratio`` compares loss *values*, and §5.5's weight sweep showed
a 16x weight range moving neither the detection loss nor the KD loss itself. This probe
measures the quantity that governs optimization instead -- gradient magnitude -- and reports
the weight that would put KD at a target share of the detection gradient.

Two modes:

``--mode kd``
    KD-side only. Pure torch, no teacher weights, no GPU, no ultralytics import. Reports how
    each ``foundation_loss`` form and each ``foundation_relation_samples`` value scales the
    gradient reaching the student feature. Runs anywhere.

``--mode ratio``
    Probe A from ``kd_gradient_analysis.md`` §5.2. Needs a real training setup (teacher weights
    + dataset). Measures ||d L_kd / d theta|| against ||d L_task / d theta|| over real batches
    and solves for ``w* = target_ratio * ||g_task|| / ||g_kd||``. Also reports the cosine between
    the two gradients, which is what revealed they are near-orthogonal.

``--mode learnable``
    Probe B from ``kd_gradient_analysis.md`` §5.3. Asks whether the KD objective is reachable at
    all: overfits one batch with the task loss switched off, once with the backbone trainable and
    once with it frozen. Separates "unreachable target" from "projector absorbs the signal" from
    "genuinely learnable".

Examples:
    $ python experiments/d2/scripts/kd_gradient_probe.py --mode kd
    $ python experiments/d2/scripts/kd_gradient_probe.py --mode ratio --batches 50 --target-ratio 0.1
    $ python experiments/d2/scripts/kd_gradient_probe.py --mode learnable --steps 300
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from ultralytics.nn.foundation.losses import cosine_kd_loss, relational_kd_loss

# wsweep actuals: batch=4, foundation_align_dim=256, imgsz=256 -> P4 grid 16x16.
DEFAULT_SHAPE = (4, 256, 16, 16)


def _l2_kd_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Mirror the ``l2`` branch of ``FoundationDistillationModel._kd_components_with_weights``."""
    return (student.float() - teacher.detach().float()).square().mean(dim=1).mean()


def _grad_norm(loss_fn, shape: tuple[int, int, int, int], seed: int) -> tuple[float, float]:
    """Return ``(loss_value, ||d loss / d student||_2)`` for one synthetic feature pair.

    Args:
        loss_fn (Callable): Maps ``(student, teacher)`` to a scalar loss.
        shape (tuple[int, int, int, int]): Student/teacher feature shape in ``(B, C, H, W)``.
        seed (int): Manual seed, so repeated calls are comparable.

    Returns:
        (tuple[float, float]): The loss value and the gradient norm at the student feature.

    Examples:
        >>> value, norm = _grad_norm(lambda s, t: (s - t).square().mean(), (2, 4, 3, 3), seed=0)
        >>> norm > 0
        True
    """
    torch.manual_seed(seed)
    student = torch.randn(*shape, requires_grad=True)
    teacher = torch.randn(*shape)
    loss = loss_fn(student, teacher)
    loss.backward()
    return float(loss.item()), float(student.grad.norm().item())


def probe_kd_side(shape: tuple[int, int, int, int] = DEFAULT_SHAPE, seed: int = 0) -> dict:
    """Compare the gradient scale of every config-selectable ``foundation_loss`` form.

    The numbers characterize each loss form's *reduction structure* (how many terms it averages
    over, whether its gradient is sign-only) under matched inputs. They are magnitudes for
    ranking, not calibrated values for a real feature distribution.

    Args:
        shape (tuple[int, int, int, int]): Feature shape in ``(B, C, H, W)``.
        seed (int): Manual seed shared by every measurement.

    Returns:
        (dict): Nested results for loss forms, relation-sample counts, and mismatch sensitivity.

    Examples:
        >>> out = probe_kd_side(shape=(2, 8, 4, 4))
        >>> sorted(out)
        ['loss_forms', 'mismatch_sensitivity', 'relation_samples']
    """
    forms = {
        "relational": lambda s, t: relational_kd_loss(s, t),
        "cosine": cosine_kd_loss,
        "l2": _l2_kd_loss,
        "hybrid": lambda s, t: cosine_kd_loss(s, t) + relational_kd_loss(s, t),
    }
    loss_forms = {name: dict(zip(("value", "grad_norm"), _grad_norm(fn, shape, seed))) for name, fn in forms.items()}

    samples = {}
    for count in (16, 32, 64, 128, 256):
        value, norm = _grad_norm(lambda s, t, n=count: relational_kd_loss(s, t, num_samples=n), shape, seed)
        samples[count] = {"value": value, "grad_norm": norm}

    # L1 on the Gram makes the gradient sign-only: being far from the teacher barely changes it.
    mismatch = {}
    for scale in (0.1, 0.5, 1.0, 4.0):
        torch.manual_seed(seed)
        student = torch.randn(*shape, requires_grad=True)
        teacher = student.detach() + scale * torch.randn(*shape)
        loss = relational_kd_loss(student, teacher)
        loss.backward()
        mismatch[scale] = {"value": float(loss.item()), "grad_norm": float(student.grad.norm().item())}

    return {"loss_forms": loss_forms, "relation_samples": samples, "mismatch_sensitivity": mismatch}


def _build_trainer(args: argparse.Namespace):
    """Set up a real DetectionTrainer with Foundation distillation active.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        (DetectionTrainer): A trainer whose ``model`` is the distillation wrapper and whose
            ``train_loader`` is ready to iterate.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": 1,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": 0,
        "device": args.device,
        "pretrained": False,
        "amp": False,
        "plots": False,
        "optimizer": "SGD",
        "lr0": 0.01,
        "foundation_enabled": True,
        "foundation_teacher": args.teacher,
        "foundation_model": args.teacher_model,
        # Measure the unweighted KD gradient; w is what we are solving for.
        "foundation_loss_weight": 1.0,
        "foundation_target_levels": args.target_levels,
        "foundation_multiscale": len(args.target_levels) > 1,
    }
    trainer = DetectionTrainer(overrides=overrides)
    # _setup_train() takes no arguments and builds the dataloader itself via
    # _build_train_pipeline(), so train_loader is available once it returns.
    trainer._setup_train()
    trainer.model.train()
    return trainer


def _summarize(values: list[float]) -> dict:
    """Mean plus the spread around it, so a mean near zero can be told from a tight zero.

    Args:
        values (list[float]): Per-batch measurements.

    Returns:
        (dict): n, mean, sample std, standard error, 95% CI on the mean, min and max.

    Examples:
        >>> out = _summarize([1.0, 1.0, 1.0])
        >>> out["mean"], out["std"]
        (1.0, 0.0)
        >>> _summarize([])["n"]
        0
    """
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    mean = sum(values) / n
    std = statistics.stdev(values) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "min": min(values),
        "max": max(values),
    }


def probe_gradient_ratio(args: argparse.Namespace) -> dict:
    """Measure ``||g_kd|| / ||g_task||`` on real batches and solve for the balancing weight.

    Builds the same Foundation distillation wrapper the trainer builds, then for each batch takes
    two separate backward passes over the shared trunk -- one from the KD term, one from the
    detection term -- and accumulates their gradient norms over the parameters feeding the tap.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        (dict): Mean gradient norms and the solved ``w*`` at each target ratio.
    """
    trainer = _build_trainer(args)
    model = trainer.model

    # Parameters upstream of the tap are what both losses compete over. The "task" side is the
    # whole non-KD objective (including the MoE auxiliary term when routing is active), because
    # that is the gradient the KD term actually competes against for the shared update.
    trunk = [p for p in model.parameters() if p.requires_grad]

    def _snapshot() -> list[torch.Tensor | None]:
        """Copy the current gradient of every trunk parameter."""
        return [None if p.grad is None else p.grad.detach().clone() for p in trunk]

    def _norm(grads: list[torch.Tensor | None]) -> float:
        return float(torch.sqrt(sum((g**2).sum() for g in grads if g is not None)).item())

    def _dot(a: list[torch.Tensor | None], b: list[torch.Tensor | None]) -> float:
        return float(sum((x * y).sum() for x, y in zip(a, b) if x is not None and y is not None).item())

    kd_norms: list[float] = []
    task_norms: list[float] = []
    cosines: list[float] = []
    for index, batch in enumerate(trainer.train_loader):
        if index >= args.batches:
            break
        batch = trainer.preprocess_batch(batch)
        captured = {}
        for which in ("kd", "task"):
            model.zero_grad(set_to_none=True)
            loss, _ = model(batch)
            # loss is cat([task_items..., foundation]); the trainer backwards loss.sum().
            component = loss[-1] if which == "kd" else loss[:-1].sum()
            component.backward()
            captured[which] = _snapshot()
        kd_norm, task_norm = _norm(captured["kd"]), _norm(captured["task"])
        kd_norms.append(kd_norm)
        task_norms.append(task_norm)
        # Negative cosine means the KD term is pulling against detection, not merely weakly:
        # a different problem from "too small", with the opposite remedy.
        cosines.append(_dot(captured["kd"], captured["task"]) / max(kd_norm * task_norm, 1e-12))

    mean_kd = sum(kd_norms) / max(len(kd_norms), 1)
    mean_task = sum(task_norms) / max(len(task_norms), 1)
    ratios = sorted({args.target_ratio, 0.05, 0.1, 0.2})
    cosine_stats = _summarize(cosines)
    return {
        "batches": len(kd_norms),
        "teacher": args.teacher,
        "target_levels": args.target_levels,
        "grad_norm_kd_at_w1": mean_kd,
        "grad_norm_task": mean_task,
        "observed_ratio_at_w1": mean_kd / max(mean_task, 1e-12),
        "w_star": {r: r * mean_task / max(mean_kd, 1e-12) for r in ratios},
        "target_ratio": args.target_ratio,
        "grad_cosine": cosine_stats,
        "grad_cosine_mean_differs_from_zero": not (cosine_stats["ci95_low"] <= 0.0 <= cosine_stats["ci95_high"]),
        "grad_conflict_fraction": sum(1 for c in cosines if c < 0) / max(len(cosines), 1),
        "per_batch_ratio": _summarize([k / max(t, 1e-12) for k, t in zip(kd_norms, task_norms)]),
        "per_batch_cosines": cosines,
    }


def probe_learnability(args: argparse.Namespace) -> dict:
    """Probe B: can the student reach the teacher's Gram structure at all?

    Fits a few batches with the task loss switched off entirely (only the KD component is
    backwarded), which removes data volume, schedule length and task interference as
    explanations. Runs two arms from identical starting weights:

    - ``full``: projector and backbone both trainable.
    - ``projector_only``: backbone frozen, so only the alignment projector can move.

    Two different questions are asked with two different instruments, which matters because they
    do not have the same answer:

    - **Reachable?** Measured on the FIT batches. If the objective barely moves even when the
      model may do nothing else, the target is unreachable and no loss weight fixes it.
    - **Does it need the backbone?** Measured on HELD-OUT batches. On the fit batches a
      trainable projector can drive almost any single batch down on its own, so comparing arms
      there is uninformative -- both saturate. Held-out is where a projector that merely
      re-mixes channels separates from a backbone that genuinely restructured its features.

    Adam is used rather than the training SGD deliberately: this asks whether the objective is
    reachable by any reasonable optimizer, not whether the configured recipe reaches it.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        (dict): Per-arm fit and held-out ``relational_raw`` plus a mechanical verdict.
    """
    trainer = _build_trainer(args)
    model = trainer.model
    loader = iter(trainer.train_loader)
    fit_batches = [trainer.preprocess_batch(next(loader)) for _ in range(args.fit_batches)]
    eval_batches = [trainer.preprocess_batch(next(loader)) for _ in range(args.eval_batches)]

    projector = model.projector
    if projector is None:
        raise RuntimeError("Foundation projector is not initialized; is foundation_enabled set?")
    projector_ids = {id(p) for p in projector.parameters()}

    trainable = [p for p in model.parameters() if p.requires_grad]
    pristine = [p.detach().clone() for p in trainable]
    original_flags = [p.requires_grad for p in trainable]
    # BatchNorm running stats also move during the overfit, so they must be rewound too or the
    # second arm would start from a state the first arm left behind.
    buffers = [b for b in model.buffers() if b.dtype.is_floating_point]
    pristine_buffers = [b.detach().clone() for b in buffers]

    def _rewind() -> None:
        with torch.no_grad():
            for param, saved, flag in zip(trainable, pristine, original_flags):
                param.copy_(saved)
                param.requires_grad_(flag)
            for buffer, saved in zip(buffers, pristine_buffers):
                buffer.copy_(saved)

    def _relational_raw() -> float:
        metrics = model.foundation_metrics()
        if "foundation_relational_raw" not in metrics:
            raise RuntimeError(
                "foundation_relational_raw missing from foundation_metrics(). The wrapper's forward() "
                "short-circuits to plain student inference unless model.training is True, so the KD "
                f"path never ran. Available keys: {sorted(metrics)}"
            )
        return float(metrics["foundation_relational_raw"])

    def _evaluate(batches: list) -> float:
        """Mean relational_raw over batches, without letting the measurement change the model.

        Must run in TRAIN mode: FoundationDistillationModel.forward() returns plain student
        inference and clears the metrics when self.training is False, so eval mode measures
        nothing. BatchNorm therefore updates running stats during the forward even under
        no_grad, so the buffers are rewound afterwards.
        """
        saved = [b.detach().clone() for b in buffers]
        model.train()
        with torch.no_grad():
            values = [(model(item), _relational_raw())[1] for item in batches]
        with torch.no_grad():
            for buffer, snapshot in zip(buffers, saved):
                buffer.copy_(snapshot)
        return sum(values) / max(len(values), 1)

    baseline_fit = _evaluate(fit_batches)
    baseline_eval = _evaluate(eval_batches)

    arms: dict[str, dict] = {}
    for arm in ("full", "projector_only"):
        # Restore the identical starting point so the two arms are genuinely comparable.
        _rewind()
        if arm == "projector_only":
            for param in trainable:
                if id(param) not in projector_ids:
                    param.requires_grad_(False)

        updating = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(updating, lr=args.probe_lr)
        trajectory: list[float] = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(fit_batches[step % len(fit_batches)])
            # loss is cat([task_items..., foundation]); take ONLY the KD component.
            loss[-1].backward()
            optimizer.step()
            trajectory.append(_relational_raw())

        fit_after, eval_after = _evaluate(fit_batches), _evaluate(eval_batches)
        arms[arm] = {
            "trainable_tensors": len(updating),
            "fit_before": baseline_fit,
            "fit_after": fit_after,
            "fit_reduction_pct": (baseline_fit - fit_after) / max(abs(baseline_fit), 1e-12) * 100.0,
            "heldout_before": baseline_eval,
            "heldout_after": eval_after,
            "heldout_reduction_pct": (baseline_eval - eval_after) / max(abs(baseline_eval), 1e-12) * 100.0,
            "trajectory_every_10": trajectory[::10],
        }

    _rewind()

    # Reachability is judged on the fit set; absorption only on held-out, because on the fit set
    # a trainable projector saturates in both arms and the comparison carries no information.
    reachable = arms["full"]["fit_reduction_pct"]
    full_gen = arms["full"]["heldout_reduction_pct"]
    residual_ratio = arms["projector_only"]["heldout_after"] / max(arms["full"]["heldout_after"], 1e-12)
    if reachable < args.learnable_threshold:
        verdict = (
            f"UNREACHABLE: KD-only fitting moved relational_raw {reachable:.1f}% on the batches it "
            f"was optimizing, below the {args.learnable_threshold}% bar. No loss weight fixes this."
        )
    elif full_gen < args.learnable_threshold:
        verdict = (
            f"FITS BUT DOES NOT GENERALIZE: {reachable:.1f}% on the fit batches, only "
            f"{full_gen:.1f}% held out. The objective is memorizable, not learnable."
        )
    elif residual_ratio <= args.absorb_factor:
        verdict = (
            f"PROJECTOR-ABSORBED: held out, freezing the backbone leaves residual error only "
            f"{residual_ratio:.2f}x the full arm's. The backbone is barely required."
        )
    else:
        verdict = (
            f"LEARNABLE: held out, freezing the backbone leaves {residual_ratio:.2f}x the residual "
            f"error of the full arm. The backbone contributes materially."
        )
    return {
        "steps": args.steps,
        "optimizer": f"Adam(lr={args.probe_lr})",
        "fit_batches": len(fit_batches),
        "heldout_batches": len(eval_batches),
        "arms": arms,
        # Residuals, not reduction percentages: near saturation two arms can both read ~99%
        # reduction while their remaining error differs by an order of magnitude.
        "heldout_residual_ratio_frozen_over_full": residual_ratio,
        "verdict": verdict,
    }


def main() -> None:
    """Parse arguments and print the requested probe as JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=("kd", "ratio", "learnable"),
        default="kd",
        help="kd: offline loss-form scales; ratio: Probe A; learnable: Probe B",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", type=int, default=50, help="ratio mode: batches to average over")
    parser.add_argument("--target-ratio", type=float, default=0.1, help="ratio mode: desired ||g_kd||/||g_task||")
    parser.add_argument("--steps", type=int, default=300, help="learnable mode: fitting steps per arm")
    parser.add_argument("--fit-batches", type=int, default=2, help="learnable mode: batches to fit on")
    parser.add_argument("--eval-batches", type=int, default=8, help="learnable mode: held-out batches")
    parser.add_argument("--probe-lr", type=float, default=1e-3, help="learnable mode: Adam learning rate")
    parser.add_argument(
        "--learnable-threshold",
        type=float,
        default=20.0,
        help="learnable mode: percent reduction in relational_raw below which the target is called unreachable",
    )
    parser.add_argument(
        "--absorb-factor",
        type=float,
        default=1.5,
        help="learnable mode: frozen-backbone held-out residual within this multiple of the full arm's is absorbed",
    )
    parser.add_argument("--model", default="ultralytics/cfg/models/26/yolo26-master-n.yaml")
    parser.add_argument("--data", default="coco128.yaml")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--teacher", default="dinov3")
    parser.add_argument("--teacher-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument(
        "--target-levels",
        default="p4",
        help="ratio/learnable mode: comma-separated student levels, e.g. 'p3,p4,p5' for the multiscale cells",
    )
    args = parser.parse_args()
    args.target_levels = [level.strip() for level in args.target_levels.split(",") if level.strip()]

    runners = {"kd": lambda: probe_kd_side(seed=args.seed), "ratio": lambda: probe_gradient_ratio(args)}
    runners["learnable"] = lambda: probe_learnability(args)
    print(json.dumps(runners[args.mode](), indent=2, default=float))


if __name__ == "__main__":
    main()
