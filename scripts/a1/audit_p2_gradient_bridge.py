"""Measure a forward-identical gradient bridge on fixed training images; never optimize weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


from scripts.a1.diagnostic_hooks import capture_head_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260829)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["A1_E2E_O2O_TAL_TOPK"] = "7"
    os.environ["A1_E2E_O2O_TAL_TOPK2"] = "1"
    os.environ["A1_E2E_O2O_CLS_GAIN"] = "1.0"
    import torch
    from run_p1_bn_frozen import enforce_p1_freeze_policy

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    torch.set_num_threads(2)
    data = check_det_dataset(str(args.data), autodownload=False)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 640, "cache": False, "workers": 0})
    dataset = build_yolo_dataset(cfg, data["train"], 1, data, mode="val", rect=False)
    if args.offset < 0 or args.images < 1 or args.offset + args.images > len(dataset):
        raise ValueError("invalid fixed image range")
    evidence = {
        "schema": "p2-gradient-bridge/v1",
        "status": "running",
        "seed": args.seed,
        "images": args.images,
        "offset": args.offset,
        "device": "cpu",
        "alpha": 0.1,
        "optimizer_steps": 0,
        "scope": "Fixed contiguous train subset, augmentation disabled, BN/base frozen, detection losses only",
        "checkpoints": {},
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    for cell in ("b", "d"):
        path = args.root / f"formal/seed{args.seed}/{cell}_formal_seed{args.seed}_15ep/weights/last.pt"
        evidence["checkpoints"][cell] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        model = YOLO(str(path), task="detect").model.float().cpu()
        model.args = get_cfg(overrides=model.args) if isinstance(model.args, dict) else model.args
        for n, p in model.named_parameters():
            p.requires_grad_(int(n.split(".")[1]) in {4, 6, 8, 23})
        model.train()
        trainer = SimpleNamespace(model=model)
        enforce_p1_freeze_policy(trainer)
        assert trainer.p1_frozen_factor_base_parameters == 459232
        before = {n: t.clone() for n, t in model.state_dict().items()}
        names, parameters = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
        head = model.model[-1]
        with capture_head_inputs(head) as inputs:
            criterion = model.init_criterion()
            native = getattr(criterion, "native_criterion", criterion)
            assert (native.one2one.assigner.topk, native.one2one.assigner.topk2) == (7, 1)
            for i in range(args.images):
                image_index = args.offset + i
                torch.manual_seed(260829 + image_index)
                batch = dataset.collate_fn([dataset[image_index]])
                batch["img"] = batch["img"].float() / 255
                preds = model(batch["img"])
                baseline_grads = {}
                row = {
                    "cell": cell,
                    "image_index": image_index,
                    "image": dataset.im_files[image_index],
                    "image_sha256": hashlib.sha256(Path(dataset.im_files[image_index]).read_bytes()).hexdigest(),
                    "gt": len(batch["cls"]),
                    "branches": {},
                }
                for tag, branch_preds, loss_fn in (
                    ("one2many", preds["one2many"], native.one2many),
                    ("native_one2one", preds["one2one"], native.one2one),
                    (
                        "bridge_one2one",
                        head.forward_head([x.detach() + 0.1 * (x - x.detach()) for x in inputs], **head.one2one),
                        native.one2one,
                    ),
                ):
                    total, _ = loss_fn.loss(branch_preds, batch)
                    grads = torch.autograd.grad(total.sum(), parameters, retain_graph=True, allow_unused=True)
                    groups = {}
                    for group, select in {
                        "factor": lambda n: int(n.split(".")[1]) in {4, 6, 8},
                        "router": lambda n: "routing" in n or "router" in n,
                        "one2one_head": lambda n: "one2one_" in n,
                    }.items():
                        vector = (
                            torch.cat(
                                [
                                    g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                                    for n, p, g in zip(names, parameters, grads)
                                    if select(n)
                                ]
                            )
                            if any(select(n) for n in names)
                            else torch.zeros(1)
                        )
                        groups[group] = {"l1": float(vector.abs().sum()), "l2": float(vector.norm())}
                        if tag == "one2many":
                            baseline_grads[group] = vector.detach().clone()
                        if tag == "bridge_one2one":
                            reference = baseline_grads[group]
                            norm = float(vector.norm() * reference.norm())
                            groups[group]["cosine_with_one2many"] = (
                                float(vector.dot(reference)) / norm if norm else None
                            )
                    row["branches"][tag] = {"loss": float(total.detach().sum()), "gradients": groups}
                    if tag == "bridge_one2one":
                        row["forward_max_error"] = max(
                            float((branch_preds[k] - preds["one2one"][k]).detach().abs().max())
                            for k in ("boxes", "scores")
                        )
                        assert row["forward_max_error"] == 0.0
                    del grads, total
                evidence["rows"].append(row)
                del preds, baseline_grads
                inputs.clear()
                if (i + 1) % 8 == 0:
                    save()
                    print(f"{cell}: {i + 1}/{args.images}", flush=True)

        changed = [n for n, t in model.state_dict().items() if not torch.equal(t, before[n])]
        evidence["checkpoints"][cell]["state_changed_keys"] = changed
        # Routing counters may advance in train mode. Parameters and frozen BN/base may not.
        evidence["checkpoints"][cell]["parameters_changed"] = [
            n for n, p in model.named_parameters() if not torch.equal(p, before[n])
        ]
        assert not evidence["checkpoints"][cell]["parameters_changed"]
        del model, before, criterion, native, parameters, branch_preds, inputs, head
    evidence["status"] = "completed"
    save()
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
