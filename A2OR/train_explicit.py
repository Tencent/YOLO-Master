"""Explicit, reproducible training entry point for the current YOLO-Master model.

Examples:
    python A2OR/train_explicit.py --mode set --data-root D:\\coding\\datasets --dataset VisDrone
    python A2OR/train_explicit.py --name baseline_explicit
    python A2OR/train_explicit.py --dynamic-topk --name dtk_explicit --lambda 0.8 --k-min 3 --k-max 10
    python A2OR/train_explicit.py --dynamic-topk --candidate-expand --name dtk_axis_explicit --expand-8-16 24
    python A2OR/train_explicit.py --candidate-expand --name axis_decay_explicit --expand-8-16 24 \
        --expand-linear-decay --expand-full-epochs 60 --expand-decay-epochs 60
    python A2OR/train_explicit.py --print-config

The script prints the complete effective training request before importing and
starting training. It refuses to overwrite an existing run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml"
DEFAULT_DATA = ROOT / "A2OR/visdrone_full.yaml"
DEFAULT_PROJECT = ROOT / "A2OR/runs"
SETTINGS_PATH = ROOT / "A2OR/.runtime_data/settings.json"
LOCAL_DATA_ROOT = Path(r"D:\coding\datasets")
CLOUD_DATA_ROOT = Path("/workspace/datasets")
DEFAULT_DATASET = "VisDrone"

PERSISTED_FIELDS = {
    "model", "data", "data_root", "dataset", "project", "epochs", "patience", "batch", "nbs", "workers",
    "imgsz", "device", "seed", "optimizer", "amp", "verbose", "single_cls", "rect", "cos_lr", "multi_scale",
    "compile", "close_mosaic", "save_period", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "warmup_momentum", "warmup_bias_lr", "box", "cls", "cls_pw", "dfl", "hsv_h", "hsv_s", "hsv_v", "degrees",
    "translate", "scale", "shear", "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix",
    "tal_topk", "tal_alpha", "tal_beta", "lambda_value", "k_min", "k_max", "small_area", "medium_area",
    "expand_0_8", "expand_8_16", "expand_linear_decay", "expand_full_epochs", "expand_decay_epochs",
    "coverage_triggered", "coverage_min_candidates", "coverage_expand_target",
    "assignment_stats", "pretrained", "dynamic_topk", "candidate_expand",
}
PERSISTED_FLAG_ALIASES = {"lambda_value": "--lambda"}

RESUME_RUNTIME_FIELDS = {
    "imgsz", "batch", "device", "close_mosaic", "save_period", "workers", "cache", "epochs", "fraction",
    "patience", "time", "freeze", "val", "plots", "distill_model", "save_dir",
}

CLI_CONFIG_FLAGS = {
    "imgsz": "--imgsz", "batch": "--batch", "device": "--device", "close_mosaic": "--close-mosaic",
    "save_period": "--save-period", "workers": "--workers", "cache": "--cache", "epochs": "--epochs",
    "fraction": "--fraction", "patience": "--patience", "time": "--time", "freeze": "--freeze",
    "val": "--val", "plots": "--plots", "distill_model": "--distill-model", "save_dir": "--save-dir",
}


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Keep the examples readable while showing option defaults."""


HELP_EPILOG = """
Experiment switches:
  --dynamic-topk       Enable Dynamic TopK for small GTs.
  --candidate-expand   Enable per-side candidate expansion.
  --coverage-triggered  Enable C1 coverage-triggered expansion for scarce short-side [8,16) GTs.
  Neither switch is enabled by default, which is the baseline configuration.

Legacy --mode values (still accepted): baseline, dtk, axis, dtk-axis, set.
Prefer the independent switches above for new experiments.

Linear candidate contraction:
  Add --expand-linear-decay to an axis mode. The expansion stays at full strength
  for --expand-full-epochs, then decreases linearly to zero over
  --expand-decay-epochs. For the requested 60+60 schedule:

  python A2OR/train_explicit.py --candidate-expand --name stal_axis_decay_60_60 \\
      --expand-0-8 16 --expand-8-16 24 --expand-linear-decay \\
      --expand-full-epochs 60 --expand-decay-epochs 60

Use --print-config to audit the complete effective request without creating a run.
"""


def _path(value: str, *, must_exist: bool = False) -> Path:
    """Resolve a user-supplied path relative to the current working directory."""
    path = Path(value).expanduser()
    path = path if path.is_absolute() else Path.cwd() / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def parse_args() -> argparse.Namespace:
    """Parse all experiment-defining settings explicitly."""
    parser = argparse.ArgumentParser(
        description="Explicit, reproducible YOLO-Master training entry point.",
        epilog=HELP_EPILOG,
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=("set", "baseline", "dtk", "axis", "dtk-axis"), default=None,
        help="Legacy experiment selector; prefer --dynamic-topk and --candidate-expand.",
    )
    parser.add_argument(
        "--dynamic-topk", action=argparse.BooleanOptionalAction, default=False,
        help="Enable Dynamic TopK for small ground truths.",
    )
    parser.add_argument(
        "--candidate-expand", action=argparse.BooleanOptionalAction, default=False,
        help="Enable per-side candidate expansion.",
    )
    parser.add_argument("--name", default=None, help="Run name under --project; generated when omitted.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument(
        "--data-root", default=None,
        help="Parent datasets directory, containing the selected dataset; saved for later runs.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset directory name under data-root, e.g. VisDrone; saved for later runs.",
    )
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--epochs", type=int, default=120, help="Total training epochs.")
    parser.add_argument("--patience", type=int, default=0, help="Early-stopping patience; 0 disables it.")
    parser.add_argument("--batch", type=int, default=16, help="Physical batch size.")
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size for loss normalization.")
    parser.add_argument("--workers", type=int, default=0, help="Data-loader workers.")
    parser.add_argument("--imgsz", type=int, default=800, help="Training image size.")
    parser.add_argument("--device", default="0", help="CUDA device, CPU, or MPS selector.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--optimizer", default="auto", help="Optimizer name or auto.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--single-cls", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rect", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-momentum", type=float, default=0.8)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--cls-pw", type=float, default=0.0)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--bgr", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--tal-topk", type=int, default=10)
    parser.add_argument("--tal-alpha", type=float, default=0.5)
    parser.add_argument("--tal-beta", type=float, default=6.0)
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=0.8, help="DTK candidate ratio lambda.")
    parser.add_argument("--k-min", type=int, default=0, help="DTK minimum K for small GTs.")
    parser.add_argument("--k-max", type=int, default=10, help="DTK maximum K; 0 disables the bound.")
    parser.add_argument("--small-area", type=float, default=1024.0, help="Small-GT area threshold in pixels^2.")
    parser.add_argument("--medium-area", type=float, default=9216.0, help="Medium-GT upper area threshold.")
    parser.add_argument(
        "--expand-0-8", type=float, default=16.0,
        help="Per-side candidate target for original side [0,8); -1 disables this interval.",
    )
    parser.add_argument(
        "--expand-8-16", type=float, default=-1.0,
        help="Per-side candidate target for original side [8,16); -1 preserves baseline.",
    )
    parser.add_argument(
        "--expand-linear-decay", action=argparse.BooleanOptionalAction, default=False,
        help="Keep full candidate expansion, then linearly contract it to the original box (axis modes only).",
    )
    parser.add_argument(
        "--expand-full-epochs", type=int, default=60,
        help="Number of full-strength epochs before contraction (60 means epochs 1-60).",
    )
    parser.add_argument(
        "--expand-decay-epochs", type=int, default=60,
        help="Number of epochs over which expansion strength reaches zero (60 means epoch 120 ends at zero).",
    )
    parser.add_argument(
        "--coverage-triggered", action=argparse.BooleanOptionalAction, default=False,
        help="C1: expand short-side [8,16) GTs only when baseline candidate count is below the minimum.",
    )
    parser.add_argument(
        "--coverage-min-candidates", type=int, default=3,
        help="C1 target candidate count; core candidates are preserved.",
    )
    parser.add_argument(
        "--coverage-expand-target", type=float, default=20.0,
        help="C1 per-side expansion target for eligible [8,16) sides.",
    )
    parser.add_argument("--assignment-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", default=None, help="Checkpoint path belonging to the selected run directory.")
    parser.add_argument("--print-config", action="store_true", help="Print the effective request and exit without training.")
    args = apply_saved_defaults(parser.parse_args())
    supplied = {item.split("=", 1)[0] for item in sys.argv[1:] if item.startswith("--")}
    args._supplied_options = supplied
    if args.mode and args.mode != "set":
        if "--dynamic-topk" not in supplied and "--no-dynamic-topk" not in supplied:
            args.dynamic_topk = args.mode in {"dtk", "dtk-axis"}
        if "--candidate-expand" not in supplied and "--no-candidate-expand" not in supplied:
            args.candidate_expand = args.mode in {"axis", "dtk-axis"}

    if args.epochs < 1 or args.batch < 1 or args.nbs < 1 or args.workers < 0:
        parser.error("epochs, batch, and nbs must be positive; workers must be non-negative")
    if args.patience < 0 or args.save_period < -1 or args.tal_topk < 1:
        parser.error("patience >= 0, save-period >= -1, and tal-topk >= 1 are required")
    if not 0 <= args.lambda_value <= 1:
        parser.error("--lambda must be within [0, 1]")
    if args.k_min < 0 or args.k_max < 0 or (args.k_max and args.k_min > args.k_max):
        parser.error("require 0 <= --k-min <= --k-max, with 0 disabling the maximum")
    if not 0 < args.small_area < args.medium_area:
        parser.error("require 0 < --small-area < --medium-area")
    for option, value, lower in (("--expand-0-8", args.expand_0_8, 8), ("--expand-8-16", args.expand_8_16, 16)):
        if value != -1 and value < lower:
            parser.error(f"{option} must be -1 or >= {lower}")
    if args.expand_full_epochs < 0 or args.expand_decay_epochs < 0:
        parser.error("--expand-full-epochs and --expand-decay-epochs must be >= 0")
    if args.expand_linear_decay and args.expand_decay_epochs < 1:
        parser.error("--expand-decay-epochs must be >= 1 when --expand-linear-decay is enabled")
    if args.coverage_min_candidates < 1:
        parser.error("--coverage-min-candidates must be >= 1")
    if args.coverage_expand_target < 16.0:
        parser.error("--coverage-expand-target must be >= 16")
    if args.coverage_triggered and (args.dynamic_topk or args.candidate_expand):
        parser.error(
            "--coverage-triggered is an independent C1 mode; do not combine it with Dynamic TopK or axis expansion"
        )
    return args


def load_saved_settings() -> dict[str, str]:
    """Load the previously selected dataset root and dataset name."""
    if not SETTINGS_PATH.is_file():
        return {}
    try:
        value = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, AttributeError):
        return {}
    if not isinstance(value, dict):
        return {}
    # New settings are stored under params; flat keys keep old path settings compatible.
    params = value.get("params", {})
    return {**{k: v for k, v in value.items() if k != "params"}, **(params if isinstance(params, dict) else {})}


def apply_saved_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply saved values only when the corresponding CLI option was omitted."""
    saved = load_saved_settings()
    supplied = {item.split("=", 1)[0] for item in sys.argv[1:] if item.startswith("--")}
    for field in PERSISTED_FIELDS:
        flag = PERSISTED_FLAG_ALIASES.get(field, "--" + field.replace("_", "-"))
        if field in saved and flag not in supplied and "--no-" + flag[2:] not in supplied:
            setattr(args, field, saved[field])
    return args


def save_data_settings(root: Path, dataset: str) -> None:
    """Persist the selected parent datasets directory and dataset name."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps({"data_root": str(root), "dataset": dataset}, indent=2) + "\n", encoding="utf-8")


def save_full_settings(args: argparse.Namespace, data_root: Path | None, dataset: str | None) -> None:
    """Persist the complete set-mode training configuration."""
    params = {field: getattr(args, field) for field in PERSISTED_FIELDS if hasattr(args, field)}
    params["data_root"] = str(data_root) if data_root else params.get("data_root")
    params["dataset"] = dataset or params.get("dataset")
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps({"params": params}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_data_yaml(
    data: Path, data_root: str | None, dataset_name: str | None
) -> tuple[Path, Path | None, str | None]:
    """Return a data YAML rooted at ``<data_root>/<dataset_name>``."""
    settings = load_saved_settings()
    explicit = bool(data_root or dataset_name)
    dataset = dataset_name or settings.get("dataset") or DEFAULT_DATASET
    if not dataset or Path(dataset).name != dataset or dataset in {".", ".."}:
        raise ValueError("--dataset must be one directory name, such as VisDrone")
    requested_root = data_root or os.environ.get("YOLO_MASTER_DATA_ROOT")
    if requested_root:
        root = _path(requested_root)
    elif data.resolve() == DEFAULT_DATA.resolve():
        saved = _path(settings["data_root"]) if settings.get("data_root") else None
        candidates = [saved, LOCAL_DATA_ROOT, CLOUD_DATA_ROOT]
        root = next((candidate for candidate in candidates if candidate and (candidate / dataset).exists()), None)
        if root is None:
            raise FileNotFoundError(
                "Dataset root is not configured. Provide --data-root <datasets directory> "
                f"and --dataset {dataset}, for example D:\\coding\\datasets or /workspace/datasets."
            )
    else:
        return data, None

    if not root.is_dir():
        raise FileNotFoundError(f"Datasets directory not found: {root}")
    dataset_dir = root / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Expected dataset directory not found: {dataset_dir}")
    if explicit:
        save_data_settings(root, dataset)
    source = data.read_text(encoding="utf-8")
    if not re.search(r"(?m)^path:\s*.*$", source):
        raise ValueError(f"Dataset YAML has no top-level path entry: {data}")
    rendered = re.sub(r"(?m)^path:\s*.*$", f"path: {dataset_dir.as_posix()}", source, count=1)
    digest = hashlib.sha256(f"{data.resolve()}::{root.resolve()}".encode()).hexdigest()[:12]
    runtime_dir = ROOT / "A2OR" / ".runtime_data"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = runtime_dir / f"{data.stem}_{digest}.yaml"
    if not runtime_yaml.exists() or runtime_yaml.read_text(encoding="utf-8") != rendered:
        runtime_yaml.write_text(rendered, encoding="utf-8")
    return runtime_yaml, root, dataset


def make_config(args: argparse.Namespace) -> tuple[dict, Path, Path, Path, Path | None, str | None]:
    """Build the explicit Ultralytics request and resolved paths."""
    model = _path(args.model, must_exist=not args.resume)
    data = _path(args.data, must_exist=True)
    data, data_root, dataset = resolve_data_yaml(data, args.data_root, args.dataset)
    project = _path(args.project)
    resume = _path(args.resume, must_exist=True) if args.resume else None
    dynamic = args.dynamic_topk
    axis = args.candidate_expand
    variant = (
        "c1-coverage"
        if args.coverage_triggered
        else "dtk-axis"
        if dynamic and axis
        else "dtk"
        if dynamic
        else "axis"
        if axis
        else "baseline"
    )
    default_name = f"{variant}_explicit_{args.epochs}e_b{args.batch}"
    # A resume checkpoint already identifies its run directory.  Use it when
    # --name is omitted so a resume does not accidentally target a new run.
    name = args.name or (resume.parent.parent.name if resume else default_name)
    run_dir = project / name
    if resume:
        if resume.suffix.lower() != ".pt" or resume.parent.parent.resolve() != run_dir.resolve():
            raise ValueError("--resume must be a .pt checkpoint under project/name/weights")
    elif run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")

    config = {
        "data": str(data), "epochs": args.epochs, "patience": args.patience,
        "batch": args.batch, "nbs": args.nbs, "workers": args.workers, "imgsz": args.imgsz,
        "device": args.device, "seed": args.seed, "deterministic": True, "optimizer": args.optimizer,
        "amp": args.amp, "verbose": args.verbose, "single_cls": args.single_cls, "rect": args.rect,
        "cos_lr": args.cos_lr, "multi_scale": args.multi_scale, "compile": args.compile,
        "cache": False, "val": True, "split": "val", "fraction": 1.0,
        "close_mosaic": args.close_mosaic, "tal_topk": args.tal_topk,
        "lr0": args.lr0, "lrf": args.lrf, "momentum": args.momentum,
        "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum, "warmup_bias_lr": args.warmup_bias_lr,
        "box": args.box, "cls": args.cls, "cls_pw": args.cls_pw, "dfl": args.dfl,
        "hsv_h": args.hsv_h, "hsv_s": args.hsv_s, "hsv_v": args.hsv_v,
        "degrees": args.degrees, "translate": args.translate, "scale": args.scale,
        "shear": args.shear, "perspective": args.perspective, "flipud": args.flipud,
        "fliplr": args.fliplr, "bgr": args.bgr, "mosaic": args.mosaic,
        "mixup": args.mixup, "cutmix": args.cutmix,
        "tal_alpha": args.tal_alpha, "tal_beta": args.tal_beta,
        "tal_dynamic_topk_small": dynamic, "tal_dynamic_topk_lambda": args.lambda_value if dynamic else 0.0,
        "tal_dynamic_topk_cap": False, "tal_dynamic_topk_min": args.k_min if dynamic else 0,
        "tal_dynamic_topk_max": args.k_max if dynamic else 0,
        "assignment_stats": args.assignment_stats, "assignment_small_area": args.small_area,
        "assignment_medium_area": args.medium_area,
        "tal_candidate_expand_0_8": args.expand_0_8 if axis else 16.0,
        "tal_candidate_expand_8_16": args.expand_8_16 if axis else -1.0,
        "tal_candidate_expand_linear_decay": args.expand_linear_decay if axis else False,
        "tal_candidate_expand_full_epochs": args.expand_full_epochs if axis else 0,
        "tal_candidate_expand_decay_epochs": args.expand_decay_epochs if axis else 0,
        "tal_candidate_expand_coverage_triggered": args.coverage_triggered,
        "tal_candidate_expand_coverage_min": args.coverage_min_candidates,
        "tal_candidate_expand_coverage_target": args.coverage_expand_target,
        "save": True, "save_period": args.save_period, "plots": True,
        "project": str(project), "name": name, "exist_ok": False,
        "pretrained": args.pretrained, "resume": str(resume) if resume else False,
    }
    return config, model, data, run_dir, data_root, dataset


def merge_resume_config(config: dict, args: argparse.Namespace, resume: Path) -> tuple[dict, dict]:
    """Make checkpoint arguments authoritative and apply explicit runtime overrides only."""
    sys.path.insert(0, str(ROOT))
    from ultralytics.nn.tasks import load_checkpoint

    checkpoint_model, _ = load_checkpoint(resume)
    checkpoint_args = checkpoint_model.args
    checkpoint_args = dict(vars(checkpoint_args)) if hasattr(checkpoint_args, "__dict__") else dict(checkpoint_args)

    final_config = {key: value for key, value in checkpoint_args.items() if key in config}
    final_config.update({"model": str(resume), "resume": str(resume)})
    supplied = getattr(args, "_supplied_options", set())
    for field in RESUME_RUNTIME_FIELDS:
        flag = CLI_CONFIG_FLAGS.get(field)
        if flag and (flag in supplied or "--no-" + flag[2:] in supplied) and field in config:
            final_config[field] = config[field]

    # The explicit entry point owns the run location, while the checkpoint
    # owns the experiment configuration.
    final_config["project"] = config["project"]
    final_config["name"] = config["name"]
    final_config["exist_ok"] = config["exist_ok"]
    return final_config, checkpoint_args


def main() -> None:
    """Validate, print, and optionally start one explicit training run."""
    args = parse_args()
    if args.mode == "set":
        data = _path(args.data, must_exist=True)
        try:
            runtime_data, data_root, dataset = resolve_data_yaml(data, args.data_root, args.dataset)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"[configuration error] {exc}") from exc
        print("=== YOLO-Master dataset settings saved ===", flush=True)
        print(f"data_root={data_root}", flush=True)
        print(f"dataset={dataset}", flush=True)
        print(f"dataset_dir={data_root / dataset}", flush=True)
        print(f"resolved_yaml={runtime_data}", flush=True)
        save_full_settings(args, data_root, dataset)
        print(f"saved_settings={SETTINGS_PATH}", flush=True)
        saved = load_saved_settings()
        print(f"saved_parameter_count={len(saved)}", flush=True)
        print(json.dumps(saved, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return
    try:
        config, model, data, run_dir, data_root, dataset = make_config(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[configuration error] {exc}") from exc
    os.chdir(ROOT)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "A2OR/.ultralytics_config"))
    os.environ.setdefault("TQDM_ASCII", "1")
    checkpoint_args = None
    if args.resume:
        try:
            config, checkpoint_args = merge_resume_config(config, args, _path(args.resume, must_exist=True))
        except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise SystemExit(f"[resume configuration error] {exc}") from exc

    print("=== Explicit YOLO-Master effective training configuration ===", flush=True)
    print(f"model={config.get('model', model)} (pretrained={config.get('pretrained')}; resume={bool(args.resume)})", flush=True)
    print(f"data_yaml={config.get('data', data)}", flush=True)
    if data_root:
        print(f"data_root={data_root} (dataset={data_root / dataset})", flush=True)
        print(f"dataset={dataset}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    if checkpoint_args is not None:
        print("config_source=checkpoint train_args + explicit runtime overrides", flush=True)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if args.print_config:
        return
    sys.path.insert(0, str(ROOT))
    from ultralytics import YOLO

    YOLO(str(_path(args.resume, must_exist=True) if args.resume else model)).train(**config)


if __name__ == "__main__":
    main()
