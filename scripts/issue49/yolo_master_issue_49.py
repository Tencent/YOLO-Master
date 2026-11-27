from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def is_repo_root(path: Path) -> bool:
    """Return whether ``path`` looks like the YOLO-Master repository root."""
    return (path / "ultralytics").exists() and ((path / "examples").exists() or (path / "scripts").exists())


def find_repo_root(start: Path) -> Path:
    """Walk upward from ``start`` until the repository root is found."""
    probe = start.resolve()
    while True:
        if is_repo_root(probe):
            return probe
        if probe == probe.parent:
            break
        probe = probe.parent
    raise FileNotFoundError("Could not locate YOLO-Master repo root from current working directory.")


def default_device() -> str:
    """Pick the default Ultralytics device string for this script."""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


ROOT = find_repo_root(Path(__file__).resolve().parent)
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tmp" / "ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO, settings
from ultralytics.data.utils import check_det_dataset

settings.update({"wandb": False})

ISSUE_DIR = ROOT / "scripts" / "issue49"
RUNS_DIR = Path(os.environ.get("YOLO_MASTER_RUNS_DIR", str(ISSUE_DIR / "runs"))).expanduser().resolve()
RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata describing a selectable model variant."""

    name: str
    cfg: Path
    uses_esmoe: bool


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata describing a selectable dataset."""

    name: str
    slug: str
    yaml: Path


@dataclass
class TrainConfig:
    """Command-line training configuration."""

    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = default_device()
    workers: int = 4
    seed: int = 42
    patience: int = 100
    amp: bool = True
    dense_eval_for_esmoe: bool = False
    lr0: float | None = None
    use_wandb: bool = True
    wandb_project: str = "yolo_master_issue49"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_group: str = "visdrone"
    run_tag: str = ""
    enable_lora: bool = False

    def wandb_enabled(self) -> bool:
        return self.use_wandb and self.wandb_mode != "disabled"


DATASET_SPECS = {
    "VisDrone": DatasetSpec(
        name="VisDrone",
        slug="visdrone",
        yaml=ROOT / "ultralytics" / "cfg" / "datasets" / "VisDrone.yaml",
    ),
    "SKU-110K": DatasetSpec(
        name="SKU-110K",
        slug="sku_110k",
        yaml=ROOT / "ultralytics" / "cfg" / "datasets" / "SKU-110K.yaml",
    ),
    "GlobalWheat2020": DatasetSpec(
        name="GlobalWheat2020",
        slug="globalwheat2020",
        yaml=ROOT / "ultralytics" / "cfg" / "datasets" / "GlobalWheat2020.yaml",
    ),
}

MODEL_SPECS = {
    "YOLO-Master-v0.1-N": ModelSpec(
        name="YOLO-Master-v0.1-N",
        cfg=ROOT / "ultralytics" / "cfg" / "models" / "master" / "v0_1" / "det" / "yolo-master-n.yaml",
        uses_esmoe=False,
    ),
    "YOLO-Master-EsMoE-N": ModelSpec(
        name="YOLO-Master-EsMoE-N",
        cfg=ROOT / "ultralytics" / "cfg" / "models" / "master" / "v0" / "det" / "yolo-master-n.yaml",
        uses_esmoe=True,
    ),
}

METRIC_COLUMNS = [
    "epoch",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/mixture_aux_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "val/mixture_aux_loss",
]

SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "run_name",
    "dense_eval",
    *METRIC_COLUMNS,
    "run_dir",
    "results_csv",
    "best_pt",
    "last_pt",
]


def build_train_overrides(cfg: TrainConfig) -> dict:
    """Return runtime overrides that keep baseline training stable by default."""
    overrides = {}

    if cfg.enable_lora:
        return {"lr0": cfg.lr0} if cfg.lr0 is not None else {}

    # Issue49 reproduces the plain model baselines by default. The repo-wide
    # default config currently enables LoRA, so pin PEFT features off unless the
    # caller explicitly opts in.
    overrides = {
        "lora_r": 0,
        "lora_save_adapters": False,
        "molora_num_experts": 0,
    }
    if cfg.lr0 is not None:
        overrides["lr0"] = cfg.lr0
    return overrides


def slugify(text: str) -> str:
    """Convert free-form text into a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def next_run_tag(project_dir: Path) -> str:
    """Return the next sequential ``runNNN`` tag for ``project_dir``."""
    existing = []
    pattern = re.compile(r"^run(\d{3})$")
    if project_dir.exists():
        for path in project_dir.iterdir():
            if not path.is_dir():
                continue
            suffix = path.name.rsplit("_", 1)[-1]
            match = pattern.match(suffix)
            if match:
                existing.append(int(match.group(1)))
    next_idx = max(existing, default=0) + 1
    return f"run{next_idx:03d}"


def read_last_row(results_csv: Path) -> dict[str, str]:
    """Read the last metrics row from an Ultralytics ``results.csv`` file."""
    if not results_csv.exists():
        return {}
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def to_float(value):
    """Best-effort conversion to ``float``."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_dataset(dataset_spec: DatasetSpec) -> dict:
    """Verify that a detection dataset is available locally."""
    return check_det_dataset(str(dataset_spec.yaml), autodownload=True)


def build_run_identity(model_slug: str, dataset_spec: DatasetSpec, run_tag: str, project_dir: Path) -> tuple[str, str]:
    """Build the stable run naming components used by this script."""
    resolved_run_tag = slugify(run_tag) if run_tag.strip() else next_run_tag(project_dir)
    run_name = f"{dataset_spec.slug}_{model_slug}_{resolved_run_tag}"
    return resolved_run_tag, run_name


def _custom_table(wandb, x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"):
    """Create a W&B area-under-curve table from interpolated series."""
    import polars as pl
    import polars.selectors as cs

    df = pl.DataFrame({"class": classes, "y": y, "x": x}).with_columns(cs.numeric().round(3))
    data = df.select(["class", "y", "x"]).rows()
    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wandb.plot_table(
        "wandb/area-under-curve/v0",
        wandb.Table(data=data, columns=["class", "y", "x"]),
        fields=fields,
        string_fields=string_fields,
    )


def _plot_curve(
    run,
    wandb,
    x,
    y,
    names=None,
    curve_id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """Log a PR-style curve to W&B using interpolated points."""
    import numpy as np

    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wandb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        run.log({title: wandb.plot.line(table, x_title, y_title, title=title)})
        return

    classes = ["mean"] * len(x_log)
    for i, yi in enumerate(y):
        x_log.extend(x_new)
        y_log.extend(np.interp(x_new, x, yi))
        classes.extend([names[i]] * len(x_new))
    run.log({curve_id: _custom_table(wandb, x_log, y_log, classes, title, x_title, y_title)}, commit=False)


def _log_plots(run, wandb, plots, step, processed_plots):
    """Log newly generated plot images once per timestamp."""
    for name, params in plots.copy().items():
        timestamp = params["timestamp"]
        if processed_plots.get(name) != timestamp:
            run.log({name.stem: wandb.Image(str(name))}, step=step)
            processed_plots[name] = timestamp


def make_dense_eval_callback():
    """Build a callback that forces dense inference for ES-MoE evaluation."""
    from ultralytics.nn.modules.moe.modules import ES_MOE
    from ultralytics.utils import LOGGER

    state = {"logged": False}

    def _callback(trainer):
        targets = []
        if getattr(trainer, "model", None) is not None:
            targets.append(trainer.model)
        ema = getattr(trainer, "ema", None)
        if ema is not None and getattr(ema, "ema", None) is not None:
            targets.append(ema.ema)

        changed = 0
        for target in targets:
            for module in target.modules():
                if isinstance(module, ES_MOE):
                    module.use_sparse_inference = False
                    changed += 1

        if changed and not state["logged"]:
            LOGGER.info(f"[issue49] dense eval enabled for {changed} ES_MOE modules")
            state["logged"] = True

    return _callback


def should_use_wandb(cfg: TrainConfig) -> bool:
    """Return whether custom W&B callbacks should be registered."""
    return cfg.wandb_enabled()


def wandb_init_custom(
    state: dict,
    run_name: str,
    model_spec: ModelSpec,
    cfg: TrainConfig,
    dense_eval: bool,
    resolved_run_tag: str,
    dataset_spec: DatasetSpec,
):
    """Initialize a W&B run for this script."""
    from ultralytics.utils import LOGGER

    if not should_use_wandb(cfg):
        return
    try:
        import wandb
    except Exception as exc:
        LOGGER.warning(f"[issue49] wandb unavailable: {exc}")
        return

    try:
        tags = [
            "issue49",
            dataset_spec.slug,
            slugify(cfg.wandb_group),
            slugify(model_spec.name),
            slugify(resolved_run_tag),
            "dense-eval" if dense_eval else "sparse-eval",
        ]
        state["wandb"] = wandb
        state["run"] = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            name=run_name,
            tags=tags,
            mode=cfg.wandb_mode,
            reinit=True,
            config={
                "dataset": dataset_spec.name,
                "dataset_slug": dataset_spec.slug,
                "wandb_group": cfg.wandb_group,
                "data": str(dataset_spec.yaml.relative_to(ROOT)),
                "model_name": model_spec.name,
                "model_cfg": str(model_spec.cfg.relative_to(ROOT)),
                "run_name": run_name,
                "run_tag": resolved_run_tag,
                "dense_eval_for_esmoe": dense_eval,
                **asdict(cfg),
            },
        )
        LOGGER.info(f"[issue49] wandb url: {getattr(state['run'], 'url', None)}")
    except Exception as exc:
        LOGGER.warning(f"[issue49] wandb init failed: {exc}")
        state["run"] = None


def wandb_log_ultralytics_defaults(state: dict, trainer, stage: str):
    """Mirror Ultralytics' default W&B logging behavior."""
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import model_info_for_loggers

    run = state["run"]
    wandb = state["wandb"]
    if run is None or wandb is None:
        return

    step = int(getattr(trainer, "epoch", 0)) + 1
    if stage == "train_epoch_end":
        try:
            run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=step)
            run.log(trainer.lr, step=step)
        except Exception as exc:
            LOGGER.warning(f"[issue49] wandb default train log failed at epoch {step}: {exc}")
        if trainer.epoch == 1:
            try:
                _log_plots(run, wandb, trainer.plots, step=step, processed_plots=state["processed_plots"])
            except Exception as exc:
                LOGGER.warning(f"[issue49] wandb default train plot log failed at epoch {step}: {exc}")
    elif stage == "fit_epoch_end":
        try:
            _log_plots(run, wandb, trainer.plots, step=step, processed_plots=state["processed_plots"])
            _log_plots(run, wandb, trainer.validator.plots, step=step, processed_plots=state["processed_plots"])
            if trainer.epoch == 0:
                run.log(model_info_for_loggers(trainer), step=step)
            run.log(trainer.metrics, step=step, commit=True)
        except Exception as exc:
            LOGGER.warning(f"[issue49] wandb default fit log failed at epoch {step}: {exc}")
    elif stage == "train_end":
        try:
            _log_plots(run, wandb, trainer.validator.plots, step=step, processed_plots=state["processed_plots"])
            _log_plots(run, wandb, trainer.plots, step=step, processed_plots=state["processed_plots"])
        except Exception as exc:
            LOGGER.warning(f"[issue49] wandb default final plot log failed: {exc}")
        try:
            art = wandb.Artifact(type="model", name=f"run_{run.id}_model")
            if trainer.best.exists():
                art.add_file(trainer.best)
                run.log_artifact(art, aliases=["best"])
        except Exception as exc:
            LOGGER.warning(f"[issue49] wandb default artifact log failed: {exc}")
        try:
            if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
                for curve_name, curve_values in zip(
                    trainer.validator.metrics.curves, trainer.validator.metrics.curves_results
                ):
                    x, y, x_title, y_title = curve_values
                    _plot_curve(
                        run,
                        wandb,
                        x,
                        y,
                        names=list(trainer.validator.metrics.names.values()),
                        curve_id=f"curves/{curve_name}",
                        title=curve_name,
                        x_title=x_title,
                        y_title=y_title,
                    )
        except Exception as exc:
            LOGGER.warning(f"[issue49] wandb default curve log failed: {exc}")


def wandb_finish_custom(state: dict):
    """Finish the active W&B run and clear callback state."""
    run = state["run"]
    if run is not None:
        try:
            run.finish()
        finally:
            state["run"] = None
            state["wandb"] = None


def make_wandb_callbacks(
    run_name: str,
    model_spec: ModelSpec,
    cfg: TrainConfig,
    dense_eval: bool,
    resolved_run_tag: str,
    dataset_spec: DatasetSpec,
):
    """Build the W&B callback mapping expected by ``YOLO.add_callback``."""
    state = {"run": None, "wandb": None, "processed_plots": {}}

    def on_train_start(trainer):
        wandb_init_custom(state, run_name, model_spec, cfg, dense_eval, resolved_run_tag, dataset_spec)

    def on_train_epoch_end(trainer):
        wandb_log_ultralytics_defaults(state, trainer, "train_epoch_end")

    def on_fit_epoch_end(trainer):
        wandb_log_ultralytics_defaults(state, trainer, "fit_epoch_end")

    def on_train_end(trainer):
        wandb_log_ultralytics_defaults(state, trainer, "train_end")
        wandb_finish_custom(state)

    return {
        "on_train_start": on_train_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }


def resolve_dataset(dataset_arg: str, dataset_name: str | None = None) -> DatasetSpec:
    """Resolve a dataset name or YAML path into a ``DatasetSpec``."""
    if dataset_arg in DATASET_SPECS:
        return DATASET_SPECS[dataset_arg]

    yaml_path = Path(dataset_arg).expanduser()
    if not yaml_path.is_absolute():
        yaml_path = (ROOT / yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    resolved_name = dataset_name or yaml_path.stem
    return DatasetSpec(name=resolved_name, slug=slugify(resolved_name), yaml=yaml_path)


def resolve_model(model_arg: str, model_name: str | None = None, uses_esmoe: bool = False) -> ModelSpec:
    """Resolve a model name or YAML path into a ``ModelSpec``."""
    if model_arg in MODEL_SPECS:
        return MODEL_SPECS[model_arg]

    cfg_path = Path(model_arg).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")

    resolved_name = model_name or cfg_path.stem
    inferred_esmoe = uses_esmoe or ("esmoe" in resolved_name.lower()) or ("esmoe" in cfg_path.as_posix().lower())
    return ModelSpec(name=resolved_name, cfg=cfg_path, uses_esmoe=inferred_esmoe)


def write_summary_csv(summary: dict, path: Path) -> None:
    """Write a one-row summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[c for c in SUMMARY_COLUMNS if c in summary])
        writer.writeheader()
        writer.writerow({k: summary.get(k) for k in writer.fieldnames})


def train_one(model_spec: ModelSpec, dataset_spec: DatasetSpec, cfg: TrainConfig) -> dict:
    """Train one selected model on one selected dataset."""
    ensure_dataset(dataset_spec)
    train_overrides = build_train_overrides(cfg)
    model_slug = slugify(model_spec.name.replace(".", ""))
    project_dir = RUNS_DIR / dataset_spec.slug / model_slug
    project_dir.mkdir(parents=True, exist_ok=True)
    tag_slug, run_name = build_run_identity(model_slug, dataset_spec, cfg.run_tag, project_dir)

    dense_eval = bool(model_spec.uses_esmoe and cfg.dense_eval_for_esmoe)
    model = YOLO(str(model_spec.cfg))

    if dense_eval:
        dense_cb = make_dense_eval_callback()
        model.add_callback("on_pretrain_routine_end", dense_cb)
        model.add_callback("on_train_start", dense_cb)

    if should_use_wandb(cfg):
        for event, callback in make_wandb_callbacks(
            run_name, model_spec, cfg, dense_eval, tag_slug, dataset_spec
        ).items():
            model.add_callback(event, callback)

    print(
        json.dumps(
            {
                "dataset": dataset_spec.name,
                "run_name": run_name,
                "model": model_spec.name,
                "cfg": str(model_spec.cfg.relative_to(ROOT) if model_spec.cfg.is_relative_to(ROOT) else model_spec.cfg),
                "data": str(
                    dataset_spec.yaml.relative_to(ROOT) if dataset_spec.yaml.is_relative_to(ROOT) else dataset_spec.yaml
                ),
                "dense_eval": dense_eval,
                "epochs": cfg.epochs,
                "imgsz": cfg.imgsz,
                "batch": cfg.batch,
                "device": cfg.device,
                "enable_lora": cfg.enable_lora,
                "train_overrides": train_overrides,
            },
            indent=2,
        )
    )

    model.train(
        data=str(dataset_spec.yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        seed=cfg.seed,
        deterministic=False,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=False,
        val=True,
        plots=True,
        patience=cfg.patience,
        amp=cfg.amp,
        verbose=True,
        **train_overrides,
    )

    run_dir = project_dir / run_name
    results_csv = run_dir / "results.csv"
    last_row = read_last_row(results_csv)
    summary = {
        "dataset": dataset_spec.name,
        "dataset_slug": dataset_spec.slug,
        "model": model_spec.name,
        "run_name": run_name,
        "run_tag": tag_slug,
        "run_dir": str(run_dir),
        "results_csv": str(results_csv),
        "best_pt": str(run_dir / "weights" / "best.pt"),
        "last_pt": str(run_dir / "weights" / "last.pt"),
        "dense_eval": dense_eval,
    }
    for key in METRIC_COLUMNS:
        summary[key] = to_float(last_row.get(key))

    with (run_dir / "issue49_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    summary_path = ISSUE_DIR / f"summary_{run_name}.csv"
    write_summary_csv(summary, summary_path)
    summary["summary_csv"] = str(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Train YOLO-Master issue49 experiments from the command line.")
    parser.add_argument("--dataset", default="VisDrone", help="Predefined dataset key or path to dataset YAML.")
    parser.add_argument("--dataset-name", help="Display name used when --dataset is a custom YAML path.")
    parser.add_argument("--model", default="YOLO-Master-v0.1-N", help="Predefined model key or path to model YAML.")
    parser.add_argument("--model-name", help="Display name used when --model is a custom YAML path.")
    parser.add_argument("--uses-esmoe", action="store_true", help="Mark a custom model as ES-MoE capable.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr0", type=float, help="Optional override for the initial learning rate.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument(
        "--dense-eval-for-esmoe", action="store_true", help="Force dense inference during eval for ES-MoE runs."
    )
    parser.add_argument("--run-tag", default="", help="Explicit run tag. Default auto-assigns run001/run002/...")
    parser.add_argument("--wandb-project", default="yolo_master_issue49")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-group", default="visdrone")
    parser.add_argument("--no-wandb", action="store_true", help="Disable custom W&B logging.")
    parser.add_argument(
        "--enable-lora",
        action="store_true",
        help="Opt into the repo-wide LoRA defaults. Off by default so issue49 runs stay baseline-only.",
    )
    parser.add_argument("--list-datasets", action="store_true", help="Print predefined dataset keys and exit.")
    parser.add_argument("--list-models", action="store_true", help="Print predefined model keys and exit.")
    return parser


def print_catalog(title: str, items: dict[str, object]) -> None:
    """Print a compact catalog of predefined datasets or models."""
    print(title)
    for key, spec in items.items():
        path = spec.yaml if hasattr(spec, "yaml") else spec.cfg
        print(f"  {key}: {path}")


def main() -> int:
    """Run the CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.list_datasets:
        print_catalog("Predefined datasets:", DATASET_SPECS)
        return 0
    if args.list_models:
        print_catalog("Predefined models:", MODEL_SPECS)
        return 0

    dataset_spec = resolve_dataset(args.dataset, args.dataset_name)
    model_spec = resolve_model(args.model, args.model_name, args.uses_esmoe)
    cfg = TrainConfig(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        lr0=args.lr0,
        amp=not args.no_amp,
        dense_eval_for_esmoe=args.dense_eval_for_esmoe,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_group=args.wandb_group,
        run_tag=args.run_tag,
        enable_lora=args.enable_lora,
    )

    print(f"repo root: {ROOT}")
    print(f"python: {sys.executable}")
    print(f"issue dir: {ISSUE_DIR}")
    print(f"runs dir: {RUNS_DIR}")

    summary = train_one(model_spec, dataset_spec, cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
