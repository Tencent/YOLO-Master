from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.utils.lora import _is_adapter_param
from ultralytics.utils.torch_utils import unwrap_model


def main() -> None:
    run_root = Path("runs/lora_rankless_quick")
    run_root.mkdir(parents=True, exist_ok=True)
    project = run_root.resolve()
    name = "ia3_cpu_smoke"

    print("[IA3] start training")
    model = YOLO("yolo11n.pt")
    model.train(
        data="coco8.yaml",
        epochs=1,
        imgsz=32,
        batch=2,
        workers=0,
        device="cpu",
        save=True,
        save_period=1,
        fraction=1.0,
        project=str(project),
        name=name,
        exist_ok=True,
        lora_type="ia3",
        lora_backend="peft",
        lora_gradient_checkpointing=False,
        plots=False,
        verbose=False,
    )

    live_model = unwrap_model(model.trainer.model)
    trainable = [(n, p.numel()) for n, p in live_model.named_parameters() if p.requires_grad]
    adapter = [(n, n_params) for n, n_params in trainable if _is_adapter_param(n)]
    non_adapter = [(n, n_params) for n, n_params in trainable if not _is_adapter_param(n)]
    trainable_total = sum(n_params for _, n_params in trainable)
    non_adapter_total = sum(n_params for _, n_params in non_adapter)

    print(f"[IA3] trainable_total={trainable_total:,}")
    print(f"[IA3] adapter_params={sum(n_params for _, n_params in adapter):,}")
    print(f"[IA3] non_adapter_params={non_adapter_total:,}")
    assert adapter, "IA3 adapter parameters were not created"
    assert trainable_total < 1_000_000, "IA3 unexpectedly re-enabled most base parameters"

    adapter_dir = project / name / "weights" / "lora_adapter_best"
    print(f"[IA3] adapter_dir={adapter_dir}")
    assert adapter_dir.exists(), f"adapter dir missing: {adapter_dir}"

    reloaded = YOLO("yolo11n.pt")
    ok = reloaded.load_lora(adapter_dir)
    print(f"[IA3] load_ok={ok}")
    assert ok, "load_lora returned False for IA3 adapter"
    ia3_names = [n for n, _ in reloaded.model.named_parameters() if "ia3_" in n.lower()]
    assert ia3_names, "no IA3 parameters found after loading adapter"

    print("[IA3] success")


if __name__ == "__main__":
    main()
