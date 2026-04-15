from pathlib import Path

import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def main() -> None:
    run_root = Path("runs/lora_e2e_quick")
    run_root.mkdir(parents=True, exist_ok=True)
    project = run_root.resolve()
    name = "cpu_smoke"

    print(f"[E2E] ultralytics_file={ultralytics.__file__}")
    print("[E2E] start training")
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
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_gradient_checkpointing=False,
        verbose=False,
    )

    train_dir = project / name / "weights"
    adapter_dir = train_dir / "lora_adapter_best"
    print(f"[E2E] adapter_dir={adapter_dir}")
    print(f"[E2E] adapter_exists={adapter_dir.exists()}")
    assert adapter_dir.exists(), f"adapter dir missing: {adapter_dir}"

    print("[E2E] start reload")
    reloaded = YOLO("yolo11n.pt")
    ok = reloaded.load_lora(adapter_dir)
    print(f"[E2E] load_ok={ok}")
    assert ok, "load_lora returned False"
    assert getattr(reloaded.model, "lora_enabled", False), "lora_enabled flag missing after load"

    results = reloaded.predict(source=str(ASSETS / "bus.jpg"), imgsz=32, device="cpu", verbose=False)
    print(f"[E2E] predict_len={len(results)}")
    assert len(results) == 1, "unexpected prediction count after loading LoRA"

    lora_param_names = [n for n, _ in reloaded.model.named_parameters() if "lora_" in n]
    print(f"[E2E] lora_param_count={len(lora_param_names)}")
    assert lora_param_names, "no lora parameters found after loading adapter"

    print("[E2E] success")


if __name__ == "__main__":
    main()
