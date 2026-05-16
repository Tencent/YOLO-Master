"""
PEFT 变体对比验证脚本 (coco128, MPS / CPU)
==========================================
目的: 验证 lora_type=full/lora/dora/loha/ia3 是否真的改变 trainable 参数 + 训练曲线
不依赖 WandB，所有结果落到本地 JSON / 控制台
"""
import os
import sys
import json
import time
from pathlib import Path

# ★ 关键：强制使用当前仓库的 ultralytics（避免被全局别的 ultralytics 包抢占）
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ---- 强制单进程 + 离线 ----
os.environ["WANDB_MODE"]   = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 关闭 ultralytics 自动 hub 上报
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE",     "false")

import torch
import ultralytics
print(f"[Boot] ultralytics loaded from: {ultralytics.__file__}")
print(f"[Boot] ultralytics version    : {ultralytics.__version__}")
assert str(REPO_ROOT) in ultralytics.__file__, (
    f"加载的不是当前仓库的 ultralytics！got {ultralytics.__file__}, expected under {REPO_ROOT}"
)

from ultralytics.utils import SETTINGS
SETTINGS["wandb"] = False                # 本地验证不需要

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
print(f"[Boot] lora_backend in default: {'lora_backend' in DEFAULT_CFG_DICT}")
print(f"[Boot] lora_type in default   : {'lora_type'    in DEFAULT_CFG_DICT}")
print(f"[Boot] lora_use_dora exists   : {'lora_use_dora' in DEFAULT_CFG_DICT}")


HERE        = Path(__file__).parent
MODEL_PATH  = HERE / "yolo11n.pt"
DATA_YAML   = "coco128.yaml"             # ultralytics 内置数据集 (auto-download)
PROJECT_DIR = HERE / "runs"
RESULTS_JSON = HERE / "peft_compare_results.json"

EPOCHS = 2          # 缩短到 2 epoch（每个变体 ~2-3 分钟）
BATCH  = 8
IMGSZ  = 320
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 强制行缓冲，方便实时观察后台进度
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 5 个变体配置；full 不传 lora_* 参数即可走原生全量微调
# 注意: 本地 ultralytics 的 lora_type 合法值 = ["lora","loha","lokr","adalora","ia3","oft","boft","hra"]
#       DoRA 通过 lora_use_dora=True 单独开关，不在 lora_type 里
LORA_R = 8
LORA_ALPHA = 16
COMMON = {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_backend": "peft", "lora_dropout": 0.05}

VARIANTS = [
    {"name": "full",  "kwargs": {}},
    {"name": "lora",  "kwargs": {"lora_type": "lora",  **COMMON}},
    {"name": "dora",  "kwargs": {"lora_type": "lora",  "lora_use_dora": True, **COMMON}},  # ★ DoRA 通过 use_dora 开关
    {"name": "loha",  "kwargs": {"lora_type": "loha",  **COMMON}},
    {"name": "ia3",   "kwargs": {"lora_type": "ia3",   "lora_backend": "peft"}},  # ia3 不需 r/alpha
]


def count_params(m: torch.nn.Module):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def detect_adapter_signature(m: torch.nn.Module):
    names = [n for n, _ in m.named_parameters()]
    return {
        "has_lora_A":             any("lora_A" in n            for n in names),
        "has_lora_B":             any("lora_B" in n            for n in names),
        "has_dora_magnitude":     any("magnitude_vector" in n  for n in names),
        "has_loha":               any("hada"   in n.lower()    for n in names),
        "has_ia3":                any("ia3"    in n.lower()    for n in names),
        "n_lora_params":          sum(1 for n in names if "lora_" in n.lower()),
    }


def run_one(variant):
    name   = variant["name"]
    kwargs = variant["kwargs"]
    print(f"\n{'='*70}\n=== Variant: {name.upper()} {'='*40}\n{'='*70}")
    print(f"kwargs = {kwargs}")

    t0 = time.time()
    model = YOLO(str(MODEL_PATH))

    # 训练前先看一次 baseline
    base_total, base_train = count_params(model.model)
    print(f"[Pre-train] total={base_total:,} trainable={base_train:,} ({base_train/base_total*100:.2f}%)")

    try:
        results = model.train(
            data    = DATA_YAML,
            epochs  = EPOCHS,
            batch   = BATCH,
            imgsz   = IMGSZ,
            device  = DEVICE,
            project = str(PROJECT_DIR),
            name    = f"v_{name}",
            exist_ok= True,
            verbose = False,
            workers = 2,
            patience= 0,         # 不早停
            plots   = False,     # 不生成各种图，加速
            save    = False,     # 不保 checkpoint
            **kwargs,
        )
        ok = True
        err = None
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
        results = None
        print(f"[ERROR] {err}")

    elapsed = time.time() - t0

    # 训练后再次统计 (PEFT 注入会发生在 train 内)
    post_total, post_train = count_params(model.model)
    sig = detect_adapter_signature(model.model)

    # 提取最终 metrics
    final_metrics = {}
    if ok and results is not None and hasattr(results, "results_dict"):
        final_metrics = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}
    elif ok:
        # 兜底: 从 trainer.metrics 取
        final_metrics = {k: float(v) for k, v in getattr(model.trainer, "metrics", {}).items() if isinstance(v, (int, float))}

    record = {
        "name":           name,
        "ok":             ok,
        "error":          err,
        "elapsed_sec":    round(elapsed, 1),
        "params_total":   post_total,
        "params_trainable": post_train,
        "trainable_pct":  round(post_train / post_total * 100, 4),
        "delta_total_vs_baseline": post_total - base_total,
        "adapter_sig":    sig,
        "lora_type":      getattr(model.model, "lora_type", None),
        "lora_backend":   getattr(model.model, "lora_backend", None),
        "final_metrics":  final_metrics,
    }
    print(f"[Post-train] total={post_total:,} trainable={post_train:,} ({record['trainable_pct']}%)")
    print(f"[Adapter Sig] {sig}")
    print(f"[Final metrics] {json.dumps(final_metrics, indent=2)}")
    return record


def main():
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH} | Imgsz: {IMGSZ}")
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {DATA_YAML}")
    PROJECT_DIR.mkdir(exist_ok=True, parents=True)

    all_records = []
    for v in VARIANTS:
        rec = run_one(v)
        all_records.append(rec)
        # 实时落盘，单个失败不丢之前结果
        RESULTS_JSON.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))

    # ============== 汇总表 ==============
    print("\n" + "="*100)
    print(f"{'Variant':<8} {'OK':<3} {'Total':>11} {'Trainable':>11} {'%':>7} {'lora_A':>7} {'dora_mag':>9} {'loha':>5} {'ia3':>4} {'mAP50-95':>10}")
    print("-"*100)
    for r in all_records:
        m = r["final_metrics"].get("metrics/mAP50-95(B)", float("nan"))
        sig = r["adapter_sig"]
        print(f"{r['name']:<8} {'Y' if r['ok'] else 'N':<3} "
              f"{r['params_total']:>11,} {r['params_trainable']:>11,} {r['trainable_pct']:>7.3f} "
              f"{str(sig['has_lora_A']):>7} {str(sig['has_dora_magnitude']):>9} "
              f"{str(sig['has_loha']):>5} {str(sig['has_ia3']):>4} "
              f"{m if isinstance(m, float) else '':>10}")
    print("="*100)
    print(f"\n详细结果: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
