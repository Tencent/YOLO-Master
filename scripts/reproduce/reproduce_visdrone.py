#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复现 VisDrone2019：YOLO-Master v0.1-N vs EsMoE-N
单模型独立训练，强制 WandB 在线，数据完整保留

用法:
    python scripts/reproduce/reproduce_visdrone.py --model v01   # 训练 v0.1-N
    python scripts/reproduce/reproduce_visdrone.py --model moe   # 训练 EsMoE-N
"""

import os
import sys
import argparse
import traceback
from multiprocessing import freeze_support

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import wandb
from ultralytics import YOLO

# ================== 环境 & WandB 强制在线 ==================
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_SERVICE_WAIT"] = "300"
os.environ["WANDB_INIT_TIMEOUT"] = "300"
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_DISABLE_SERVICE"] = "True"
os.environ["WANDB_SILENT"] = "false"
os.environ["WANDB_LOG_MODEL"] = "false"

# ================== 固定超参（针对 VisDrone 优化）==================
DATA_YAML = "ultralytics/cfg/datasets/VisDrone.yaml"
IMG_SIZE   = 800           # VisDrone 推荐较高分辨率以保留小目标
EPOCHS     = 120
BATCH      = 6             # 8G 显存安全值，若 OOM 可改为 4
DEVICE     = "0"
PROJECT    = "VisDrone_Reproduce"
NAME_BASE  = "VisDrone"

NOTE_V01 = (
    "YOLO-Master v0.1-N on VisDrone2019; "
    "imgsz=800, batch=6, amp, cache=disk"
)
NOTE_MOE = (
    "YOLO-Master EsMoE-N on VisDrone2019; "
    "imgsz=800, batch=6, amp, cache=disk"
)

def ensure_wandb_login():
    """确保 WandB 可用，无 API key 则用匿名模式"""
    try:
        if wandb.api.api_key:
            print("✅ WandB API key 已配置")
            return True
    except Exception:
        pass
    key = os.environ.get("WANDB_API_KEY")
    if key:
        wandb.login(key=key)
        return True
    print("⚠️ 未检测到 API key，使用匿名模式（数据仍保留，链接公开）")
    wandb.init(anonymous="allow", project=PROJECT, mode="online")
    wandb.finish()
    return False

def init_run(run_name, notes):
    """初始化 WandB run，带超时保护"""
    for attempt in range(3):
        try:
            run = wandb.init(
                project=PROJECT,
                name=run_name,
                notes=notes,
                resume="allow",
                config={"imgsz": IMG_SIZE, "batch": BATCH, "epochs": EPOCHS},
                settings=wandb.Settings(_service_wait=300, init_timeout=120)
            )
            print(f"✅ WandB Run 初始化成功, ID: {run.id}")
            return run
        except Exception as e:
            print(f"❌ 初始化失败 (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                import time
                time.sleep(10)
            else:
                raise RuntimeError("无法连接 WandB，请检查网络或 API key") from e

def train(model_cfg, note, run_name):
    """单次训练流程（针对 VisDrone 密集小目标优化）"""
    print(f"\n{'='*60}")
    print(f"🚀 正在训练: {run_name}")
    print(f"📁 数据: {DATA_YAML}")
    print(f"⚙️ 模型: {model_cfg}")
    print(f"🍃 参数: imgsz={IMG_SIZE}, batch={BATCH}, epochs={EPOCHS}")
    print(f"{'='*60}")

    ensure_wandb_login()
    run = init_run(run_name, note)

    try:
        model = YOLO(model_cfg)
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            amp=True,
            # 使用磁盘缓存，避免内存爆炸
            cache="disk",
            # Windows 下 workers=0 最稳定，避免 I/O 死锁
            workers=0,
            # 数据增强（复现原始 VisDrone 脚本中的经验设置）
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            # 最后 15 个 epoch 关闭 mosaic，稳定最终精度
            close_mosaic=15,
            # 早停，避免无意义训练
            patience=20,
            # 不启用矩形训练（VisDrone 图像尺寸差异大，rect 无益）
            rect=False,
            project=PROJECT,
            name=run_name,
            exist_ok=True,
            save=True,
            save_period=10,
            plots=True,
        )
        print(f"✅ {run_name} 训练完成")

        # 最终验证
        val_results = model.val(data=DATA_YAML, split='val')
        print(f"📊 验证结果: mAP50={val_results.box.map50:.4f}, mAP50-95={val_results.box.map:.4f}")
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        traceback.print_exc()
    finally:
        print("⏳ 同步 WandB 数据...")
        wandb.finish()
        print("✅ WandB Run 已结束，数据已完整上传。")

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="VisDrone 复现训练")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["v01", "moe"],
        help="选择要训练的模型: v01 (YOLO-Master v0.1-N) 或 moe (EsMoE-N)"
    )
    args = parser.parse_args()

    if args.model == "v01":
        train(
            model_cfg="ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",
            note=NOTE_V01,
            run_name=f"{NAME_BASE}_v01_800_ep{EPOCHS}"
        )
    else:  # moe
        train(
            model_cfg="ultralytics/cfg/models/master/exp/yolo-master-v0_10.yaml",
            note=NOTE_MOE,
            run_name=f"{NAME_BASE}_MoE_800_ep{EPOCHS}"
        )

    print("\n🎉 训练完毕。前往 https://wandb.ai 查看实验。")
    print("📘 将项目设置为 Public 即可获取公开对比链接。")