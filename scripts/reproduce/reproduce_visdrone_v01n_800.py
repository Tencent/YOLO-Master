import os
import wandb
from multiprocessing import freeze_support
from ultralytics import YOLO

def main():
    # Wandb配置（不需要可以注释掉，不影响训练）
    # os.environ["WANDB_MODE"] = "online"
    # os.environ["WANDB__SERVICE_WAIT"] = "300"
    # wandb.login()
    # wandb_run = wandb.init(
    #     project="VisDrone",
    #     name="v0.1_N_800_ep120",
    #     config={
    #         "dataset": "VisDrone2019",
    #         "imgsz": 800,
    #         "batch_size": 6,
    #         "epochs": 120
    #     }
    # )

    # 加载v0.1-N模型配置
    model = YOLO("ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml")

    # 启动训练（参数和你本次训练完全一致，保证可复现）
    model.train(
        data="ultralytics/cfg/datasets/VisDrone.yaml",
        imgsz=800,
        batch=6,
        epochs=120,
        project="VisDrone",
        name="v0.1_N_800_ep120",
        workers=4,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        patience=20,
        close_mosaic=15,
        plots=True
    )

    # 训练结束自动跑验证，输出最终指标
    results = model.val(
        data="ultralytics/cfg/datasets/VisDrone.yaml",
        imgsz=800,
        batch=8,
        plots=True,
        save_json=True
    )

    print("===== 最终验证指标 =====")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    # wandb.finish()

if __name__ == '__main__':
    freeze_support()
    main()
