# scripts/reproduce/reproduce_visdrone.py
import wandb
from ultralytics import YOLO


def main():
    # 说明：因算力限制，依据 Issue “可选，换用其他的垂类公开数据集” 的规则，采用 BCCD 垂类数据集进行复现
    wandb.init(project="YOLO-Master-Issue49", name="YOLO-Master-v0.1-N-BCCD")

    # 加载 v0.1-N 的基础配置
    model = YOLO("ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml")

    # 启动普通基线训练
    model.train(
        data="../datasets/BCCD-Blood-Cells-1/data.yaml",  # ✅ 已经替换为你真实的路径
        epochs=100,
        imgsz=640,
        batch=16,
        amp=False,
        optimizer="AdamW",
        project="YOLO-Master-Issue49",
        name="BCCD-v0.1-N-Baseline",
    )
    wandb.finish()


if __name__ == "__main__":
    main()
