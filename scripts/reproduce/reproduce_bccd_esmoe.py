# scripts/reproduce/reproduce_sku110k.py
import torch
import wandb
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend


# ==================== 🐵 猴子补丁：修复框架热身 Bug ====================
# 替换掉原版容易产生 NaN 的 torch.empty，确保 EsMoE 路由安全通过 final_eval
def safe_warmup(self, imgsz=(1, 3, 640, 640)):
    im = torch.zeros(
        *imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device
    )
    self.forward(im)


AutoBackend.warmup = safe_warmup


# =========================================================================


def main():
    # 记录为 EsMoE 的专属名称
    wandb.init(project="YOLO-Master-Issue49", name="YOLO-Master-EsMoE-N-BCCD")

    # 统一使用基础 yaml，通过下方参数动态激活 MoE
    model = YOLO("ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml")

    model.train(
        data="../datasets/BCCD-Blood-Cells-1/data.yaml",  # ✅ 已经替换为你真实的路径
        epochs=100,
        imgsz=640,
        batch=16,
        amp=False,
        optimizer="AdamW",
        warmup_epochs=5,
        project="YOLO-Master-Issue49",
        name="BCCD-EsMoE-N-Result",
        # ==================== 🧠 激活 EsMoE 核心参数 ====================
        moe_num_experts=8,
        moe_top_k=2,
        moe_balance_loss=0.01,
        moe_router_z_loss=0.001,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
