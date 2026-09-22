"""A1 S1 阶段1：验证 C 格训练中 o2m/o2o 两条 loss 均正常下降。

背景：E2ELoss.__call__ 只把 one2one 分支的 items 上报日志（loss.py:1217-1221），
one2many 分支的 loss 数值不可见。本脚本 monkeypatch E2ELoss.__call__ 补打两条 loss，
复用训练官方管线跑 3 epochs，观察两条 loss 是否均下降。

用法：
    python scripts/a1/verify_o2m_o2o_loss.py
"""

from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss

_PRINT_HEADER = "[o2m_o2o_verify]"
_orig_call = E2ELoss.__call__


def _patched_call(self, preds, batch):
    """复刻原 __call__ 逻辑，补打 o2m/o2o 两条 loss 后再返回原组合结果。

    注：本仓库 v8DetectionLoss 的 loss[0] 为 3 元素 tensor（box/cls/dfl 分项，
    各乘 gain），非标量；items（loss[1]）为 detach 后的同结构分项。
    """
    preds = self.one2many.parse_output(preds)
    one2many, one2one = preds["one2many"], preds["one2one"]
    loss_m = self.one2many.loss(one2many, batch)
    loss_o = self.one2one.loss(one2one, batch)
    m_items = [round(x, 4) for x in loss_m[0].detach().tolist()]
    o_items = [round(x, 4) for x in loss_o[0].detach().tolist()]
    m_sum = round(loss_m[0].detach().sum().item(), 4)
    o_sum = round(loss_o[0].detach().sum().item(), 4)
    print(f"{_PRINT_HEADER} o2m box/cls/dfl={m_items} sum={m_sum}")
    print(f"{_PRINT_HEADER} o2o box/cls/dfl={o_items} sum={o_sum}")
    return loss_m[0] * self.o2m + loss_o[0] * self.o2o, loss_o[1]


def main() -> None:
    E2ELoss.__call__ = _patched_call
    try:
        model = YOLO("ultralytics/cfg/models/26/yolo26.yaml")
        model.train(
            data="coco8.yaml",
            epochs=3,
            imgsz=64,
            batch=16,
            device=0,
            seed=0,
            project="runs/agent",
            name="a1-s1-smoke-c-lossverify",
            plots=False,
        )
    finally:
        E2ELoss.__call__ = _orig_call


if __name__ == "__main__":
    main()
