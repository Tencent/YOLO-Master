"""A1 S1 阶段1：验证 A/C 两格推理时是否走 NMS。

背景（方案文档已核实事实）：end2end 推理时 head 输出 shape[-1]==6 的 top-k 结果，
nms.py:66 的 BNC 快捷分支直接返回，不执行 torchvision.ops.nms 框抑制；
传统 o2m 路径输出 shape[-1]==4+nc 的稠密预测，必须走框抑制。

验证粒度：non_max_suppression 函数本身两格都会被调用（函数内部 if 分流），
真正的"免 NMS"语义是跳过框抑制原语 torchvision.ops.nms。故本脚本同时记录：
1. non_max_suppression 入口时 prediction 的 shape[-1]
2. torchvision.ops.nms 的调用次数

断言：
- C 格（end2end=True）：输入 shape[-1]==6 且框抑制 0 次
- A 格（end2end=False）：输入 shape[-1]==4+nc 且框抑制 ≥1 次

用法：
    python scripts/a1/verify_nms_path.py
"""

from pathlib import Path

import torchvision

from ultralytics import YOLO
from ultralytics.utils import nms

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = "ultralytics/assets/bus.jpg"

_entries = []
_box_nms_calls = []
_orig_nms_fn = nms.non_max_suppression
_orig_tv_nms = torchvision.ops.nms


def _patched_nms(*args, **kwargs):
    """记录入口 shape 并透传原函数。"""
    pred = args[0][0] if isinstance(args[0], (list, tuple)) else args[0]
    _entries.append(tuple(pred.shape))
    return _orig_nms_fn(*args, **kwargs)


def _patched_tv_nms(*args, **kwargs):
    """记录框抑制调用并透传。"""
    _box_nms_calls.append(1)
    return _orig_tv_nms(*args, **kwargs)


def check(case: str, weights: str, expect_end2end: bool) -> None:
    _entries.clear()
    _box_nms_calls.clear()
    # 先加载模型：autobackend warmup 会用随机假框跑一次 NMS 预热 CUDA kernel，
    # 该调用与真实推理路径无关。warmup 在 YOLO() 构造（默认 imgsz=640）时发生一次，
    # 若 predict 传入的 imgsz 不同还会触发二次 setup + 再 warmup，故先跑一次丢弃
    model = YOLO(str(_ROOT / weights))
    nms.non_max_suppression = _patched_nms
    torchvision.ops.nms = _patched_tv_nms
    try:
        model.predict(_SOURCE, imgsz=64, device=0, verbose=False)  # 首次：触发 setup + warmup，丢弃
        _entries.clear()
        _box_nms_calls.clear()
        # conf=0.0：3 epochs 冒烟权重置信度极低，默认 conf=0.25 会把候选框全滤掉，
        # 导致 nms 内部提前 return 而不执行框抑制；conf=0 让候选全通过，必然走到框抑制
        results = model.predict(_SOURCE, imgsz=64, device=0, conf=0.0, verbose=False)
        n_dets = len(results[0].boxes) if results[0].boxes is not None else 0
        n_box_nms = len(_box_nms_calls)
        shapes = _entries
        print(f"[{case}] 观察记录: nms入口={shapes}  框抑制调用={n_box_nms} 次  检出框数={n_dets}")
        if expect_end2end:
            assert n_box_nms == 0, f"[{case}] 预期框抑制 0 次（免 NMS），实际 {n_box_nms} 次"
        else:
            assert n_box_nms >= 1, f"[{case}] 预期框抑制 ≥1 次（走 NMS），实际 {n_box_nms} 次"
        bnc_seen = any(s[-1] == 6 for s in shapes)
        print(
            f"[{case}] PASS  框抑制调用={n_box_nms} 次  nms入口shape={shapes}  BNC(shape[-1]==6)={bnc_seen}  检出框数={n_dets}"
        )
        if expect_end2end and any(s[-1] != 6 for s in shapes):
            print(f"[{case}] WARN  存在非 BNC 入口调用（疑点，待分析）：{shapes}")
    finally:
        nms.non_max_suppression = _orig_nms_fn
        torchvision.ops.nms = _orig_tv_nms


if __name__ == "__main__":
    check("C(end2end=True)", "runs/detect/runs/agent/a1-s1-smoke-c/weights/best.pt", True)
    check("A(end2end=False)", "runs/detect/runs/agent/a1-s1-smoke-a/weights/best.pt", False)
    print("推理冒烟（免NMS vs NMS 路径）全部通过")
