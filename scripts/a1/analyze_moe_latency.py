"""A1 MoE latency 归因分析:B 格 forward 的 20ms 去哪了。

三个消融口径(B 格 MoE vs A 格 dense 同位置模块):
1. 逐模块计时:hook 记录每个骨干模块的 GPU 耗时 → MoE 模块(A2C2fMoE)总占比
2. 同步检查消融:monkeypatch torch.isfinite 为恒 True → 测 isfinite().all() 同步的总开销
3. 同位置对比:A 格相同层位(C3k2)模块耗时 vs B 格 MoE 模块耗时

用法:
    python scripts/a1/analyze_moe_latency.py \
        --model-b <B格best.pt> [--model-a <A格best.pt>] [--device 0]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # YOLO-Master 仓库根
sys.path.insert(0, str(REPO_ROOT))

os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("YOLO_VERBOSE", "false")

import torch

from ultralytics import YOLO

N_RUNS = 100


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-b", required=True, help="B 格(MoE)best.pt")
    ap.add_argument("--model-a", default=None, help="A 格(dense)best.pt,可选对照")
    ap.add_argument("--device", default="0")
    return ap.parse_args()


def timed_forward(model: torch.nn.Module, x: torch.Tensor, n: int = N_RUNS) -> tuple[float, float]:
    """计时 forward,返回 (mean_ms, std_ms)。"""
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(x)
        times = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    import statistics

    return statistics.fmean(times), statistics.stdev(times)


def module_breakdown(model: torch.nn.Module, x: torch.Tensor) -> dict[str, float]:
    """hook 记录各骨干模块 GPU 耗时(ms/forward,取中位数)。"""
    from ultralytics.nn.modules.moe.modules import A2C2fMoE

    # 挂 hook 的模块:骨干里的 A2C2fMoE(第 4/6 层)与对照 C3k2
    mods = {}
    for name, m in model.named_modules():
        if isinstance(m, A2C2fMoE):
            mods[name] = m
    acc: dict[str, list[float]] = {name: [] for name in mods}

    handles, marks = [], {}
    for name, m in mods.items():

        def make_pre(n):
            def pre(_, __):
                torch.cuda.synchronize()
                marks[n] = time.perf_counter()

            return pre

        def make_post(n):
            def post(_, __, ___):
                torch.cuda.synchronize()
                acc[n].append((time.perf_counter() - marks[n]) * 1e3)

            return post

        handles.append(m.register_forward_pre_hook(make_pre(name)))
        handles.append(m.register_forward_hook(make_post(name)))

    with torch.no_grad():
        for _ in range(30):
            model(x)
    for h in handles:
        h.remove()

    out = {}
    for name, ts in acc.items():
        if not ts:
            print(f"  [跳过] {name}: hook 未触发(30 次 forward 内未执行)")
            continue
        ts_sorted = sorted(ts)
        out[name] = round(ts_sorted[len(ts_sorted) // 2], 4)  # 中位数
    return out


def submodule_breakdown(model: torch.nn.Module, x: torch.Tensor, layer_name: str) -> dict[str, float]:
    """对指定 MoE 层内部拆解:attn / routing / shared_expert / 各专家 的 GPU 耗时。"""
    layer = dict(model.named_modules()).get(layer_name)
    if layer is None:
        return {}
    mods = {}
    for name, m in layer.named_modules():
        if (
            name.endswith(("attn", "routing", "mlp"))
            or "shared_expert" in name
            or ("experts" in name and "." in name)  # experts.<i> 叶子模块
        ):
            mods[name] = m
    acc: dict[str, list[float]] = {name: [] for name in mods}
    marks = {}
    handles = []
    for name, m in mods.items():

        def make_pre(n):
            def pre(_, __):
                torch.cuda.synchronize()
                marks[n] = time.perf_counter()

            return pre

        def make_post(n):
            def post(_, __, ___):
                torch.cuda.synchronize()
                acc[n].append((time.perf_counter() - marks[n]) * 1e3)

            return post

        handles.append(m.register_forward_pre_hook(make_pre(name)))
        handles.append(m.register_forward_hook(make_post(name)))
    with torch.no_grad():
        for _ in range(30):
            model(x)
    for h in handles:
        h.remove()
    out = {}
    for name, ts in acc.items():
        if not ts:
            print(f"  [跳过] {name}: hook 未触发(30 次 forward 内未执行)")
            continue
        ts_sorted = sorted(ts)
        out[name] = round(ts_sorted[len(ts_sorted) // 2], 4)
    # 专家合并为一个"专家计算合计"
    expert_total = sum(t for n, t in out.items() if n.startswith("experts."))
    return {
        "attn": out.get("attn", 0),
        "routing": out.get("routing", 0),
        "shared_expert": out.get("shared_expert", 0),
        "experts_total": round(expert_total, 4),
    }


def main() -> None:
    """执行三口径消融并打印归因表。"""
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    import torchvision  # noqa: F401  统一 NMS 快路径环境

    b = YOLO(args.model_b)
    model_b = b.model
    model_b.fuse()
    model_b.eval().to(device)
    x = torch.randn(1, 3, 640, 640, device=device)

    print("== B 格(MoE)forward ==")
    mean_b, std_b = timed_forward(model_b, x)
    print(f"基线: {mean_b:.2f} ± {std_b:.2f} ms")

    # 口径 2:同步检查消融(monkeypatch torch.isfinite 恒 True)
    orig_isfinite = torch.isfinite

    def fake_isfinite(t):
        return torch.ones(1, dtype=torch.bool, device=t.device)

    torch.isfinite = fake_isfinite
    mean_no_sync, std_no_sync = timed_forward(model_b, x)
    torch.isfinite = orig_isfinite
    print(
        f"关闭 isfinite().all() 同步: {mean_no_sync:.2f} ± {std_no_sync:.2f} ms → 同步检查开销 ≈ {mean_b - mean_no_sync:.2f} ms"
    )

    # 口径 1:逐模块计时
    breakdown = module_breakdown(model_b, x)
    moe_total = sum(breakdown.values())
    print("\n逐模块(GPU 计时,ms/forward):")
    for name, t in sorted(breakdown.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<30} {t:>8.2f} ms")
    print(f"  MoE 模块合计: {moe_total:.2f} ms(占 forward {moe_total / mean_b * 100:.0f}%)")

    # 口径 1b:最贵 MoE 层(model.4)内部拆解:attn / 路由 / 共享专家 / 专家计算
    sub = submodule_breakdown(model_b, x, "model.4")
    print("\nmodel.4 内部拆解(GPU 计时,ms/forward):")
    layer_total = breakdown.get("model.4", 0.0)
    sub_sum = sum(sub.values())
    for name, t in sorted(sub.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<35} {t:>8.2f} ms")
    print(
        f"  子模块合计 {sub_sum:.2f} ms | 层总 {layer_total:.2f} ms | 其余(聚合/gather-scatter 等)≈ {layer_total - sub_sum:.2f} ms"
    )

    # 口径 3:A 格同位置对照
    if args.model_a:
        a = YOLO(args.model_a)
        model_a = a.model
        model_a.fuse()
        model_a.eval().to(device)
        mean_a, std_a = timed_forward(model_a, x)
        print("\n== A 格(dense)对照 ==")
        print(f"forward: {mean_a:.2f} ± {std_a:.2f} ms")
        print(f"B - A 差 = {mean_b - mean_a:.2f} ms,其中 MoE 模块合计 {moe_total:.2f} ms")


if __name__ == "__main__":
    main()
