#!/usr/bin/env python3
"""Where does Core ML run each op of an .mlpackage? (macOS 14.4+ and coremltools 8+ only.)

    python coreml_export/compute_plan.py model.mlpackage [--units ane|gpu|all|cpu] [--top 30]

Prints, for the chosen compute-unit setting, the estimated-cost share and op count per device the
plan assigns, then the ops the Neural Engine cannot take (type, name, cost share), largest first.
This is the Xcode performance report as text: the tool that tells whether an "ANE" number measured
by the Bench is the Neural Engine or a CPU fallback with sync boundaries.
"""
import argparse, collections, sys

import coremltools as ct

try:   # the native bridge: absent when pip could only install the pure-Python part (no wheel for this Python)
    from coremltools import libcoremlpython  # noqa: F401
except ImportError:
    sys.exit(f"coremltools {ct.__version__} on Python {sys.version.split()[0]} has no native bridge "
             "(libcoremlpython): pip installed the pure-Python package only, so nothing can be compiled or "
             "planned. Use a Python that coremltools ships binaries for (3.11 or 3.12), e.g.\n"
             "  python3.12 -m venv ~/ctplan && ~/ctplan/bin/pip install -U coremltools\n"
             "  ~/ctplan/bin/python coreml_export/compute_plan.py model.mlpackage --units ane")
from coremltools.models.compute_plan import MLComputePlan

UNITS = {"ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU,
         "all": ct.ComputeUnit.ALL, "cpu": ct.ComputeUnit.CPU_ONLY}


def device_name(dev) -> str:
    n = type(dev).__name__
    return {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU", "MLCPUComputeDevice": "CPU"}.get(n, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--units", default="ane", choices=sorted(UNITS))
    ap.add_argument("--top", type=int, default=30, help="how many unsupported ops to list")
    a = ap.parse_args()

    compiled = ct.utils.compile_model(a.model) if not a.model.endswith(".mlmodelc") else a.model
    plan = MLComputePlan.load_from_path(path=compiled, compute_units=UNITS[a.units])
    program = plan.model_structure.program
    if program is None:
        print("not an ML program; nothing to report"); return 1
    ops = list(program.functions["main"].block.operations)

    pref_cost, pref_n, cap_n = collections.Counter(), collections.Counter(), collections.Counter()
    off_ane, demoted = [], []
    total, n_real = 0.0, 0
    for op in ops:
        if op.operator_name == "const" or "constexpr" in op.operator_name:   # weights and their dequantizers
            continue
        n_real += 1
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        w = float(cost.weight) if cost is not None and cost.weight is not None else 0.0
        total += w
        dev = device_name(usage.preferred_compute_device) if usage is not None else "?"
        pref_cost[dev] += w; pref_n[dev] += 1
        supported = {device_name(d) for d in (usage.supported_compute_devices if usage is not None else [])}
        for d in supported: cap_n[d] += 1
        name = op.outputs[0].name if op.outputs else "?"
        if "ANE" not in supported:
            off_ane.append((w, op.operator_name, name, ",".join(sorted(supported)) or "?"))
        elif dev != "ANE":
            demoted.append((w, op.operator_name, name, dev))

    print(f"{a.model}  units={a.units}  ops={n_real} (weights and dequantizers excluded)")
    if total == 0:
        print("  (Core ML gave no cost estimates for this plan; shares are by op count)")
    print("  scheduled on (preferred device):")
    for dev in sorted(pref_n, key=lambda d: -(pref_cost[d] if total else pref_n[d])):
        share = 100 * (pref_cost[dev] / total if total else pref_n[dev] / n_real)
        print(f"    {dev:>4}: {pref_n[dev]:5d} ops  {share:5.1f}%")
    print("  capable of (supported devices): " + ", ".join(f"{d}:{n}" for d, n in cap_n.most_common()))
    off_ane.sort(reverse=True)
    print(f"ops the ANE cannot run: {len(off_ane)}")
    kinds = collections.Counter(t for _, t, _, _ in off_ane)
    if kinds:
        print("  by type: " + ", ".join(f"{t}:{n}" for t, n in kinds.most_common()))
    for w, t, name, sup in off_ane[: a.top]:
        print(f"  {100 * w / total if total else 0:5.2f}%  {t:<24} {name:<44} runs on {sup}")
    if demoted:
        kinds = collections.Counter(t for _, t, _, _ in demoted)
        print(f"ANE-capable ops that Core ML scheduled elsewhere anyway: {len(demoted)} "
              f"(by type: {', '.join(f'{t}:{n}' for t, n in kinds.most_common(12))})")
        print("  every op says it can run on the ANE, yet the plan puts the segment on another unit: Core ML rejected")
        print("  the ANE partition as a whole, typically a tensor beyond the ANE's shape limits or an op that fails")
        print("  at ANE compile time. Compare with --units all and with the fp16 package to narrow it down.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
