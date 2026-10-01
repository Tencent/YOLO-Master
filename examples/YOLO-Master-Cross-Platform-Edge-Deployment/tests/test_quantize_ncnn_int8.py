#!/usr/bin/env python3
"""Oracle tests for the structural INT8 exclusion builder in ``scripts/quantize_ncnn_int8.py``.

Plain python (no pytest):  ``python tests/test_quantize_ncnn_int8.py``  from the repo root.
Every oracle is pinned to a real ``.param`` in the repo; an oracle whose file is absent is
SKIPPED, a violated one FAILS with the offending assertion. Exit code 1 on any failure.
No ncnn tool or wheel is touched: this only parses text and runs the ``--dry-run`` CLI path.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import quantize_ncnn_int8 as q  # noqa: E402

MOA = REPO / "models/mixture/moa-n_ncnn/model.ncnn.param"
MOLORA = REPO / "models/mixture/molora-merged_ncnn/model.ncnn.param"
P03 = REPO / "tempo-ncnn/models/p03_v01n.ncnn.param"
P03_TABLE = REPO / "tempo-ncnn/models/p03_v01n.table"
P03_TABLE_S = REPO / "tempo-ncnn/models/p03_v01n_sensitive.table"
ESMOE = REPO / "models/esmoe_n_visdrone_ncnn/model.ncnn.param"
SEG = REPO / "models/v0.1-seg-n_ncnn/model.ncnn.param"

ROUTER_ORACLE = [
    (
        "linear_361 fcsigmoid_0 convdw_1067 conv_92 convsigmoid_0 convdw_1068 conv_94 convsigmoid_1 "
        "linear_363 convdw_1069 conv_96 conv_97"
    ).split(),
    (
        "linear_364 fcsigmoid_1 convdw_1077 conv_124 convsigmoid_4 convdw_1078 conv_126 convsigmoid_5 "
        "linear_366 convdw_1079 conv_128 conv_129"
    ).split(),
    (
        "linear_367 fcsigmoid_2 convdw_1087 conv_156 convsigmoid_8 convdw_1088 conv_158 convsigmoid_9 "
        "linear_369 convdw_1089 conv_160 conv_161"
    ).split(),
]
P03_EXCLUDED = {"conv_16", "conv_17", "conv_32", "conv_58", "conv_84", "conv_146"}
ESMOE_HEAD = {"conv_131", "conv_134", "conv_137", "conv_140", "conv_143", "conv_146"}

RESULTS = {"pass": 0, "fail": 0, "skip": 0}


def check(cond, msg):
    """Record one assertion."""
    RESULTS["pass" if cond else "fail"] += 1
    print(("  ok   " if cond else "  FAIL ") + msg)


def skip(msg):
    """Record one skipped oracle."""
    RESULTS["skip"] += 1
    print("  skip " + msg)


def test_mixture(param: Path):
    print(f"[{param.parent.name}]")
    if not param.exists():
        return skip(f"{param} absent")
    text = param.read_text()
    check(q.is_router_emulated(text), "router-emulated fingerprint (amax_ AND 1.000000e30)")
    ex = q.build_exclusions(text)
    check(ex["quantizable_total"] == 265, f"265 quantizable (got {ex['quantizable_total']})")
    check(ex["stem"] == ["conv_81", "conv_82"], f"stem pair conv_81, conv_82 (got {ex['stem']})")
    check(ex["dfl"] == ["conv_292"], f"DFL conv_292 (got {ex['dfl']})")
    check(len(ex["router"]) == 3, f"three router windows (got {len(ex['router'])})")
    for i, want in enumerate(ROUTER_ORACLE):
        got = ex["router"][i] if i < len(ex["router"]) else None
        check(got == want, f"router{i} window == 12-name oracle in file order (got {got})")
    check(
        ex["logit_heads"] == [] and ex["sk_gates"] == [],
        f"no extra logit/sk exclusions ({ex['logit_heads']}, {ex['sk_gates']})",
    )
    check(len(ex["excluded"]) == 39, f"36 router + stem 2 + dfl 1 == 39 excluded (got {len(ex['excluded'])})")
    g = q.parse_param(text)
    for r in ex["routers"]:
        # window body = [lower .. sink]; Split names repeat in this export, so pick the entry
        # Split as the closest one preceding the window's first quantizable layer
        first_q = g.by_name(r["window"][0]).index
        lo = max(l.index for l in g.layers if l.type == "Split" and l.name == r["lower"] and l.index < first_q)
        hi = min(l.index for l in g.layers if l.type == "BinaryOp" and l.name == r["sink"] and l.index > first_q)
        body = g.layers[lo : hi + 1]
        softmax = sum(l.type == "Softmax" for l in body)
        mem = [l for l in body if l.type == "MemoryData" and set(l.kv) == {"0"} and int(l.kv["0"]) > 1]
        consts = sum(any(v in q.ROUTER_CONSTANTS for v in l.kv.values()) for l in body)
        check(
            softmax == 1 and len(mem) == 1 and consts == 3,
            f"{r['lower']}..{r['sink']}: softmax={softmax} memorydata(E={r['experts']})={len(mem)} constants={consts}",
        )
        check(r["experts"] > 1, f"E={r['experts']} > 1")
        check(
            len(r["cone"]) == 10 and set(r["cone"]) <= set(r["window"]),
            f"cone scope = 10 sink ancestors within the window (got {len(r['cone'])})",
        )
    cone = q.build_exclusions(text, router_scope="cone")
    check(
        all(len(r) == 10 for r in cone["router"]) and len(cone["excluded"]) == 33,
        f"--router-scope cone excludes 3x10 + 3 (got {len(cone['excluded'])})",
    )


def test_p03():
    print("[p03_v01n]")
    if not P03.exists():
        return skip(f"{P03} absent")
    text = P03.read_text()
    check(not q.is_router_emulated(text), "dense fingerprint")
    ex = q.build_exclusions(text)
    check(ex["stem"] == ["conv_16", "conv_17"], f"stem pair conv_16, conv_17 (got {ex['stem']})")
    check(ex["dfl"] == ["conv_146"], f"DFL conv_146 (got {ex['dfl']})")
    check(ex["logit_heads"] == ["conv_32", "conv_58", "conv_84"], f"image-level logit heads (got {ex['logit_heads']})")
    check(ex["sk_gates"] == [] and ex["router"] == [], "no sk gates / routers on p03")
    check(
        set(ex["excluded"]) == P03_EXCLUDED and len(ex["excluded"]) == 6,
        f"default exclusions == {sorted(P03_EXCLUDED)} (got {ex['excluded']})",
    )
    if P03_TABLE.exists() and P03_TABLE_S.exists():
        keys = lambda p: {ln.split()[0] for ln in p.read_text().splitlines() if ln.strip()}  # noqa: E731
        base = lambda k: k[: -len("_param_0")] if k.endswith("_param_0") else k  # noqa: E731
        diff = keys(P03_TABLE) - keys(P03_TABLE_S)
        check({base(k) for k in diff} == P03_EXCLUDED, f"table diff oracle: {sorted(diff)}")
        both_kinds = all(any(k == n or k == n + "_param_0" for n in P03_EXCLUDED) for k in diff)
        check(both_kinds and len(diff) == 12, "table diff is exactly both row kinds of the six names")
        kept, dropped, warn = q.strip_table(P03_TABLE.read_text(), ex["excluded"])
        check(sorted(dropped) == sorted(diff), "strip_table on the real table reproduces the prior-art table diff")
        kept_keys = {ln.split()[0] for ln in kept.splitlines() if ln.strip()}
        check(keys(P03_TABLE_S) == kept_keys, "strip_table kept rows == p03_v01n_sensitive.table rows")
        check(warn == [], f"no warnings on the real table (got {warn})")
    else:
        skip("p03 table pair absent")


def test_esmoe():
    print("[esmoe_n_visdrone]")
    if not ESMOE.exists():
        return skip(f"{ESMOE} absent")
    text = ESMOE.read_text()
    check(not q.is_router_emulated(text), "dense fingerprint")
    ex = q.build_exclusions(text)
    check(ex["stem"] == ["conv_22", "conv_23"], f"stem pair (got {ex['stem']})")
    check(ex["dfl"] == ["conv_147"], f"DFL conv_147 (got {ex['dfl']})")
    check(ex["head"] == [], "head not excluded by default")
    exh = q.build_exclusions(text, exclude_head=True)
    check(set(exh["head"]) == ESMOE_HEAD, f"--exclude-head == {sorted(ESMOE_HEAD)} (got {exh['head']})")
    check(len(ex["sk_gates"]) == 8, f"8 sk-gate names excluded by default (got {ex['sk_gates']})")
    g = q.parse_param(text)
    check(all(g.by_name(n).out_channels <= 32 for n in ex["sk_gates"]), "sk-gate convs are tiny (<= 32 out-ch)")
    kept = q.build_exclusions(text, keep_sk_gates=True)
    check(
        kept["sk_gates"] == [] and kept["sk_gates_kept"] == ex["sk_gates"],
        "--keep-sk-gates moves them to sk_gates_kept",
    )
    check(ex["logit_heads"] == [] and ex["router"] == [], "no logit heads / routers on esmoe")
    check(
        set(ex["excluded"]) == set(ex["stem"] + ex["dfl"] + ex["sk_gates"]), "default excluded = stem + dfl + sk gates"
    )


def test_seg():
    print("[v0.1-seg-n]")
    if not SEG.exists():
        return skip(f"{SEG} absent")
    text = SEG.read_text()
    ex = q.build_exclusions(text, exclude_head=True)
    check(ex["stem"] == ["conv_22", "conv_23"], f"stem pair (got {ex['stem']})")
    check(ex["dfl"] == ["conv_159"], f"DFL conv_159 (got {ex['dfl']})")
    check(len(ex["head"]) == 10, f"--exclude-head == 10 names (got {len(ex['head'])}: {ex['head']})")
    g = q.parse_param(text)
    ch = [g.by_name(n).out_channels for n in ex["head"]]
    check(ch.count(32) == 4, f"four 32-channel head convs (mask-coef + proto tail); out-ch = {ch}")
    check(len(ex["sk_gates"]) == 8, f"8 sk-gate names (got {ex['sk_gates']})")


def test_strip_table_synthetic():
    print("[strip_table synthetic]")
    table = "conv_1_param_0 1.0 2.0\nconv_2_param_0 3.0\nconv_3_param_0 4.0\nconv_1 0.5\nconv_2 0.6\nconv_3 0.7\n"
    kept, dropped, warn = q.strip_table(table, ["conv_2", "ghost"])
    check(dropped == ["conv_2_param_0", "conv_2"], f"both row kinds dropped (got {dropped})")
    want = ["conv_1_param_0 1.0 2.0", "conv_3_param_0 4.0", "conv_1 0.5", "conv_3 0.7"]
    check(kept.splitlines() == want, f"kept rows intact/in order (got {kept.splitlines()})")
    check(warn == ["ghost"], f"warning for an unseen excluded name (got {warn})")
    check(kept.endswith("\n"), "kept text ends with newline")


def test_fingerprints_and_parse():
    print("[fingerprints]")
    try:
        q.is_router_emulated("7767517\n2 2\nInput in0 0 1 in0\nReduction amax_0 1 1 in0 1 0=4\n")
        check(False, "amax_ without 1.000000e30 must raise")
    except ValueError:
        check(True, "amax_ without 1.000000e30 raises")
    try:
        q.is_router_emulated("7767517\n2 2\nInput in0 0 1 in0\nBinaryOp mul_0 1 1 in0 1 2=1.000000e30\n")
        check(False, "1.000000e30 without amax_ must raise")
    except ValueError:
        check(True, "1.000000e30 without amax_ raises")
    g = q.parse_param(
        "7767517\n3 3\nInput in0 0 1 in0\nConvolution conv_0 1 1 in0 1 0=1 1=1 5=0 6=16\n"
        "InnerProduct fc 1 1 1 2 0=4 2=64\n"
    )
    check(q.dfl_layers(g) == ["conv_0"], "DFL fingerprint 0=1 1=1 5=0 6=16")
    check(q.stem_pair(g) == ["conv_0", "fc"], "stem pair = first two quantizable, first is a Convolution on in0")
    check(g.descendants(1) == {2} and g.ancestors(2) == {1}, "descendants/ancestors over quantizable layers")
    if MOA.exists():
        gm = q.parse_param(MOA.read_text())
        names = [l.name for l in gm.layers]
        check(len(names) != len(set(names)), "moa-n layer names are NOT unique (graph must be index-keyed)")
        qn = [l.name for l in gm.quantizable]
        check(len(qn) == len(set(qn)), "moa-n quantizable names ARE unique (table keys are safe)")


def test_dry_run_cli():
    print("[--dry-run p03]")
    binf = P03.with_suffix(".bin")
    if not P03.exists() or not binf.exists():
        return skip("p03 param/bin absent")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "p03-int8_ncnn"
        script = str(REPO / "scripts/quantize_ncnn_int8.py")
        cmd = [sys.executable, script, "--param", str(P03), "--bin", str(binf), "--dry-run", "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        check(r.returncode == 0, f"dry-run exits 0 (stderr: {r.stderr.strip()[-300:]})")
        m = out / "quant_manifest.json"
        check(m.exists(), "quant_manifest.json written")
        if m.exists():
            d = json.loads(m.read_text())
            check(d["dry_run"] is True and set(d["excluded"]) == P03_EXCLUDED, "manifest carries the p03 exclusion set")
            cov_ok = d["quantizable"] == 145 and abs(d["layer_coverage"] - (145 - 6) / 145) < 1e-9
            check(cov_ok, f"coverage {d['layer_coverage']:.4f}")
            check(d["calib"]["skipped"] is True, "calibration skipped without --calib-dir")
        check("[coverage]" in r.stdout and "dense" in r.stdout, "prints classification and coverage")
        check(not (out / "model.ncnn.param").exists(), "no int8 param produced in dry-run")


def main():
    tests = (
        test_fingerprints_and_parse, test_strip_table_synthetic, test_p03, test_esmoe, test_seg, test_dry_run_cli
    )
    for t in tests:
        t()
    test_mixture(MOA)
    test_mixture(MOLORA)
    print(f"\n{RESULTS['pass']} passed, {RESULTS['fail']} failed, {RESULTS['skip']} skipped")
    return 1 if RESULTS["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
