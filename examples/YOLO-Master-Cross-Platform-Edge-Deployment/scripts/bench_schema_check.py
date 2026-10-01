#!/usr/bin/env python3
"""Check a yolomaster-bench/v1 JSON document (any producer: Linux CLI, server, macOS CLI, apps).

    python scripts/bench_schema_check.py bench.json [--iters N] [--frames N] [--images N] [--accuracy]

Exit 0 when the document has every required block and key, the floor-rank convention, a 64-hex
image-list hash and (optionally) the expected sample counts. Used by tests/run_tests.sh (T19) and
mac/tests/run_mac_tests.sh so every producer is held to the same shape.
"""
import argparse
import json
import sys

STAGE_KEYS = ("n", "mean", "median", "p90", "p95", "p99", "min", "max")
REQUIRED = {
    "model": ("id", "path", "backend", "runtime", "execution_provider", "precision", "nc", "is_seg", "imgsz"),
    "environment": ("host", "os", "cpu_model", "cpu_count", "mem_bytes", "gpu_name", "threads", "version"),
    "protocol": ("mode", "conf", "iou", "max_det", "multi_label", "slicing", "tile_size", "warmup", "iters",
                 "minutes", "probe", "probe_mode", "dataset", "image_count", "image_list_sha256"),
}


def check(doc, iters=None, frames=None, images=None, accuracy=False):
    errs = []

    def need(d, keys, where):
        for k in keys:
            if k not in d:
                errs.append(f"{where}.{k} missing")

    if doc.get("schema_version") != "yolomaster-bench/v1":
        errs.append("schema_version != yolomaster-bench/v1")
    if doc.get("stats_convention") != "floor_rank":
        errs.append("stats_convention != floor_rank")
    if doc.get("tool") not in ("cli", "server", "android", "ios", "macos"):
        errs.append(f"tool {doc.get('tool')!r} unknown")
    if not doc.get("timestamp"):
        errs.append("timestamp missing")
    for block, keys in REQUIRED.items():
        if block not in doc:
            errs.append(f"{block} missing")
        else:
            need(doc[block], keys, block)
    p = doc.get("protocol", {})
    if len(p.get("image_list_sha256", "")) != 64:
        errs.append("protocol.image_list_sha256 is not 64 hex chars")
    if p.get("probe_mode") not in ("infer_only", "full"):
        errs.append(f"protocol.probe_mode {p.get('probe_mode')!r}")
    if images is not None and p.get("image_count") != images:
        errs.append(f"protocol.image_count {p.get('image_count')} != {images}")
    if "host_meters" in doc:   # optional (macOS app): per-cell thermal / temperature / power summary
        hm = doc["host_meters"]
        need(hm, ("seconds", "thermal_start", "thermal_peak", "power_state"), "host_meters")
        for k in ("die_celsius", "power_w"):
            if k in hm and hm[k] is not None and not isinstance(hm[k], list):
                errs.append(f"host_meters.{k} is not a list")
        if isinstance(hm.get("seconds"), int) and hm.get("die_celsius") and len(hm["die_celsius"]) > hm["seconds"]:
            errs.append("host_meters.die_celsius longer than host_meters.seconds")
    if "moe_export" in doc.get("model", {}) and doc["model"]["moe_export"] not in ("ane_safe_rank", "topk_gather", None):
        errs.append(f"model.moe_export {doc['model']['moe_export']!r} unknown")
    has_cold, has_sus = "cold" in doc, "sustained" in doc
    if not has_cold and not has_sus:
        errs.append("neither cold nor sustained block")
    if has_cold:
        need(doc["cold"].get("infer_ms", {}), STAGE_KEYS, "cold.infer_ms")
        if iters is not None and doc["cold"].get("infer_ms", {}).get("n") != iters:
            errs.append(f"cold.infer_ms.n {doc['cold'].get('infer_ms', {}).get('n')} != {iters}")
    if has_sus:
        s = doc["sustained"]
        need(s, ("infer_ms", "cold_median_ms", "sustained_median_ms", "throttle_pct", "sparkline", "duration_s", "probe_mode"), "sustained")
        need(s.get("infer_ms", {}), STAGE_KEYS, "sustained.infer_ms")
    if "dataset" in doc:
        d = doc["dataset"]
        need(d, ("frames", "total_dets", "pre_ms", "infer_ms", "post_ms", "total_ms", "model_fps", "wall_s"), "dataset")
        for k in ("pre_ms", "infer_ms", "post_ms", "total_ms"):
            need(d.get(k, {}), STAGE_KEYS, f"dataset.{k}")
        if frames is not None and d.get("frames") != frames:
            errs.append(f"dataset.frames {d.get('frames')} != {frames}")
    elif frames is not None:
        errs.append("dataset block missing")
    if accuracy:
        if "accuracy" not in doc:
            errs.append("accuracy block missing")
        else:
            a = doc["accuracy"]
            need(a, ("protocol", "labels", "images", "map50", "map5095", "per_class", "timings"), "accuracy")
            need(a.get("protocol", {}), ("conf", "iou", "max_det", "multi_label"), "accuracy.protocol")
            for c in a.get("per_class", []):
                need(c, ("class_id", "n_gt", "n_pred", "ap50", "ap5095"), "accuracy.per_class[]")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--images", type=int)
    ap.add_argument("--accuracy", action="store_true", help="require the accuracy block")
    a = ap.parse_args()
    with open(a.json) as f:
        doc = json.load(f)
    errs = check(doc, a.iters, a.frames, a.images, a.accuracy)
    for e in errs:
        print("  " + e, file=sys.stderr)
    if errs:
        print(f"{a.json}: {len(errs)} schema problem(s)", file=sys.stderr)
        return 1
    print(f"{a.json}: yolomaster-bench/v1 OK (tool={doc['tool']}, ep={doc['model']['execution_provider']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
