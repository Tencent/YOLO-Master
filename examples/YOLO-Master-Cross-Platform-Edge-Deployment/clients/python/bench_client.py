#!/usr/bin/env python3
"""Closed-loop API benchmark: streams every image of a directory through /v1/infer.

    python bench_client.py --url http://localhost:8080 --model v01n-trt --images /data/datasets/coco/val2017 \
        --concurrency 1 --dump-txt out/api_v01n_trt --summary out/api_v01n_trt.json [--limit N] [--warmup 20]

Measures client-observed wall latency per request (p50/p95/p99), throughput (img/s), and the
server-reported stage timings (queue/decode/pre/infer/post) taken from the JSON response.
--dump-txt writes the CLI's 'class conf x1 y1 x2 y2' txt per image (from return=json boxes, same
numbers the txt endpoint prints) so scripts/eval_map.py can score API and bare CLI runs alike.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request, urllib.error

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20, help="requests excluded from the statistics")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--max-det", type=int, default=None)
    ap.add_argument("--dump-txt", default="")
    ap.add_argument("--summary", default="")
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()

    imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in EXTS)
    if a.limit:
        imgs = imgs[:a.limit]
    if not imgs:
        sys.exit("no images")
    q = {"model": a.model, "return": "json", "names": "0"}
    if a.conf is not None: q["conf"] = a.conf
    if a.iou is not None: q["iou"] = a.iou
    if a.max_det is not None: q["max_det"] = a.max_det
    url = a.url.rstrip("/") + "/v1/infer?" + "&".join(f"{k}={v}" for k, v in q.items())
    if a.dump_txt:
        os.makedirs(a.dump_txt, exist_ok=True)

    lock = threading.Lock()
    rows = []          # (idx, wall_ms, timings dict, count, status)
    errors = []

    def one(i, p):
        data = p.read_bytes()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "image/jpeg")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                body = r.read()
            wall = (time.perf_counter() - t0) * 1000
            j = json.loads(body)
            if a.dump_txt:
                with open(Path(a.dump_txt) / (p.stem + ".txt"), "w") as f:
                    for d in j["detections"]:
                        b = d["box"]
                        f.write(f"{d['class_id']} {d['conf']} {b[0]} {b[1]} {b[2]} {b[3]}\n")
            with lock:
                rows.append((i, wall, j["timings_ms"], j["count"], 200))
        except urllib.error.HTTPError as e:
            wall = (time.perf_counter() - t0) * 1000
            with lock:
                errors.append((p.name, e.code, e.read()[:200].decode(errors="replace")))
                rows.append((i, wall, None, 0, e.code))
        except Exception as e:  # noqa
            with lock:
                errors.append((p.name, 0, str(e)))

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(lambda ip: one(*ip), enumerate(imgs)))
    wall_total = time.perf_counter() - t_start

    rows.sort()
    ok = [r for r in rows if r[4] == 200]
    meas = [r for r in ok if r[0] >= a.warmup]
    walls = [r[1] for r in meas]
    stages = {}
    for k in ("queue", "decode", "pre", "infer", "post", "total"):
        v = [r[2][k] for r in meas]
        stages[k] = {"mean": statistics.fmean(v) if v else 0, "p50": pct(v, .5), "p95": pct(v, .95), "p99": pct(v, .99)}
    summary = {
        "url": url, "model": a.model, "images": len(imgs), "ok": len(ok), "errors": len(errors), "concurrency": a.concurrency,
        "warmup_excluded": a.warmup, "measured": len(meas),
        "wall_s": wall_total, "throughput_img_s": len(ok) / wall_total if wall_total else 0,
        "client_ms": {"mean": statistics.fmean(walls) if walls else 0, "p50": pct(walls, .5), "p95": pct(walls, .95), "p99": pct(walls, .99),
                      "min": min(walls) if walls else 0, "max": max(walls) if walls else 0},
        "server_ms": stages,
        "api_overhead_ms_p50": pct(walls, .5) - stages["total"]["p50"] - stages["queue"]["p50"],
        "total_dets": sum(r[3] for r in ok),
        "first_errors": errors[:5],
    }
    print(json.dumps(summary, indent=2))
    if a.summary:
        Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(a.summary).write_text(json.dumps(summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
