#!/usr/bin/env python3
"""Numeric parity of two prediction-dump directories ('class conf x1 y1 x2 y2' per line).

    python scripts/server/parity_txt.py cli_txt api_txt [--tol 0.01] [--conf-tol 1e-4]

Detections are matched per image by class and IoU (greedy, best IoU first) rather than by line
order, because the CLI prints 6 significant digits and the API full floats, so near-tie rows can
legitimately swap order. Reports unmatched detections, box coordinate deltas (px) and confidence
deltas; parity=OK when every detection matches within the tolerances. Exit 1 otherwise.
"""
import sys, argparse
from pathlib import Path
import numpy as np


def load(p):
    rows = [l.split() for l in Path(p).read_text().splitlines()]
    rows = [r for r in rows if len(r) == 6]
    if not rows:
        return np.zeros((0, 6), np.float64)
    return np.array(rows, dtype=np.float64)


def iou_matrix(a, b):
    ax1, ay1, ax2, ay2 = a[:, 2:3], a[:, 3:4], a[:, 4:5], a[:, 5:6]
    bx1, by1, bx2, by2 = b[:, 2], b[:, 3], b[:, 4], b[:, 5]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.maximum(ua, 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=0.01); ap.add_argument("--conf-tol", type=float, default=1e-4)
    x = ap.parse_args()
    fa = {p.name: p for p in Path(x.a).glob("*.txt")}; fb = {p.name: p for p in Path(x.b).glob("*.txt")}
    common = sorted(set(fa) & set(fb)); missing = sorted((set(fa) | set(fb)) - set(common))
    dets = matched = unmatched = bad_box = bad_conf = count_mismatch = 0
    max_box = max_conf = 0.0
    for n in common:
        A, B = load(fa[n]), load(fb[n])
        dets += len(A)
        if len(A) != len(B):
            count_mismatch += 1
        if len(A) == 0 or len(B) == 0:
            unmatched += len(A) + len(B); continue
        M = iou_matrix(A, B)
        M[A[:, 0:1] != B[:, 0][None, :]] = -1          # class must agree
        used_b = np.zeros(len(B), bool)
        order = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
        used_a = np.zeros(len(A), bool)
        for i, j in order:
            if M[i, j] < 0.5:
                break
            if used_a[i] or used_b[j]:
                continue
            used_a[i] = used_b[j] = True; matched += 1
            d = float(np.abs(A[i, 2:6] - B[j, 2:6]).max()); c = float(abs(A[i, 1] - B[j, 1]))
            max_box = max(max_box, d); max_conf = max(max_conf, c)
            if d > x.tol: bad_box += 1
            if c > x.conf_tol: bad_conf += 1
        unmatched += int((~used_a).sum()) + int((~used_b).sum())
    ok = not missing and unmatched == 0 and bad_box == 0 and bad_conf == 0
    print(f"files={len(common)} missing={len(missing)} dets={dets} matched={matched} unmatched={unmatched} "
          f"count_mismatch_files={count_mismatch} box_over_tol={bad_box} conf_over_tol={bad_conf} "
          f"max_box_delta={max_box:.5f} max_conf_delta={max_conf:.6f} parity={'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
