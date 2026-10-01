#!/usr/bin/env python3
"""Deterministic COCO val2017 subset for the on-device accuracy bench (release asset, never committed).

Takes every N-th image of the sorted val2017 file list (default every 10th -> 500 images, the 48
label-less images included with zero ground truth, exactly as the full-val evaluators treat them),
copies the images and their YOLO labels, writes a dataset yaml with the 80 COCO names and an
image_list_sha256.txt whose LIST line is the sha256 of the sorted basenames joined by '\\n' (the
string bench::image_list_sha256 hashes, so a bench JSON can be tied to this exact list).

  python scripts/make_coco_subset.py --src /data/datasets/coco --out datasets/coco500
"""
import argparse, hashlib, shutil
from pathlib import Path

COCO80 = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
    "bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
    "vase","scissors","teddy bear","hair drier","toothbrush"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/data/datasets/coco", help="COCO root with <split>/ images and labels/<split>/")
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", default="datasets/coco500")
    ap.add_argument("--name", default=None, help="dataset name (default: out dir name)")
    a = ap.parse_args()
    src = Path(a.src)
    img_dir = src / a.split if (src / a.split).is_dir() else src / "images" / a.split
    lab_dir = src / "labels" / a.split
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".jpg")
    pick = imgs[::a.every]
    out = Path(a.out); name = a.name or out.name
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    n_lab = 0
    for p in pick:
        shutil.copyfile(p, out / "images" / p.name)
        l = lab_dir / (p.stem + ".txt")
        if l.exists():
            shutil.copyfile(l, out / "labels" / l.name); n_lab += 1
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO80))
    (out / f"{name}.yaml").write_text(
        f"# {len(pick)} images = every {a.every}th of sorted COCO {a.split} ({n_lab} with labels)\n"
        f"path: .\ntrain: images\nval: images\nnames:\n{names}\n")
    basenames = sorted(p.name for p in pick)
    lines = [f"{hashlib.sha256((out / 'images' / b).read_bytes()).hexdigest()}  images/{b}" for b in basenames]
    lines.append("LIST " + hashlib.sha256("\n".join(basenames).encode()).hexdigest())
    (out / "image_list_sha256.txt").write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(pick)} images ({n_lab} labelled), list sha256 {lines[-1].split()[1][:16]}...")


if __name__ == "__main__":
    main()
