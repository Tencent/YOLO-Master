#!/usr/bin/env python3
"""Segment an exported YOLO-Master ONNX into alternating fp16 / fp32 runs for MNN's multi-path session.

MNN applies BackendConfig::precision per ScheduleConfig, not per op, and runs the configs of a
createMultiPathSession() one after another. So a graph whose MoE routing must stay fp32 while the
rest runs fp16 on CUDA is expressed as an ordered list of Tensor-mode paths: every maximal run of
routing nodes ("/routing/" in the node name, the same rule scripts/server/prepare_models.py uses
for the fp16 ONNX) becomes an fp32 segment, everything between becomes fp16. The runs follow the
graph's node order, which torch.onnx.export emits topologically sorted, so each segment's inputs
are produced by earlier segments or are graph inputs.

Output: <onnx>.paths.json next to the ONNX (or --out), consumed by MnnBackend when the MNN model
carries a sibling "<model>.paths.json" (deploy/pod/prepare_models_pod.sh copies it):

  {"schema": "mnn-paths/v1", "source": "model.onnx", "inputs": [...], "outputs": [...],
   "segments": [{"precision": "fp16"|"fp32", "nodes": N, "inputs": [tensor...], "outputs": [tensor...]}, ...]}

Tensor names are the ONNX names; MNNConvert keeps them (checked with --verify against the .mnn).
"""
import argparse, json, sys
from pathlib import Path

import onnx


def grouped_topological(nodes, available, marker):
    """Kahn's algorithm with a preference for staying in the current kind (routing / non-routing):
    among the ready nodes, keep emitting the kind of the last emitted node while any is ready.
    torch.onnx.export interleaves a block's routing chain with its expert convs one node at a time
    (41 segments on EsMoE-N); grouping turns each block's routing chain into a single run."""
    produced = set(available)
    remaining = list(range(len(nodes)))
    order = []
    cur_kind = None
    while remaining:
        ready = [i for i in remaining if all((t == "" or t in produced) for t in nodes[i].input)]
        if not ready:
            raise RuntimeError("graph is not topologically sortable (dangling input)")
        same = [i for i in ready if (marker in nodes[i].name) == cur_kind] if cur_kind is not None else []
        pick = same[0] if same else ready[0]
        cur_kind = marker in nodes[pick].name
        order.append(nodes[pick])
        produced.update(nodes[pick].output)
        remaining.remove(pick)
    return order


def segments_of(model: onnx.ModelProto, marker: str, min_run: int):
    g = model.graph
    inits = {t.name for t in g.initializer}
    graph_in = {i.name for i in g.input} - inits
    graph_out = [o.name for o in g.output]
    nodes = grouped_topological(list(g.node), inits | graph_in, marker)
    flags = [marker in n.name for n in nodes]
    # maximal runs; a routing run shorter than min_run is merged into its neighbours (it is then
    # computed in fp16, which prepare_models.py's fp16 ONNX would also have refused: keep min_run=1)
    runs = []
    for i, f in enumerate(flags):
        if runs and runs[-1][0] == f:
            runs[-1][1].append(i)
        else:
            runs.append([f, [i]])
    merged = []
    for f, idx in runs:
        if f and len(idx) < min_run and merged:
            merged[-1][1].extend(idx)
        else:
            merged.append([f, idx])
    # producer map
    producer = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            producer[o] = i
    seg_of_node = {}
    for si, (_, idx) in enumerate(merged):
        for i in idx:
            seg_of_node[i] = si
    consumers_after = {}
    for i, n in enumerate(nodes):
        for t in n.input:
            if t in producer and seg_of_node[producer[t]] != seg_of_node[i]:
                consumers_after.setdefault(t, set()).add(seg_of_node[i])
    out = []
    for si, (f, idx) in enumerate(merged):
        ins, outs = [], []
        for i in idx:
            for t in nodes[i].input:
                if t in inits or t == "":
                    continue
                if t in graph_in or (t in producer and seg_of_node[producer[t]] != si):
                    if t not in ins:
                        ins.append(t)
            for t in nodes[i].output:
                if t in graph_out or (t in consumers_after and any(s != si for s in consumers_after[t])):
                    if t not in outs:
                        outs.append(t)
        out.append({"precision": "fp32" if f else "fp16", "nodes": len(idx), "inputs": ins, "outputs": outs,
                    "first": nodes[idx[0]].name, "last": nodes[idx[-1]].name})
    return list(graph_in), graph_out, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--out", default=None, help="default: <onnx>.paths.json")
    ap.add_argument("--marker", default="/routing/")
    ap.add_argument("--min-run", type=int, default=1)
    a = ap.parse_args()
    m = onnx.load(a.onnx)
    gin, gout, segs = segments_of(m, a.marker, a.min_run)
    if not any(s["precision"] == "fp32" for s in segs):
        print(f"{a.onnx}: no '{a.marker}' nodes, nothing to pin (dense model)", file=sys.stderr)
    doc = {"schema": "mnn-paths/v1", "source": Path(a.onnx).name, "marker": a.marker,
           "inputs": gin, "outputs": gout, "segments": segs}
    out = Path(a.out) if a.out else Path(a.onnx).with_suffix(Path(a.onnx).suffix + ".paths.json")
    out.write_text(json.dumps(doc, indent=1))
    n32 = sum(s["nodes"] for s in segs if s["precision"] == "fp32")
    print(f"{out}: {len(segs)} segments, {n32} fp32 nodes of {sum(s['nodes'] for s in segs)}")
    for s in segs:
        print(f"  {s['precision']:4s} {s['nodes']:4d} nodes  in={len(s['inputs'])} out={len(s['outputs'])}  {s['first']} .. {s['last']}")


if __name__ == "__main__":
    main()
