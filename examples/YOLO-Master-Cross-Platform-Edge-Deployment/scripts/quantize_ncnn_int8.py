#!/usr/bin/env python3
"""Mixed-INT8 quantization of an ncnn float export with a structural exclusion set.

Port of the p03 recipe (``git show 9d0f537:tempo-ncnn/export_ncnn_p03.py`` steps 3-5) made
repeatable for every ncnn dir in ``models/``:

  1. build the exclusion set STRUCTURALLY from the ``.param`` text (pnnx names are generic,
     so nothing is keyed on names): stem pair, DFL conv, image-level logit heads (p03 rule),
     selective-kernel gates, optional detect/seg head, and -- for the router-emulated mixture
     models -- the whole router window between the entry ``Split`` and the routing ``div``;
  2. pre-render an even-spread TRAIN-split calibration set as letterboxed ``imgsz x imgsz``
     gray-114 BGR PNGs (cached under ``calib/ncnn_<domain>_<n>_<imgsz>/``), because
     ``ncnn2table``'s own plain resize distorts aspect and never shows the pad the runtime feeds
     the stem;
  3. ``ncnn2table`` (KL) -> strip BOTH the ``<name>`` and ``<name>_param_0`` rows of every
     excluded layer -> ``ncnn2int8``;
  4. hard post-checks (same layer list, ``{quantizable} - {8= layers} == excluded`` exactly, no
     requantize chaining, no ``8=`` on non-quantizable types, bin <= 0.6x float) and a
     ``metadata.yaml`` / ``quant_manifest.json`` receipt.

Everything before the tools is pure text processing over the ``.param`` (numpy / cv2 / pyyaml
only; the python ``ncnn`` wheel is never imported).

Usage::

    python scripts/quantize_ncnn_int8.py --model models/esmoe_n_visdrone_ncnn \\
        --calib-dir /data/datasets/VisDrone/images/train --calib-n 1024 \\
        --out models/esmoe_n_visdrone-int8_ncnn
    python scripts/quantize_ncnn_int8.py --param tempo-ncnn/models/p03_v01n.ncnn.param \\
        --bin tempo-ncnn/models/p03_v01n.ncnn.bin --calib-list tempo-ncnn/models/calib.txt \\
        --out models/p03_v01n-int8_ncnn
    python scripts/quantize_ncnn_int8.py --model models/mixture/moa-n_ncnn --dry-run --out /tmp/x
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NCNN_TAG = "20260526"
QUANTIZABLE_TYPES = ("Convolution", "ConvolutionDepthWise", "InnerProduct")
NEVER_INT8_TYPES = ("Deconvolution", "DeconvolutionDepthWise", "MatMul", "GroupNorm")
ROUTER_CONSTANTS = ("1.000000e-9", "1.000000e30")
ACTIVATION_TYPES = ("Swish", "ReLU", "Sigmoid", "HardSwish", "HardSigmoid", "GELU", "Mish", "TanH", "ELU", "SELU")
LOGIT_PASSTHROUGH = ("Clip", "BinaryOp", "Eltwise")
SK_PASSTHROUGH = ACTIVATION_TYPES + ("Reshape", "Flatten")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LETTERBOX_PAD = 114


# --------------------------------------------------------------------------------------------
# .param parsing
# --------------------------------------------------------------------------------------------
@dataclass
class Layer:
    """One ``.param`` row.

    Attributes:
        index: 0-based position among layer rows (file order == topological order in ncnn).
        type: ncnn layer type, e.g. ``Convolution``.
        name: layer name (NOT unique in the mixture exports: ``splitncnn_0..6`` repeat).
        ins: input blob names.
        outs: output blob names.
        kv: raw ``key=value`` params, keys kept as strings (``"0"``, ``"-23303"``).
    """

    index: int
    type: str
    name: str
    ins: list[str]
    outs: list[str]
    kv: dict[str, str] = field(default_factory=dict)

    @property
    def quantizable(self) -> bool:
        """True for the three layer types ``ncnn2int8`` can quantize."""
        return self.type in QUANTIZABLE_TYPES

    @property
    def out_channels(self) -> int:
        """``0=`` (num_output) or -1 when absent."""
        return int(self.kv.get("0", -1))

    @property
    def weight_elems(self) -> int:
        """Weight element count: ``6=`` for conv types, ``2=`` for InnerProduct, else 0."""
        key = "2" if self.type == "InnerProduct" else "6"
        return int(self.kv.get(key, 0)) if self.quantizable else 0


@dataclass
class ParamGraph:
    """Parsed ``.param``: layers in file order plus blob producer / consumer maps."""

    magic: str
    layer_count: int
    blob_count: int
    layers: list[Layer]
    producer: dict[str, int]  # blob -> layer index
    consumers: dict[str, list[int]]  # blob -> layer indices
    _desc: dict[int, frozenset[int]] = field(default_factory=dict, repr=False)
    _anc: dict[int, frozenset[int]] = field(default_factory=dict, repr=False)

    @property
    def quantizable(self) -> list[Layer]:
        """Quantizable layers in file order."""
        return [l for l in self.layers if l.quantizable]

    def by_name(self, name: str) -> Layer:
        """Return the unique quantizable layer with this name (quantizable names ARE unique)."""
        hits = [l for l in self.layers if l.name == name and l.quantizable]
        if len(hits) != 1:
            raise KeyError(f"{name}: {len(hits)} quantizable layers with that name")
        return hits[0]

    def descendants(self, index: int) -> frozenset[int]:
        """Indices of all quantizable layers reachable downstream of ``index`` (excluding itself).

        Propagates through non-quantizable layers; memoized in reverse file order.
        """
        if index in self._desc:
            return self._desc[index]
        acc: set[int] = set()
        for blob in self.layers[index].outs:
            for c in self.consumers.get(blob, ()):
                if self.layers[c].quantizable:
                    acc.add(c)
                acc |= self.descendants(c)
        self._desc[index] = frozenset(acc)
        return self._desc[index]

    def ancestors(self, index: int) -> frozenset[int]:
        """Indices of all quantizable layers upstream of ``index`` (excluding itself)."""
        if index in self._anc:
            return self._anc[index]
        acc: set[int] = set()
        for blob in self.layers[index].ins:
            p = self.producer.get(blob)
            if p is None:
                continue
            if self.layers[p].quantizable:
                acc.add(p)
            acc |= self.ancestors(p)
        self._anc[index] = frozenset(acc)
        return self._anc[index]

    def all_ancestors(self, index: int, floor: int = 0) -> set[int]:
        """Indices of EVERY layer upstream of ``index`` with index >= ``floor`` (any type)."""
        seen: set[int] = set()
        stack = [index]
        while stack:
            i = stack.pop()
            for blob in self.layers[i].ins:
                p = self.producer.get(blob)
                if p is not None and p >= floor and p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen


def parse_param(text: str) -> ParamGraph:
    """Parse ncnn ``.param`` text into a :class:`ParamGraph`.

    Args:
        text: full ``.param`` file contents (magic, counts, then one layer per line).

    Returns:
        ParamGraph with layers in file order and blob producer/consumer maps.

    Raises:
        ValueError: on a bad magic, a malformed row, or a blob with two producers.

    Examples:
        >>> g = parse_param("7767517\\n2 2\\nInput in0 0 1 in0\\nConvolution conv_0 1 1 in0 1 0=8 1=3 6=216\\n")
        >>> [l.name for l in g.quantizable], g.layers[1].out_channels, g.consumers["in0"]
        (['conv_0'], 8, [1])
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2 or lines[0].strip() != "7767517":
        raise ValueError("not an ncnn .param (magic 7767517 missing)")
    counts = lines[1].split()
    layer_count, blob_count = int(counts[0]), int(counts[1])
    layers: list[Layer] = []
    producer: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for row in lines[2:]:
        parts = row.split()
        if len(parts) < 4:
            raise ValueError(f"malformed .param row: {row!r}")
        ltype, name, nin, nout = parts[0], parts[1], int(parts[2]), int(parts[3])
        ins = parts[4 : 4 + nin]
        outs = parts[4 + nin : 4 + nin + nout]
        kv: dict[str, str] = {}
        for tok in parts[4 + nin + nout :]:
            k, _, v = tok.partition("=")
            kv[k] = v
        idx = len(layers)
        layers.append(Layer(idx, ltype, name, ins, outs, kv))
        for b in outs:
            if b in producer:
                raise ValueError(f"blob {b} produced twice (layers {producer[b]} and {idx})")
            producer[b] = idx
        for b in ins:
            consumers.setdefault(b, []).append(idx)
    if len(layers) != layer_count:
        raise ValueError(f"header says {layer_count} layers, found {len(layers)}")
    return ParamGraph("7767517", layer_count, blob_count, layers, producer, consumers)


# --------------------------------------------------------------------------------------------
# structural fingerprints
# --------------------------------------------------------------------------------------------
def is_router_emulated(text: str) -> bool:
    """True when the export carries the trace-time router emulation.

    The emulation leaves ``Reduction amax_*`` layers and literal ``1.000000e30`` constants
    (fp16-unsafe); both fingerprints must agree.

    Args:
        text: ``.param`` contents.

    Returns:
        bool: router-emulated (float remainder must stay fp32) or dense (fp16-safe).

    Raises:
        ValueError: if exactly one of the two fingerprints is present.

    Examples:
        >>> is_router_emulated("7767517\\n1 1\\nInput in0 0 1 in0\\n")
        False
    """
    has_amax = bool(re.search(r"^Reduction\s+amax_", text, re.M))
    has_big = "1.000000e30" in text
    if has_amax != has_big:
        raise ValueError(f"router fingerprint mismatch: amax_={has_amax} 1.000000e30={has_big}")
    return has_amax


def stem_pair(g: ParamGraph) -> list[str]:
    """The first two quantizable layers in file order (the stem convs).

    Raises:
        ValueError: if the first one is not a ``Convolution`` consuming blob ``in0``.
    """
    q = g.quantizable
    if len(q) < 2:
        raise ValueError("fewer than two quantizable layers")
    if q[0].type != "Convolution" or "in0" not in q[0].ins:
        raise ValueError(f"first quantizable layer {q[0].name} is not a Convolution on in0")
    return [q[0].name, q[1].name]


def dfl_layers(g: ParamGraph) -> list[str]:
    """DFL integral convs: ``Convolution 0=1 1=1 5=0 6=16`` (16 weights, no bias)."""
    want = {"0": "1", "1": "1", "5": "0", "6": "16"}
    return [l.name for l in g.layers if l.type == "Convolution" and all(l.kv.get(k) == v for k, v in want.items())]


def _walk_forward(g: ParamGraph, start_blob: str, passthrough: tuple[str, ...], target: str, max_hops: int = 12):
    """DFS from ``start_blob`` through ``passthrough`` layer types until a ``target`` layer.

    Returns:
        list[int] | None: the layer-index path (passthrough layers, then the target) or None.
    """
    stack: list[tuple[str, list[int], int]] = [(start_blob, [], 0)]
    while stack:
        blob, path, hops = stack.pop()
        for c in g.consumers.get(blob, ()):
            lay = g.layers[c]
            if lay.type == target:
                return path + [c]
            if lay.type in passthrough and hops < max_hops:
                for ob in lay.outs:
                    stack.append((ob, path + [c], hops + 1))
    return None


def image_level_logit_heads(g: ParamGraph, max_out: int = 32) -> list[str]:
    """Image-level router logit heads (p03 rule).

    ``Convolution`` (out-ch <= ``max_out``) -> ``Reduction 0=3`` (spatial mean) ->
    ``[Clip | BinaryOp | Eltwise]*`` -> ``Softmax``.
    """
    heads = []
    for l in g.layers:
        if l.type != "Convolution" or l.out_channels > max_out:
            continue
        for c in g.consumers.get(l.outs[0], ()):
            red = g.layers[c]
            if red.type == "Reduction" and red.kv.get("0") == "3":
                if _walk_forward(g, red.outs[0], LOGIT_PASSTHROUGH, "Softmax") is not None:
                    heads.append(l.name)
                    break
    return heads


def _is_gap(l: Layer) -> bool:
    return l.type == "Pooling" and l.name.startswith("gap_") and l.kv.get("0") == "1" and l.kv.get("4") == "1"


def _sk_passthrough_ok(l: Layer) -> bool:
    if l.type == "Convolution":
        return l.kv.get("1") == "1" and l.kv.get("11", "1") == "1"
    return l.type == "InnerProduct" or l.type in SK_PASSTHROUGH


def sk_gate_chains(g: ParamGraph, max_out: int = 16) -> list[tuple[int, list[str]]]:
    """Selective-kernel gates as ``(gap_index, [quantizable names on the chain])``.

    Chain: ``Pooling gap_* (0=1 4=1)`` -> ``{Conv 1x1 | InnerProduct | activation | Reshape |
    Flatten}+`` -> ``Softmax``; the gate-logit layer (the last quantizable one, whose output
    the Softmax normalises over the kernel branches) has out-ch <= ``max_out``. The bound is
    NOT applied to the reducer conv: the P5 gate of EsMoE-N / v0.1-seg-N reduces to 32 channels.
    """
    gates = []
    for l in g.layers:
        if not _is_gap(l):
            continue
        stack: list[tuple[str, list[int]]] = [(l.outs[0], [])]
        found = None
        while stack and found is None:
            blob, path = stack.pop()
            for c in g.consumers.get(blob, ()):
                lay = g.layers[c]
                if lay.type == "Softmax":
                    found = path
                    break
                if _sk_passthrough_ok(lay) and len(path) < 12:
                    stack.append((lay.outs[0], path + [c]))
        if found is None:
            continue
        q = [g.layers[i] for i in found if g.layers[i].quantizable]
        if q and q[-1].out_channels <= max_out:
            gates.append((l.index, [x.name for x in q]))
    return gates


def sk_gates(g: ParamGraph, max_out: int = 16) -> list[str]:
    """Names of the selective-kernel gate convs (see :func:`sk_gate_chains`), file order."""
    return [n for _, names in sk_gate_chains(g, max_out) for n in names]


def head_layers(g: ParamGraph) -> list[str]:
    """Detect/seg head convs: quantizable layers whose quantizable descendants are a subset of the DFL conv(s).

    The DFL conv itself is reported by :func:`dfl_layers`, not here.
    """
    dfl = {g.by_name(n).index for n in dfl_layers(g)}
    return [l.name for l in g.quantizable if l.index not in dfl and g.descendants(l.index) <= dfl]


@dataclass
class RouterWindow:
    """One emulated router: file-order window ``[lower, sink]`` around a pair of ``amax_*``."""

    lower: int
    sink: int
    amax: tuple[int, int]
    experts: int
    window: list[str]  # all quantizable names inside the window
    cone: list[str]  # window members that are quantizable ancestors of the sink

    def names(self, scope: str) -> list[str]:
        """Excluded names for ``scope`` (``window`` or ``cone``)."""
        if scope not in ("window", "cone"):
            raise ValueError(f"router scope must be window|cone, got {scope}")
        return list(self.window if scope == "window" else self.cone)


def _router_entry_gap(g: ParamGraph, gap: Layer) -> bool:
    """A router's pooled-input gap feeds an ``InnerProduct`` through Reshape/Flatten only."""
    return _walk_forward(g, gap.outs[0], ("Reshape", "Flatten"), "InnerProduct", max_hops=4) is not None


def router_windows(g: ParamGraph) -> list[RouterWindow]:
    """Router windows of a router-emulated export.

    For each consecutive pair of ``Reduction amax_*`` layers: the sink is the layer just before
    the first quantizable layer after the second ``amax`` (a ``BinaryOp div_*``); the lower
    bound is the ``Split`` feeding the nearest preceding router-entry ``Pooling gap_*`` (the
    gap whose pooled output reaches an ``InnerProduct`` via Reshape/Flatten -- SE-block gaps,
    which feed a 1x1 conv, are skipped) that is an ancestor of the first ``amax``.

    Each window must hold exactly one ``Softmax``, one ``MemoryData 0=E`` (E > 1, the expert
    count) and three ``1.000000e-9`` / ``1.000000e30`` constants.

    Raises:
        ValueError: on an odd ``amax_`` count or a window violating the invariants.
    """
    amax = [l.index for l in g.layers if l.type == "Reduction" and l.name.startswith("amax_")]
    if len(amax) % 2:
        raise ValueError(f"odd number of amax_ reductions: {len(amax)}")
    windows: list[RouterWindow] = []
    prev_sink = -1
    for a, b in zip(amax[0::2], amax[1::2]):
        nxt = next((l.index for l in g.layers if l.index > b and l.quantizable), None)
        if nxt is None:
            raise ValueError(f"no quantizable layer after {g.layers[b].name}")
        sink = nxt - 1
        s = g.layers[sink]
        if not (s.type == "BinaryOp" and s.name.startswith("div_")):
            raise ValueError(f"router sink {s.type} {s.name} is not a BinaryOp div_*")
        anc = g.all_ancestors(a, floor=prev_sink + 1)
        gaps = [i for i in sorted(anc) if i < a and _is_gap(g.layers[i]) and _router_entry_gap(g, g.layers[i])]
        if not gaps:
            raise ValueError(f"no router-entry gap_ before {g.layers[a].name}")
        gap = g.layers[gaps[-1]]
        lower = g.producer.get(gap.ins[0])
        if lower is None or g.layers[lower].type != "Split":
            raise ValueError(f"{gap.name} is not fed by a Split")
        body = g.layers[lower : sink + 1]
        softmax = [l for l in body if l.type == "Softmax"]
        mem = [l for l in body if l.type == "MemoryData" and set(l.kv) == {"0"} and int(l.kv["0"]) > 1]
        consts = [l for l in body if any(v in ROUTER_CONSTANTS for v in l.kv.values())]
        if len(softmax) != 1 or len(mem) != 1 or len(consts) != 3:
            raise ValueError(
                f"router window {g.layers[lower].name}..{s.name}: softmax={len(softmax)} "
                f"memorydata(E>1)={len(mem)} constants={len(consts)} (want 1/1/3)"
            )
        qwin = [l for l in body if l.quantizable]
        sink_anc = g.ancestors(sink)
        windows.append(
            RouterWindow(
                lower=lower,
                sink=sink,
                amax=(a, b),
                experts=int(mem[0].kv["0"]),
                window=[l.name for l in qwin],
                cone=[l.name for l in qwin if l.index in sink_anc],
            )
        )
        prev_sink = sink
    return windows


def build_exclusions(
    text: str, exclude_head: bool = False, keep_sk_gates: bool = False, router_scope: str = "window"
) -> dict:
    """Build the INT8 exclusion set from ``.param`` text.

    Categories are disjoint, claimed in priority order stem > dfl > router > logit_heads >
    sk_gates > head, so ``excluded`` is their concatenation (file order within each).

    Args:
        text: ``.param`` contents.
        exclude_head: also exclude the detect/seg head convs (:func:`head_layers`).
        keep_sk_gates: keep the selective-kernel gate convs quantized (reported under
            ``sk_gates_kept`` instead).
        router_scope: ``window`` (every quantizable layer inside each router window) or
            ``cone`` (only the sink's quantizable ancestors, for ablation).

    Returns:
        dict with ``router_emulated``, ``quantizable_total``, ``stem``, ``dfl``, ``router``
        (list of per-router lists), ``logit_heads``, ``sk_gates``, ``head``, ``excluded``,
        ``sk_gates_kept``, ``head_available`` and ``routers`` (per-router metadata).
    """
    g = parse_param(text)
    emulated = is_router_emulated(text)
    claimed: set[str] = set()

    def claim(names):
        out = [n for n in names if n not in claimed]
        claimed.update(out)
        return out

    stem = claim(stem_pair(g))
    dfl = claim(dfl_layers(g))
    windows = router_windows(g) if emulated else []
    router = [claim(w.names(router_scope)) for w in windows]
    logit = claim(image_level_logit_heads(g))
    sk_found = sk_gates(g)
    sk = [] if keep_sk_gates else claim(sk_found)
    head_all = head_layers(g)
    head = claim(head_all) if exclude_head else []
    excluded = stem + dfl + [n for r in router for n in r] + logit + sk + head
    return {
        "router_emulated": emulated,
        "quantizable_total": len(g.quantizable),
        "stem": stem,
        "dfl": dfl,
        "router": router,
        "logit_heads": logit,
        "sk_gates": sk,
        "head": head,
        "excluded": excluded,
        "sk_gates_kept": sk_found if keep_sk_gates else [],
        "head_available": head_all,
        "router_scope": router_scope,
        "routers": [
            {
                "lower": g.layers[w.lower].name,
                "sink": g.layers[w.sink].name,
                "amax": [g.layers[i].name for i in w.amax],
                "experts": w.experts,
                "window": w.window,
                "cone": w.cone,
            }
            for w in windows
        ],
    }


# --------------------------------------------------------------------------------------------
# calibration table handling
# --------------------------------------------------------------------------------------------
def strip_table(table_text: str, excluded_names) -> tuple[str, list[str], list[str]]:
    """Drop the ``<name>`` and ``<name>_param_0`` rows of every excluded layer from a table.

    Args:
        table_text: ``ncnn2table`` output.
        excluded_names: layer names to keep in float.

    Returns:
        tuple: ``(kept_text, dropped_row_keys, warnings)`` where ``warnings`` lists excluded
        names that had no row at all (typos or layers ncnn2table never calibrated).

    Examples:
        >>> kept, dropped, warn = strip_table("a_param_0 1 2\\nb_param_0 3\\na 0.5\\nb 0.7\\n", ["a", "zz"])
        >>> kept.splitlines(), dropped, warn
        (['b_param_0 3', 'b 0.7'], ['a_param_0', 'a'], ['zz'])
    """
    excluded = set(excluded_names)
    keep, dropped, seen = [], [], set()
    for line in table_text.splitlines():
        parts = line.split()
        key = parts[0] if parts else ""
        base = key[: -len("_param_0")] if key.endswith("_param_0") else key
        if base in excluded:
            dropped.append(key)
            seen.add(base)
        else:
            keep.append(line)
    warnings = [n for n in excluded_names if n not in seen]
    return "\n".join(keep) + "\n", dropped, warnings


# --------------------------------------------------------------------------------------------
# calibration images
# --------------------------------------------------------------------------------------------
def letterbox_bgr(img, imgsz: int = 640):
    """Letterbox a BGR uint8 image onto a gray-114 ``imgsz x imgsz`` canvas (runtime preprocessing).

    Same geometry as ``scripts/quantize_int8.py:letterbox`` (round-scaled, centered); returns
    the BGR canvas so ``ncnn2table pixel=RGB`` does the BGR->RGB swap and ``norm`` the /255.
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = round(w * r), round(h * r)
    canvas = np.full((imgsz, imgsz, 3), LETTERBOX_PAD, np.uint8)
    px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[py : py + nh, px : px + nw] = cv2.resize(img, (nw, nh))
    return canvas


def md5_of(data: bytes) -> str:
    """Hex MD5 of ``data``."""
    return hashlib.md5(data).hexdigest()


def md5_file(path: Path) -> str:
    """Hex MD5 of a file, streamed."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_domain(source: Path) -> str:
    """Short dataset tag from a calibration source path (``coco``, ``visdrone``, else basename)."""
    s = str(source).lower()
    for tag in ("coco", "visdrone"):
        if tag in s:
            return tag
    return re.sub(r"[^a-z0-9]+", "-", source.name.lower()).strip("-") or "custom"


def select_calibration_sources(calib_dir: Path | None, calib_list: Path | None, n: int) -> tuple[list[Path], str]:
    """Pick calibration source images.

    ``calib_list`` (one path per line) wins and reproduces a previous run exactly; otherwise
    the sorted images of ``calib_dir`` are sampled at ``np.linspace`` positions (even spread).

    Returns:
        tuple: ``(paths, source_label)``.
    """
    import numpy as np

    if calib_list is not None:
        paths = [Path(p.strip()) for p in calib_list.read_text().splitlines() if p.strip()]
        if not paths:
            raise SystemExit(f"--calib-list {calib_list} is empty")
        return paths, str(calib_list)
    if calib_dir is None:
        raise SystemExit("need --calib-dir or --calib-list")
    imgs = sorted(p for p in calib_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        raise SystemExit(f"no images under {calib_dir}")
    k = min(n, len(imgs))
    idx = np.unique(np.linspace(0, len(imgs) - 1, k).astype(int))
    return [imgs[i] for i in idx], str(calib_dir)


def render_calibration(
    paths: list[Path], source: str, imgsz: int, cache_root: Path, domain: str
) -> tuple[Path, list[Path], str]:
    """Pre-render letterboxed PNGs into ``cache_root/ncnn_<domain>_<n>_<imgsz>/`` (cached).

    The cache is reused when its ``manifest.json`` carries the same list md5 / imgsz and every
    PNG exists; otherwise it is re-rendered from scratch.

    Returns:
        tuple: ``(cache_dir, png_paths, list_md5)``.
    """
    import cv2

    list_md5 = md5_of("\n".join(str(p) for p in paths).encode())
    cache = cache_root / f"ncnn_{domain}_{len(paths)}_{imgsz}"
    manifest = cache / "manifest.json"
    pngs = [cache / f"{i:05d}.png" for i in range(len(paths))]
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            m = {}
        if m.get("list_md5") == list_md5 and m.get("imgsz") == imgsz and all(p.exists() for p in pngs):
            print(f"[calib] reusing {cache} ({len(pngs)} png, md5 {list_md5[:8]})")
            return cache, pngs, list_md5
        print(f"[calib] stale cache {cache}; re-rendering")
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for i, (src, dst) in enumerate(zip(paths, pngs)):
        img = cv2.imread(str(src))
        if img is None:
            raise SystemExit(f"cannot read calibration image {src}")
        cv2.imwrite(str(dst), letterbox_bgr(img, imgsz))
        if (i + 1) % 200 == 0 or i + 1 == len(paths):
            print(f"[calib] rendered {i + 1}/{len(paths)}")
    manifest.write_text(
        json.dumps(
            {
                "source": source,
                "n": len(paths),
                "imgsz": imgsz,
                "list_md5": list_md5,
                "letterbox": {
                    "pad": LETTERBOX_PAD,
                    "mode": "center",
                    "colour_order": "BGR png -> ncnn2table pixel=RGB",
                },
                "sources": [str(p) for p in paths],
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=1,
        )
    )
    return cache, pngs, list_md5


# --------------------------------------------------------------------------------------------
# ncnn tools
# --------------------------------------------------------------------------------------------
def check_ncnn_root(root: Path) -> str:
    """Locate ``ncnn2table``/``ncnn2int8`` under ``root/bin`` and verify ``VERSION.txt``'s tag.

    Returns:
        str: the tag recorded in ``VERSION.txt`` (or ``"unknown"`` when the file is absent).

    Raises:
        SystemExit: when a tool is missing or the tag is not :data:`NCNN_TAG`.
    """
    for tool in ("ncnn2table", "ncnn2int8"):
        if not (root / "bin" / tool).exists():
            raise SystemExit(f"{root / 'bin' / tool} missing (run scripts/build_ncnn_x86.sh)")
    version = root / "VERSION.txt"
    if not version.exists():
        print(f"[ncnn] WARNING: {version} missing; tag unverified")
        return "unknown"
    txt = version.read_text()
    m = re.search(r"tag=(\S+)", txt) or re.search(r"\b(\d{8})\b", txt)
    tag = m.group(1) if m else txt.strip().split()[0]
    if tag != NCNN_TAG:
        raise SystemExit(f"{version}: tag {tag} != required {NCNN_TAG}")
    return tag


def run_tool(root: Path, tool: str, args: list[str]) -> None:
    """Run ``root/bin/<tool>`` with ``root/lib`` on ``LD_LIBRARY_PATH``; raise on failure."""
    env = {**os.environ, "LD_LIBRARY_PATH": str(root / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    cmd = [str(root / "bin" / tool), *args]
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


# --------------------------------------------------------------------------------------------
# post-conversion checks
# --------------------------------------------------------------------------------------------
def post_checks(float_text: str, int8_text: str, excluded: list[str], float_bin: Path, int8_bin: Path) -> dict:
    """Hard checks on the ``ncnn2int8`` output.

    Returns:
        dict with ``int8_layers`` (names), ``quantizable``, ``layer_coverage``,
        ``weight_byte_coverage``, ``bin_ratio`` and ``int8_flags`` (histogram of ``8=`` values).

    Raises:
        SystemExit: on any violated invariant.
    """
    gf, gi = parse_param(float_text), parse_param(int8_text)
    if [(l.type, l.name) for l in gf.layers] != [(l.type, l.name) for l in gi.layers]:
        raise SystemExit("int8 param does not carry the identical layer list (type, name, order)")
    q = gi.quantizable
    int8 = [l for l in q if "8" in l.kv]
    flags: dict[str, int] = {}
    for l in int8:
        flags[l.kv["8"]] = flags.get(l.kv["8"], 0) + 1
    got_float = {l.name for l in q} - {l.name for l in int8}
    if got_float != set(excluded):
        extra, missing = sorted(got_float - set(excluded)), sorted(set(excluded) - got_float)
        raise SystemExit(f"float remainder != exclusion set: unexpectedly float {extra}; unexpectedly int8 {missing}")
    if flags.get("101") or flags.get("102"):
        # 8=101/102 = ncnn's fused int8->int8 requantize on adjacent quantized layers (no float
        # dequantize between them). It is a legitimate ncnn path, not a conversion error: models
        # whose every conv is followed by Swish (p03) get none; models with adjacent 1x1 convs
        # (EsMoE-N, seg-N head/proto) get a few. It changes rounding vs float-bounded int8, so it
        # is recorded (the flag histogram lands in the manifest) and judged by the accuracy gate.
        n_req = flags.get("101", 0) + flags.get("102", 0)
        print(f"[warn] requantize chaining (8=101/102) on {n_req} layer(s): {flags} - legitimate fused "
              f"int8->int8 path; rounding differs, so rely on tests/certify_ncnn_int8.py for the verdict")
    bad = [f"{l.type} {l.name}" for l in gi.layers if l.type in NEVER_INT8_TYPES and "8" in l.kv]
    if bad:
        raise SystemExit(f"8= on non-quantizable layers: {bad}")
    ratio = int8_bin.stat().st_size / float_bin.stat().st_size
    if ratio > 0.6:
        raise SystemExit(f"int8 bin is {ratio:.2f}x the float bin (> 0.6x)")
    w_all = sum(l.weight_elems for l in q)
    w_int8 = sum(l.weight_elems for l in int8)
    return {
        "int8_layers": [l.name for l in int8],
        "quantizable": len(q),
        "layer_coverage": len(int8) / len(q),
        "weight_byte_coverage": w_int8 / w_all if w_all else 0.0,
        "bin_ratio": ratio,
        "int8_flags": flags,
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def print_exclusions(ex: dict) -> float:
    """Print the classification and per-category exclusion counts; return the layer coverage."""
    total, excluded = ex["quantizable_total"], len(ex["excluded"])
    kind = "router-emulated (float remainder fp32)" if ex["router_emulated"] else "dense (float remainder fp16)"
    print(f"[classify] {kind}; quantizable={total}")
    for cat in ("stem", "dfl", "logit_heads", "sk_gates", "head"):
        print(f"  {cat:12s} {len(ex[cat]):3d}  {ex[cat]}")
    for i, r in enumerate(ex["router"]):
        print(f"  router[{i}]    {len(r):3d}  {r}")
    if ex["sk_gates_kept"]:
        print(f"  (kept quantized sk_gates: {ex['sk_gates_kept']})")
    if not ex["head"] and ex["head_available"]:
        print(f"  (head convs left int8: {ex['head_available']}; --exclude-head to keep float)")
    cov = (total - excluded) / total if total else 0.0
    print(f"[coverage] excluded={excluded}  int8 layers={total - excluded}/{total}  coverage={cov:.4f}")
    return cov


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("input")
    src.add_argument("--model", type=Path, help="float ncnn dir with model.ncnn.param/.bin (+ metadata.yaml)")
    src.add_argument("--param", type=Path, help="float .param (with --bin) instead of --model")
    src.add_argument("--bin", type=Path, help="float .bin (with --param)")
    cal = ap.add_argument_group("calibration")
    cal.add_argument("--calib-dir", type=Path, help="TRAIN-split image dir (even-spread sampled)")
    cal.add_argument("--calib-n", type=int, default=1024, help="number of calibration images (default 1024)")
    cal.add_argument("--calib-list", type=Path, help="explicit image list (one path per line); overrides --calib-dir")
    cal.add_argument("--domain", help="dataset tag for the calib cache dir (inferred from the source path)")
    cal.add_argument("--calib-root", type=Path, default=REPO / "calib", help="cache root (default <repo>/calib)")
    cal.add_argument("--method", default="aciq",
                     help="ncnn2table calibration method (default aciq). Measured on p03_v01n COCO val: KL's "
                          "histogram clipping lost ~28%% of detections at conf 0.25 (mAP50-95 -3.2 pts, box match "
                          "0.77); ACIQ's analytical clipping kept detection parity (-2.0 pts, match 0.91) at "
                          "identical layer coverage. kl / eq remain available for ablation.")
    cal.add_argument("--imgsz", type=int, default=640)
    exc = ap.add_argument_group("exclusions")
    exc.add_argument("--exclude-head", action="store_true", help="keep the detect/seg head convs float")
    exc.add_argument("--keep-sk-gates", action="store_true", help="quantize the selective-kernel gate convs")
    exc.add_argument("--router-scope", choices=("window", "cone"), default="window")
    run = ap.add_argument_group("run")
    run.add_argument("--out", type=Path, help="output dir (default <model>-int8_ncnn next to --model)")
    run.add_argument("--ncnn-root", type=Path, default=REPO / "third_party" / "ncnn-x86-20260526")
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--dry-run", action="store_true", help="everything except ncnn2table/ncnn2int8")
    return ap


def resolve_inputs(args) -> tuple[Path, Path, Path | None, Path]:
    """Resolve ``(param, bin, metadata_or_None, out_dir)`` from the CLI arguments."""
    if args.model:
        d = args.model
        param, binf, meta = d / "model.ncnn.param", d / "model.ncnn.bin", d / "metadata.yaml"
        if not meta.exists():
            meta = None
        default_out = d.parent / (d.name[: -len("_ncnn")] if d.name.endswith("_ncnn") else d.name)
        default_out = default_out.with_name(default_out.name + "-int8_ncnn")
    elif args.param and args.bin:
        param, binf = args.param, args.bin
        cand = param.with_name("metadata.yaml")
        meta = cand if cand.exists() else None
        default_out = param.parent / (param.name.split(".")[0] + "-int8_ncnn")
    else:
        raise SystemExit("need --model <dir> or --param + --bin")
    for p in (param, binf):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    return param, binf, meta, (args.out or default_out)


def write_metadata(meta_src: Path | None, dst: Path, imgsz: int, quant: dict) -> bool:
    """Write ``metadata.yaml`` = source yaml + ``precision: int8`` + ``quant:``; synthesize when absent.

    Returns:
        bool: True when the metadata was synthesized (no source yaml).
    """
    import yaml

    synthesized = meta_src is None
    meta = yaml.safe_load(meta_src.read_text()) if meta_src else {"imgsz": [imgsz, imgsz], "end2end": False}
    meta = dict(meta or {})
    meta["precision"] = "int8"
    meta["quant"] = quant
    dst.write_text(yaml.safe_dump(meta, sort_keys=False, default_flow_style=False, allow_unicode=True))
    return synthesized


def main(argv=None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    param, binf, meta_src, out = resolve_inputs(args)
    text = param.read_text()
    print(f"[input] {param} ({binf.stat().st_size / 1e6:.1f} MB bin)")
    ex = build_exclusions(text, args.exclude_head, args.keep_sk_gates, args.router_scope)
    layer_cov = print_exclusions(ex)
    out.mkdir(parents=True, exist_ok=True)
    qdir = out / "quant"
    qdir.mkdir(exist_ok=True)

    calib_info: dict = {"skipped": True}
    if args.calib_dir or args.calib_list:
        sources, source_label = select_calibration_sources(args.calib_dir, args.calib_list, args.calib_n)
        domain = args.domain or infer_domain(Path(source_label))
        cache, pngs, list_md5 = render_calibration(sources, source_label, args.imgsz, args.calib_root, domain)
        list_path = qdir / "calib_list.txt"
        list_path.write_text("\n".join(str(p.resolve()) for p in pngs) + "\n")
        calib_info = {
            "skipped": False,
            "source": source_label,
            "n": len(pngs),
            "cache_dir": str(cache),
            "list": str(list_path),
            "list_md5": list_md5,
            "png_list_md5": md5_of(list_path.read_bytes()),
        }
        print(f"[calib] {len(pngs)} images from {source_label} -> {list_path}")
    elif not args.dry_run:
        raise SystemExit("need --calib-dir or --calib-list (or --dry-run)")
    else:
        print("[calib] skipped (dry-run without --calib-dir/--calib-list)")

    manifest = {
        "source_param": str(param),
        "source_bin": str(binf),
        "source_bin_md5": md5_file(binf),
        "dry_run": args.dry_run,
        "router_emulated": ex["router_emulated"],
        "router_scope": args.router_scope,
        "quantizable": ex["quantizable_total"],
        "excluded": ex["excluded"],
        "excluded_by_category": {k: ex[k] for k in ("stem", "dfl", "router", "logit_heads", "sk_gates", "head")},
        "sk_gates_kept": ex["sk_gates_kept"],
        "head_available": ex["head_available"],
        "routers": ex["routers"],
        "layer_coverage": layer_cov,
        "calib": calib_info,
        "method": args.method,
        "imgsz": args.imgsz,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.dry_run:
        (out / "quant_manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"[dry-run] wrote {out / 'quant_manifest.json'}; tools not invoked")
        return 0

    tag = check_ncnn_root(args.ncnn_root)
    table = qdir / "model.table"
    run_tool(
        args.ncnn_root,
        "ncnn2table",
        [
            str(param),
            str(binf),
            calib_info["list"],
            str(table),
            "mean=[0,0,0]",
            "norm=[0.003922,0.003922,0.003922]",
            f"shape=[{args.imgsz},{args.imgsz},3]",
            "pixel=RGB",
            f"thread={args.threads}",
            f"method={args.method}",
        ],
    )
    table_text = table.read_text()
    kept, dropped, warnings = strip_table(table_text, ex["excluded"])
    stripped = qdir / "model_stripped.table"
    stripped.write_text(kept)
    for w in warnings:
        print(f"[table] WARNING: excluded layer {w} had no table rows")
    n_rows = len([ln for ln in table_text.splitlines() if ln.strip()])
    print(f"[table] rows={n_rows} dropped={len(dropped)} kept={n_rows - len(dropped)}")

    out_param, out_bin = out / "model.ncnn.param", out / "model.ncnn.bin"
    run_tool(args.ncnn_root, "ncnn2int8", [str(param), str(binf), str(out_param), str(out_bin), str(stripped)])
    checks = post_checks(text, out_param.read_text(), ex["excluded"], binf, out_bin)
    print(
        f"[check] int8 layers={len(checks['int8_layers'])}/{checks['quantizable']} "
        f"layer coverage={checks['layer_coverage']:.4f} weight-byte coverage={checks['weight_byte_coverage']:.4f} "
        f"bin {out_bin.stat().st_size / 1e6:.1f} MB ({checks['bin_ratio']:.2f}x) flags={checks['int8_flags']}"
    )

    quant = {
        "source_bin_md5": manifest["source_bin_md5"],
        "ncnn_tag": tag,
        "method": args.method,
        "calib_source": calib_info["source"],
        "calib_n": calib_info["n"],
        "imgsz": args.imgsz,
        "letterbox": f"center, pad {LETTERBOX_PAD}, BGR png -> ncnn2table pixel=RGB norm 1/255",
        "excluded": {k: len(ex[k]) for k in ("stem", "dfl", "logit_heads", "sk_gates", "head")}
        | {"router": sum(len(r) for r in ex["router"]), "total": len(ex["excluded"])},
        "int8_layers": len(checks["int8_layers"]),
        "quantizable": checks["quantizable"],
        "weight_byte_coverage": round(checks["weight_byte_coverage"], 4),
        "float_remainder": "fp32" if ex["router_emulated"] else "fp16",
        "generated": manifest["generated"],
    }
    synthesized = write_metadata(meta_src, out / "metadata.yaml", args.imgsz, quant)
    if synthesized:
        print("[meta] NOTE: no source metadata.yaml; synthesized a minimal one (imgsz, end2end: false, no names)")
    manifest.update(
        {
            "ncnn_tag": tag,
            "int8_layers": checks["int8_layers"],
            "int8_flags": checks["int8_flags"],
            "weight_byte_coverage": checks["weight_byte_coverage"],
            "bin_ratio": checks["bin_ratio"],
            "table": {"path": str(table), "rows": n_rows, "dropped_rows": dropped, "kept_rows": n_rows - len(dropped)},
            "table_warnings": warnings,
            "stripped_table": str(stripped),
            "metadata_synthesized": synthesized,
        }
    )
    (out / "quant_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
