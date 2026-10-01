# YOLO-Master CoreML Runner 1.1.0

Three inference upgrades, ported from upstream [YOLO-Master](https://github.com/Tencent/YOLO-Master)
to run fully on-device: **traditional (dense) tiling**, **Sparse SAHI Mode**, and
**Cluster-Weighted NMS (CW-NMS)**. All three target the verticals this project cares about
(VisDrone, AI-TOD-v2, SKU-110K), where tiny objects vanish when a large frame is squeezed into
the model's fixed 640-px input.

## New: Slicing (sidebar section)

Applies to **images and folder batches**; video and webcam stay single-pass.

- **Dense** - a global full-image pass plus every 640-px tile of a 20%-overlap grid.
  Slowest, best small-object recall.
- **Sparse SAHI** - the global pass paints a 1/8-resolution objectness map from its own
  detections; only tiles where that map exceeds 0.15 run. If *no* tile qualifies, every tile
  runs (the upstream all-or-nothing fallback). Near-dense accuracy at a fraction of the tiles.
- Tile counts (run / grid, fallback, cap) appear in the Inference stats card.
  A 256-tile-per-image safety cap keeps gigapixel inputs from hanging; hitting it is
  reported as "(capped)".
- **Tile size is adjustable**: from the model input size (native scale, the default) up to
  1/4 of the source's short side. Larger tiles are letterboxed down to the model input
  (fewer forwards, coarser detail - upstream `slice_size` semantics). The bound is enforced
  per image, so a mixed-size folder clamps each image to its own limit; the stats card shows
  the tile size actually used (a range when it varies).
- Changing the slicing mode, tile size, or the masks toggle re-runs inference (like the
  Preprocess control); conf / IoU / NMS-mode / sigma tuning stays instant on the cached
  candidate pool. The tile-size slider commits on release, not per drag tick.
- **Segmentation masks in sliced modes are opt-in via "Masks (global pass)"**: when enabled,
  detections from the full-image pass render masks as in 1.0.0, while tile detections stay
  boxes-only - tile mask coefficients are meaningless against a full-image prototype tensor,
  and caching a prototype tensor per tile would be prohibitive in folder mode. With the
  toggle off, sliced modes are boxes-only and the Overlay picker hides.

## New: NMS mode (Detection section)

- **Standard** - the classic greedy suppression (unchanged default).
- **Cluster-Weighted** - after standard selection, each surviving box's coordinates are
  refined as the score-and-proximity-weighted average of its same-class overlapping
  candidates (`w = score · exp(-(1-IoU)²/σ)`, σ adjustable 0.01-0.5, default 0.1).
  Survivor count, scores and classes are identical to Standard; only localization changes.
  Applies everywhere: images, folders, video, live camera, exports - and it is the merge
  used for sliced inference when selected.

## New: Zoom (images + paused video)

Pinch to zoom (cursor-anchored, up to 8x), drag to pan, ⌘0 to reset,
⌘+ / ⌘− for stepped zoom. Works on the annotated image preview and on video **while paused**
(the overlay boxes scale with the pixels); zoom resets automatically on play, when browsing
to another image, and on re-run. Scrubbing while paused keeps the zoom, for frame-to-frame
comparison at magnification.

## New: annotation export (Export labels…)

Turn inference results into training data, in three formats - **YOLO TXT**, **COCO JSON**,
**Pascal VOC XML** - exported WYSIWYG at the current conf / IoU / NMS-mode / sigma:

- **Single image** -> one annotation file (save panel; YOLO also writes a sibling classes.txt).
- **Folder** -> `<folder>_labels/` with one file per image (COCO: one `annotations.coco.json`).
- **Video** -> `<video>_annotations/` with `frames/` (extracted JPEGs, source-frame-indexed)
  plus `labels/` (or one COCO json referencing `frames/…`). The **sampling rate** is selectable
  in the scrubber bar: every frame, one per second (default), or every 5th/10th/30th.
- **Segmentation models export polygons** (YOLO-seg lines, COCO polygon segmentation),
  traced from the instance masks; Pascal VOC has no standard segmentation field and always
  writes boxes. Detection models export boxes. In tiled modes the export follows what the
  preview shows: with "Masks (global pass)" on, global-pass instances carry real polygons and
  tile instances fall back to their box as a 4-point polygon; with it off, the export is
  box-dialect. Empty results still write (empty .txt = verified negative).
- Format details: COCO uses 1-based category ids (cls+1), polygon shoelace areas, and keeps
  a `score` key per annotation; VOC uses 1-based inclusive integer boxes with no score
  element; YOLO coordinates are normalized to 6 decimal places.

## CLI

```
yolomaster-coreml ... [--slicing off|dense|sparse [--tile-size N] [--slicing-masks]] [--cw-nms [--sigma 0.1]]
```

`--slicing` on a video source warns and runs single-pass.

## Fidelity notes (deliberate deviations from upstream)

The port follows upstream's **code** (`predictor.py:542-699`, `nms.py:167-207`), with four
documented deviations:

1. **No per-tile NMS.** Upstream NMSes each tile, then merge-NMSes. This runner pools
   *pre-NMS* candidates from the global pass and all tiles and applies one NMS at the user's
   conf/IoU - which is what keeps slider tuning instant, and gives CW-NMS the full candidate
   pool its weighting is defined over.
2. **Tile detections are clipped to real crop content.** Upstream lets boxes live on the
   gray padding of edge tiles.
3. **CW-NMS survivor guard.** Upstream's top-3000 candidate cap can collapse a survivor with
   no qualifying neighbors to (0,0,0,0); here it keeps its original box.
4. **CW-NMS actually runs at predict time, and (when selected) merges the tiles.** Upstream's
   README describes both, but its code wires CW-NMS only into the validator and merges SAHI
   tiles with plain NMS. Selecting Cluster-Weighted here realizes the documented behavior.

Port fidelity was verified against upstream directly: the tile grid matches the Python
reference exactly on 20 image sizes (including degenerate ones), and CW-NMS output matches
upstream's torch implementation to float precision on class-separated toy scenes.

## Also

- Sliced inference reports "Model-only" time as the **sum of all forwards** for the image
  (global + tiles), so throughput numbers stay honest.
- The Stretch preprocess option applies to the global pass only; tiles always run at native
  scale with bottom-right gray padding (`_pad_slice` semantics).

## Coming next

The same three features land in the shared C++ runtime (ONNX / ncnn / MNN / TensorRT) and the
Windows GUI runner.
