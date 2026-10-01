# Windows Runner (GUI + C++ CLI) - complete change log, v1.0.0 -> v1.1.0

Compiled from `git diff 581efe4..HEAD -- gui/ cpp/` (8 commits, +1,672 / -49 lines across
21 files), not from memory. Commit hashes cite the branch `gui/win-v1.1.0-tiled-inference`.
This release ports the Mac v1.1.0 feature set (mac/CHANGELOG-1.1.0.md sections 1, 3-7) to
the shared C++ runtime, the CLI and the Dear ImGui GUI, across the ONNX Runtime / ncnn /
MNN backends. The Mac video performance work (its section 2) is intentionally NOT ported:
the Windows video path is a different artifact (OpenCV VideoCapture + D3D11, no AVFoundation).

| file | delta |
|---|---|
| gui/src/app.cpp | +605 net (slicing/NMS/export cards, zoom, export threads) |
| cpp/src/annotate_writers.cpp | +195 (new) |
| cpp/src/annotate_export.cpp | +185 (new) |
| cpp/src/slicing.cpp | +138 (new) |
| cpp/src/main.cpp | +133 (CLI flags + slicing/export wiring) |
| cpp/include/{slicing,slicing_core,annotate,annotate_export}.hpp | +286 (new) |
| cpp/src/common.cpp | +79 net (CW-NMS + dynamic class offset) |
| gui/src/app.hpp | +50 (state + Platform::save_file) |
| gui/src/main_win.cpp | +17 (GetSaveFileNameA dialog) |
| CMakeLists (both), package scripts, about.hpp | build wiring + 1.1.0 |

---

## 1. New features

### 1.1 Sliced inference ("Slicing", fe7cb4e, 6b4983b)
Port of upstream YOLO-Master Sparse SAHI (predictor.py:542-699) plus a dense variant,
implemented once in the shared runtime (`sliced_candidates`, cpp/src/slicing.cpp) and used
by both the GUI and the CLI on all three backends.
- **Dense**: global full-image pass + every tile of a 20%-overlap grid.
- **Sparse SAHI**: the global pass paints a 1/8-scale objectness mask from its own
  detections (score-max composited, floor-NMS at fixed conf 0.05 / IoU 0.5 so selection is
  independent of the UI sliders and the NMS mode); only tiles whose mask max exceeds 0.15
  (strict) run; if NO tile qualifies, all tiles run (upstream's all-or-nothing fallback).
  Grid replicates upstream exactly: stride = tile - overlap, right/bottom tiles clipped,
  sub-8px-collapse drop rule.
- **Zero backend changes**: each tile crop is pre-padded onto a tile x tile gray-114 canvas,
  so the unmodified backend letterbox sees a square and scales uniformly by imgsz/tile -
  upstream _pad_slice semantics. ncnn's fixed input size and MNN's session shape never
  change across global + tile forwards.
- **Adjustable tile size**: bounded per image to [model imgsz, max(imgsz, shortSide/4)];
  larger tiles letterbox down (upstream slice_size semantics). GUI slider ceiling derives
  from the source (folder: max across cached images), commits on release; a too-small
  source disables the slider with a caption.
- **Masks in sliced modes, opt-in** ("Masks (global pass)"): the global pass keeps coeffs +
  proto so its detections still render masks; tile detections are always boxes-only (tile
  coeffs are meaningless against a full-image proto).
- Applies to images and folder batches; video and webcam stay single-pass (mode disabled
  with an "Images and folders only" caption). Mode/size/masks changes re-infer (like
  Preprocess); conf/IoU/NMS tuning stays cached-instant because the pooled candidates are
  cached pre-NMS at the 0.05 floor.
- 256-tiles-run safety cap; tiles run/grid, tile size (range over a folder), fallback and
  capped counts surfaced in the INFERENCE stats card; sliced "Model" time = SUM of forwards.
- Deliberate deviations from upstream (same as the Mac port): no per-tile NMS (one final
  NMS over the pooled pre-NMS candidates), tile detections clipped to real crop content.
- `sliced_candidates` rewrites the backend's cached candidates/letterbox/proto to describe
  the merged run (documented postcondition), so every existing consumer - GUI re-NMS,
  folder cache snapshot, CLI decode - works unchanged.

### 1.2 Cluster-Weighted NMS (0ce2530, 6b4983b)
`NmsMode::ClusterWeighted` + `cw_sigma` on the shared `Config` (GUI: NMS picker + sigma
slider 0.01-0.5 in the DETECTION card; CLI: `--cw-nms --sigma`): after standard greedy
selection inside `nms_and_cap`, each survivor's coordinates are rewritten as the
score-and-proximity-weighted average of its same-class pre-NMS cluster
(`w = score * exp(-(1-IoU)^2/sigma)`, pool = conf-filtered candidates capped 3000, IoU on
the class-offset boxes so cross-class terms are zero). Survivor count/order/scores/classes
unchanged. Applies everywhere including the sliced merge and the live webcam; toggling is
a cheap re-NMS on the cached candidates. Guard against upstream's latent collapse-to-origin
bug (zero-weight survivor keeps its box). Verified against upstream torch
`non_max_suppression(cluster=True)`: survivor sets identical, coordinates match to float32
box-storage noise (~1e-2 px at 4000 px coordinates).

Also in this commit: the per-class NMS offset (was a fixed 8192) is now
`2*max(orig_w, orig_h) + 8192`, since sliced originals can exceed 8192 px. Standard-mode
output verified bit-identical across the change on visdrone50.

### 1.3 Zoom and pan (6b4983b, fdb5dd6)
Images always, video when paused, never the live webcam: cursor-anchored mouse-wheel zoom
1x-8x (the pixel under the cursor stays put), click-drag pan when zoomed (clamped so
content never leaves the frame), Ctrl+= / Ctrl+- center-anchored steps, Ctrl+0 reset,
snap-home below ~102%, zoom% badge ("N% - Ctrl+0 to reset"). Boxes and the seg-mask
overlay follow for free because the preview draws everything from one origin/scale pair;
content clips to the preview panel. Auto-reset on new source, folder browse, model reload,
webcam start and video play. The preview child takes `NoScrollWithMouse` so the wheel zooms
instead of scrolling; input lands on an InvisibleButton over the image area.

### 1.4 Annotation export (bb7df47, 6b4983b, 2753678)
New EXPORT card (images, folders, videos; hidden for the webcam), WYSIWYG at the current
conf/IoU/NMS/sigma:
- **Labels**: YOLO TXT / COCO JSON / Pascal VOC XML.
  - image -> one file via the new save dialog (+ sibling classes.txt for YOLO);
  - folder -> a labels folder built entirely from the candidate cache (no re-forward);
  - video -> frames/ (extracted JPEGs, source-frame-indexed `<stem>_%06d.jpg`) + labels/,
    with sampling (every frame / one per second / every 5th/10th/30th). Per-frame protos
    are cached during pre-inference, so video seg export needs no re-inference either.
- **Rendered**: image -> "Save image..."; folder -> "This image" / "All..." (JPEGs);
  video -> "This frame" / "All (annotated video)..." (mp4 via cv::VideoWriter). Rendered
  files compose CPU-side with the shared draw() style and honor the Overlay mode
  (both / masks / boxes).
- Bulk exports create an enclosed source-named subfolder (`<source>-labels` /
  `<source>-rendered`) inside the picked directory, suffixing -2/-3 instead of merging
  into an existing non-empty folder (2753678; the Windows folder picker selects an
  existing directory, unlike the Mac save panel's name-a-folder idiom).
- Seg models export real polygons: `seg_polygons` traces instance masks over the proto
  grid (threshold at the preview's smoothstep center, cell centers clipped to the box,
  cv::findContours external contours + approxPolyDP at 2 px, area-sorted, cap 8).
  Coeff-less instances fall back to the box as a 4-point polygon; a seg model sliced
  without kept masks exports det-dialect (matches the preview). COCO: one json per set,
  category_id = cls+1, shoelace areas, score kept as an extension key, video image ids =
  source frame index + 1; VOC: boxes only, 1-based ints, no score element. Empty results
  still write (valid-negative convention). Writers are hand-emitted (no JSON/XML library
  added; binary streams, LF endings).
- Exports run on a background thread with a progress bar and Cancel; the sidebar locks
  model/source/preprocess/slicing while an export reads the caches (conf/IoU stay live;
  the export snapshots settings at click time). Windows gains `Platform::save_file`
  (GetSaveFileNameA, overwrite prompt, default-extension handling) in main_win.cpp.
- Writer output validated with real parsers (pycocotools incl. annToMask round-trip,
  ElementTree, YOLO round-trip against --save-txt).

## 2. Shared runtime changes (cpp/, compiled into both CLI and GUI)

New public surface:
- `NmsMode`, `Config::nms_mode` / `Config::cw_sigma` (defaulted - existing call sites
  byte-compatible); reworked `nms_and_cap` (dynamic offset + CW post-pass, shared
  `box_iou` helper).
- `slicing_core.hpp` (header-only, OpenCV-free): `SliceMode`, `TileRect`,
  `clamped_tile_size`, `tile_grid`, `TileStats`.
- `slicing.hpp/.cpp`: `SliceConfig`, `SliceOutput`, `sliced_candidates(Backend&, ...)`
  with a cancel token for background folder runs.
- `annotate.hpp` + `annotate_writers.cpp` (OpenCV-free): `annot::Format`, IR
  (`annot::Instance` / `annot::Image`), pure writers `yolo_lines` / `classes_txt` /
  `voc_xml` / `coco_json`.
- `annotate_export.hpp/.cpp`: `seg_polygons`, `annotation_instances`, streaming
  `AnnotationSink` (shared by the CLI and both GUI export loops), `write_jpg` (stb,
  moved from main.cpp's private helper).
- All new .cpp files are added to BOTH cpp/CMakeLists.txt and gui/CMakeLists.txt (the GUI
  compiles the runtime sources directly); the GUI defines HAVE_VIDEOIO.
- The PORTABLE CLI build (core+imgproc only) keeps working: slicing and image/folder label
  export need nothing beyond imgproc; video paths stay behind HAVE_VIDEOIO.

## 3. CLI (yolomaster_edge, 1d971f8)

- `--slicing off|dense|sparse`, `--tile-size N`, `--slicing-masks`, `--cw-nms`,
  `--sigma S`. Per-image log gains `tiles=R/T @Npx [fallback] [capped]`; summary gains
  `nms=cw(sigma=S)` and a `[slicing]` line (mode, tiles, size range, fallbacks, capped).
  No `--tiling` compat aliases (the C++ CLI never shipped them).
- `--export-labels DIR`, `--label-format yolo|coco|voc`, `--sampling all|1s|N` (video:
  DIR/frames/ + DIR/labels/, COCO ids = frame index + 1, 1s = round(fps)). Summary gains a
  `[labels]` line (format, images, instances, destination). The label sink is created
  after the first forward, so the file dialect follows what the run actually produced.
- Slicing on a video source warns and runs single-pass. Slicing on the TensorRT backend
  warns and runs single-pass (TRT never populates the cached candidates the slicer pools).

## 4. UX / copy changes (6b4983b, b409e34, fdb5dd6)

- New SLICING card between PREPROCESS and DETECTION ("Slicing" naming throughout - the
  feature never shipped here under the "Tiling" name).
- "IoU (NMS)" -> "IoU" in the DETECTION card.
- Stats card additions: Slicing mode, Tile size (range over folders), Tiles run/grid,
  fallback and capped counts.
- Double-click zoom reset removed (both runners); reset is Ctrl+0 / snap-home (fdb5dd6).
- Em/en-dash pass: remaining dashes in cpp/gui comments replaced with ASCII hyphens; all
  new user-facing strings are dash-free (b409e34).
- Detection settings stay LIVE during video playback (unlike the Mac runner): the Windows
  playback overlay re-NMSes the shown frame from cache per frame, which is cheap; the Mac
  freeze existed for its whole-video bake, which is not part of this port.

## 5. Packaging / version (b409e34)

- about.hpp kAppVersion 1.0.0 -> 1.1.0.
- Default version 1.0.0 -> 1.1.0 in package.cmd, package-cuda.cmd and package.ps1.

## 6. Verification (Linux-side, shared runtime + CLI)

- Existing robustness battery (cpp/run_tests.sh): 16/16 on the rebuilt binary.
- Standard-NMS output bit-identical before/after the dynamic-offset change (50 images).
- Grid parity vs a transcription of upstream predictor.py: 4,608 grids + 72 clamps, zero
  mismatches.
- CW-NMS vs torchvision-greedy + upstream cluster refinement: identical survivor sets,
  coords to float32 storage noise.
- Sparse painting/gating/fallback/cap/clip/postcondition: 9 scripted stub-backend
  scenarios vs a Python simulation, all pass.
- Writers: pycocotools (det + seg, masks rasterize), ElementTree bounds checks, YOLO
  round-trip; dialect rules confirmed (seg 13-field lines, sliced-without-masks degrades
  to 5-field det lines, sliced-with-masks 9-field).
- End-to-end CLI on visdrone50 matches the Mac runner's smoke: dense 12/12 tiles, sparse
  6/12, --tile-size 900 clamps to 640 on 765-px-short-side sources.
- The GUI itself is compile-verified on Linux against the vendored ImGui only; functional
  GUI testing is on Windows.

## Known behavior notes (unchanged or intentional)

- Mac v1.1.0 video performance work (pipelined decode, playback compositor, Accelerate
  mask batching, bake machinery) is NOT ported; the Windows playback path is unchanged
  from 1.0.0 (per-frame decode + cached-candidate re-NMS).
- Live webcam is out of scope for slicing/zoom/export.
- Sliced-mode masks are global-pass only by design; tile instances stay boxes.
- TensorRT (the Jetson artifact, not shipped in the Windows GUI) does not support
  slicing or label export; the CLI degrades to single-pass with a warning.
- Rendered exports use the shared draw() style; the GUI's hud/neon box styles are
  screen-only draw-list effects.
- Sampling "every frame" on long videos writes tens of GBs of JPEGs; the default is 1/s.
