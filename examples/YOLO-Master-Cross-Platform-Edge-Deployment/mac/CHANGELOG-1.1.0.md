# macOS Core ML Runner - complete change log, v1.0.0 -> v1.1.0

Compiled from `git diff v1.0.0-macos..HEAD -- mac/` (22 commits, +2,152 / -456 lines across
19 files), not from memory. Commit hashes cite the branch `gui/mac-v1.1.0-tiled-inference`.

| file | delta |
|---|---|
| Sources/YOLOMasterApp/YOLOMasterApp.swift | +917 net (rewired) |
| Sources/YOLOMasterKit/Detector.swift | +422 |
| Sources/YOLOMasterKit/AnnotationExport.swift | +369 (new) |
| Sources/YOLOMasterKit/Tiling.swift | +228 (new) |
| Sources/YOLOMasterApp/Zoom.swift | +137 (new) |
| Sources/YOLOMasterKit/Pipeline.swift | +111 |
| Sources/YOLOMasterCoreML/main.swift | +36 |
| Info.swift / Camera.swift / License.swift / Annotate.swift | threading + copy passes |
| Resources/AppIcon.{png,icns} | replaced |
| RELEASE_NOTES-1.1.0.md | new (1.0.0 notes, README, DISTRIBUTING removed from main earlier, faaf946) |

---

## 1. New features

### 1.1 Sliced inference ("Slicing", e366f47, 4da4ea5, ca3a221)
Port of upstream YOLO-Master Sparse SAHI (predictor.py:542-699) plus a dense variant.
- **Dense**: global full-image pass + every tile of a 20%-overlap grid.
- **Sparse SAHI**: the global pass paints a 1/8-scale objectness mask from its own detections
  (score-max composited); only tiles whose mask max exceeds 0.15 (strict) run; if NO tile
  qualifies, all tiles run (upstream's all-or-nothing fallback). Grid replicates upstream
  exactly: stride = tile - overlap, right/bottom tiles clipped, sub-8px-collapse drop rule.
- **Adjustable tile size** (`TilingConfig.tileSize`): bounded per image to
  [model imgsz, max(imgsz, shortSide/4)]; larger tiles letterbox down (upstream slice_size
  semantics); slider ceiling derived from the source (folder: max across images), commits on
  release; stats show the size actually used (a range when per-image clamping varies).
- **Masks in sliced modes, opt-in** ("Masks (global pass)", `keepGlobalMasks`): the global
  pass keeps coeffs + proto so its detections render masks; tile detections are always
  boxes-only (tile coeffs are meaningless against a full-image proto).
- Applies to images and folder batches only; video/webcam stay single-pass. Mode/size/masks
  changes re-infer (like Preprocess); conf/IoU/NMS tuning stays cached-instant.
- 256-tiles-run safety cap; run/grid/fallback/capped surfaced in the stats card.
- Deliberate deviations from upstream (documented in RELEASE_NOTES): no per-tile NMS (one
  final NMS over the pooled pre-NMS candidates), tile detections clipped to real crop
  content, pad-to-square via new `Detector.forwardPadded` at native scale.
- Verified against upstream Python directly: tile grid exact-matched on 20 image sizes;
  see also 1.2.

### 1.2 Cluster-Weighted NMS (e366f47)
`NMSMode.clusterWeighted`, a global NMS option (sidebar picker + sigma slider 0.01-0.5,
CLI `--cw-nms --sigma`): after standard greedy selection, each survivor's coordinates are
rewritten as the score-and-proximity-weighted average of its same-class pre-NMS cluster
(`w = score * exp(-(1-IoU)^2/sigma)`, pool = score>conf capped 3000). Survivor count/scores/
classes unchanged. Applies everywhere including the sliced merge and live camera. Guard added
against upstream's latent collapse-to-origin bug (zero-weight survivor keeps its box).
Verified against upstream torch `non_max_suppression(cluster=True)` to 3.2e-5.

### 1.3 Zoom (20c0ec4, 3868e84; new Zoom.swift)
Images always, video when paused: cursor-anchored pinch 1x-8x, click-drag pan when zoomed,
cmd-0 reset, cmd-+/- steps, zoom% badge. Video wraps the whole
player+overlay stack so boxes zoom with pixels; auto-reset on play/browse/re-run/new source.
Gestures live on a transparent layer ABOVE the content because the AVPlayerLayer NSView
swallows mouse events before SwiftUI ancestor gestures fire (3868e84).

### 1.4 Annotation export (20c0ec4, 3f50f60, 3868e84; new AnnotationExport.swift)
"Export labels" for YOLO TXT / COCO JSON / Pascal VOC XML, WYSIWYG at current
conf/IoU/NMS/sigma:
- image -> one file via save panel (+ sibling classes.txt for YOLO); folder -> a labels
  folder; video -> frames/ (extracted JPEGs, source-frame-indexed) + labels/, with sampling
  (every frame / 1 per second / every 5th/10th/30th) picked in the scrubber bar.
- Folder/video destination chosen via save panel (name-a-new-folder idiom).
- Seg models export real polygons: new `Detector.maskPolygons` traces instance masks
  (threshold at the preview's smoothstep center, clip to box, 8-connected components,
  Moore boundary trace, Douglas-Peucker 2px, outer contours only, area-sorted, cap 8).
  Coeff-less instances fall back to the box as a 4-point polygon; a seg model sliced
  without kept masks exports det-dialect (matches the preview). COCO: one json per set,
  category_id = cls+1, shoelace areas, score kept as an extension key; VOC: boxes only,
  1-based ints, no score element. Empty results still write (verified-negative convention).
- Writer output validated with real parsers (pycocotools incl. annToMask round-trip,
  ElementTree, YOLO round-trip); tracer validated on 8 synthetic grids.

### 1.5 New app icon + dev-run branding (4e56f06, af41fdb)
`Resources/AppIcon.png` replaced (new 1024x1024 artwork), icns rebuilt via the existing
`scripts/make_icon.py` (11 PNG-encoded sizes, container verified). Under `swift run` (no
bundle) the About-page logos, author avatar and sidebar app mark now load from the checkout
via a `#filePath`-derived fallback; packaged apps unchanged (bundle path wins).

## 2. Performance (video)

### 2.1 Inference throughput (e9e1b8a, f27c795)
`inferVideo` was strictly serial per frame (decode -> forward -> candidate extraction).
Now a 3-stage pipeline: frame n+1 decodes while frame n forwards; candidates run on a
detached task overlapping the next forward (in-flight bound ~3 keeps memory flat; one decode
task at a time so the AVAssetReader is never touched concurrently). Upright videos decode via
a single-memcpy CGImage wrap instead of a full CIContext color-managed render (rotated tracks
keep the CI path). Measured on-device: ~30 -> ~50+ overall fps mid-series, further after the
decode fix; model-only was ~160.

### 2.2 Playback overlay: the webcam architecture (ea8d220; enablers b821b37, e9e1b8a, f27c795, 769a985)
v1.0.0 drew the overlay by chasing the player clock through SwiftUI per tick (retrieval +
NMS + hundreds of box paths + label resolution on the main thread). Final architecture: while
playing, a self-paced worker follows the player clock and composes boxes+labels+masks for the
frame under the playhead into ONE transparent full-res CGImage off-main (latest-frame-wins,
generation-token teardown); the Canvas blits that single image - the same cost as the camera
preview overlay. Paused/scrub composes one frame at full detail. Boxes/labels/masks are now
composed from the SAME frame by construction, eliminating the box/mask desync class of bug.
Enablers retained along the way:
- Detection settings freeze during playback (sidebar disables them) -> whole-video post-NMS
  bake becomes legal; per-frame single-slot NMS cache; sorted-input early break in
  `Detector.nms` (f27c795, b821b37, e9e1b8a).
- Periodic time observer 30 Hz -> 10 Hz (it only feeds the scrubber + stats/paused-compose).
- Stats card throttled to ~4 Hz during playback (full rate paused).

### 2.3 Mask math on the AMX matrix units (769a985)
`maskOverlay` computed a scalar coeffs.proto matmul PER DETECTION, re-reading (fp16:
re-converting) the whole proto tensor each time - seconds at low-conf detection counts.
Now: proto unpacked to contiguous Float32 once, one `cblas_sgemm` ([N,32]x[32,25600]) for all
masks, sigmoid vectorized (vDSP/vvexpf/vvrecf); per-mask work is the tint pass + GPU-backed
CG composite. Identical visual output.

### 2.4 Cross-run degradation fixes (b45ace9)
Re-inferring made the app progressively laggy. Two accumulators removed:
- Superseded whole-video NMS bakes ran to completion (every settings tick spawned one) ->
  bakes now carry an OSAllocatedUnfairLock cancellation token and stop at the next frame.
- Re-inferring a seg video held BOTH generations of per-frame tensors (GBs) -> `runVideo`
  releases the previous run's caches up front, and new `RawOutput.maskOnly()` halves
  per-frame retention (keeps proto + geometry, drops the detection tensor y).

## 3. Bug fixes

- **SwiftUI type-checker failures on build** (f853091, c702032, 5a8d666): the grown modifier
  chain exceeded the one-expression budget; body staged into typed sub-properties
  (mainStage -> tuningObservers -> sourceObservers -> keyboardLayer) and the video-overlay
  call pair extracted. (c702032 itself clobbered 586 lines via a bad edit anchor; restored
  and re-applied correctly in 5a8d666 - both commits are in history.)
- **Public IR types had internal memberwise inits** (3f50f60): AnnotationInstance /
  AnnotatedImage / AnnotationExportResult got explicit public inits (app target could not
  construct them).
- **Swift-6 concurrency warnings** (3f50f60): pixelSize was MainActor-isolated but called
  from Task.detached -> file-scope free function; mutable capture across a MainActor hop
  replaced with an immutable let.
- **Zoom pan dead over video** (3868e84): AVPlayerLayer's NSView swallowed drags; gestures
  moved to an overlay layer.
- **Playback overlay stuck at ~5 fps** (b821b37 partial, e9e1b8a): triple NMS per tick +
  30 Hz sidebar diffs; and a TimelineView whose context was never read - SwiftUI registered
  no schedule dependency, so the "vsync canvas" of 2ad87a6 silently never fired. Superseded
  by the 2.2 architecture.
- **Masks trailing/decoupled from boxes during playback** (2ad87a6 regression, fixed ea8d220):
  a fixed 4 Hz mask throttle showed masks up to 0.25 s stale; latest-wins coalescing, then the
  single-compositor architecture removed the desync class entirely.
- **Export menus could not match button width** (dcfa041): macOS Menu hugs its label
  regardless of frame(maxWidth:); replaced with real full-width buttons opening popovers.
- **Finder arrow keys drifted diagonally** (ca3a221): the icon grid used ADAPTIVE columns
  while the key handler estimated the count with a different formula; one shared
  `finderCols()` now drives both layout (fixed columns) and navigation.
- **NMS scanned the whole candidate array** (e9e1b8a): sorted-input early break at the conf
  threshold.

## 4. UX / copy changes

- "Tiling" -> **"Slicing"** across UI, stats, captions, release notes; CLI flags renamed
  `--slicing` / `--slicing-masks` (old `--tiling*` accepted as silent aliases) (ca3a221).
- "IoU (NMS)" -> "IoU"; dense-mode caption removed; too-small caption reworded to "The input
  dimensions are too small for larger tiles. Slicing runs at the model input (N px)." (ca3a221).
- Folder "Save image"+"Export all" -> one "Export rendered images" (This frame / All);
  video "Save frame"+"Export video" -> "Export rendered" (This frame / All (annotated
  video)) (dcfa041, 2ad87a6).
- Finder: selection auto-scrolls into view (both modes); icon view uses Finder-exact arrows -
  left/right within the current row only, up/down within the current column (ca3a221).
- Detection settings disabled (with caption) while an inferred video plays (f27c795).
- Every em/en-dash removed from sources + release notes; verified zero Unicode
  dash-punctuation codepoints remain (b9d8347).
- Stats card additions: Slicing mode, tile size (range), tiles run/grid, fallback count,
  NMS mode; "Model-only" for sliced runs = per-image SUM of forwards (documented).

## 5. Kit API changes (YOLOMasterKit)

New public surface:
- `NMSMode`; `Detector.nms(_:conf:iou:maxDet:mode:sigma:)` (defaulted - existing call sites
  byte-compatible); `decode`/`detect` gain mode/sigma.
- `TilingMode`, `TilingConfig`, `TiledOutput`, `TileStats`; `Detector.tileGrid`,
  `Detector.clampedTileSize`, `Detector.tiledCandidates`, `Detector.detectTiled`,
  `Detector.forwardPadded` (internal), `RawOutput.maskOnly()`.
- `Detector.maskPolygons`; `AnnotationFormat`, `VideoSampling`, `AnnotationInstance`,
  `AnnotatedImage`, `AnnotationExportResult`, `annotationInstances`, `AnnotationWriter`
  (yoloLines/classesTXT/vocXML/cocoJSON), `exportAnnotationsFolder`, `exportAnnotationsVideo`.
- `FolderItem` gains `width`/`height` (+ explicit public init); `inferFolder` returns
  `(items, summary, tileStats?)` and takes `tiling:`; `runFolder`/`runVideo`/
  `exportFolderCached`/`exportVideoCached` gain mode/sigma (and tiling where applicable).
- `maskOverlay` rewritten (Accelerate-batched; same signature/output). First Accelerate,
  Codable/JSONEncoder and os.OSAllocatedUnfairLock usages in the package.

## 6. CLI (yolomaster-coreml)

`--slicing off|dense|sparse`, `--tile-size N`, `--slicing-masks`, `--cw-nms`, `--sigma S`
(+ `--tiling*` compat aliases). `[det]` line prints `tiles=R/T @Npx (fallback)(capped)`.
Slicing on a video source warns and runs single-pass.

## 7. Packaging / docs

- Version 1.0.0 -> 1.1.0 in `make_app.sh`, `scripts/release.sh`, Info.swift fallback.
- `RELEASE_NOTES-1.1.0.md` added (features, deviations from upstream, format details).
- mac/README.md, DISTRIBUTING.md, RELEASE_NOTES-1.0.0.md were removed from the repo on main
  before this branch (faaf946) - not part of the v1.1.0 work but present in the diff.

## Known behavior notes (unchanged or intentional)

- Live camera is out of scope for slicing/zoom (unchanged from 1.0.0 behavior).
- Sliced-mode masks are global-pass only by design; tile instances stay boxes.
- The video inference "overall" fps remains below model-only fps (serial reader + JPEG-free
  decode still cost per frame); the gap narrowed substantially but is not zero.
- Sampling "every frame" on long videos writes tens of GBs of JPEGs; the default is 1/s.
