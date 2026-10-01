// Tiled inference for YOLOMasterKit - a faithful port of upstream YOLO-Master's Sparse SAHI
// Mode (ultralytics/engine/predictor.py:542-699, `_run_sparse_sahi_single`) plus a dense
// ("traditional tiling") variant that runs every grid tile unconditionally.
//
// Shape of the port (deviations from upstream are deliberate and documented):
//   * Upstream NMSes each tile at the caller's conf/iou, then merges with one more NMS.
//     Here NO per-tile NMS runs: the merged pool is PRE-NMS candidates (conf floor 0.05) from
//     the global pass + every executed tile, offset into full-image coordinates. The single
//     user-conf/iou NMS at render time subsumes per-tile NMS, keeps the cached pool independent
//     of the UI sliders (the app's cache-and-retune pattern), and hands CW-NMS the pre-NMS pool
//     its weighting wants.
//   * Tile detections are clipped to real crop content (via forwardPadded's origW/H = crop
//     dims); upstream lets boxes live on the gray padding.
//   * TILE detections are always boxes-only: their mask coefficients are meaningless against a
//     full-image proto tensor and are stripped. GLOBAL-pass detections keep coeffs (and the
//     proto tensor rides along in TiledOutput.globalRaw) only when keepGlobalMasks is set, so
//     seg masks can still render for what the global pass found.
//   * A maxTiles safety cap bounds the tiles RUN per image (row-major truncation, surfaced in
//     `TiledOutput.capped`) so a gigapixel input degrades gracefully instead of hanging.
import Foundation
import CoreGraphics

/// Tiled-inference mode. `.dense` = global pass + every grid tile ("traditional tiling").
/// `.sparse` = Sparse SAHI: global pass + only the tiles where the global pass found evidence
/// (objectness mask > threshold), with an all-or-nothing dense fallback when nothing is found.
public enum TilingMode: String, CaseIterable, Sendable {
    case off, dense, sparse
    public var label: String {
        switch self {
        case .off: return "Off"
        case .dense: return "Dense"
        case .sparse: return "Sparse SAHI"
        }
    }
}

/// Tiling parameters. The UI exposes `mode`, `tileSize` and `keepGlobalMasks`; the rest stays
/// at the upstream defaults (default.yaml:75-79) and is exposed here for the CLI/tests.
public struct TilingConfig: Sendable {
    public var mode: TilingMode = .off
    /// Requested tile edge in source pixels. nil = the model's imgsz (native scale, the
    /// upstream default). Clamped PER IMAGE to [imgsz, max(imgsz, min(w,h)/4)] - no smaller
    /// than the model input, no larger than a quarter of the image's short side; tiles larger
    /// than imgsz are letterboxed down to the model input (upstream slice_size semantics).
    public var tileSize: Int? = nil
    /// Keep the GLOBAL pass's mask coefficients (+ proto tensor) so segmentation masks still
    /// render for global-pass detections in tiled modes. Tile detections are always boxes-only:
    /// their coefficients are meaningless against a full-image proto tensor.
    public var keepGlobalMasks: Bool = false
    public var overlapRatio: CGFloat = 0.2        // overlap = Int(tile * ratio); upstream 0.2
    public var objectnessThreshold: Float = 0.15  // strict > gate on the painted mask
    public var fallback: Bool = true              // sparse only: zero active tiles -> run ALL
    public var maxTiles: Int = 256                // safety cap on tiles RUN per image
    public init(mode: TilingMode = .off, tileSize: Int? = nil, keepGlobalMasks: Bool = false) {
        self.mode = mode; self.tileSize = tileSize; self.keepGlobalMasks = keepGlobalMasks
    }
}

/// Aggregate tile statistics over one or many images (a folder run), for the stats card.
public struct TileStats: Sendable {
    public var tilesRun = 0
    public var tilesTotal = 0
    public var fallbacks = 0    // images where the sparse all-or-nothing fallback fired
    public var capped = 0       // images where the maxTiles cap truncated the run
    public var tileSizeMin = 0  // per-image clamping can vary the tile edge across a folder
    public var tileSizeMax = 0
    public init() {}
    public mutating func add(_ o: TiledOutput) {
        tilesRun += o.tilesRun; tilesTotal += o.tilesTotal
        fallbacks += o.usedFallback ? 1 : 0; capped += o.capped ? 1 : 0
        tileSizeMin = tileSizeMin == 0 ? o.tileSizeUsed : Swift.min(tileSizeMin, o.tileSizeUsed)
        tileSizeMax = Swift.max(tileSizeMax, o.tileSizeUsed)
    }
    /// "640" or "640-750" for the stats card.
    public var tileSizeLabel: String {
        tileSizeMin == tileSizeMax ? "\(tileSizeMin)" : "\(tileSizeMin)-\(tileSizeMax)"
    }
}

/// Result of a tiled inference: the merged candidate pool plus run statistics for the UI.
public struct TiledOutput: @unchecked Sendable {   // globalRaw is immutable after init
    public let candidates: [Detection]  // global + tile pool, sorted score desc (tile coeffs stripped)
    public let tilesTotal: Int          // grid tiles surviving the drop check
    public let tilesRun: Int
    public let tileSizeUsed: Int        // per-image clamped tile edge actually used
    public let usedFallback: Bool       // sparse: zero active tiles -> dense fallback fired
    public let capped: Bool             // maxTiles truncation hit
    public let inferMs: Double          // SUM of all model-only forwards (global + tiles)
    /// The global pass's raw output, retained ONLY when `keepGlobalMasks` on a seg model -
    /// hands the app the proto tensor so global-pass detections can still render masks.
    public let globalRaw: Detector.RawOutput?
}

extension Detector {
    /// Upstream's mask resolution divisor (predictor.py:556, `mask_scale = 8`). Fixed, not
    /// derived from slice size. Also drives the tile drop check.
    private static let maskScale = 8

    /// The user-facing tile-size bound: no smaller than the model input, no larger than a
    /// quarter of the image's short side. When the image is too small for that upper bound
    /// (shortSide/4 < imgsz) the floor wins and tiling runs at imgsz.
    public static func clampedTileSize(_ requested: Int?, imgsz: Int, width: Int, height: Int) -> Int {
        let ceiling = max(imgsz, min(width, height) / 4)
        return min(max(requested ?? imgsz, imgsz), ceiling)
    }

    /// The upstream slice grid (predictor.py:573-590), exactly: stride = sliceSize - overlap
    /// from 0 while < image dim; right/bottom tiles CLIPPED (not shifted back); a tile is
    /// dropped when integer division by the mask scale collapses it in either axis
    /// (`y/8 == yMax/8` - so a 7-px sliver at offset 1 survives, one at offset 0 doesn't).
    public static func tileGrid(width: Int, height: Int, sliceSize: Int, overlap: Int)
        -> [(x: Int, y: Int, w: Int, h: Int)] {
        let step = sliceSize - overlap
        guard step > 0, width > 0, height > 0 else { return [] }
        var tiles: [(x: Int, y: Int, w: Int, h: Int)] = []
        for y in stride(from: 0, to: height, by: step) {
            for x in stride(from: 0, to: width, by: step) {
                let yMax = min(y + sliceSize, height), xMax = min(x + sliceSize, width)
                let mY1 = y / maskScale, mY2 = yMax / maskScale
                let mX1 = x / maskScale, mX2 = xMax / maskScale
                guard mY2 > mY1 && mX2 > mX1 else { continue }
                tiles.append((x: x, y: y, w: xMax - x, h: yMax - y))
            }
        }
        return tiles
    }

    /// Global forward + (dense: all tiles | sparse: mask-selected tiles) -> merged pre-NMS pool.
    /// See the file header for the exact semantics. `config.mode` must not be `.off` - callers
    /// branch to the ordinary single-pass path before getting here.
    public func tiledCandidates(_ image: CGImage, config: TilingConfig,
                                confFloor: Float = 0.05) throws -> TiledOutput {
        precondition(config.mode != .off, "tiledCandidates called with mode .off")
        let w = image.width, h = image.height

        // ---- global pass (honors the current preprocess mode, like any single-pass run) ----
        var totalMs = 0.0
        let globalCands: [Detection]
        var globalRaw: Detector.RawOutput? = nil
        do {
            let raw = try forward(image)          // released at scope exit unless masks are kept
            totalMs += raw.inferMs
            globalCands = candidates(raw, confFloor: confFloor)
            if config.keepGlobalMasks && isSegment { globalRaw = raw }
        }

        let tile = Detector.clampedTileSize(config.tileSize, imgsz: imgsz, width: w, height: h)
        let overlap = Int(CGFloat(tile) * config.overlapRatio)
        let grid = Detector.tileGrid(width: w, height: h, sliceSize: tile, overlap: overlap)

        // ---- tile selection ----
        var selected: [(x: Int, y: Int, w: Int, h: Int)]
        var usedFallback = false
        switch config.mode {
        case .dense, .off:
            selected = grid
        case .sparse:
            // 1/8-scale objectness mask painted from the global pass (predictor.py:555-568).
            // Painted from floor-NMS'd candidates at a FIXED IoU (0.5) so tile selection is
            // independent of the UI conf/iou sliders; the 0.15 objectness gate does the work.
            let mH = h / Detector.maskScale + 1, mW = w / Detector.maskScale + 1
            var mask = [Float](repeating: 0, count: mH * mW)
            for d in Detector.nms(globalCands, conf: confFloor, iou: 0.5) {
                // Upstream: (xyxy / 8).astype(int) - truncation - then clamp to the mask dims.
                let x1 = max(0, Int(d.rect.minX) / Detector.maskScale)
                let y1 = max(0, Int(d.rect.minY) / Detector.maskScale)
                let x2 = min(mW, Int(d.rect.maxX) / Detector.maskScale)
                let y2 = min(mH, Int(d.rect.maxY) / Detector.maskScale)
                guard x2 > x1, y2 > y1 else { continue }   // sub-8px box paints nothing (upstream)
                for yy in y1..<y2 {
                    for xx in x1..<x2 {
                        mask[yy * mW + xx] = max(mask[yy * mW + xx], d.score)
                    }
                }
            }
            func tileMax(_ t: (x: Int, y: Int, w: Int, h: Int)) -> Float {
                let mY1 = t.y / Detector.maskScale, mY2 = (t.y + t.h) / Detector.maskScale
                let mX1 = t.x / Detector.maskScale, mX2 = (t.x + t.w) / Detector.maskScale
                var v: Float = 0
                for yy in mY1..<min(mY2, mH) {
                    for xx in mX1..<min(mX2, mW) { v = max(v, mask[yy * mW + xx]) }
                }
                return v
            }
            selected = grid.filter { tileMax($0) > config.objectnessThreshold }   // strict >
            // Upstream's all-or-nothing fallback (predictor.py:632-654): ONLY when zero tiles
            // are active, run the whole grid. If any tile is active, inactive ones are skipped.
            if selected.isEmpty && config.fallback && !grid.isEmpty {
                selected = grid
                usedFallback = true
            }
        }

        let capped = selected.count > config.maxTiles
        if capped { selected = Array(selected.prefix(config.maxTiles)) }

        // ---- run tiles sequentially, offset into full-image coords ----
        // Tile coeffs are ALWAYS stripped (meaningless vs the full-image proto). Global-pass
        // coeffs are kept only when the caller asked for masks (keepGlobalMasks).
        var pool: [Detection] = config.keepGlobalMasks && isSegment
            ? globalCands
            : globalCands.map { Detection(cls: $0.cls, score: $0.score, rect: $0.rect) }
        for t in selected {
            guard let crop = image.cropping(to: CGRect(x: t.x, y: t.y, width: t.w, height: t.h))
            else { continue }
            let raw = try forwardPadded(crop, tileSize: tile)
            totalMs += raw.inferMs
            for d in candidates(raw, confFloor: confFloor) {
                pool.append(Detection(cls: d.cls, score: d.score,
                                      rect: d.rect.offsetBy(dx: CGFloat(t.x), dy: CGFloat(t.y))))
            }
        }
        pool.sort { $0.score > $1.score }   // nms() requires score-desc input; concat isn't sorted

        return TiledOutput(candidates: pool, tilesTotal: grid.count, tilesRun: selected.count,
                           tileSizeUsed: tile, usedFallback: usedFallback, capped: capped,
                           inferMs: totalMs, globalRaw: globalRaw)
    }

    /// Convenience for the CLI: tiled forward + the selected NMS in one call.
    public func detectTiled(_ image: CGImage, conf: Float, iou iouT: CGFloat,
                            tiling: TilingConfig, nmsMode: NMSMode = .standard,
                            sigma: Float = 0.1, maxDet: Int = 300) throws -> (Result, TiledOutput) {
        let out = try tiledCandidates(image, config: tiling)
        let dets = Detector.nms(out.candidates, conf: conf, iou: iouT, maxDet: maxDet, mode: nmsMode, sigma: sigma)
        return (Result(detections: dets, inferMs: out.inferMs), out)
    }
}
