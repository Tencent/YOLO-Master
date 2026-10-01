// Folder-batch and video pipelines - shared by the CLI and the GUI so both process
// images / folders / videos through the same Detector + annotate path. Progress callbacks
// let a GUI drive a progress bar and live preview.
import Foundation
import CoreGraphics
import AVFoundation
import CoreImage
import CoreVideo
import ImageIO   // CGImagePropertyOrientation

public enum PipelineError: Error { case noVideoTrack, readerInit, writerInit }

public let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp"]
public let videoExtensions: Set<String> = ["mp4", "mov", "m4v", "avi"]

public enum SourceKind: Sendable { case image, folder, video, unknown }

/// image / folder / video / unknown from a URL on disk.
public func classifySource(_ url: URL) -> SourceKind {
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return .unknown }
    if isDir.boolValue { return .folder }
    let ext = url.pathExtension.lowercased()
    if videoExtensions.contains(ext) { return .video }
    if imageExtensions.contains(ext) { return .image }
    return .unknown
}

public func listImages(_ dir: URL) -> [URL] {
    ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
        .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

/// Visible (non-hidden) entries in `dir` that are NOT images - subfolders or other file types.
/// A non-empty result means the folder is "mixed" and should be rejected for batch inference.
public func folderNonImages(_ dir: URL) -> [URL] {
    ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
        .filter { !$0.lastPathComponent.hasPrefix(".") }                       // ignore .DS_Store & friends
        .filter { !imageExtensions.contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

// ---- source resize (output-resolution knob; independent of the model's fixed inference size) ----
public func resizeExact(_ image: CGImage, _ w: Int, _ h: Int) -> CGImage {
    guard w > 0, h > 0, (w != image.width || h != image.height),
          let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return image }
    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage() ?? image
}
public func fitLongSide(_ w: Int, _ h: Int, _ longSide: Int) -> (Int, Int) {
    if longSide <= 0 || max(w, h) == 0 { return (w, h) }
    let s = Double(longSide) / Double(max(w, h))
    return (max(1, Int((Double(w) * s).rounded())), max(1, Int((Double(h) * s).rounded())))
}
public func resizeLong(_ image: CGImage, _ longSide: Int) -> CGImage {
    let (nw, nh) = fitLongSide(image.width, image.height, longSide)
    return resizeExact(image, nw, nh)
}


/// "stem", or "stem-2", "stem-3", ... if already used; inserts the result. Guards every
/// per-source-file writer: two sources with the same stem (1.jpg + 1.png in one folder)
/// must never overwrite each other's output.
public func uniqueStem(_ used: inout Set<String>, _ stem: String) -> String {
    if used.insert(stem).inserted { return stem }
    var i = 2
    while !used.insert("\(stem)-\(i)").inserted { i += 1 }
    return "\(stem)-\(i)"
}

public struct BatchStats: Sendable { public let processed: Int; public let total: Int; public let meanMs: Double }
public struct VideoStats: Sendable {
    public let frames: Int; public let meanMs: Double
    public let outW: Int; public let outH: Int; public let fps: Int
}

/// Run every image in `input`, annotate with the given params, write to `output` (nil = don't save).
/// `progress(done, total, lastAnnotated)` fires per image; `onResult(source, image, result)` fires
/// before annotation with the per-stage timings (the CLI's txt dumps and bench samples hook here).
/// `limit` > 0 stops after that many images (the Linux `--limit`).
@discardableResult
public func runFolder(_ det: Detector, input: URL, output: URL?,
                      conf: Float, iou: CGFloat, style: BoxStyle, label: LabelMode, resize: Int = 0,
                      tiling: TilingConfig = TilingConfig(), nmsMode: NMSMode = .standard, sigma: Float = 0.1, maxDet: Int = 300,
                      limit: Int = 0, annotateImages: Bool = true,
                      onResult: ((_ source: URL, _ image: CGImage, _ result: Detector.Result) -> Void)? = nil,
                      progress: ((_ done: Int, _ total: Int, _ lastAnnotated: CGImage?) -> Void)? = nil) -> BatchStats {
    var files = listImages(input)
    if limit > 0 && files.count > limit { files = Array(files.prefix(limit)) }
    if let output { try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true) }
    var times: [Double] = []
    var usedStems = Set<String>()
    for (i, src) in files.enumerated() {
        guard var cg = loadCGImage(src) else { continue }
        if resize > 0 { cg = resizeLong(cg, resize) }
        let res: Detector.Result
        if tiling.mode == .off {
            guard let r = try? det.detect(cg, conf: conf, iou: iou, mode: nmsMode, sigma: sigma, maxDet: maxDet) else { continue }
            res = r
        } else {
            guard let (r, _) = try? det.detectTiled(cg, conf: conf, iou: iou, tiling: tiling,
                                                    nmsMode: nmsMode, sigma: sigma) else { continue }
            res = r
        }
        times.append(res.inferMs)
        onResult?(src, cg, res)
        let annotated = (annotateImages || output != nil) ? annotate(cg, res.detections, names: det.classNames, style: style, label: label) : nil
        if let output, let a = annotated {
            let stem = uniqueStem(&usedStems, src.deletingPathExtension().lastPathComponent)
            saveCGImage(a, to: output.appendingPathComponent(stem + ".jpg"))
        }
        progress?(i + 1, files.count, annotated)
    }
    let steady = times.count > 1 ? times[1...].reduce(0, +) / Double(times.count - 1) : (times.first ?? 0)
    return BatchStats(processed: times.count, total: files.count, meanMs: steady)
}

/// Decode `input` video, detect+annotate each frame, encode to `output` (h264, size/fps preserved;
/// nil = detect only, no file). `progress(frames, lastAnnotated)` fires periodically. With a `tracker`
/// every frame's detections are replaced by the tracker's confirmed tracks (ids in `Detection.trackId`)
/// before annotation; `onResult(frameIndex, image, result)` sees the tracked result.
@discardableResult
public func runVideo(_ det: Detector, input: URL, output: URL?,
                     conf: Float, iou: CGFloat, style: BoxStyle, label: LabelMode, resize: Int = 0,
                     nmsMode: NMSMode = .standard, sigma: Float = 0.1, maxDet: Int = 300,
                     tracker: Tracker? = nil, motion: CameraMotionEstimator? = nil,
                     onResult: ((_ frameIndex: Int, _ image: CGImage, _ result: Detector.Result) -> Void)? = nil,
                     progress: ((_ frames: Int, _ lastAnnotated: CGImage?) -> Void)? = nil) async throws -> VideoStats {
    let asset = AVURLAsset(url: input)
    guard let tracks = try? await asset.loadTracks(withMediaType: .video), let track = tracks.first else {
        throw PipelineError.noVideoTrack
    }
    let nominalFps = (try? await track.load(.nominalFrameRate)) ?? 0
    let fps = nominalFps > 0 ? nominalFps : 30
    let transform = (try? await track.load(.preferredTransform)) ?? .identity
    let orient = videoOrientation(transform)
    let naturalSize = (try? await track.load(.naturalSize)) ?? .zero
    let swap = (orient == .right || orient == .left)
    let natW = Int((swap ? naturalSize.height : naturalSize.width).rounded())
    let natH = Int((swap ? naturalSize.width : naturalSize.height).rounded())
    let (outW, outH) = resize > 0 ? fitLongSide(natW, natH, resize) : (natW, natH)

    guard let reader = try? AVAssetReader(asset: asset) else { throw PipelineError.readerInit }
    let rout = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    rout.alwaysCopiesSampleData = false
    reader.add(rout)

    var writer: AVAssetWriter? = nil
    var winput: AVAssetWriterInput? = nil
    var adaptor: AVAssetWriterInputPixelBufferAdaptor? = nil
    if let output {
        try? FileManager.default.removeItem(at: output)
        let ftype: AVFileType = output.pathExtension.lowercased() == "mov" ? .mov : .mp4
        guard let w = try? AVAssetWriter(outputURL: output, fileType: ftype) else { throw PipelineError.writerInit }
        let wi = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: outW, AVVideoHeightKey: outH])
        wi.expectsMediaDataInRealTime = false
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: wi, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outW, kCVPixelBufferHeightKey as String: outH])
        w.add(wi)
        writer = w; winput = wi
    }
    reader.startReading()
    if let writer { writer.startWriting(); writer.startSession(atSourceTime: .zero) }

    let cictx = CIContext()
    var n = 0, times: [Double] = []
    while reader.status == .reading {
        guard let sb = rout.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sb) else { break }
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        let ci = CIImage(cvPixelBuffer: pb).oriented(orient)   // upright, matching AVPlayer
        guard var cg = cictx.createCGImage(ci, from: ci.extent) else { continue }
        if resize > 0 { cg = resizeExact(cg, outW, outH) }
        guard var res = try? det.detect(cg, conf: conf, iou: iou, mode: nmsMode, sigma: sigma, maxDet: maxDet) else { continue }
        if let tracker {
            // frames arrive in order here (sequential reader), which is what the tracker requires
            let t0 = Date()
            let m = motion?.estimate(cg)
            let tracked = tracker.update(res.detections, motion: m)
            res = Detector.Result(detections: tracked, inferMs: res.inferMs, preMs: res.preMs,
                                  postMs: res.postMs + Date().timeIntervalSince(t0) * 1000)
        }
        times.append(res.inferMs)
        onResult?(n, cg, res)
        guard let adaptor, let winput else { n += 1; if n % 10 == 0 { progress?(n, nil) }; continue }
        guard let annotated = annotate(cg, res.detections, names: det.classNames, style: style, label: label)
        else { continue }
        guard let pool = adaptor.pixelBufferPool else { continue }
        var opb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &opb)
        guard let dst = opb else { continue }
        CVPixelBufferLockBaseAddress(dst, [])
        if let base = CVPixelBufferGetBaseAddress(dst),
           let c = CGContext(data: base, width: outW, height: outH, bitsPerComponent: 8,
                             bytesPerRow: CVPixelBufferGetBytesPerRow(dst), space: CGColorSpaceCreateDeviceRGB(),
                             bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) {
            c.draw(annotated, in: CGRect(x: 0, y: 0, width: outW, height: outH))
        }
        CVPixelBufferUnlockBaseAddress(dst, [])
        while !winput.isReadyForMoreMediaData { usleep(2000) }
        adaptor.append(dst, withPresentationTime: pts)
        n += 1
        if n % 10 == 0 { progress?(n, annotated) }
    }
    if let winput, let writer { winput.markAsFinished(); await writer.finishWriting() }
    let mean = times.count > 1 ? times[1...].reduce(0, +) / Double(times.count - 1) : (times.first ?? 0)
    return VideoStats(frames: n, meanMs: mean, outW: outW, outH: outH, fps: Int(fps.rounded()))
}

/// Video duration in seconds (0 if unknown) - for a scrubber range.
public func videoDuration(_ url: URL) async -> Double {
    let d = (try? await AVURLAsset(url: url).load(.duration)) ?? .zero
    return d.seconds.isFinite ? d.seconds : 0
}

/// Decode a single upright frame at `atSeconds` - for the in-app preview/tune finder.
public func extractFrame(_ url: URL, atSeconds t: Double) async -> CGImage? {
    let gen = AVAssetImageGenerator(asset: AVURLAsset(url: url))
    gen.appliesPreferredTrackTransform = true
    gen.requestedTimeToleranceBefore = .zero
    gen.requestedTimeToleranceAfter = .zero
    let time = CMTime(seconds: max(0, t), preferredTimescale: 600)
    return try? await gen.image(at: time).image
}

/// Reusable frame grabber for fast scrubbing/playback: one AVAssetImageGenerator with a
/// ~1-frame time tolerance (much faster than exact-frame decode), full-res (candidates are
/// in original coords, so the preview must not be downscaled).
public final class FrameGrabber {
    private let gen: AVAssetImageGenerator
    public init(url: URL) {
        gen = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        gen.appliesPreferredTrackTransform = true
        let tol = CMTime(seconds: 0.033, preferredTimescale: 600)
        gen.requestedTimeToleranceBefore = tol
        gen.requestedTimeToleranceAfter = tol
    }
    public func frame(atSeconds t: Double) async -> CGImage? {
        try? await gen.image(at: CMTime(seconds: max(0, t), preferredTimescale: 600)).image
    }
}

// ---- two-phase folder flow: infer-all-once (cache candidates) -> tune -> export ----
public struct FolderItem: Sendable {
    public let url: URL
    public let candidates: [Detection]
    public let width: Int    // source pixel dims, captured at infer time (annotation export
    public let height: Int   // normalizes coords without re-decoding 548 images)
    public init(url: URL, candidates: [Detection], width: Int = 0, height: Int = 0) {
        self.url = url; self.candidates = candidates; self.width = width; self.height = height
    }
}

/// Aggregate inference timing over a set of forward passes.
/// `meanMs`/`fps` are MODEL-ONLY (forward pass). `wallMeanMs`/`wallFps` are OVERALL
/// (wall-clock ÷ count - includes image/frame decode, candidate decode, I/O).
public struct InferSummary: Sendable {
    public let count: Int, meanMs: Double, minMs: Double, maxMs: Double, totalMs: Double, wallMs: Double
    /// Mean preprocess / postprocess per item when the caller collected them (empty arrays = not measured).
    public let preMeanMs: Double, postMeanMs: Double
    public var fps: Double { meanMs > 0 ? 1000 / meanMs : 0 }                    // model-only
    public var wallMeanMs: Double { count > 0 ? wallMs / Double(count) : 0 }      // overall per item
    public var wallFps: Double { wallMeanMs > 0 ? 1000 / wallMeanMs : 0 }         // overall
    public var hasStages: Bool { preMeanMs > 0 || postMeanMs > 0 }
    public init(_ times: [Double], wallMs: Double, pre: [Double] = [], post: [Double] = []) {
        count = times.count
        totalMs = times.reduce(0, +)
        meanMs = count > 0 ? totalMs / Double(count) : 0
        minMs = times.min() ?? 0
        maxMs = times.max() ?? 0
        self.wallMs = wallMs
        preMeanMs = pre.isEmpty ? 0 : pre.reduce(0, +) / Double(pre.count)
        postMeanMs = post.isEmpty ? 0 : post.reduce(0, +) / Double(post.count)
    }
}

/// Phase 1: forward every image once, caching pre-NMS candidates (conf/iou tuning stays cheap).
/// Returns the cached items + inference timing summary. With `tiling.mode != .off` each image
/// runs the tiled pipeline instead (FolderItem shape is unchanged - the cache just holds the
/// merged pool), `times` entries become the per-image SUM of all forwards, and aggregate tile
/// stats are returned for the UI.
public func inferFolder(_ det: Detector, input: URL, confFloor: Float = 0.05,
                        tiling: TilingConfig = TilingConfig(),
                        progress: ((_ done: Int, _ total: Int) -> Void)? = nil)
    -> (items: [FolderItem], summary: InferSummary, tileStats: TileStats?) {
    let files = listImages(input)
    var out: [FolderItem] = []; out.reserveCapacity(files.count)
    var times: [Double] = [], pres: [Double] = []
    var stats: TileStats? = tiling.mode == .off ? nil : TileStats()
    let t0 = Date()
    for (i, url) in files.enumerated() {
        if let cg = loadCGImage(url) {
            if tiling.mode == .off {
                if let raw = try? det.forward(cg) {
                    out.append(FolderItem(url: url, candidates: det.candidates(raw, confFloor: confFloor),
                                          width: cg.width, height: cg.height))
                    times.append(raw.inferMs); pres.append(raw.preMs)
                }
            } else if let tiled = try? det.tiledCandidates(cg, config: tiling, confFloor: confFloor) {
                out.append(FolderItem(url: url, candidates: tiled.candidates,
                                      width: cg.width, height: cg.height))
                times.append(tiled.inferMs)
                stats?.add(tiled)
            }
        }
        progress?(i + 1, files.count)
    }
    return (out, InferSummary(times, wallMs: Date().timeIntervalSince(t0) * 1000, pre: pres), stats)
}

/// Phase 3: write annotated images from cached candidates + the tuned params - NO inference.
@discardableResult
public func exportFolderCached(_ items: [FolderItem], output: URL, names: [String],
                               conf: Float, iou: CGFloat, style: BoxStyle, label: LabelMode,
                               detector: Detector? = nil, overlay: SegOverlay = .both,
                               nmsMode: NMSMode = .standard, sigma: Float = 0.1, maxDet: Int = 300,
                               progress: ((_ done: Int, _ total: Int) -> Void)? = nil) -> Int {
    try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let seg = detector?.isSegment == true
    var written = 0
    var usedStems = Set<String>()
    for (i, item) in items.enumerated() {
        if let cg = loadCGImage(item.url) {
            let dets = Detector.nms(item.candidates, conf: conf, iou: iou, maxDet: maxDet, mode: nmsMode, sigma: sigma)
            var masks: [MaskBitmap] = [], drawBoxes = true
            if seg, overlay != .boxes, let det = detector, let raw = try? det.forward(cg) {
                masks = dets.compactMap { det.maskImage($0, raw) }
                drawBoxes = overlay != .masks
            }
            if let a = annotate(cg, dets, names: names, style: style, label: label, masks: masks, drawBoxes: drawBoxes) {
                let stem = uniqueStem(&usedStems, item.url.deletingPathExtension().lastPathComponent)
                saveCGImage(a, to: output.appendingPathComponent(stem + ".jpg"))
                written += 1
            }
        }
        progress?(i + 1, items.count)
    }
    return written
}

// ---- two-phase video flow: infer every frame once (cache candidates) -> scrub/tune -> export ----

/// CGImagePropertyOrientation matching a video track's preferredTransform, so each CIImage frame is
/// oriented the SAME way AVFoundation displays it. Using CIImage.oriented(_:) (rather than a raw
/// transformed(by:), which mishandles CIImage's bottom-left origin) keeps a rotated video's overlay
/// aligned instead of 180-flipped.
func videoOrientation(_ t: CGAffineTransform) -> CGImagePropertyOrientation {
    switch (t.a.rounded(), t.b.rounded(), t.c.rounded(), t.d.rounded()) {
    case (0, 1, -1, 0):  return .right   // 90 CW (portrait)
    case (0, -1, 1, 0):  return .left    // 90 CCW
    case (-1, 0, 0, -1): return .down    // 180
    default:             return .up      // identity
    }
}

/// Phase 1: stream + forward every frame once, caching per-frame candidates + timing.
/// Per-frame camera motion collected while a video is inferred once, so tracking can later be
/// (re)run over the cached candidates without touching the video again. Index-aligned with `frames`.
public final class MotionLog {
    public let estimator: CameraMotionEstimator
    public private(set) var entries: [CameraMotion?] = []
    public init(estimator: CameraMotionEstimator) { self.estimator = estimator }
    func record(_ frame: CGImage) { entries.append(estimator.estimate(frame)) }
    public func motion(at index: Int) -> CameraMotion? { index < entries.count ? entries[index] : nil }
}

/// Tracks over cached per-frame candidates: NMS at the given settings then the tracker, frame by frame
/// in order (the pure function of cache + motions + settings the GUI re-runs when a slider moves).
public func trackCached(_ framesCands: [[Detection]], conf: Float, iou: CGFloat, maxDet: Int = 300,
                        nmsMode: NMSMode = .standard, sigma: Float = 0.1,
                        kind: TrackerKind, fps: Double, trackBuffer: Int = 30, motions: MotionLog? = nil,
                        shouldStop: (() -> Bool)? = nil, each: ((Int, [Detection]) -> Void)? = nil) -> [[Detection]] {
    let tracker = Tracker(kind: kind, fps: fps, trackBuffer: trackBuffer)
    // the second association wants the low-score candidates the user's conf would drop
    let floor = min(conf, tracker.config.detectorFloor)
    var out: [[Detection]] = []
    out.reserveCapacity(framesCands.count)
    for (i, cands) in framesCands.enumerated() {
        if shouldStop?() == true { break }
        let dets = Detector.nms(cands, conf: floor, iou: iou, maxDet: maxDet, mode: nmsMode, sigma: sigma)
        let tracked = tracker.update(dets, motion: kind == .botSort ? motions?.motion(at: i) : nil)
        // show only tracks at or above the user's conf, like untracked playback would
        let shown = tracked.filter { $0.score >= conf }
        out.append(shown)
        each?(i, shown)
    }
    return out
}

public func inferVideo(_ det: Detector, input: URL, confFloor: Float = 0.05, motionLog: MotionLog? = nil,
                       progress: ((_ done: Int, _ estTotal: Int) -> Void)? = nil) async throws -> (frames: [[Detection]], raws: [Detector.RawOutput?], summary: InferSummary, fps: Double, size: CGSize) {
    let asset = AVURLAsset(url: input)
    guard let tracks = try? await asset.loadTracks(withMediaType: .video), let track = tracks.first else { throw PipelineError.noVideoTrack }
    let nominalFps = (try? await track.load(.nominalFrameRate)) ?? 0
    let fps = nominalFps > 0 ? nominalFps : 30
    let transform = (try? await track.load(.preferredTransform)) ?? .identity
    let orient = videoOrientation(transform)
    let naturalSize = (try? await track.load(.naturalSize)) ?? .zero
    let dur = (try? await asset.load(.duration))?.seconds ?? 0
    let estTotal = Int((dur * Double(fps)).rounded())
    let swap = (orient == .right || orient == .left)   // 90° rotation swaps width/height
    let natW = Int((swap ? naturalSize.height : naturalSize.width).rounded())
    let natH = Int((swap ? naturalSize.width : naturalSize.height).rounded())
    guard let reader = try? AVAssetReader(asset: asset) else { throw PipelineError.readerInit }
    let rout = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    rout.alwaysCopiesSampleData = false; reader.add(rout); reader.startReading()
    let cictx = CIContext()
    var frames: [[Detection]] = [], raws: [Detector.RawOutput?] = [], times: [Double] = [], pres: [Double] = [], n = 0
    let seg = det.isSegment   // only seg needs the proto tensor cached (for masks); else keep memory flat
    let t0 = Date()
    // 3-stage pipeline: frame decode (CIImage->CGImage, the expensive CPU step) for frame n+1
    // overlaps the Core ML forward of frame n, and candidate extraction of frame n runs on a
    // detached task overlapping the next forward. This closes most of the model-only vs overall
    // fps gap (the stages used to run strictly serially). Exactly ONE decode task exists at a
    // time (awaited before the next spawns), so the AVAssetReader is never touched concurrently;
    // candidates() reads a finished MLMultiArray (concurrent-read safe). The in-flight bound
    // keeps at most ~3 raw output tensors alive, so memory stays flat on long videos.
    func decodeOne() -> CGImage? {
        guard reader.status == .reading,
              let sb = rout.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sb) else { return nil }
        // Upright video (the overwhelmingly common case): wrap the BGRA buffer as a CGImage with
        // ONE memcpy - the CIContext render (full-res color-managed draw) was the decode-stage
        // bottleneck capping overall fps at ~1/3 of model-only. Rotated tracks keep the CI path.
        if orient == .up { return Detector.cgImage(from: pb) }
        let ci = CIImage(cvPixelBuffer: pb).oriented(orient)   // upright, matching AVPlayer
        return cictx.createCGImage(ci, from: ci.extent)
    }
    var candTasks: [Task<([Detection], Detector.RawOutput?), Never>] = []
    func drainOne() async {
        guard !candTasks.isEmpty else { return }
        let (c, r) = await candTasks.removeFirst().value
        frames.append(c); raws.append(r)
    }
    var nextDecode = Task.detached { decodeOne() }
    while let cg = await nextDecode.value {
        nextDecode = Task.detached { decodeOne() }            // decode n+1 while forwarding n
        motionLog?.record(cg)                                 // sequential, index-aligned with frames
        if let raw = try? det.forward(cg) {
            times.append(raw.inferMs); pres.append(raw.preMs)
            candTasks.append(Task.detached { (det.candidates(raw, confFloor: confFloor), seg ? raw.maskOnly() : nil) })
        } else {
            candTasks.append(Task { ([], nil) })              // keeps frame/candidate alignment
        }
        if candTasks.count >= 3 { await drainOne() }          // bound in-flight raw tensors
        n += 1
        if n % 4 == 0 { progress?(n, estTotal) }
    }
    while !candTasks.isEmpty { await drainOne() }
    return (frames, raws, InferSummary(times, wallMs: Date().timeIntervalSince(t0) * 1000, pre: pres), Double(fps), CGSize(width: natW, height: natH))
}

/// Phase 3: re-stream frames in order, apply cached candidates[i] + tuned params, encode. NO inference.
@discardableResult
public func exportVideoCached(input: URL, output: URL, framesCands: [[Detection]], names: [String],
                              conf: Float, iou: CGFloat, style: BoxStyle, label: LabelMode, resize: Int = 0,
                              raws: [Detector.RawOutput?] = [], detector: Detector? = nil, overlay: SegOverlay = .both,
                              nmsMode: NMSMode = .standard, sigma: Float = 0.1, maxDet: Int = 300,
                              tracked: [[Detection]]? = nil,
                              progress: ((_ done: Int, _ total: Int) -> Void)? = nil) async throws -> VideoStats {
    let seg = detector?.isSegment == true && overlay != .boxes
    let asset = AVURLAsset(url: input)
    guard let tracks = try? await asset.loadTracks(withMediaType: .video), let track = tracks.first else { throw PipelineError.noVideoTrack }
    let nominalFps = (try? await track.load(.nominalFrameRate)) ?? 0
    let fps = nominalFps > 0 ? nominalFps : 30
    let transform = (try? await track.load(.preferredTransform)) ?? .identity
    let orient = videoOrientation(transform)
    let naturalSize = (try? await track.load(.naturalSize)) ?? .zero
    let swap = (orient == .right || orient == .left)
    let natW = Int((swap ? naturalSize.height : naturalSize.width).rounded())
    let natH = Int((swap ? naturalSize.width : naturalSize.height).rounded())
    let (outW, outH) = resize > 0 ? fitLongSide(natW, natH, resize) : (natW, natH)
    guard let reader = try? AVAssetReader(asset: asset) else { throw PipelineError.readerInit }
    let rout = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    rout.alwaysCopiesSampleData = false; reader.add(rout)
    try? FileManager.default.removeItem(at: output)
    let ftype: AVFileType = output.pathExtension.lowercased() == "mov" ? .mov : .mp4
    guard let writer = try? AVAssetWriter(outputURL: output, fileType: ftype) else { throw PipelineError.writerInit }
    let winput = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: outW, AVVideoHeightKey: outH])
    winput.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: winput, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: outW, kCVPixelBufferHeightKey as String: outH])
    writer.add(winput); reader.startReading(); writer.startWriting(); writer.startSession(atSourceTime: .zero)
    let cictx = CIContext(); var n = 0; let total = framesCands.count
    while reader.status == .reading {
        guard let sb = rout.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sb) else { break }
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        let ci = CIImage(cvPixelBuffer: pb).oriented(orient)   // upright, matching AVPlayer
        guard var cg = cictx.createCGImage(ci, from: ci.extent) else { n += 1; continue }
        if resize > 0 { cg = resizeExact(cg, outW, outH) }
        // `tracked` (from trackCached, index-aligned) replaces the per-frame NMS when tracking is on
        let dets = tracked.map { n < $0.count ? $0[n] : [] }
            ?? Detector.nms(n < framesCands.count ? framesCands[n] : [], conf: conf, iou: iou,
                            maxDet: maxDet, mode: nmsMode, sigma: sigma)
        var masks: [MaskBitmap] = []
        if seg, let det = detector, n < raws.count, let raw = raws[n] { masks = dets.compactMap { det.maskImage($0, raw) } }
        let drawBoxes = !(detector?.isSegment == true && overlay == .masks)
        guard let annotated = annotate(cg, dets, names: names, style: style, label: label, masks: masks, drawBoxes: drawBoxes),
              let pool = adaptor.pixelBufferPool else { n += 1; continue }
        var opb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &opb)
        guard let dst = opb else { n += 1; continue }
        CVPixelBufferLockBaseAddress(dst, [])
        if let base = CVPixelBufferGetBaseAddress(dst),
           let c = CGContext(data: base, width: outW, height: outH, bitsPerComponent: 8,
                             bytesPerRow: CVPixelBufferGetBytesPerRow(dst), space: CGColorSpaceCreateDeviceRGB(),
                             bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) {
            c.draw(annotated, in: CGRect(x: 0, y: 0, width: outW, height: outH))
        }
        CVPixelBufferUnlockBaseAddress(dst, [])
        while !winput.isReadyForMoreMediaData { usleep(2000) }
        adaptor.append(dst, withPresentationTime: pts)
        n += 1
        if n % 10 == 0 { progress?(n, total) }
    }
    winput.markAsFinished(); await writer.finishWriting()
    return VideoStats(frames: n, meanMs: 0, outW: outW, outH: outH, fps: Int(fps.rounded()))
}
