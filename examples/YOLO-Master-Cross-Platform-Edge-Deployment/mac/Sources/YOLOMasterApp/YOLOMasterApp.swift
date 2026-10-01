// YOLOMasterApp - SwiftUI frontend for the Core ML runner (YOLOMasterKit backend).
//
// Pipeline:  choose model + source  ->  RUN (infer the whole set once, progress bar)  ->
//            browse the Finder + tune conf/iou/style/label in real time (cheap NMS/redraw
//            from cached candidates, NO re-inference)  ->  Export writes with the tuned params.
//   image  -> Run infers 1 -> tune -> Save
//   folder -> Run infers all (cache) -> Finder (Icons/List/Gallery) + arrows to browse -> Export folder
//   video  -> scrub a frame (infers it) -> tune -> Export video
//
// Build & run:  swift run -c release --package-path mac YOLOMasterApp   |   Bundle: mac/make_app.sh
import SwiftUI
import AppKit
import UniformTypeIdentifiers
import CoreGraphics
import ImageIO
import AVFoundation
import os   // OSAllocatedUnfairLock: thread-safe bake-cancellation token
@preconcurrency import YOLOMasterKit   // Detector/RawOutput aren't Sendable; we hop them to main safely

let brandColor = Color.accentColor   // default system accent (blue)

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ note: Notification) {
        NSApp.setActivationPolicy(.regular); NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct YOLOMasterApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    var body: some Scene {
        WindowGroup("YOLO-Master") { ContentView().frame(minWidth: 1120, minHeight: 720) }
            .windowStyle(.titleBar)
    }
}

// ---- async, cached thumbnails ----
func makeThumbnail(_ url: URL, max: CGFloat) -> NSImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    let opts: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                 kCGImageSourceThumbnailMaxPixelSize: max,
                                 kCGImageSourceCreateThumbnailWithTransform: true]
    guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }
    return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
}
final class ThumbCache {
    static let shared = ThumbCache()
    private let cache = NSCache<NSString, NSImage>()
    private let queue = DispatchQueue(label: "com.yolomaster.thumb", qos: .userInitiated, attributes: .concurrent)
    init() { cache.countLimit = 800 }
    func thumb(_ url: URL, max: CGFloat, _ done: @escaping (NSImage?) -> Void) {
        let key = "\(Int(max))|\(url.path)" as NSString
        if let img = cache.object(forKey: key) { done(img); return }
        queue.async { [weak self] in
            let img = makeThumbnail(url, max: max)
            if let img { self?.cache.setObject(img, forKey: key) }
            DispatchQueue.main.async { done(img) }
        }
    }
}
struct AsyncThumb: View {
    let url: URL; var max: CGFloat = 128; var fit: Bool = false
    @State private var image: NSImage?
    var body: some View {
        Group {
            if let image {
                if fit { Image(nsImage: image).resizable().scaledToFit() }
                else { Image(nsImage: image).resizable().scaledToFill() }
            } else {
                Rectangle().fill(Color.gray.opacity(0.15))
            }
        }
        .onAppear { if image == nil { ThumbCache.shared.thumb(url, max: max) { image = $0 } } }
    }
}

/// Image pixel dims from the header only (no decode) - cheap even across a large folder.
/// File-scope (not actor-isolated) so background tasks can call it.
func imagePixelSize(_ url: URL) -> (w: Int, h: Int)? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
          let w = props[kCGImagePropertyPixelWidth] as? Int,
          let h = props[kCGImagePropertyPixelHeight] as? Int else { return nil }
    return (w, h)
}

// ---------- stats models ----------
struct StatModelInfo: Equatable { let name: String; let imgsz: Int; let nc: Int; let compute: String }
struct ClassCount: Identifiable, Equatable { var id: String { name }; let name: String; let count: Int }

// ---------- inference engine (two-phase: forward-once + cheap tuning) ----------
final class InferenceEngine: ObservableObject, @unchecked Sendable {   // state is guarded by `queue` + main-hops
    @Published var resultImage: NSImage?
    @Published var detCount = 0
    @Published var modelSummary = ""
    @Published var status = "Choose a model (.mlpackage) + a source, then Run."
    @Published var busy = false
    @Published var exporting = false
    @Published var hasResults = false          // folder: inference cache ready
    @Published var progress: Double?
    @Published var outputURL: URL?
    @Published var modelInfo: StatModelInfo?   // model name / imgsz / classes / compute
    @Published var infer: InferSummary?        // count / avg / min / max / total / fps
    @Published var classCounts: [ClassCount] = []   // per-class breakdown of the current frame
    @Published var modelIsSegment = false      // drives the Masks/Boxes/Both overlay control
    @Published var tileStats: TileStats?       // tiled modes: tiles run/total (+fallback/cap), nil when off

    private var detector: Detector?
    /// Preprocessing device applied to every detector the engine drives (the sidebar Preprocess picker).
    var preprocDevice: PreprocDevice = .gpu
    private var resultsTiled = false           // current image/folder cache was built tiled
    private var tiledMasksKept = false         // tiled cache retained global-pass masks (keepGlobalMasks)
    private var currentRaw: Detector.RawOutput?    // cached forward pass for the shown image (seg masks need protos)
    private var key = ""
    private var detNames: [String] = []
    private var currentCG: CGImage?
    private var currentCands: [Detection] = []
    private var currentMs = 0.0
    private var lastAnnotated: CGImage?
    private var folderCache: [FolderItem] = []
    private var folderInput: URL?
    private var imageInput: URL?                 // single-image source (annotation export stem)
    private var videoCache: [[Detection]] = []
    private var videoRaws: [Detector.RawOutput?] = []   // per-frame proto (seg only) for on-demand masks
    private var videoDet: Detector?                       // the seg detector, to compute mask overlays
    @Published private(set) var videoFps: Double = 30
    @Published private(set) var videoURL: URL?
    @Published private(set) var videoSize: CGSize = .zero
    private var videoInput: URL?
    private let queue = DispatchQueue(label: "com.yolomaster.inference")
    // ---- tracking over the cached video: a pure function of (cached candidates, per-frame camera
    // motion recorded during inference, conf / IoU / NMS settings, tracker kind), recomputed off-main
    // whenever one of them changes and never touching the video again ----
    @Published var trackKind: TrackerKind? = nil        // nil = off
    @Published private(set) var trackGen = 0             // bumped when a tracked pass lands
    @Published private(set) var trackCount = 0           // distinct ids in the current tracked pass
    private var videoMotion: MotionLog? = nil
    private var videoTracks: [[Detection]]? = nil
    private var videoTracksKey = ""
    private var trackCancel: OSAllocatedUnfairLock<Bool>?

    func resetResults() {
        hasResults = false; folderCache = []; folderInput = nil; videoCache = []; videoInput = nil; videoURL = nil; videoSize = .zero; outputURL = nil
        resultImage = nil; detCount = 0; currentCG = nil; currentCands = []; currentRaw = nil
        videoRaws = []; videoDet = nil
        videoMotion = nil; videoTracks = nil; videoTracksKey = ""; trackCount = 0
        infer = nil; classCounts = []; tileStats = nil; resultsTiled = false; tiledMasksKept = false
        imageInput = nil
        baked = []; bakedKey = ""; bakeGen += 1; detsCacheKey = ""; detsCacheVal = []
        bakeCancel?.withLock { $0 = true }; bakeCancel = nil
        stopVideoOverlayLoop(); videoOverlayImg = nil
        status = "Ready - press Run."
    }

    // ---- image / video-frame: forward one (or tiled), cache candidates, render ----
    func previewURL(model: URL, image: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay, preprocess: Detector.PreprocessMode,
                    tiling: TilingConfig = TilingConfig(), nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) {
        guard let cg = loadCGImage(image) else { publish(error: "Could not read image."); return }
        imageInput = image
        preview(model: model, cg: cg, compute: compute, conf: conf, iou: iou, style: style, label: label, overlay: overlay, preprocess: preprocess, tiling: tiling, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet)
    }
    func preview(model: URL, cg: CGImage, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay, preprocess: Detector.PreprocessMode,
                 tiling: TilingConfig = TilingConfig(), nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) {
        busy = true; progress = nil; status = tiling.mode == .off ? "Inferring…" : "Inferring (tiled)…"
        let k = model.path + "|" + compute.rawValue
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det = try self.reuseDetector(model: model, compute: compute, key: k)
                det.preprocess = preprocess; det.preprocDevice = self.preprocDevice
                let s: InferSummary
                var stats: TileStats? = nil
                if tiling.mode == .off {
                    let raw = try det.forward(cg)
                    self.currentCG = cg; self.currentCands = det.candidates(raw); self.currentMs = raw.inferMs
                    self.currentRaw = det.isSegment ? raw : nil
                    self.resultsTiled = false
                    s = InferSummary([raw.inferMs], wallMs: raw.inferMs + raw.preMs, pre: [raw.preMs])
                } else {
                    let t0 = Date()
                    let tiled = try det.tiledCandidates(cg, config: tiling)
                    self.currentCG = cg; self.currentCands = tiled.candidates; self.currentMs = tiled.inferMs
                    self.currentRaw = tiled.globalRaw           // non-nil only when keepGlobalMasks on a seg model
                    self.resultsTiled = true
                    self.tiledMasksKept = tiled.globalRaw != nil
                    var st = TileStats(); st.add(tiled); stats = st
                    s = InferSummary([tiled.inferMs], wallMs: Date().timeIntervalSince(t0) * 1000)
                }
                self.detNames = det.classNames
                let info = StatModelInfo(name: model.lastPathComponent, imgsz: det.imgsz, nc: det.nc, compute: compute.label)
                DispatchQueue.main.async { self.modelInfo = info; self.infer = s; self.modelIsSegment = det.isSegment; self.tileStats = stats }
                self.render(conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet)
            } catch { self.publish(error: "Inference failed: \(error.localizedDescription)") }
        }
    }

    // ---- folder: infer ALL once (progress), cache candidates ----
    func runFolder(model: URL, input: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay, preprocess: Detector.PreprocessMode,
                   tiling: TilingConfig = TilingConfig(), nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) {
        busy = true; exporting = false; hasResults = false; progress = 0; outputURL = nil
        status = tiling.mode == .off ? "Inferring folder…" : "Inferring folder (tiled)…"
        let k = model.path + "|" + compute.rawValue
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let det = try self.reuseDetector(model: model, compute: compute, key: k)
                det.preprocess = preprocess; det.preprocDevice = self.preprocDevice
                self.detNames = det.classNames
                let (items, summary, stats) = inferFolder(det, input: input, confFloor: 0.05, tiling: tiling) { done, total in
                    DispatchQueue.main.async {
                        self.progress = total > 0 ? Double(done) / Double(total) : nil
                        self.status = "Inferring \(done)/\(total)…"
                    }
                }
                self.folderCache = items; self.folderInput = input
                self.resultsTiled = tiling.mode != .off
                self.tiledMasksKept = tiling.mode != .off && tiling.keepGlobalMasks && det.isSegment
                let info = StatModelInfo(name: model.lastPathComponent, imgsz: det.imgsz, nc: det.nc, compute: compute.label)
                if let first = items.first, let cg = loadCGImage(first.url) {
                    self.currentCG = cg; self.currentCands = first.candidates; self.currentMs = 0
                    self.currentRaw = (det.isSegment && (tiling.mode == .off || tiling.keepGlobalMasks)) ? try? det.forward(cg) : nil
                }
                DispatchQueue.main.async {
                    self.modelInfo = info; self.infer = summary; self.hasResults = !items.isEmpty
                    self.busy = false; self.progress = nil; self.modelIsSegment = det.isSegment
                    self.tileStats = stats
                    self.status = "Inferred \(items.count) images - browse & tune, then Export"
                }
                self.render(conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet)
            } catch { self.publish(error: "Inference failed: \(error.localizedDescription)") }
        }
    }

    // ---- show a cached folder item (instant; re-forwards only for seg masks, never when tiled) ----
    func showFolder(index i: Int, url: URL, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay,
                    nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) {
        queue.async { [weak self] in
            guard let self, let cg = loadCGImage(url) else { return }
            self.currentCG = cg
            self.currentCands = self.folderCache.indices.contains(i) ? self.folderCache[i].candidates : []
            self.currentMs = 0
            let wantMasks = self.detector?.isSegment == true && (!self.resultsTiled || self.tiledMasksKept)
            self.currentRaw = wantMasks ? try? self.detector?.forward(cg) : nil
            self.render(conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet)
        }
    }

    // ---- tuning: cheap re-NMS + redraw of the current frame ----
    private var pendingRestyle: DispatchWorkItem?
    func restyle(conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay,
                 nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) {
        pendingRestyle?.cancel()
        let item = DispatchWorkItem { [weak self] in self?.render(conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet) }
        pendingRestyle = item
        queue.async(execute: item)
    }
    private func render(conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay,
                        nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) {
        guard let cg = currentCG, !detNames.isEmpty else { DispatchQueue.main.async { self.busy = false }; return }
        let dets = Detector.nms(currentCands, conf: Float(conf), iou: CGFloat(iou), maxDet: maxDet, mode: nmsMode, sigma: Float(sigma))
        var masks: [MaskBitmap] = [], drawBoxes = true
        if let det = detector, det.isSegment, let raw = currentRaw, overlay != .boxes {
            masks = dets.compactMap { det.maskImage($0, raw) }
            drawBoxes = overlay != .masks
        }
        let annotated = annotate(cg, dets, names: detNames, style: style, label: label, masks: masks, drawBoxes: drawBoxes) ?? cg
        self.lastAnnotated = annotated
        let ns = NSImage(cgImage: annotated, size: NSSize(width: cg.width, height: cg.height))
        let ms = self.currentMs
        var byClass: [Int: Int] = [:]
        for d in dets { byClass[d.cls, default: 0] += 1 }
        let breakdown = byClass.sorted { $0.value > $1.value }.map {
            ClassCount(name: self.detNames.indices.contains($0.key) ? self.detNames[$0.key] : "class\($0.key)", count: $0.value)
        }
        DispatchQueue.main.async {
            self.resultImage = ns; self.detCount = dets.count; self.busy = false; self.classCounts = breakdown
            self.status = ms > 0 ? "\(dets.count) detections · \(String(format: "%.1f", ms)) ms" : "\(dets.count) detections"
        }
    }

    // ---- export ----
    func exportFolder(conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay,
                      nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) {
        guard let input = folderInput, !folderCache.isEmpty else { return }
        busy = true; exporting = true; progress = 0; outputURL = nil; status = "Exporting folder…"
        let out = input.deletingLastPathComponent().appendingPathComponent(input.lastPathComponent + "_annotated")
        let cache = folderCache, names = detNames, det = detector
        let ov: SegOverlay = (resultsTiled && !tiledMasksKept) ? .boxes : overlay   // tiled-without-masks cache has no coeffs
        queue.async { [weak self] in
            guard let self else { return }
            let n = exportFolderCached(cache, output: out, names: names, conf: Float(conf), iou: CGFloat(iou), style: style, label: label, detector: det, overlay: ov,
                                       nmsMode: nmsMode, sigma: Float(sigma), maxDet: maxDet) { done, total in
                DispatchQueue.main.async { self.progress = total > 0 ? Double(done)/Double(total) : nil; self.status = "Exporting \(done)/\(total)…" }
            }
            DispatchQueue.main.async {
                self.outputURL = out; self.busy = false; self.exporting = false; self.progress = nil
                self.status = "Exported \(n) images"
            }
        }
    }
    // ---- video: infer ALL frames once (progress), cache candidates ----
    func runVideo(model: URL, input: URL, compute: ComputeMode, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, preprocess: Detector.PreprocessMode, overlay: SegOverlay,
                  nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) {
        busy = true; exporting = false; hasResults = false; progress = 0; outputURL = nil; status = "Inferring video…"
        // Release the PREVIOUS run's per-frame tensors before the new run allocates its own:
        // re-inferring a seg video otherwise holds both generations at once (GBs) and pushes
        // the machine into memory pressure that outlives the run.
        videoCache = []; videoRaws = []; videoDet = nil
        videoMotion = nil; videoTracks = nil; videoTracksKey = ""; trackCount = 0
        trackCancel?.withLock { $0 = true }; trackCancel = nil
        baked = []; bakedKey = ""; bakeGen += 1
        bakeCancel?.withLock { $0 = true }; bakeCancel = nil
        detsCacheKey = ""; detsCacheVal = []
        stopVideoOverlayLoop(); videoOverlayImg = nil
        Task { [weak self] in
            guard let self else { return }
            do {
                let det = try Detector(modelURL: model, compute: compute)
                det.preprocess = preprocess; det.preprocDevice = self.preprocDevice
                self.detNames = det.classNames
                // camera motion per frame is recorded now (a few ms of Vision per frame) so tracking can be
                // switched on / re-tuned later without re-reading the video
                let motion = MotionLog(estimator: VisionCameraMotion())
                let (frames, raws, summary, fps, size) = try await inferVideo(det, input: input, confFloor: 0.05, motionLog: motion) { done, est in
                    DispatchQueue.main.async {
                        self.progress = est > 0 ? min(1, Double(done) / Double(est)) : nil
                        self.status = "Inferring frame \(done)…"
                    }
                }
                let info = StatModelInfo(name: model.lastPathComponent, imgsz: det.imgsz, nc: det.nc, compute: compute.label)
                DispatchQueue.main.async {
                    self.videoRunGen += 1
                    self.videoCache = frames; self.videoRaws = raws; self.videoDet = det.isSegment ? det : nil
                    self.videoMotion = motion
                    self.modelIsSegment = det.isSegment
                    self.videoFps = fps; self.videoInput = input; self.videoURL = input; self.videoSize = size
                    self.modelInfo = info; self.infer = summary; self.hasResults = !frames.isEmpty
                    self.busy = false; self.progress = nil
                    self.status = "Inferred \(frames.count) frames - play / scrub & tune, then Export"
                    self.setVideoFrameStats(time: 0, conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet)
                    self.requestOverlayFrame(time: 0, conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet,
                                             style: style, label: label, overlay: overlay)
                }
            } catch { DispatchQueue.main.async { self.status = "Inference failed: \(error.localizedDescription)"; self.busy = false; self.progress = nil } }
        }
    }

    // ---- video overlay data: AVPlayer displays frames; we draw boxes from cached candidates at time ----
    var names: [String] { detNames }
    var videoIsSegment: Bool { videoDet != nil }
    private func videoFrameIndex(_ time: Double) -> Int {
        min(max(0, Int((time * videoFps).rounded())), max(0, videoCache.count - 1))
    }
    // Playback detections come from a BAKED per-frame post-NMS array: detection settings are
    // frozen while the video plays (the sidebar disables them), so the whole video's NMS is
    // computed ONCE in the background at the current settings and playback is a pure array
    // index + draw. The baker outruns the playhead within a fraction of a second; frames it
    // hasn't reached yet fall back to the single-slot on-demand cache below.
    private var baked: [[Detection]?] = []
    private var bakedKey = ""
    private var bakeGen = 0
    private var bakeCancel: OSAllocatedUnfairLock<Bool>?   // current bake's cancellation token
    private var detsCacheKey = ""
    private var detsCacheVal: [Detection] = []
    fileprivate var videoRunGen = 0            // bumped when a new video cache is installed
    func detsAt(time: Double, conf: Double, iou: Double, nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) -> [Detection] {
        guard !videoCache.isEmpty else { return [] }
        let idx = videoFrameIndex(time)
        if let tr = currentTracks(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet), idx < tr.count { return tr[idx] }
        if idx < baked.count, bakedKey == bakeKey(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet),
           let b = baked[idx] { return b }
        let key = "\(videoRunGen)|\(idx)|\(conf)|\(iou)|\(nmsMode.rawValue)|\(sigma)|\(maxDet)"
        if key == detsCacheKey { return detsCacheVal }
        let dets = Detector.nms(videoCache[idx], conf: Float(conf), iou: CGFloat(iou), maxDet: maxDet, mode: nmsMode, sigma: Float(sigma))
        detsCacheKey = key; detsCacheVal = dets
        return dets
    }
    private func bakeKey(conf: Double, iou: Double, nmsMode: NMSMode, sigma: Double, maxDet: Int) -> String {
        "\(videoRunGen)|\(conf)|\(iou)|\(nmsMode.rawValue)|\(sigma)|\(maxDet)"
    }
    private func trackKey(conf: Double, iou: Double, nmsMode: NMSMode, sigma: Double, maxDet: Int) -> String {
        "\(trackKind?.rawValue ?? "off")|" + bakeKey(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet)
    }
    /// The tracked per-frame detections for these settings, if the pass for them has landed.
    private func currentTracks(conf: Double, iou: Double, nmsMode: NMSMode, sigma: Double, maxDet: Int) -> [[Detection]]? {
        guard trackKind != nil, videoTracksKey == trackKey(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet) else { return nil }
        return videoTracks
    }
    /// (Re)run the tracker over the cached video for these settings, off-main; a newer request
    /// cancels the running one. Publishes `trackGen` when the pass lands (the view re-renders).
    func ensureTracked(conf: Double, iou: Double, nmsMode: NMSMode, sigma: Double, maxDet: Int) {
        let key = trackKey(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet)
        guard key != videoTracksKey, !videoCache.isEmpty else { return }
        trackCancel?.withLock { $0 = true }
        guard let kind = trackKind else { videoTracks = nil; videoTracksKey = key; trackCount = 0; trackGen += 1; return }
        let token = OSAllocatedUnfairLock(initialState: false)
        trackCancel = token
        let cache = videoCache, motions = videoMotion, fps = videoFps
        status = "Tracking (\(kind.rawValue))…"
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let tracks = trackCached(cache, conf: Float(conf), iou: CGFloat(iou), maxDet: maxDet, nmsMode: nmsMode, sigma: Float(sigma),
                                     kind: kind, fps: fps, motions: motions, shouldStop: { token.withLock { $0 } })
            if token.withLock({ $0 }) { return }   // superseded
            var ids = Set<Int>()
            for f in tracks { for d in f { if let id = d.trackId { ids.insert(id) } } }
            DispatchQueue.main.async {
                guard let self, !token.withLock({ $0 }) else { return }
                self.videoTracks = tracks; self.videoTracksKey = key; self.trackCount = ids.count
                self.status = "Tracked \(ids.count) objects (\(kind.rawValue)) - play / scrub, then Export"
                self.trackGen += 1
            }
        }
    }
    /// Re-bake the whole video's post-NMS detections at the given settings (no-op if already
    /// baked for them). Runs on a global queue in 32-frame chunks; a newer bake supersedes.
    func ensureBaked(conf: Double, iou: Double, nmsMode: NMSMode, sigma: Double, maxDet: Int) {
        let key = bakeKey(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet)
        guard key != bakedKey, !videoCache.isEmpty else { return }
        bakedKey = key
        bakeGen += 1
        let gen = bakeGen
        // CANCEL the superseded bake. Without this, every settings change (each TICK of a
        // slider drag!) spawned a full-video NMS pass that ran to completion - a few drags
        // left dozens of concurrent passes pegging the CPU, the "gets laggier every re-run"
        // syndrome. The token is an unfair-lock-guarded Bool: safe to read off-main.
        bakeCancel?.withLock { $0 = true }
        let token = OSAllocatedUnfairLock(initialState: false)
        bakeCancel = token
        let cache = videoCache
        baked = Array(repeating: nil, count: cache.count)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            var chunk: [(Int, [Detection])] = []
            for i in cache.indices {
                if token.withLock({ $0 }) { return }   // superseded -> stop immediately
                chunk.append((i, Detector.nms(cache[i], conf: Float(conf), iou: CGFloat(iou),
                                              maxDet: maxDet, mode: nmsMode, sigma: Float(sigma))))
                if chunk.count == 32 || i == cache.count - 1 {
                    let batch = chunk; chunk = []
                    DispatchQueue.main.async {
                        guard let self, self.bakeGen == gen else { return }
                        for (j, d) in batch { self.baked[j] = d }
                    }
                }
            }
        }
    }
    /// Compute the seg mask overlay for the frame at `time` (background) and publish it. No-op for
    /// detection models or masks-off. Recomputed as the shown frame / conf / iou / overlay change.
    // ---- self-paced video overlay compositor (the webcam model) ----
    // The live camera is smooth because its overlay is SELF-PACED: a worker renders the latest
    // state as fast as it can and the view blits the newest image while the preview runs free.
    // Video playback now works identically: while playing, a worker loop chases the player
    // clock, composing boxes + labels + masks for the frame under the playhead into ONE
    // transparent full-res image (NMS + batched mask math + annotate, all off-main - legal
    // because detection settings are frozen during playback), and the Canvas draws that single
    // image. Paused / scrubbing composes one frame on demand at full detail.
    @Published var videoOverlayImg: CGImage?
    private let overlayQueue = DispatchQueue(label: "com.yolomaster.videooverlay", qos: .userInitiated)
    private let overlayGen = OSAllocatedUnfairLock(initialState: 0)

    func stopVideoOverlayLoop() { overlayGen.withLock { $0 += 1 } }

    private struct OverlaySnapshot {
        let cache: [[Detection]], raws: [Detector.RawOutput?], det: Detector?, names: [String]
        let fps: Double, size: CGSize
        let conf: Float, iou: CGFloat, nmsMode: NMSMode, sigma: Float, maxDet: Int
        let style: BoxStyle, label: LabelMode, overlay: SegOverlay
        let tracked: [[Detection]]?      // the tracked pass for these settings when tracking is on and it has landed
    }
    private func overlaySnapshot(conf: Double, iou: Double, nmsMode: NMSMode, sigma: Double, maxDet: Int,
                                 style: BoxStyle, label: LabelMode, overlay: SegOverlay) -> OverlaySnapshot? {
        guard !videoCache.isEmpty, videoSize.width > 0, videoSize.height > 0 else { return nil }
        return OverlaySnapshot(cache: videoCache, raws: videoRaws, det: videoDet, names: detNames,
                               fps: videoFps, size: videoSize,
                               conf: Float(conf), iou: CGFloat(iou), nmsMode: nmsMode, sigma: Float(sigma), maxDet: maxDet,
                               style: style, label: label, overlay: overlay,
                               tracked: currentTracks(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet))
    }
    private static func compose(_ s: OverlaySnapshot, idx: Int, maskCap: Int) -> CGImage? {
        guard s.cache.indices.contains(idx) else { return nil }
        let dets = s.tracked.flatMap { idx < $0.count ? $0[idx] : nil }
            ?? Detector.nms(s.cache[idx], conf: s.conf, iou: s.iou, maxDet: s.maxDet, mode: s.nmsMode, sigma: s.sigma)
        let w = Int(s.size.width), h = Int(s.size.height)
        var base: CGImage? = nil
        if let det = s.det, s.overlay != .boxes, s.raws.indices.contains(idx), let raw = s.raws[idx] {
            let md = dets.count > maskCap ? Array(dets.prefix(maskCap)) : dets
            base = det.maskOverlay(md, raw)
        }
        if base == nil {   // transparent canvas when no masks (det model / boxes-only / no proto)
            guard let c = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                    space: CGColorSpaceCreateDeviceRGB(),
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
            base = c.makeImage()
        }
        guard let b = base else { return nil }
        let drawBoxes = !((s.det?.isSegment ?? false) && s.overlay == .masks)
        if !drawBoxes && s.label == .off { return b }
        return annotate(b, dets, names: s.names, style: s.style, label: s.label, masks: [], drawBoxes: drawBoxes) ?? b
    }
    /// PLAYING: chase the player clock, latest-frame-wins, as fast as compose allows.
    func startVideoOverlayLoop(player: AVPlayer, conf: Double, iou: Double, nmsMode: NMSMode, sigma: Double, maxDet: Int,
                               style: BoxStyle, label: LabelMode, overlay: SegOverlay) {
        guard let snap = overlaySnapshot(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet,
                                         style: style, label: label, overlay: overlay) else { return }
        let gen: Int = overlayGen.withLock { $0 += 1; return $0 }
        overlayQueue.async { [weak self] in
            var lastIdx = -1
            while true {
                guard let strong = self, strong.overlayGen.withLock({ $0 }) == gen else { return }
                let t = player.currentTime().seconds
                let idx = min(max(0, Int(((t.isFinite ? t : 0) * snap.fps).rounded())), snap.cache.count - 1)
                if idx == lastIdx { usleep(4000); continue }   // frame unchanged -> tiny nap
                lastIdx = idx
                let img = InferenceEngine.compose(snap, idx: idx, maskCap: 250)
                DispatchQueue.main.async {
                    guard let s2 = self, s2.overlayGen.withLock({ $0 }) == gen else { return }
                    s2.videoOverlayImg = img
                }
            }
        }
    }
    /// PAUSED / scrub / tuning: compose the shown frame once, full detail (no mask cap).
    func requestOverlayFrame(time: Double, conf: Double, iou: Double, nmsMode: NMSMode, sigma: Double, maxDet: Int,
                             style: BoxStyle, label: LabelMode, overlay: SegOverlay) {
        guard let snap = overlaySnapshot(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet,
                                         style: style, label: label, overlay: overlay) else { return }
        let gen: Int = overlayGen.withLock { $0 }
        let idx = min(max(0, Int((max(0, time) * snap.fps).rounded())), snap.cache.count - 1)
        overlayQueue.async { [weak self] in
            let img = InferenceEngine.compose(snap, idx: idx, maskCap: Int.max)
            DispatchQueue.main.async {
                guard let self, self.overlayGen.withLock({ $0 }) == gen else { return }
                self.videoOverlayImg = img
            }
        }
    }
    /// Update the 'this frame' summary stats for the video frame at `time`.
    private var lastVideoStatsAt = Date.distantPast
    func setVideoFrameStats(time: Double, conf: Double, iou: Double, nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300, throttled: Bool = false) {
        if throttled {   // playback: classCounts rebuild + sidebar diff at 30 Hz starves the Canvas
            guard Date().timeIntervalSince(lastVideoStatsAt) > 0.25 else { return }
            lastVideoStatsAt = Date()
        }
        let dets = detsAt(time: time, conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet)
        detCount = dets.count
        var byClass: [Int: Int] = [:]; for d in dets { byClass[d.cls, default: 0] += 1 }
        classCounts = byClass.sorted { $0.value > $1.value }.map {
            ClassCount(name: detNames.indices.contains($0.key) ? detNames[$0.key] : "class\($0.key)", count: $0.value)
        }
    }

    // ---- export video from cached candidates (NO inference) ----
    func exportVideo(conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay,
                     nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) {
        guard let input = videoInput, !videoCache.isEmpty else { return }
        busy = true; exporting = true; progress = 0; outputURL = nil; status = "Exporting video…"
        let out = input.deletingLastPathComponent().appendingPathComponent(input.deletingPathExtension().lastPathComponent + "_annotated.mp4")
        let frames = videoCache, names = detNames, rw = videoRaws, det = videoDet
        // tracked frames: the landed pass for these settings, else computed here (fast: no inference)
        var tracked = currentTracks(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet)
        if tracked == nil, let kind = trackKind {
            tracked = trackCached(frames, conf: Float(conf), iou: CGFloat(iou), maxDet: maxDet, nmsMode: nmsMode, sigma: Float(sigma),
                                  kind: kind, fps: videoFps, motions: videoMotion)
        }
        let trackedFrames = tracked
        Task { [weak self] in
            guard let self else { return }
            do {
                let stats = try await exportVideoCached(input: input, output: out, framesCands: frames, names: names, conf: Float(conf), iou: CGFloat(iou), style: style, label: label, raws: rw, detector: det, overlay: overlay,
                                                        nmsMode: nmsMode, sigma: Float(sigma), maxDet: maxDet, tracked: trackedFrames) { done, total in
                    DispatchQueue.main.async { self.progress = total > 0 ? Double(done) / Double(total) : nil; self.status = "Exporting \(done)/\(total)…" }
                }
                DispatchQueue.main.async {
                    self.outputURL = out; self.busy = false; self.exporting = false; self.progress = nil
                    self.status = "Exported \(stats.frames) frames @\(stats.fps)fps"
                }
            } catch { DispatchQueue.main.async { self.status = "Export failed: \(error.localizedDescription)"; self.busy = false; self.exporting = false } }
        }
    }

    // ---- annotation export (YOLO TXT / COCO JSON / Pascal VOC XML) ----
    /// Source kind inferred from the caches, mirroring the image/folder/video export methods.
    /// WYSIWYG: exports the cached candidates filtered through the CURRENT tuned parameters.
    func exportAnnotations(format: AnnotationFormat, sampling: VideoSampling,
                           conf: Double, iou: Double, nmsMode: NMSMode, sigma: Double, maxDet: Int) {
        if let input = videoInput, !videoCache.isEmpty {
            // let the user place the export root (frames/ + labels/ land inside it)
            guard let root = Self.chooseExportDir(
                suggested: input.deletingPathExtension().lastPathComponent + "_annotations",
                startingIn: input.deletingLastPathComponent(),
                message: "Choose where to save the extracted frames and label files") else { return }
            busy = true; exporting = true; progress = 0; outputURL = nil; status = "Exporting labels…"
            let frames = videoCache, nm = detNames, rw = videoRaws, det = videoDet, fps = videoFps
            let includePolys = det != nil   // video is never tiled; videoDet is non-nil only for seg
            Task { [weak self] in
                guard let self else { return }
                do {
                    let res = try await exportAnnotationsVideo(
                        input: input, root: root, framesCands: frames, names: nm,
                        format: format, sampling: sampling, fps: fps,
                        conf: Float(conf), iou: CGFloat(iou), nmsMode: nmsMode, sigma: Float(sigma), maxDet: maxDet,
                        raws: rw, detector: det, includePolygons: includePolys) { done, total in
                        DispatchQueue.main.async {
                            self.progress = total > 0 ? Double(done) / Double(total) : nil
                            self.status = "Exporting labels \(done)/\(total)…"
                        }
                    }
                    DispatchQueue.main.async {
                        self.outputURL = res.output; self.busy = false; self.exporting = false; self.progress = nil
                        self.status = "Exported \(res.images) frames + labels (\(res.instances) objects)"
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.status = "Label export failed: \(error.localizedDescription)"
                        self.busy = false; self.exporting = false; self.progress = nil
                    }
                }
            }
        } else if let input = folderInput, !folderCache.isEmpty {
            guard let out = Self.chooseExportDir(
                suggested: input.lastPathComponent + "_labels",
                startingIn: input.deletingLastPathComponent(),
                message: "Choose where to save the label files") else { return }
            busy = true; exporting = true; progress = 0; outputURL = nil; status = "Exporting labels…"
            let cache = folderCache, nm = detNames, det = detector
            let includePolys = det?.isSegment == true && (!resultsTiled || tiledMasksKept)
            queue.async { [weak self] in
                guard let self else { return }
                do {
                    let res = try exportAnnotationsFolder(cache, output: out, names: nm, format: format,
                        conf: Float(conf), iou: CGFloat(iou), nmsMode: nmsMode, sigma: Float(sigma), maxDet: maxDet,
                        detector: det, includePolygons: includePolys) { done, total in
                        DispatchQueue.main.async {
                            self.progress = total > 0 ? Double(done) / Double(total) : nil
                            self.status = "Exporting labels \(done)/\(total)…"
                        }
                    }
                    DispatchQueue.main.async {
                        self.outputURL = res.output; self.busy = false; self.exporting = false; self.progress = nil
                        self.status = "Exported \(res.images) label files (\(res.instances) objects)"
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.status = "Label export failed: \(error.localizedDescription)"
                        self.busy = false; self.exporting = false; self.progress = nil
                    }
                }
            }
        } else if currentCG != nil {
            // Build the IR on the engine queue (touches private caches + non-Sendable RawOutput),
            // then hop to main for the save panel.
            queue.async { [weak self] in
                guard let self, let cg = self.currentCG else { return }
                let dets = Detector.nms(self.currentCands, conf: Float(conf), iou: CGFloat(iou),
                                        maxDet: maxDet, mode: nmsMode, sigma: Float(sigma))
                let det = self.detector
                let includePolys = det?.isSegment == true && (!self.resultsTiled || self.tiledMasksKept)
                let insts = annotationInstances(dets, detector: det, raw: self.currentRaw, includePolygons: includePolys)
                let stem = self.imageInput?.deletingPathExtension().lastPathComponent ?? "annotations"
                let srcName = self.imageInput?.lastPathComponent ?? (stem + ".jpg")
                let img = AnnotatedImage(name: stem, width: cg.width, height: cg.height, instances: insts)
                let nm = self.detNames
                DispatchQueue.main.async {
                    let panel = NSSavePanel()
                    panel.nameFieldStringValue = stem + (format == .cocoJSON ? ".coco." : ".") + format.fileExtension
                    guard panel.runModal() == .OK, let url = panel.url else { return }
                    do {
                        switch format {
                        case .yoloTXT:
                            try AnnotationWriter.yoloLines(img, segDialect: includePolys)
                                .write(to: url, atomically: true, encoding: .utf8)
                            try? AnnotationWriter.classesTXT(nm)
                                .write(to: url.deletingLastPathComponent().appendingPathComponent("classes.txt"),
                                       atomically: true, encoding: .utf8)
                        case .pascalVOC:
                            try AnnotationWriter.vocXML(img, names: nm, folder: url.deletingLastPathComponent().lastPathComponent)
                                .write(to: url, atomically: true, encoding: .utf8)
                        case .cocoJSON:
                            try AnnotationWriter.cocoJSON([img], fileNames: [srcName], names: nm, segDialect: includePolys)
                                .write(to: url)
                        }
                        self.status = "Exported \(insts.count) objects → \(url.lastPathComponent)"
                    } catch { self.status = "Label export failed: \(error.localizedDescription)" }
                }
            }
        }
    }

    /// Save-panel idiom for exporting a NEW directory: the user names/places the folder, the
    /// export creates it. Returns nil on cancel. Must run on the main thread (runModal).
    private static func chooseExportDir(suggested: String, startingIn: URL, message: String) -> URL? {
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = suggested
        panel.directoryURL = startingIn
        panel.message = message
        panel.prompt = "Export"
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func reuseDetector(model: URL, compute: ComputeMode, key k: String) throws -> Detector {
        if let d = detector, key == k { return d }
        let d = try Detector(modelURL: model, compute: compute); detector = d; key = k; return d
    }
    private func publish(error: String) { DispatchQueue.main.async { self.status = error; self.busy = false; self.exporting = false; self.progress = nil } }
    func save() {   // single image / current folder item (lastAnnotated is refreshed on every render)
        guard let cg = lastAnnotated else { return }
        let panel = NSSavePanel(); panel.allowedContentTypes = [.jpeg, .png]; panel.nameFieldStringValue = "annotated.jpg"
        if panel.runModal() == .OK, let url = panel.url { saveCGImage(cg, to: url) }
    }
    /// Annotate + save the single video frame shown at `time` (the video overlay is drawn in a Canvas,
    /// not baked into an image, so we re-extract + annotate here).
    func saveVideoFrame(time: Double, conf: Double, iou: Double, style: BoxStyle, label: LabelMode, overlay: SegOverlay,
                        nmsMode: NMSMode = .standard, sigma: Double = 0.1, maxDet: Int = 300) {
        guard let input = videoInput, !videoCache.isEmpty else { return }
        let idx = videoFrameIndex(time)
        let cands = videoCache[idx]
        let raw = videoRaws.indices.contains(idx) ? videoRaws[idx] : nil
        let det = videoDet, names = detNames
        Task {
            guard let cg = await extractFrame(input, atSeconds: time) else { return }
            let dets = Detector.nms(cands, conf: Float(conf), iou: CGFloat(iou), maxDet: maxDet, mode: nmsMode, sigma: Float(sigma))
            var masks: [MaskBitmap] = [], drawBoxes = true
            if let det, let raw, overlay != .boxes {
                masks = dets.compactMap { det.maskImage($0, raw) }
                drawBoxes = overlay != .masks
            }
            let annotated = annotate(cg, dets, names: names, style: style, label: label, masks: masks, drawBoxes: drawBoxes) ?? cg
            await MainActor.run {
                let panel = NSSavePanel(); panel.allowedContentTypes = [.jpeg, .png]
                panel.nameFieldStringValue = String(format: "frame_%.2fs.jpg", time)
                if panel.runModal() == .OK, let url = panel.url { saveCGImage(annotated, to: url) }
            }
        }
    }
    func reveal() { if let u = outputURL { NSWorkspace.shared.activateFileViewerSelecting([u]) } }
}

// ---------- Finder (Icons / List / Gallery) ----------
enum FinderMode: String, CaseIterable { case icons, list }

/// Icon-grid column count for the fixed 380-pt Finder pane. Used by BOTH the grid layout and
/// the arrow-key navigation: the old adaptive grid let SwiftUI pick a column count while the
/// key handler guessed with a different formula, so up/down could jump a wrong stride and the
/// selection drifted diagonally.
func finderCols(_ iconSize: Double) -> Int { max(1, Int((380.0 - 20 + 8) / (iconSize + 8))) }

struct FinderView: View {
    let images: [URL]
    @Binding var selected: Int
    @Binding var mode: FinderMode
    @Binding var iconSize: Double
    let onSelect: (Int) -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Picker("", selection: $mode) {
                    Image(systemName: "square.grid.2x2").tag(FinderMode.icons)
                    Image(systemName: "list.bullet").tag(FinderMode.list)
                }.pickerStyle(.segmented).labelsHidden().fixedSize()
                Spacer()
                Text("\(images.count) images").font(.caption).foregroundStyle(.secondary)
                if mode == .icons { Slider(value: $iconSize, in: 64...200).frame(width: 90) }
            }.padding(8)
            Divider()
            switch mode { case .icons: icons; case .list: list }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
    private var icons: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVGrid(columns: Array(repeating: GridItem(.fixed(iconSize), spacing: 8), count: finderCols(iconSize)), spacing: 8) {
                    ForEach(images.indices, id: \.self) { i in
                        VStack(spacing: 3) {
                            AsyncThumb(url: images[i], max: 220)
                                .frame(width: iconSize, height: iconSize * 0.72).clipped().cornerRadius(5)
                                .overlay(RoundedRectangle(cornerRadius: 5).stroke(i == selected ? brandColor : .clear, lineWidth: 3))
                            Text(images[i].lastPathComponent).font(.caption2).lineLimit(1).truncationMode(.middle).frame(width: iconSize)
                        }.contentShape(Rectangle()).onTapGesture { onSelect(i) }.id(i)
                    }
                }.padding(10)
            }
            .onChange(of: selected) { proxy.scrollTo(selected) }   // the view follows the highlight (Finder behavior)
            .onAppear { proxy.scrollTo(selected) }
        }
    }
    private var list: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 1) {
                    ForEach(images.indices, id: \.self) { i in
                        HStack(spacing: 8) {
                            AsyncThumb(url: images[i], max: 90).frame(width: 54, height: 38).clipped().cornerRadius(3)
                            Text(images[i].lastPathComponent).font(.callout).lineLimit(1).truncationMode(.middle)
                            Spacer(minLength: 0)
                        }
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(i == selected ? brandColor.opacity(0.25) : .clear)
                        .contentShape(Rectangle()).onTapGesture { onSelect(i) }.id(i)
                    }
                }
            }
            .onChange(of: selected) { proxy.scrollTo(selected) }   // the view follows the highlight
            .onAppear { proxy.scrollTo(selected) }
        }
    }
}

// ---------- AVPlayer video stage (real-time playback) + live detection overlay ----------
final class PlayerController: ObservableObject {
    let player = AVPlayer()
    @Published var currentTime: Double = 0     // playhead (drives the slider during playback)
    @Published var displayTime: Double = 0     // time of the frame actually ON SCREEN (drives the overlay)
    @Published var isPlaying = false
    private var loaded: URL?
    private var timeObs: Any?
    private var endObs: NSObjectProtocol?
    func load(_ url: URL) {
        guard loaded != url else { return }
        loaded = url
        player.replaceCurrentItem(with: AVPlayerItem(url: url))
        currentTime = 0; displayTime = 0; isPlaying = false
        if timeObs == nil {
            // 10 Hz: feeds the scrubber + stats/paused-compose triggers only. The playback
            // overlay is composed by the self-paced worker reading the player clock directly
            // and does not depend on this cadence.
            timeObs = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: 10), queue: .main) { [weak self] t in
                guard let self else { return }
                let s = t.seconds.isFinite ? t.seconds : 0
                self.currentTime = s
                if self.isPlaying { self.displayTime = s }   // during playback, boxes track the shown frame
            }
        }
        if endObs == nil {
            endObs = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: nil, queue: .main) { [weak self] _ in
                self?.player.seek(to: .zero); if self?.isPlaying == true { self?.player.play() }
            }
        }
    }
    func togglePlay() { isPlaying.toggle(); isPlaying ? player.play() : player.pause() }
    func pause() { isPlaying = false; player.pause() }
    func seek(_ t: Double) {
        player.seek(to: CMTime(seconds: max(0, t), preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] done in
            if done { DispatchQueue.main.async { self?.displayTime = t } }   // draw boxes only after the frame is on screen
        }
    }
    deinit { if let o = timeObs { player.removeTimeObserver(o) }; if let e = endObs { NotificationCenter.default.removeObserver(e) } }
}

final class PlayerContainer: NSView {
    let playerLayer = AVPlayerLayer()
    override func makeBackingLayer() -> CALayer { playerLayer.videoGravity = .resizeAspect; return playerLayer }
}
struct PlayerView: NSViewRepresentable {
    let player: AVPlayer
    func makeNSView(context: Context) -> PlayerContainer { let v = PlayerContainer(); v.wantsLayer = true; v.playerLayer.player = player; return v }
    func updateNSView(_ v: PlayerContainer, context: Context) { v.playerLayer.player = player }
}

let overlayPalette: [Color] = [
    Color(red: 0.98, green: 0.26, blue: 0.30), Color(red: 0.20, green: 0.71, blue: 0.98), Color(red: 0.16, green: 0.85, blue: 0.52),
    Color(red: 0.99, green: 0.79, blue: 0.12), Color(red: 0.72, green: 0.40, blue: 0.98), Color(red: 0.99, green: 0.55, blue: 0.18),
    Color(red: 0.10, green: 0.83, blue: 0.80), Color(red: 0.98, green: 0.36, blue: 0.66), Color(red: 0.55, green: 0.82, blue: 0.28),
    Color(red: 0.40, green: 0.52, blue: 0.98)]

/// AVPlayer plays the raw video (hardware-decoded, real-time); a Canvas overlays boxes for the
/// current play-head time from cached candidates, so playback is smooth AND boxes stay live-tunable.
struct VideoStage: View {
    @ObservedObject var engine: InferenceEngine
    @ObservedObject var pc: PlayerController
    let conf: Double, iou: Double
    let nmsMode: NMSMode, sigma: Double
    let maxDet: Int
    let overlay: SegOverlay
    let style: BoxStyle, label: LabelMode
    var body: some View {
        ZStack {
            PlayerView(player: pc.player)
            // The webcam model: the compositor publishes ONE transparent full-res overlay image
            // (boxes + labels + masks, rendered off-main); this canvas just blits the newest
            // one into the video's fitted rect. Redraw cost is a single image draw.
            Canvas { ctx, size in
                let vid = engine.videoSize
                guard vid.width > 0, vid.height > 0 else { return }
                let scale = Swift.min(size.width / vid.width, size.height / vid.height)
                let dw = vid.width * scale, dh = vid.height * scale
                let ox = (size.width - dw) / 2, oy = (size.height - dh) / 2
                if let ov = engine.videoOverlayImg {
                    ctx.draw(Image(decorative: ov, scale: 1), in: CGRect(x: ox, y: oy, width: dw, height: dh))
                }
            }
            .allowsHitTesting(false)
        }
    }
}

// ---------- main UI ----------
struct ContentView: View {
    @StateObject private var engine = InferenceEngine()
    // Default to the model bundled in the app (Resources); user can still pick another. nil under `swift run`.
    @State private var modelURL: URL? = Bundle.main.url(forResource: "v0.1-seg-N", withExtension: "mlpackage")
    @State private var sourceURL: URL?
    @State private var conf = 0.25
    @State private var iou = 0.50
    @State private var style: BoxStyle = .hud
    @State private var label: LabelMode = .full
    @State private var overlay: SegOverlay = .both     // segmentation: masks / boxes / both
    @State private var preprocess: Detector.PreprocessMode = .letterbox   // input fit: letterbox vs force-resize to imgsz
    @State private var tiling: TilingMode = .off       // tiled inference (images/folders only)
    @State private var tileSize = 640.0                // requested tile edge (clamped per image in Kit)
    @State private var tileCeil = 0                    // slider ceiling: max over source of shortSide/4 (0 = unknown)
    @State private var tilingMasks = false             // tiled modes: keep global-pass seg masks
    @State private var nmsMode: NMSMode = .standard    // global NMS variant (also the tiled merge)
    @State private var sigma = 0.1                     // CW-NMS gaussian width
    @State private var maxDet = 300.0                  // NMS keep cap (10...5000)
    @State private var sampling: VideoSampling = .onePerSecond   // annotation-export frame sampling
    @State private var showLabelExport = false    // "Export labels" popover
    @State private var showRenderExport = false   // "Export rendered images" popover
    @State private var compute: ComputeMode = .cpuAndGPU
    @State private var showPicker = false
    @State private var pickTarget: PickTarget = .model
    @State private var folderImages: [URL] = []
    @State private var sourceError: String?      // set when the chosen source is invalid (e.g. mixed folder)
    @State private var selectedIndex = 0
    @State private var finderMode: FinderMode = .icons
    @State private var iconSize: Double = 108
    @State private var videoDur = 0.0
    @State private var scrubTime = 0.0
    @State private var scrubbing = false
    @State private var wasPlaying = false
    @StateObject private var pc = PlayerController()
    @StateObject private var zoom = ZoomModel()   // image + paused-video magnification (stages are exclusive)
    @State private var cameraOn = false   // live-camera mode; the session lives in LiveCameraView (isolated observation)
    @State private var cameraIsSegment = false   // set by LiveCameraView once its detector is built
    @State private var cameraMirror = true       // live-camera selfie mirror (toggled from the stage)
    @State private var showInfo = false          // About & Licenses sheet
    @State private var trackMode = "off"         // video: off | bytetrack | botsort (ids over the cached candidates)
    @State private var preprocDevice: PreprocDevice = .gpu   // letterbox on the Metal GPU (default) or the CPU path
    // ---- the two major modes: Inference (everything of 1.1) and Bench (its own sidebar + dashboard) ----
    @State private var appMode: AppMode = .inference
    @StateObject private var bench = BenchModel()
    @State private var selectedBenchRecord: UUID?
    @FocusState private var kbFocused: Bool

    private enum PickTarget { case model, source }
    private var sourceKind: SourceKind { sourceURL.map(classifySource) ?? .unknown }
    /// The adjustable NMS cap is a tiled-inference feature (merged tile pools can
    /// legitimately exceed 300 boxes); every single-pass run keeps the default.
    private var effectiveMaxDet: Int {
        (tiling != .off && (sourceKind == .image || sourceKind == .folder) && !cameraOn) ? Int(maxDet) : 300
    }
    private var modelInfoImgsz: String { engine.modelInfo.map { "\($0.imgsz)×\($0.imgsz)" } ?? "the model's imgsz" }
    private var isSegModel: Bool { cameraOn ? cameraIsSegment : engine.modelIsSegment }   // drives the Overlay control in both modes
    private var kindLabel: String {
        switch sourceKind { case .image: "image"; case .folder: "folder"; case .video: "video"; case .unknown: "unsupported" }
    }
    private var pickerTypes: [UTType] {
        if pickTarget == .source { return [.image, .movie, .mpeg4Movie, .folder] }
        let byId = ["com.apple.coreml.mlpackage", "com.apple.coreml.mlmodelc", "com.apple.coreml.model"].compactMap { UTType($0) }
        let byExt = ["mlpackage", "mlmodelc", "mlmodel"].compactMap { UTType(filenameExtension: $0) }
        let all = byId + byExt + [.package]; return all.isEmpty ? [.item] : all
    }

    // The full modifier chain (~25 modifiers) exceeds the SwiftUI type-checker budget as ONE
    // expression ("unable to type-check this expression in reasonable time", pointing at
    // whichever modifier the budget dies on). Staging it through explicitly-typed computed
    // properties gives each stage its own budget. Behavior is identical.
    var body: some View { keyboardLayer }

    /// Stage 1: layout + panels (sheet / importer / drop).
    private var mainStage: some View {
        HStack(spacing: 0) {
            controls.frame(width: 300).padding(16)
            Divider()
            if appMode == .bench {
                BenchDashboard(bench: bench, store: bench.store, selectedRecord: $selectedBenchRecord, brand: brandColor)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                if !cameraOn && sourceKind == .folder && engine.hasResults && !engine.exporting {
                    FinderView(images: folderImages, selected: $selectedIndex, mode: $finderMode, iconSize: $iconSize) { selectAndShow($0) }
                        .frame(width: 380)
                    Divider()
                }
                VStack(spacing: 0) {
                    preview.frame(maxWidth: .infinity, maxHeight: .infinity)
                        .overlay(alignment: .bottom) { if engine.busy && !cameraOn { progressBar } }
                    if !cameraOn && sourceKind == .video && engine.hasResults && !engine.exporting { scrubberBar }
                }
            }
        }
        .onChange(of: appMode) {
            // Bench owns the machine while it runs: leave the camera and playback behind, and seed the
            // bench model list with the model Inference is using
            if appMode == .bench {
                if cameraOn { stopCamera() }
                if pc.isPlaying { pc.pause() }
                if let m = modelURL { bench.addModel(m) }
            }
        }
        .sheet(isPresented: $showInfo) { InfoView() }
        .fileImporter(isPresented: $showPicker, allowedContentTypes: pickerTypes) { if case .success(let u) = $0 { assign(u) } }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for p in providers { _ = p.loadObject(ofClass: URL.self) { url, _ in guard let url else { return }; DispatchQueue.main.async { assign(url) } } }
            return true
        }
    }

    /// Stage 2: cheap re-render observers (no re-inference).
    private var tuningObservers: some View {
        mainStage
            .onChange(of: conf) { rerender() }
            .onChange(of: iou) { rerender() }
            .onChange(of: maxDet) { rerender() }
            .onChange(of: nmsMode) { rerender() }
            .onChange(of: sigma) { if nmsMode == .clusterWeighted { rerender() } }
            .onChange(of: style) { rerender() }
            .onChange(of: label) { rerender() }
            .onChange(of: overlay) { rerender() }
            .onChange(of: trackMode) { engine.trackKind = TrackerKind(rawValue: trackMode); rerender() }
            .onChange(of: engine.trackGen) { if sourceKind == .video { refreshVideoOverlays() } }   // a tracked pass landed
    }

    /// Stage 3: observers that re-infer or re-target the source / follow playback.
    private var sourceObservers: some View {
        tuningObservers
            .onChange(of: preprocess) {   // preprocessing changes the forward pass -> re-infer (not a cheap re-render)
                if cameraOn { return }    // LiveCameraView hot-swaps the detector itself
                guard !engine.busy, engine.hasResults || engine.resultImage != nil else { return }
                runInfer()
            }
            .onChange(of: preprocDevice) {   // same: the tensor changes, so the cached forward passes do
                engine.preprocDevice = preprocDevice
                if cameraOn { return }
                guard !engine.busy, engine.hasResults || engine.resultImage != nil else { return }
                runInfer()
            }
            .onChange(of: tiling) {       // tiling changes the forward pass(es) -> re-infer, images/folders only
                guard !cameraOn, sourceKind == .image || sourceKind == .folder else { return }
                guard !engine.busy, engine.hasResults || engine.resultImage != nil else { return }
                runInfer()
            }
            .onChange(of: tilingMasks) {  // toggling mask retention changes what the tiled cache holds
                guard tiling != .off, !cameraOn, sourceKind == .image || sourceKind == .folder else { return }
                guard !engine.busy, engine.hasResults || engine.resultImage != nil else { return }
                runInfer()
            }
            .onChange(of: modelURL) { setupSource() }
            .onChange(of: sourceURL) { setupSource() }
            .onChange(of: scrubTime) { if scrubbing { pc.seek(scrubTime) } }   // seek while dragging
            .onChange(of: pc.currentTime) { if pc.isPlaying && !scrubbing { scrubTime = pc.currentTime } }   // slider follows playback
            .onChange(of: pc.displayTime) { refreshVideoOverlays() }
            .onChange(of: pc.isPlaying) {
                if pc.isPlaying {
                    zoom.reset()   // zoom is a paused-video feature
                    engine.startVideoOverlayLoop(player: pc.player, conf: conf, iou: iou, nmsMode: nmsMode,
                                                 sigma: sigma, maxDet: effectiveMaxDet, style: style, label: label, overlay: overlay)
                } else {
                    engine.stopVideoOverlayLoop()
                    engine.requestOverlayFrame(time: pc.displayTime, conf: conf, iou: iou, nmsMode: nmsMode,
                                               sigma: sigma, maxDet: effectiveMaxDet, style: style, label: label, overlay: overlay)
                }
            }
    }

    /// Stage 4: keyboard focus + shortcuts + tint.
    private var keyboardLayer: some View {
        sourceObservers
            .focusable().focused($kbFocused).focusEffectDisabled().onAppear { DispatchQueue.main.async { kbFocused = true } }
            .onKeyPress(.leftArrow)  { step(-1, vertical: false); return .handled }
            .onKeyPress(.rightArrow) { step(1,  vertical: false); return .handled }
            .onKeyPress(.upArrow)    { step(-1, vertical: true);  return .handled }
            .onKeyPress(.downArrow)  { step(1,  vertical: true);  return .handled }
            .onKeyPress(.space) { toggleVideoPlayback() }   // space toggles play/pause of the inferred video
            .tint(brandColor)   // teal accent for buttons/controls (Live Camera keeps its own pink tint)
    }

    private func assign(_ url: URL) {
        switch url.pathExtension.lowercased() {
        case "mlpackage", "mlmodelc", "mlmodel": modelURL = url
        default: sourceURL = url
        }
    }
    private func setupSource() {
        pc.pause()
        engine.resetResults()
        zoom.reset()
        recomputeTileCeil()
        sourceError = nil; folderImages = []
        guard let s = sourceURL else { return }
        switch classifySource(s) {
        case .folder:
            let others = folderNonImages(s)
            if !others.isEmpty {
                let sample = others.prefix(3).map { $0.lastPathComponent }.joined(separator: ", ")
                let more = others.count > 3 ? " (+\(others.count - 3) more)" : ""
                sourceError = "This folder isn’t images-only - it contains: \(sample)\(more). Pick a folder that holds only image files."
            } else {
                let imgs = listImages(s)
                if imgs.isEmpty { sourceError = "This folder has no images." }
                else { folderImages = imgs; selectedIndex = 0 }
            }
        case .video:
            pc.load(s)
            Task { let dur = await videoDuration(s); await MainActor.run { videoDur = dur; scrubTime = 0 } }
        case .unknown:
            sourceError = "Unsupported source. Choose an image, a video, or a folder of images."
        default: break
        }
    }
    private var tilingConfig: TilingConfig {
        TilingConfig(mode: tiling, tileSize: Int(tileSize), keepGlobalMasks: tilingMasks)
    }
    private func runInfer() {
        guard let m = modelURL, let s = sourceURL, sourceError == nil else { return }
        zoom.reset()
        switch sourceKind {
        case .image:  engine.previewURL(model: m, image: s, compute: compute, conf: conf, iou: iou, style: style, label: label, overlay: overlay, preprocess: preprocess,
                                        tiling: tilingConfig, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet)
        case .folder: engine.runFolder(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label, overlay: overlay, preprocess: preprocess,
                                       tiling: tilingConfig, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet)
        case .video:  engine.runVideo(model: m, input: s, compute: compute, conf: conf, iou: iou, style: style, label: label, preprocess: preprocess, overlay: overlay,
                                      nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet)
        default: break
        }
    }
    // ---- live camera (session lifecycle + detector build handled inside LiveCameraView) ----
    private func toggleVideoPlayback() -> KeyPress.Result {   // extracted so the view body type-checks
        guard appMode == .inference, sourceKind == .video, engine.hasResults, !cameraOn else { return .ignored }
        pc.togglePlay()
        return .handled
    }
    private func startCamera() { guard modelURL != nil, !engine.busy else { return }; pc.pause(); zoom.reset(); cameraOn = true }
    private func stopCamera() { cameraOn = false }

    private func selectAndShow(_ i: Int) {
        guard folderImages.indices.contains(i) else { return }
        selectedIndex = i
        zoom.reset()
        engine.showFolder(index: i, url: folderImages[i], conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet)
    }
    /// Re-derive the shown video frame's stats + seg-mask overlay from the cache. Extracted from
    /// the body's onChange chain: inline, these two many-argument calls blow the SwiftUI
    /// type-checker budget ("unable to type-check this expression in reasonable time").
    private func refreshVideoOverlays() {
        guard sourceKind == .video, engine.hasResults else { return }
        let t: Double = pc.displayTime
        // Tracking is a pure function of the cache + settings: keep the tracked pass current (a cheap
        // key compare when nothing changed; settings are locked during playback anyway).
        engine.ensureTracked(conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet)
        engine.setVideoFrameStats(time: t, conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet, throttled: pc.isPlaying)
        if !pc.isPlaying {   // paused/scrub: one full-detail compose; the loop owns playback
            engine.requestOverlayFrame(time: t, conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet,
                                       style: style, label: label, overlay: overlay)
        }
    }
    private func rerender() {
        if cameraOn { return }   // camera overlay reads conf/iou/style/label live - no engine re-render
        if sourceKind == .video {
            refreshVideoOverlays()   // overlay redraws on conf/iou/label automatically
        } else {
            engine.restyle(conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet)
        }
    }
    private func step(_ dir: Int, vertical: Bool) {
        if appMode == .bench {   // up / down move the inspected cell through the results table / side-by-side bars
            guard vertical, !bench.running else { return }
            let cells = selectedBenchRecord.flatMap { id in bench.store.records.first { $0.id == id }?.cells } ?? bench.cells
            let ids = cells.map(\.id)
            guard !ids.isEmpty else { return }
            let i = bench.selectedCellID.flatMap { ids.firstIndex(of: $0) } ?? 0
            bench.selectedCellID = ids[max(0, min(ids.count - 1, i + dir))]
            return
        }
        switch sourceKind {
        case .folder where engine.hasResults && !folderImages.isEmpty:
            // Finder icon-view semantics: left/right move within the CURRENT ROW only (no
            // wrapping onto the next row); up/down move within the CURRENT COLUMN only.
            // List view: any arrow steps the list.
            let n = folderImages.count
            var target = selectedIndex
            if finderMode == .icons {
                let cols = finderCols(iconSize)
                if vertical {
                    let t = selectedIndex + dir * cols
                    if t >= 0 && t < n { target = t }
                } else {
                    let col = selectedIndex % cols
                    let t = selectedIndex + dir
                    if t >= 0 && t < n && ((dir > 0 && col < cols - 1) || (dir < 0 && col > 0)) { target = t }
                }
            } else {
                let t = selectedIndex + dir
                if t >= 0 && t < n { target = t }
            }
            if target != selectedIndex { selectAndShow(target) }
        case .video where engine.hasResults:
            scrubTime = min(max(0, scrubTime + Double(dir) * (vertical ? 1.0 : 0.2)), max(videoDur, 0.0))
            pc.seek(scrubTime)
        default: break
        }
    }

    private var controls: some View {
        VStack(spacing: 14) {
            HStack(spacing: 10) {
                Image(nsImage: appMark())
                    .resizable().frame(width: 30, height: 30)
                VStack(alignment: .leading, spacing: 0) {
                    Text("YOLO-Master").font(.headline)
                    Text("Core ML runner").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button { showInfo = true } label: { Image(systemName: "info.circle").font(.system(size: 16)) }
                    .buttonStyle(.borderless).help("About & Licenses")
            }
            SegmentedButtons(options: AppMode.allCases.map { ($0, $0.rawValue) }, selection: $appMode, tint: brandColor)
            .disabled(engine.busy || bench.running)

            if appMode == .bench {
                BenchSidebar(bench: bench, store: bench.store, selectedRecord: $selectedBenchRecord, brand: brandColor)
                Text("© 2026 Thomas Li").font(.system(size: 9)).foregroundStyle(.tertiary)
            } else {
            ScrollView {
                VStack(spacing: 14) {
                    sectionBox("Files", "folder") {
                        fileRow(icon: "cube.box.fill", title: "Model",
                                value: modelURL?.lastPathComponent ?? "Choose .mlpackage…", set: modelURL != nil) {
                            pickTarget = .model; DispatchQueue.main.async { showPicker = true }
                        }
                        Divider()
                        fileRow(icon: "photo.on.rectangle.angled", title: "Source",
                                value: sourceURL.map { "\($0.lastPathComponent) · \(kindLabel)" } ?? "Choose image / folder / video…",
                                set: sourceURL != nil) {
                            pickTarget = .source; DispatchQueue.main.async { showPicker = true }
                        }
                    }
                    sectionBox("Preprocess", "aspectratio") {
                        segRow("Input fit") {
                            SegmentedButtons(options: [(Detector.PreprocessMode.letterbox, "Letterbox"), (Detector.PreprocessMode.stretch, "Stretch")],
                                             icons: ["rectangle.inset.filled", "arrow.left.and.right.square"], selection: $preprocess, tint: brandColor)
                                .disabled(cameraOn)
                        }
                        segRow("Device") {
                            SegmentedButtons(options: [(PreprocDevice.gpu, "GPU"), (PreprocDevice.cpu, "CPU")], icons: ["rectangle.stack.fill", "cpu"], selection: $preprocDevice, tint: brandColor)
                        }
                    }
                    sectionBox("Slicing", "square.grid.3x3") {
                        segRow("Mode") {
                            SegmentedButtons(options: TilingMode.allCases.map { ($0, $0.label) }, selection: $tiling, tint: brandColor)
                                .disabled(cameraOn || sourceKind == .video)
                        }
                        if tiling != .off && !(cameraOn || sourceKind == .video) {
                            tileSizeRow
                            if engine.modelIsSegment {
                                segRow("Masks (global pass)") {
                                    SegmentedButtons(options: [(false, "Off"), (true, "On")], icons: ["square.dashed", "square.fill.on.square"], selection: $tilingMasks, tint: brandColor)
                                }
                                Text("Masks come from the full-image pass; tile detections stay boxes-only.")
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                        if cameraOn || sourceKind == .video {
                            Text("Slicing applies to images and folders only.")
                                .font(.caption2).foregroundStyle(.secondary)
                        } else if tiling == .sparse {
                            Text("Global pass + \(modelInfoImgsz) tiles where the global pass found objects; runs every tile if it found none.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    sectionBox("Detection", "slider.horizontal.3") {
                        if videoTuningLocked {
                            Text("Pause the video to tune detection settings.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        sliderRow("Confidence", $conf, 0.05...0.95).disabled(videoTuningLocked)
                        sliderRow("IoU", $iou, 0.10...0.90).disabled(videoTuningLocked)
                        if tiling != .off {
                            intSliderRow("Max detections", $maxDet, 10...5000).disabled(videoTuningLocked)
                            Text("Tiled runs only: caps the merged tile pool after NMS. Single-pass runs keep 300.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        segRow("NMS") {
                            SegmentedButtons(options: NMSMode.allCases.map { ($0, $0.label) }, icons: ["rectangle.on.rectangle", "rectangle.3.group"], selection: $nmsMode, tint: brandColor)
                                .disabled(videoTuningLocked)
                        }
                        if nmsMode == .clusterWeighted {
                            sliderRow("Sigma", $sigma, 0.01...0.5).disabled(videoTuningLocked)
                            Text("Survivor boxes are refined by score-weighted averaging over overlapping candidates.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        if sourceKind == .video && !cameraOn {
                            segRow("Tracking") {
                                SegmentedButtons(options: [("off", "Off"), ("bytetrack", "ByteTrack"), ("botsort", "BoT-SORT")], selection: $trackMode, tint: brandColor)
                                    .disabled(videoTuningLocked)
                            }
                            if trackMode != "off" {
                                Text(trackMode == "botsort"
                                     ? "Kalman + two-stage IoU association with camera-motion compensation (Vision); ids persist across frames and the export."
                                     : "Kalman + two-stage IoU association; ids persist across frames and the export.")
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                    sectionBox("Appearance", "paintbrush.fill") {
                        if isSegModel && tiledActive && !tilingMasks {
                            Text("Masks are off in sliced modes - enable \"Masks (global pass)\" in Slicing.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        if isSegModel && (!tiledActive || tilingMasks) {
                            segRow("Overlay") {
                                SegmentedButtons(options: SegOverlay.allCases.map { ($0, $0.rawValue.capitalized) }, selection: $overlay, tint: brandColor)
                            }
                        }
                        if !(isSegModel && overlay == .masks) {   // box style is irrelevant with boxes hidden
                            segRow("Box style") {
                                SegmentedButtons(options: BoxStyle.allCases.map { ($0, $0.rawValue.capitalized) }, selection: $style, tint: brandColor)
                            }
                        }
                        segRow("Label") {   // labels stay adjustable even in masks-only mode
                            SegmentedButtons(options: LabelMode.allCases.map { ($0, $0.rawValue.capitalized) }, selection: $label, tint: brandColor)
                        }
                    }
                    sectionBox("Device", "cpu") {
                        MenuButton(options: [ComputeMode.cpuAndGPU, .all, .cpu].map { ($0, $0.label) }, icons: ["rectangle.stack.fill", "sparkles", "cpu"], selection: $compute, tint: brandColor)
                            .disabled(cameraOn)
                        if cameraOn {
                            Text("Stop the camera to change the compute backend.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    sectionBox("System", "gauge.with.dots.needle.33percent") { MetersStrip(meters: bench.meters) }
                    sectionBox("Inference", "chart.bar.doc.horizontal") { summaryContent }
                }
            }
            .disabled(engine.busy)   // lock every control during image/folder/video inference; tune after it finishes (camera isn't engine.busy)

            actionRow
            Text("© 2026 Thomas Li").font(.system(size: 9)).foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder private var actionRow: some View {
        VStack(spacing: 8) {
            if cameraOn {
                primaryButton("Stop Camera", "stop.fill") { stopCamera() }
            } else {
                sourceActions
                VStack(spacing: 4) {
                    Button { startCamera() } label: {
                        Label("Live Camera", systemImage: "camera.fill").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent).tint(.pink).controlSize(.large)
                    .disabled(modelURL == nil || engine.busy)
                    if modelURL == nil {
                        Text("Load a model to enable the live camera").font(.caption2).foregroundStyle(.secondary)
                    } else if engine.busy {
                        Text("Finish or wait for the current inference before starting the camera.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    @ViewBuilder private var sourceActions: some View {
        VStack(spacing: 8) {
            switch sourceKind {
            case .image:
                primaryButton("Run", "play.fill") { runInfer() }.disabled(sourceURL == nil || engine.busy || sourceError != nil)
                secondaryButton("Save…", "square.and.arrow.down") { engine.save() }.disabled(engine.resultImage == nil)
                exportLabelsMenu.disabled(engine.resultImage == nil || engine.busy)
            case .folder:
                primaryButton(engine.hasResults ? "Re-run inference" : "Run inference", "play.fill") { runInfer() }
                    .disabled(sourceURL == nil || engine.busy || sourceError != nil)
                HStack(spacing: 8) {
                    secondaryButton("Export rendered images", "photo.on.rectangle") { showRenderExport = true }
                        .popover(isPresented: $showRenderExport, arrowEdge: .trailing) {
                            VStack(alignment: .leading, spacing: 2) {
                                popoverRow("This frame") { showRenderExport = false; engine.save() }
                                popoverRow("All") {
                                    showRenderExport = false
                                    engine.exportFolder(conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet)
                                }
                            }
                            .padding(8).frame(minWidth: 150)
                        }
                        .disabled(!engine.hasResults || engine.busy)
                    if engine.outputURL != nil { revealButton }
                }
                exportLabelsMenu.disabled(!engine.hasResults || engine.busy)
            case .video:
                primaryButton(engine.hasResults ? "Re-run inference" : "Run inference", "play.fill") { runInfer() }
                    .disabled(sourceURL == nil || engine.busy || sourceError != nil)
                HStack(spacing: 8) {
                    secondaryButton("Export rendered", "photo.on.rectangle") { showRenderExport = true }
                        .popover(isPresented: $showRenderExport, arrowEdge: .trailing) {
                            VStack(alignment: .leading, spacing: 2) {
                                popoverRow("This frame") {
                                    showRenderExport = false
                                    engine.saveVideoFrame(time: pc.displayTime, conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet)
                                }
                                popoverRow("All (annotated video)") {
                                    showRenderExport = false
                                    engine.exportVideo(conf: conf, iou: iou, style: style, label: label, overlay: overlay, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet)
                                }
                            }
                            .padding(8).frame(minWidth: 180)
                        }
                        .disabled(!engine.hasResults || engine.busy)
                    if engine.outputURL != nil { revealButton }
                }
                exportLabelsMenu.disabled(!engine.hasResults || engine.busy)
            case .unknown:
                EmptyView()
            }
        }
    }

    /// Annotation export: a real full-width button (identical chrome/width to every other action
    /// button - macOS `Menu` hugs its content and never honors a full-width label) opening a
    /// popover with the three formats. Video exports honor the scrubber-bar sampling picker.
    @ViewBuilder private var exportLabelsMenu: some View {
        secondaryButton("Export labels", "doc.badge.arrow.up") { showLabelExport = true }
            .popover(isPresented: $showLabelExport, arrowEdge: .trailing) {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(AnnotationFormat.allCases, id: \.self) { f in
                        popoverRow(f.label) {
                            showLabelExport = false
                            engine.exportAnnotations(format: f, sampling: sampling,
                                                     conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet)
                        }
                    }
                    if sourceKind == .video {
                        Divider().padding(.vertical, 2)
                        Text("Sampling: \(sampling.label) - change in the scrubber bar")
                            .font(.caption2).foregroundStyle(.secondary).padding(.horizontal, 8)
                    }
                }
                .padding(8).frame(minWidth: 190)
            }
    }

    /// A menu-item-like row for action popovers.
    private func popoverRow(_ title: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title).frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 8).padding(.vertical, 5)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
    }

    private func primaryButton(_ title: String, _ icon: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) { Label(title, systemImage: icon).frame(maxWidth: .infinity) }
            .buttonStyle(.borderedProminent).controlSize(.large)
    }
    private func secondaryButton(_ title: String, _ icon: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) { Label(title, systemImage: icon).frame(maxWidth: .infinity) }.controlSize(.large)
    }
    private var revealButton: some View {
        Button { engine.reveal() } label: { Image(systemName: "magnifyingglass") }.controlSize(.large)
    }

    private var progressBar: some View {
        VStack(spacing: 4) {
            if let p = engine.progress { ProgressView(value: p) } else { ProgressView().progressViewStyle(.linear) }
            Text(engine.status).font(.caption)
        }.padding(10).frame(maxWidth: .infinity).background(.ultraThinMaterial)
    }

    private var preview: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor)
            if cameraOn {
                LiveCameraView(modelURL: modelURL, compute: compute, preprocess: preprocess, preprocDevice: preprocDevice,
                               conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet,
                               overlay: overlay, style: style, label: label,
                               isSegment: $cameraIsSegment, mirror: $cameraMirror).padding(12)
            } else if let err = sourceError {
                VStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 44)).foregroundStyle(.orange)
                    Text(err).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 380)
                }.padding(24)
            } else if sourceKind == .video && engine.hasResults {
                ZoomContainer(zoom: zoom, enabled: !pc.isPlaying) {
                    VideoStage(engine: engine, pc: pc, conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: effectiveMaxDet, overlay: overlay, style: style, label: label).padding(12)
                }
            } else if let img = engine.resultImage {
                ZoomContainer(zoom: zoom) {
                    Image(nsImage: img).resizable().scaledToFit().padding(12)
                }
            } else if (sourceKind == .folder || sourceKind == .video) && !engine.hasResults && !engine.busy {
                VStack(spacing: 8) {
                    Image(systemName: sourceKind == .video ? "film" : "folder").font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text(sourceKind == .video ? "Press Run to infer the video"
                                              : "\(folderImages.count) images - press Run to infer").foregroundStyle(.secondary)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: sourceKind == .video ? "film" : "photo").font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text(sourceURL != nil ? "Press Run"
                         : modelURL == nil ? "Choose a model + source"
                         : "Choose an image / folder / video - or start Live Camera").foregroundStyle(.secondary)
                }
            }
        }
    }

    private var scrubberBar: some View {
        HStack(spacing: 12) {
            Button { pc.togglePlay() } label: {
                Image(systemName: pc.isPlaying ? "pause.fill" : "play.fill").font(.title3).frame(width: 22)
            }.buttonStyle(.borderless)
            VStack(spacing: 2) {
                Slider(value: $scrubTime, in: 0...max(videoDur, 0.01)) { editing in
                    scrubbing = editing
                    if editing { wasPlaying = pc.isPlaying; pc.pause() }
                    else { pc.seek(scrubTime); if wasPlaying { pc.togglePlay() } }
                }
                Text("\(String(format: "%.2f", scrubTime))s / \(String(format: "%.1f", videoDur))s")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 2) {   // annotation-export frame sampling
                Text("Label sampling").font(.caption2).foregroundStyle(.secondary)
                Picker("", selection: $sampling) {
                    Text("Every frame").tag(VideoSampling.allFrames)
                    Text("1 / second").tag(VideoSampling.onePerSecond)
                    Text("Every 5th").tag(VideoSampling.everyNth(5))
                    Text("Every 10th").tag(VideoSampling.everyNth(10))
                    Text("Every 30th").tag(VideoSampling.everyNth(30))
                }.pickerStyle(.menu).labelsHidden().frame(width: 130)
            }
        }.padding(.horizontal, 12).padding(.vertical, 8).background(Color(nsColor: .windowBackgroundColor))
    }

    private func sectionBox<C: View>(_ title: String, _ icon: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon)                       // title: left edge aligns with the card below
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                .padding(.leading, 2)
            VStack(alignment: .leading, spacing: 14) { content() }   // roomier gap between option rows
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)                                          // breathing room inside the card
                .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
        }
    }
    private func fileRow(icon: String, title: String, value: String, set: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: icon).font(.system(size: 15)).foregroundStyle(set ? brandColor : .secondary).frame(width: 20)
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.caption).foregroundStyle(.secondary)
                    Text(value).font(.callout).lineLimit(1).truncationMode(.middle).foregroundStyle(set ? Color.primary : .secondary)
                }
                Spacer(minLength: 4)
                Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())   // whole row is the hit target, not just the text
        }.buttonStyle(.plain)
    }
    private func sliderRow(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(title).font(.callout)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue)).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    .padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule())
            }
            Slider(value: value, in: range)
        }
    }
    private func intSliderRow(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>, step: Double = 10) -> some View {
        // no Slider step: parameter - stepped sliders draw tick marks on macOS (an ugly
        // dashed line under the track at 500 ticks); round in the binding instead.
        let rounded = Binding<Double>(get: { value.wrappedValue },
                                      set: { value.wrappedValue = ($0 / step).rounded() * step })
        return VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(title).font(.callout)
                Spacer()
                Text("\(Int(value.wrappedValue))").font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    .padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule())
            }
            Slider(value: rounded, in: range)
        }
    }
    /// Tile-size slider. Bound: [model imgsz, shortSide/4 of the source] (Kit re-clamps per
    /// image). Commits on slider RELEASE - each change re-runs tiled inference, so per-tick
    /// re-inference during a drag would be a storm of forwards.
    @ViewBuilder private var tileSizeRow: some View {
        let degenerate = tileCeil <= tileFloor
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text("Tile size").font(.callout)
                Spacer()
                Text("\(Int(tileSize)) px").font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    .padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule())
            }
            Slider(value: $tileSize, in: tileRange) { editing in
                if !editing {
                    guard !engine.busy, engine.hasResults || engine.resultImage != nil else { return }
                    runInfer()
                }
            }
            .disabled(degenerate)
            Text(degenerate
                 ? "The input dimensions are too small for larger tiles. Slicing runs at the model input (\(tileFloor) px)."
                 : "Min = model input (\(tileFloor) px) · max = 1/4 of the source's short side (\(tileCeil) px). Larger tiles run faster but see less detail.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }
    private func segRow<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 4) { Text(title).font(.callout); content() }
    }

    // ---- inference summary panel ----
    @ViewBuilder private var summaryContent: some View {
        if let s = engine.infer {
            if let info = engine.modelInfo {
                statRow("Model", info.name)
                statRow("Input", "\(info.imgsz) × \(info.imgsz) px")
                statRow("Classes", "\(info.nc)")
                statRow("Compute", info.compute)
            }
            Divider()
            statRow(isVideoSource ? "Frames" : (s.count > 1 ? "Images" : "Frame"), "\(s.count)")
            statRow("Model-only", speedText(s.meanMs, s.fps))
            if s.preMeanMs > 0 { statRow("Preprocess", String(format: "%.2f ms", s.preMeanMs)) }
            if s.postMeanMs > 0 { statRow("Postprocess", String(format: "%.2f ms", s.postMeanMs)) }
            statRow("Overall", speedText(s.wallMeanMs, s.wallFps))
            if s.count > 1 {
                statRow("Model min/max", String(format: "%.1f / %.1f ms", s.minMs, s.maxMs))
                statRow("Total time", String(format: "%.2fs wall · %.2fs model", s.wallMs / 1000, s.totalMs / 1000))
            }
            if let t = engine.tileStats {
                Divider()
                statRow("Slicing", tiling.label)
                statRow("Tile size", t.tileSizeLabel + " px")
                statRow("Tiles", "\(t.tilesRun) run / \(t.tilesTotal) grid" + (t.capped > 0 ? " (capped)" : ""))
                if t.fallbacks > 0 { statRow("Fallback", "\(t.fallbacks) image(s) ran all tiles") }
            }
            if nmsMode == .clusterWeighted { statRow("NMS", nmsMode.label) }
            Divider()
            if engine.trackKind != nil, isVideoSource { statRow("Tracks", "\(engine.trackCount) ids (\(trackMode))") }
            statRow("Detections", "\(engine.detCount)  (this frame)")
            if !engine.classCounts.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(engine.classCounts.prefix(12)) { c in
                        HStack {
                            Text(c.name).font(.caption)
                            Spacer(minLength: 8)
                            Text("\(c.count)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        }
                    }
                }.padding(.top, 2)
            }
        } else {
            HStack { Image(systemName: "info.circle").foregroundStyle(.tertiary)
                     Text("Run inference to see stats.").font(.caption).foregroundStyle(.secondary) }
        }
    }

    private var isVideoSource: Bool { sourceKind == .video }
    /// Slider floor: the model input size (the tile-size lower bound). 640 until a model reports.
    private var tileFloor: Int { engine.modelInfo?.imgsz ?? 640 }
    /// Slider range honoring the bound "no less than model res, no bigger than shortSide/4".
    /// When the source is too small (ceil < floor) the floor wins and the slider is moot.
    private var tileRange: ClosedRange<Double> {
        let lo = Double(tileFloor)
        return lo...Swift.max(lo, Double(tileCeil))
    }
    /// Recompute the tile-size ceiling for the current source: shortSide/4 for an image, the MAX
    /// of shortSide/4 across a folder (per-image clamping in Kit enforces each image's own bound).
    private func recomputeTileCeil() {
        tileCeil = 0
        guard let src = sourceURL else { return }
        let kind = sourceKind
        Task.detached(priority: .utility) {
            // computed as a let so the MainActor hop captures an immutable value (Swift 6 ready)
            let ceil: Int = {
                if kind == .image, let d = imagePixelSize(src) { return Swift.min(d.w, d.h) / 4 }
                if kind == .folder {
                    return listImages(src).reduce(0) { acc, u in
                        guard let d = imagePixelSize(u) else { return acc }
                        return Swift.max(acc, Swift.min(d.w, d.h) / 4)
                    }
                }
                return 0
            }()
            await MainActor.run {
                tileCeil = ceil
                tileSize = Swift.min(Swift.max(tileSize, Double(tileFloor)), tileRange.upperBound)
            }
        }
    }
    /// Tiled modes affect only image/folder sources; video/camera keep single-pass masks.
    private var tiledActive: Bool { tiling != .off && !cameraOn && (sourceKind == .image || sourceKind == .folder) }
    /// Detection tuning is FROZEN while an inferred video plays: playback reads a pre-baked
    /// whole-video post-NMS array (pure index + draw); pause to tune, the bake refreshes then.
    private var videoTuningLocked: Bool { !cameraOn && sourceKind == .video && engine.hasResults && pc.isPlaying }
    private func speedText(_ ms: Double, _ fps: Double) -> String {
        String(format: "%.1f", ms) + (isVideoSource ? " ms/frame · " : " ms/img · ")
            + String(format: "%.1f", fps) + (isVideoSource ? " fps" : " img/s")
    }
    private func statRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(value).font(.caption.monospacedDigit()).lineLimit(1).truncationMode(.middle)
        }
    }
}
