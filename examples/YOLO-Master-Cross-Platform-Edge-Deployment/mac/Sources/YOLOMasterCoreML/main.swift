// yolomaster-coreml - command-line Core ML runner. Thin CLI over YOLOMasterKit
// (shared inference backend + folder/video pipelines).
//
// --source: image | folder/ | video.(mp4|mov|m4v) - mode auto-detected. Modes:
//   image   -> annotated image (--out out.jpg)
//   folder  -> annotated folder (--out preds/) + batch timing
//   video   -> annotated video  (--out out.mp4), size/fps preserved
//   --benchmark -> model-only latency (percentiles + img/s)
// Compute defaults cpuAndGPU (the ANE can crash on this fragmented MoE graph); --compute ane|all|cpu.
//
// Build:  swift build -c release --package-path mac
import Foundation
import CoreGraphics
import YOLOMasterKit

// ---------- args ----------
func argValue(_ name: String, _ def: String? = nil) -> String? {
    let a = CommandLine.arguments
    if let i = a.firstIndex(of: name), i + 1 < a.count { return a[i + 1] }
    return def
}
func hasFlag(_ name: String) -> Bool { CommandLine.arguments.contains(name) }
func die(_ msg: String, _ code: Int32 = 1) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!); exit(code)
}
func logErr(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
func f1(_ v: Double) -> String { String(format: "%.1f", v) }
func fps(_ ms: Double) -> String { f1(ms > 0 ? 1000 / ms : 0) }

// --core-selftest: handshake with the portable C++ core (no model needed); exits 0 on known answers
if hasFlag("--core-selftest") {
    let r = CoreSelfTest.run()
    print(r.report)
    exit(r.ok ? 0 : 1)
}

guard let modelPath = argValue("--model"), let srcPath = argValue("--source") else {
    die("usage: yolomaster-coreml --model M.mlpackage --source img|dir/|vid.mp4 [--out o] " +
        "[--conf 0.25] [--iou 0.5] [--compute cpuAndGPU|ane|all|cpu] [--style hud|solid|neon] " +
        "[--label full|min|off] [--resize N] [--limit N] [--no-save] [--save-txt DIR] " +
        "[--slicing off|dense|sparse [--tile-size N] [--slicing-masks] [--max-det N]] [--cw-nms [--sigma 0.1]] " +
        "[--bench off|cold|sustained] [--bench-iters 50] [--bench-warmup 10] [--bench-minutes 2] [--bench-json PATH] " +
        "[--accuracy auto|LABELS_DIR] [--track off|botsort|bytetrack] [--track-buffer 30] " +
        "[--cpu-preproc] [--dump-input DIR] " +
        "[--benchmark [--iters 200]]  (legacy alias of --bench cold)", 2)
}
let conf = Float(argValue("--conf", "0.25")!) ?? 0.25
let iouT = CGFloat(Float(argValue("--iou", "0.5")!) ?? 0.5)
let outArg = argValue("--out")
let compute = ComputeMode(argValue("--compute", "cpuAndGPU")!)
let benchmark = hasFlag("--benchmark")
let noSave = hasFlag("--no-save")
let iters = Int(argValue("--iters", "200")!) ?? 200
let resize = Int(argValue("--resize", "0")!) ?? 0
let boxStyle = BoxStyle(rawValue: (argValue("--style", "hud")!).lowercased()) ?? .hud
let maxDet = Int(argValue("--max-det", "300")!) ?? 300
let labelMode = LabelMode(rawValue: (argValue("--label", "full")!).lowercased()) ?? .full
// --slicing is the user-facing name; --tiling accepted as a compat alias
let slicingArg = argValue("--slicing") ?? argValue("--tiling", "off")!
let tilingMode = TilingMode(rawValue: slicingArg.lowercased()) ?? .off
// --tile-size: clamped per image in Kit to [model imgsz, max(imgsz, shortSide/4)]
let tileSizeArg = Int(argValue("--tile-size", "0")!) ?? 0
let tilingCfg = TilingConfig(mode: tilingMode, tileSize: tileSizeArg > 0 ? tileSizeArg : nil,
                             keepGlobalMasks: hasFlag("--slicing-masks") || hasFlag("--tiling-masks"))
let nmsMode: NMSMode = hasFlag("--cw-nms") ? .clusterWeighted : .standard
// the adjustable cap is a tiled-inference feature (merged tile pools can exceed 300)
if hasFlag("--max-det") && tilingMode == .off {
    die("--max-det is only available with tiled inference (add --slicing dense|sparse)", 2)
}
let sigma = Float(argValue("--sigma", "0.1")!) ?? 0.1
let limit = Int(argValue("--limit", "0")!) ?? 0
let saveTxt = argValue("--save-txt")
// bench mode: the Linux CLI's flags; the old --benchmark [--iters N] is an alias of --bench cold
var benchMode = BenchMode(rawValue: (argValue("--bench", "off")!).lowercased()) ?? .off
let benchIters = Int(argValue("--bench-iters") ?? (benchmark ? String(iters) : "50")) ?? 50
let benchWarmup = Int(argValue("--bench-warmup", "10")!) ?? 10
let benchMinutes = Double(argValue("--bench-minutes", "2")!) ?? 2
let benchJson = argValue("--bench-json")
let accuracy = argValue("--accuracy")          // "auto" | labels dir; implies --bench cold
if benchmark && benchMode == .off { benchMode = .cold }
if accuracy != nil && benchMode == .off { benchMode = .cold }
let trackArg = (argValue("--track", "off")!).lowercased()
let trackKind: TrackerKind? = trackArg == "off" ? nil : TrackerKind(rawValue: trackArg)
if trackArg != "off" && trackKind == nil { die("--track must be off|botsort|bytetrack", 2) }
let trackBuffer = Int(argValue("--track-buffer", "30")!) ?? 30
let cpuPreproc = hasFlag("--cpu-preproc")          // default: Metal letterbox when a GPU exists
let dumpInput = argValue("--dump-input")           // raw float32 NCHW input tensors, one .f32 per image (parity check)

// ---------- backend (shared) ----------
let detector: Detector
do { detector = try Detector(modelURL: URL(fileURLWithPath: modelPath), compute: compute,
                             forceCompute: CommandLine.arguments.contains("--compute")) }
catch { die("model load failed: \(error)", 3) }
detector.preprocDevice = cpuPreproc ? .cpu : .gpu
print("[model] \(detector.summary) preproc=\(detector.effectivePreprocDevice.rawValue)")
if let dumpInput {
    // the tensor the model would see, for scripts/preproc_compare.py against the Linux preprocess_nchw
    let dir = URL(fileURLWithPath: dumpInput)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let src0 = URL(fileURLWithPath: srcPath)
    var list = classifySource(src0) == .folder ? listImages(src0) : [src0]
    if limit > 0 && list.count > limit { list = Array(list.prefix(limit)) }
    var n = 0
    for u in list {
        guard let cg = loadCGImage(u), let bytes = detector.inputTensorBytes(cg) else { continue }
        let stem = u.deletingPathExtension().lastPathComponent
        try? bytes.write(to: dir.appendingPathComponent(stem + ".f32"))
        // the decoded source pixels the preprocessing saw (ImageIO's JPEG decoder differs from libjpeg by a
        // few levels), so scripts/preproc_compare.py can isolate the kernel from the decoder
        if let sp = MetalPreprocessor.sourcePixels(cg) {
            var tight = Data(capacity: sp.width * sp.height * 4)
            sp.data.withUnsafeBytes { raw in
                for y in 0..<sp.height { tight.append(raw.baseAddress!.advanced(by: y * sp.bytesPerRow).assumingMemoryBound(to: UInt8.self), count: sp.width * 4) }
            }
            try? tight.write(to: dir.appendingPathComponent("\(stem).\(sp.width)x\(sp.height).\(sp.bgra ? "bgra" : "rgba")"))
        }
        n += 1
    }
    print("[dump-input] \(n) tensors (\(detector.imgsz)x\(detector.imgsz) float32 NCHW, preproc=\(detector.effectivePreprocDevice.rawValue)) -> \(dir.path)")
}

// ---------- shared per-frame bookkeeping (txt dumps, bench samples, the Linux [summary] line) ----------
var samples = BenchSamples()
var txtWriter: TxtDumpWriter? = nil
if let saveTxt {
    do { txtWriter = try TxtDumpWriter(directory: URL(fileURLWithPath: saveTxt)) }
    catch { die("cannot create --save-txt directory \(saveTxt): \(error)", 4) }
}
let tStart = Date()
func record(_ stem: String, _ res: Detector.Result) {
    samples.add(res)
    txtWriter?.write(stem: stem, res.detections)
}
/// `%g`-style like the C++ ostream (six significant digits) for the summary line.
func g(_ v: Double) -> String { String(format: "%g", v) }
func printSummary() {
    guard samples.frames > 0 else { logErr("no frames processed"); exit(5) }
    let wall = Date().timeIntervalSince(tStart)
    let n = Double(samples.frames)
    let pre = samples.pre.reduce(0, +) / n, inf = samples.infer.reduce(0, +) / n, post = samples.post.reduce(0, +) / n
    let avg = pre + inf + post
    var line = "\n[summary] frames=\(samples.frames)  total_dets=\(samples.totalDets)  avg/frame: pre=\(g(pre)) infer=\(g(inf)) post=\(g(post)) total=\(g(avg))ms  model-FPS=\(g(1000 / avg))  wall=\(g(wall))s"
    if nmsMode == .clusterWeighted { line += "  nms=cw(sigma=\(g(Double(sigma))))" }
    print(line)
}

// ---------- single image ----------
func processImage(_ path: String, _ outPath: String) {
    guard var cg = loadCGImage(URL(fileURLWithPath: path)) else { logErr("skip (unreadable): \(path)"); return }
    if resize > 0 { cg = resizeLong(cg, resize) }
    let res: Detector.Result
    var tileNote = ""
    if tilingMode == .off {
        guard let r = try? detector.detect(cg, conf: conf, iou: iouT, mode: nmsMode, sigma: sigma) else { logErr("predict failed: \(path)"); return }
        res = r
    } else {
        guard let (r, t) = try? detector.detectTiled(cg, conf: conf, iou: iouT, tiling: tilingCfg, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet) else { logErr("predict failed: \(path)"); return }
        res = r
        tileNote = "  tiles=\(t.tilesRun)/\(t.tilesTotal) @\(t.tileSizeUsed)px" + (t.usedFallback ? " (fallback)" : "") + (t.capped ? " (capped)" : "")
    }
    record(URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent, res)
    print("[det] \((path as NSString).lastPathComponent)  dets=\(res.detections.count)  infer=\(f1(res.inferMs))ms" + tileNote)
    if !noSave, let a = annotate(cg, res.detections, names: detector.classNames, style: boxStyle, label: labelMode) {
        saveCGImage(a, to: URL(fileURLWithPath: outPath))
        print("[saved] \(outPath)")
    }
}

// ---------- dispatch (source auto-detected) ----------
let src = URL(fileURLWithPath: srcPath)
let kind = classifySource(src)
guard kind != .unknown else { die("source not found / unsupported: \(srcPath)", 4) }
if benchMode != .off && kind == .video { die("--bench / --accuracy take an image or folder source, not a video", 2) }
if trackKind != nil && kind != .video { die("--track needs a video source", 2) }

var doc: BenchDocument? = nil
var images: [URL] = kind == .folder ? listImages(src) : (kind == .image ? [src] : [])
if limit > 0 && images.count > limit { images = Array(images.prefix(limit)) }

if benchMode != .off {
    // the probe run doubles as the warm-up of the dataset pass (same order as the Linux CLI)
    let cold = BenchRunner.coldRun(detector, warmup: benchWarmup, iters: benchIters)
    print("[bench] cold probe \(cold.probe_mode): infer median=\(g(cold.infer_ms.median))ms p90=\(g(cold.infer_ms.p90)) min=\(g(cold.infer_ms.min)) (n=\(cold.infer_ms.n))")
    var card = BenchEnvironment.model(detector)
    card.ep_note = "preproc=\(detector.effectivePreprocDevice.rawValue)"
    doc = BenchDocument(timestamp: YMCore.timestampUTC(), tool: "macos", model: card,
                        environment: BenchEnvironment.collect(),
                        protocol: .init(mode: benchMode.rawValue, conf: conf, iou: Float(iouT),
                                        max_det: tilingMode == .off ? 300 : maxDet, multi_label: true,
                                        slicing: tilingMode.rawValue, tile_size: tileSizeArg,
                                        warmup: benchWarmup, iters: benchIters, minutes: benchMinutes,
                                        probe: "gray114", probe_mode: cold.probe_mode,
                                        dataset: src.deletingPathExtension().lastPathComponent, image_count: images.count,
                                        image_list_sha256: YMCore.imageListSha256(images.map { $0.path })),
                        cold: cold)
}

switch kind {
case .video:
    if tilingMode != .off { logErr("[warn] slicing applies to images/folders only - video runs single-pass") }
    let out: URL? = noSave ? nil : URL(fileURLWithPath: outArg ?? "out.mp4")
    let videoStem = src.deletingPathExtension().lastPathComponent
    var tracker: Tracker? = nil
    var detConf = conf
    if let k = trackKind {
        tracker = Tracker(kind: k, fps: 30, trackBuffer: trackBuffer)
        // the second association wants the low-score candidates: drop the detector floor unless --conf was given
        if !hasFlag("--conf") { detConf = tracker!.config.detectorFloor }
    }
    do {
        let s = try await runVideo(detector, input: src, output: out, conf: detConf, iou: iouT,
                                   style: boxStyle, label: labelMode, resize: resize,
                                   nmsMode: nmsMode, sigma: sigma, tracker: tracker,
                                   motion: trackKind == .botSort ? VisionCameraMotion() : nil,
                                   onResult: { i, _, res in record(TxtDumpWriter.frameStem(videoStem, i), res) }) { n, _ in
            if n % 60 == 0 { logErr("  \(n) frames…") }
        }
        if let out { print("[video] \(s.frames) frames -> \(out.path)  (\(s.outW)x\(s.outH) @\(s.fps)fps)") }
        else { print("[video] \(s.frames) frames (--no-save)  (\(s.outW)x\(s.outH) @\(s.fps)fps)") }
        print("[video] model-infer mean \(f1(s.meanMs))ms -> \(fps(s.meanMs)) fps (model-only)")
        if let t = tracker { print("[track] \(t.config.kind.rawValue)  frames=\(t.frameCount)") }
    } catch { die("video failed: \(error)", 5) }
case .folder:
    let out = noSave ? nil : URL(fileURLWithPath: outArg ?? "preds")
    print("[batch] \(src.lastPathComponent) -> \(out?.path ?? "(--no-save)")")
    let s = runFolder(detector, input: src, output: out, conf: conf, iou: iouT,
                      style: boxStyle, label: labelMode, resize: resize,
                      tiling: tilingCfg, nmsMode: nmsMode, sigma: sigma,
                      maxDet: tilingMode == .off ? 300 : maxDet, limit: limit, annotateImages: false,
                      onResult: { url, _, res in record(url.deletingPathExtension().lastPathComponent, res) })
    print("[batch] \(s.processed)/\(s.total) ok  |  model-infer mean \(f1(s.meanMs))ms -> \(fps(s.meanMs)) img/s steady")
case .image:
    processImage(srcPath, outArg ?? "out.jpg")
case .unknown:
    die("unsupported source: \(srcPath)", 4)
}
printSummary()
if let w = txtWriter { print("[save-txt] \(w.files) files -> \(w.directory.path)") }

if var d = doc {
    d.dataset = samples.dataset(wallS: Date().timeIntervalSince(tStart))
    if benchMode == .sustained {
        let su = BenchRunner.sustainedLoop(detector, warmup: benchWarmup, minutes: benchMinutes, coldIters: benchIters)
        print("[bench] sustained \(g(su.duration_s))s \(su.probe_mode): cold median=\(g(su.cold_median_ms))ms sustained median=\(g(su.sustained_median_ms))ms throttle=\(g(su.throttle_pct))%")
        d.sustained = su
    }
    if let acc = accuracy {
        // second pass at the val protocol; a --save-txt dump written during THIS pass is what scores
        // identically with scripts/eval_map*.py (the first pass ran at the user's conf)
        var accWriter: TxtDumpWriter? = nil
        if let saveTxt { accWriter = try? TxtDumpWriter(directory: URL(fileURLWithPath: saveTxt).appendingPathComponent("val")) }
        let o = AccuracyRunner.run(detector, images: images, labels: acc, resize: resize,
                                   dump: { url, dets in accWriter?.write(stem: url.deletingPathExtension().lastPathComponent, dets) })
        print(o.line)
        if let w = accWriter { print("[save-txt] val protocol dump: \(w.files) files -> \(w.directory.path)") }
        d.accuracy = o.document()
    }
    let jpath = benchJson ?? ((outArg ?? ".") + "/bench.json")
    let jurl = URL(fileURLWithPath: jpath)
    try? FileManager.default.createDirectory(at: jurl.deletingLastPathComponent(), withIntermediateDirectories: true)
    do { try d.json().write(to: jurl); print("[bench] json -> \(jpath)") }
    catch { die("cannot write \(jpath): \(error)", 5) }
}
