// Benchmark mode: the `yolomaster-bench/v1` document the Linux CLI and server emit, produced on
// macOS from the Core ML path. Same statistics (floor-rank percentiles through the portable core),
// same probe (gray 114 at the model input size), same sustained definition (median of the slowest
// quarter vs the cold median), so a Mac row sits next to a Linux or phone row in one table.
//
// Cold run = `Detector.inferOnly` (Core ML prediction alone, probe_mode "infer_only"). The
// dataset pass collects Result.preMs / inferMs / postMs per image; the accuracy pass runs the val
// protocol (conf 0.001, IoU 0.7, max_det 300; the Kit's decode is multi-label already) and scores it
// in process with the txt-rounded values, so the number equals scoring a --save-txt dump.
import Foundation
import CoreGraphics
import Metal

public enum BenchMode: String, CaseIterable, Sendable { case off, cold, sustained }

// MARK: - the document (mirrors cpp/src/bench.cpp to_json key for key)

public struct BenchDocument: Codable {
    public struct Model: Codable {
        public var id, path, backend, runtime, execution_provider, ep_note, precision: String
        public var nc: Int; public var is_seg: Bool; public var imgsz: Int
        /// How the MoE mixture was exported (`ane_safe_rank` or `topk_gather`, from the package metadata; absent on other models).
        public var moe_export: String?
    }
    public struct Environment: Codable {
        public var host, os, cpu_model: String
        public var cpu_count: Int; public var mem_bytes: Int64
        public var gpu_name: String; public var threads: Int
        public var git_commit, build_flags, version: String
    }
    public struct ProtocolInfo: Codable {
        public var mode: String; public var conf, iou: Float; public var max_det: Int; public var multi_label: Bool
        public var slicing: String; public var tile_size: Int
        public var warmup, iters: Int; public var minutes: Double
        public var probe, probe_mode: String
        public var dataset: String; public var image_count: Int; public var image_list_sha256: String
        public init(mode: String, conf: Float, iou: Float, max_det: Int, multi_label: Bool, slicing: String, tile_size: Int,
                    warmup: Int, iters: Int, minutes: Double, probe: String, probe_mode: String,
                    dataset: String, image_count: Int, image_list_sha256: String) {
            self.mode = mode; self.conf = conf; self.iou = iou; self.max_det = max_det; self.multi_label = multi_label
            self.slicing = slicing; self.tile_size = tile_size; self.warmup = warmup; self.iters = iters; self.minutes = minutes
            self.probe = probe; self.probe_mode = probe_mode; self.dataset = dataset; self.image_count = image_count
            self.image_list_sha256 = image_list_sha256
        }
    }
    public struct Cold: Codable { public var infer_ms: StageStats; public var probe_mode: String }
    public struct Sustained: Codable {
        public var infer_ms: StageStats
        public var cold_median_ms, sustained_median_ms, throttle_pct: Double
        public var sparkline: [Double]; public var duration_s: Double; public var probe_mode: String
        /// macOS / iOS only, one sample per second in the thermal bands Cool / Normal / Hot / Critical:
        /// the die temperature bands (< 50, < 80, < 110 C) when the producer reads the SMC (the macOS
        /// app), else ProcessInfo's pressure states mapped onto the same four names (nominal -> Cool,
        /// fair -> Normal, serious -> Hot, critical -> Critical).
        public var thermal: [String]?
    }
    public struct Dataset: Codable {
        public var frames, total_dets: Int
        public var pre_ms, infer_ms, post_ms, total_ms: StageStats
        public var model_fps, wall_s: Double
    }
    /// macOS only, optional: what the machine did while THIS cell ran, sampled once per second. Thermal
    /// names are the app's die-temperature bands (Cool / Normal / Hot / Critical); `die_celsius` is the
    /// SMC sensor mean (absent without a sensor), `power_w` the battery flow (positive = charging,
    /// negative = discharging; absent without a battery).
    public struct HostMeters: Codable {
        public var seconds: Int
        public var thermal_start, thermal_peak: String
        public var die_celsius_start, die_celsius_peak, die_celsius_mean: Double?
        public var power_state: String
        public var power_w_mean, power_w_min, power_w_max: Double?
        public var die_celsius: [Double]?
        public var power_w: [Double]?
        public init(seconds: Int, thermal_start: String, thermal_peak: String, die_celsius_start: Double?, die_celsius_peak: Double?,
                    die_celsius_mean: Double?, power_state: String, power_w_mean: Double?, power_w_min: Double?, power_w_max: Double?,
                    die_celsius: [Double]?, power_w: [Double]?) {
            self.seconds = seconds; self.thermal_start = thermal_start; self.thermal_peak = thermal_peak
            self.die_celsius_start = die_celsius_start; self.die_celsius_peak = die_celsius_peak; self.die_celsius_mean = die_celsius_mean
            self.power_state = power_state; self.power_w_mean = power_w_mean; self.power_w_min = power_w_min; self.power_w_max = power_w_max
            self.die_celsius = die_celsius; self.power_w = power_w
        }
    }
    public struct AccuracyProtocol: Codable { public var conf, iou: Float; public var max_det: Int; public var multi_label: Bool }
    public struct PerClass: Codable { public var class_id, n_gt, n_pred: Int; public var ap50, ap5095: Double }
    public struct Accuracy: Codable {
        public var `protocol`: AccuracyProtocol
        public var labels: String; public var images: Int
        public var map50, map5095: Double
        public var per_class: [PerClass]
        public var timings: [String: StageStats]
    }

    public var schema_version = "yolomaster-bench/v1"
    public var timestamp: String
    public var tool: String
    public var model: Model
    public var environment: Environment
    public var `protocol`: ProtocolInfo
    public var stats_convention = "floor_rank"
    public var cold: Cold?
    public var sustained: Sustained?
    public var dataset: Dataset?
    public var accuracy: Accuracy?
    public var host_meters: HostMeters?

    public init(timestamp: String, tool: String, model: Model, environment: Environment, protocol proto: ProtocolInfo,
                cold: Cold? = nil, sustained: Sustained? = nil, dataset: Dataset? = nil, accuracy: Accuracy? = nil) {
        self.timestamp = timestamp; self.tool = tool; self.model = model; self.environment = environment
        self.protocol = proto; self.cold = cold; self.sustained = sustained; self.dataset = dataset; self.accuracy = accuracy
    }

    public func json() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try enc.encode(self)
    }
}

// MARK: - environment / model cards

public enum BenchEnvironment {
    static func sysctlString(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "" }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "" }
        return String(cString: buf)
    }
    static func sysctlInt64(_ name: String) -> Int64 {
        var v: Int64 = 0; var size = MemoryLayout<Int64>.size
        return sysctlbyname(name, &v, &size, nil, 0) == 0 ? v : 0
    }
    /// Runtime version: the app / package bundle version, else the repo VERSION file next to the
    /// package (development builds), else "".
    public static var version: String {
        if let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String, !v.isEmpty { return v }
        let here = URL(fileURLWithPath: #filePath)   // mac/Sources/YOLOMasterKit/Bench.swift -> repo root VERSION
        let root = here.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        if let s = try? String(contentsOf: root.appendingPathComponent("VERSION"), encoding: .utf8) {
            return s.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return ""
    }
    public static func collect(threads: Int = 0) -> BenchDocument.Environment {
        let pi = ProcessInfo.processInfo
        let osv = pi.operatingSystemVersion
        let cpu = sysctlString("machdep.cpu.brand_string")
        let gpu = MTLCreateSystemDefaultDevice()?.name ?? ""
        return BenchDocument.Environment(
            host: pi.hostName, os: "macOS \(osv.majorVersion).\(osv.minorVersion).\(osv.patchVersion) \(sysctlString("hw.machine"))",
            cpu_model: cpu, cpu_count: pi.processorCount, mem_bytes: sysctlInt64("hw.memsize"),
            gpu_name: gpu, threads: threads, git_commit: "", build_flags: "coreml", version: version)
    }
    public static func model(_ det: Detector) -> BenchDocument.Model {
        let path = det.modelURL.path
        let id = det.modelURL.deletingPathExtension().lastPathComponent
        let ep: String = {
            switch det.computeMode {
            case .cpuAndNeuralEngine: return "CoreML-ANE"
            case .all: return "CoreML-ALL"
            case .cpuAndGPU: return "CoreML-GPU"
            case .cpu: return "CoreML-CPU"
            }
        }()
        let precision = det.metadata["precision"] ?? det.metadata["quantization"] ?? "unknown"
        return BenchDocument.Model(id: id, path: path, backend: "coreml", runtime: "coreml", execution_provider: ep,
                                   ep_note: "", precision: precision, nc: det.nc, is_seg: det.isSegment, imgsz: det.imgsz,
                                   moe_export: det.metadata["moe_export"])
    }
}

// MARK: - runner

public struct BenchSamples {
    public var pre: [Double] = [], infer: [Double] = [], post: [Double] = [], total: [Double] = []
    public var frames = 0, totalDets = 0
    public init() {}
    public mutating func add(_ r: Detector.Result) {
        pre.append(r.preMs); infer.append(r.inferMs); post.append(r.postMs); total.append(r.preMs + r.inferMs + r.postMs)
        frames += 1; totalDets += r.detections.count
    }
    public var avgTotalMs: Double { frames > 0 ? total.reduce(0, +) / Double(frames) : 0 }
    public func dataset(wallS: Double) -> BenchDocument.Dataset {
        BenchDocument.Dataset(frames: frames, total_dets: totalDets, pre_ms: StageStats(pre), infer_ms: StageStats(infer),
                              post_ms: StageStats(post), total_ms: StageStats(total),
                              model_fps: avgTotalMs > 0 ? 1000 / avgTotalMs : 0, wall_s: wallS)
    }
}

public enum BenchRunner {
    /// The probe the Linux runtime uses: a gray-114 image at the model input size (letterbox is a no-op).
    public static func probeImage(_ imgsz: Int) -> CGImage? {
        var px = [UInt8](repeating: 114, count: imgsz * imgsz * 4)
        for i in stride(from: 3, to: px.count, by: 4) { px[i] = 255 }
        return px.withUnsafeMutableBytes { raw -> CGImage? in
            guard let ctx = CGContext(data: raw.baseAddress, width: imgsz, height: imgsz, bitsPerComponent: 8,
                                      bytesPerRow: imgsz * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
            return ctx.makeImage()
        }
    }

    /// `warmup` untimed then `iters` timed Core ML predictions on the probe (probe_mode "infer_only").
    /// `onSample(index, ms)` streams every timed iteration (a live chart); `cancel()` stops early and
    /// the statistics cover what was collected.
    public static func coldRun(_ det: Detector, warmup: Int, iters: Int,
                                 cancel: (() -> Bool)? = nil, onSample: ((Int, Double) -> Void)? = nil) -> BenchDocument.Cold {
        guard let probe = probeImage(det.imgsz) else { return BenchDocument.Cold(infer_ms: StageStats([]), probe_mode: "infer_only") }
        for _ in 0..<max(warmup, 0) {
            if cancel?() == true { break }
            _ = try? det.inferOnly(probe)
        }
        var v: [Double] = []
        v.reserveCapacity(max(iters, 0))
        for i in 0..<max(iters, 0) {
            if cancel?() == true { break }
            autoreleasepool {
                if let t = try? det.inferOnly(probe) { v.append(t); onSample?(i, t) }
            }
        }
        return BenchDocument.Cold(infer_ms: StageStats(v), probe_mode: "infer_only")
    }

    /// Timed loop for `minutes` after `warmup`; cold baseline = median of the first `coldIters`;
    /// one sparkline median and one thermal-state sample per second. `cancel()` stops early.
    public static func sustainedLoop(_ det: Detector, warmup: Int, minutes: Double, coldIters: Int = 50,
                                     cancel: (() -> Bool)? = nil, tick: ((Double, Double) -> Void)? = nil,
                                     onSample: ((Int, Double) -> Void)? = nil,
                                     thermalSampler: (() -> String)? = nil) -> BenchDocument.Sustained {
        var r = BenchDocument.Sustained(infer_ms: StageStats([]), cold_median_ms: 0, sustained_median_ms: 0, throttle_pct: 0,
                                        sparkline: [], duration_s: 0, probe_mode: "infer_only", thermal: [])
        guard let probe = probeImage(det.imgsz) else { return r }
        for _ in 0..<max(warmup, 0) { _ = try? det.inferOnly(probe) }
        var all: [Double] = [], second: [Double] = []
        let t0 = Date(); var secStart = t0
        let budget = minutes * 60
        while true {
            if cancel?() == true { break }
            let elapsed = Date().timeIntervalSince(t0)
            if elapsed >= budget && all.count >= coldIters { break }
            guard let t = (try? autoreleasepool { try det.inferOnly(probe) }) else { break }
            all.append(t); second.append(t)
            onSample?(all.count - 1, t)
            if Date().timeIntervalSince(secStart) >= 1 {
                r.sparkline.append(StageStats(second).median)
                r.thermal?.append(thermalSampler?() ?? thermalName(ProcessInfo.processInfo.thermalState))
                tick?(elapsed, StageStats(second).median)
                second.removeAll(); secStart = Date()
            }
        }
        if !second.isEmpty { r.sparkline.append(StageStats(second).median) }
        r.duration_s = Date().timeIntervalSince(t0)
        r.infer_ms = StageStats(all)
        let s = SustainedSummary(all, coldIters: coldIters)
        r.cold_median_ms = s.coldMedianMs; r.sustained_median_ms = s.sustainedMedianMs; r.throttle_pct = s.throttlePct
        return r
    }

    /// ProcessInfo's pressure state in the document's four-band vocabulary.
    public static func thermalName(_ s: ProcessInfo.ThermalState) -> String {
        switch s { case .nominal: return "Cool"; case .fair: return "Normal"; case .serious: return "Hot"; case .critical: return "Critical"
        @unknown default: return "Cool" }
    }
}

// MARK: - accuracy pass (the Linux --accuracy protocol)

public struct AccuracyProtocolValues {
    /// conf 0.001 / IoU 0.7 / max_det 300, multi-label. The Kit's decode keeps candidates with
    /// score > floor where the C++ keeps >= conf, so the floor is the largest Float below 0.001.
    public static let conf: Float = 0.001
    public static let confFloor: Float = Float(0.001).nextDown
    public static let iou: CGFloat = 0.7
    public static let maxDet = 300
}

public enum AccuracyRunner {
    public struct Outcome {
        public let map: MapResult
        public let inferMs: StageStats
        public let labels: String
        public var line: String { String(format: "[accuracy] images=%d  mAP50=%.4f  mAP50-95=%.4f", map.images, map.map50, map.map5095) }
        public func document() -> BenchDocument.Accuracy {
            BenchDocument.Accuracy(
                protocol: .init(conf: AccuracyProtocolValues.conf, iou: Float(AccuracyProtocolValues.iou),
                                max_det: AccuracyProtocolValues.maxDet, multi_label: true),
                labels: labels, images: map.images, map50: map.map50, map5095: map.map5095,
                per_class: map.perClass.map { .init(class_id: $0.cls, n_gt: $0.nGt, n_pred: $0.nPred, ap50: $0.ap50, ap5095: $0.ap5095) },
                timings: ["infer_ms": inferMs])
        }
    }

    /// "auto" = the ultralytics images -> labels rule, anything else is a labels directory.
    public static func labelsDir(_ spec: String) -> String? { spec == "auto" || spec.isEmpty ? nil : spec }

    /// Second pass over `images` at the val protocol; the boxes are scored with the txt rounding so
    /// the number equals scoring a --save-txt dump. `dump` receives every image's detections.
    public static func run(_ det: Detector, images: [URL], labels spec: String, resize: Int = 0,
                           dump: ((URL, [Detection]) -> Void)? = nil,
                           progress: ((Int, Int) -> Void)? = nil,
                           onInfer: ((Int, Double) -> Void)? = nil) -> Outcome {
        let ldir = labelsDir(spec)
        var evals: [ImageEval] = []
        evals.reserveCapacity(images.count)
        var infer: [Double] = []
        for (i, src) in images.enumerated() {
            autoreleasepool {
                guard var cg = loadCGImage(src) else { return }
                if resize > 0 { cg = resizeLong(cg, resize) }
                guard let res = try? det.detect(cg, conf: AccuracyProtocolValues.confFloor, iou: AccuracyProtocolValues.iou,
                                                maxDet: AccuracyProtocolValues.maxDet) else { return }
                infer.append(res.inferMs)
                onInfer?(i, res.inferMs)
                dump?(src, res.detections)
                let gts = MapEvaluator.loadLabels(MapEvaluator.labelPath(forImage: src.path, labelsDir: ldir),
                                                  imageWidth: cg.width, imageHeight: cg.height)
                evals.append(ImageEval(preds: res.detections.map(TxtDump.predBox), gts: gts))
            }
            progress?(i + 1, images.count)
        }
        return Outcome(map: MapEvaluator.evaluate(evals), inferMs: StageStats(infer), labels: ldir ?? "auto")
    }
}
