// Swift face of the portable C++ core (Sources/YOLOMasterCore, imported through its C header).
// Everything numeric that must agree with the Linux runtime to the last digit goes through here:
// the in-process mAP, the benchmark statistics, the hashes and the tracker.
import Foundation
import YOLOMasterCore

public enum YMCore {
    public static var version: Int { Int(ym_core_version()) }

    /// What the C++ CLI's `--save-txt` writer prints for a float (`%g`, six significant digits),
    /// read back as a Double. Feeding rounded values to the scorer makes the in-process mAP equal
    /// to scoring the txt dump with `scripts/eval_map*.py`.
    public static func round6(_ v: Double) -> Double { ym_round6(v) }

    public static func sha256Hex(_ data: Data) -> String {
        var out = [CChar](repeating: 0, count: 65)
        data.withUnsafeBytes { raw in
            ym_sha256_hex(raw.bindMemory(to: UInt8.self).baseAddress, data.count, &out)
        }
        return String(cString: out)
    }

    /// sha256 over the sorted basenames joined by "\n" (the `image_list_sha256` of the bench schema).
    public static func imageListSha256(_ paths: [String]) -> String {
        var out = [CChar](repeating: 0, count: 65)
        let cstrs: [UnsafePointer<CChar>?] = paths.map { UnsafePointer(strdup($0)) }
        defer { cstrs.forEach { free(UnsafeMutablePointer(mutating: $0)) } }
        cstrs.withUnsafeBufferPointer { ym_image_list_sha256($0.baseAddress, Int32(paths.count), &out) }
        return String(cString: out)
    }

    public static func timestampUTC() -> String {
        var out = [CChar](repeating: 0, count: 32)
        ym_timestamp_utc(&out)
        return String(cString: out)
    }
}

// MARK: - benchmark statistics (floor-rank percentiles, the phones' convention)

public struct StageStats: Codable, Equatable {
    public var n: Int
    public var mean, median, p90, p95, p99, min, max: Double

    public init(_ samples: [Double]) {
        var s = YmStageStats()
        samples.withUnsafeBufferPointer { ym_stats_reduce($0.baseAddress, Int32(samples.count), &s) }
        n = Int(s.n); mean = s.mean; median = s.median; p90 = s.p90; p95 = s.p95; p99 = s.p99; min = s.min; max = s.max
    }
}

public struct SustainedSummary {
    public let coldMedianMs, sustainedMedianMs, throttlePct: Double
    /// `all` in acquisition order; cold = median of the first `coldIters`, sustained = median of the slowest quarter.
    public init(_ all: [Double], coldIters: Int) {
        var s = YmSustained()
        all.withUnsafeBufferPointer { ym_sustained_summary($0.baseAddress, Int32(all.count), Int32(coldIters), &s) }
        coldMedianMs = s.cold_median_ms; sustainedMedianMs = s.sustained_median_ms; throttlePct = s.throttle_pct
    }
}

// MARK: - in-process mAP

public struct GtBox { public var x1, y1, x2, y2: Double; public var cls: Int
    public init(x1: Double, y1: Double, x2: Double, y2: Double, cls: Int) { self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2; self.cls = cls } }
public struct PredBox { public var x1, y1, x2, y2, conf: Double; public var cls: Int
    public init(x1: Double, y1: Double, x2: Double, y2: Double, conf: Double, cls: Int) { self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2; self.conf = conf; self.cls = cls } }
public struct ImageEval { public var preds: [PredBox]; public var gts: [GtBox]
    public init(preds: [PredBox], gts: [GtBox]) { self.preds = preds; self.gts = gts } }
public struct ClassAP: Codable { public let cls: Int; public let ap: [Double]; public let nGt, nPred: Int
    public var ap50: Double { ap[0] }
    public var ap5095: Double { ap.reduce(0, +) / 10 } }
public struct MapResult: Codable {
    public let images: Int
    public let map50, map5095: Double
    public let perClass: [ClassAP]
}

public enum MapEvaluator {
    /// COCO-style mAP 0.50:0.95, identical to `scripts/eval_map*.py` and the Linux `--accuracy` pass.
    public static func evaluate(_ images: [ImageEval]) -> MapResult {
        let preds = images.map { $0.preds.map { YmPredBox(x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2, conf: $0.conf, cls: Int32($0.cls)) } }
        let gts = images.map { $0.gts.map { YmGtBox(x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2, cls: Int32($0.cls)) } }
        var evals: [YmImageEval] = []
        evals.reserveCapacity(images.count)
        // keep every per-image array alive and pinned for the duration of the call
        var holders: [(UnsafeMutableBufferPointer<YmPredBox>, UnsafeMutableBufferPointer<YmGtBox>)] = []
        for i in images.indices {
            let p = UnsafeMutableBufferPointer<YmPredBox>.allocate(capacity: max(preds[i].count, 1))
            _ = p.initialize(from: preds[i])
            let g = UnsafeMutableBufferPointer<YmGtBox>.allocate(capacity: max(gts[i].count, 1))
            _ = g.initialize(from: gts[i])
            holders.append((p, g))
            evals.append(YmImageEval(preds: p.baseAddress, n_preds: Int32(preds[i].count), gts: g.baseAddress, n_gts: Int32(gts[i].count)))
        }
        defer { holders.forEach { $0.0.deallocate(); $0.1.deallocate() } }
        var r = YmMapResult()
        evals.withUnsafeBufferPointer { ym_map_evaluate($0.baseAddress, Int32(evals.count), &r) }
        defer { ym_map_free(&r) }
        var perClass: [ClassAP] = []
        for k in 0..<Int(r.n_classes) {
            let c = r.per_class[k]
            let ap = withUnsafeBytes(of: c.ap) { $0.bindMemory(to: Double.self).map { $0 } }
            perClass.append(ClassAP(cls: Int(c.cls), ap: ap, nGt: Int(c.n_gt), nPred: Int(c.n_pred)))
        }
        return MapResult(images: Int(r.images), map50: r.map50, map5095: r.map5095, perClass: perClass)
    }

    /// YOLO ("cls cx cy w h" normalized) or VisDrone comma labels; a missing / empty file is an empty ground truth.
    public static func loadLabels(_ path: String, imageWidth: Int, imageHeight: Int) -> [GtBox] {
        var out: UnsafeMutablePointer<YmGtBox>? = nil
        var n: Int32 = 0
        _ = ym_load_yolo_labels(path, Int32(imageWidth), Int32(imageHeight), &out, &n)
        defer { ym_gt_free(out) }
        guard let o = out else { return [] }
        return (0..<Int(n)).map { GtBox(x1: o[$0].x1, y1: o[$0].y1, x2: o[$0].x2, y2: o[$0].y2, cls: Int(o[$0].cls)) }
    }

    /// ultralytics images -> labels rule when `labelsDir` is nil, else `<labelsDir>/<stem>.txt`.
    public static func labelPath(forImage image: String, labelsDir: String?) -> String {
        var buf = [CChar](repeating: 0, count: 4096)
        let n = labelsDir.map { ym_label_path_for(image, $0, &buf, buf.count) } ?? ym_label_path_for(image, nil, &buf, buf.count)
        return n > 0 ? String(cString: buf) : ""
    }
}

// MARK: - self test (slice-1 handshake between the Swift package and the core)

public enum CoreSelfTest {
    /// Known answers: a perfectly matched two-box image scores 0.9950 / 0.9950 (the 101-point
    /// interpolation of eval_map.py, whose last sample is the appended precision 0); one detection moving 3 px
    /// per frame keeps id 1 across two frames on both trackers; the floor-rank median of 1..10 is 6.
    public static func run() -> (ok: Bool, report: String) {
        var lines: [String] = ["core version \(YMCore.version)"]
        var ok = true
        let im = ImageEval(preds: [PredBox(x1: 10, y1: 10, x2: 60, y2: 60, conf: 0.9, cls: 0),
                                   PredBox(x1: 100, y1: 100, x2: 150, y2: 180, conf: 0.8, cls: 1)],
                           gts: [GtBox(x1: 10, y1: 10, x2: 60, y2: 60, cls: 0), GtBox(x1: 100, y1: 100, x2: 150, y2: 180, cls: 1)])
        let m = MapEvaluator.evaluate([im])
        lines.append(String(format: "map images=%d mAP50=%.4f mAP50-95=%.4f classes=%d", m.images, m.map50, m.map5095, m.perClass.count))
        ok = ok && m.images == 1 && abs(m.map50 - 0.995) < 1e-9 && abs(m.map5095 - 0.995) < 1e-9 && m.perClass.count == 2
        let st = StageStats((1...10).map(Double.init))
        lines.append("stats n=\(st.n) median=\(st.median) p90=\(st.p90) max=\(st.max)")
        ok = ok && st.n == 10 && st.median == 6 && st.p90 == 10 && st.max == 10
        lines.append("round6(123.456789)=\(YMCore.round6(123.456789)) sha256('')=\(YMCore.sha256Hex(Data()).prefix(12))")
        ok = ok && YMCore.round6(123.456789) == 123.457 && YMCore.sha256Hex(Data()).hasPrefix("e3b0c44298fc")
        for kind in [TrackerKind.byteTrack, .botSort] {
            let t = Tracker(kind: kind)
            let d1 = [Detection(cls: 0, score: 0.9, rect: CGRect(x: 10, y: 10, width: 50, height: 50))]
            let d2 = [Detection(cls: 0, score: 0.9, rect: CGRect(x: 13, y: 10, width: 50, height: 50))]
            let a = t.update(d1), b = t.update(d2)
            lines.append("\(kind.rawValue) frame1 ids=\(a.map { $0.trackId ?? -1 }) frame2 ids=\(b.map { $0.trackId ?? -1 }) frames=\(t.frameCount)")
            ok = ok && a.count == 1 && b.count == 1 && a[0].trackId == 1 && b[0].trackId == 1 && t.frameCount == 2
        }
        lines.append(ok ? "SELFTEST OK" : "SELFTEST FAILED")
        return (ok, lines.joined(separator: "\n"))
    }
}
