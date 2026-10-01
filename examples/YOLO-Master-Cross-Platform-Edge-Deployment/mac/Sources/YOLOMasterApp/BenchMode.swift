// Bench mode: the app's second major mode. The sidebar holds the protocol (models, compute
// units, preprocessing device, warm-up, timed iterations, sustained minutes, dataset / accuracy
// set) and the run history; the stage becomes a dashboard that streams the run as it happens:
// a per-iteration latency chart, the per-second sustained trace with its thermal band, a
// thermometer, the stat cards, the results table and the accuracy per-class chart.
//
// The measurements come from the Kit's BenchRunner / AccuracyRunner (the same probe, statistics
// and protocol as the CLI's --bench / --accuracy, through the portable core), so every run also
// yields a yolomaster-bench/v1 document that can be saved or exported.
import SwiftUI
import Charts
import AppKit
import IOKit
import IOKit.ps
import UniformTypeIdentifiers
import YOLOMasterKit

enum AppMode: String, CaseIterable { case inference = "Inference", bench = "Bench" }

// MARK: - data

enum BenchKind: String, CaseIterable, Codable {
    case cold = "Cold run", sustained = "Sustained", accuracy = "Accuracy"
    var icon: String {
        switch self { case .cold: return "bolt.fill"; case .sustained: return "flame.fill"; case .accuracy: return "checkmark.seal" }
    }
    var blurb: String {
        switch self {
        case .cold: return "10 warm-up predictions, then 100 timed model-only predictions on a gray probe at the input size."
        case .sustained: return "A timed loop after 10 warm-up predictions."
        case .accuracy: return "Val protocol (conf 0.001, IoU 0.7, max_det 300) over a labelled folder, scored in process: mAP50 / mAP50-95 per class."
        }
    }
}

enum ComputeChoice: String, CaseIterable, Codable, Identifiable {
    case auto = "Auto", gpu = "GPU", cpu = "CPU"
    var id: String { rawValue }
    /// Older records carry "ANE" (the forced Neural Engine unit, dropped: every segment the ANE
    /// refuses lands on the CPU, which measures nothing useful); they read back as Auto.
    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = ComputeChoice(rawValue: raw) ?? (raw == "ANE" ? .auto : .gpu)
    }
    /// ANE = CPU + Neural Engine (the ANE runs everything it supports, the CPU the rest; the GPU is
    /// never used), GPU = CPU + GPU, CPU = CPU only. Not `.all`: that is Core ML's own partitioning
    /// across the three units and measures the scheduler, not the Neural Engine.
    /// Auto = every unit, Core ML partitions the graph itself (the setting the iOS app's "ANE" button
    /// measures): the ANE takes the segments it accepts and the GPU the rest.
    var mode: ComputeMode { switch self { case .auto: return .all; case .gpu: return .cpuAndGPU; case .cpu: return .cpu } }
    var ep: String { self == .auto ? "CoreML-ALL" : "CoreML-" + rawValue }
    var icon: String { switch self { case .auto: return "sparkles"; case .gpu: return "rectangle.stack.fill"; case .cpu: return "cpu" } }
}

/// One (model x compute unit) cell of a run.
struct BenchCell: Identifiable, Codable {
    var id = UUID()
    let modelName: String
    let modelPath: String
    let compute: ComputeChoice
    let preproc: String
    var cold: StageStats?
    var sustained: BenchDocument.Sustained?
    var dataset: BenchDocument.Dataset?
    var accuracy: BenchDocument.Accuracy?
    var classNames: [String] = []        // from the model metadata (the accuracy chart names classes on hover)
    var samples: [Double] = []           // the per-iteration series (model ms)
    var sampleTimes: [Double] = []       // seconds since the cell's run started, parallel to samples
    var thermal: [Int] = []              // thermal level per second (sustained) or per sample bucket
    var celsius: [Double]? = nil         // die temperature per second while this cell ran (nil: no sensor)
    var power: [Double]? = nil           // battery watts per second while this cell ran (nil: no battery)
    var document: BenchDocument?
    var headlineMs: Double? { cold?.median ?? sustained?.sustained_median_ms ?? dataset?.infer_ms.median ?? accuracy?.timings["infer_ms"]?.median }
    var fps: Double { (headlineMs ?? 0) > 0 ? 1000 / headlineMs! : 0 }
}

/// A saved run (JSON-persisted, newest first).
struct BenchRecord: Identifiable, Codable {
    var id = UUID()
    var name: String
    let date: Date
    let kind: BenchKind
    let warmup: Int, iters: Int, minutes: Double
    var cells: [BenchCell]
    let hostName: String, cpuModel: String, osVersion: String
    var fastest: BenchCell? { cells.min { ($0.headlineMs ?? .infinity) < ($1.headlineMs ?? .infinity) } }
}

final class BenchStore: ObservableObject {
    @Published private(set) var records: [BenchRecord] = []
    private let url: URL
    init() {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("YOLOMaster", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        url = dir.appendingPathComponent("bench_history.json")
        if let d = try? Data(contentsOf: url) {
            if let r = BenchStore.decode(d) { records = r }
            else if let arr = (try? JSONSerialization.jsonObject(with: d)) as? [[String: Any]] {
                // older files: a renamed protocol, or a protocol that no longer exists (dropped)
                let kept = arr.filter { ($0["kind"] as? String) != "Dataset" }.map { rec -> [String: Any] in
                    var r = rec; if let k = r["kind"] as? String, k.hasPrefix("Cold "), k != "Cold run" { r["kind"] = "Cold run" }; return r
                }
                if let d2 = try? JSONSerialization.data(withJSONObject: kept), let r = BenchStore.decode(d2) { records = r }
            }
            if records.isEmpty, let arr = (try? JSONSerialization.jsonObject(with: d)) as? [Any], !arr.isEmpty {
                // parseable JSON that still yields nothing: keep the file out of harm's way instead of
                // letting the next save() overwrite it with an empty list
                let backup = url.deletingPathExtension().appendingPathExtension("unreadable.json")
                try? FileManager.default.removeItem(at: backup)
                try? FileManager.default.copyItem(at: url, to: backup)
            }
        }
    }
    /// The file is written with ISO-8601 dates (`save()`); decode it the same way, and accept the
    /// numeric form as a fallback so a file from another writer still loads.
    private static func decode(_ d: Data) -> [BenchRecord]? {
        let iso = JSONDecoder(); iso.dateDecodingStrategy = .iso8601
        if let r = try? iso.decode([BenchRecord].self, from: d) { return r }
        return try? JSONDecoder().decode([BenchRecord].self, from: d)
    }
    func add(_ r: BenchRecord) { records.insert(r, at: 0); save() }
    /// Replace a record in place (a sustained session growing by one cell per run); adds it if it is gone.
    func replace(_ id: UUID, with r: BenchRecord) {
        if let i = records.firstIndex(where: { $0.id == id }) { records[i] = r } else { records.insert(r, at: 0) }
        save()
    }
    func remove(_ ids: Set<UUID>) { records.removeAll { ids.contains($0.id) }; save() }
    func rename(_ id: UUID, _ name: String) { if let i = records.firstIndex(where: { $0.id == id }) { records[i].name = name; save() } }
    func clear() { records = []; save() }
    private func save() {
        let enc = JSONEncoder(); enc.dateEncodingStrategy = .iso8601
        if let d = try? enc.encode(records) { try? d.write(to: url) }
    }
    /// One row per cell of every record: what a spreadsheet wants. The first 21 columns are the
    /// 1.2.0 layout; everything the dashboard shows since (percentiles, FPS, thermal and power,
    /// precision, MoE export, host) follows, so old sheets keep their column positions.
    static let csvHeader = ["run", "date", "kind", "model", "compute", "preproc", "warmup", "iters", "minutes",
                            "cold_median_ms", "cold_p90_ms", "cold_p99_ms", "cold_min_ms", "sustained_median_ms", "throttle_pct",
                            "dataset_pre_ms", "dataset_infer_ms", "dataset_post_ms", "map50", "map5095", "images",
                            "cold_mean_ms", "cold_p95_ms", "cold_max_ms", "samples", "fps",
                            "sustained_cold_median_ms", "sustained_duration_s",
                            "thermal_start", "thermal_peak", "die_c_start", "die_c_peak", "die_c_mean",
                            "power_state", "power_w_mean", "power_w_min", "power_w_max",
                            "precision", "moe_export", "execution_provider", "imgsz", "host", "cpu", "os", "cell_id"]
    func csv() -> String {
        var out = [BenchStore.csvHeader.joined(separator: ",")]
        let f = ISO8601DateFormatter()
        func q(_ v: String) -> String {   // RFC 4180: quote when the field carries a comma, a quote or a newline
            v.contains(",") || v.contains("\"") || v.contains("\n") ? "\"" + v.replacingOccurrences(of: "\"", with: "\"\"") + "\"" : v
        }
        for r in records {
            for c in r.cells {
                func s(_ v: Double?) -> String { v.map { String(format: "%.4f", $0) } ?? "" }
                func s1(_ v: Double?) -> String { v.map { String(format: "%.1f", $0) } ?? "" }
                let hm = c.document?.host_meters, m = c.document?.model
                let row: [String] = [
                    q(r.name), f.string(from: r.date), r.kind.rawValue, q(c.modelName), c.compute.rawValue, c.preproc,
                    "\(r.warmup)", "\(r.iters)", "\(r.minutes)",
                    s(c.cold?.median), s(c.cold?.p90), s(c.cold?.p99), s(c.cold?.min),
                    s(c.sustained?.sustained_median_ms), s(c.sustained?.throttle_pct),
                    s(c.dataset?.pre_ms.median), s(c.dataset?.infer_ms.median), s(c.dataset?.post_ms.median),
                    s(c.accuracy?.map50), s(c.accuracy?.map5095), c.accuracy.map { "\($0.images)" } ?? "",
                    s(c.cold?.mean), s(c.cold?.p95), s(c.cold?.max), c.cold.map { "\($0.n)" } ?? "\(c.samples.count)", s1(c.fps),
                    s(c.sustained?.cold_median_ms), s1(c.sustained?.duration_s),
                    hm?.thermal_start ?? (c.thermal.first.map(thermalName) ?? ""), hm?.thermal_peak ?? (c.thermal.max().map(thermalName) ?? ""),
                    s1(hm?.die_celsius_start), s1(hm?.die_celsius_peak), s1(hm?.die_celsius_mean),
                    hm?.power_state ?? "", s(hm?.power_w_mean), s(hm?.power_w_min), s(hm?.power_w_max),
                    m?.precision ?? "", m?.moe_export ?? "", m?.execution_provider ?? c.compute.ep, m.map { "\($0.imgsz)" } ?? "",
                    q(r.hostName), q(r.cpuModel), q(r.osVersion), c.id.uuidString,
                ]
                out.append(row.joined(separator: ","))
            }
        }
        return out.joined(separator: "\n") + "\n"
    }
}

// MARK: - thermal

func thermalLevel(_ s: ProcessInfo.ThermalState) -> Int {
    switch s { case .nominal: return 0; case .fair: return 1; case .serious: return 2; case .critical: return 3; @unknown default: return 0 }
}
func thermalName(_ level: Int) -> String { ["Cool", "Normal", "Hot", "Critical"][max(0, min(3, level))] }

// MARK: - battery power flow (IOKit AppleSmartBattery: instantaneous current x voltage)

enum PowerState: String { case pluggedIn = "Plugged in", charging = "Charging", onBattery = "On battery" }

struct BatterySample {
    let present: Bool
    let watts: Double          // > 0 into the battery, < 0 out of it, 0 when idle / no battery
    let state: PowerState      // from the IOKit Power Sources API, not inferred from the sign of the current
    let percent: Int?
    static let none = BatterySample(present: false, watts: 0, state: .pluggedIn, percent: nil)
}
enum BatteryReader {
    /// State and charge come from IOPowerSources (kIOPSPowerSourceStateKey / kIOPSIsChargingKey); the
    /// instantaneous power comes from the AppleSmartBattery registry entry (current x voltage), which
    /// the Power Sources API does not expose.
    static func read() -> BatterySample {
        var state: PowerState? = nil
        var percent: Int? = nil
        if let blob = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
           let list = IOPSCopyPowerSourcesList(blob)?.takeRetainedValue() as? [CFTypeRef] {
            for ps in list {
                guard let d = IOPSGetPowerSourceDescription(blob, ps)?.takeUnretainedValue() as? [String: Any],
                      (d[kIOPSTypeKey as String] as? String) == kIOPSInternalBatteryType else { continue }
                let onAC = (d[kIOPSPowerSourceStateKey as String] as? String) == kIOPSACPowerValue
                let charging = (d[kIOPSIsChargingKey as String] as? Bool) ?? false
                state = onAC ? (charging ? .charging : .pluggedIn) : .onBattery
                if let cur = d[kIOPSCurrentCapacityKey as String] as? Int, let mx = d[kIOPSMaxCapacityKey as String] as? Int, mx > 0 {
                    percent = Int((Double(cur) / Double(mx) * 100).rounded())
                }
                break
            }
        }
        guard let state else { return .none }          // no internal battery: a desktop Mac
        var watts = 0.0
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSmartBattery"))
        if service != 0 {
            defer { IOObjectRelease(service) }
            var propsRef: Unmanaged<CFMutableDictionary>?
            if IORegistryEntryCreateCFProperties(service, &propsRef, kCFAllocatorDefault, 0) == KERN_SUCCESS,
               let props = propsRef?.takeRetainedValue() as? [String: Any] {
                // InstantAmperage is the live value (Amperage is a rolling average that lags by a minute). A
                // discharge is negative and the registry may hand it over as an unsigned 64-bit or a 32-bit
                // two's-complement pattern, so decode the bit pattern instead of trusting a plain Int cast.
                func milliamps(_ key: String) -> Int64? {
                    guard let n = props[key] as? NSNumber else { return nil }
                    var v: Int64
                    if let i = n as? Int64 { v = i } else { v = Int64(bitPattern: n.uint64Value) }
                    if v > Int64(Int32.max) && v <= Int64(UInt32.max) { v -= Int64(1) << 32 }
                    return v
                }
                let mA = milliamps("InstantAmperage") ?? milliamps("Amperage") ?? 0
                let mV = Double((props["Voltage"] as? Int) ?? 0)
                watts = Double(mA) / 1000 * mV / 1000
            }
        }
        return BatterySample(present: true, watts: watts, state: state, percent: percent)
    }
}
func thermalColor(_ level: Int) -> Color { [Color.blue, .green, .orange, .red][max(0, min(3, level))] }

// MARK: - die temperature (AppleSMC user client: the same keys smctemp / iStat read)

/// Reads the CPU / GPU die temperature sensors through the AppleSMC IOKit user client. ProcessInfo's
/// thermalState only says whether macOS is already under thermal PRESSURE (throttling); a machine
/// whose fans keep up stays "nominal" however hot the die is. The SMC exposes the sensors themselves:
/// on Apple silicon the CPU dies are the Tp / Te / Tf keys and the GPU dies the Tg keys, all "flt "
/// values in degrees Celsius. The key set is discovered once (the SMC lists its keys by index) and
/// then read on every tick; the hottest sensor is the reported temperature.
final class SMCTemperature {
    // SMCKeyInfoData is 12 bytes in C (9 + 3 padding); Swift lays fields out at size, not stride, so
    // the padding is explicit to keep the 80-byte SMCKeyData_t layout the kernel expects
    private struct KeyInfo { var dataSize: UInt32 = 0; var dataType: UInt32 = 0; var dataAttributes: UInt8 = 0; var pad: (UInt8, UInt8, UInt8) = (0, 0, 0) }
    private struct KeyData {          // the 80-byte SMCKeyData_t of the AppleSMC user client
        var key: UInt32 = 0
        var vers: (UInt8, UInt8, UInt8, UInt8, UInt16) = (0, 0, 0, 0, 0)
        var pLimit: (UInt16, UInt16, UInt32, UInt32, UInt32) = (0, 0, 0, 0, 0)
        var keyInfo = KeyInfo()
        var result: UInt8 = 0
        var status: UInt8 = 0
        var data8: UInt8 = 0
        var data32: UInt32 = 0
        var bytes: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                    UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8) =
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    }
    private static let kSMCHandleYPCEvent: UInt32 = 2
    private static let kSMCReadKey: UInt8 = 5, kSMCGetKeyFromIndex: UInt8 = 8, kSMCGetKeyInfo: UInt8 = 9

    private var conn: io_connect_t = 0
    private var sensors: [(name: String, key: UInt32, info: KeyInfo)] = []   // key info cached: one IOKit call per sensor per read
    var sensorNames: [String] { sensors.map(\.name) }
    var available: Bool { conn != 0 && !sensors.isEmpty }

    init() {
        guard MemoryLayout<KeyData>.size == 80 else { return }   // layout guard: never talk to the SMC with a wrong struct
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"))
        guard service != 0 else { return }
        defer { IOObjectRelease(service) }
        guard IOServiceOpen(service, mach_task_self_, 0, &conn) == KERN_SUCCESS else { conn = 0; return }
        discover()
    }
    deinit { if conn != 0 { IOServiceClose(conn) } }

    private static func code(_ s: String) -> UInt32 { s.utf8.reduce(0) { ($0 << 8) | UInt32($1) } }
    private static func name(_ c: UInt32) -> String {
        String(bytes: [UInt8(c >> 24 & 0xff), UInt8(c >> 16 & 0xff), UInt8(c >> 8 & 0xff), UInt8(c & 0xff)], encoding: .ascii) ?? "????"
    }
    private func call(_ input: inout KeyData) -> KeyData? {
        var output = KeyData()
        var outSize = MemoryLayout<KeyData>.stride
        let rc = withUnsafePointer(to: &input) { ip in
            IOConnectCallStructMethod(conn, SMCTemperature.kSMCHandleYPCEvent, ip, MemoryLayout<KeyData>.stride, &output, &outSize)
        }
        return rc == KERN_SUCCESS && output.result == 0 ? output : nil
    }
    private func keyInfo(_ key: UInt32) -> KeyInfo? {
        var q = KeyData(); q.key = key; q.data8 = SMCTemperature.kSMCGetKeyInfo
        return call(&q)?.keyInfo
    }
    private func readFloat(_ key: UInt32, _ info: KeyInfo) -> Double? {
        var q = KeyData(); q.key = key; q.keyInfo = info; q.data8 = SMCTemperature.kSMCReadKey
        guard let r = call(&q) else { return nil }
        let b = withUnsafeBytes(of: r.bytes) { Array($0.prefix(Int(info.dataSize))) }
        switch SMCTemperature.name(info.dataType) {
        case "flt ": guard b.count >= 4 else { return nil }; return Double(Float(bitPattern: UInt32(b[0]) | UInt32(b[1]) << 8 | UInt32(b[2]) << 16 | UInt32(b[3]) << 24))
        case "sp78": guard b.count >= 2 else { return nil }; return Double(Int16(bitPattern: UInt16(b[0]) << 8 | UInt16(b[1]))) / 256   // Intel Macs
        case "ui8 ": return b.first.map(Double.init)
        case "ui16": guard b.count >= 2 else { return nil }; return Double(UInt16(b[0]) << 8 | UInt16(b[1]))
        default: return nil
        }
    }
    /// Enumerate every key once; keep the die sensors (Tp / Te / Tf CPU, Tg GPU, Tc CPU on Intel) that
    /// read a plausible temperature.
    private func discover() {
        guard let cnt = keyInfo(SMCTemperature.code("#KEY")), let n = readUInt32(SMCTemperature.code("#KEY"), cnt), n > 0, n < 20000 else { return }
        var found: [(String, UInt32, KeyInfo)] = []
        for i in 0..<n {
            var q = KeyData(); q.data8 = SMCTemperature.kSMCGetKeyFromIndex; q.data32 = UInt32(i)
            guard let r = call(&q) else { continue }
            let nm = SMCTemperature.name(r.key)
            guard nm.hasPrefix("Tp") || nm.hasPrefix("Te") || nm.hasPrefix("Tf") || nm.hasPrefix("Tg") || nm.hasPrefix("Tc") else { continue }
            guard let info = keyInfo(r.key), let v = readFloat(r.key, info), v > 5, v < 130 else { continue }
            found.append((nm, r.key, info))
        }
        sensors = found
    }
    private func readUInt32(_ key: UInt32, _ info: KeyInfo) -> Int? {
        var q = KeyData(); q.key = key; q.keyInfo = info; q.data8 = SMCTemperature.kSMCReadKey
        guard let r = call(&q) else { return nil }
        let b = withUnsafeBytes(of: r.bytes) { Array($0.prefix(4)) }
        return Int(UInt32(b[0]) << 24 | UInt32(b[1]) << 16 | UInt32(b[2]) << 8 | UInt32(b[3]))
    }
    /// The die temperature right now, in degrees Celsius: the mean of the die sensors (what the
    /// temperature apps report as "CPU / GPU temperature"), not the single hottest spot, which on a
    /// many-sensor SoC is a permanent outlier. nil when nothing could be read.
    func temperature() -> Double? {
        var sum = 0.0, n = 0
        for s in sensors {
            guard let v = readFloat(s.key, s.info), v > 5, v < 130 else { continue }
            sum += v; n += 1
        }
        return n > 0 ? sum / Double(n) : nil
    }
}

/// Die temperature to the four meter levels: Cool below 50 C, Normal to 80, Hot to 110, Critical above.
func thermalLevel(celsius: Double) -> Int { celsius < 50 ? 0 : (celsius < 80 ? 1 : (celsius < 110 ? 2 : 3)) }

// MARK: - the meters (their own observable so their 10 Hz ticks re-render only the two gauges)

final class MeterModel: ObservableObject {
    @Published private(set) var thermal = thermalLevel(ProcessInfo.processInfo.thermalState)
    @Published private(set) var thermalPeak = 0
    @Published private(set) var celsius: Double? = nil        // hottest die sensor (nil: no SMC sensor, pressure state only)
    @Published private(set) var celsiusPeak: Double? = nil
    @Published private(set) var battery = BatteryReader.read()
    let smc = SMCTemperature()
    private var thermalTimer: Timer?
    private var powerTimer: Timer?
    private let smcQueue = DispatchQueue(label: "com.yolomaster.smc", qos: .utility)
    private var smcBusy = false
    var tracking = false          // a run is on: keep the peak

    init() {
        // die temperature once a second: the SMC reads (one IOKit call per sensor) run off the main
        // thread and the result is published in one animated step
        thermalTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self, !self.smcBusy else { return }
            self.smcBusy = true
            let pressure = thermalLevel(ProcessInfo.processInfo.thermalState)
            self.smcQueue.async {
                let c = self.smc.available ? self.smc.temperature() : nil
                DispatchQueue.main.async {
                    self.smcBusy = false
                    let l = c.map { thermalLevel(celsius: $0) } ?? pressure
                    if let c, abs((self.celsius ?? -1) - c) >= 0.5 { self.celsius = c }
                    if l != self.thermal { self.thermal = l }
                    if self.tracking {
                        if l > self.thermalPeak { self.thermalPeak = l }
                        if let c { self.celsiusPeak = max(self.celsiusPeak ?? c, c) }
                    }
                }
            }
        }
        // the battery's instantaneous current at 10 Hz (an IOKit registry read, well under a millisecond)
        powerTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self else { return }
            let b = BatteryReader.read()
            if abs(b.watts - self.battery.watts) > 0.01 || b.state != self.battery.state || b.present != self.battery.present {
                self.battery = b
            }
        }
    }
    func startRun() { tracking = true; thermalPeak = thermal; celsiusPeak = celsius }
    func endRun() { tracking = false }
}

final class BenchModel: ObservableObject {
    // protocol
    @Published var models: [URL] = []
    @Published var selectedModels: Set<URL> = []
    @Published var computes: Set<ComputeChoice> = [.gpu]
    @Published var preproc: PreprocDevice = .gpu
    @Published var kind: BenchKind = .cold { didSet { if kind == .sustained { collapseToSingle() } } }
    /// Sustained runs one model on one unit: a timed loop only makes sense against one thermal history,
    /// and running cells back to back would hand the second one a pre-heated machine.
    var singleCell: Bool { kind == .sustained }
    func collapseToSingle() {
        if selectedModels.count > 1, let keep = models.first(where: { selectedModels.contains($0) }) { selectedModels = [keep] }
        if computes.count > 1, let keep = ComputeChoice.allCases.first(where: { computes.contains($0) }) { computes = [keep] }
    }
    let warmup = 10.0                            // fixed by the protocol (the Linux CLI's defaults)
    let iters = 100.0
    @Published var minutes = 2.0
    @Published var datasetURL: URL?             // dataset / accuracy: the images folder
    @Published var datasetLimit = 0.0           // 0 = all
    let conf = 0.25, iou = 0.5                   // recorded in the document's protocol block
    // live
    @Published private(set) var running = false
    @Published private(set) var phase = ""          // what is happening now
    @Published private(set) var progress: Double?   // 0...1 when known
    @Published private(set) var liveSamples: [(t: Double, ms: Double)] = []   // t = seconds since the cell's run started
    @Published private(set) var liveStart = Date()
    @Published private(set) var liveSeconds: [(t: Double, med: Double, thermal: Int)] = []
    @Published private(set) var liveCell: BenchCell?
    let meters = MeterModel()
    private(set) var runPowerW: [Double] = []      // battery watts sampled once per second during a run
    @Published private(set) var cells: [BenchCell] = []     // the current / last run
    @Published private(set) var lastRecord: BenchRecord?
    @Published var note = ""
    @Published var selectedCellID: UUID? = nil      // the cell the dashboard inspects (table row / bar / arrow keys)
    let store = BenchStore()

    private let queue = DispatchQueue(label: "com.yolomaster.bench", qos: .userInitiated)
    private var cancelFlag = false
    private let cancelLock = NSLock()
    private var powerLogTimer: Timer?
    private var pendingSamples: [(t: Double, ms: Double)] = []
    private var cellStart = Date()
    private var cellGen = 0                 // bumped per cell: samples still in flight from the previous cell are dropped
    private var flushScheduled = false
    private var detectors: [String: Detector] = [:]

    init() {
        // once a second during a run: log the battery draw (not published; the record keeps it)
        powerLogTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self, self.running, self.meters.battery.present else { return }
            self.runPowerW.append(self.meters.battery.watts)
        }
    }

    static let maxModels = 5
    func addModel(_ url: URL) {
        if !models.contains(url) {
            guard models.count < BenchModel.maxModels else { note = "At most \(BenchModel.maxModels) models in the list; remove one first."; return }
            models.append(url)
        }
        if singleCell { selectedModels = [url] } else { selectedModels.insert(url) }
    }
    func removeModel(_ url: URL) { models.removeAll { $0 == url }; selectedModels.remove(url) }

    private var isCancelled: Bool { cancelLock.lock(); defer { cancelLock.unlock() }; return cancelFlag }
    func cancel() { cancelLock.lock(); cancelFlag = true; cancelLock.unlock() }

    private func detector(_ url: URL, _ c: ComputeChoice) throws -> Detector {
        let key = url.path + "|" + c.rawValue
        if let d = detectors[key] { d.preprocDevice = preproc; return d }
        let d = try Detector(modelURL: url, compute: c.mode)
        d.preprocDevice = preproc
        detectors[key] = d
        return d
    }

    // streaming: samples are batched onto the main thread at ~20 Hz so the chart never starves the run
    private func push(_ ms: Double) {
        let t = Date().timeIntervalSince(cellStart), gen = cellGen
        DispatchQueue.main.async { [weak self] in
            guard let self, gen == self.cellGen else { return }
            self.pendingSamples.append((t, ms))
            if !self.flushScheduled {
                self.flushScheduled = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    if gen == self.cellGen { self.liveSamples.append(contentsOf: self.pendingSamples) }
                    self.pendingSamples.removeAll(); self.flushScheduled = false
                }
            }
        }
    }
    private func main(_ f: @escaping () -> Void) { DispatchQueue.main.async(execute: f) }

    func run() {
        guard !running else { return }
        let targets = models.filter { selectedModels.contains($0) }
        guard !targets.isEmpty else { note = "Add and select at least one model."; return }
        guard !computes.isEmpty else { note = "Select at least one compute unit."; return }
        if singleCell && (targets.count > 1 || computes.count > 1) { note = "Sustained runs one model on one compute unit at a time."; return }
        if kind == .accuracy && datasetURL == nil { note = "Choose the images folder first."; return }
        cancelLock.lock(); cancelFlag = false; cancelLock.unlock()
        // Sustained runs one cell at a time, so consecutive sustained runs extend one record (the
        // previous cells stay in the table and the comparison chart); any other kind starts a new one.
        let carried: [BenchCell] = (kind == .sustained && lastRecord?.kind == .sustained) ? (lastRecord?.cells ?? []) : []
        let continuing: UUID? = carried.isEmpty ? nil : lastRecord?.id
        running = true; cells = carried; liveSamples = []; liveSeconds = []; liveCell = nil; progress = nil; note = ""
        runPowerW = []; meters.startRun()
        let kind = self.kind, warm = Int(warmup), iters = Int(iters), minutes = self.minutes
        let computes = ComputeChoice.allCases.filter { self.computes.contains($0) }
        let dataset = datasetURL, limit = Int(datasetLimit), conf = Float(self.conf), iou = CGFloat(self.iou)
        queue.async { [weak self] in
            guard let self else { return }
            var done: [BenchCell] = carried
            outer: for url in targets {
                for c in computes {
                    if self.isCancelled { break outer }
                    let name = url.deletingPathExtension().lastPathComponent
                    self.main { self.phase = "Loading \(name) on \(c.rawValue)…" }
                    let det: Detector
                    do { det = try self.detector(url, c) } catch {
                        self.main { self.note = "\(name) on \(c.rawValue): \(error.localizedDescription)" }
                        continue
                    }
                    var cell = BenchCell(modelName: name, modelPath: url.path, compute: c, preproc: det.effectivePreprocDevice.rawValue)
                    cell.classNames = det.classNames
                    // new cell: new clock and generation; the chart and any in-flight samples of the previous cell are dropped
                    self.cellStart = Date()
                    let cellStart = self.cellStart
                    let gen = self.cellGen + 1
                    self.cellGen = gen
                    self.main { self.liveCell = cell; self.liveStart = cellStart; self.liveSamples = []; self.liveSeconds = []; self.pendingSamples = [] }
                    var card = BenchEnvironment.model(det); card.ep_note = "preproc=\(det.effectivePreprocDevice.rawValue)"
                    var doc = BenchDocument(timestamp: YMCore.timestampUTC(), tool: "macos", model: card, environment: BenchEnvironment.collect(),
                                            protocol: .init(mode: kind == .sustained ? "sustained" : "cold", conf: conf, iou: Float(iou), max_det: 300,
                                                            multi_label: true, slicing: "off", tile_size: 0, warmup: warm, iters: iters, minutes: minutes,
                                                            probe: "gray114", probe_mode: "infer_only", dataset: dataset?.lastPathComponent ?? "",
                                                            image_count: 0, image_list_sha256: YMCore.imageListSha256([])))
                    var thermalTrack: [Int] = [], celsiusTrack: [Double] = [], powerTrack: [Double] = []
                    let t0 = Date(); var lastSec = -1
                    let sampleMeters: () -> Void = {   // one row per second: level, die temperature, battery watts
                        thermalTrack.append(self.meters.thermal)
                        if let c = self.meters.celsius { celsiusTrack.append(c) }
                        if self.meters.battery.present { powerTrack.append(self.meters.battery.watts) }
                    }
                    let sampleThermal: () -> Void = {
                        let sec = Int(Date().timeIntervalSince(t0))
                        if sec != lastSec { lastSec = sec; sampleMeters() }
                    }
                    switch kind {
                    case .cold:
                        self.main { self.phase = "\(name) · \(c.rawValue): warm-up \(warm), then \(iters) timed iterations" }
                        let cold = BenchRunner.coldRun(det, warmup: warm, iters: iters, cancel: { self.isCancelled }) { i, ms in
                            cell.samples.append(ms); cell.sampleTimes.append(Date().timeIntervalSince(cellStart)); self.push(ms); sampleThermal()
                            if i % 5 == 0 { let p = Double(i + 1) / Double(max(iters, 1)); self.main { self.progress = p } }
                        }
                        cell.cold = cold.infer_ms; doc.cold = cold
                    case .sustained:
                        self.main { self.phase = "\(name) · \(c.rawValue): sustained \(String(format: "%.1f", minutes)) min" }
                        let su = BenchRunner.sustainedLoop(det, warmup: warm, minutes: minutes, coldIters: iters,
                                                           cancel: { self.isCancelled },
                                                           tick: { elapsed, med in
                                                               let th = self.meters.thermal
                                                               sampleMeters()
                                                               self.main { self.liveSeconds.append((elapsed, med, th)); self.progress = min(1, elapsed / (minutes * 60)) }
                                                           },
                                                           onSample: { _, ms in
                                                               cell.samples.append(ms); cell.sampleTimes.append(Date().timeIntervalSince(cellStart)); self.push(ms)
                                                           },
                                                           thermalSampler: { thermalName(self.meters.thermal) })   // the die-temperature band, as on the meter
                        cell.sustained = su; doc.sustained = su
                        var coldStats = StageStats(Array(cell.samples.prefix(max(iters, 1))))
                        coldStats.n = min(cell.samples.count, max(iters, 1))
                        cell.cold = coldStats
                    case .accuracy:
                        guard let ds = dataset else { break }
                        var files = listImages(ds)
                        if limit > 0 && files.count > limit { files = Array(files.prefix(limit)) }
                        let sibling = ds.deletingLastPathComponent().appendingPathComponent("labels")
                        let labels = FileManager.default.fileExists(atPath: sibling.path) ? sibling.path : "auto"
                        self.main { self.phase = "\(name) · \(c.rawValue): scoring \(files.count) images (val protocol)" }
                        if let probe = BenchRunner.probeImage(det.imgsz) { for _ in 0..<warm { _ = try? det.inferOnly(probe) } }
                        let o = AccuracyRunner.run(det, images: files, labels: labels,
                                                   progress: { done, total in
                                                       if done % 5 == 0 { let p = Double(done) / Double(total); self.main { self.progress = p } }
                                                   },
                                                   onInfer: { _, ms in
                                                       cell.samples.append(ms); cell.sampleTimes.append(Date().timeIntervalSince(cellStart)); self.push(ms); sampleThermal()
                                                   })
                        cell.accuracy = o.document(); doc.accuracy = o.document()
                        doc.protocol.image_count = files.count; doc.protocol.image_list_sha256 = YMCore.imageListSha256(files.map { $0.path })
                        cell.cold = o.inferMs
                    }
                    cell.thermal = thermalTrack
                    cell.celsius = celsiusTrack.isEmpty ? nil : celsiusTrack
                    cell.power = powerTrack.isEmpty ? nil : powerTrack
                    doc.host_meters = BenchModel.hostMeters(thermal: thermalTrack, celsius: cell.celsius, power: cell.power,
                                                            state: self.meters.battery.present ? self.meters.battery.state.rawValue : "no battery")
                    cell.document = doc
                    if self.isCancelled && cell.headlineMs == nil { break outer }   // stopped before any statistic: nothing to keep
                    done.append(cell)
                    let snapshot = done
                    self.main { self.cells = snapshot; self.liveCell = cell; self.progress = nil }
                }
            }
            let cancelled = self.isCancelled
            let env = BenchEnvironment.collect()
            let record = BenchRecord(id: continuing ?? UUID(), name: BenchModel.defaultName(kind, done), date: Date(), kind: kind,
                                     warmup: warm, iters: iters, minutes: minutes,
                                     cells: done, hostName: env.host, cpuModel: env.cpu_model, osVersion: env.os)
            self.main {
                self.running = false; self.progress = nil; self.meters.endRun()
                self.phase = cancelled ? "Stopped." : "Done."
                if done.count > carried.count {
                    if let id = continuing { self.store.replace(id, with: record) } else { self.store.add(record) }
                    self.lastRecord = record
                }
            }
        }
    }

    /// Remove one run from the record it belongs to (the store entry is rewritten or, if it is now
    /// empty, dropped). Returns true when the whole record went away.
    @discardableResult
    func deleteCell(_ id: UUID, from record: BenchRecord?) -> Bool {
        guard !running else { return false }
        if selectedCellID == id { selectedCellID = nil }
        cells.removeAll { $0.id == id }
        guard var r = record else { return false }
        r.cells.removeAll { $0.id == id }
        if r.cells.isEmpty {
            store.remove([r.id])
            if lastRecord?.id == r.id { lastRecord = nil; cells = [] }
            return true
        }
        store.replace(r.id, with: r)
        if lastRecord?.id == r.id { lastRecord = r; cells = r.cells }
        return false
    }

    private static func defaultName(_ kind: BenchKind, _ cells: [BenchCell]) -> String {
        let f = DateFormatter(); f.dateFormat = "MMM d HH:mm"
        let m = Set(cells.map(\.modelName)).sorted().joined(separator: "+")
        return "\(kind.rawValue) · \(m.isEmpty ? "-" : m) · \(f.string(from: Date()))"
    }

    static func hostMeters(thermal: [Int], celsius: [Double]?, power: [Double]?, state: String) -> BenchDocument.HostMeters? {
        guard !thermal.isEmpty else { return nil }
        func mean(_ a: [Double]) -> Double { a.reduce(0, +) / Double(a.count) }
        return .init(seconds: thermal.count, thermal_start: thermalName(thermal.first ?? 0), thermal_peak: thermalName(thermal.max() ?? 0),
                     die_celsius_start: celsius?.first, die_celsius_peak: celsius?.max(), die_celsius_mean: celsius.map(mean),
                     power_state: state, power_w_mean: power.map(mean), power_w_min: power?.min(), power_w_max: power?.max(),
                     die_celsius: celsius, power_w: power)
    }

    // ---- export ----
    /// Every cell of a record as one JSON array of `yolomaster-bench/v1` documents (each element
    /// validates on its own with scripts/bench_schema_check.py).
    func exportRecordJSON(_ record: BenchRecord) {
        let docs = record.cells.compactMap(\.document)
        guard !docs.isEmpty else { note = "Nothing to export: this run has no documents"; return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "bench-\(record.name.replacingOccurrences(of: " · ", with: "-").replacingOccurrences(of: " ", with: "_")).json"
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try enc.encode(docs).write(to: url); note = "Exported \(docs.count) cells to \(url.lastPathComponent)"
        } catch { note = "Export failed: \(error.localizedDescription)" }
    }
    func saveJSON(_ cell: BenchCell) {
        guard let doc = cell.document else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "bench-\(cell.modelName)-\(cell.compute.rawValue).json"
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try doc.json().write(to: url); note = "Saved \(url.lastPathComponent)" } catch { note = "Save failed: \(error.localizedDescription)" }
    }
    func exportCSV() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "yolomaster-bench-history.csv"
        panel.allowedContentTypes = [.commaSeparatedText]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try store.csv().write(to: url, atomically: true, encoding: .utf8); note = "Exported \(store.records.count) runs" }
        catch { note = "Export failed: \(error.localizedDescription)" }
    }
}

// MARK: - sidebar

struct BenchSidebar: View {
    @ObservedObject var bench: BenchModel
    @ObservedObject var store: BenchStore
    @Binding var selectedRecord: UUID?
    let brand: Color
    @State private var confirmClear = false
    @State private var pendingDelete: UUID? = nil

    var body: some View {
        sidebarScroll
            .confirmationDialog("Delete every saved run?", isPresented: $confirmClear, titleVisibility: .visible) {
                Button("Delete all runs", role: .destructive) { store.clear(); selectedRecord = nil }
                Button("Cancel", role: .cancel) {}
            } message: { Text("\(store.records.count) runs will be removed from the history. This cannot be undone.") }
            .confirmationDialog("Delete this run?", isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }), titleVisibility: .visible) {
                Button("Delete run", role: .destructive) {
                    if let id = pendingDelete { store.remove([id]); if selectedRecord == id { selectedRecord = nil } }
                    pendingDelete = nil
                }
                Button("Cancel", role: .cancel) { pendingDelete = nil }
            } message: { Text(store.records.first { $0.id == pendingDelete }?.name ?? "") }
    }
    private var sidebarScroll: some View {
        ScrollView {
            VStack(spacing: 14) {
                box("Models", "cube.box.fill") {
                    ForEach(Array(bench.models.enumerated()), id: \.element) { i, u in
                        if i > 0 { Divider() }
                        let on = bench.selectedModels.contains(u)
                        HStack(spacing: 6) {
                            Button {
                                if bench.singleCell { bench.selectedModels = [u] }
                                else if on { bench.selectedModels.remove(u) } else { bench.selectedModels.insert(u) }
                            } label: {
                                HStack(spacing: 10) {
                                    Image(systemName: on ? "checkmark.circle.fill" : "circle").font(.system(size: 15))
                                        .foregroundStyle(on ? brand : .secondary).frame(width: 20)
                                    Text(u.deletingPathExtension().lastPathComponent).font(.callout).lineLimit(1).truncationMode(.middle)
                                        .foregroundStyle(on ? Color.primary : .secondary)
                                    Spacer(minLength: 4)
                                }
                                .contentShape(Rectangle())
                            }.buttonStyle(.plain).disabled(bench.running)
                            Button { bench.removeModel(u) } label: { Image(systemName: "minus.circle").foregroundStyle(.tertiary) }
                                .buttonStyle(.borderless).help("Remove from the list")
                        }
                    }
                    if !bench.models.isEmpty { Divider() }
                    if bench.models.count < BenchModel.maxModels {
                        fileRow(icon: "plus.circle", title: "Add model", value: "Choose .mlpackage / .mlmodelc…", set: false) { addModel() }
                    } else {
                        Text("Five models at most; remove one to add another.").font(.caption2).foregroundStyle(.tertiary)
                    }
                }
                box("Compute", "cpu") {
                    row("Units") {
                        HStack(spacing: 8) {
                            ForEach(ComputeChoice.allCases) { c in
                                let on = bench.computes.contains(c)
                                Button {
                                    if bench.singleCell { bench.computes = [c] }
                                    else if on { bench.computes.remove(c) } else { bench.computes.insert(c) }
                                } label: {
                                    Label(c.rawValue, systemImage: c.icon).font(.callout)
                                        .frame(maxWidth: .infinity).padding(.vertical, 4)
                                }
                                .buttonStyle(.bordered).tint(on ? brand : .secondary)
                                .background(RoundedRectangle(cornerRadius: 6).fill(on ? brand.opacity(0.14) : .clear))
                            }
                        }
                    }.disabled(bench.running)
                    row("Preprocess") {
                        SegmentedButtons(options: [(PreprocDevice.gpu, "GPU"), (PreprocDevice.cpu, "CPU")], icons: ["rectangle.stack.fill", "cpu"], selection: $bench.preproc, tint: brand)
                    }.disabled(bench.running)
                    if bench.singleCell {
                        Text("Sustained runs one model on one unit; pick one of each.").font(.caption2).foregroundStyle(.secondary)
                    }
                }
                box("Protocol", "list.bullet.clipboard") {
                    MenuButton(options: BenchKind.allCases.map { ($0, $0.rawValue) }, icons: BenchKind.allCases.map(\.icon), selection: $bench.kind, tint: brand)
                        .disabled(bench.running)
                    Text(bench.kind.blurb).font(.caption2).foregroundStyle(.secondary)
                    if bench.kind == .sustained { row("Duration") { TimerDial(minutes: $bench.minutes).disabled(bench.running) } }
                    if bench.kind == .accuracy {
                        fileRow(icon: "photo.on.rectangle.angled", title: "Labelled set (images folder)",
                                value: bench.datasetURL?.lastPathComponent ?? "Choose images folder…", set: bench.datasetURL != nil) { pickDataset() }
                        Text("Labels are read from the sibling labels/ folder (the ultralytics layout) or next to each image.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                if !store.records.isEmpty {
                    box("History", "clock.arrow.circlepath") {
                        ForEach(store.records) { r in
                            HStack(spacing: 8) {
                                Image(systemName: r.kind.icon).foregroundStyle(selectedRecord == r.id ? brand : .secondary).frame(width: 16)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(r.name).font(.caption).lineLimit(1).truncationMode(.middle)
                                    Text(r.fastest.map { String(format: "%.2f ms · %@ · %@", $0.headlineMs ?? 0, $0.compute.rawValue, r.kind.rawValue) } ?? r.kind.rawValue)
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                                Spacer(minLength: 4)
                            }
                            .contentShape(Rectangle())
                            .padding(6)
                            .background(RoundedRectangle(cornerRadius: 6).fill(selectedRecord == r.id ? brand.opacity(0.12) : .clear))
                            .onTapGesture { selectedRecord = selectedRecord == r.id ? nil : r.id }
                            .contextMenu {
                                Button("Delete…") { pendingDelete = r.id }
                            }
                        }
                        HStack {
                            Button { bench.exportCSV() } label: { Label("Export CSV", systemImage: "square.and.arrow.up") }
                            Spacer()
                            Button(role: .destructive) { confirmClear = true } label: { Label("Clear…", systemImage: "trash") }
                        }.controlSize(.small)
                    }
                }
            }
        }
    }

    private func addModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = true; panel.allowsMultipleSelection = true
        panel.treatsFilePackagesAsDirectories = false
        panel.message = "Choose Core ML models (.mlpackage / .mlmodelc)"
        guard panel.runModal() == .OK else { return }
        for u in panel.urls where ["mlpackage", "mlmodelc", "mlmodel"].contains(u.pathExtension.lowercased()) { bench.addModel(u) }
    }
    private func pickDataset() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.allowsMultipleSelection = false
        panel.message = "Choose the images folder of a labelled set (labels/ next to it)"
        guard panel.runModal() == .OK, let u = panel.url else { return }
        bench.datasetURL = u
    }

    // small helpers (the Inference sidebar has private twins)
    private func fileRow(icon: String, title: String, value: String, set: Bool, chevron: Bool = true, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: icon).font(.system(size: 15)).foregroundStyle(set ? brand : .secondary).frame(width: 20)
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.caption).foregroundStyle(.secondary)
                    Text(value).font(.callout).lineLimit(1).truncationMode(.middle).foregroundStyle(set ? Color.primary : .secondary)
                }
                Spacer(minLength: 4)
                if chevron { Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary) }
            }
            .contentShape(Rectangle())
        }.buttonStyle(.plain).disabled(bench.running)
    }
    private func box<C: View>(_ title: String, _ icon: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon).font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.leading, 2)
            VStack(alignment: .leading, spacing: 12) { content() }
                .frame(maxWidth: .infinity, alignment: .leading).padding(16)
                .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
        }
    }
    private func row<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 4) { Text(title).font(.callout); content() }
    }
    private func slider(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack { Text(title).font(.callout); Spacer()
                Text(String(format: "%.2f", value.wrappedValue)).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    .padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule()) }
            Slider(value: value, in: range)
        }.disabled(bench.running)
    }
    private func intRow(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>, step: Double) -> some View {
        let rounded = Binding<Double>(get: { value.wrappedValue }, set: { value.wrappedValue = ($0 / step).rounded() * step })
        return VStack(alignment: .leading, spacing: 3) {
            HStack { Text(title).font(.callout); Spacer()
                Text("\(Int(value.wrappedValue))").font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    .padding(.horizontal, 7).padding(.vertical, 1).background(.quaternary, in: Capsule()) }
            Slider(value: rounded, in: range)
        }.disabled(bench.running)
    }
}

// MARK: - dashboard

struct BenchDashboard: View {
    @ObservedObject var bench: BenchModel
    @ObservedObject var store: BenchStore
    @Binding var selectedRecord: UUID?
    let brand: Color

    private var shownCells: [BenchCell] {
        if let id = selectedRecord, let r = store.records.first(where: { $0.id == id }) { return r.cells }
        return bench.cells
    }
    private var shownRecord: BenchRecord? {
        if let id = selectedRecord { return store.records.first { $0.id == id } }
        return bench.lastRecord
    }

    /// The protocol being shown: the record's for a history record, else the live selection.
    private var shownKind: BenchKind { selectedRecord != nil ? (shownRecord?.kind ?? bench.kind) : (bench.running || bench.lastRecord == nil ? bench.kind : bench.lastRecord!.kind) }
    /// Colour means SPEED (the iOS StatsHUD rule with Mac breakpoints): the cell's median model time
    /// decides it, whatever the model or unit. Under 20 ms purple, 20 to 40 green, 40 to 100 orange, slower red.
    static func msColor(_ ms: Double) -> Color {
        switch ms {
        case ..<20: return Color(red: 0.69, green: 0.32, blue: 0.87)
        case ..<40: return Color(red: 0.20, green: 0.84, blue: 0.29)
        case ..<100: return Color(red: 1.00, green: 0.58, blue: 0.00)
        default: return Color(red: 0.96, green: 0.26, blue: 0.21)
        }
    }
    private func cellColor(_ c: BenchCell) -> Color {
        if shownKind == .sustained { return brand }   // the sustained traces are all blue; the throttle figure is the story there
        if bench.running, bench.liveCell?.id == c.id, bench.liveSamples.count > 1 { return BenchDashboard.msColor(StageStats(bench.liveSamples.map(\.ms)).median) }
        return BenchDashboard.msColor(c.headlineMs ?? 0)
    }
    /// A light wash of a colour for bands and row highlights. Translucent on purpose: it composites over
    /// whatever the card draws in the current appearance, so a light window never gets a band blended
    /// against the dark window colour (that happened when the app switched modes).
    private static func tint(_ c: Color, _ f: CGFloat) -> Color { c.opacity(1 - f) }
    /// A darker shade of a colour (mixed with black, opaque).
    private static func shade(_ c: Color, _ f: CGFloat) -> Color {
        let n = NSColor(c).usingColorSpace(.sRGB) ?? .gray
        return Color(nsColor: n.blended(withFraction: f, of: .black) ?? n)
    }
    private func cellSolid(_ c: BenchCell) -> Color { cellColor(c) }
    private func cellFill(_ c: BenchCell) -> AnyShapeStyle { AnyShapeStyle(cellSolid(c)) }
    /// Line dash by compute unit for the time chart: ANE solid, GPU dashed, CPU dotted.
    private func cellDash(_ c: BenchCell) -> [CGFloat] { [] }
    private func swatch(_ c: BenchCell) -> some View {
        RoundedRectangle(cornerRadius: 2).fill(cellSolid(c)).frame(width: 14, height: 10)
    }
    // ---- zoom: the charts open on the full data range; the user zooms the value axis (pinch or the
    // Y buttons) and the time axis (X buttons; the chart then scrolls sideways) ----
    @State private var zoomX: Double = 1          // histogram only: narrows the latency axis around the median
    @State private var pinchBase: Double = 1
    // ---- the cell the distribution card shows: clicked in the side-by-side chart or the results table,
    // or moved with the up / down arrow keys ----
    private var focusCell: BenchCell? {
        if bench.running { return bench.liveCell }
        let cells = shownCells
        return cells.first { $0.id == bench.selectedCellID }
            ?? (shownKind == .sustained ? (cells.last { $0.headlineMs != nil } ?? cells.last) : cells.first)
    }
    private func deleteRun(_ id: UUID) {
        if bench.deleteCell(id, from: shownRecord) { selectedRecord = nil }
    }
    @State private var hoverRowID: UUID? = nil     // the comparison row under the pointer (for its context menu)
    private func zoomBar() -> some View {
        HStack(spacing: 4) {
            Text("X").font(.caption2).foregroundStyle(.tertiary)
            Button { zoomX = max(1, zoomX / 1.5) } label: { Image(systemName: "minus.magnifyingglass") }.disabled(zoomX <= 1)
            Button { zoomX = min(64, zoomX * 1.5) } label: { Image(systemName: "plus.magnifyingglass") }
            Button { zoomX = 1 } label: { Image(systemName: "arrow.counterclockwise") }.disabled(zoomX == 1).padding(.leading, 6)
        }
        .buttonStyle(.borderless).controlSize(.mini).foregroundStyle(.secondary)
    }

    /// Which lower card is expanded: the main chart folds to half the stage and the other card retracts.
    enum Expanded { case comparison, results }
    @State private var expanded: Expanded? = nil
    private func expandButton(_ which: Expanded) -> some View {
        let on = expanded == which
        return Button { withAnimation(.easeInOut(duration: 0.2)) { expanded = on ? nil : which } } label: {
            Image(systemName: on ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right")
        }
        .buttonStyle(.borderless).controlSize(.small).foregroundStyle(.secondary)
        .help(on ? "Restore the layout" : "Expand this card (the main chart folds to half, the other card retracts)")
    }
    /// The charts column: stat cards, the main chart (the whole stage, or half of it while a lower card
    /// is expanded), then the comparison card and the results table (the expanded one takes the rest,
    /// the other retracts).
    private func stageContent(_ stage: GeometryProxy) -> some View {
        let half = max(stage.size.height / 2, 160)
        let sustainedLive = bench.running && shownKind == .sustained   // the earlier cells fold away while a new one streams
        let showComparison = expanded != .results && !sustainedLive && (shownCells.count > 1 || (bench.running && !bench.cells.isEmpty && shownKind == .cold))
        // the height the column needs with the main chart at its minimum; past that, the column scrolls
        let showResults = !shownCells.isEmpty && expanded != .comparison && !sustainedLive
        var needed: CGFloat = 64 + 14 + 240
        if shownKind == .sustained && expanded == nil { needed += 14 + 260 }
        if showComparison { needed += 14 + (expanded == .comparison ? 5 * 40 + 70 : comparisonHeight) }
        if showResults { needed += 14 + 26 * CGFloat(min(shownCells.count, 5)) + 70 }
        let scrolls = expanded == nil && needed > stage.size.height
        let column = VStack(spacing: 14) {
            statCards
            mainChart.frame(minHeight: expanded == nil ? 220 : 0, maxHeight: scrolls ? 320 : (expanded == nil ? .infinity : half))
            if shownKind == .sustained && expanded == nil { sustainedChart.frame(height: 260) }
            if showComparison {
                if expanded == .comparison { comparisonChart } else { comparisonChart.frame(height: comparisonHeight) }
            }
            if showResults { resultsTable }
        }
        return Group {
            if scrolls {
                ScrollView(.vertical) { column.frame(width: stage.size.width) }
                    .frame(width: stage.size.width, height: stage.size.height)
            } else {
                column.frame(width: stage.size.width, height: stage.size.height, alignment: .top)
            }
        }
    }
    @ViewBuilder private var mainChart: some View {
        switch shownKind {
        case .sustained: timeChart
        case .cold: histogramChart
        case .accuracy:
            if let acc = focusCell?.accuracy { accuracyChart(acc) } else { accuracyPending }
        }
    }
    var body: some View {
        VStack(spacing: 14) {
            header
            HStack(alignment: .top, spacing: 14) {
                GeometryReader { stage in stageContent(stage) }
                ThermometerView(meters: bench.meters, running: bench.running).frame(width: 96)
                PowerMeterView(meters: bench.meters).frame(width: 96)
            }
            if shownCells.isEmpty && !bench.running {
                VStack(spacing: 6) {
                    Image(systemName: "gauge.with.dots.needle.67percent").font(.system(size: 40)).foregroundStyle(.tertiary)
                    Text("Pick models, compute units and a protocol in the sidebar, then Run.").font(.callout).foregroundStyle(.secondary)
                    Text("Every run streams here as it happens and is kept in History.").font(.caption).foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity, minHeight: 160)
            }
        }
        .padding(16)
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(selectedRecord != nil ? (shownRecord?.name ?? "Run") : (bench.running ? "Running" : (bench.lastRecord?.name ?? "Benchmark")))
                    .font(.title3.weight(.semibold)).lineLimit(1)
                Text(bench.running ? bench.phase : (bench.note.isEmpty ? (shownRecord.map { "\($0.cpuModel) · \($0.osVersion)" } ?? bench.phase) : bench.note))
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            if let p = bench.progress, bench.running { ProgressView(value: p).frame(maxWidth: .infinity).padding(.horizontal, 24) }
            if bench.running {
                Button(role: .destructive) { bench.cancel() } label: { Label("Stop", systemImage: "stop.fill") }
                    .onAppear { zoomX = 1; pinchBase = 1; bench.selectedCellID = nil; expanded = nil }
            } else {
                if let r = shownRecord, r.cells.contains(where: { $0.document != nil }) {
                    Button { bench.exportRecordJSON(r) } label: { Label("Export run", systemImage: "square.and.arrow.up") }
                        .help("Every cell of this run as one JSON array of yolomaster-bench/v1 documents")
                }
                Button { bench.run() } label: { Label("Run", systemImage: "play.fill") }
                    .buttonStyle(.borderedProminent).tint(brand).keyboardShortcut(.return, modifiers: .command)
                    .disabled(bench.selectedModels.isEmpty || bench.computes.isEmpty)
            }
        }
    }

    private var liveStats: StageStats? {
        if bench.running { return bench.liveSamples.count > 1 ? StageStats(bench.liveSamples.map(\.ms)) : bench.liveCell?.cold }
        return shownCells.first?.cold
    }
    private var statCards: some View {
        let s = liveStats
        let cell = bench.running ? bench.liveCell : shownCells.first
        return HStack(spacing: 10) {
            card("Median", s.map { String(format: "%.2f ms  ·  %.1f fps", $0.median, $0.median > 0 ? 1000 / $0.median : 0) } ?? "-")
            card("p90 / p99", s.map { String(format: "%.2f / %.2f ms", $0.p90, $0.p99) } ?? "-")
            card("Min / max", s.map { String(format: "%.2f / %.2f ms", $0.min, $0.max) } ?? "-")
            card("Samples", s.map { "\($0.n)" } ?? "-")
            if let su = cell?.sustained {
                card("Throttle", String(format: "%+.1f%%", su.throttle_pct))
            } else if let a = cell?.accuracy {
                card("mAP50-95", String(format: "%.4f", a.map5095))
            }
        }
    }
    private func card(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(.title3, design: .rounded).weight(.semibold).monospacedDigit()).lineLimit(1).minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    /// One series of (seconds since the cell started, model ms).
    private struct Series { let name: String; let color: Color; let dash: [CGFloat]; let cell: BenchCell?; let points: [(t: Double, ms: Double)] }
    /// While a run streams: the current cell, a rolling last-minute window. Once it is done (and for
    /// history records): every cell's full series overlaid from its own 0 s, one colour per cell.
    private var seriesToShow: (series: [Series], tStart: Double, tEnd: Double) {
        if bench.running, let c = bench.liveCell {
            let pts = bench.liveSamples
            let tEnd = max(pts.map(\.t).max() ?? 0, 1)
            // once the window rolls its width is exactly windowSeconds, so the slot width (and hence the
            // slot edges) stays constant from one redraw to the next
            let tStart = max(0, tEnd - BenchDashboard.windowSeconds)
            return ([Series(name: "\(c.modelName) · \(c.compute.rawValue)", color: cellSolid(c), dash: cellDash(c), cell: c, points: pts)],
                    tStart, tStart > 0 ? tStart + BenchDashboard.windowSeconds : max(tEnd, 1))
        }
        let cells = focusCell.map { [$0] } ?? []   // one run at a time: the selected cell (arrow keys / clicks switch it)
        let series = cells.map { c in
            Series(name: "\(c.modelName) · \(c.compute.rawValue)", color: cellSolid(c), dash: cellDash(c), cell: c,
                   points: c.samples.enumerated().map { ($0.offset < c.sampleTimes.count ? c.sampleTimes[$0.offset] : Double($0.offset), $0.element) })
        }
        let tEnd = max(series.flatMap { $0.points.map(\.t) }.max() ?? 0, 1)
        return (series, 0, tEnd)
    }
    private static let windowSeconds = 30.0
    /// The full data range with padding (nothing is clamped away by default).
    private static func yRange(_ values: [Double]) -> ClosedRange<Double> {
        guard let mn = values.min(), let mx = values.max() else { return 0...1 }
        let pad = max((mx - mn) * 0.05, 0.05)
        return (mn - pad)...(mx + pad)
    }
    private struct Bucket: Identifiable { let id: Int; let series: String; let t, lo, hi, mean, trend: Double }
    /// Decimate one series into time-aligned slots (min / max band + mean) and smooth the means.
    /// Slots are anchored to absolute time (slot k covers [k * width, (k + 1) * width)), so a rolling
    /// window never re-partitions the samples it already showed; the 8-slot moving average is warmed
    /// up on the slots preceding `tStart` and only slots inside the window are returned.
    private static func decimate(_ pts: [(t: Double, ms: Double)], name: String, from tStart: Double, to tEnd: Double,
                                 into range: ClosedRange<Double>, live: Bool) -> [Bucket] {
        guard !pts.isEmpty else { return [] }
        // live: a constant slot width (the rolling window's) even while the axis is still growing, so
        // slot edges never move; finished runs: 400 slots across the whole run
        let width = live ? BenchDashboard.windowSeconds / 400 : max((tEnd - tStart) / 400, 0.01)
        let smooth = 8
        let leadIn = tStart - Double(smooth) * width
        let clamp: (Double) -> Double = { min(max($0, range.lowerBound), range.upperBound) }
        // group by slot (the samples arrive in time order)
        var slots: [(k: Int, lo: Double, hi: Double, sum: Double, n: Int)] = []
        for p in pts where p.t >= leadIn {
            let k = Int((p.t / width).rounded(.down))
            if let last = slots.last, last.k == k {
                slots[slots.count - 1] = (k, min(last.lo, p.ms), max(last.hi, p.ms), last.sum + p.ms, last.n + 1)
            } else {
                slots.append((k, p.ms, p.ms, p.ms, 1))
            }
        }
        var out: [Bucket] = []; out.reserveCapacity(slots.count)
        var window: [Double] = []
        for sl in slots {
            let mean = sl.sum / Double(sl.n)
            window.append(mean); if window.count > smooth { window.removeFirst() }
            let t = Double(sl.k) * width
            if t + width < tStart { continue }   // lead-in slot: it only warmed the average up
            out.append(Bucket(id: sl.k, series: name, t: t, lo: clamp(sl.lo), hi: clamp(sl.hi), mean: clamp(mean),
                              trend: clamp(window.reduce(0, +) / Double(window.count))))
        }
        return out
    }

    /// Adaptive range for the sustained trace: the 1st to 99th percentile of the visible samples with
    /// 10 % padding, so a single outlier does not flatten the trace (it is clamped into the band).
    private static func adaptiveRange(_ values: [Double]) -> ClosedRange<Double> {
        guard values.count > 1 else { let v = values.first ?? 0; return (v - 0.5)...(v + 0.5) }
        let sorted = values.sorted()
        let lo = sorted[Int(Double(sorted.count) * 0.01)], hi = sorted[min(Int(Double(sorted.count) * 0.99), sorted.count - 1)]
        let pad = max((hi - lo) * 0.10, 0.05)
        return (lo - pad)...(hi + pad)
    }
    private var timeChart: some View {
        let shown = seriesToShow
        let visible = shown.series.flatMap { s in (shown.tStart > 0 ? s.points.filter { $0.t >= shown.tStart } : s.points).map(\.ms) }
        let range = BenchDashboard.adaptiveRange(visible)
        let med = visible.isEmpty ? 0 : StageStats(visible).median
        let buckets = shown.series.map { BenchDashboard.decimate($0.points, name: $0.name, from: shown.tStart, to: shown.tEnd, into: range, live: bench.running) }
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Model time per iteration").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
            }
            Chart {
                ForEach(Array(buckets.enumerated()), id: \.offset) { k, bs in
                    ForEach(bs) { b in
                        AreaMark(x: .value("s", b.t), yStart: .value("min", b.lo), yEnd: .value("max", b.hi), series: .value("series", b.series + " band"))
                            .foregroundStyle(BenchDashboard.tint(shown.series[k].color, 0.8))
                        LineMark(x: .value("s", b.t), y: .value("ms", b.trend), series: .value("series", b.series))
                            .foregroundStyle(shown.series[k].color).lineStyle(StrokeStyle(lineWidth: 2, dash: shown.series[k].dash)).interpolationMethod(.monotone)
                    }
                }
                if !visible.isEmpty && shown.series.count == 1 {
                    RuleMark(y: .value("median", med)).foregroundStyle(.secondary.opacity(0.6)).lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                }
            }
            .chartYAxisLabel("ms").chartXAxisLabel("seconds")
            .chartXScale(domain: shown.tStart...max(shown.tEnd, shown.tStart + 1))
            .chartYScale(domain: range)
            .chartLegend(.hidden)
            if shown.series.count > 1 {   // legend with the unit dash visible
                HStack(spacing: 12) {
                    ForEach(Array(shown.series.enumerated()), id: \.offset) { _, sr in
                        HStack(spacing: 4) {
                            Path { p in p.move(to: .init(x: 0, y: 5)); p.addLine(to: .init(x: 22, y: 5)) }
                                .stroke(sr.color, style: StrokeStyle(lineWidth: 2, dash: sr.dash)).frame(width: 22, height: 10)
                            Text(sr.name).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    /// Cold run: the latency distribution of ONE cell's timed iterations (the live cell while a run
    /// streams, else the cell picked in the side-by-side chart or the results table), in its model's
    /// colour, with median / p90 / p99 marked. Pinch or the X buttons zoom the ms axis.
    private var histogramChart: some View {
        let cell = focusCell
        let values: [Double] = bench.running ? bench.liveSamples.map(\.ms) : (cell?.samples ?? [])
        var range = BenchDashboard.yRange(values)
        if zoomX > 1 {   // narrow the axis around the median
            let med = values.count > 1 ? StageStats(values).median : range.lowerBound
            let half = (range.upperBound - range.lowerBound) / 2 / zoomX
            range = max(range.lowerBound, med - half)...min(range.upperBound, med + half)
        }
        let binCount = 48
        let width = max((range.upperBound - range.lowerBound) / Double(binCount), 1e-6)
        struct Bin: Identifiable { let id: Int; let lo: Double; let hi: Double; let n: Int }
        var counts = [Int](repeating: 0, count: binCount)
        for v in values where v >= range.lowerBound && v <= range.upperBound {
            counts[min(max(Int((v - range.lowerBound) / width), 0), binCount - 1)] += 1
        }
        // bars span their bin (xStart / xEnd): a real histogram, whatever the axis scale
        let bins = counts.enumerated().filter { $0.element > 0 }.map {
            Bin(id: $0.offset, lo: range.lowerBound + Double($0.offset) * width, hi: range.lowerBound + Double($0.offset + 1) * width, n: $0.element)
        }
        let stats = values.count > 1 ? StageStats(values) : nil
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Latency distribution (iterations per bin)").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                if let c = cell {
                    swatch(c)
                    Text("\(c.modelName) · \(c.compute.rawValue)").font(.caption.weight(.semibold)).foregroundStyle(.primary)
                    if !bench.running && shownCells.count > 1 { Text("click a bar or a row to switch").font(.caption2).foregroundStyle(.tertiary) }
                }
                Spacer()
                zoomBar()
            }
            Chart {
                ForEach(bins) { b in   // each bin takes the speed colour of its own latency
                    RectangleMark(xStart: .value("from", b.lo), xEnd: .value("to", b.hi), yStart: .value("zero", 0), yEnd: .value("count", b.n))
                        .foregroundStyle(BenchDashboard.msColor((b.lo + b.hi) / 2))
                }
                if let st = stats {   // labels above their rule; the chart keeps a top margin for them
                    RuleMark(x: .value("median", st.median)).foregroundStyle(.primary).lineStyle(StrokeStyle(lineWidth: 1.5))
                        .annotation(position: .top, alignment: .leading) { Text("median").font(.caption2) }
                    RuleMark(x: .value("p90", st.p90)).foregroundStyle(.secondary).lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                        .annotation(position: .top, alignment: .leading) { Text("p90").font(.caption2).foregroundStyle(.secondary) }
                    RuleMark(x: .value("p99", st.p99)).foregroundStyle(.secondary).lineStyle(StrokeStyle(lineWidth: 1, dash: [2, 3]))
                        .annotation(position: .top, alignment: .leading) { Text("p99").font(.caption2).foregroundStyle(.secondary) }
                }
            }
            .chartXAxisLabel("ms")
            .chartXScale(domain: range)
            .chartYScale(domain: 0...Double(max(counts.max() ?? 1, 1)) * 1.05)
            .padding(.top, 16)   // room above the plot for the median / p90 / p99 labels (the y-axis label used to provide it)
            .gesture(MagnifyGesture().onChanged { v in zoomX = max(1, min(64, pinchBase * v.magnification)) }.onEnded { _ in pinchBase = zoomX })
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    /// One row of the side-by-side chart per cell; the card is capped and scrolls beyond six rows.
    private var comparisonRows: Int { max(bench.running ? bench.cells.count : shownCells.count, 1) }
    private var comparisonHeight: CGFloat { min(CGFloat(comparisonRows) * 40 + 70, 5 * 40 + 70) }
    private struct Row: Identifiable { let id: UUID; let name: String; let value: Double; let hi: Double; let color: Color; let fill: AnyShapeStyle }
    /// Accuracy row: the mAP50 extension (a lighter bar of the same colour continuing the mAP50-95 bar)
    /// and the label past the 1.0 edge.
    @ChartContentBuilder private func accuracyMarks(_ r: Row, xMax: Double) -> some ChartContent {
        if r.hi > r.value {
            BarMark(xStart: .value("lo", r.value), xEnd: .value("hi", min(r.hi, xMax)), y: .value("cell", r.name), height: .ratio(0.62))
                .foregroundStyle(r.color.opacity(0.28))
        }
        PointMark(x: .value("end", xMax), y: .value("cell", r.name)).opacity(0)
            .annotation(position: .trailing, spacing: 6) {
                Text(r.hi > 0 ? String(format: "%.4f / %.4f", r.value, r.hi) : String(format: "%.4f", r.value)).font(.caption2.monospacedDigit())
            }
    }
    /// Latency row: the p90 diamond and the "median (p90)" label after it.
    @ChartContentBuilder private func latencyMarks(_ r: Row) -> some ChartContent {
        if r.hi > 0 {
            PointMark(x: .value("hi", r.hi), y: .value("cell", r.name)).symbol(.diamond).foregroundStyle(.primary).symbolSize(30)
                .annotation(position: .trailing, spacing: 6) {
                    Text(String(format: "%.2f ms  (p90 %.2f)", r.value, r.hi)).font(.caption2.monospacedDigit())
                }
        } else {
            PointMark(x: .value("value", r.value), y: .value("cell", r.name)).opacity(0)
                .annotation(position: .trailing, spacing: 6) {
                    Text(String(format: "%.2f ms", r.value)).font(.caption2.monospacedDigit())
                }
        }
    }
    /// Cells side by side: median with the p90 whisker (cold / sustained / dataset) or mAP50-95 (accuracy).
    private var comparisonChart: some View {
        let cells = bench.running ? bench.cells : shownCells
        let accuracyMode = shownKind == .accuracy
        var seen: [String: Int] = [:]
        let rows = cells.map { c -> Row in
            let base = "\(c.modelName) · \(c.compute.rawValue)"
            seen[base, default: 0] += 1
            let name = seen[base]! > 1 ? "\(base) #\(seen[base]!)" : base
            if accuracyMode { return Row(id: c.id, name: name, value: c.accuracy?.map5095 ?? 0, hi: c.accuracy?.map50 ?? 0, color: cellSolid(c), fill: cellFill(c)) }
            return Row(id: c.id, name: name, value: c.cold?.median ?? 0, hi: c.cold?.p90 ?? 0, color: cellSolid(c), fill: cellFill(c))
        }
        let xMax: Double = accuracyMode ? 1.0 : max((rows.map(\.hi).max() ?? 1) * 1.3, 0.1)
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(accuracyMode ? "Accuracy Comparison per Run" : "Latency Comparison per Run")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Text(accuracyMode ? "mAP50-95 / mAP50" : "median (p90)")   // the column of labels past the axis
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                if comparisonRows > 5 || expanded == .comparison { expandButton(.comparison).padding(.leading, 6) }
            }
            ScrollViewReader { sp in
            ScrollView(.vertical) {
            ZStack(alignment: .top) {
            Chart {
                ForEach(rows) { r in
                    BarMark(x: .value("value", r.value), y: .value("cell", r.name), width: .ratio(0.62))
                        .foregroundStyle(r.fill)
                    if accuracyMode { accuracyMarks(r, xMax: xMax) } else { latencyMarks(r) }
                }
            }
            .chartXAxisLabel(accuracyMode ? "mAP" : "ms")
            .chartXScale(domain: 0...xMax)
            .chartBackground { proxy in   // the selected cell: a wash across its whole row, sized from the chart's own band geometry
                GeometryReader { geo in
                    if let anchor = proxy.plotFrame, let i = rows.firstIndex(where: { $0.id == focusCell?.id }) {
                        let plot = geo[anchor]
                        let band = plot.height / CGFloat(max(rows.count, 1))
                        RoundedRectangle(cornerRadius: 6).fill(BenchDashboard.tint(rows[i].color, 0.86))
                            .frame(width: geo.size.width + (accuracyMode ? 110 : 0), height: band)   // out to the labels past the axis
                            .offset(y: plot.minY + CGFloat(i) * band)
                    }
                }
            }
            .padding(.trailing, accuracyMode ? 110 : 0)   // the "map / map50" labels live past the 1.0 edge
            .chartOverlay { proxy in   // a click (not hover) picks the row under the pointer
                GeometryReader { geo in
                    ZStack {
                        Rectangle().fill(.clear).contentShape(Rectangle())
                            .onTapGesture { location in
                                guard let anchor = proxy.plotFrame else { return }
                                let plot = geo[anchor]
                                let y = location.y - plot.origin.y
                                if let name: String = proxy.value(atY: y), let r = rows.first(where: { $0.name == name }) { bench.selectedCellID = r.id }
                            }
                            .onContinuousHover { phase in
                                switch phase {
                                case .active(let loc):
                                    guard let anchor = proxy.plotFrame else { return }
                                    let plot = geo[anchor]
                                    if let name: String = proxy.value(atY: loc.y - plot.origin.y), let r = rows.first(where: { $0.name == name }) { hoverRowID = r.id } else { hoverRowID = nil }
                                case .ended: hoverRowID = nil
                                }
                            }
                            .contextMenu {   // right-click: the row under the pointer
                                if let id = hoverRowID, let r = rows.first(where: { $0.id == id }) {
                                    Button("Delete run \"\(r.name)\"", role: .destructive) { deleteRun(id) }.disabled(bench.running)
                                }
                            }
                    }
                }
            }
            .frame(height: CGFloat(max(rows.count, 1)) * 40 + 40)
            .padding(.trailing, 14)   // room for the scrollbar
            VStack(spacing: 0) {   // one invisible anchor per row so the selection can be scrolled to
                ForEach(rows) { r in Color.clear.frame(height: 40).id(r.id) }
            }
            .padding(.top, 8).allowsHitTesting(false)
            }
            }
            .onChange(of: bench.selectedCellID) { if let id = bench.selectedCellID { withAnimation { sp.scrollTo(id, anchor: .center) } } }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: expanded == .comparison ? .infinity : nil, alignment: .top)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    private var accuracyPending: some View {
        VStack(spacing: 8) {
            if bench.running {
                ProgressView(value: bench.progress ?? 0).frame(width: 260)
                Text(bench.phase).font(.caption).foregroundStyle(.secondary)
                if bench.liveSamples.count > 1 {
                    let st = StageStats(bench.liveSamples.map(\.ms))
                    Text(String(format: "%d images scored · model median %.2f ms", st.n, st.median)).font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
                }
            } else {
                Image(systemName: "checkmark.seal").font(.system(size: 30)).foregroundStyle(.tertiary)
                Text("The per-class AP chart appears once the accuracy pass has scored the set.").font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    /// Sustained: one median per second, coloured by the thermal state of that second.
    private var sustainedChart: some View {
        let points: [(t: Double, med: Double, thermal: Int)] = {
            if bench.running, !bench.liveSeconds.isEmpty { return bench.liveSeconds }
            if let c = focusCell, let su = c.sustained {
                // the thermal track is sampled once per completed second; the sparkline's last (partial)
                // second has no sample of its own and keeps the last known state, not "Cool"
                return su.sparkline.enumerated().map { (Double($0.offset), $0.element, $0.offset < c.thermal.count ? c.thermal[$0.offset] : (c.thermal.last ?? 0)) }
            }
            return []
        }()
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Sustained: FPS and thermal state").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                ForEach(0..<4, id: \.self) { l in HStack(spacing: 3) { Circle().fill(thermalColor(l)).frame(width: 7, height: 7); Text(thermalName(l)).font(.caption2).foregroundStyle(.secondary) } }
            }
            let fps = points.map { (t: $0.t, fps: $0.med > 0 ? 1000 / $0.med : 0, thermal: $0.thermal) }
            let tMax = max(fps.map(\.t).max() ?? 1, 1)
            // one smooth line; the thermal state colours it along x through a hard-stop gradient (the line's
            // frame spans 0...tMax, the x domain, so a stop at t / tMax lands exactly on that second)
            let stops: [Gradient.Stop] = {
                var out: [Gradient.Stop] = []
                for (i, p) in fps.enumerated() {
                    let color = thermalColor(p.thermal)
                    let x0 = i == 0 ? 0 : (fps[i - 1].t + p.t) / 2 / tMax     // the colour changes halfway between samples
                    let x1 = i == fps.count - 1 ? 1 : (p.t + fps[i + 1].t) / 2 / tMax
                    out.append(.init(color: color, location: x0)); out.append(.init(color: color, location: x1))
                }
                return out.isEmpty ? [.init(color: .secondary, location: 0), .init(color: .secondary, location: 1)] : out
            }()
            Chart {
                ForEach(Array(fps.enumerated()), id: \.offset) { _, p in
                    LineMark(x: .value("s", p.t), y: .value("fps", p.fps))
                }
                .interpolationMethod(.monotone)
                .lineStyle(StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                .foregroundStyle(LinearGradient(stops: stops, startPoint: .leading, endPoint: .trailing))
            }
            .chartYAxisLabel("fps").chartXAxisLabel("seconds")
            .chartXScale(domain: 0...tMax)
            .chartYScale(domain: {   // one value per second: show them all, padded, never outside the plot
                let ys = fps.map(\.fps); let lo = ys.min() ?? 0, hi = ys.max() ?? 1; let pad = max((hi - lo) * 0.12, 0.5)
                return (lo - pad)...(hi + pad)
            }())
            .chartLegend(.hidden)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    private var resultsTable: some View {
        let headers = ["Model", "Unit", "Pre", "Median ms", "p90", "p99", "Min", "FPS"]
        let rowH: CGFloat = 26
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Results").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                if shownCells.count > 5 || expanded == .results { expandButton(.results) }
            }
            // header outside the scroll view so it never scrolls away; the same column widths as the rows
            resultsRow(headers, bold: true, trailing: { Spacer().frame(width: 22, height: 1) })
                .frame(height: 18).padding(.horizontal, 6)
            Divider()
            ScrollViewReader { sp in
            ScrollView(.vertical) {
                VStack(spacing: 0) {
                    ForEach(shownCells) { c in
                        resultsRow(rowValues(c), bold: false) {
                            Button { bench.saveJSON(c) } label: { Image(systemName: "square.and.arrow.down") }
                                .buttonStyle(.borderless).help("Save the yolomaster-bench/v1 JSON").disabled(c.document == nil).frame(width: 22)
                        }
                        .frame(height: rowH)
                        .padding(.horizontal, 6)
                        .background(RoundedRectangle(cornerRadius: 5).fill(focusCell?.id == c.id ? BenchDashboard.tint(cellColor(c), 0.82) : .clear))
                        .contentShape(Rectangle())
                        .onTapGesture { bench.selectedCellID = c.id }
                        .contextMenu {
                            Button("Delete run \"\(c.modelName) · \(c.compute.rawValue)\"", role: .destructive) { deleteRun(c.id) }.disabled(bench.running)
                        }
                        .id(c.id)
                    }
                }
                .padding(.trailing, 14)   // room for the scrollbar
            }
            .frame(height: expanded == .results ? nil : rowH * CGFloat(min(shownCells.count, 5)))
            .frame(maxHeight: expanded == .results ? .infinity : nil)
            .onChange(of: bench.selectedCellID) { if let id = bench.selectedCellID { withAnimation { sp.scrollTo(id, anchor: .center) } } }
            }
        }
        .fixedSize(horizontal: false, vertical: expanded != .results)   // the card is exactly as tall as header + rows unless expanded
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }
    /// One table line: a fixed model column, equal shares for the numbers.
    private func resultsRow<T: View>(_ cells: [String], bold: Bool, @ViewBuilder trailing: () -> T) -> some View {
        HStack(spacing: 8) {
            ForEach(Array(cells.enumerated()), id: \.offset) { i, v in
                Text(v).font(bold ? .caption.weight(.semibold) : .callout.monospacedDigit()).lineLimit(1).truncationMode(.middle)
                    .frame(width: i == 0 ? 150 : nil, alignment: .leading)
                    .frame(maxWidth: i == 0 ? 150 : .infinity, alignment: .leading)
            }
            trailing()
        }
    }
    private func rowValues(_ c: BenchCell) -> [String] {
        func f2(_ v: Double?) -> String { v.map { String(format: "%.2f", $0) } ?? "-" }
        return [c.modelName, c.compute.rawValue, c.preproc, f2(c.cold?.median), f2(c.cold?.p90), f2(c.cold?.p99), f2(c.cold?.min),
                String(format: "%.1f", c.fps)]
    }
    private func cellText(_ v: String) -> some View {
        Text(v).font(.callout.monospacedDigit()).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
    }

    /// AP colour bands: below 0.1 red, 0.1 to 0.2 orange, 0.2 to 0.3 yellow, 0.3 to 0.6 green, 0.6 and up purple.
    static func apColor(_ ap: Double) -> Color {
        switch ap {
        case ..<0.1: return Color(red: 0.96, green: 0.26, blue: 0.21)
        case ..<0.2: return Color(red: 1.00, green: 0.58, blue: 0.00)
        case ..<0.3: return Color(red: 0.98, green: 0.80, blue: 0.18)
        case ..<0.6: return Color(red: 0.20, green: 0.84, blue: 0.29)
        default: return Color(red: 0.69, green: 0.32, blue: 0.87)
        }
    }
    @State private var hoverClass: Int? = nil
    private func accuracyChart(_ a: BenchDocument.Accuracy) -> some View {
        let rows = a.per_class.sorted { $0.ap5095 > $1.ap5095 }
        let names = focusCell?.classNames ?? []
        func name(_ id: Int) -> String { id < names.count ? names[id] : "class \(id)" }
        let hovered = rows.first { $0.class_id == hoverClass }
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Accuracy per class (AP50-95, \(rows.count) classes)").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                if let h = hovered {
                    Text(name(h.class_id)).font(.caption.weight(.semibold)).foregroundStyle(.primary)
                    Text(String(format: "AP50-95 %.3f · AP50 %.3f · %d GT · %d predictions", h.ap5095, h.ap50, h.n_gt, h.n_pred))
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                } else {
                    Text("hover a bar for the class").font(.caption2).foregroundStyle(.tertiary)
                }
                Spacer()
            }
            Chart {
                ForEach(rows, id: \.class_id) { c in
                    BarMark(x: .value("class", "\(c.class_id)"), y: .value("AP", c.ap5095))
                        .foregroundStyle(hoverClass == nil || hoverClass == c.class_id ? brand : brand.opacity(0.35))
                }
                RuleMark(y: .value("mAP", a.map5095)).foregroundStyle(.secondary).lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
            }
            .chartYScale(domain: 0...1)
            .chartXAxis(.hidden)   // eighty class ids do not fit; the hover names the class instead
            .chartOverlay { proxy in
                GeometryReader { geo in
                    Rectangle().fill(.clear).contentShape(Rectangle())
                        .onContinuousHover { phase in
                            switch phase {
                            case .active(let loc):
                                guard let anchor = proxy.plotFrame else { return }
                                let plot = geo[anchor]
                                if let id: String = proxy.value(atX: loc.x - plot.origin.x) { hoverClass = Int(id) } else { hoverClass = nil }
                            case .ended: hoverClass = nil
                            }
                        }
                }
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }
}

/// A timer-style duration control: big MM : SS digits with steppers, like the Clock app's timer.
/// Minutes 0 to 30, seconds in quarter-minute steps; the value is minutes as a Double.
struct TimerDial: View {
    @Binding var minutes: Double
    private var mm: Int { Int(minutes) }
    private var ss: Int { Int(((minutes - Double(mm)) * 60).rounded()) }
    /// The seconds steppers snap to the quarter-minute grid: from a typed :08, up goes to :15 and down to :00.
    private func nextQuarter(_ s: Int, up: Bool) -> Int {
        if up { return (s / 15 + 1) * 15 }
        return s % 15 == 0 ? s - 15 : (s / 15) * 15
    }
    /// Seconds carry into minutes like a clock (:45 + 15 s -> next minute :00, :00 - 15 s -> previous :45);
    /// the total is held to 0:15 ... 30:00, so 29:45 + 15 s lands on 30:00 and stops there.
    private func set(_ m: Int, _ s: Int) {
        var total = m * 60 + s
        total = max(15, min(30 * 60, total))
        minutes = Double(total) / 60
    }
    var body: some View {
        HStack(spacing: 6) {
            Spacer(minLength: 0)
            digitColumn(value: mm, label: "min", up: { set(mm + 1, ss) }, down: { set(mm - 1, ss) })
            Text(":").font(.system(size: 34, weight: .light, design: .rounded)).foregroundStyle(.secondary).padding(.bottom, 14)
            digitColumn(value: ss, label: "sec", up: { set(mm, nextQuarter(ss, up: true)) }, down: { set(mm, nextQuarter(ss, up: false)) })
            Spacer(minLength: 0)
        }
        .padding(.vertical, 14)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.04)))
    }
    @State private var editing: String? = nil       // "min" | "sec" while a group is being typed
    @State private var draft = ""
    @FocusState private var focused: Bool
    private func commit() {
        defer { editing = nil; draft = "" }
        guard let n = Int(draft.trimmingCharacters(in: .whitespaces)) else { return }
        if editing == "min" { set(min(n, 30), n >= 30 ? 0 : ss) }
        else if editing == "sec" { set(mm, min(max(n, 0), 59)) }
    }
    private func digitColumn(value: Int, label: String, up: @escaping () -> Void, down: @escaping () -> Void) -> some View {
        VStack(spacing: 0) {
            Button(action: up) { Image(systemName: "chevron.up").font(.caption2) }.buttonStyle(.borderless).foregroundStyle(.secondary)
                .frame(height: 20)
            if editing == label {   // double-clicked: type the number (Return commits, Escape cancels)
                TextField("", text: $draft)
                    .textFieldStyle(.plain).multilineTextAlignment(.center)
                    .font(.system(size: 34, weight: .light, design: .rounded).monospacedDigit())
                    .focused($focused)
                    .onSubmit { commit() }
                    .onExitCommand { editing = nil; draft = "" }
                    .onChange(of: focused) { if !focused && editing == label { commit() } }
                    .onAppear { focused = true }
            } else {
                Text(String(format: "%02d", value))
                    .font(.system(size: 34, weight: .light, design: .rounded).monospacedDigit())
                    .contentTransition(.numericText())
                    .animation(.easeInOut(duration: 0.15), value: value)
                    .contentShape(Rectangle())
                    .onTapGesture(count: 2) { draft = String(value); editing = label }
                    .help("Double-click to type a value (30 minutes at most)")
            }
            Button(action: down) { Image(systemName: "chevron.down").font(.caption2) }.buttonStyle(.borderless).foregroundStyle(.secondary)
                .frame(height: 20)
            Text(label).font(.caption2).foregroundStyle(.tertiary)
        }
        .frame(width: 64)
    }
}

/// Equal-width bordered buttons that fill their row (the Units row look), used wherever a segmented
/// picker would size itself to its labels and leave the row ragged.
struct SegmentedButtons<T: Hashable>: View {
    let options: [(T, String)]
    var icons: [String] = []          // optional SF Symbols, parallel to options
    @Binding var selection: T
    let tint: Color
    var body: some View {
        HStack(spacing: 8) {
            ForEach(Array(options.enumerated()), id: \.offset) { i, o in
                let on = selection == o.0
                Button { selection = o.0 } label: {
                    Group {
                        if i < icons.count { Label(o.1, systemImage: icons[i]) } else { Text(o.1) }
                    }
                    .font(.callout).lineLimit(1).minimumScaleFactor(0.8).frame(maxWidth: .infinity).padding(.vertical, 4)
                }
                .buttonStyle(.bordered).tint(on ? tint : .secondary)
                .background(RoundedRectangle(cornerRadius: 6).fill(on ? tint.opacity(0.14) : .clear))
            }
        }
    }
}

/// A dropdown in the same dress as `SegmentedButtons`: full width, tinted, the current choice with its
/// icon and a chevron. `Picker(.menu)` on macOS is an NSPopUpButton that sizes to its content and
/// ignores `frame(maxWidth:)`; a `Menu` with our own label does not.
struct MenuButton<T: Hashable>: View {
    let options: [(T, String)]
    var icons: [String] = []
    @Binding var selection: T
    let tint: Color
    var body: some View {
        Menu {
            Picker("", selection: $selection) {
                ForEach(Array(options.enumerated()), id: \.offset) { i, o in
                    Group { if i < icons.count { Label(o.1, systemImage: icons[i]) } else { Text(o.1) } }.tag(o.0)
                }
            }.pickerStyle(.inline).labelsHidden()
        } label: {
            HStack(spacing: 6) {
                if let i = options.firstIndex(where: { $0.0 == selection }) {
                    if i < icons.count { Image(systemName: icons[i]) }
                    Text(options[i].1)
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.up.chevron.down").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            }
            .font(.callout).lineLimit(1).padding(.vertical, 7).padding(.horizontal, 10).frame(maxWidth: .infinity)
            .contentShape(Rectangle())
        }
        .menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden)
        .foregroundStyle(tint)
        .background(RoundedRectangle(cornerRadius: 6).fill(tint.opacity(0.14)))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(tint.opacity(0.35), lineWidth: 1))
        .frame(maxWidth: .infinity)
    }
}

/// A thermometer: four zones, the bulb filled to the current thermal state, the peak of the run marked.
struct ThermometerView: View {
    @ObservedObject var meters: MeterModel
    let running: Bool
    var body: some View {
        let level = meters.thermal
        // fill: the die temperature on a 30..110 C scale when a sensor exists, else the pressure level
        let frac: CGFloat = meters.celsius.map { CGFloat(min(max(($0 - 30) / 90, 0.04), 1)) } ?? CGFloat(level + 1) / 4   // 30..120 C
        return VStack(spacing: 8) {
            Text(thermalName(level)).font(.caption.weight(.semibold)).foregroundStyle(thermalColor(level)).lineLimit(1)
            GeometryReader { g in
                let h = g.size.height, w: CGFloat = 22
                let fill = h * frac
                ZStack(alignment: .bottom) {
                    Capsule().fill(Color.primary.opacity(0.08)).frame(width: w)
                    Capsule().fill(thermalColor(level))                     // the whole column takes the zone colour
                        .frame(width: w).mask(alignment: .bottom) { Rectangle().frame(height: max(w, fill)) }
                        .animation(.easeInOut(duration: 0.6), value: frac)
                        .animation(.easeInOut(duration: 0.4), value: level)
                }.frame(maxWidth: .infinity)
            }
            Text(meters.celsius.map { String(format: "%.0f °C", $0) } ?? thermalName(level))
                .font(.caption.weight(.semibold).monospacedDigit()).foregroundStyle(thermalColor(level))
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

}

/// Battery power flow: the bar grows downward from the zero line while discharging (watts drawn
/// from the battery) and upward while charging; a Mac on mains with a full battery sits at zero.
/// The two gauges as one compact card for the Inference sidebar: horizontal tubes, one row each.
/// Same readings and colours as the Bench dashboard's meters, in the sidebar's card style.
struct MetersStrip: View {
    @ObservedObject var meters: MeterModel
    private let tube: CGFloat = 10
    var body: some View {
        let level = meters.thermal
        let frac = meters.celsius.map { CGFloat(min(max(($0 - 30) / 90, 0.04), 1)) } ?? CGFloat(level + 1) / 4   // 30..120 C
        let b = meters.battery, w = b.watts
        VStack(spacing: 16) {
            gaugeRow(icon: "thermometer.medium", title: thermalName(level), color: thermalColor(level),
                     value: meters.celsius.map { String(format: "%.0f °C", $0) } ?? "no sensor") { width in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.08)).frame(width: width, height: tube)
                    Capsule().fill(thermalColor(level)).frame(width: max(tube, width * frac), height: tube)
                        .animation(.easeInOut(duration: 0.6), value: frac)
                        .animation(.easeInOut(duration: 0.4), value: level)
                }.frame(width: width, height: tube)
            }
            gaugeRow(icon: b.present ? (b.state == .onBattery ? "battery.50percent" : "powerplug.fill") : "powerplug",
                     title: b.present ? b.state.rawValue : "Power",
                     color: b.present ? (w < -0.05 ? .orange : (w > 0.05 ? .green : .secondary)) : .secondary,
                     value: b.present ? String(format: "%@%.1f W", w < 0 ? "-" : "+", abs(w)) : "no battery") { width in
                let half = width / 2, len = min(half, half * CGFloat(abs(w)) / 100)   // full half-bar = 100 W
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.08)).frame(width: width, height: tube)
                    if b.present {   // a rounded fill growing from the zero line, right when charging, left when draining
                        Capsule().fill(w < 0 ? Color.orange : Color.green)
                            .frame(width: max(tube, len), height: tube)
                            .offset(x: w < 0 ? half - max(tube, len) : half)
                            .animation(.easeInOut(duration: 0.25), value: w)
                    }
                    Rectangle().fill(Color.primary.opacity(0.35)).frame(width: 1, height: tube + 6).offset(x: half)   // zero line
                }.frame(width: width, height: tube)
            }
        }
    }
    private func gaugeRow<C: View>(icon: String, title: String, color: Color, value: String, @ViewBuilder bar: @escaping (CGFloat) -> C) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: icon).foregroundStyle(color).frame(width: 14)
                Text(title).font(.caption.weight(.semibold)).foregroundStyle(color).lineLimit(1)
                Spacer(minLength: 4)
                Text(value).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            GeometryReader { g in bar(g.size.width) }.frame(height: tube)
        }
    }
}

struct PowerMeterView: View {
    @ObservedObject var meters: MeterModel
    var body: some View {
        let b = meters.battery
        let w = b.watts
        let scale = 100.0                       // full bar = 100 W either way
        return VStack(spacing: 8) {
            Text(b.present ? b.state.rawValue : "Power").font(.caption.weight(.semibold)).foregroundStyle(.secondary).lineLimit(1)
            GeometryReader { g in
                let h = g.size.height, barW: CGFloat = 22
                let half = h / 2
                let len = min(half, half * CGFloat(abs(w)) / scale)
                ZStack(alignment: .center) {
                    Capsule().fill(Color.primary.opacity(0.08)).frame(width: barW)
                    Rectangle().fill(Color.primary.opacity(0.35)).frame(width: barW + 10, height: 1)       // zero line
                    if b.present {
                        Rectangle()
                            .fill(w < 0 ? LinearGradient(colors: [.orange, .red], startPoint: .top, endPoint: .bottom)
                                        : LinearGradient(colors: [.green, .mint], startPoint: .bottom, endPoint: .top))
                            .frame(width: barW, height: max(2, len))
                            .offset(y: w < 0 ? len / 2 : -len / 2)
                            .frame(width: barW, height: h)
                            .clipShape(Capsule())                 // the tube's outline crops the bar
                            .animation(.easeInOut(duration: 0.25), value: w)
                    }
                }.frame(maxWidth: .infinity)
            }
            if b.present {
                Text(String(format: "%@%.1f W", w < 0 ? "-" : "+", abs(w))).font(.caption.weight(.semibold).monospacedDigit())
                    .foregroundStyle(w < -0.05 ? Color.orange : (w > 0.05 ? Color.green : Color.secondary))
            } else {
                Text("no battery").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }
}

