// On-device benchmark, rebuilt in the shared HUD design language: tachometer
// dials, color-ramp bars, the thermal tach, a live throttle sparkline, haptics,
// and the animated export toast - matching the Live and Photo tabs.
//
// Methodology (from the Orin/S26 work): flagships throttle, so a real-time claim
// needs BOTH a cold median and a sustained (thermal) number. Headline metric is
// inferOnly (pure model predict wall-time); each result also carries one
// forward+decode pass for a pre/inf/dec stage breakdown. The synthetic gray input
// yields ~no detections, so the decode stage reads the empty-scene floor -
// inference is the number that matters.
import CoreGraphics
import SwiftUI
import UIKit
import YOLOMasterKit

struct BenchResult: Identifiable, Codable {
    var id: String { "\(modelId)|\(compute.rawValue)" }
    let modelId: String
    let shortID: String
    let fullName: String
    let compute: ComputeChoice
    let coldMedian: Double
    let coldP90: Double
    let coldMin: Double
    let pre: Double
    let inf: Double
    let dec: Double
    var sustainedMedian: Double? = nil
    var throttlePct: Double? = nil
    var fpsEquiv: Double { coldMedian > 0 ? 1000.0 / coldMedian : 0 }
}

/// Run-control flags the detached bench loop polls each iteration (word-sized
/// reads; a data flag, eventual consistency is fine). Lets a run be PAUSED
/// (suspended, resumable) rather than only cancelled, and auto-paused when the
/// tab loses focus so benchmarking never competes with Live/Photo inference.
final class BenchControl: @unchecked Sendable {
    var paused = false
    var cancelled = false
}

/// A saved benchmark run in the permanent history (JSON-persisted).
struct BenchRun: Identifiable, Codable {
    var id = UUID()
    var name: String
    let date: Date
    let mode: String                 // "Cold Sweep" | "Sustained"
    let results: [BenchResult]
    var sparkline: [Double]? = nil    // sustained: the throttle trend
    var thermalTimeline: [Int]? = nil // sustained: thermal level sampled over the run
    var thermalStart: Int? = nil      // sustained: thermal level at start
    var thermalEnd: Int? = nil        // thermal level at the captured moment / run end
    var thermalPeak: Int? = nil       // sustained: worst level reached
    var durationSec: Int? = nil       // sustained: actual elapsed run time (seconds)
    var fastest: BenchResult? { results.min { $0.coldMedian < $1.coldMedian } }
}

/// Persistent, JSON-backed store for saved benchmark runs (newest first).
final class BenchHistory: ObservableObject {
    static let shared = BenchHistory()   // one store shared by Bench + Settings
    @Published var runs: [BenchRun] = []
    private let fileURL: URL = {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return dir.appendingPathComponent("bench_history.json")
    }()
    init() {
        if let data = try? Data(contentsOf: fileURL),
           let decoded = try? JSONDecoder().decode([BenchRun].self, from: data) { runs = decoded }
    }
    private func save() {
        try? FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(runs) { try? data.write(to: fileURL) }
    }
    func add(_ run: BenchRun) { runs.insert(run, at: 0); save() }
    func rename(_ id: UUID, _ name: String) {
        if let i = runs.firstIndex(where: { $0.id == id }) { runs[i].name = name; save() }
    }
    func delete(_ ids: Set<UUID>) { runs.removeAll { ids.contains($0.id) }; save() }
    func clear() { runs.removeAll(); try? FileManager.default.removeItem(at: fileURL) }
}

/// Thermal-state colour shared by the tach and the history rows.
func benchUnitIcon(_ c: ComputeChoice) -> String {
    switch c {
    case .ane: return "bolt.fill"
    case .gpu: return "square.stack.3d.up.fill"
    case .cpu: return "cpu"
    }
}

/// Compact run length, e.g. "45s", "3m 12s", "30m".
func durationText(_ sec: Int) -> String {
    if sec < 60 { return "\(sec)s" }
    let m = sec / 60, s = sec % 60
    return s == 0 ? "\(m)m" : "\(m)m \(s)s"
}

func thermalLevelColor(_ level: Int) -> Color {
    switch level { case 0: return .blue; case 1: return .green; case 2: return .orange; default: return .red }
}

struct BenchView: View {
    enum Mode: String, CaseIterable, Identifiable {
        case sweep = "Cold Sweep", sustained = "Sustained"
        var id: String { rawValue }
    }
    enum Phase: Equatable {
        case idle
        case loadingModel(String, String)            // model shortID, unit
        case benchmarking(String, String, Int, Int)  // model, unit, done, total
        case sustained(String, String, Int, Int)     // model, unit, elapsedS, totalS
        case done
    }

    @State private var models: [BundledModel] = []
    @State private var results: [BenchResult] = []            // active mode's results (shown)
    @State private var stashedResults: [BenchResult] = []     // the OTHER mode's, kept hidden
    @State private var mode: Mode = .sweep
    @State private var phase: Phase = .idle
    @State private var running = false
    @State private var paused = false
    @State private var control = BenchControl()
    @ObservedObject private var history = BenchHistory.shared
    @State private var showHistory = false
    @State private var runThermalStart = 0
    @State private var runThermalPeak = 0
    @Environment(\.scenePhase) private var scenePhase
    @State private var showSettings = false
    @State private var expandedCards: Set<String> = []
    // sustained target + settings
    @State private var selectedModel: BundledModel?
    @State private var selectedCompute: ComputeChoice = ComputeChoice.deviceDefault
    @State private var sustainedMinutes = 3.0
    @State private var iters = 50
    @State private var warmup = 10
    // live readouts
    @State private var liveMs = 0.0
    @State private var runDuration = 0   // sustained: elapsed seconds (live, then final)
    @State private var sparkSamples: [Double] = []
    @State private var sparkThermal: [Int] = []
    @State private var sparkBaseline: Double? = nil
    @State private var thermal = ProcessInfo.processInfo.thermalState
    // export toast
    @State private var exportMsg: String? = nil
    @State private var exportOK = false
    @State private var exportGen = 0

    @State private var loopTask: Task<Void, Never>?
    private let lightHaptic = UIImpactFeedbackGenerator(style: .light)
    private let medHaptic = UIImpactFeedbackGenerator(style: .medium)
    private let notifyHaptic = UINotificationFeedbackGenerator()

    var body: some View {
        ZStack {
            if results.isEmpty && !running {
                // vertically centered, matching the Photo tab's "No photos" empty state
                ContentUnavailableView("No benchmarks yet",
                    systemImage: "speedometer",
                    description: Text("Run a cold sweep across every model and compute unit."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(spacing: 10) {
                        if running { progressCard }
                        if mode == .sustained, sparkSamples.count > 1 { sustainedGraphCard }
                        if mode == .sweep, let hero = fastest { heroCard(hero) }
                        ForEach(modelsWithResults, id: \.self) { modelCard($0) }
                    }
                    .padding(.horizontal, 10)
                    .padding(.bottom, 24)
                }
            }
            if let msg = exportMsg {
                ExportToast(success: exportOK, message: msg)
                    .transition(.scale(scale: 0.82).combined(with: .opacity))
                    .onTapGesture { withAnimation(.easeOut(duration: 0.2)) { exportMsg = nil } }
            }
        }
        .safeAreaInset(edge: .top) { topBar }
        .onAppear {
            lightHaptic.prepare(); medHaptic.prepare(); notifyHaptic.prepare()
            guard models.isEmpty else { return }
            Task.detached {
                let found = BundledModel.discover()
                await MainActor.run {
                    models = found
                    if selectedModel == nil { selectedModel = BundledModel.preferred(in: found) }
                }
            }
        }
        .onReceive(NotificationCenter.default.publisher(
            for: ProcessInfo.thermalStateDidChangeNotification)) { _ in
            thermal = ProcessInfo.processInfo.thermalState
            if running { runThermalPeak = max(runThermalPeak, thermalLevel) }
        }
        // leaving the tab (or backgrounding) auto-pauses the run so it never
        // competes with Live/Photo inference; the user resumes with play
        .onDisappear { autoPause() }
        .onChange(of: scenePhase) { _, ph in if ph != .active { autoPause() } }
        .onChange(of: mode) { _, _ in
            // preserve each mode's results: stash the current, restore the other
            let tmp = results; results = stashedResults; stashedResults = tmp
            expandedCards = []
            sparkSamples = []; sparkThermal = []
        }
        .sheet(isPresented: $showHistory) { HistoryView(history: history) }
    }

    private func autoPause() {
        if running && !paused { paused = true; control.paused = true }
    }

    // MARK: - top bar

    private var topBar: some View {
        VStack(spacing: 8) {
            HStack {
                Picker("Mode", selection: $mode) {
                    ForEach(Mode.allCases) { m in Text(m.rawValue).tag(m) }
                }
                .pickerStyle(.segmented)
                .disabled(running)
                Button { withAnimation { showSettings.toggle() } } label: {
                    Image(systemName: "slider.horizontal.3")
                }
                .buttonStyle(.bordered)
                Button { showHistory = true } label: {
                    Image(systemName: "clock.arrow.circlepath")
                }
                .buttonStyle(.bordered)
                if !results.isEmpty {
                    Button { exportCSV() } label: { Image(systemName: "square.and.arrow.up") }
                        .buttonStyle(.bordered)
                        .disabled(running)
                }
                if running {
                    Button { togglePause() } label: {
                        Image(systemName: paused ? "play.fill" : "pause.fill").frame(minWidth: 40)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(paused ? .accentColor : .orange)
                    Button { stop() } label: { Image(systemName: "xmark") }
                        .buttonStyle(.bordered)
                        .tint(.red)
                } else {
                    Button { start() } label: {
                        Image(systemName: "play.fill").frame(minWidth: 40)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(models.isEmpty || (mode == .sustained && selectedModel == nil))
                }
            }
            // sustained target + duration are ALWAYS visible in sustained mode
            // (not hidden behind the gear); the gear holds advanced iters/warmup
            if mode == .sustained { sustainedConfig }
            if showSettings { advancedSettings }
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 10)
        .onChange(of: mode) { _, _ in lightHaptic.impactOccurred(); lightHaptic.prepare() }
    }

    private var sustainedConfig: some View {
        VStack(spacing: 8) {
            HStack {
                Menu {
                    Picker("Model", selection: $selectedModel) {
                        ForEach(models) { m in Text(m.fullName).tag(Optional(m)) }
                    }
                } label: {
                    HStack(spacing: 3) {
                        Text(selectedModel?.shortID ?? "Model").lineLimit(1)
                        Image(systemName: "chevron.up.chevron.down").font(.caption2)
                    }
                }
                .fixedSize()
                Picker("Unit", selection: $selectedCompute) {
                    ForEach(ComputeChoice.allCases) { c in Text(c.rawValue).tag(c) }
                }
                .pickerStyle(.segmented)
            }
            // manual duration selection: a fine stepper plus quick presets
            HStack(spacing: 8) {
                Stepper(value: $sustainedMinutes, in: 1...Double(maxSustained), step: 1) {
                    Text("Duration: \(Int(sustainedMinutes)) min")
                        .font(.caption.monospacedDigit())
                }
                ForEach([3, 5, 10, 20].filter { $0 < maxSustained }, id: \.self) { p in
                    Button("\(p)") { sustainedMinutes = Double(p) }
                        .font(.caption2)
                        .buttonStyle(.bordered)
                        .tint(Int(sustainedMinutes) == p ? .accentColor : .secondary)
                }
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .disabled(running)
        .onChange(of: selectedCompute) { _, _ in
            lightHaptic.impactOccurred(); lightHaptic.prepare()
            if sustainedMinutes > Double(maxSustained) { sustainedMinutes = Double(maxSustained) }
        }
        .onChange(of: sustainedMinutes) { _, _ in lightHaptic.impactOccurred(); lightHaptic.prepare() }
    }

    // CPU inference is slow and hot - cap sustained CPU stress at 3 minutes.
    private var maxSustained: Int { selectedCompute == .cpu ? 3 : 60 }

    private var advancedSettings: some View {
        VStack(spacing: 6) {
            Stepper(value: $iters, in: 20...200, step: 10) {
                Text("Timed iters: \(iters)").font(.caption)
            }
            Stepper(value: $warmup, in: 0...50, step: 5) {
                Text("Warmup: \(warmup)").font(.caption)
            }
            Text(mode == .sustained
                 ? "In Sustained mode these set the cold baseline the throttle is measured against."
                 : "Per model x unit cell in the cold sweep.")
                .font(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .disabled(running)
    }

    // MARK: - cards

    private var progressCard: some View {
        VStack(spacing: 8) {
            if paused {
                HStack(spacing: 4) {
                    Image(systemName: "pause.circle.fill")
                    Text("Paused").font(.caption.bold())
                }
                .foregroundStyle(.orange)
                .frame(maxWidth: .infinity)
            }
            HStack(alignment: .center, spacing: 12) {
                // leading flexible column: the ProgressView fills THIS width only,
                // so it can never run under the thermal dial on the right
                phaseContent
                    .frame(maxWidth: .infinity, alignment: .leading)
                ThermalTach(level: thermalLevel, color: thermalColor, size: 46)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    /// Sustained throttle graph: a rolling 1-minute window while running that
    /// smoothly transitions to the full-run trend when it finishes, plus a
    /// colored thermal timeline bar. Persists after the run until the next start.
    private var sustainedGraphCard: some View {
        VStack(spacing: 6) {
            SparklineView(samples: sparkSamples, baseline: sparkBaseline, color: .blue)
                .frame(height: 64)
            if !sparkThermal.isEmpty {
                ThermalBar(levels: sparkThermal).frame(height: 8)
            }
            HStack(spacing: 6) {
                if let b = sparkBaseline {
                    Text("cold \(fmt(b))").font(.caption2).foregroundStyle(.secondary)
                }
                Text("→ \(running ? "live" : "final") \(fmt(liveMs)) ms")
                    .font(.caption2.monospacedDigit())
                Spacer()
                if runDuration > 0 {
                    Text(durationText(runDuration))
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    @ViewBuilder private var phaseContent: some View {
        switch phase {
        case .loadingModel(let m, let u):
            HStack(spacing: 8) {
                ProgressView()
                Text("Loading \(m) to \(u)").font(.caption)
            }
        case .benchmarking(let m, let u, let d, let t):
            VStack(alignment: .leading, spacing: 4) {
                Text("Benchmarking \(m) @ \(u)").font(.caption)
                ProgressView(value: Double(d), total: Double(max(t, 1)))
                Text("\(d)/\(t)").font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
        case .sustained(let m, let u, let e, let tot):
            VStack(alignment: .leading, spacing: 4) {
                Text("Sustained \(m) @ \(u)").font(.caption)
                ProgressView(value: Double(e), total: Double(max(tot, 1)))
                Text("\(e / 60)m \(e % 60)s / \(tot / 60)m")
                    .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
        default:
            EmptyView()
        }
    }

    private func heroCard(_ r: BenchResult) -> some View {
        HStack(spacing: 16) {
            Gauge(value: min(max(r.coldMedian, 30), 100), in: 30...100) {
                Text("ms")
            } currentValueLabel: {
                Text("\(Int(r.coldMedian.rounded()))")
                    .font(.system(.title3, design: .rounded).bold())
            }
            .gaugeStyle(.accessoryCircular)
            .tint(msColor(r.coldMedian))
            .frame(width: 60, height: 60)
            VStack(alignment: .leading, spacing: 2) {
                Text("FASTEST").font(.caption2).foregroundStyle(.secondary)
                Text(r.fullName).font(.headline).lineLimit(1).minimumScaleFactor(0.6)
                Text("\(r.compute.rawValue)  ·  \(fmt(r.coldMedian)) ms  ·  \(Int(r.fpsEquiv)) FPS")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private func modelCard(_ mid: String) -> some View {
        let rows = results.filter { $0.modelId == mid }.sorted { $0.coldMedian < $1.coldMedian }
        let title = rows.first?.fullName ?? mid
        let fastestUnit = rows.first?.compute
        let expanded = expandedCards.contains(mid)
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title).font(.subheadline.bold()).lineLimit(1).minimumScaleFactor(0.6)
                Spacer()
                if let f = fastestUnit {
                    Text("fastest \(f.rawValue)").font(.caption2)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(.green.opacity(0.2), in: Capsule())
                }
                Image(systemName: expanded ? "chevron.up" : "chevron.down")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
            ForEach(rows) { r in
                DetailBar(icon: unitIcon(r.compute), name: r.compute.rawValue,
                          ms: r.coldMedian, fullScale: 50,
                          color: msColor(r.coldMedian), value: "\(fmt(r.coldMedian))")
                if let s = r.sustainedMedian, let tp = r.throttlePct {
                    DetailBar(icon: "flame", name: "sustained", ms: s, fullScale: 50,
                              color: msColor(s), value: "+\(Int(tp))%")
                }
            }
            if expanded {
                Divider()
                ForEach(rows) { r in
                    Text("\(r.compute.rawValue) · \(Int(r.fpsEquiv)) FPS · p90 \(fmt(r.coldP90))")
                        .font(.caption2).foregroundStyle(.secondary)
                    DetailBar(icon: "aspectratio", name: "preprocess", ms: r.pre)
                    DetailBar(icon: "cpu", name: "inference", ms: r.inf)
                    DetailBar(icon: "rectangle.dashed", name: "decode", ms: r.dec)
                    DetailBar(icon: "timer", name: "end to end", ms: r.pre + r.inf + r.dec,
                              fullScale: 100, greenUntil: 50)
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation(.spring(duration: 0.3)) {
                if expanded { expandedCards.remove(mid) } else { expandedCards.insert(mid) }
            }
        }
    }

    // MARK: - derived

    private var fastest: BenchResult? { results.min { $0.coldMedian < $1.coldMedian } }
    private var modelsWithResults: [String] {
        models.map(\.id).filter { id in results.contains { $0.modelId == id } }
    }
    private func fmt(_ v: Double) -> String { String(format: "%.1f", v) }
    /// Average `arr` into up to `n` buckets to smooth per-inference jitter.
    private func bucketed(_ arr: [Double], _ n: Int) -> [Double] {
        guard !arr.isEmpty else { return [] }
        let bs = max(1, arr.count / n)
        var out: [Double] = []; var i = 0
        while i < arr.count {
            let end = min(i + bs, arr.count)
            out.append(arr[i..<end].reduce(0, +) / Double(end - i)); i = end
        }
        return out
    }
    private func thermalLevelNow() -> Int {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return 0; case .fair: return 1; case .serious: return 2; case .critical: return 3
        @unknown default: return 0
        }
    }
    private func unitIcon(_ c: ComputeChoice) -> String {
        switch c {
        case .ane: return "bolt.fill"
        case .gpu: return "square.stack.3d.up.fill"
        case .cpu: return "cpu"
        }
    }
    private var thermalLevel: Int {
        switch thermal {
        case .nominal: return 0
        case .fair: return 1
        case .serious: return 2
        case .critical: return 3
        @unknown default: return 0
        }
    }
    private var thermalColor: Color {
        switch thermal {
        case .nominal: return .blue
        case .fair: return .green
        case .serious: return .orange
        case .critical: return .red
        @unknown default: return .gray
        }
    }

    // MARK: - run control

    private func start() {
        guard !running else { return }
        control.paused = false; control.cancelled = false
        running = true; paused = false
        runThermalStart = thermalLevel; runThermalPeak = thermalLevel
        medHaptic.impactOccurred(); medHaptic.prepare()
        sparkSamples = []; sparkThermal = []; sparkBaseline = nil; liveMs = 0
        if mode == .sweep {
            results = []; expandedCards = []
            runSweep()
        } else {
            runSustained()
        }
    }

    private func stop() {
        control.cancelled = true
        loopTask?.cancel(); loopTask = nil
        running = false; paused = false
        phase = .idle
    }

    private func togglePause() {
        paused.toggle()
        control.paused = paused
        lightHaptic.impactOccurred(); lightHaptic.prepare()
    }

    /// Suspend the detached loop while paused; returns false if cancelled.
    private func pauseGate() async -> Bool {
        while control.paused && !control.cancelled {
            try? await Task.sleep(nanoseconds: 120_000_000)
        }
        return !control.cancelled
    }

    /// Called on SUCCESSFUL completion (not on cancel): saves the run to history.
    private func finishRun(sustained: Bool, thermalTimeline: [Int]? = nil, durationSec: Int? = nil) {
        if !results.isEmpty {
            let name = sustained
                ? "\(results.first?.shortID ?? "run") · \(selectedCompute.rawValue) · \(Int(sustainedMinutes))min"
                : "Sweep · \(modelsWithResults.count) models"
            history.add(BenchRun(
                name: name, date: Date(),
                mode: sustained ? "Sustained" : "Cold Sweep",
                results: results,
                sparkline: sustained ? sparkSamples : nil,
                thermalTimeline: sustained ? thermalTimeline : nil,
                thermalStart: sustained ? runThermalStart : nil,
                thermalEnd: thermalLevel,
                thermalPeak: sustained ? runThermalPeak : nil,
                durationSec: sustained ? durationSec : nil))
        }
        running = false; paused = false; phase = .done
        notifyHaptic.notificationOccurred(.success); notifyHaptic.prepare()
    }

    /// 640x640 mid-gray test image; inferOnly measures predict() wall time.
    private func testImage() -> CGImage {
        let ctx = CGContext(data: nil, width: 640, height: 640, bitsPerComponent: 8,
                            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.setFillColor(CGColor(red: 0.45, green: 0.45, blue: 0.45, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: 640, height: 640))
        return ctx.makeImage()!
    }

    /// One forward+decode pass -> pre/inf/dec stage ms (empty-scene decode floor).
    private func stageBreakdown(_ det: Detector, _ img: CGImage) -> (Double, Double, Double) {
        var pre = 0.0, inf = 0.0, dec = 0.0
        autoreleasepool {
            let a = CFAbsoluteTimeGetCurrent()
            guard let raw = try? det.forward(img) else { return }
            let b = CFAbsoluteTimeGetCurrent()
            _ = det.decode(raw, conf: 0.25, iou: 0.5)
            let c = CFAbsoluteTimeGetCurrent()
            inf = raw.inferMs
            pre = max((b - a) * 1000 - raw.inferMs, 0)
            dec = (c - b) * 1000
        }
        return (pre, inf, dec)
    }

    private func runSweep() {
        let warmN = warmup, iterN = iters
        loopTask = Task.detached(priority: .userInitiated) {
            let img = testImage()
            let units = ComputeChoice.allCases
            let total = models.count * units.count
            var done = 0
            for m in models {
                for c in units {
                    if control.cancelled { return }
                    if !(await pauseGate()) { return }
                    await MainActor.run { phase = .loadingModel(m.fullName, c.rawValue) }
                    guard let url = try? m.compiledURL(),
                          let det = try? Detector(modelURL: url, compute: c.mode) else {
                        done += 1; continue
                    }
                    await MainActor.run { phase = .benchmarking(m.fullName, c.rawValue, done, total) }
                    for _ in 0..<warmN { autoreleasepool { _ = try? det.inferOnly(img) } }
                    var ms: [Double] = []
                    for _ in 0..<iterN {
                        if control.cancelled { return }
                        autoreleasepool { if let t = try? det.inferOnly(img) { ms.append(t) } }
                    }
                    guard !ms.isEmpty else { done += 1; continue }
                    ms.sort()
                    let (pre, inf, dec) = stageBreakdown(det, img)
                    let r = BenchResult(modelId: m.id, shortID: m.shortID, fullName: m.fullName, compute: c,
                        coldMedian: ms[ms.count / 2],
                        coldP90: ms[min(Int(Double(ms.count) * 0.9), ms.count - 1)],
                        coldMin: ms.first ?? 0, pre: pre, inf: inf, dec: dec)
                    done += 1
                    let d = done
                    await MainActor.run {
                        withAnimation(.spring(duration: 0.3)) { results.append(r) }
                        phase = .benchmarking(m.fullName, c.rawValue, d, total)
                        lightHaptic.impactOccurred(); lightHaptic.prepare()
                    }
                }
            }
            await MainActor.run { finishRun(sustained: false) }
        }
    }

    private func runSustained() {
        guard let m = selectedModel else { running = false; return }
        let c = selectedCompute, warmN = warmup, iterN = iters
        let totalS = Int(sustainedMinutes * 60)
        loopTask = Task.detached(priority: .userInitiated) {
            let img = testImage()
            await MainActor.run { phase = .loadingModel(m.fullName, c.rawValue) }
            guard let url = try? m.compiledURL(),
                          let det = try? Detector(modelURL: url, compute: c.mode) else {
                await MainActor.run { running = false; phase = .idle }; return
            }
            // cold baseline
            for _ in 0..<warmN { autoreleasepool { _ = try? det.inferOnly(img) } }
            var cold: [Double] = []
            for _ in 0..<iterN { autoreleasepool { if let t = try? det.inferOnly(img) { cold.append(t) } } }
            cold.sort()
            let coldMed = cold.isEmpty ? 0 : cold[cold.count / 2]
            await MainActor.run { sparkBaseline = coldMed; sparkSamples = []; sparkThermal = []; runDuration = 0 }
            // sustained loop with a rolling 1-minute display window
            var t0 = CFAbsoluteTimeGetCurrent()
            var all: [Double] = []          // every inference ms
            var atimes: [Double] = []       // its elapsed time
            var therm: [(t: Double, lvl: Int)] = []   // thermal sampled per publish
            var lastPub = 0.0
            var winStart = 0
            while CFAbsoluteTimeGetCurrent() - t0 < Double(totalS) {
                if control.cancelled { return }
                if control.paused {
                    let ps = CFAbsoluteTimeGetCurrent()
                    if !(await pauseGate()) { return }
                    t0 += CFAbsoluteTimeGetCurrent() - ps   // exclude paused time from elapsed
                }
                let st = CFAbsoluteTimeGetCurrent() - t0
                autoreleasepool { if let t = try? det.inferOnly(img) { all.append(t); atimes.append(st) } }
                let now = CFAbsoluteTimeGetCurrent()
                if now - lastPub >= 0.1 {
                    lastPub = now
                    let elapsed = now - t0
                    therm.append((elapsed, thermalLevelNow()))
                    // rolling 1-minute window
                    let cutoff = elapsed - 60
                    while winStart < atimes.count && atimes[winStart] < cutoff { winStart += 1 }
                    let winMs = Array(all[winStart...])
                    let winTherm = therm.filter { $0.t >= cutoff }.map { $0.lvl }
                    let pts = bucketed(winMs, 100)
                    let liveWin = all.suffix(30).sorted()
                    let liveMed = liveWin.isEmpty ? 0 : liveWin[liveWin.count / 2]
                    await MainActor.run {
                        sparkSamples = pts; sparkThermal = winTherm; liveMs = liveMed
                        runDuration = Int(elapsed)
                        phase = .sustained(m.fullName, c.rawValue, Int(elapsed), totalS)
                    }
                }
            }
            let durationSec = Int(CFAbsoluteTimeGetCurrent() - t0)
            let sortedMs = all.sorted()
            let tail = Array(sortedMs.suffix(max(sortedMs.count / 4, 1)))
            let sust = tail.isEmpty ? coldMed : tail[tail.count / 2]
            let p90 = sortedMs.isEmpty ? 0 : sortedMs[min(Int(Double(sortedMs.count) * 0.9), sortedMs.count - 1)]
            let (pre, inf, dec) = stageBreakdown(det, img)
            let fullSpark = bucketed(all, 120)      // smooth full-run trend
            let fullTherm = therm.map { $0.lvl }
            let r = BenchResult(modelId: m.id, shortID: m.shortID, fullName: m.fullName, compute: c,
                coldMedian: coldMed, coldP90: p90, coldMin: sortedMs.first ?? 0,
                pre: pre, inf: inf, dec: dec,
                sustainedMedian: sust, throttlePct: coldMed > 0 ? (sust - coldMed) / coldMed * 100 : 0)
            await MainActor.run {
                // smooth transition from the rolling window to the full-run graph
                withAnimation(.easeInOut(duration: 0.5)) {
                    sparkSamples = fullSpark; sparkThermal = fullTherm; liveMs = sust
                }
                runDuration = durationSec
                results.removeAll { $0.modelId == m.id && $0.compute == c }
                withAnimation(.spring(duration: 0.3)) { results.append(r) }
                finishRun(sustained: true, thermalTimeline: fullTherm, durationSec: durationSec)
            }
        }
    }

    // MARK: - export

    private func buildCSV() -> String {
        let header = "model,compute,cold_median_ms,cold_p90_ms,cold_min_ms,pre_ms,inf_ms,dec_ms,fps_equiv,sustained_ms,throttle_pct"
        let rows = results.map { r in
            String(format: "%@,%@,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%@,%@",
                   r.modelId, r.compute.rawValue, r.coldMedian, r.coldP90, r.coldMin,
                   r.pre, r.inf, r.dec, r.fpsEquiv,
                   r.sustainedMedian.map { String(format: "%.2f", $0) } ?? "",
                   r.throttlePct.map { String(format: "%.1f", $0) } ?? "")
        }
        return ([header] + rows).joined(separator: "\n")
    }

    private func exportCSV() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("benchmark.csv")
        let ok = (try? buildCSV().data(using: .utf8)?.write(to: url)) != nil
        lightHaptic.impactOccurred(); lightHaptic.prepare()
        notifyHaptic.notificationOccurred(ok ? .success : .error); notifyHaptic.prepare()
        exportGen += 1
        let gen = exportGen
        withAnimation(.spring(duration: 0.3)) {
            exportOK = ok
            exportMsg = ok ? "benchmark.csv ready" : "Export failed"
        }
        if ok { share(url) }
        Task {
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            if exportGen == gen { withAnimation(.easeOut(duration: 0.25)) { exportMsg = nil } }
        }
    }

    private func share(_ url: URL) {
        guard let scene = UIApplication.shared.connectedScenes
                .first(where: { $0.activationState == .foregroundActive }) as? UIWindowScene,
              let root = scene.keyWindow?.rootViewController else { return }
        var top = root
        while let presented = top.presentedViewController { top = presented }
        let av = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        av.popoverPresentationController?.sourceView = top.view
        av.popoverPresentationController?.sourceRect = CGRect(
            x: top.view.bounds.midX, y: top.view.bounds.midY, width: 0, height: 0)
        top.present(av, animated: true)
    }
}


/// Horizontal thermal-timeline bar: run-length segments coloured by the thermal
/// level sampled over a run, so the temperature CHANGE reads as parallel zones.
struct ThermalBar: View {
    let levels: [Int]
    var body: some View {
        GeometryReader { g in
            HStack(spacing: 0) {
                ForEach(Array(segments.enumerated()), id: \.offset) { _, seg in
                    thermalLevelColor(seg.level)
                        .frame(width: max(1, g.size.width * seg.frac))
                }
            }
        }
        .clipShape(Capsule())
        .animation(.easeInOut(duration: 0.4), value: levels)
    }
    private var segments: [(level: Int, frac: Double)] {
        guard !levels.isEmpty else { return [] }
        var segs: [(Int, Int)] = []
        for l in levels {
            if let last = segs.last, last.0 == l { segs[segs.count - 1].1 += 1 }
            else { segs.append((l, 1)) }
        }
        let total = Double(levels.count)
        return segs.map { (level: $0.0, frac: Double($0.1) / total) }
    }
}

/// Permanent benchmark history: searchable, sortable, renameable, exportable.
/// Rows are simplified by default and EXPAND on tap to reveal the sustained
/// throttle sparkline + thermal-timeline bar.
struct HistoryView: View {
    @ObservedObject var history: BenchHistory
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var sort: SortKey = .time
    @State private var renaming: BenchRun?
    @State private var newName = ""
    @State private var expanded: Set<UUID> = []
    @State private var fpsRuns: Set<UUID> = []   // cold-sweep cards showing FPS instead of latency
    @Environment(\.editMode) private var editMode
    @State private var selection: Set<UUID> = []
    @State private var exportMsg: String?
    @State private var exportOK = false
    @State private var exportGen = 0

    enum SortKey: String, CaseIterable, Identifiable { case time = "Time", name = "Name"; var id: String { rawValue } }

    private var shown: [BenchRun] {
        var r = history.runs
        if !query.isEmpty {
            let q = query
            r = r.filter { run in
                run.name.localizedCaseInsensitiveContains(q)
                || run.mode.localizedCaseInsensitiveContains(q)
                || run.results.contains { $0.fullName.localizedCaseInsensitiveContains(q) }
            }
        }
        switch sort {
        case .time: r.sort { $0.date > $1.date }
        case .name: r.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        }
        return r
    }
    private func fmt(_ v: Double) -> String { String(format: "%.1f", v) }

    var body: some View {
        NavigationStack {
            ZStack {
                if history.runs.isEmpty {
                    ContentUnavailableView("No saved runs", systemImage: "clock.arrow.circlepath",
                        description: Text("Completed cold sweeps and sustained runs are saved here."))
                } else {
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(shown) { run in card(run) }
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                    }
                    .searchable(text: $query, prompt: "Search runs or models")
                }
                if let msg = exportMsg {
                    ExportToast(success: exportOK, message: msg)
                        .transition(.scale(scale: 0.82).combined(with: .opacity))
                        .onTapGesture { withAnimation(.easeOut(duration: 0.2)) { exportMsg = nil } }
                }
            }
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Menu {
                        Picker("Sort", selection: $sort) {
                            ForEach(SortKey.allCases) { Text($0.rawValue).tag($0) }
                        }
                    } label: {
                        Image(systemName: "arrow.up.arrow.down")
                    }
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    if !history.runs.isEmpty {
                        Button { exportRuns() } label: { Image(systemName: "square.and.arrow.up") }
                        EditButton()
                    }
                    Button("Done") { dismiss() }
                }
            }
            .alert("Rename run", isPresented: Binding(
                get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
                TextField("Name", text: $newName)
                Button("Cancel", role: .cancel) { renaming = nil }
                Button("Save") {
                    if let r = renaming, !newName.isEmpty { history.rename(r.id, newName) }
                    renaming = nil
                }
            }
        }
    }

    private func card(_ run: BenchRun) -> some View {
        let isOpen = expanded.contains(run.id)
        let editing = editMode?.wrappedValue == .active
        let selected = selection.contains(run.id)
        // The header + summary are FIXED; only the detail block below them grows in,
        // pushing the rest of the scroll content down (same feel as the bench cards).
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                if editing {
                    Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(selected ? Color.accentColor : Color.secondary)
                }
                Text(run.name).font(.subheadline.bold()).lineLimit(1)
                Spacer()
                Text(run.mode).font(.caption2)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background((run.mode == "Sustained" ? Color.orange : Color.blue).opacity(0.2),
                                in: Capsule())
                if !editing {
                    Image(systemName: "chevron.down")
                        .font(.caption2).foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isOpen ? 180 : 0))
                }
            }
            HStack(spacing: 6) {
                Text(run.date.formatted(date: .abbreviated, time: .shortened))
                if run.mode == "Sustained", let d = run.durationSec {
                    Text("· \(durationText(d))").monospacedDigit()
                }
            }
            .font(.caption2).foregroundStyle(.secondary)
            if let f = run.fastest {
                Text("fastest \(f.shortID) @ \(f.compute.rawValue) · \(fmt(f.coldMedian)) ms · \(Int(f.fpsEquiv)) FPS")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            if isOpen {
                VStack(alignment: .leading, spacing: 6) {
                    if run.mode == "Sustained", let spark = run.sparkline, spark.count > 1 {
                        SparklineView(samples: spark, color: .blue).frame(height: 44)
                    }
                    if run.mode == "Sustained", let tl = run.thermalTimeline, !tl.isEmpty {
                        ThermalBar(levels: tl).frame(height: 8).padding(.top, 8)
                        if let tp = run.fastest?.throttlePct {
                            Text("throttle +\(Int(tp))%").font(.caption2.monospacedDigit())
                                .foregroundStyle(.orange)
                        }
                    }
                    let scale = max(run.results.map(\.coldMedian).max() ?? 50, 1)
                    let asFPS = fpsRuns.contains(run.id)
                    ForEach(run.results.sorted {
                        $0.modelId == $1.modelId
                            ? $0.coldMedian < $1.coldMedian    // within a model, fastest unit first
                            : $0.modelId < $1.modelId          // group by model, not by global rank
                    }) { r in
                        DetailBar(icon: benchUnitIcon(r.compute),
                                  name: "\(r.shortID) \(r.compute.rawValue)",
                                  ms: r.coldMedian, fullScale: scale,
                                  color: msColor(r.coldMedian),
                                  value: asFPS ? "\(Int(r.fpsEquiv)) fps" : "\(fmt(r.coldMedian)) ms",
                                  valueWidth: 56)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                withAnimation(.easeInOut(duration: 0.15)) {
                                    if asFPS { fpsRuns.remove(run.id) } else { fpsRuns.insert(run.id) }
                                }
                            }
                    }
                }
                .padding(.top, 2)
                .transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(selected ? Color.accentColor : Color.clear, lineWidth: 2))
        .contentShape(Rectangle())
        .onTapGesture {
            if editing {
                if selected { selection.remove(run.id) } else { selection.insert(run.id) }
            } else {
                withAnimation(.spring(response: 0.34, dampingFraction: 0.86)) {
                    if isOpen { expanded.remove(run.id) } else { expanded.insert(run.id) }
                }
            }
        }
        .contextMenu {
            Button { renaming = run; newName = run.name } label: { Label("Rename", systemImage: "pencil") }
            Button { exportRuns([run]) } label: { Label("Export", systemImage: "square.and.arrow.up") }
            Button(role: .destructive) { history.delete([run.id]) } label: { Label("Delete", systemImage: "trash") }
        }
    }

    private func exportRuns(_ explicit: [BenchRun]? = nil) {
        let runs = explicit ?? (selection.isEmpty ? shown : history.runs.filter { selection.contains($0.id) })
        guard !runs.isEmpty else { return }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("bench_history.csv")
        let ok = (try? runsCSV(runs).data(using: .utf8)?.write(to: url)) != nil
        exportGen += 1
        let gen = exportGen
        let tap = UIImpactFeedbackGenerator(style: .light); tap.impactOccurred()
        UINotificationFeedbackGenerator().notificationOccurred(ok ? .success : .error)
        withAnimation(.spring(duration: 0.3)) {
            exportOK = ok
            exportMsg = ok ? "\(runs.count) run\(runs.count == 1 ? "" : "s") exported" : "Export failed"
        }
        if ok { share(url) }
        Task {
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            if exportGen == gen { withAnimation(.easeOut(duration: 0.25)) { exportMsg = nil } }
        }
    }

    private func runsCSV(_ runs: [BenchRun]) -> String {
        var lines = ["run,date,mode,model,compute,cold_median_ms,cold_p90_ms,fps_equiv,sustained_ms,throttle_pct,thermal_start,thermal_end,thermal_peak"]
        for run in runs {
            let name = "\"" + run.name + "\""
            let date = run.date.ISO8601Format()
            let ts = run.thermalStart.map(String.init) ?? ""
            let te = run.thermalEnd.map(String.init) ?? ""
            let tp = run.thermalPeak.map(String.init) ?? ""
            for r in run.results {
                let cm = String(format: "%.2f", r.coldMedian)
                let cp = String(format: "%.2f", r.coldP90)
                let fps = String(format: "%.1f", r.fpsEquiv)
                let sm = r.sustainedMedian.map { String(format: "%.2f", $0) } ?? ""
                let th = r.throttlePct.map { String(format: "%.1f", $0) } ?? ""
                var cols: [String] = [name, date, run.mode, r.modelId, r.compute.rawValue]
                cols.append(contentsOf: [cm, cp, fps, sm, th, ts, te, tp])
                lines.append(cols.joined(separator: ","))
            }
        }
        return lines.joined(separator: "\n")
    }

    private func share(_ url: URL) {
        guard let scene = UIApplication.shared.connectedScenes
                .first(where: { $0.activationState == .foregroundActive }) as? UIWindowScene,
              let root = scene.keyWindow?.rootViewController else { return }
        var top = root
        while let presented = top.presentedViewController { top = presented }
        let av = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        av.popoverPresentationController?.sourceView = top.view
        av.popoverPresentationController?.sourceRect = CGRect(
            x: top.view.bounds.midX, y: top.view.bounds.midY, width: 0, height: 0)
        top.present(av, animated: true)
    }
}
