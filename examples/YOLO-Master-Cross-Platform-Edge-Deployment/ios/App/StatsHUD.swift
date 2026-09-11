// Shared stats HUD: FPS tachometer (smooth color ramp), per-stage latency bars
// (green -> orange as a stage misbehaves), detection count, thermal state.
import SwiftUI
import YOLOMasterKit

// MARK: - color ramps

private typealias RGB = (r: Double, g: Double, b: Double)
private let cRed: RGB = (0.96, 0.26, 0.21)
private let cOrange: RGB = (1.00, 0.58, 0.00)
private let cGreen: RGB = (0.20, 0.84, 0.29)
private let cPurple: RGB = (0.69, 0.32, 0.87)

private func lerp(_ a: RGB, _ b: RGB, _ t: Double) -> RGB {
    let k = min(max(t, 0), 1)
    return (a.r + (b.r - a.r) * k, a.g + (b.g - a.g) * k, a.b + (b.b - a.b) * k)
}

private func color(_ c: RGB) -> Color { Color(red: c.r, green: c.g, blue: c.b) }

/// Pure band colors - 0-10 red, 10-20 orange, 20-<30 green, and full purple
/// once the dial pegs at 30 (model outrunning the camera). Band changes
/// animate in TIME (fast gradient); values never mix bands.
func fpsColor(_ fps: Double) -> Color {
    switch fps {
    case ..<10: return color(cRed)
    case ..<20: return color(cOrange)
    case ..<29.5: return color(cGreen)
    default: return color(cPurple)
    }
}

/// Stage bars: green while legit, blending to orange past `greenUntil` (default
/// ~20ms, "extraordinarily slow" for any single stage of a 30FPS budget). Rows
/// measuring a whole pipeline pass a higher threshold.
func stageColor(_ ms: Double, greenUntil: Double = 20) -> Color {
    color(lerp(cGreen, cOrange, (ms - greenUntil) / 15))
}

// MARK: - HUD

enum DialMode { case fps, ms }

/// <30ms purple (outrunning the camera budget), 30-50 green, 50-100 orange,
/// >100ms red. Same fast temporal band transition as the FPS dial.
func msColor(_ ms: Double) -> Color {
    switch ms {
    case ..<30: return Color(red: 0.69, green: 0.32, blue: 0.87)
    case ..<50: return Color(red: 0.20, green: 0.84, blue: 0.29)
    case ..<100: return Color(red: 1.00, green: 0.58, blue: 0.00)
    default: return Color(red: 0.96, green: 0.26, blue: 0.21)
    }
}

struct StatsHUD: View {
    let fps: Double            // dial value: FPS (fps mode) or e2e ms (ms mode)
    let pre: Double            // ms
    let inf: Double            // ms
    let dec: Double            // ms
    let dets: Int
    var fpsLabel: String = "FPS"
    var active: Bool = true    // false while no model is loaded/running: N/A + grey
    var extras: [(String, String)] = []   // optional detail rows (stats-card mode)
    var mode: DialMode = .fps
    var fullWidth: Bool = false
    var mask: Double? = nil    // seg mask-compose ms; nil hides the bar (det models)

    @State private var thermal = ProcessInfo.processInfo.thermalState
    @State private var expanded = false   // tap toggles the detailed-stats section
    @State private var rowWidth: CGFloat = 0   // measured main-row width: the expanded
                                               // section is pinned to it so toggling
                                               // never changes the card width

    var body: some View {
        VStack(spacing: 8) {
            mainRow
                .frame(maxWidth: fullWidth ? .infinity : nil)
                .background(GeometryReader { g in
                    Color.clear
                        .onAppear { if !expanded { rowWidth = g.size.width } }
                        .onChange(of: g.size.width) { _, w in
                            if !expanded { rowWidth = w }   // collapsed layout owns the width
                        }
                })
            if expanded {
                Group {
                    Divider()
                    details
                    if !extras.isEmpty {
                        Divider()
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())],
                                  alignment: .leading, spacing: 3) {
                            ForEach(extras.indices, id: \.self) { i in
                                HStack(spacing: 4) {
                                    Text(extras[i].0).foregroundStyle(.secondary)
                                    Text(extras[i].1).monospacedDigit()
                                }
                                .font(.caption2)
                            }
                        }
                    }
                }
                .frame(width: fullWidth || rowWidth <= 0 ? nil : rowWidth)
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .contentShape(Rectangle())
        .onTapGesture { withAnimation(.spring(duration: 0.3)) { expanded.toggle() } }
        .onReceive(NotificationCenter.default.publisher(
            for: ProcessInfo.thermalStateDidChangeNotification)) { _ in
            thermal = ProcessInfo.processInfo.thermalState
        }
    }

    /// Expanded section: every stage as [symbol] metric ---- bar, plus derived
    /// postprocess and end-to-end rows on wider scales.
    private var details: some View {
        VStack(alignment: .leading, spacing: 5) {
            DetailBar(icon: "aspectratio", name: "preprocess", ms: pre)
            DetailBar(icon: "cpu", name: "inference", ms: inf)
            DetailBar(icon: "rectangle.dashed", name: "decode", ms: dec)
            if let mask {
                DetailBar(icon: "person.and.background.dotted", name: "mask", ms: mask)
            }
            DetailBar(icon: "gearshape.2", name: "postprocess", ms: dec + (mask ?? 0))
            DetailBar(icon: "timer", name: "end to end",
                      ms: pre + inf + dec + (mask ?? 0), fullScale: 100, greenUntil: 50)
        }
    }

    private var mainRow: some View {
        HStack(spacing: 14) {
            // ms mode: anything under 30ms rests at the LEFTMOST position;
            // 30-100ms sweeps left -> right; beyond 100 pegs right.
            Gauge(value: active ? (mode == .ms ? min(max(fps, 30), 100)
                                               : min(max(fps, 0), 30)) : (mode == .ms ? 30 : 0),
                  in: mode == .ms ? 30...100 : 0...30) {
                Text(fpsLabel)
            } currentValueLabel: {
                Text(active ? "\(Int(fps.rounded()))" : "N/A")
                    .font(.system(active ? .title3 : .footnote, design: .rounded).bold())
                    .foregroundStyle(active ? .primary : .secondary)
            }
            .gaugeStyle(.accessoryCircular)
            .tint(active ? dialColor : Color.gray.opacity(0.5))
            .animation(.easeInOut(duration: 0.25), value: active ? dialColor : .gray)
            .frame(width: 56, height: 56)

            ThermalTach(level: Int(thermalLevel), color: thermalColor)

            // compact stage bars only while collapsed - the expanded view has
            // the same stages as named detail rows, no duplication
            if !expanded {
                VStack(alignment: .leading, spacing: 5) {
                    StageBar(icon: "aspectratio", ms: pre)                       // preprocess
                    StageBar(icon: "cpu", ms: inf)                               // inference
                    StageBar(icon: "rectangle.dashed", ms: dec)                  // decode
                    if let mask {
                        StageBar(icon: "person.and.background.dotted", ms: mask) // mask compose
                    }
                }
            }

            VStack(spacing: 0) {
                Text("\(dets)")
                    .font(.system(.title3, design: .rounded).bold())
                    .foregroundStyle(.green)
                Text("dets").font(.caption2).foregroundStyle(.secondary)
            }
            .frame(width: 40)
        }
    }

    private var dialColor: Color {
        mode == .fps ? fpsColor(fps) : msColor(fps)
    }

    private var thermalLevel: Double {
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

}

struct StageBar: View {
    let icon: String   // SF Symbol naming the stage
    let ms: Double
    var color: Color? = nil   // override the stage-color ramp (bench uses msColor bands)
    private let fullScale = 50.0   // bar saturates at 50ms

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 24, alignment: .center)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    Capsule().fill(color ?? stageColor(ms))
                        .frame(width: max(3, min(ms / fullScale, 1) * g.size.width))
                }
            }
            .frame(width: 90, height: 6)
            .animation(.easeOut(duration: 0.15), value: ms)
            Text(String(format: "%4.1f", ms))
                .font(.caption2.monospacedDigit())
                .frame(width: 36, alignment: .trailing)
        }
    }
}

/// Discrete thermal tachometer: a 270-degree dial split into FOUR separate arc
/// segments (one per thermal state), filled up to the current level in the
/// state's color - no needle, no dot. A large centered thermometer symbol,
/// nudged down like the FPS dial's value.
struct ThermalTach: View {
    let level: Int     // 0 nominal ... 3 critical
    let color: Color
    var size: CGFloat = 56   // overall diameter; scales the ring, stroke, and glyph

    private let span = 0.75          // 270 degrees, same as the accessory gauges
    // round line caps extend half the line width past each trim end, so the
    // nominal gap must be wide enough to stay visible after both caps bite in
    private let gap = 0.055
    private var segLen: Double { (span - 3 * gap) / 4 }

    var body: some View {
        ZStack {
            ForEach(0..<4, id: \.self) { i in
                Circle()
                    .trim(from: Double(i) * (segLen + gap),
                          to: Double(i) * (segLen + gap) + segLen)
                    .stroke(i <= level ? AnyShapeStyle(color) : AnyShapeStyle(.quaternary),
                            style: StrokeStyle(lineWidth: size * 0.107, lineCap: .round))
                    .rotationEffect(.degrees(135))   // start bottom-left, like the FPS dial
                    .frame(width: size - 2, height: size - 2)
            }
            Image(systemName: "thermometer.medium")
                .font(.system(size: size * 0.43, weight: .medium))
                .foregroundStyle(color)
                .offset(y: size * 0.054)
        }
        .frame(width: size, height: size)
        .animation(.easeOut(duration: 0.3), value: level)
    }
}

/// Expanded-HUD row: [symbol] metric name, then a full-width bar + ms value.
struct DetailBar: View {
    let icon: String
    let name: String
    let ms: Double
    var fullScale = 50.0
    var greenUntil = 20.0   // whole-pipeline rows warn later than single stages
    var color: Color? = nil // override the ramp (bench latency uses msColor bands)
    var value: String? = nil // override the trailing readout text (default: ms)
    var barWidth: CGFloat? = nil // cap the bar length (history comparison uses a short bar); nil = flexible
    var valueWidth: CGFloat = 40 // width of the trailing readout column

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 20, alignment: .center)
            Text(name)
                .font(.caption2)
                .frame(width: 78, alignment: .leading)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    Capsule().fill(color ?? stageColor(ms, greenUntil: greenUntil))
                        .frame(width: max(3, min(ms / fullScale, 1) * g.size.width))
                }
            }
            .frame(width: barWidth, height: 6)
            .animation(.easeOut(duration: 0.15), value: ms)
            if barWidth != nil { Spacer(minLength: 6) }
            Text(value ?? String(format: "%5.1f", ms))
                .font(.caption2.monospacedDigit())
                .lineLimit(1)
                .frame(width: valueWidth, alignment: .trailing)
        }
    }
}

/// Reusable Canvas line-plot for the bench sustained/throttle view: draws
/// `samples` normalized to the view bounds, with an optional dashed `baseline`
/// (the cold median) so you can see how far the phone has throttled.
struct SparklineView: View {
    let samples: [Double]
    var baseline: Double? = nil
    var color: Color = .orange

    var body: some View {
        Canvas { ctx, size in
            guard samples.count > 1 else { return }
            let lo = min(samples.min() ?? 0, baseline ?? .greatestFiniteMagnitude)
            let hi = max(samples.max() ?? 1, baseline ?? 0)
            let span = max(hi - lo, 1e-6)
            // inset vertically so the peak/trough (and the 2pt stroke) never clip
            // against the canvas edges
            let pad: CGFloat = 4
            func y(_ v: Double) -> CGFloat {
                let h = max(size.height - 2 * pad, 1)
                return size.height - pad - CGFloat((v - lo) / span) * h
            }
            let dx = size.width / CGFloat(samples.count - 1)
            if let b = baseline {
                var base = Path()
                base.move(to: CGPoint(x: 0, y: y(b)))
                base.addLine(to: CGPoint(x: size.width, y: y(b)))
                ctx.stroke(base, with: .color(.secondary.opacity(0.5)),
                           style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            }
            var line = Path()
            for (i, v) in samples.enumerated() {
                let p = CGPoint(x: CGFloat(i) * dx, y: y(v))
                if i == 0 { line.move(to: p) } else { line.addLine(to: p) }
            }
            ctx.stroke(line, with: .color(color),
                       style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
        }
        .animation(.easeOut(duration: 0.2), value: samples.count)
    }
}

/// Export-result popup: a stroke-animated tick (or cross) drawing itself in,
/// then the card auto-dismisses. Tap dismisses early. Shared by Photo + Bench.
struct ExportToast: View {
    let success: Bool
    let message: String
    @State private var drawn = false

    var body: some View {
        VStack(spacing: 10) {
            ZStack {
                Circle()
                    .trim(from: 0, to: drawn ? 1 : 0)
                    .stroke(success ? Color.green : Color.red,
                            style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .frame(width: 46, height: 46)
                Group {
                    if success {
                        TickShape()
                            .trim(from: 0, to: drawn ? 1 : 0)
                            .stroke(Color.green,
                                    style: StrokeStyle(lineWidth: 3.5, lineCap: .round,
                                                       lineJoin: .round))
                    } else {
                        CrossShape()
                            .trim(from: 0, to: drawn ? 1 : 0)
                            .stroke(Color.red,
                                    style: StrokeStyle(lineWidth: 3.5, lineCap: .round))
                    }
                }
                .frame(width: 20, height: 20)
            }
            Text(message).font(.caption)
        }
        .padding(20)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .onAppear {
            withAnimation(.easeOut(duration: 0.45).delay(0.05)) { drawn = true }
        }
    }
}

struct TickShape: Shape {
    func path(in r: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: r.minX, y: r.midY + r.height * 0.08))
        p.addLine(to: CGPoint(x: r.minX + r.width * 0.36, y: r.maxY - r.height * 0.08))
        p.addLine(to: CGPoint(x: r.maxX, y: r.minY + r.height * 0.1))
        return p
    }
}

struct CrossShape: Shape {
    func path(in r: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: r.minX, y: r.minY))
        p.addLine(to: CGPoint(x: r.maxX, y: r.maxY))
        p.move(to: CGPoint(x: r.maxX, y: r.minY))
        p.addLine(to: CGPoint(x: r.minX, y: r.maxY))
        return p
    }
}

/// conf / IoU tuning sliders + box style - the detection knobs, one panel.
struct TuningPanel: View {
    @Binding var conf: Double
    @Binding var iou: Double
    var style: Binding<BoxStyle>? = nil
    var hudVisible: Binding<Bool>? = nil
    var segOverlay: Binding<SegOverlay>? = nil   // non-nil only for seg models
    var body: some View {
        VStack(spacing: 6) {
            if let hudVisible {
                Toggle("Show stats HUD", isOn: hudVisible)
                    .font(.caption)
            }
            if let style {
                Picker("Style", selection: style) {
                    ForEach(BoxStyle.allCases) { s in
                        Text(s.label).lineLimit(1).fixedSize().tag(s)
                    }
                }
                .pickerStyle(.segmented)
            }
            if let segOverlay {
                Picker("Overlay", selection: segOverlay) {
                    ForEach(SegOverlay.allCases, id: \.self) { o in
                        Text(o.rawValue.capitalized).lineLimit(1).fixedSize().tag(o)
                    }
                }
                .pickerStyle(.segmented)
            }
            HStack {
                Text("conf").font(.caption).frame(width: 34, alignment: .leading)
                Slider(value: $conf, in: 0.05...0.9)
                Text(String(format: "%.2f", conf)).font(.caption.monospacedDigit()).frame(width: 34)
            }
            HStack {
                Text("IoU").font(.caption).frame(width: 34, alignment: .leading)
                Slider(value: $iou, in: 0.1...0.9)
                Text(String(format: "%.2f", iou)).font(.caption.monospacedDigit()).frame(width: 34)
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

/// Shared mutable tuning state readable from the detection loop without
/// main-actor hops (word-sized reads; eventual consistency is fine here).
final class Tuning: ObservableObject {
    @Published var conf: Double = 0.25
    @Published var iou: Double = 0.5
    @Published var segOverlay: SegOverlay = .both   // masks / boxes / both (mac default)
}
