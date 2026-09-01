// Shared detection-overlay renderer with selectable styles.
// solid / neon / hud mirror the Mac Kit's Annotate styles (same 10-color class
// palette); chip is the classic label-plate look; minimal is a thin monochrome.
import SwiftUI
import YOLOMasterKit

enum BoxStyle: String, CaseIterable, Identifiable {
    case solid, neon, hud, chip, minimal
    var id: String { rawValue }
    var label: String {
        switch self {
        case .solid: return "Solid"
        case .neon: return "Neon"
        case .hud: return "HUD"
        case .chip: return "Chip"
        case .minimal: return "Minimal"
        }
    }
}

/// COCO-80 fallback when a model carries no usable names metadata.
let cocoNames = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

/// Same palette as mac/Sources/YOLOMasterKit/Annotate.swift and the C++ runner.
private let palette: [Color] = [
    Color(red: 0.98, green: 0.26, blue: 0.30), Color(red: 0.20, green: 0.71, blue: 0.98),
    Color(red: 0.16, green: 0.85, blue: 0.52), Color(red: 0.99, green: 0.79, blue: 0.12),
    Color(red: 0.72, green: 0.40, blue: 0.98), Color(red: 0.99, green: 0.55, blue: 0.18),
    Color(red: 0.10, green: 0.83, blue: 0.80), Color(red: 0.98, green: 0.36, blue: 0.66),
    Color(red: 0.55, green: 0.82, blue: 0.28), Color(red: 0.40, green: 0.52, blue: 0.98),
]
private let darkText: Set<Int> = [2, 3, 6, 8]   // light palette entries -> black label text

extension BoxStyle {
    /// Nearest Kit `annotate` style for baked exports (chip/minimal have no
    /// direct Kit equivalent).
    var kitStyle: YOLOMasterKit.BoxStyle {
        switch self {
        case .neon: return .neon
        case .hud, .minimal: return .hud
        default: return .solid
        }
    }
}

enum DetOverlay {
    /// `boxes: false` (segmentation masks-only mode) keeps the per-style labels but
    /// skips the box geometry - the mask carries the shape.
    static func draw(_ ctx: GraphicsContext, _ dets: [Detection],
                     scale: CGFloat, ox: CGFloat, oy: CGFloat,
                     style: BoxStyle = .chip, names: [String] = cocoNames,
                     boxes: Bool = true) {
        for d in dets {
            let r = CGRect(x: d.rect.minX * scale + ox, y: d.rect.minY * scale + oy,
                           width: d.rect.width * scale, height: d.rect.height * scale)
            let ci = ((d.cls % palette.count) + palette.count) % palette.count
            let col = palette[ci]
            let name = d.cls < names.count ? names[d.cls] : "\(d.cls)"
            let title = "\(name) \(Int(d.score * 100))%"
            let corner = min(min(r.width, r.height) * 0.14, 10)

            switch style {
            case .solid:
                if boxes {
                    ctx.stroke(Path(roundedRect: r, cornerRadius: corner),
                               with: .color(col), lineWidth: 2.4)
                }
                chipLabel(ctx, title, at: r, color: col, dark: darkText.contains(ci))

            case .neon:
                if boxes {
                    var glow = ctx
                    glow.addFilter(.shadow(color: col.opacity(0.95), radius: 8))
                    glow.stroke(Path(roundedRect: r, cornerRadius: corner),
                                with: .color(col), lineWidth: 2.6)
                    ctx.stroke(Path(roundedRect: r, cornerRadius: corner),
                               with: .color(col), lineWidth: 1.2)
                }
                var t = ctx
                t.addFilter(.shadow(color: col.opacity(0.9), radius: 5))
                let txt = t.resolve(Text(title).font(.caption2.bold()).foregroundStyle(col))
                t.draw(txt, at: CGPoint(x: r.minX + 2, y: max(r.minY - 9, 6)), anchor: .leading)

            case .hud:
                if boxes {
                    ctx.fill(Path(r), with: .color(col.opacity(0.08)))
                    ctx.stroke(Path(r), with: .color(col.opacity(0.35)), lineWidth: 1)
                    let arm = min(min(r.width, r.height) * 0.28, 26)
                    var p = Path()
                    for (cx, cy, sx, sy): (CGFloat, CGFloat, CGFloat, CGFloat) in
                        [(r.minX, r.minY, 1, 1), (r.maxX, r.minY, -1, 1),
                         (r.minX, r.maxY, 1, -1), (r.maxX, r.maxY, -1, -1)] {
                        p.move(to: CGPoint(x: cx + arm * sx, y: cy))
                        p.addLine(to: CGPoint(x: cx, y: cy))
                        p.addLine(to: CGPoint(x: cx, y: cy + arm * sy))
                    }
                    ctx.stroke(p, with: .color(col), lineWidth: 2.8)
                }
                let txt = ctx.resolve(Text(title.uppercased())
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(col))
                ctx.draw(txt, at: CGPoint(x: r.minX + 3, y: max(r.minY - 9, 6)), anchor: .leading)

            case .chip:
                if boxes {
                    ctx.stroke(Path(roundedRect: r, cornerRadius: 3),
                               with: .color(col), lineWidth: 2)
                }
                chipLabel(ctx, title, at: r, color: col, dark: darkText.contains(ci))

            case .minimal:
                if boxes {
                    ctx.stroke(Path(r), with: .color(.white.opacity(0.9)), lineWidth: 1)
                }
                let txt = ctx.resolve(Text(title)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.white.opacity(0.9)))
                ctx.draw(txt, at: CGPoint(x: r.minX + 2, y: max(r.minY - 8, 5)), anchor: .leading)
            }
        }
    }

    private static func chipLabel(_ ctx: GraphicsContext, _ title: String,
                                  at r: CGRect, color: Color, dark: Bool) {
        let t = ctx.resolve(Text(title).font(.caption2.bold())
            .foregroundStyle(dark ? Color.black : Color.white))
        let sz = t.measure(in: CGSize(width: 320, height: 40))
        let chip = CGRect(x: r.minX, y: max(r.minY - sz.height - 4, 2),
                          width: sz.width + 8, height: sz.height + 3)
        ctx.fill(Path(roundedRect: chip, cornerRadius: 3), with: .color(color))
        ctx.draw(t, at: CGPoint(x: chip.minX + 4, y: chip.midY), anchor: .leading)
    }
}
