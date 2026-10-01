// `--save-txt` dumps in the C++ CLI's format: one file per source, one line per detection,
// `class conf x1 y1 x2 y2 [track_id]` in original-image pixels, every number printed the way
// `std::ostream << float` prints it (`%g`, six significant digits). Byte-compatible with
// cpp/src/main.cpp, so the dumps score with scripts/eval_map*.py, yolomaster_score and
// scripts/server/parity_txt.py exactly like a Linux dump.
import Foundation
import CoreGraphics

public enum TxtDump {
    /// The five numbers of a detection as the txt line carries them, computed in Float the way the
    /// C++ writer does (x2 = x + width in float arithmetic), before formatting.
    public static func values(_ d: Detection) -> (cls: Int, conf: Float, x1: Float, y1: Float, x2: Float, y2: Float) {
        let x = Float(d.rect.minX), y = Float(d.rect.minY), w = Float(d.rect.width), h = Float(d.rect.height)
        return (d.cls, d.score, x, y, x + w, y + h)
    }
    /// `%g` of a float promoted to double: six significant digits, trailing zeros trimmed.
    public static func g(_ v: Float) -> String { String(format: "%g", Double(v)) }

    public static func line(_ d: Detection) -> String {
        let v = values(d)
        var s = "\(v.cls) \(g(v.conf)) \(g(v.x1)) \(g(v.y1)) \(g(v.x2)) \(g(v.y2))"
        if let id = d.trackId { s += " \(id)" }
        return s
    }

    /// The scorer's view of a detection: the txt-rounded values (what the dump reads back as), so the
    /// in-process mAP equals scoring the file.
    public static func predBox(_ d: Detection) -> PredBox {
        let v = values(d)
        return PredBox(x1: YMCore.round6(Double(v.x1)), y1: YMCore.round6(Double(v.y1)),
                       x2: YMCore.round6(Double(v.x2)), y2: YMCore.round6(Double(v.y2)),
                       conf: YMCore.round6(Double(v.conf)), cls: v.cls)
    }
}

/// One dump directory; stems are made unique the way every per-file writer does (`uniqueStem`).
public final class TxtDumpWriter {
    public let directory: URL
    private var used = Set<String>()
    public private(set) var files = 0

    public init(directory: URL) throws {
        self.directory = directory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
    /// Writes `<stem>.txt` (empty file for an image without detections; a verified negative).
    @discardableResult
    public func write(stem: String, _ dets: [Detection]) -> URL {
        let name = uniqueStem(&used, stem) + ".txt"
        let url = directory.appendingPathComponent(name)
        let body = dets.map(TxtDump.line).joined(separator: "\n") + (dets.isEmpty ? "" : "\n")
        try? body.write(to: url, atomically: true, encoding: .utf8)
        files += 1
        return url
    }
    /// Video frames: `<videoStem>_%06d`, the C++ CLI's frame-unique naming.
    public static func frameStem(_ videoStem: String, _ index: Int) -> String { videoStem + String(format: "_%06d", index) }
}
