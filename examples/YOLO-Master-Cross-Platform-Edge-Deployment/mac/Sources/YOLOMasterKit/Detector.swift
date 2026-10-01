// YOLOMasterKit - shared Core ML inference backend for YOLO-Master detectors.
//
// Extracted VERBATIM from the CLI runner (Sources/YOLOMasterCoreML/main.swift) so the
// command-line tool and the SwiftUI app run the exact same letterbox → Core ML →
// decode([1,4+nc,anchors]) → per-class NMS path. One backend, two frontends.
import Foundation
import CoreML
import CoreGraphics
import CoreVideo
import Accelerate   // batched mask math: SGEMM (AMX) + vectorized sigmoid

/// IEEE half bits (UInt16) -> Float32, done by hand. `Float16` is entirely UNAVAILABLE in macOS on
/// x86_64 (arm64-only type), so it can't be named at all - half-precision Core ML tensors are read
/// as raw UInt16 straight from `dataPointer` (`load(fromByteOffset:as:)`) and converted here. Keeps
/// the universal (arm64 + x86_64) build compiling with identical numerics on both slices.
@inline(__always) func halfToFloat(_ h: UInt16) -> Float32 {
    let sign = UInt32(h & 0x8000) << 16
    let exp  = UInt32(h & 0x7C00) >> 10
    let mant = UInt32(h & 0x03FF)
    let bits: UInt32
    if exp == 0 {
        if mant == 0 { bits = sign }                              // +/-0
        else {                                                    // subnormal -> normalized
            var e: UInt32 = 127 - 15 + 1, m = mant
            while (m & 0x0400) == 0 { m <<= 1; e -= 1 }
            bits = sign | (e << 23) | ((m & 0x03FF) << 13)
        }
    } else if exp == 0x1F {
        bits = sign | 0x7F80_0000 | (mant << 13)                  // +/-inf / NaN
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13)   // normal
    }
    return Float32(bitPattern: bits)
}

/// A single detection in ORIGINAL-image pixel coordinates (top-left origin).
/// `maskCoeffs` is non-empty only for segmentation models (nm mask-prototype coefficients).
public struct Detection: Sendable {
    public let cls: Int
    public let score: Float
    public let rect: CGRect
    public let maskCoeffs: [Float]
    /// Track id when the detection came out of a `Tracker` (video / camera with tracking on); nil otherwise.
    public let trackId: Int?
    public init(cls: Int, score: Float, rect: CGRect, maskCoeffs: [Float] = [], trackId: Int? = nil) {
        self.cls = cls; self.score = score; self.rect = rect; self.maskCoeffs = maskCoeffs; self.trackId = trackId
    }
    /// The same detection tagged with a track id.
    public func withTrackId(_ id: Int?) -> Detection { Detection(cls: cls, score: score, rect: rect, maskCoeffs: maskCoeffs, trackId: id) }
}

/// A rendered instance mask (segmentation): a proto-resolution tinted RGBA image, the unit
/// sub-rect of the proto grid mapping to the full original image, and the box to clip it to.
public struct MaskBitmap: @unchecked Sendable {
    public let image: CGImage
    public let protoCrop: CGRect   // unit coords of the proto grid (top-left origin) = full original image
    public let clip: CGRect        // detection box, original-image pixels (top-left origin)
}

/// Core ML compute unit selection. Default cpuAndGPU: the ANE can crash on this
/// fragmented MoE+attention graph.
///
/// `cpuAndNeuralEngine` is the Neural Engine measurement: every op the ANE supports runs there and
/// the rest falls back to the CPU, never the GPU. `all` hands the whole decision to Core ML, which
/// partitions the graph across all three units and can end up slower than the GPU alone on a
/// graph with many ANE-unsupported segments (each boundary is a sync and a copy).
public enum ComputeMode: String, CaseIterable, Sendable {
    case cpuAndGPU, cpuAndNeuralEngine, all, cpu
    public var mlUnits: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpu: return .cpuOnly
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        }
    }
    /// Human-readable label for the UI.
    public var label: String {
        switch self {
        case .cpuAndGPU: return "CPU + GPU"
        case .cpuAndNeuralEngine: return "CPU + Neural Engine"
        case .all: return "Auto (ANE + GPU + CPU)"
        case .cpu: return "CPU only"
        }
    }
    public init(_ s: String) {
        switch s.lowercased() {
        case "all": self = .all
        case "ane", "neuralengine", "cpuandneuralengine", "cpuandane": self = .cpuAndNeuralEngine
        case "cpu", "cpuonly": self = .cpu
        default: self = .cpuAndGPU
        }
    }
}

public enum DetectorError: Error { case inputBuildFailed, badOutput }

/// Which NMS variant post-processing uses. `.standard` is the classic greedy suppression.
/// `.clusterWeighted` (CW-NMS, from upstream YOLO-Master) runs the same greedy selection and
/// then REFINES each survivor's coordinates as the score-and-proximity-weighted average of its
/// same-class cluster in the pre-NMS candidate pool - trading a little post-processing time for
/// better localization. Survivor count, scores and classes are identical to `.standard`.
public enum NMSMode: String, CaseIterable, Sendable {
    case standard, clusterWeighted
    public var label: String { self == .standard ? "Standard" : "Cluster-Weighted" }
}

/// IoU of two rects (used by NMS).
func rectIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
    let i = a.intersection(b); if i.isNull { return 0 }
    let ia = i.width * i.height
    return ia / (a.width * a.height + b.width * b.height - ia + 1e-6)
}

/// Loads a `.mlpackage`/`.mlmodelc` YOLO-Master detector and runs inference.
/// `imgsz`, class count and names are read from the model (works for any exported
/// ultralytics detect model), so preprocessing always matches the checkpoint.
public final class Detector {
    public let imgsz: Int
    public let nc: Int
    public let classNames: [String]
    public let computeMode: ComputeMode

    public let isSegment: Bool
    public let nm: Int   // mask-coeff count (segmentation); 0 for detection
    public let end2end: Bool   // NMS-free head: output [1, max_det, 6] instead of [1, 4+nc, anchors]
    public let modelURL: URL
    /// The exporter's creator-defined metadata (names, imgsz, task, precision when stamped, ...).
    public let metadata: [String: String]

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let protoName: String

    /// `forceCompute: true` keeps the requested compute units even for end2end models,
    /// which otherwise auto-downgrade to CPU (see below).
    public init(modelURL: URL, compute: ComputeMode = .cpuAndGPU, forceCompute: Bool = false) throws {
        let compiledURL = modelURL.pathExtension.lowercased() == "mlmodelc"
            ? modelURL : try MLModel.compileModel(at: modelURL)
        func load(_ mode: ComputeMode) throws -> MLModel {
            let cfg = MLModelConfiguration(); cfg.computeUnits = mode.mlUnits
            return try MLModel(contentsOf: compiledURL, configuration: cfg)
        }
        var loaded = try load(compute)
        var effectiveCompute = compute
        let md = loaded.modelDescription

        let inName = md.inputDescriptionsByName.keys.sorted().first ?? "images"
        let meta = md.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        let metaNames = meta["names"]?.split(separator: ",").map(String.init)
            ?? ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
        let outName = meta["output"] ?? md.outputDescriptionsByName.keys.sorted().first ?? "output0"
        // end2end (NMS-free) detect head: output is [1, max_det, 6] rows of
        // [x1,y1,x2,y2,score,cls] in letterboxed-input pixels. The exporter stamps an
        // `end2end` metadata key (authoritative); the shape heuristic mirrors the C++
        // runtime's looks_end2end (dim2 == 6 && dim1 >= 32) for models without it.
        let e2eFromShape: Bool = {
            if let sh = md.outputDescriptionsByName[outName]?.multiArrayConstraint?.shape,
               sh.count == 3, sh[2].intValue == 6, sh[1].intValue >= 32 { return true }
            return false
        }()
        let e2eResolved = meta["end2end"].map { $0 == "true" } ?? e2eFromShape
        // class count from the output shape [1, 4+nc, anchors] (authoritative for ANY model);
        // fall back to the metadata names count. An end2end tensor's dim1 is max_det, not
        // 4+nc, so it must use the names count.
        let ncFromShape: Int? = {
            if let sh = md.outputDescriptionsByName[outName]?.multiArrayConstraint?.shape,
               sh.count >= 2, sh[1].intValue > 4 { return sh[1].intValue - 4 }
            return nil
        }()
        let ncResolved = e2eResolved ? metaNames.count : (ncFromShape ?? metaNames.count)
        // Input resolution is FIXED at export time - read it from the model ([1,3,H,W]).
        let szResolved: Int = {
            if let shape = md.inputDescriptionsByName[inName]?.multiArrayConstraint?.shape,
               shape.count >= 4, shape[2].intValue > 0 { return shape[2].intValue }
            if let s = meta["imgsz"], let v = Int(s), v > 0 { return v }
            return 640
        }()

        // segmentation: detection tensor is [1, 4+nc+nm, anchors] -> nc from names, plus a proto output
        let segTask = (meta["task"] ?? "detect") == "segment"
        let ncFinal = segTask ? metaNames.count : ncResolved

        // MPSGraph aborts compiling end2end mixture graphs on GPU (a hard assertion at
        // the first predict, not catchable in-process), and CPU is orders of magnitude
        // faster on these fragmented graphs anyway (measured: 22ms CPU vs 7.5s GPU-path
        // on the anchors sibling). Auto-downgrade unless the caller insists.
        if e2eResolved && !segTask && effectiveCompute != .cpu && !forceCompute {
            effectiveCompute = .cpu
            loaded = try load(.cpu)
        }
        self.computeMode = effectiveCompute
        self.modelURL = modelURL
        self.metadata = meta
        self.model = loaded

        self.inputName = inName
        self.outputName = outName
        // proto/nm: metadata keys when the exporter stamped them; else recover from
        // shapes (proto = the rank-4 output that isn't the det tensor, nm = its dim1).
        // Without this, a seg model missing `nm` decodes with empty coeffs and
        // renders zero masks with no error.
        let protoResolved: String = {
            if let p = meta["proto"], !p.isEmpty { return p }
            guard segTask else { return "" }
            return md.outputDescriptionsByName.first {
                $0.key != outName && ($0.value.multiArrayConstraint?.shape.count ?? 0) == 4
            }?.key ?? ""
        }()
        let nmResolved: Int = {
            guard segTask else { return 0 }
            if let v = Int(meta["nm"] ?? ""), v > 0 { return v }
            if let sh = md.outputDescriptionsByName[protoResolved]?.multiArrayConstraint?.shape,
               sh.count == 4 { return sh[1].intValue }
            if let sh = md.outputDescriptionsByName[outName]?.multiArrayConstraint?.shape,
               sh.count == 3, sh[1].intValue > 4 + metaNames.count {
                return sh[1].intValue - 4 - metaNames.count
            }
            return 0
        }()
        self.protoName = protoResolved
        self.isSegment = segTask
        self.nm = nmResolved
        self.nc = ncFinal
        self.classNames = metaNames.count == ncFinal ? metaNames : (0..<ncFinal).map { "class\($0)" }
        self.imgsz = szResolved
        self.end2end = e2eResolved && !segTask
    }

    /// Human-readable one-line model summary (parity with the CLI `[model]` banner).
    public var summary: String {
        "input=\(inputName) [\(imgsz)x\(imgsz)] output=\(outputName) classes=\(nc) compute=\(computeMode.rawValue)"
            + (end2end ? " layout=end2end" : "")
    }

    // ---------- preprocess ----------
    /// How the source image is fit into the model's fixed imgsz×imgsz input. This is a PREPROCESSING
    /// choice (changes the forward-pass input), not a tuning param - switching it requires re-inference.
    /// `.letterbox`: aspect-preserving fit + gray padding (YOLO default). `.stretch`: force-resize the
    /// whole image to imgsz×imgsz (no padding; the size the model was trained on), distorting aspect.
    public enum PreprocessMode: String, CaseIterable, Sendable { case letterbox, stretch }
    public var preprocess: PreprocessMode = .letterbox
    /// Where the letterbox + tensor build runs: `.cpu` (CGContext + vDSP, the 1.1.x path) or `.gpu`
    /// (the Metal kernel in Preproc.swift, integer pads, bilinear). Falls back to the CPU path when
    /// Metal is unavailable. Default cpu so existing hosts (the iOS app) are unchanged; the macOS CLI
    /// and app opt in.
    public var preprocDevice: PreprocDevice = .cpu
    private lazy var metal: MetalPreprocessor? = MetalPreprocessor()
    private var gpuActive: Bool { preprocDevice == .gpu && metal != nil }
    /// The preprocessing path actually in use (a `.gpu` request without Metal reports `.cpu`).
    public var effectivePreprocDevice: PreprocDevice { gpuActive ? .gpu : .cpu }

    private struct LB { let px: [UInt8]; let scaleX: CGFloat; let scaleY: CGFloat; let padX: CGFloat; let padY: CGFloat }

    private func letterbox(_ image: CGImage) -> LB {
        let size = imgsz
        let w = image.width, h = image.height
        let scaleX: CGFloat, scaleY: CGFloat, nw: Int, nh: Int, padX: CGFloat, padY: CGFloat
        switch preprocess {
        case .letterbox:                                    // aspect-preserving fit + centered padding
            let s = min(CGFloat(size) / CGFloat(w), CGFloat(size) / CGFloat(h))
            scaleX = s; scaleY = s
            nw = Int((CGFloat(w) * s).rounded()); nh = Int((CGFloat(h) * s).rounded())
            padX = CGFloat(size - nw) / 2; padY = CGFloat(size - nh) / 2
        case .stretch:                                      // force-resize to the full imgsz square
            scaleX = CGFloat(size) / CGFloat(w); scaleY = CGFloat(size) / CGFloat(h)
            nw = size; nh = size; padX = 0; padY = 0
        }
        var px = [UInt8](repeating: 114, count: size * size * 4)
        px.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: size, height: size, bitsPerComponent: 8,
                                      bytesPerRow: size * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: padX, y: padY, width: CGFloat(nw), height: CGFloat(nh)))  // no flip -> top-down
        }
        return LB(px: px, scaleX: scaleX, scaleY: scaleY, padX: padX, padY: padY)
    }

    private func fillInput(_ raster: [UInt8]) -> MLDictionaryFeatureProvider? {
        guard let arr = try? MLMultiArray(shape: [1, 3, NSNumber(value: imgsz), NSNumber(value: imgsz)], dataType: .float32)
        else { return nil }
        let p = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
        let plane = imgsz * imgsz
        // vDSP: interleaved RGBX u8 -> planar float RGB, then one /255 over all
        // three planes. Replaces a ~400k-iteration scalar loop (mobile hot path).
        raster.withUnsafeBufferPointer { rb in
            guard let src = rb.baseAddress else { return }
            for ch in 0..<3 {
                vDSP_vfltu8(src + ch, 4, p + ch * plane, 1, vDSP_Length(plane))
            }
        }
        var inv255 = Float32(1.0 / 255.0)
        vDSP_vsmul(p, 1, &inv255, p, 1, vDSP_Length(3 * plane))
        return try? MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
    }

    // ---------- decode + NMS (split so forward is cached once, tuning stays cheap) ----------
    /// Rows the decoder will read down dim 1 (anchor layout) or across dim 2 (end2end rows).
    /// `nc` and `nm` come from the package's own metadata, so a package whose names list or `nm`
    /// stamp exceeds its head would otherwise send the raw-pointer scan past the tensor.
    private func outputFits(_ y: MLMultiArray) -> Bool {
        guard y.shape.count == 3 else { return false }
        if end2end { return y.shape[2].intValue >= 6 }
        return y.shape[1].intValue >= 4 + nc + nm
    }
    private func validatedOutput(_ out: MLFeatureProvider) throws -> MLMultiArray {
        guard let y = out.featureValue(for: outputName)?.multiArrayValue, outputFits(y) else { throw DetectorError.badOutput }
        return y
    }

    /// All boxes above `confFloor` (NO NMS), ORIGINAL-image pixels, sorted by score desc.
    /// Cache this once per image after `forward`; then re-run `nms(_:conf:iou:)` for cheap tuning.
    public func candidates(_ raw: RawOutput, confFloor: Float = 0.05) -> [Detection] {
        let y = raw.y
        guard outputFits(y) else { return [] }   // a mask-only raw ([1]) or a foreign RawOutput: nothing to decode
        let na = y.shape[2].intValue
        let s1 = y.strides[1].intValue, s2 = y.strides[2].intValue
        let scaleX = raw.scaleX, scaleY = raw.scaleY, padX = raw.padX, padY = raw.padY
        let origW = raw.origW, origH = raw.origH
        var dets: [Detection] = []
        // end2end layout [1, max_det, 6]: rows [x1,y1,x2,y2,score,cls] in letterboxed px,
        // already one-to-one (no NMS needed downstream, though it stays harmless).
        // Same accessor convention as decodeAnchors: at(dim1Index, dim2Index).
        func decodeEndToEnd(_ at: (Int, Int) -> Float32) {
            let n = y.shape[1].intValue
            for r in 0..<n {
                let s = at(r, 4)
                if s <= confFloor { continue }
                let c = Int(at(r, 5))
                if c < 0 || c >= nc { continue }
                var x1 = (CGFloat(at(r, 0)) - padX) / scaleX, y1 = (CGFloat(at(r, 1)) - padY) / scaleY
                var x2 = (CGFloat(at(r, 2)) - padX) / scaleX, y2 = (CGFloat(at(r, 3)) - padY) / scaleY
                x1 = max(0, min(CGFloat(origW), x1)); x2 = max(0, min(CGFloat(origW), x2))
                y1 = max(0, min(CGFloat(origH), y1)); y2 = max(0, min(CGFloat(origH), y2))
                if x2 > x1 && y2 > y1 {
                    dets.append(Detection(cls: c, score: s,
                                          rect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1),
                                          maskCoeffs: []))
                }
            }
        }
        if end2end {
            if y.dataType == .float16 {
                let rawPtr = y.dataPointer
                decodeEndToEnd { r, f in halfToFloat(rawPtr.load(fromByteOffset: (r * s1 + f * s2) * 2, as: UInt16.self)) }
            } else {
                y.withUnsafeBufferPointer(ofType: Float32.self) { buf in
                    guard let yp = buf.baseAddress else { return }
                    decodeEndToEnd { r, f in yp[r * s1 + f * s2] }
                }
            }
            dets.sort { $0.score > $1.score }
            return dets
        }
        // fast path: contiguous anchors, float32 OR float16, det AND seg. The fp16
        // case is the ANE's native output - the generic path costs ~100ms/frame
        // there (per-element closure + scalar half->float); here it is one
        // vectorized vImage conversion followed by the same vDSP-gated scan.
        if !end2end, s2 == 1,
           y.dataType == .float32 || y.dataType == .float16 {
            func scan(_ yp: UnsafePointer<Float32>) {
                for c in 0..<nc {
                    let row = yp + (4 + c) * s1
                    var mx: Float = 0
                    vDSP_maxv(row, 1, &mx, vDSP_Length(na))
                    if mx <= confFloor { continue }
                    for a in 0..<na {
                        let s = row[a]
                        if s <= confFloor { continue }
                        let cx = CGFloat(yp[a]), cy = CGFloat(yp[s1 + a])
                        let bw = CGFloat(yp[2 * s1 + a]), bh = CGFloat(yp[3 * s1 + a])
                        var x1 = (cx - bw / 2 - padX) / scaleX, y1 = (cy - bh / 2 - padY) / scaleY
                        var x2 = (cx + bw / 2 - padX) / scaleX, y2 = (cy + bh / 2 - padY) / scaleY
                        x1 = max(0, min(CGFloat(origW), x1)); x2 = max(0, min(CGFloat(origW), x2))
                        y1 = max(0, min(CGFloat(origH), y1)); y2 = max(0, min(CGFloat(origH), y2))
                        if x2 > x1 && y2 > y1 {
                            // seg: gather the nm coeffs down this anchor's column,
                            // survivors only (the whole-tensor work stays vectorized)
                            var coeffs: [Float] = []
                            if nm > 0 {
                                coeffs.reserveCapacity(nm)
                                let base = yp + (4 + nc) * s1 + a
                                for k in 0..<nm { coeffs.append(base[k * s1]) }
                            }
                            dets.append(Detection(cls: c, score: s,
                                                  rect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1),
                                                  maskCoeffs: coeffs))
                        }
                    }
                }
            }
            if y.dataType == .float32 {
                y.withUnsafeBufferPointer(ofType: Float32.self) { buf in
                    if let yp = buf.baseAddress { scan(yp) }
                }
            } else {
                // Convert the full STRIDED extent, not the logical count: ANE
                // tensors may pad rows (s1 > na), and converting only y.count
                // elements leaves the tail rows (the highest class indices)
                // reading garbage - the "wall of last-class detections" bug.
                let n = y.shape[1].intValue * s1
                var tmp = [Float32](repeating: 0, count: n)
                tmp.withUnsafeMutableBufferPointer { dstBuf in
                    var src = vImage_Buffer(data: y.dataPointer, height: 1,
                                            width: vImagePixelCount(n), rowBytes: n * 2)
                    var dst = vImage_Buffer(data: dstBuf.baseAddress, height: 1,
                                            width: vImagePixelCount(n), rowBytes: n * 4)
                    vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
                    if let yp = dstBuf.baseAddress { scan(yp) }
                }
            }
            dets.sort { $0.score > $1.score }
            return dets
        }
        func decodeAnchors(_ at: (Int, Int) -> Float32) {
            for a in 0..<na {
                let cx = CGFloat(at(0, a)), cy = CGFloat(at(1, a)), bw = CGFloat(at(2, a)), bh = CGFloat(at(3, a))
                for c in 0..<nc {
                    let s = at(4 + c, a)
                    if s <= confFloor { continue }
                    var x1 = (cx - bw / 2 - padX) / scaleX, y1 = (cy - bh / 2 - padY) / scaleY
                    var x2 = (cx + bw / 2 - padX) / scaleX, y2 = (cy + bh / 2 - padY) / scaleY
                    x1 = max(0, min(CGFloat(origW), x1)); x2 = max(0, min(CGFloat(origW), x2))
                    y1 = max(0, min(CGFloat(origH), y1)); y2 = max(0, min(CGFloat(origH), y2))
                    if x2 > x1 && y2 > y1 {
                        var coeffs: [Float] = []
                        if nm > 0 {
                            coeffs.reserveCapacity(nm)
                            for k in 0..<nm { coeffs.append(Float(at(4 + nc + k, a))) }
                        }
                        dets.append(Detection(cls: c, score: s,
                                              rect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1),
                                              maskCoeffs: coeffs))
                    }
                }
            }
        }
        if y.dataType == .float16 {
            let raw = y.dataPointer   // Float16 is unnameable on x86_64; read raw half bytes as UInt16
            decodeAnchors { c, a in halfToFloat(raw.load(fromByteOffset: (c * s1 + a * s2) * 2, as: UInt16.self)) }
        } else {
            y.withUnsafeBufferPointer(ofType: Float32.self) { buf in
                guard let yp = buf.baseAddress else { return }
                decodeAnchors { c, a in yp[c * s1 + a * s2] }
            }
        }
        dets.sort { $0.score > $1.score }
        return dets
    }

    /// Filter cached `candidates` by `conf` + per-class greedy NMS (cap 300). Cheap - no model call.
    /// `mode: .clusterWeighted` additionally rewrites survivor coordinates via CW refinement
    /// (see `cwRefine`); defaults keep every existing call site's behavior byte-identical.
    public static func nms(_ dets: [Detection], conf: Float, iou iouT: CGFloat, maxDet: Int = 300,
                           mode: NMSMode = .standard, sigma: Float = 0.1) -> [Detection] {
        var keep: [Detection] = []
        for d in dets {
            if d.score <= conf { break }   // input is score-sorted (see candidates()) -> nothing further passes
            if keep.count >= maxDet { break }
            if !keep.contains(where: { $0.cls == d.cls && rectIoU($0.rect, d.rect) > iouT }) { keep.append(d) }
        }
        guard mode == .clusterWeighted, !keep.isEmpty else { return keep }
        return cwRefine(keep, pool: dets, conf: conf, iou: iouT, sigma: sigma)
    }

    /// Cluster-Weighted refinement - the `cluster` variant of upstream YOLO-Master's CW-NMS
    /// (ultralytics/utils/nms.py, commit 31c3fe3). One shot, no iteration:
    ///   pool  = pre-NMS candidates with score > conf, capped at the top 3000 (input is score-sorted,
    ///           so prefix(while:) + prefix(3000) equals upstream's `scores > conf` + `topk(3000)`)
    ///   w_km  = score_m * exp(-(1 - IoU_km)^2 / sigma) for same-class pool boxes with IoU_km > iouT
    ///           (upstream separates classes via the class-offset trick; `cls ==` is equivalent.
    ///            NOTE the exponent is /sigma exactly - NOT the Gaussian /(2*sigma^2))
    ///   box_k = (sum_m w_km * box_m) / (sum_m w_km + 1e-6), all from ORIGINAL pre-update coords
    /// Kept set, scores, classes and mask coeffs are unchanged; only survivor rects move.
    /// Deliberate deviation from upstream: a survivor whose weight sum is ~0 keeps its original
    /// rect (upstream's top-3000 cap can collapse such a box to (0,0,0,0) - a latent bug).
    private static func cwRefine(_ keep: [Detection], pool dets: [Detection], conf: Float,
                                 iou iouT: CGFloat, sigma: Float) -> [Detection] {
        let pool = Array(dets.prefix(while: { $0.score > conf }).prefix(3000))
        guard !pool.isEmpty else { return keep }
        return keep.map { k in
            var sw: Float = 0, ax1: Float = 0, ay1: Float = 0, ax2: Float = 0, ay2: Float = 0
            for m in pool where m.cls == k.cls {
                let ov = rectIoU(k.rect, m.rect)
                guard ov > iouT else { continue }
                let w = m.score * expf(-powf(1 - Float(ov), 2) / sigma)
                sw += w
                ax1 += w * Float(m.rect.minX); ay1 += w * Float(m.rect.minY)
                ax2 += w * Float(m.rect.maxX); ay2 += w * Float(m.rect.maxY)
            }
            guard sw > 1e-6 else { return k }
            let inv = 1 / (sw + 1e-6)
            let x1 = CGFloat(ax1 * inv), y1 = CGFloat(ay1 * inv)
            let x2 = CGFloat(ax2 * inv), y2 = CGFloat(ay2 * inv)
            return Detection(cls: k.cls, score: k.score,
                             rect: CGRect(x: x1, y: y1, width: max(0, x2 - x1), height: max(0, y2 - y1)),
                             maskCoeffs: k.maskCoeffs)
        }
    }

    // ---------- public inference ----------
    /// `inferMs` is the Core ML prediction alone; `preMs` letterbox + input tensor build; `postMs` decode + NMS
    /// (the Linux runtime's pre / infer / post split, so bench numbers line up across platforms).
    public struct Result: Sendable {
        public let detections: [Detection]
        public let inferMs: Double
        public let preMs: Double
        public let postMs: Double
        public init(detections: [Detection], inferMs: Double, preMs: Double = 0, postMs: Double = 0) {
            self.detections = detections; self.inferMs = inferMs; self.preMs = preMs; self.postMs = postMs
        }
    }

    /// Cached forward-pass output + letterbox geometry. Hold onto this and re-decode with
    /// different conf/iou via `decode(_:conf:iou:)` - no second model call. Post-processing
    /// (conf/iou threshold, NMS) is a frontend concern, not an inference one.
    public final class RawOutput {
        fileprivate let y: MLMultiArray
        fileprivate let proto: MLMultiArray?   // segmentation prototypes [1, nm, mh, mw]
        fileprivate let scaleX, scaleY, padX, padY: CGFloat   // per-axis (scaleX==scaleY for letterbox)
        public let origW, origH: Int
        public let inferMs: Double
        /// Preprocess wall time (letterbox + tensor build, or texture upload + Metal kernel + wait), the Linux `pre_ms`.
        public let preMs: Double
        /// GPU command-buffer time of the Metal preprocess (0 on the CPU path); a sub-metric of `preMs`.
        public let preGpuMs: Double
        fileprivate init(y: MLMultiArray, proto: MLMultiArray?, scaleX: CGFloat, scaleY: CGFloat, padX: CGFloat, padY: CGFloat,
                         origW: Int, origH: Int, inferMs: Double, preMs: Double = 0, preGpuMs: Double = 0) {
            self.y = y; self.proto = proto; self.scaleX = scaleX; self.scaleY = scaleY; self.padX = padX; self.padY = padY
            self.origW = origW; self.origH = origH; self.inferMs = inferMs; self.preMs = preMs; self.preGpuMs = preGpuMs
        }

        /// A copy retaining ONLY what mask rendering needs (proto tensor + letterbox geometry),
        /// dropping the detection tensor `y` - the larger half of a RawOutput. Per-frame video
        /// caches keep these: ~halves seg-video memory. nil for non-seg outputs.
        /// (maskImage/maskOverlay never touch `y`; a 1-element placeholder satisfies the field.)
        public func maskOnly() -> RawOutput? {
            guard let p = proto, let tiny = try? MLMultiArray(shape: [1], dataType: .float32) else { return nil }
            return RawOutput(y: tiny, proto: p, scaleX: scaleX, scaleY: scaleY, padX: padX, padY: padY,
                             origW: origW, origH: origH, inferMs: inferMs, preMs: preMs, preGpuMs: preGpuMs)
        }
    }

    /// Core ML forward pass only (letterbox → predict). Cache the result and re-`decode`.
    public func forward(_ image: CGImage) throws -> RawOutput {
        if gpuActive, let mp = metal {
            let tp = Date()
            guard let tex = mp.texture(from: image),
                  let o = mp.run(texture: tex, srcW: image.width, srcH: image.height, imgsz: imgsz, stretch: preprocess == .stretch)
            else { return try forwardCPU(image) }          // any Metal hiccup: the CPU path, same contract
            return try predict(o, preMs: Date().timeIntervalSince(tp) * 1000, origW: image.width, origH: image.height)
        }
        return try forwardCPU(image)
    }
    private func forwardCPU(_ image: CGImage) throws -> RawOutput {
        let tp = Date()
        let lb = letterbox(image)
        guard let input = fillInput(lb.px) else { throw DetectorError.inputBuildFailed }
        let t0 = Date()
        let preMs = t0.timeIntervalSince(tp) * 1000
        let out = try model.prediction(from: input)
        let infMs = Date().timeIntervalSince(t0) * 1000
        let y = try validatedOutput(out)
        let proto = isSegment ? out.featureValue(for: protoName)?.multiArrayValue : nil
        return RawOutput(y: y, proto: proto, scaleX: lb.scaleX, scaleY: lb.scaleY, padX: lb.padX, padY: lb.padY,
                         origW: image.width, origH: image.height, inferMs: infMs, preMs: preMs)
    }
    /// Prediction on a Metal-preprocessed tensor (integer-pad geometry).
    private func predict(_ o: MetalPreprocessor.Output, preMs: Double, origW: Int, origH: Int) throws -> RawOutput {
        guard let input = try? MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: o.array)]) else {
            throw DetectorError.inputBuildFailed
        }
        let t0 = Date()
        let out = try model.prediction(from: input)
        let infMs = Date().timeIntervalSince(t0) * 1000
        let y = try validatedOutput(out)
        let proto = isSegment ? out.featureValue(for: protoName)?.multiArrayValue : nil
        let g = o.geometry
        return RawOutput(y: y, proto: proto, scaleX: g.scaleX, scaleY: g.scaleY, padX: CGFloat(g.padX), padY: CGFloat(g.padY),
                         origW: origW, origH: origH, inferMs: infMs, preMs: preMs, preGpuMs: o.gpuMs)
    }

    /// Low-latency forward from a camera `CVPixelBuffer` (BGRA). On the GPU path the buffer is read by
    /// the Metal kernel directly (zero copy through the texture cache); on the CPU path it is wrapped
    /// as a CGImage with a single copy (no CIContext) then letterboxed as usual.
    public func forward(_ pixelBuffer: CVPixelBuffer) throws -> RawOutput {
        if gpuActive, let mp = metal, let (tex, holder) = mp.texture(from: pixelBuffer) {
            let tp = Date()
            let w = CVPixelBufferGetWidth(pixelBuffer), h = CVPixelBufferGetHeight(pixelBuffer)
            if let o = mp.run(texture: tex, srcW: w, srcH: h, imgsz: imgsz, stretch: preprocess == .stretch) {
                _ = holder   // keeps the CVMetalTexture alive through the pass
                return try predict(o, preMs: Date().timeIntervalSince(tp) * 1000, origW: w, origH: h)
            }
        }
        guard let cg = Detector.cgImage(from: pixelBuffer) else { throw DetectorError.inputBuildFailed }
        return try forward(cg)
    }

    /// The model input tensor for `image` on the current preprocessing device (no prediction), as
    /// raw float32 NCHW bytes: what `--dump-input` writes for the Linux parity check.
    public func inputTensorBytes(_ image: CGImage) -> Data? {
        if gpuActive, let mp = metal, let tex = mp.texture(from: image),
           let o = mp.run(texture: tex, srcW: image.width, srcH: image.height, imgsz: imgsz, stretch: preprocess == .stretch) {
            return Data(bytes: o.array.dataPointer, count: 3 * imgsz * imgsz * MemoryLayout<Float>.size)
        }
        let lb = letterbox(image)
        guard let input = fillInput(lb.px), let arr = input.featureValue(for: inputName)?.multiArrayValue else { return nil }
        return Data(bytes: arr.dataPointer, count: 3 * imgsz * imgsz * MemoryLayout<Float>.size)
    }

    /// Forward a tile crop, padded bottom-right with gray 114 to tileSize×tileSize and (when
    /// tileSize > imgsz) letterboxed down to the model input - upstream `_pad_slice`
    /// (predictor.py:523-530) followed by its standard preprocess. tileSize == imgsz is the
    /// native-scale fast path (scale 1). Returned geometry: scale = imgsz/tileSize, pads = 0,
    /// origW/H = CROP dims so `candidates()`'s existing clamp trims detections to real crop
    /// content (deliberate deviation from upstream, which lets boxes live on the padding).
    /// Precondition: crop ≤ tileSize in both axes (guaranteed by `tileGrid`).
    internal func forwardPadded(_ crop: CGImage, tileSize: Int? = nil) throws -> RawOutput {
        let tile = tileSize ?? imgsz
        let cw = crop.width, ch = crop.height
        precondition(cw <= tile && ch <= tile, "tile crop exceeds tile size")
        let s = CGFloat(imgsz) / CGFloat(tile)   // 1 when tile == imgsz; <1 shrinks bigger tiles
        let tp = Date()
        var px = [UInt8](repeating: 114, count: imgsz * imgsz * 4)
        px.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: imgsz, height: imgsz, bitsPerComponent: 8,
                                      bytesPerRow: imgsz * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
            ctx.interpolationQuality = .high
            // Same no-flip convention as letterbox(): raster top-left = CG-space y (imgsz - ch*s).
            ctx.draw(crop, in: CGRect(x: 0, y: CGFloat(imgsz) - CGFloat(ch) * s,
                                      width: CGFloat(cw) * s, height: CGFloat(ch) * s))
        }
        guard let input = fillInput(px) else { throw DetectorError.inputBuildFailed }
        let t0 = Date()
        let preMs = t0.timeIntervalSince(tp) * 1000
        let out = try model.prediction(from: input)
        let infMs = Date().timeIntervalSince(t0) * 1000
        let y = try validatedOutput(out)
        // proto deliberately nil: tile coeffs are meaningless against a full-image proto tensor.
        return RawOutput(y: y, proto: nil, scaleX: s, scaleY: s, padX: 0, padY: 0,
                         origW: cw, origH: ch, inferMs: infMs, preMs: preMs)
    }

    /// Cheap BGRA `CVPixelBuffer` → `CGImage` (one memcpy via a buffer-backed context; no Core Image).
    public static func cgImage(from pb: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let bmp = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue  // 32BGRA
        guard let ctx = CGContext(data: base, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bmp) else { return nil }
        return ctx.makeImage()
    }

    /// Decode + per-class NMS from a cached forward pass. Cheap - no model call.
    public func decode(_ raw: RawOutput, conf: Float, iou iouT: CGFloat,
                       mode: NMSMode = .standard, sigma: Float = 0.1, maxDet: Int = 300) -> [Detection] {
        Detector.nms(candidates(raw, confFloor: conf), conf: conf, iou: iouT, maxDet: maxDet, mode: mode, sigma: sigma)
    }

    /// Convenience: forward + decode in one call (used by the CLI). `inferMs` is model-only latency;
    /// `preMs` / `postMs` are the preprocess and decode + NMS stages around it.
    public func detect(_ image: CGImage, conf: Float = 0.25, iou iouT: CGFloat = 0.5,
                       mode: NMSMode = .standard, sigma: Float = 0.1, maxDet: Int = 300) throws -> Result {
        let raw = try forward(image)
        let t0 = Date()
        let dets = decode(raw, conf: conf, iou: iouT, mode: mode, sigma: sigma, maxDet: maxDet)
        let postMs = Date().timeIntervalSince(t0) * 1000
        return Result(detections: dets, inferMs: raw.inferMs, preMs: raw.preMs, postMs: postMs)
    }

    // ---------- segmentation masks ----------
    /// Instance mask for one detection: threshold(sigmoid(coeffs · protos)), tinted by class color,
    /// as a proto-resolution RGBA image plus the unit sub-rect of the proto grid that maps to the full
    /// original image (`protoCrop`) and the box to clip it to (`clip`, original-image pixels). nil if
    /// the model isn't segmentation or there's no proto tensor.
    public func maskImage(_ det: Detection, _ raw: RawOutput, threshold: Float = 0.5, alpha: UInt8 = 165) -> MaskBitmap? {
        guard let proto = raw.proto, proto.shape.count == 4, !det.maskCoeffs.isEmpty else { return nil }
        let cm = proto.shape[1].intValue, mh = proto.shape[2].intValue, mw = proto.shape[3].intValue
        let nmv = min(cm, det.maskCoeffs.count)
        let s1 = proto.strides[1].intValue, s2 = proto.strides[2].intValue, s3 = proto.strides[3].intValue
        let coeffs = det.maskCoeffs
        // tint from the class palette (context is premultipliedLast -> RGB carries color*alpha)
        let comps = classColor(det.cls).components ?? [1, 0.25, 0.25, 1]
        let cr = Float(comps[0]), cg = Float(comps[1]), cb = Float(comps[2]), aMax = Float(alpha)
        // Anti-aliased coverage instead of a hard threshold: smoothstep the sigmoid across a soft band
        // around `threshold`. Combined with bilinear upscaling this removes the classic serrated edge.
        let band: Float = 0.14, e0 = threshold - band, e1 = threshold + band, inv = 1 / (e1 - e0)
        var px = [UInt8](repeating: 0, count: mw * mh * 4)
        func fill(_ at: (Int, Int, Int) -> Float32) {
            for i in 0..<mh {
                for j in 0..<mw {
                    var acc: Float = 0
                    for k in 0..<nmv { acc += coeffs[k] * Float(at(k, i, j)) }
                    let v = 1 / (1 + expf(-acc))
                    var t = (v - e0) * inv
                    if t <= 0 { continue }
                    if t > 1 { t = 1 }
                    let a = t * t * (3 - 2 * t) * aMax          // smoothstep coverage * max alpha
                    let o = (i * mw + j) * 4
                    px[o] = UInt8(min(255, cr * a)); px[o + 1] = UInt8(min(255, cg * a))
                    px[o + 2] = UInt8(min(255, cb * a)); px[o + 3] = UInt8(min(255, a))
                }
            }
        }
        if proto.dataType == .float16 {
            let raw = proto.dataPointer   // Float16 is unnameable on x86_64; read raw half bytes as UInt16
            fill { k, i, j in halfToFloat(raw.load(fromByteOffset: (k * s1 + i * s2 + j * s3) * 2, as: UInt16.self)) }
        } else {
            proto.withUnsafeBufferPointer(ofType: Float32.self) { buf in
                guard let pp = buf.baseAddress else { return }
                fill { k, i, j in pp[k * s1 + i * s2 + j * s3] }
            }
        }
        let bmp = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(data: &px, width: mw, height: mh, bitsPerComponent: 8, bytesPerRow: mw * 4,
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bmp),
              let img = ctx.makeImage() else { return nil }
        // proto grid covers the letterboxed imgsz×imgsz input; the original image occupies
        // [pad, pad + orig*scale]. Express that as a unit crop of the proto grid (top-left origin).
        let sz = CGFloat(imgsz)
        let crop = CGRect(x: raw.padX / sz, y: raw.padY / sz,
                          width: CGFloat(raw.origW) * raw.scaleX / sz, height: CGFloat(raw.origH) * raw.scaleY / sz)
        return MaskBitmap(image: img, protoCrop: crop, clip: det.rect)
    }

    /// Transparent original-image-sized overlay compositing every detection's instance mask (box-clipped),
    /// for the Canvas-based stages (video / live camera) that draw over a raw frame rather than through
    /// `annotate`. Returns nil for non-seg models or when no mask survives. `dets` should be post-NMS.
    ///
    /// BATCHED math (Accelerate): the old path ran a scalar coeffs·proto matmul PER DETECTION,
    /// re-reading (and for fp16 re-converting) the whole proto tensor each time - seconds at
    /// hundreds of detections. Now: proto is unpacked to contiguous Float32 ONCE, all
    /// detections' coefficients form one [N,nm] matrix, and a single SGEMM ([N,nm]x[nm,plane],
    /// dispatched to the AMX matrix units) plus vectorized sigmoid produce every mask grid in
    /// one shot. Only the cheap per-pixel tint + the GPU-backed CG composite remain per mask.
    /// `maxSide` > 0 renders the composite at a downscaled resolution (longest side capped);
    /// callers draw the overlay stretched over the image, and the proto grid is only 160px,
    /// so nothing is lost - it just avoids a full-res RGBA canvas per large photo.
    public func maskOverlay(_ dets: [Detection], _ raw: RawOutput, maxSide: Int = 0) -> CGImage? {
        guard isSegment, let proto = raw.proto, proto.shape.count == 4, !dets.isEmpty else { return nil }
        guard raw.origW > 0, raw.origH > 0 else { return nil }
        let f: CGFloat = maxSide > 0 ? min(1, CGFloat(maxSide) / CGFloat(max(raw.origW, raw.origH))) : 1
        let w = max(1, Int((CGFloat(raw.origW) * f).rounded())), h = max(1, Int((CGFloat(raw.origH) * f).rounded()))
        let cm = proto.shape[1].intValue, mh = proto.shape[2].intValue, mw = proto.shape[3].intValue
        let plane = mh * mw
        guard cm > 0, plane > 0 else { return nil }
        let usable = dets.filter { !$0.maskCoeffs.isEmpty }
        guard !usable.isEmpty else { return nil }
        let N = usable.count

        // ---- proto -> contiguous [cm, plane] Float32, once ----
        // The unpack MUST be vectorized: 32x160x160 = 819k elements per frame, and
        // a scalar half->float loop here alone costs ~100ms on-device (the same
        // failure mode as the old scalar decode path). Contiguous rows (s3 == 1,
        // the CoreML/ANE layout) take one bulk vImage convert of the full strided
        // extent + row memcpys; anything else falls back to the scalar walk.
        let s1 = proto.strides[1].intValue, s2 = proto.strides[2].intValue, s3 = proto.strides[3].intValue
        var protoF = [Float](repeating: 0, count: cm * plane)
        if proto.dataType == .float16 {
            let rp = proto.dataPointer
            if s3 == 1 {
                let n = cm * s1
                var tmp = [Float](repeating: 0, count: n)
                tmp.withUnsafeMutableBufferPointer { t in
                    var src = vImage_Buffer(data: rp, height: 1,
                                            width: vImagePixelCount(n), rowBytes: n * 2)
                    var dst = vImage_Buffer(data: t.baseAddress, height: 1,
                                            width: vImagePixelCount(n), rowBytes: n * 4)
                    vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
                }
                protoF.withUnsafeMutableBufferPointer { dst in
                    tmp.withUnsafeBufferPointer { sp in
                        for k in 0..<cm {
                            for i in 0..<mh {
                                memcpy(dst.baseAddress! + (k * plane + i * mw),
                                       sp.baseAddress! + (k * s1 + i * s2), mw * 4)
                            }
                        }
                    }
                }
            } else {
                protoF.withUnsafeMutableBufferPointer { dst in
                    for k in 0..<cm {
                        let ko = k * plane
                        for i in 0..<mh {
                            let ro = ko + i * mw, so = k * s1 + i * s2
                            for j in 0..<mw {
                                dst[ro + j] = halfToFloat(rp.load(fromByteOffset: (so + j * s3) * 2, as: UInt16.self))
                            }
                        }
                    }
                }
            }
        } else {
            proto.withUnsafeBufferPointer(ofType: Float32.self) { buf in
                guard let pp = buf.baseAddress else { return }
                protoF.withUnsafeMutableBufferPointer { dst in
                    if s3 == 1 {
                        for k in 0..<cm {
                            for i in 0..<mh {
                                memcpy(dst.baseAddress! + (k * plane + i * mw),
                                       pp + (k * s1 + i * s2), mw * 4)
                            }
                        }
                    } else {
                        for k in 0..<cm {
                            let ko = k * plane
                            for i in 0..<mh {
                                let ro = ko + i * mw, so = k * s1 + i * s2
                                for j in 0..<mw { dst[ro + j] = pp[so + j * s3] }
                            }
                        }
                    }
                }
            }
        }

        // ---- one SGEMM for all masks + batch sigmoid ----
        var A = [Float](repeating: 0, count: N * cm)
        for (r, d) in usable.enumerated() {
            let nmv = min(cm, d.maskCoeffs.count)
            for k in 0..<nmv { A[r * cm + k] = d.maskCoeffs[k] }
        }
        var acc = [Float](repeating: 0, count: N * plane)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(N), Int32(plane), Int32(cm),
                    1, A, Int32(cm), protoF, Int32(plane), 0, &acc, Int32(plane))
        var cnt = Int32(N * plane)
        vDSP_vneg(acc, 1, &acc, 1, vDSP_Length(N * plane))       // -x
        vvexpf(&acc, acc, &cnt)                                   // e^-x
        var one: Float = 1
        vDSP_vsadd(acc, 1, &one, &acc, 1, vDSP_Length(N * plane)) // 1 + e^-x
        vvrecf(&acc, acc, &cnt)                                   // sigmoid

        // ---- per-mask tint + composite (identical visual pipeline to maskImage) ----
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.interpolationQuality = .high   // bilinear upscale proto->full res -> smooth mask edges
        let sz = CGFloat(imgsz)
        let crop = CGRect(x: raw.padX / sz, y: raw.padY / sz,
                          width: CGFloat(raw.origW) * raw.scaleX / sz, height: CGFloat(raw.origH) * raw.scaleY / sz)
        let threshold: Float = 0.5, alphaMax: Float = 165
        let band: Float = 0.14, e0 = threshold - band, e1 = threshold + band, inv = 1 / (e1 - e0)
        var drew = false
        // Edge quality: the proto grid is only imgsz/4 (160px at 640), so thresholding
        // AT proto resolution bakes its staircase into the contour no matter how the
        // bitmap is upscaled later. Instead, Lanczos-upscale the CONTINUOUS sigmoid
        // field 4x first, then smoothstep-threshold at that resolution: the iso-line
        // follows interpolated curves (rounded, blended edges), not grid steps.
        let u = 4
        let umw = mw * u, umh = mh * u, uplane = umw * umh
        var field = [Float](repeating: 0, count: uplane)   // upscaled sigmoid field
        var hbuf = [Float](repeating: 0, count: mh * umw)  // horizontal-pass scratch
        var px = [UInt8](repeating: 0, count: uplane * 4)
        // per-det tint, fully vectorized (the scalar smoothstep loop was the other
        // debug-visible hot spot): coverage a = smoothstep(t) * alphaMax as vDSP
        // planes, premultiplied channel planes, one vImage 4-plane interleave.
        var aF = [Float](repeating: 0, count: uplane)   // coverage (ends premultiplied alpha)
        var tt = [Float](repeating: 0, count: uplane)   // t^2 scratch
        var chF = [Float](repeating: 0, count: uplane)  // channel scratch
        var chR = [UInt8](repeating: 0, count: uplane), chG = chR, chB = chR, chA = chR
        let vn = vDSP_Length(uplane)
        for (r, d) in usable.enumerated() {
            let comps = classColor(d.cls).components ?? [1, 0.25, 0.25, 1]
            let cr = Float(comps[0]), cg = Float(comps[1]), cb = Float(comps[2])
            // Separable BILINEAR upscale, two vDSP passes. Deliberately NOT
            // Lanczos (vImageScale): its negative lobes ring on a near-binary
            // field, dipping interior values below the threshold band - seen
            // on-device as pinholes and uneven opacity. Bilinear is monotone:
            // interiors stay saturated, contours stay smooth curves.
            acc.withUnsafeBufferPointer { ap in
                let sp = ap.baseAddress! + r * plane
                hbuf.withUnsafeMutableBufferPointer { hb in
                    let hp = hb.baseAddress!
                    for p in 0..<u {
                        var w1 = Float(p) / Float(u)
                        var w0 = 1 - w1
                        for row in 0..<mh {
                            let s = sp + row * mw
                            vDSP_vsmsma(s, 1, &w0, s + 1, 1, &w1,
                                        hp + row * umw + p, vDSP_Stride(u), vDSP_Length(mw - 1))
                            hp[row * umw + (mw - 1) * u + p] = s[mw - 1]   // edge extend
                        }
                    }
                }
            }
            hbuf.withUnsafeBufferPointer { hb in
                let hp = hb.baseAddress!
                field.withUnsafeMutableBufferPointer { fb in
                    let fp = fb.baseAddress!
                    for p in 0..<u {
                        var w1 = Float(p) / Float(u)
                        var w0 = 1 - w1
                        for row in 0..<mh {
                            let nxt = min(row + 1, mh - 1)                 // edge extend
                            vDSP_vsmsma(hp + row * umw, 1, &w0, hp + nxt * umw, 1, &w1,
                                        fp + (row * u + p) * umw, 1, vDSP_Length(umw))
                        }
                    }
                }
            }
            // t = clip((sigmoid - e0) * inv, 0, 1); a = t*t*(3 - 2t) * alphaMax
            field.withUnsafeBufferPointer { ap in
                var m = inv, add = -e0 * inv
                vDSP_vsmsa(ap.baseAddress!, 1, &m, &add, &aF, 1, vn)
            }
            var lo: Float = 0, hi: Float = 1
            vDSP_vclip(aF, 1, &lo, &hi, &aF, 1, vn)
            vDSP_vsq(aF, 1, &tt, 1, vn)
            var mneg2: Float = -2, add3: Float = 3
            vDSP_vsmsa(aF, 1, &mneg2, &add3, &aF, 1, vn)   // 3 - 2t
            vDSP_vmul(tt, 1, aF, 1, &aF, 1, vn)            // t^2 * (3 - 2t)
            var aMaxV = alphaMax
            vDSP_vsmul(aF, 1, &aMaxV, &aF, 1, vn)          // * alphaMax
            func plane8(_ scale: Float, _ out: inout [UInt8]) {
                var s = scale
                vDSP_vsmul(aF, 1, &s, &chF, 1, vn)
                vDSP_vfixru8(chF, 1, &out, 1, vn)
            }
            plane8(cr, &chR); plane8(cg, &chG); plane8(cb, &chB); plane8(1, &chA)
            px.withUnsafeMutableBytes { pd in
                chR.withUnsafeMutableBytes { rr in
                    chG.withUnsafeMutableBytes { gg in
                        chB.withUnsafeMutableBytes { bb in
                            chA.withUnsafeMutableBytes { aa in
                                func vb(_ p: UnsafeMutableRawBufferPointer, _ rb: Int) -> vImage_Buffer {
                                    vImage_Buffer(data: p.baseAddress, height: vImagePixelCount(umh),
                                                  width: vImagePixelCount(umw), rowBytes: rb)
                                }
                                var vr = vb(rr, umw), vg = vb(gg, umw), vbP = vb(bb, umw), va = vb(aa, umw)
                                var dest = vb(pd, umw * 4)
                                // planes interleave in argument order -> RGBA bytes
                                vImageConvert_Planar8toARGB8888(&vr, &vg, &vbP, &va, &dest,
                                                                vImage_Flags(kvImageNoFlags))
                            }
                        }
                    }
                }
            }
            guard let mctx = CGContext(data: &px, width: umw, height: umh, bitsPerComponent: 8, bytesPerRow: umw * 4,
                                       space: CGColorSpaceCreateDeviceRGB(),
                                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
                  let img = mctx.makeImage() else { continue }
            let sub = CGRect(x: crop.minX * CGFloat(umw), y: crop.minY * CGFloat(umh),
                             width: crop.width * CGFloat(umw), height: crop.height * CGFloat(umh))
            guard sub.width > 0, sub.height > 0, let cropped = img.cropping(to: sub) else { continue }
            ctx.saveGState()
            ctx.clip(to: CGRect(x: d.rect.minX * f, y: CGFloat(h) - d.rect.maxY * f,
                                width: d.rect.width * f, height: d.rect.height * f))
            ctx.draw(cropped, in: CGRect(x: 0, y: 0, width: w, height: h))  // upright, matches Canvas top-left mapping
            ctx.restoreGState()
            drew = true
        }
        return drew ? ctx.makeImage() : nil
    }

    // ---------- segmentation mask -> polygon contours (annotation export) ----------
    /// Contours of one detection's instance mask, in ORIGINAL-image pixels (top-left origin).
    /// Used by the annotation exporters (YOLO-seg / COCO polygons).
    ///
    /// Semantics match what the preview shows: hard threshold at the CENTER of `maskImage`'s
    /// smoothstep band (sigmoid(coeffs . proto) > threshold), restricted to `det.rect` (the
    /// preview clips masks to the box). OUTER contours only - holes are unrepresentable in
    /// YOLO-seg and COCO-polygon anyway. Components smaller than 2 proto cells are dropped
    /// (speckle). Rings are Douglas-Peucker-simplified with `epsilon` in original pixels,
    /// sorted by area (desc) and capped at `maxPolygons`.
    ///
    /// Bias note: contours pass through proto CELL CENTERS, so polygons sit up to half a proto
    /// cell (~2 px at 640/160) inside the rendered mask edge - comparable to `epsilon`, and
    /// tight-side annotations are the safer convention.
    /// Returns [] for non-seg models, missing proto, empty coeffs, or an all-background mask.
    public func maskPolygons(_ det: Detection, _ raw: RawOutput, threshold: Float = 0.5,
                             epsilon: CGFloat = 2.0, maxPolygons: Int = 8) -> [[CGPoint]] {
        guard let proto = raw.proto, proto.shape.count == 4, !det.maskCoeffs.isEmpty else { return [] }
        let cm = proto.shape[1].intValue, mh = proto.shape[2].intValue, mw = proto.shape[3].intValue
        guard mh > 0, mw > 0 else { return [] }
        let nmv = min(cm, det.maskCoeffs.count)
        let s1 = proto.strides[1].intValue, s2 = proto.strides[2].intValue, s3 = proto.strides[3].intValue
        let coeffs = det.maskCoeffs
        let sz = CGFloat(imgsz)

        // det.rect (original px) <-> proto grid (same mapping as maskImage's protoCrop, K:468-473):
        //   grid = (pad + orig * scale) / imgsz * gridDim
        func gridX(_ x: CGFloat) -> CGFloat { (raw.padX + x * raw.scaleX) / sz * CGFloat(mw) }
        func gridY(_ y: CGFloat) -> CGFloat { (raw.padY + y * raw.scaleY) / sz * CGFloat(mh) }
        func origX(_ gx: CGFloat) -> CGFloat { (gx / CGFloat(mw) * sz - raw.padX) / raw.scaleX }
        func origY(_ gy: CGFloat) -> CGFloat { (gy / CGFloat(mh) * sz - raw.padY) / raw.scaleY }

        // sub-grid covering det.rect, expanded by 1 cell, clamped to the proto
        let gx0 = max(0, Int(gridX(det.rect.minX).rounded(.down)) - 1)
        let gy0 = max(0, Int(gridY(det.rect.minY).rounded(.down)) - 1)
        let gx1 = min(mw, Int(gridX(det.rect.maxX).rounded(.up)) + 1)
        let gy1 = min(mh, Int(gridY(det.rect.maxY).rounded(.up)) + 1)
        guard gx1 > gx0, gy1 > gy0 else { return [] }
        let subW = gx1 - gx0, subH = gy1 - gy0
        // padded (+2 each axis) with a guaranteed-false border ring so every contour closes
        let gw = subW + 2, gh = subH + 2
        var grid = [Bool](repeating: false, count: gw * gh)

        func fill(_ at: (Int, Int, Int) -> Float32) {
            for i in 0..<subH {
                let pi = gy0 + i
                let cy = origY(CGFloat(pi) + 0.5)
                if cy < det.rect.minY || cy > det.rect.maxY { continue }   // preview clips to box
                for j in 0..<subW {
                    let pj = gx0 + j
                    let cx = origX(CGFloat(pj) + 0.5)
                    if cx < det.rect.minX || cx > det.rect.maxX { continue }
                    var acc: Float = 0
                    for k in 0..<nmv { acc += coeffs[k] * Float(at(k, pi, pj)) }
                    if 1 / (1 + expf(-acc)) > threshold { grid[(i + 1) * gw + (j + 1)] = true }
                }
            }
        }
        if proto.dataType == .float16 {
            let rp = proto.dataPointer   // Float16 unnameable on x86_64; read half bits (cf. maskImage)
            fill { k, i, j in halfToFloat(rp.load(fromByteOffset: (k * s1 + i * s2 + j * s3) * 2, as: UInt16.self)) }
        } else {
            proto.withUnsafeBufferPointer(ofType: Float32.self) { buf in
                guard let pp = buf.baseAddress else { return }
                fill { k, i, j in pp[k * s1 + i * s2 + j * s3] }
            }
        }

        // ---- 8-connected component labeling (iterative flood fill); < 2 cells = speckle ----
        var label = [Int](repeating: 0, count: gw * gh)
        var starts: [Int] = []                       // topmost-leftmost cell per kept component
        var nextLabel = 0
        var stack: [Int] = []
        for idx0 in 0..<(gw * gh) where grid[idx0] && label[idx0] == 0 {
            nextLabel += 1
            var count = 0
            stack.removeAll(keepingCapacity: true)
            stack.append(idx0); label[idx0] = nextLabel
            while let idx = stack.popLast() {
                count += 1
                let cy = idx / gw, cx = idx % gw
                for dy in -1...1 {
                    for dx in -1...1 where dx != 0 || dy != 0 {
                        let ny = cy + dy, nx = cx + dx
                        guard ny >= 0, ny < gh, nx >= 0, nx < gw else { continue }
                        let nidx = ny * gw + nx
                        if grid[nidx] && label[nidx] == 0 { label[nidx] = nextLabel; stack.append(nidx) }
                    }
                }
            }
            if count >= 2 { starts.append(idx0) }    // row-major scan => idx0 is topmost-leftmost
        }
        guard !starts.isEmpty else { return [] }

        // ---- Moore-neighbor boundary trace (clockwise, Jacob's stopping criterion) ----
        // Neighbor offsets clockwise from West in a top-down grid.
        let nbr: [(dx: Int, dy: Int)] = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
        func traceRing(from start: Int, component: Int) -> [(x: Int, y: Int)] {
            let sx = start % gw, sy = start / gw
            var ring: [(x: Int, y: Int)] = [(sx, sy)]
            var cur = (x: sx, y: sy)
            var back = 0                              // entered scanning from West (border ring guarantees background there)
            var firstDir = -1
            let maxSteps = 4 * gw * gh
            for _ in 0..<maxSteps {
                var found = false
                for t in 0..<8 {
                    let d = (back + 1 + t) % 8        // clockwise from just past the backtrack
                    let nx = cur.x + nbr[d].dx, ny = cur.y + nbr[d].dy
                    guard nx >= 0, nx < gw, ny >= 0, ny < gh, label[ny * gw + nx] == component else { continue }
                    // Jacob's criterion: back at start entering in the same direction as the first move
                    if nx == sx && ny == sy {
                        if firstDir == -1 { firstDir = d }
                        else if d == firstDir { return ring }
                    }
                    if firstDir == -1 { firstDir = d }
                    cur = (nx, ny)
                    if !(nx == sx && ny == sy) { ring.append(cur) }
                    back = (d + 4) % 8               // backtrack = direction we came FROM
                    found = true
                    break
                }
                if !found { return ring }            // isolated pair edge case: ring so far
                if cur.x == sx && cur.y == sy && ring.count > 1 { return ring }
            }
            return ring
        }

        // ---- Douglas-Peucker on an open polyline ----
        func perpDist(_ p: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
            let vx = b.x - a.x, vy = b.y - a.y
            let len = (vx * vx + vy * vy).squareRoot()
            if len < 1e-9 { return ((p.x - a.x) * (p.x - a.x) + (p.y - a.y) * (p.y - a.y)).squareRoot() }
            return abs(vx * (a.y - p.y) - (a.x - p.x) * vy) / len
        }
        func dp(_ pts: [CGPoint]) -> [CGPoint] {
            guard pts.count > 2 else { return pts }
            var keep = [Bool](repeating: false, count: pts.count)
            keep[0] = true; keep[pts.count - 1] = true
            var spans: [(Int, Int)] = [(0, pts.count - 1)]
            while let (i0, i1) = spans.popLast() {
                guard i1 > i0 + 1 else { continue }
                var maxD: CGFloat = -1; var maxI = i0
                for i in (i0 + 1)..<i1 {
                    let d = perpDist(pts[i], pts[i0], pts[i1])
                    if d > maxD { maxD = d; maxI = i }
                }
                if maxD > epsilon { keep[maxI] = true; spans.append((i0, maxI)); spans.append((maxI, i1)) }
            }
            return pts.indices.filter { keep[$0] }.map { pts[$0] }
        }
        func shoelace(_ pts: [CGPoint]) -> CGFloat {
            guard pts.count >= 3 else { return 0 }
            var a: CGFloat = 0
            for i in pts.indices {
                let q = pts[(i + 1) % pts.count]
                a += pts[i].x * q.y - q.x * pts[i].y
            }
            return abs(a) / 2
        }

        var rings: [[CGPoint]] = []
        for start in starts {
            let cells = traceRing(from: start, component: label[start])
            guard cells.count >= 3 else { continue }
            // padded-grid cell centers -> original px, clamped to det.rect intersect image
            let pts = cells.map { c -> CGPoint in
                let gx = CGFloat(gx0 + c.x - 1) + 0.5, gy = CGFloat(gy0 + c.y - 1) + 0.5
                let x = min(max(origX(gx), max(0, det.rect.minX)), min(CGFloat(raw.origW), det.rect.maxX))
                let y = min(max(origY(gy), max(0, det.rect.minY)), min(CGFloat(raw.origH), det.rect.maxY))
                return CGPoint(x: x, y: y)
            }
            // closed-ring DP: anchor at the pair of farthest-apart vertices, DP each half
            var farI = 0; var farD: CGFloat = -1
            for i in pts.indices {
                let dx = pts[i].x - pts[0].x, dy = pts[i].y - pts[0].y
                let d = dx * dx + dy * dy
                if d > farD { farD = d; farI = i }
            }
            let simplified: [CGPoint]
            if farI > 0 {
                let half1 = dp(Array(pts[0...farI]))
                let half2 = dp(Array(pts[farI...]) + [pts[0]])
                simplified = half1 + half2.dropFirst().dropLast()
            } else {
                simplified = dp(pts)
            }
            if simplified.count >= 3 { rings.append(simplified) }
        }
        rings.sort { shoelace($0) > shoelace($1) }
        return Array(rings.prefix(maxPolygons))
    }

    /// Model-only forward (no decode/draw) - for latency benchmarking. Returns ms.
    @discardableResult
    public func inferOnly(_ image: CGImage) throws -> Double {
        let lb = letterbox(image)
        guard let input = fillInput(lb.px) else { throw DetectorError.inputBuildFailed }
        let t0 = Date()
        _ = try model.prediction(from: input)
        return Date().timeIntervalSince(t0) * 1000
    }
}
