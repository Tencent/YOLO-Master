// Annotation export - turn cached inference results into training data.
//
// Three formats (YOLO TXT, COCO JSON, Pascal VOC XML), three source kinds (image, folder,
// video-with-frame-extraction), det + seg models. Writers are pure functions over a
// format-neutral intermediate representation so they never touch Detection/Detector.
//
// Dialect rule (WYSIWYG): the FILE dialect (det boxes vs seg polygons) follows what the
// preview shows - seg polygons only when the model is seg AND mask data exists for the run
// (single-pass, or tiled with keepGlobalMasks). Within a polygon-dialect export, an instance
// with no usable mask (coeff-less tile detection, or a mask that thresholds to nothing) emits
// its BOX as a 4-point polygon so every line/segmentation entry stays structurally valid.
// Pascal VOC has no standard segmentation field and always writes boxes.
import Foundation
import CoreGraphics
import CoreImage
import AVFoundation

// ---------- user-facing options ----------

public enum AnnotationFormat: String, CaseIterable, Sendable {
    case yoloTXT, cocoJSON, pascalVOC
    public var label: String {
        switch self {
        case .yoloTXT: return "YOLO (TXT)"
        case .cocoJSON: return "COCO (JSON)"
        case .pascalVOC: return "Pascal VOC (XML)"
        }
    }
    public var fileExtension: String {
        switch self {
        case .yoloTXT: return "txt"
        case .cocoJSON: return "json"
        case .pascalVOC: return "xml"
        }
    }
}

/// Which video frames get extracted + annotated.
public enum VideoSampling: Hashable, Sendable {
    case allFrames, onePerSecond, everyNth(Int)
    public func stride(fps: Double) -> Int {
        switch self {
        case .allFrames: return 1
        case .onePerSecond: return max(1, Int(fps.rounded()))
        case .everyNth(let n): return max(1, n)
        }
    }
    public var label: String {
        switch self {
        case .allFrames: return "Every frame"
        case .onePerSecond: return "One per second"
        case .everyNth(let n): return "Every \(n)th frame"
        }
    }
}

// ---------- format-neutral intermediate representation ----------

public struct AnnotationInstance: Sendable {
    public let cls: Int
    public let score: Float
    public let rect: CGRect            // ORIGINAL px, top-left origin
    public let polygons: [[CGPoint]]   // ORIGINAL px; empty = box-only instance
    public init(cls: Int, score: Float, rect: CGRect, polygons: [[CGPoint]] = []) {
        self.cls = cls; self.score = score; self.rect = rect; self.polygons = polygons
    }
}

public struct AnnotatedImage: Sendable {
    public let name: String            // stem (no extension)
    public let width: Int, height: Int
    public let instances: [AnnotationInstance]
    public init(name: String, width: Int, height: Int, instances: [AnnotationInstance]) {
        self.name = name; self.width = width; self.height = height; self.instances = instances
    }
}

public struct AnnotationExportResult: Sendable {
    public let images: Int
    public let instances: Int
    public let output: URL
    public init(images: Int, instances: Int, output: URL) {
        self.images = images; self.instances = instances; self.output = output
    }
}

/// Build the IR from post-NMS detections (WYSIWYG - pass exactly what the preview shows).
/// `includePolygons` = the run's dialect (seg model with mask data available). Instances whose
/// mask is unavailable/empty in a polygon export fall back to the box as a 4-point polygon.
public func annotationInstances(_ dets: [Detection], detector: Detector?,
                                raw: Detector.RawOutput?, includePolygons: Bool) -> [AnnotationInstance] {
    dets.map { d in
        var polys: [[CGPoint]] = []
        if includePolygons {
            if let det = detector, let raw, !d.maskCoeffs.isEmpty {
                polys = det.maskPolygons(d, raw)
            }
            if polys.isEmpty {   // coeff-less (tile det) or empty mask -> box as a valid polygon
                polys = [[CGPoint(x: d.rect.minX, y: d.rect.minY), CGPoint(x: d.rect.maxX, y: d.rect.minY),
                          CGPoint(x: d.rect.maxX, y: d.rect.maxY), CGPoint(x: d.rect.minX, y: d.rect.maxY)]]
            }
        }
        return AnnotationInstance(cls: d.cls, score: d.score, rect: d.rect, polygons: polys)
    }
}

// ---------- writers (pure) ----------

public enum AnnotationWriter {
    private static func f6(_ v: CGFloat) -> String { String(format: "%.6f", v) }

    /// One image's YOLO .txt. det: "cls cx cy w h" normalized. seg dialect: "cls x1 y1 x2 y2 …"
    /// using the LARGEST polygon per instance (YOLO-seg has no multi-part syntax; polygons
    /// arrive area-sorted from maskPolygons). Empty detections -> empty string (valid negative).
    public static func yoloLines(_ img: AnnotatedImage, segDialect: Bool) -> String {
        let W = CGFloat(max(1, img.width)), H = CGFloat(max(1, img.height))
        func nx(_ v: CGFloat) -> CGFloat { min(1, max(0, v / W)) }
        func ny(_ v: CGFloat) -> CGFloat { min(1, max(0, v / H)) }
        var lines: [String] = []
        for inst in img.instances {
            if segDialect, let poly = inst.polygons.first, poly.count >= 3 {
                let coords = poly.flatMap { [f6(nx($0.x)), f6(ny($0.y))] }.joined(separator: " ")
                lines.append("\(inst.cls) \(coords)")
            } else {
                let r = inst.rect
                lines.append("\(inst.cls) \(f6(nx(r.midX))) \(f6(ny(r.midY))) \(f6(nx(r.width))) \(f6(ny(r.height)))")
            }
        }
        return lines.joined(separator: "\n") + (lines.isEmpty ? "" : "\n")
    }

    public static func classesTXT(_ names: [String]) -> String {
        names.joined(separator: "\n") + "\n"
    }

    /// Per-image Pascal VOC XML. Boxes only (VOC has no polygon field); 1-based inclusive ints
    /// clamped to the image; no score element (strictest-parser compatibility).
    public static func vocXML(_ img: AnnotatedImage, names: [String], folder: String) -> String {
        func esc(_ s: String) -> String {
            s.replacingOccurrences(of: "&", with: "&amp;")
             .replacingOccurrences(of: "<", with: "&lt;")
             .replacingOccurrences(of: ">", with: "&gt;")
             .replacingOccurrences(of: "\"", with: "&quot;")
        }
        var x = """
        <annotation>
        \t<folder>\(esc(folder))</folder>
        \t<filename>\(esc(img.name)).jpg</filename>
        \t<size>
        \t\t<width>\(img.width)</width>
        \t\t<height>\(img.height)</height>
        \t\t<depth>3</depth>
        \t</size>
        \t<segmented>0</segmented>

        """
        for inst in img.instances {
            let name = inst.cls < names.count ? names[inst.cls] : "class\(inst.cls)"
            let xmin = max(1, min(img.width, Int(inst.rect.minX.rounded()) + 1))
            let ymin = max(1, min(img.height, Int(inst.rect.minY.rounded()) + 1))
            let xmax = max(xmin, min(img.width, Int(inst.rect.maxX.rounded())))
            let ymax = max(ymin, min(img.height, Int(inst.rect.maxY.rounded())))
            x += """
            \t<object>
            \t\t<name>\(esc(name))</name>
            \t\t<pose>Unspecified</pose>
            \t\t<truncated>0</truncated>
            \t\t<difficult>0</difficult>
            \t\t<bndbox>
            \t\t\t<xmin>\(xmin)</xmin>
            \t\t\t<ymin>\(ymin)</ymin>
            \t\t\t<xmax>\(xmax)</xmax>
            \t\t\t<ymax>\(ymax)</ymax>
            \t\t</bndbox>
            \t</object>

            """
        }
        x += "</annotation>\n"
        return x
    }

    // COCO document model (private): standard keys + a "score" extension key (GT readers
    // ignore unknown keys; prediction workflows get the confidence for free).
    private struct COCOInfo: Encodable { let description: String }
    private struct COCOImage: Encodable { let id: Int; let file_name: String; let width: Int; let height: Int }
    private struct COCOAnnotation: Encodable {
        let id: Int; let image_id: Int; let category_id: Int
        let bbox: [Double]; let area: Double; let iscrowd: Int
        let segmentation: [[Double]]?
        let score: Double
    }
    private struct COCOCategory: Encodable { let id: Int; let name: String }
    private struct COCODoc: Encodable {
        let info: COCOInfo
        let images: [COCOImage]; let annotations: [COCOAnnotation]; let categories: [COCOCategory]
    }

    /// ONE JSON for the whole set. `fileNames[i]` pairs with `images[i]` (may carry a relative
    /// dir like "frames/clip_000090.jpg"). `imageIDs` optional explicit ids (video: source frame
    /// index + 1 so ids survive re-export at a different sampling); default 1-based sequence.
    public static func cocoJSON(_ images: [AnnotatedImage], fileNames: [String],
                                names: [String], segDialect: Bool,
                                imageIDs: [Int]? = nil) throws -> Data {
        func shoelace(_ pts: [CGPoint]) -> Double {
            guard pts.count >= 3 else { return 0 }
            var a = 0.0
            for i in pts.indices {
                let q = pts[(i + 1) % pts.count]
                a += Double(pts[i].x) * Double(q.y) - Double(q.x) * Double(pts[i].y)
            }
            return abs(a) / 2
        }
        var imgs: [COCOImage] = [], anns: [COCOAnnotation] = []
        var annID = 1
        for (i, img) in images.enumerated() {
            let iid = imageIDs?[i] ?? (i + 1)
            imgs.append(COCOImage(id: iid, file_name: fileNames[i], width: img.width, height: img.height))
            for inst in img.instances {
                let r = inst.rect
                let segs: [[Double]]? = segDialect
                    ? inst.polygons.filter { $0.count >= 3 }.map { $0.flatMap { [Double($0.x), Double($0.y)] } }
                    : nil
                let area = segDialect
                    ? inst.polygons.map(shoelace).reduce(0, +)
                    : Double(r.width * r.height)
                anns.append(COCOAnnotation(
                    id: annID, image_id: iid, category_id: inst.cls + 1,
                    bbox: [Double(r.minX), Double(r.minY), Double(r.width), Double(r.height)],
                    area: area > 0 ? area : Double(r.width * r.height), iscrowd: 0,
                    segmentation: segs, score: Double(inst.score)))
                annID += 1
            }
        }
        let cats = names.enumerated().map { COCOCategory(id: $0.offset + 1, name: $0.element) }
        let doc = COCODoc(info: COCOInfo(description: "YOLO-Master macOS export"),
                          images: imgs, annotations: anns, categories: cats)
        let enc = JSONEncoder(); enc.outputFormatting = [.sortedKeys]
        return try enc.encode(doc)
    }
}

// ---------- orchestrators ----------

/// Folder (cached items) -> `output` dir. YOLO: <stem>.txt per image + classes.txt; VOC:
/// <stem>.xml per image; COCO: single annotations.coco.json. Det export never decodes pixels
/// (dims ride on FolderItem); seg polygon export re-forwards each image for its proto (the
/// same pattern/cost as exportFolderCached's mask path).
@discardableResult
public func exportAnnotationsFolder(_ items: [FolderItem], output: URL, names: [String],
                                    format: AnnotationFormat,
                                    conf: Float, iou: CGFloat, nmsMode: NMSMode = .standard, sigma: Float = 0.1, maxDet: Int = 300,
                                    detector: Detector? = nil, includePolygons: Bool = false,
                                    progress: ((_ done: Int, _ total: Int) -> Void)? = nil) throws -> AnnotationExportResult {
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    var cocoImages: [AnnotatedImage] = [], cocoNames: [String] = []
    var nInstances = 0
    var usedStems = Set<String>()
    for (i, item) in items.enumerated() {
        let dets = Detector.nms(item.candidates, conf: conf, iou: iou, maxDet: maxDet, mode: nmsMode, sigma: sigma)
        var raw: Detector.RawOutput? = nil
        var w = item.width, h = item.height
        if includePolygons, let det = detector, let cg = loadCGImage(item.url) {
            raw = try? det.forward(cg)
            if w == 0 { w = cg.width; h = cg.height }   // legacy cache without dims
        }
        let stem = uniqueStem(&usedStems, item.url.deletingPathExtension().lastPathComponent)
        let insts = annotationInstances(dets, detector: detector, raw: raw, includePolygons: includePolygons)
        let img = AnnotatedImage(name: stem, width: w, height: h, instances: insts)
        nInstances += insts.count
        switch format {
        case .yoloTXT:
            try AnnotationWriter.yoloLines(img, segDialect: includePolygons)
                .write(to: output.appendingPathComponent(stem + ".txt"), atomically: true, encoding: .utf8)
        case .pascalVOC:
            try AnnotationWriter.vocXML(img, names: names, folder: output.lastPathComponent)
                .write(to: output.appendingPathComponent(stem + ".xml"), atomically: true, encoding: .utf8)
        case .cocoJSON:
            cocoImages.append(img); cocoNames.append(item.url.lastPathComponent)
        }
        progress?(i + 1, items.count)
    }
    switch format {
    case .yoloTXT:
        try AnnotationWriter.classesTXT(names)
            .write(to: output.appendingPathComponent("classes.txt"), atomically: true, encoding: .utf8)
    case .cocoJSON:
        let data = try AnnotationWriter.cocoJSON(cocoImages, fileNames: cocoNames,
                                                 names: names, segDialect: includePolygons)
        try data.write(to: output.appendingPathComponent("annotations.coco.json"))
    case .pascalVOC: break
    }
    return AnnotationExportResult(images: items.count, instances: nInstances, output: output)
}

/// Video -> `root/frames/<stem>_%06d.jpg` (source-frame-indexed, upright) + labels.
/// YOLO/VOC labels land in `root/labels/` (+ classes.txt for YOLO); COCO becomes ONE
/// `root/annotations.coco.json` with file_name = "frames/…". Reader loop mirrors
/// exportVideoCached exactly (frame n == framesCands[n], including `n += 1; continue` on
/// decode failures); non-sampled frames skip the expensive CIImage->CGImage conversion.
@discardableResult
public func exportAnnotationsVideo(input: URL, root: URL, framesCands: [[Detection]], names: [String],
                                   format: AnnotationFormat, sampling: VideoSampling, fps: Double,
                                   conf: Float, iou: CGFloat, nmsMode: NMSMode = .standard, sigma: Float = 0.1, maxDet: Int = 300,
                                   raws: [Detector.RawOutput?] = [], detector: Detector? = nil,
                                   includePolygons: Bool = false,
                                   progress: ((_ done: Int, _ total: Int) -> Void)? = nil) async throws -> AnnotationExportResult {
    let framesDir = root.appendingPathComponent("frames")
    let labelsDir = root.appendingPathComponent("labels")
    try FileManager.default.createDirectory(at: framesDir, withIntermediateDirectories: true)
    if format != .cocoJSON {
        try FileManager.default.createDirectory(at: labelsDir, withIntermediateDirectories: true)
    }
    let stem = input.deletingPathExtension().lastPathComponent
    let strideN = sampling.stride(fps: fps)

    let asset = AVURLAsset(url: input)
    guard let tracks = try? await asset.loadTracks(withMediaType: .video), let track = tracks.first else {
        throw PipelineError.noVideoTrack
    }
    let transform = (try? await track.load(.preferredTransform)) ?? .identity
    let orient = videoOrientation(transform)
    guard let reader = try? AVAssetReader(asset: asset) else { throw PipelineError.readerInit }
    let rout = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    rout.alwaysCopiesSampleData = false; reader.add(rout); reader.startReading()

    let cictx = CIContext()
    var n = 0
    let total = framesCands.count
    var cocoImages: [AnnotatedImage] = [], cocoNames: [String] = [], cocoIDs: [Int] = []
    var nImages = 0, nInstances = 0
    while reader.status == .reading {
        guard let sb = rout.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sb) else { break }
        guard n % strideN == 0, n < framesCands.count else { n += 1; continue }
        let ci = CIImage(cvPixelBuffer: pb).oriented(orient)   // upright, matching AVPlayer
        guard let cg = cictx.createCGImage(ci, from: ci.extent) else { n += 1; continue }
        let frameStem = String(format: "%@_%06d", stem, n)
        saveCGImage(cg, to: framesDir.appendingPathComponent(frameStem + ".jpg"))
        let dets = Detector.nms(framesCands[n], conf: conf, iou: iou, maxDet: maxDet, mode: nmsMode, sigma: sigma)
        let raw = n < raws.count ? raws[n] : nil
        let insts = annotationInstances(dets, detector: detector, raw: raw, includePolygons: includePolygons)
        let img = AnnotatedImage(name: frameStem, width: cg.width, height: cg.height, instances: insts)
        nImages += 1; nInstances += insts.count
        switch format {
        case .yoloTXT:
            try AnnotationWriter.yoloLines(img, segDialect: includePolygons)
                .write(to: labelsDir.appendingPathComponent(frameStem + ".txt"), atomically: true, encoding: .utf8)
        case .pascalVOC:
            try AnnotationWriter.vocXML(img, names: names, folder: "frames")
                .write(to: labelsDir.appendingPathComponent(frameStem + ".xml"), atomically: true, encoding: .utf8)
        case .cocoJSON:
            cocoImages.append(img); cocoNames.append("frames/" + frameStem + ".jpg"); cocoIDs.append(n + 1)
        }
        n += 1
        if nImages % 10 == 0 { progress?(min(n, total), total) }
    }
    switch format {
    case .yoloTXT:
        try AnnotationWriter.classesTXT(names)
            .write(to: labelsDir.appendingPathComponent("classes.txt"), atomically: true, encoding: .utf8)
    case .cocoJSON:
        let data = try AnnotationWriter.cocoJSON(cocoImages, fileNames: cocoNames,
                                                 names: names, segDialect: includePolygons, imageIDs: cocoIDs)
        try data.write(to: root.appendingPathComponent("annotations.coco.json"))
    case .pascalVOC: break
    }
    progress?(total, total)
    return AnnotationExportResult(images: nImages, instances: nInstances, output: root)
}
