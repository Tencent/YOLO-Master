// Batch photo detection: up to 100 images, gallery (3-up grid) or pager view,
// macOS-style phase progress (Loading/iCloud -> Inference), per-image stage
// averages + img/s dial. Slider retunes re-decode cached passes - no re-runs.
import CoreML
import Photos
import PhotosUI
import SwiftUI
import UIKit
import YOLOMasterKit

struct PhotoTestView: View {
    enum ViewMode { case gallery, pager }
    enum Phase: Equatable { case idle, loading, loadingModel, inferring }

    @State private var models: [BundledModel] = []
    @State private var selectedModel: BundledModel?
    @State private var compute: ComputeChoice = ComputeChoice.deviceDefault
    @AppStorage("allowCPU") private var allowCPU = false
    @State private var style: BoxStyle = .chip
    @State private var picks: [PhotosPickerItem] = []
    @State private var images: [CGImage] = []
    @State private var fileNames: [String] = []
    @State private var fileTypes: [String] = []
    @State private var results: [[Detection]] = []
    @State private var page = 0
    @State private var viewMode: ViewMode = .gallery
    @State private var conf = 0.25
    @State private var iou = 0.5
    @State private var showTuning = false
    @State private var showHUD = true
    @State private var isRunning = false
    @State private var phase: Phase = .idle
    @State private var progress = 0
    @State private var progressTotal = 0
    @State private var statPre: Double = 0
    @State private var statInf: Double = 0
    @State private var statDec: Double = 0
    @State private var throughput: Double = 0
    @State private var wallSeconds: Double = 0
    @State private var perPre: [Double] = []
    @State private var perInf: [Double] = []
    @State private var perDec: [Double] = []
    @State private var perMask: [Double] = []
    @State private var statMask: Double = 0
    @State private var errorText = ""
    @State private var lastDet: Detector?
    @State private var lastRaws: [Detector.RawOutput] = []
    @State private var maskImages: [CGImage?] = []   // seg composites, per image
    @State private var segOverlay: SegOverlay = .both
    @State private var isSeg = false
    @State private var classNames: [String] = cocoNames
    @State private var retuneGen = 0                 // drops stale async mask renders
    @State private var exporting = false
    @State private var exportMsg: String?            // non-nil -> result popup
    @State private var exportOK = false
    @State private var exportToastGen = 0            // auto-dismiss generation

    private let lightHaptic = UIImpactFeedbackGenerator(style: .light)
    private let notifyHaptic = UINotificationFeedbackGenerator()

    private let gridCols = [GridItem(.flexible(), spacing: 4),
                            GridItem(.flexible(), spacing: 4),
                            GridItem(.flexible(), spacing: 4)]

    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                content
                if phase != .idle { progressBar }
                if showTuning {
                    TuningPanel(conf: $conf, iou: $iou, style: $style,
                                hudVisible: $showHUD,
                                segOverlay: isSeg ? $segOverlay : nil)
                        .padding(.horizontal, 8)
                }
                if showHUD { hud.padding(.bottom, 20) }
                if !errorText.isEmpty {
                    Text(errorText).font(.caption2).foregroundStyle(.red)
                        .lineLimit(3).padding(.horizontal, 12)
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .safeAreaInset(edge: .top) { controls }
            .onAppear {
                lightHaptic.prepare()
                notifyHaptic.prepare()
                guard models.isEmpty else { return }
                Task.detached {
                    let found = BundledModel.discover()
                    await MainActor.run {
                        models = found
                        if selectedModel == nil {
                            selectedModel = BundledModel.preferred(in: found)
                        }
                    }
                }
            }
            .onChange(of: picks) { _, items in load(items) }
            .onChange(of: selectedModel) { _, _ in run() }
            .onChange(of: compute) { _, _ in run() }
            .onChange(of: allowCPU) { _, on in if !on && compute == .cpu { compute = ComputeChoice.deviceDefault } }
            .onChange(of: conf) { _, _ in retune() }
            .onChange(of: iou) { _, _ in retune() }
            .overlay {
                if let msg = exportMsg {
                    ExportToast(success: exportOK, message: msg)
                        .transition(.scale(scale: 0.82).combined(with: .opacity))
                        .onTapGesture {
                            withAnimation(.easeOut(duration: 0.2)) { exportMsg = nil }
                        }
                }
            }
        }
    }

    /// Pager: THIS image's stage times + its own capability FPS.
    /// Gallery: batch averages + overall FPS + detail rows.
    private var hud: some View {
        let perImage = viewMode == .pager && perInf.indices.contains(page)
        let p = perImage ? perPre[page] : statPre
        let i = perImage ? perInf[page] : statInf
        let d = perImage ? perDec[page] : statDec
        let mk = perImage && perMask.indices.contains(page) ? perMask[page] : statMask
        let e2e = p + i + d + (isSeg ? mk : 0)
        var extras: [(String, String)] = []
        if statInf > 0 {
            if perImage {
                let img = images[page]
                extras = [("image", "\(page + 1)/\(images.count)"),
                          ("size", "\(img.width)x\(img.height)"),
                          ("file", fileNames.indices.contains(page) ? fileNames[page] : "-"),
                          ("type", fileTypes.indices.contains(page) ? fileTypes[page] : "-"),
                          ("e2e", String(format: "%.1f ms", e2e)),
                          ("dets", "\(results.indices.contains(page) ? results[page].count : 0)")]
            } else {
                let totalDets = results.reduce(0) { $0 + $1.count }
                extras = [("images", "\(images.count)"),
                          ("wall", String(format: "%.2f s", wallSeconds)),
                          ("total dets", "\(totalDets)"),
                          ("avg e2e", String(format: "%.1f ms",
                                             statPre + statInf + statDec + (isSeg ? statMask : 0)))]
            }
        }
        // pager dial = this image's END-TO-END ms (0-30 scale, band colors);
        // gallery dial = batch FPS
        return StatsHUD(fps: perImage ? e2e : throughput, pre: p, inf: i, dec: d,
                        dets: results.indices.contains(page) ? results[page].count : 0,
                        fpsLabel: perImage ? "ms" : "FPS", active: statInf > 0,
                        extras: extras, mode: perImage ? .ms : .fps, fullWidth: true,
                        mask: isSeg ? mk : nil)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
    }

    // MARK: pieces

    @ViewBuilder private var content: some View {
        if images.isEmpty {
            ContentUnavailableView("No photos", systemImage: "photo.on.rectangle.angled",
                                   description: Text("Pick up to 100 images"))
                .frame(maxHeight: .infinity)
        } else if viewMode == .gallery {
            ScrollView {
                LazyVGrid(columns: gridCols, spacing: 4) {
                    ForEach(images.indices, id: \.self) { i in
                        galleryCell(i)
                    }
                }
                .padding(.horizontal, 8)
            }
            .transition(.scale(scale: 0.92).combined(with: .opacity))
        } else {
            TabView(selection: $page) {
                ForEach(images.indices, id: \.self) { i in
                    ZoomableScrollView(onPinchExit: {
                        withAnimation(.spring(duration: 0.35)) { viewMode = .gallery }
                    }) {
                        annotated(images[i], results.indices.contains(i) ? results[i] : [],
                                  maskImages.indices.contains(i) ? maskImages[i] : nil)
                    }
                    .tag(i)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .automatic))
            .transition(.scale(scale: 1.08).combined(with: .opacity))
        }
    }

    private func galleryCell(_ i: Int) -> some View {
        GeometryReader { geo in
            let img = images[i]
            let scale = geo.size.width / CGFloat(img.width)
            ZStack(alignment: .topLeading) {
                Image(decorative: img, scale: 1).resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: geo.size.width, height: geo.size.width)
                    .clipped()
                // mask composite shares the photo's aspect: identical modifiers
                // guarantee pixel alignment under the .fill crop
                if segOverlay != .boxes, maskImages.indices.contains(i),
                   let mask = maskImages[i] {
                    Image(decorative: mask, scale: 1).resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: geo.size.width, height: geo.size.width)
                        .clipped()
                }
                Canvas { ctx, _ in
                    if results.indices.contains(i) {
                        DetOverlay.draw(ctx, results[i], scale: scale, ox: 0, oy: 0,
                                        style: style, names: classNames,
                                        boxes: !(isSeg && segOverlay == .masks))
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .contentShape(Rectangle())
            .onTapGesture {
                page = i
                withAnimation(.spring(duration: 0.35)) { viewMode = .pager }
            }
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private var progressBar: some View {
        HStack(spacing: 10) {
            if phase == .loadingModel {
                ProgressView()
                Text("Loading the model to \(compute.rawValue)")
                    .font(.caption)
                Spacer(minLength: 0)
            } else {
                Text(phase == .loading ? "Loading / iCloud" : "Inference")
                    .font(.caption)
                ProgressView(value: Double(progress), total: Double(max(progressTotal, 1)))
                    .progressViewStyle(.linear)
                Text("\(progress)/\(progressTotal)")
                    .font(.caption.monospacedDigit())
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 8)
    }

    private var controls: some View {
        HStack {
            Menu {
                Picker("Model", selection: $selectedModel) {
                    ForEach(models) { m in
                        Text(m.fullName).tag(Optional(m))
                    }
                }
            } label: {
                HStack(spacing: 3) {
                    Text(selectedModel?.shortID ?? "Model").lineLimit(1)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2)
                }
            }
            .fixedSize()
            .disabled(isRunning)
            Picker("Compute", selection: $compute) {
                ForEach(ComputeChoice.available(allowCPU: allowCPU)) { c in
                    Text(c.rawValue).lineLimit(1).fixedSize().tag(c)
                }
            }
            .fixedSize()
            .disabled(isRunning)
            Spacer(minLength: 0)
            if !images.isEmpty {
                Button {
                    lightHaptic.impactOccurred()
                    lightHaptic.prepare()
                    withAnimation { viewMode = viewMode == .gallery ? .pager : .gallery }
                } label: {
                    Image(systemName: viewMode == .gallery
                          ? "rectangle.portrait" : "square.grid.3x3")
                }
                .buttonStyle(.bordered)
                // pager: export THIS image; gallery: export the whole batch
                Button {
                    lightHaptic.impactOccurred()
                    lightHaptic.prepare()
                    exportImages(viewMode == .pager ? [page] : Array(images.indices))
                } label: {
                    if exporting {
                        ProgressView()
                    } else {
                        Image(systemName: viewMode == .pager
                              ? "square.and.arrow.down"
                              : "square.and.arrow.down.on.square")
                    }
                }
                .buttonStyle(.bordered)
                .disabled(exporting || isRunning)
            }
            Button {
                withAnimation { showTuning.toggle() }
            } label: {
                Image(systemName: "slider.horizontal.3")
            }
            .buttonStyle(.bordered)
            PhotosPicker(selection: $picks, maxSelectionCount: 100, matching: .images,
                         photoLibrary: .shared()) {
                Image(systemName: "photo.badge.plus")
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning)
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 8)
    }

    private func annotated(_ img: CGImage, _ dets: [Detection], _ mask: CGImage?) -> some View {
        GeometryReader { geo in
            let scale = min(geo.size.width / CGFloat(img.width),
                            geo.size.height / CGFloat(img.height))
            let w = CGFloat(img.width) * scale
            let h = CGFloat(img.height) * scale
            ZStack(alignment: .topLeading) {
                Image(decorative: img, scale: 1).resizable()
                    .frame(width: w, height: h)
                if let mask, segOverlay != .boxes {
                    Image(decorative: mask, scale: 1).resizable()
                        .frame(width: w, height: h)
                }
                Canvas { ctx, _ in
                    DetOverlay.draw(ctx, dets, scale: scale, ox: 0, oy: 0, style: style,
                                    names: classNames,
                                    boxes: !(isSeg && segOverlay == .masks))
                }
                .frame(width: w, height: h)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding(.horizontal, 8)
    }

    // MARK: pipeline

    private func load(_ items: [PhotosPickerItem]) {
        guard !items.isEmpty else { return }
        phase = .loading
        progress = 0
        progressTotal = items.count
        errorText = ""
        Task {
            var loaded: [CGImage] = []
            var names: [String] = []
            var types: [String] = []
            // album-style names (IMG_xxxx) come from PHAssetResource via the
            // picker's itemIdentifier (photoLibrary: .shared() enables it)
            if PHPhotoLibrary.authorizationStatus(for: .readWrite) == .notDetermined {
                _ = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
            }
            for (idx, item) in items.enumerated() {
                var name = "IMG_\(idx + 1)"
                var type = item.supportedContentTypes.first?
                    .preferredFilenameExtension?.uppercased() ?? "-"
                if let id = item.itemIdentifier {
                    let assets = PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil)
                    if let asset = assets.firstObject,
                       let res = PHAssetResource.assetResources(for: asset).first {
                        let full = res.originalFilename          // e.g. IMG_4231.HEIC
                        name = (full as NSString).deletingPathExtension
                        let ext = (full as NSString).pathExtension
                        if !ext.isEmpty { type = ext.uppercased() }
                    }
                }
                let data = try? await item.loadTransferable(type: Data.self)
                if let data,
                   let src = CGImageSourceCreateWithData(data as CFData, nil),
                   let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                       kCGImageSourceCreateThumbnailWithTransform: true,   // EXIF
                       kCGImageSourceCreateThumbnailFromImageAlways: true,
                       kCGImageSourceThumbnailMaxPixelSize: 2048,
                   ] as CFDictionary) {
                    loaded.append(img)
                    names.append(name)
                    types.append(type)
                }
                let done = idx + 1
                await MainActor.run { progress = done }
            }
            let imgs = loaded, ns = names, ts = types
            await MainActor.run {
                images = imgs
                fileNames = ns
                fileTypes = ts
                results = Array(repeating: [], count: imgs.count)
                maskImages = Array(repeating: nil, count: imgs.count)
                page = 0
                viewMode = imgs.count > 1 ? .gallery : .pager
                phase = .idle
                if imgs.isEmpty { errorText = "no loadable photos" }
            }
            run()
        }
    }

    private func run() {
        guard !images.isEmpty, let m = selectedModel else { return }
        let mode = compute.mode
        let confNow = Float(conf), iouNow = CGFloat(iou)
        let imgs = images
        isRunning = true
        phase = .loadingModel
        progress = 0
        progressTotal = imgs.count
        errorText = ""
        perPre = []; perInf = []; perDec = []; perMask = []
        maskImages = Array(repeating: nil, count: imgs.count)
        Task.detached {
            do {
                let det = try Detector(modelURL: m.compiledURL(), compute: mode)
                await MainActor.run { phase = .inferring }
                var raws: [Detector.RawOutput] = []
                var allDets: [[Detection]] = []
                var sumPre = 0.0, sumInf = 0.0, sumDec = 0.0, sumMask = 0.0
                let t0 = CFAbsoluteTimeGetCurrent()
                // cache raw passes for slider retune only on modest batches:
                // 100 tensors x ~3MB is a jetsam invitation, and seg raws also
                // carry the proto tensor - roughly half the headroom
                let cacheRaws = imgs.count <= (det.isSegment ? 20 : 40)
                for (idx, img) in imgs.enumerated() {
                    let step: (Detector.RawOutput, [Detection], CGImage?,
                               Double, Double, Double, Double) =
                        try autoreleasepool {
                            let a = CFAbsoluteTimeGetCurrent()
                            let raw = try det.forward(img)
                            let b = CFAbsoluteTimeGetCurrent()
                            let d = det.decode(raw, conf: confNow, iou: iouNow)
                            let c = CFAbsoluteTimeGetCurrent()
                            // masks render for EVERY image, cached raw or not -
                            // beyond the cache cap they just stay at run-time
                            // conf/iou (same as the boxes there)
                            let mk = det.isSegment
                                ? det.maskOverlay(d, raw, maxSide: 1024) : nil
                            let e = CFAbsoluteTimeGetCurrent()
                            return (raw, d, mk, (b - a) * 1000 - raw.inferMs,
                                    raw.inferMs, (c - b) * 1000, (e - c) * 1000)
                        }
                    let (raw, d, mk, pPreV, pInfV, pDecV, pMaskV) = step
                    sumPre += pPreV
                    sumInf += pInfV
                    sumDec += pDecV
                    sumMask += pMaskV
                    if cacheRaws { raws.append(raw) }
                    allDets.append(d)
                    let done = idx + 1
                    let dSnapshot = d
                    await MainActor.run {
                        progress = done
                        if results.indices.contains(idx) { results[idx] = dSnapshot }
                        if maskImages.indices.contains(idx) { maskImages[idx] = mk }
                        perPre.append(pPreV); perInf.append(pInfV); perDec.append(pDecV)
                        perMask.append(pMaskV)
                    }
                }
                let wall = CFAbsoluteTimeGetCurrent() - t0
                let n = Double(imgs.count)
                let rawsF = raws, detsF = allDets
                await MainActor.run {
                    results = detsF
                    lastDet = det
                    lastRaws = rawsF
                    isSeg = det.isSegment
                    classNames = det.classNames
                    statPre = sumPre / n
                    statInf = sumInf / n
                    statDec = sumDec / n
                    statMask = sumMask / n
                    throughput = n / max(wall, 0.001)
                    wallSeconds = wall
                    phase = .idle
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    errorText = "ERROR: \(error)"
                    phase = .idle
                    isRunning = false
                }
            }
        }
    }

    /// Export annotated images (as rendered: mask layer + boxes per the current
    /// overlay mode and style) to the photo library, then pop the result.
    private func exportImages(_ idxs: [Int]) {
        guard !exporting, !idxs.isEmpty else { return }
        exporting = true
        let imgs = images, dets = results, masks = maskImages
        let drawMask = isSeg && segOverlay != .boxes
        let drawBoxes = !(isSeg && segOverlay == .masks)
        let names = classNames
        let kitStyle = style.kitStyle
        Task.detached(priority: .userInitiated) {
            var uis: [UIImage] = []
            for i in idxs where imgs.indices.contains(i) {
                autoreleasepool {
                    let cg = composeExport(imgs[i],
                                           dets: dets.indices.contains(i) ? dets[i] : [],
                                           mask: masks.indices.contains(i) ? masks[i] : nil,
                                           drawMask: drawMask, drawBoxes: drawBoxes,
                                           names: names, style: kitStyle)
                    uis.append(UIImage(cgImage: cg))
                }
            }
            if PHPhotoLibrary.authorizationStatus(for: .addOnly) == .notDetermined {
                _ = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            }
            var ok = false
            do {
                let toSave = uis
                try await PHPhotoLibrary.shared().performChanges {
                    for ui in toSave {
                        PHAssetChangeRequest.creationRequestForAsset(from: ui)
                    }
                }
                ok = !toSave.isEmpty
            } catch {
                ok = false
            }
            let n = uis.count
            await MainActor.run {
                exporting = false
                exportOK = ok
                exportToastGen += 1
                let gen = exportToastGen
                // App Store purchase-style feedback: the system success
                // notification haptic paired with the tick drawing in
                notifyHaptic.notificationOccurred(ok ? .success : .error)
                notifyHaptic.prepare()
                withAnimation(.spring(duration: 0.3)) {
                    exportMsg = ok
                        ? (n == 1 ? "Saved 1 image to Photos" : "Saved \(n) images to Photos")
                        : "Could not save to Photos"
                }
                Task {
                    try? await Task.sleep(nanoseconds: ok ? 1_600_000_000 : 2_400_000_000)
                    if exportToastGen == gen {
                        withAnimation(.easeOut(duration: 0.25)) { exportMsg = nil }
                    }
                }
            }
        }
    }

    /// Slider retune: decode-only over the cached forward passes. Boxes update
    /// synchronously (cheap); seg mask re-renders go off-main with a generation
    /// token so slider drags never hitch and stale renders are dropped.
    private func retune() {
        guard let det = lastDet, !lastRaws.isEmpty else { return }
        let confNow = Float(conf), iouNow = CGFloat(iou)
        let newDets = lastRaws.map { det.decode($0, conf: confNow, iou: iouNow) }
        results = newDets
        guard det.isSegment else { return }
        retuneGen += 1
        let gen = retuneGen, raws = lastRaws
        Task.detached(priority: .userInitiated) {
            var masks: [CGImage?] = []
            for (raw, d) in zip(raws, newDets) {
                autoreleasepool { masks.append(det.maskOverlay(d, raw, maxSide: 1024)) }
            }
            let m = masks
            await MainActor.run {
                guard gen == retuneGen else { return }
                for (i, mk) in m.enumerated() where maskImages.indices.contains(i) {
                    maskImages[i] = mk
                }
            }
        }
    }
}

// ExportToast / TickShape / CrossShape moved to StatsHUD.swift (shared with Bench).

/// Bake the on-screen annotation (mask composite under Kit-rendered boxes and
/// labels) into a standalone image for export.
private func composeExport(_ img: CGImage, dets: [Detection], mask: CGImage?,
                           drawMask: Bool, drawBoxes: Bool,
                           names: [String], style: YOLOMasterKit.BoxStyle) -> CGImage {
    var out = img
    if drawMask, let mask,
       let ctx = CGContext(data: nil, width: img.width, height: img.height,
                           bitsPerComponent: 8, bytesPerRow: 0,
                           space: CGColorSpaceCreateDeviceRGB(),
                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
        ctx.interpolationQuality = .high
        let full = CGRect(x: 0, y: 0, width: img.width, height: img.height)
        ctx.draw(img, in: full)
        ctx.draw(mask, in: full)
        if let o = ctx.makeImage() { out = o }
    }
    if !dets.isEmpty {
        out = annotate(out, dets, names: names, style: style,
                       label: .full, drawBoxes: drawBoxes) ?? out
    }
    return out
}

/// Photos-app zoom behavior via UIScrollView: pinch zoom anchored at the
/// fingers, one-finger pan ONLY while zoomed (1x swipes pass through to the
/// pager), double-tap zooms to the tapped point, and pinching below minimum
/// zoom (bounce region) exits to the gallery.
struct ZoomableScrollView<Content: View>: UIViewRepresentable {
    var onPinchExit: (() -> Void)?
    @ViewBuilder var content: Content

    func makeCoordinator() -> Coordinator {
        Coordinator(host: UIHostingController(rootView: content))
    }

    func makeUIView(context: Context) -> UIScrollView {
        let sv = UIScrollView()
        sv.minimumZoomScale = 1
        sv.maximumZoomScale = 5
        sv.bouncesZoom = true
        sv.showsHorizontalScrollIndicator = false
        sv.showsVerticalScrollIndicator = false
        sv.contentInsetAdjustmentBehavior = .never
        sv.backgroundColor = .clear
        sv.delegate = context.coordinator
        let hostView = context.coordinator.host.view!
        hostView.translatesAutoresizingMaskIntoConstraints = false
        hostView.backgroundColor = .clear
        sv.addSubview(hostView)
        NSLayoutConstraint.activate([
            hostView.leadingAnchor.constraint(equalTo: sv.contentLayoutGuide.leadingAnchor),
            hostView.trailingAnchor.constraint(equalTo: sv.contentLayoutGuide.trailingAnchor),
            hostView.topAnchor.constraint(equalTo: sv.contentLayoutGuide.topAnchor),
            hostView.bottomAnchor.constraint(equalTo: sv.contentLayoutGuide.bottomAnchor),
            hostView.widthAnchor.constraint(equalTo: sv.frameLayoutGuide.widthAnchor),
            hostView.heightAnchor.constraint(equalTo: sv.frameLayoutGuide.heightAnchor),
        ])
        let dt = UITapGestureRecognizer(target: context.coordinator,
                                        action: #selector(Coordinator.handleDoubleTap(_:)))
        dt.numberOfTapsRequired = 2
        sv.addGestureRecognizer(dt)
        return sv
    }

    func updateUIView(_ sv: UIScrollView, context: Context) {
        context.coordinator.host.rootView = content
        context.coordinator.onPinchExit = onPinchExit
    }

    final class Coordinator: NSObject, UIScrollViewDelegate {
        let host: UIHostingController<Content>
        var onPinchExit: (() -> Void)?
        private var exitFired = false
        init(host: UIHostingController<Content>) { self.host = host }

        func viewForZooming(in scrollView: UIScrollView) -> UIView? { host.view }

        func scrollViewDidZoom(_ sv: UIScrollView) {
            // bounce region below minimum zoom = the gallery-exit affordance;
            // only reachable when the pinch started at 1x
            if !exitFired, sv.zoomScale < sv.minimumZoomScale * 0.85,
               sv.pinchGestureRecognizer?.state == .changed {
                exitFired = true
                onPinchExit?()
            }
        }

        func scrollViewDidEndZooming(_ sv: UIScrollView, with view: UIView?,
                                     atScale scale: CGFloat) {
            exitFired = false
        }

        @objc func handleDoubleTap(_ gr: UITapGestureRecognizer) {
            guard let sv = gr.view as? UIScrollView else { return }
            if sv.zoomScale > sv.minimumZoomScale * 1.3 {
                sv.setZoomScale(sv.minimumZoomScale, animated: true)
            } else {
                let p = gr.location(in: host.view)
                let w = sv.bounds.width / 2.5, h = sv.bounds.height / 2.5
                sv.zoom(to: CGRect(x: p.x - w / 2, y: p.y - h / 2, width: w, height: h),
                        animated: true)
            }
        }
    }
}
