// Live camera detection: preview layer + Canvas overlay + model/compute pickers.
// Detection loop: grab freshest frame -> Kit forward(pixelBuffer) -> decode -> draw.
import AudioToolbox
import AVFoundation
import Photos
import SwiftUI
import YOLOMasterKit

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    /// Tap callback with BOTH spaces: capture-device point (for focus) and
    /// layer point (for drawing the focus indicator).
    var onTap: ((CGPoint, CGPoint) -> Void)? = nil
    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        let tap = UITapGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.tapped(_:)))
        v.addGestureRecognizer(tap)
        return v
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {
        context.coordinator.parent = self
    }
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    final class Coordinator: NSObject {
        var parent: CameraPreview
        init(_ p: CameraPreview) { parent = p }
        @objc func tapped(_ gr: UITapGestureRecognizer) {
            guard let v = gr.view as? PreviewView else { return }
            let lp = gr.location(in: v)
            let dp = v.videoPreviewLayer.captureDevicePointConverted(fromLayerPoint: lp)
            parent.onTap?(dp, lp)
        }
    }
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

struct LiveView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var camera = CameraController()
    @State private var models: [BundledModel] = []
    @State private var initializing = true    // first-launch discovery/compile in flight
    @State private var selectedModel: BundledModel?
    @State private var compute: ComputeChoice = ComputeChoice.deviceDefault
    @AppStorage("allowCPU") private var allowCPU = false
    @State private var detections: [Detection] = []
    @State private var frameSize = CGSize(width: 720, height: 1280)
    @State private var statPre: Double = 0
    @State private var statInf: Double = 0
    @State private var statDec: Double = 0
    @State private var statMask: Double = 0
    @State private var statHz: Double = 0
    @State private var statDets = 0
    @State private var statLoopHz: Double = 0   // ACTUAL label refresh rate
    @State private var statCamHz: Double = 0    // sensor->app frame delivery rate
    @State private var statAgeMs: Double = 0    // frame age at delivery (pipeline latency)
    @State private var maskImage: CGImage?          // seg: per-frame composite overlay
    @State private var isSegModel = false           // set once the running Detector loads
    @State private var classNames: [String] = cocoNames
    @State private var showTuning = false
    @State private var showHUD = true
    @State private var style: BoxStyle = .chip
    @StateObject private var tuning = Tuning()
    @State private var zoomBase: CGFloat = 1  // display zoom factor at gesture start
    @State private var focusPoint: CGPoint?   // last tap, preview-layer coords
    @State private var focusVisible = false
    @State private var shutterFlash = false   // brief black blink on capture
    @State private var capturing = false      // capture->compose->save in flight
    @State private var lensExpanded = false   // zoom chip expanded to lens buttons
    @State private var lensCollapse: Task<Void, Never>?
    @State private var lensRect: CGRect = .zero   // global frame of the lens picker
    private let haptic = UIImpactFeedbackGenerator(style: .medium)
    private let lightHaptic = UIImpactFeedbackGenerator(style: .light)
    private let heavyHaptic = UIImpactFeedbackGenerator(style: .heavy)
    @State private var longPressFired = false  // swallow the tap that follows a long press
    @State private var loadingModel = false   // Detector init in flight (compile + unit load)
    @State private var running = false        // loop actually alive
    @State private var wantRun = true         // the user's intent - survives tab
                                              // switches and app backgrounding;
                                              // true at launch = cold-start auto-run
    @State private var loopTask: Task<Void, Never>?

    var body: some View {
        ZStack {
            if models.isEmpty {
                if initializing {
                    // first run after install: .mlpackage -> .mlmodelc compile
                    HStack(spacing: 10) {
                        ProgressView()
                        Text("Initializing...")
                    }
                    .font(.caption)
                    .padding(14)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                } else {
                    ContentUnavailableView("No models bundled",
                                           systemImage: "shippingbox",
                                           description: Text("Copy .mlpackage files into ios/Models/, re-run xcodegen, rebuild."))
                }
            } else if camera.authorized {
                // preview and overlay MUST share one coordinate space: both
                // full-screen, safe area ignored together, or boxes shear by
                // the notch/home-bar insets
                ZStack {
                    CameraPreview(session: camera.session) { dp, lp in
                        // dead zone: taps on or near the lens picker are zoom
                        // interaction, never focus selection (the preview is
                        // full-screen, so its coords equal global coords)
                        if lensRect.insetBy(dx: -28, dy: -28).contains(lp) { return }
                        camera.focus(atDevicePoint: dp)
                        focusPoint = lp
                        withAnimation(.easeOut(duration: 0.15)) { focusVisible = true }
                        Task {
                            try? await Task.sleep(nanoseconds: 900_000_000)
                            withAnimation(.easeOut(duration: 0.3)) { focusVisible = false }
                        }
                    }
                    overlay
                    if let fp = focusPoint, focusVisible {
                        RoundedRectangle(cornerRadius: 3)
                            .stroke(.yellow, lineWidth: 1.5)
                            .frame(width: 72, height: 72)
                            .position(fp)
                            .transition(.opacity)
                    }
                }
                .ignoresSafeArea()
                .gesture(
                    MagnifyGesture()
                        .onChanged { v in camera.setZoom(zoomBase * v.magnification) }
                        .onEnded { v in zoomBase = camera.setZoom(zoomBase * v.magnification) }
                )
            } else {
                ContentUnavailableView("Camera access required",
                                       systemImage: "camera.fill")
            }
            Color.black
                .opacity(shutterFlash ? 1 : 0)
                .ignoresSafeArea()
                .allowsHitTesting(false)
            if loadingModel {
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Loading the model to \(compute.rawValue)")
                }
                .font(.caption)
                .padding(14)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
            VStack {
                Spacer()
                if camera.authorized, !models.isEmpty {
                    lensControl
                        .padding(.bottom, 6)
                    Button { captureFrame() } label: {
                        ZStack {
                            Circle().stroke(.white.opacity(0.9), lineWidth: 3)
                                .frame(width: 58, height: 58)
                            if capturing {
                                ProgressView()
                                    .tint(.white)
                                    .scaleEffect(1.4)
                            } else {
                                Circle().fill(.white)
                                    .frame(width: 46, height: 46)
                            }
                        }
                    }
                    .buttonStyle(ShutterStyle())
                    .disabled(capturing)
                    .padding(.bottom, 8)
                }
                if showHUD {
                    StatsHUD(fps: statHz, pre: statPre, inf: statInf,
                             dec: statDec, dets: statDets,
                             active: running && !loadingModel,
                             extras: running && !loadingModel
                                 ? [("camera", String(format: "%.1f fps", statCamHz)),
                                    ("loop", String(format: "%.1f fps", statLoopHz)),
                                    ("frame age", String(format: "%.0f ms", statAgeMs))]
                                 : [],
                             mask: isSegModel ? statMask : nil)
                        .padding(.bottom, 20)
                }
            }
        }
        .safeAreaInset(edge: .top) {
            // same vertical anchor as the photo tab's top bar
            VStack(spacing: 8) {
                controls
                if showTuning {
                    TuningPanel(conf: $tuning.conf, iou: $tuning.iou, style: $style,
                                hudVisible: $showHUD,
                                segOverlay: isSegModel ? $tuning.segOverlay : nil)
                        .padding(.horizontal)
                }
            }
        }
        .onChange(of: allowCPU) { _, on in if !on && compute == .cpu { compute = ComputeChoice.deviceDefault } }
        .onAppear {
            if !allowCPU && compute == .cpu { compute = ComputeChoice.deviceDefault }
            camera.start()
            haptic.prepare()
            lightHaptic.prepare()
            heavyHaptic.prepare()
            // first performChanges spins up the photo-library connection (seconds
            // on a cold start) - warm it here instead of on the first shutter tap
            if PHPhotoLibrary.authorizationStatus(for: .addOnly) == .authorized {
                Task.detached(priority: .utility) {
                    try? await PHPhotoLibrary.shared().performChanges {}
                }
            }
            if models.isEmpty {
                // discovery compiles bundled packages on first launch - keep it
                // off the main thread, behind the Initializing card
                Task.detached(priority: .userInitiated) {
                    let found = BundledModel.discover()
                    await MainActor.run {
                        models = found
                        initializing = false
                        if selectedModel == nil {
                            selectedModel = BundledModel.preferred(in: found)
                        }
                        if wantRun { startLoop() }     // cold-start auto-run
                    }
                }
            } else if wantRun {
                startLoop()                            // auto-resume on tab return
            }
        }
        .onDisappear { suspendLoop(); camera.stop() }  // pause, intent preserved
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                camera.start()
                if wantRun { startLoop() }             // auto-resume from background
            } else {
                suspendLoop()
            }
        }
    }

    /// Camera-app style: whole numbers bare (1x, 2x), fractions one decimal (0.5x, 2.4x).
    private func zoomLabel(_ z: CGFloat) -> String {
        let r = (z * 10).rounded() / 10
        return r == r.rounded() ? "\(Int(r))x" : String(format: "%.1fx", r)
    }

    /// The native lens the current zoom is riding on (nearest stop at or below).
    private var activeLens: CGFloat {
        camera.lensFactors.last { camera.displayZoom >= $0 - 0.05 }
            ?? camera.lensFactors.first ?? 1
    }

    /// Collapsed: one chip with the current factor. Tapped: one button per
    /// native lens (camera-app style); collapses back after 3s of no input.
    private var lensControl: some View {
        HStack(spacing: 4) {
            if lensExpanded {
                ForEach(camera.lensFactors, id: \.self) { f in
                    let isActive = f == activeLens
                    Button {
                        zoomBase = camera.setZoom(f)
                        bumpLensTimer()
                    } label: {
                        // active button reads the LIVE zoom, others their stop
                        Text(isActive ? zoomLabel(camera.displayZoom) : zoomLabel(f))
                            .font(.caption.monospacedDigit().bold())
                            .foregroundStyle(isActive ? .yellow : .primary)
                            .frame(minWidth: 34)
                            .padding(.vertical, 6)
                    }
                }
            } else {
                Button {
                    withAnimation(.spring(duration: 0.25)) { lensExpanded = true }
                    bumpLensTimer()
                } label: {
                    Text(zoomLabel(camera.displayZoom))
                        .font(.caption.monospacedDigit().bold())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                }
            }
        }
        .padding(.horizontal, lensExpanded ? 8 : 0)
        .background(.ultraThinMaterial, in: Capsule())
        .buttonStyle(.plain)
        .background(GeometryReader { g in
            Color.clear
                .onAppear { lensRect = g.frame(in: .global) }
                .onChange(of: g.frame(in: .global)) { _, r in lensRect = r }
        })
    }

    /// Restart the 3s auto-collapse window.
    private func bumpLensTimer() {
        lensCollapse?.cancel()
        lensCollapse = Task {
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            guard !Task.isCancelled else { return }
            withAnimation(.spring(duration: 0.25)) { lensExpanded = false }
        }
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
            .disabled(wantRun)          // pause inference before switching (mac GUI rule)
            Picker("Compute", selection: $compute) {
                ForEach(ComputeChoice.available(allowCPU: allowCPU)) { c in
                    Text(c.rawValue).lineLimit(1).fixedSize().tag(c)
                }
            }
            .fixedSize()
            .disabled(wantRun)
            Button {
                lightHaptic.impactOccurred()
                lightHaptic.prepare()
                camera.setTorch(!camera.torchOn)
            } label: {
                // flashlight glyphs are taller than the other symbols - pin the
                // label box so every bordered button comes out the same height
                Image(systemName: camera.torchOn ? "flashlight.on.fill" : "flashlight.off.fill")
                    .font(.system(size: 15))
                    .frame(width: 22, height: 20)
            }
            .buttonStyle(.bordered)
            .tint(camera.torchOn ? .yellow : nil)
            Button {
                withAnimation { showTuning.toggle() }
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .frame(width: 22, height: 20)
            }
            .buttonStyle(.bordered)
            Button {
                if longPressFired { longPressFired = false; return }
                togglePlay(strong: false)
            } label: {
                Image(systemName: wantRun ? "pause.fill" : "play.fill")
                    .frame(minWidth: 44, minHeight: 20)
            }
            .buttonStyle(.borderedProminent)
            .tint(wantRun ? .red : .accentColor)
            .disabled(selectedModel == nil)
            .simultaneousGesture(
                LongPressGesture(minimumDuration: 0.4).onEnded { _ in
                    longPressFired = true
                    togglePlay(strong: true)
                    // if the finger lifts outside the button no tap follows -
                    // do not leave the swallow flag armed
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                        longPressFired = false
                    }
                }
            )
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)   // no top padding: aligns with the photo tab's bar
    }

    private var overlay: some View {
        GeometryReader { geo in
            Canvas(rendersAsynchronously: true) { ctx, size in
                // the preview layer is resizeAspectFill: one uniform scale (the
                // larger axis ratio), surplus cropped equally. Mirror exactly
                // that mapping or boxes shear toward the center.
                let scale = max(size.width / frameSize.width,
                                size.height / frameSize.height)
                let ox = (size.width - frameSize.width * scale) / 2
                let oy = (size.height - frameSize.height * scale) / 2
                // mask composite is camera-frame-sized: same rect as the frame
                // itself under aspect-fill, drawn beneath the boxes/labels
                if let mask = maskImage {
                    ctx.draw(Image(decorative: mask, scale: 1),
                             in: CGRect(x: ox, y: oy, width: frameSize.width * scale,
                                        height: frameSize.height * scale))
                }
                DetOverlay.draw(ctx, detections, scale: scale, ox: ox, oy: oy, style: style,
                                names: classNames,
                                boxes: !(isSegModel && tuning.segOverlay == .masks))
            }
        }
        .allowsHitTesting(false)
    }

    private func startLoop() {
        guard let model = selectedModel, !running else { return }
        running = true
        loadingModel = true
        let mode = compute.mode
        let tuningRef = tuning      // plain class ref: detached-safe, no wrapper
        let cameraRef = camera
        loopTask = Task.detached(priority: .userInitiated) {
            guard let url = try? model.compiledURL(),
                  let det = try? Detector(modelURL: url, compute: mode) else {
                await MainActor.run { running = false; loadingModel = false }
                return
            }
            let seg = det.isSegment
            let names = det.classNames
            await MainActor.run { isSegModel = seg; classNames = names; loadingModel = false }
            var uiTicks = 0
            var uiWindowStart = CFAbsoluteTimeGetCurrent()
            var uiHz = 0.0
            var lastFrameID: UInt64 = 0
            while !Task.isCancelled {
                guard let (pb, fid) = cameraRef.grabLatest() else {
                    try? await Task.sleep(nanoseconds: 20_000_000); continue
                }
                if fid == lastFrameID {                    // stale frame - don't reprocess
                    try? await Task.sleep(nanoseconds: 5_000_000); continue
                }
                lastFrameID = fid
                let confNow = Float(tuningRef.conf)
                let iouNow = CGFloat(tuningRef.iou)
                let t0 = CFAbsoluteTimeGetCurrent()
                // autoreleasepool per frame: without it, autoreleased CoreML/CG
                // transients accumulate on the cooperative thread until jetsam
                // kills the app (~40s in CPU mode, whose transients are largest)
                // masks only when wanted this frame: boxes-only mode skips the
                // whole SGEMM+composite, and the stage bar drops to ~0
                let segNow = seg && tuningRef.segOverlay != .boxes
                let frame: ([Detection], CGImage?, Double, Double, Double, Double)? = autoreleasepool {
                    guard let raw = try? det.forward(pb) else { return nil }
                    let t1 = CFAbsoluteTimeGetCurrent()
                    let d = det.decode(raw, conf: confNow, iou: iouNow)
                    let t2 = CFAbsoluteTimeGetCurrent()
                    // maxSide 640: quarter the composite pixels vs the 720x1280
                    // frame; the proto grid is 160px, so nothing visible is lost
                    let mask: CGImage? = segNow && !d.isEmpty
                        ? det.maskOverlay(Array(d.prefix(100)), raw, maxSide: 640) : nil
                    let t3 = CFAbsoluteTimeGetCurrent()
                    let fwd = (t1 - t0) * 1000
                    return (d, mask, fwd - raw.inferMs, raw.inferMs,
                            (t2 - t1) * 1000, (t3 - t2) * 1000)
                }
                guard let (dets, mask, preMS, infMS, decMS, maskMS) = frame else { continue }
                uiTicks += 1
                let nowT = CFAbsoluteTimeGetCurrent()
                if nowT - uiWindowStart >= 1.0 {
                    uiHz = Double(uiTicks) / (nowT - uiWindowStart)
                    uiTicks = 0; uiWindowStart = nowT
                }
                let camHz = cameraRef.deliveryRate()
                let ageMs = cameraRef.frameAgeMs()

                let w = CGFloat(CVPixelBufferGetWidth(pb))
                let h = CGFloat(CVPixelBufferGetHeight(pb))
                // fire-and-forget: NEVER block the inference loop on the main
                // thread - a busy render loop otherwise throttles detection to
                // its leftover scheduling gaps
                // model end-to-end capability, NOT the paced loop rate: this
                // can exceed the camera's 30 and peg the dial purple
                let e2eFPS = 1000.0 / max(preMS + infMS + decMS + maskMS, 0.1)
                Task { @MainActor in
                    guard running else { return }   // zombie frame after pause: drop
                    detections = dets
                    maskImage = mask
                    statPre = preMS
                    statInf = infMS
                    statDec = decMS
                    statMask = maskMS
                    statHz = e2eFPS
                    statLoopHz = uiHz
                    statCamHz = camHz
                    statAgeMs = ageMs
                    statDets = dets.count
                    frameSize = CGSize(width: w, height: h)
                }
                // pace to ~30FPS: cools the phone, nothing visible is faster anyway
                let spent = (CFAbsoluteTimeGetCurrent() - t0)   // full iteration cost
                if spent < 0.033 {
                    try? await Task.sleep(nanoseconds: UInt64((0.033 - spent) * 1e9))
                }
            }
        }
    }

    /// Shutter: sound + haptic + black blink, then a FULL-RESOLUTION still via
    /// AVCapturePhotoOutput, cropped to the video FOV so the live overlay maps
    /// onto it by pure scaling; mask composite + Kit-rendered boxes/labels are
    /// baked in off-main and the result saved to Photos. Falls back to the
    /// 720p video frame if the photo capture fails.
    private func captureFrame() {
        guard !capturing else { return }
        capturing = true
        AudioServicesPlaySystemSound(1108)                       // classic shutter
        haptic.impactOccurred()
        haptic.prepare()
        shutterFlash = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
            withAnimation(.easeOut(duration: 0.2)) { shutterFlash = false }
        }
        let dets = detections
        let mask = maskImage
        let drawMask = isSegModel && tuning.segOverlay != .boxes
        let drawBoxes = !(isSegModel && tuning.segOverlay == .masks)
        let names = classNames
        let fSize = frameSize
        let kitStyle = style.kitStyle
        let fallback = camera.grabLatest().flatMap { Detector.cgImage(from: $0.0) }
        let t0 = CFAbsoluteTimeGetCurrent()
        camera.capturePhoto { photo in
            Task.detached(priority: .userInitiated) {
                var base: CGImage? = nil
                var s: CGFloat = 1
                if let photo {
                    // photo sensor FOV is wider than the 16:9 video FOV the
                    // detections live in - crop to the centered video-aspect
                    // region, then overlay coords map by one scale factor
                    let pw = CGFloat(photo.width), ph = CGFloat(photo.height)
                    let va = fSize.width / fSize.height
                    var crop = CGRect(x: 0, y: 0, width: pw, height: ph)
                    if pw / ph > va {
                        let cw = (ph * va).rounded()
                        crop = CGRect(x: ((pw - cw) / 2).rounded(), y: 0, width: cw, height: ph)
                    } else if pw / ph < va {
                        let ch = (pw / va).rounded()
                        crop = CGRect(x: 0, y: ((ph - ch) / 2).rounded(), width: pw, height: ch)
                    }
                    if let c = photo.cropping(to: crop) {
                        base = c
                        s = crop.height / fSize.height
                    }
                }
                if base == nil { base = fallback; s = 1 }
                guard var img = base else {
                    await rearmShutter(t0)
                    return
                }
                let scaled = s == 1 ? dets : dets.map {
                    Detection(cls: $0.cls, score: $0.score,
                              rect: CGRect(x: $0.rect.minX * s, y: $0.rect.minY * s,
                                           width: $0.rect.width * s, height: $0.rect.height * s),
                              maskCoeffs: $0.maskCoeffs)
                }
                if drawMask, let mask,
                   let ctx = CGContext(data: nil, width: img.width, height: img.height,
                                       bitsPerComponent: 8, bytesPerRow: 0,
                                       space: CGColorSpaceCreateDeviceRGB(),
                                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
                    ctx.interpolationQuality = .high
                    let full = CGRect(x: 0, y: 0, width: img.width, height: img.height)
                    ctx.draw(img, in: full)
                    ctx.draw(mask, in: full)   // mask covers the video FOV = the cropped base
                    if let out = ctx.makeImage() { img = out }
                }
                if !scaled.isEmpty {
                    img = annotate(img, scaled, names: names, style: kitStyle,
                                   label: .full, drawBoxes: drawBoxes) ?? img
                }
                let ui = UIImage(cgImage: img)
                if PHPhotoLibrary.authorizationStatus(for: .addOnly) == .notDetermined {
                    _ = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
                }
                try? await PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.creationRequestForAsset(from: ui)
                }
                await rearmShutter(t0)
            }
        }
    }

    /// Re-enable the shutter no sooner than ~1s after the tap: the system
    /// shutter sound is about that long, and a fast save re-arming earlier lets
    /// rapid taps overlap sounds into a stutter.
    private func rearmShutter(_ t0: CFAbsoluteTime) async {
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        if elapsed < 1.0 {
            try? await Task.sleep(nanoseconds: UInt64((1.0 - elapsed) * 1e9))
        }
        await MainActor.run { capturing = false }
    }

    /// Stop the loop and WIPE the overlay; keeps `wantRun` untouched so
    /// lifecycle pauses auto-resume.
    private func suspendLoop() {
        running = false
        loadingModel = false
        loopTask?.cancel()
        loopTask = nil
        detections = []
        maskImage = nil          // same autowipe rule as the boxes
        // full HUD reset - only the thermal state stays (it is device-ambient,
        // not a property of the stopped run)
        statPre = 0
        statInf = 0
        statDec = 0
        statMask = 0
        statHz = 0
        statLoopHz = 0
        statCamHz = 0
        statAgeMs = 0
        statDets = 0
    }

    /// User-initiated stop (play/pause button).
    private func stopLoop() {
        suspendLoop()
    }

    /// Play/pause with haptics: medium on tap, heavy full-intensity on the
    /// force/long press path.
    private func togglePlay(strong: Bool) {
        if strong {
            heavyHaptic.impactOccurred(intensity: 1.0)
            heavyHaptic.prepare()
        } else {
            haptic.impactOccurred()
            haptic.prepare()
        }
        if wantRun { wantRun = false; stopLoop() }
        else { wantRun = true; startLoop() }
    }
}

/// Camera-app shutter feel: quick shrink while pressed.
struct ShutterStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.85 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}
