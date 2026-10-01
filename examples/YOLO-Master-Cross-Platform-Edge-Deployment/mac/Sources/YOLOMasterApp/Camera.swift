// Live-camera real-time inference. Latency-first design:
//   • AVCaptureVideoPreviewLayer shows the raw camera feed (hardware, zero-latency, always smooth).
//   • AVCaptureVideoDataOutput(alwaysDiscardsLateVideoFrames = true) delivers frames on a private
//     serial queue; while we're inferring, AVFoundation DROPS the frames that pile up and hands us
//     only the next fresh one - so we always process the newest frame and never accrue a backlog.
//   • A SwiftUI Canvas overlays boxes from the latest inference (lags the live feed by ~one inference
//     period, imperceptibly). conf/iou/style/label are read live - no re-inference to tune them.
import SwiftUI
import AppKit
import AVFoundation
import CoreML
import CoreGraphics
import CoreVideo
import QuartzCore
@preconcurrency import YOLOMasterKit   // Detector/RawOutput aren't Sendable; hopped to main safely

// ---------- capture + inference driver ----------
final class CameraController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {   // guarded by camQueue + main-hops
    let session = AVCaptureSession()
    // Raw per-frame output (pre-NMS). CameraStage applies conf/iou/overlay reactively via SwiftUI props
    // - so tuning is instant and never routed through a side channel that could miss updates.
    @Published var candidates: [Detection] = []     // pre-NMS, camera-pixel coords, confFloor 0.05
    @Published var lastRaw: Detector.RawOutput?     // this frame's proto/geometry (for seg masks)
    @Published var isSegment = false                // active model is segmentation
    @Published var frameSize: CGSize = .zero        // camera frame pixel dims (for overlay mapping)
    @Published var latencyMs: Double = 0            // per-frame compute latency (EMA)
    @Published var fps: Double = 0                  // achieved throughput (EMA)
    @Published var running = false
    @Published var errorMsg: String?
    @Published var detNames: [String] = []          // class names of the active model

    private let camQueue = DispatchQueue(label: "com.yolomaster.camera", qos: .userInteractive)
    private let output = AVCaptureVideoDataOutput()
    private var detector: Detector?                 // inference - touched only on camQueue
    private var maskDet: Detector?                  // same model, used for main-thread mask compositing (read-only math)
    private var configured = false
    private var mirrored = true                     // selfie mirror (applied to the data-output connection)
    private var lastT = 0.0, latEMA = 0.0, fpsEMA = 0.0

    /// Toggle selfie mirroring live. Mirrors the delivered frames (so inference/overlay follow) to match
    /// the preview layer's mirroring, which the view updates in lockstep.
    func setMirrored(_ on: Bool) {
        camQueue.async { [weak self] in
            guard let self else { return }
            self.mirrored = on
            guard let conn = self.output.connection(with: .video), conn.isVideoMirroringSupported else { return }
            self.session.beginConfiguration()
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = on
            self.session.commitConfiguration()
        }
    }

    /// Compose the seg mask overlay for `dets` from the latest frame's proto. Called on main from the
    /// view (read-only proto math on an immutable RawOutput snapshot - no model call, safe off camQueue).
    func makeMask(_ dets: [Detection]) -> CGImage? {
        guard let det = maskDet, let raw = lastRaw else { return nil }
        return det.maskOverlay(dets, raw)
    }

    /// Begin streaming with `det`. Requests camera permission + configures the session on first use.
    func start(detector det: Detector) {
        DispatchQueue.main.async { self.detNames = det.classNames; self.isSegment = det.isSegment; self.maskDet = det }
        camQueue.async { [weak self] in
            guard let self else { return }
            self.detector = det
            self.configureIfNeeded { ok in
                guard ok else { return }
                if !self.session.isRunning { self.session.startRunning() }
                DispatchQueue.main.async { self.running = true; self.errorMsg = nil }
            }
        }
    }
    /// Swap the model/compute/preprocess live without tearing down the session.
    func updateDetector(_ det: Detector) {
        DispatchQueue.main.async { self.detNames = det.classNames; self.isSegment = det.isSegment; self.maskDet = det }
        camQueue.async { [weak self] in self?.detector = det }
    }
    func stop() {
        camQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning { self.session.stopRunning() }
            self.lastT = 0; self.latEMA = 0; self.fpsEMA = 0
            DispatchQueue.main.async { self.running = false; self.candidates = []; self.lastRaw = nil; self.latencyMs = 0; self.fps = 0 }
        }
    }

    private func configureIfNeeded(_ done: @escaping (Bool) -> Void) {
        if configured { done(true); return }
        let build: () -> Bool = { [weak self] in
            guard let self else { return false }
            self.session.beginConfiguration()
            // 720p is plenty (we downscale to imgsz anyway) and cheaper to convert than 1080p -> lower latency
            self.session.sessionPreset = self.session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high
            guard let dev = AVCaptureDevice.default(for: .video),
                  let input = try? AVCaptureDeviceInput(device: dev), self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                DispatchQueue.main.async { self.errorMsg = "No camera found." }
                return false
            }
            self.session.addInput(input)
            self.output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            self.output.alwaysDiscardsLateVideoFrames = true                 // drop stale frames -> no backlog
            self.output.setSampleBufferDelegate(self, queue: self.camQueue)
            guard self.session.canAddOutput(self.output) else { self.session.commitConfiguration(); return false }
            self.session.addOutput(self.output)
            // Mirror the delivered frames (selfie view). Inference + candidates + masks are then all in
            // mirrored coordinates, matching the mirrored preview layer - so the overlay stays aligned.
            if let conn = self.output.connection(with: .video), conn.isVideoMirroringSupported {
                conn.automaticallyAdjustsVideoMirroring = false
                conn.isVideoMirrored = self.mirrored
            }
            self.session.commitConfiguration()
            self.configured = true
            return true
        }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            done(build())
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                self?.camQueue.async {
                    guard granted else { DispatchQueue.main.async { self?.errorMsg = "Camera access denied." }; done(false); return }
                    done(build())
                }
            }
        default:
            DispatchQueue.main.async { self.errorMsg = "Camera access denied - enable it in System Settings ▸ Privacy & Security ▸ Camera." }
            done(false)
        }
    }

    // Called on camQueue. Runs synchronously here; AVFoundation drops frames that arrive meanwhile.
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let det = detector, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let t0 = CACurrentMediaTime()
        let dt = lastT > 0 ? t0 - lastT : 0
        lastT = t0
        do {
            let raw = try det.forward(pb)
            let cands = det.candidates(raw, confFloor: 0.05)   // pre-NMS; the view applies conf/iou/overlay live
            let compute = (CACurrentMediaTime() - t0) * 1000
            latEMA = latEMA == 0 ? compute : latEMA * 0.8 + compute * 0.2
            if dt > 0 { let f = 1.0 / dt; fpsEMA = fpsEMA == 0 ? f : fpsEMA * 0.8 + f * 0.2 }
            let seg = det.isSegment, w = raw.origW, h = raw.origH, lat = latEMA, f = fpsEMA
            DispatchQueue.main.async {
                self.candidates = cands; self.lastRaw = seg ? raw : nil
                self.frameSize = CGSize(width: w, height: h)
                self.latencyMs = lat; self.fps = f
            }
        } catch {
            DispatchQueue.main.async { self.errorMsg = "Inference error: \(error.localizedDescription)" }
        }
    }
}

// ---------- live preview layer (un-mirrored so boxes align with the un-mirrored data output) ----------
final class CameraPreviewNSView: NSView {
    let previewLayer = AVCaptureVideoPreviewLayer()
    override func makeBackingLayer() -> CALayer { previewLayer.videoGravity = .resizeAspect; return previewLayer }
    override func layout() { super.layout(); disableHardwareMirror() }
    // Keep the preview's OWN connection un-mirrored. The visual selfie flip is applied in SwiftUI
    // (.scaleEffect on the view) so it works even on cameras whose preview connection doesn't support
    // isVideoMirrored - the exact case that left the preview un-mirrored while the data output/overlay
    // was mirrored. Retry until the connection exists (it's nil right after the session attaches).
    private func disableHardwareMirror(retries: Int = 15) {
        if let c = previewLayer.connection {
            if c.isVideoMirroringSupported { c.automaticallyAdjustsVideoMirroring = false; c.isVideoMirrored = false }
        } else if retries > 0 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) { [weak self] in self?.disableHardwareMirror(retries: retries - 1) }
        }
    }
}
struct CameraPreviewView: NSViewRepresentable {
    let session: AVCaptureSession
    func makeNSView(context: Context) -> CameraPreviewNSView {
        let v = CameraPreviewNSView(); v.wantsLayer = true; v.previewLayer.session = session; return v
    }
    func updateNSView(_ v: CameraPreviewNSView, context: Context) { v.previewLayer.session = session }
}

// ---------- lifecycle owner: builds the detector, starts/stops the session, isolates observation ----------
// Owns the CameraController via @StateObject so it's built once and only THIS subtree re-renders per
// frame - the parent (sidebar) never observes the high-frequency stats. Start/stop follow view presence.
struct LiveCameraView: View {
    let modelURL: URL?
    let compute: ComputeMode
    let preprocess: Detector.PreprocessMode
    var preprocDevice: PreprocDevice = .gpu          // Metal letterbox straight from the camera buffer (zero copy)
    let conf: Double, iou: Double
    let nmsMode: NMSMode, sigma: Double
    let maxDet: Int
    let overlay: SegOverlay
    let style: BoxStyle, label: LabelMode
    @Binding var isSegment: Bool                    // reported up so the sidebar can show the Overlay control
    @Binding var mirror: Bool                        // selfie mirror, toggled live from the stage
    @StateObject private var cam = CameraController()

    var body: some View {
        // conf/iou/overlay are plain props of CameraStage -> tuning is instantly reactive (no side channel)
        CameraStage(cam: cam, conf: conf, iou: iou, nmsMode: nmsMode, sigma: sigma, maxDet: maxDet, overlay: overlay, style: style, label: label, mirror: $mirror)
            .onAppear { cam.setMirrored(mirror); rebuild { cam.start(detector: $0) } }
            .onChange(of: mirror) { cam.setMirrored(mirror) }
            .onDisappear { cam.stop() }
            .onChange(of: modelURL) { rebuild { cam.updateDetector($0) } }       // hot-swap model
            .onChange(of: compute) { rebuild { cam.updateDetector($0) } }        // hot-swap compute unit
            .onChange(of: preprocess) { rebuild { cam.updateDetector($0) } }     // hot-swap letterbox/stretch
            .onChange(of: preprocDevice) { rebuild { cam.updateDetector($0) } }  // hot-swap CPU / Metal preprocessing
    }
    private func rebuild(_ apply: @escaping (Detector) -> Void) {
        guard let m = modelURL else { cam.errorMsg = "Choose a model first."; return }
        let comp = compute, pp = preprocess, pd = preprocDevice
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let d = try Detector(modelURL: m, compute: comp); d.preprocess = pp; d.preprocDevice = pd
                DispatchQueue.main.async { isSegment = d.isSegment; apply(d) }
            } catch {
                DispatchQueue.main.async { cam.errorMsg = "Could not load model: \(error.localizedDescription)" }
            }
        }
    }
}

// ---------- camera stage: live preview + box overlay + real-time FPS/latency HUD ----------
struct CameraStage: View {
    @ObservedObject var cam: CameraController
    let conf: Double, iou: Double
    let nmsMode: NMSMode, sigma: Double
    let maxDet: Int
    let overlay: SegOverlay
    let style: BoxStyle, label: LabelMode
    @Binding var mirror: Bool
    var body: some View {
        let dets = Detector.nms(cam.candidates, conf: Float(conf), iou: CGFloat(iou), maxDet: maxDet, mode: nmsMode, sigma: Float(sigma))   // live conf/iou/NMS-mode
        let masksOnly = cam.isSegment && overlay == .masks
        let drawBoxes = !masksOnly
        let mask: CGImage? = (cam.isSegment && overlay != .boxes) ? cam.makeMask(dets) : nil
        return ZStack(alignment: .topLeading) {
            CameraPreviewView(session: cam.session).scaleEffect(x: mirror ? -1 : 1, y: 1)   // selfie flip in SwiftUI (reliable)
            Canvas { ctx, size in
                let vid = cam.frameSize
                guard vid.width > 0, vid.height > 0 else { return }
                let scale = Swift.min(size.width / vid.width, size.height / vid.height)
                let dw = vid.width * scale
                let ox = (size.width - dw) / 2, oy = (size.height - vid.height * scale) / 2
                let lw = Swift.max(1.5, dw / 640 * 1.5)
                if let mask {                     // segmentation overlay, scaled to the video rect
                    ctx.draw(Image(decorative: mask, scale: 1),
                             in: CGRect(x: ox, y: oy, width: dw, height: vid.height * scale))
                }
                if !drawBoxes && label == .off { return }   // masks-only with no labels -> nothing more to draw
                for d in dets {
                    let color = overlayPalette[((d.cls % overlayPalette.count) + overlayPalette.count) % overlayPalette.count]
                    let r = CGRect(x: ox + d.rect.minX * scale, y: oy + d.rect.minY * scale,
                                   width: d.rect.width * scale, height: d.rect.height * scale)
                    let rp = Path(roundedRect: r, cornerRadius: 3)
                    if drawBoxes {
                    switch style {
                    case .solid:
                        ctx.stroke(rp, with: .color(color), lineWidth: lw * 1.2)
                    case .neon:
                        var g = ctx; g.addFilter(.shadow(color: color, radius: lw * 4))
                        g.stroke(rp, with: .color(color), lineWidth: lw * 1.3)
                    case .hud:
                        ctx.fill(Path(r), with: .color(color.opacity(0.08)))
                        ctx.stroke(Path(r), with: .color(color.opacity(0.35)), lineWidth: lw * 0.6)
                        let arm = Swift.min(Swift.min(r.width, r.height) * 0.28, lw * 22)
                        var br = Path()
                        br.move(to: CGPoint(x: r.minX + arm, y: r.minY)); br.addLine(to: CGPoint(x: r.minX, y: r.minY)); br.addLine(to: CGPoint(x: r.minX, y: r.minY + arm))
                        br.move(to: CGPoint(x: r.maxX - arm, y: r.minY)); br.addLine(to: CGPoint(x: r.maxX, y: r.minY)); br.addLine(to: CGPoint(x: r.maxX, y: r.minY + arm))
                        br.move(to: CGPoint(x: r.minX, y: r.maxY - arm)); br.addLine(to: CGPoint(x: r.minX, y: r.maxY)); br.addLine(to: CGPoint(x: r.minX + arm, y: r.maxY))
                        br.move(to: CGPoint(x: r.maxX, y: r.maxY - arm)); br.addLine(to: CGPoint(x: r.maxX, y: r.maxY)); br.addLine(to: CGPoint(x: r.maxX - arm, y: r.maxY))
                        ctx.stroke(br, with: .color(color), lineWidth: lw * 1.4)
                    }
                    }   // if drawBoxes
                    if label != .off {
                        let name = d.cls < cam.names.count ? cam.names[d.cls] : "class\(d.cls)"
                        let txt = label == .min ? name : "\(name) \(String(format: "%.2f", d.score))"
                        let resolved = ctx.resolve(Text(txt).font(.system(size: Swift.max(9, dw / 95))).bold().foregroundColor(.white))
                        let ts = resolved.measure(in: size)
                        let chip = CGRect(x: r.minX, y: Swift.max(oy, r.minY - ts.height - 4), width: ts.width + 6, height: ts.height + 4)
                        ctx.fill(Path(roundedRect: chip, cornerRadius: 3), with: .color(color.opacity(0.85)))
                        ctx.draw(resolved, at: CGPoint(x: chip.minX + 3, y: chip.minY + 2), anchor: .topLeading)
                    }
                }
            }
            hud(objects: dets.count)
            mirrorButton
            if !cam.running, let e = cam.errorMsg {
                VStack(spacing: 8) {
                    Image(systemName: "video.slash").font(.system(size: 40)).foregroundStyle(.tertiary)
                    Text(e).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 360)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if !cam.running {
                ProgressView("Starting camera…").frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private var mirrorButton: some View {
        Button { mirror.toggle() } label: {
            Label(mirror ? "Mirrored" : "Mirror", systemImage: "arrow.left.and.right")
                .font(.caption.weight(.semibold))
                .padding(.horizontal, 11).padding(.vertical, 7)
                .background(.black.opacity(mirror ? 0.6 : 0.35), in: Capsule())
                .foregroundStyle(mirror ? .white : .white.opacity(0.75))
        }
        .buttonStyle(.plain)
        .padding(14)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
    }

    private func hud(objects: Int) -> some View {
        HStack(spacing: 14) {
            stat("\(String(format: "%.1f", cam.fps))", "FPS", .green)
            stat("\(String(format: "%.0f", cam.latencyMs))", "ms/frame", .cyan)
            stat("\(objects)", "objects", .orange)
            Circle().fill(cam.running ? Color.red : Color.gray).frame(width: 8, height: 8)
                .overlay(Circle().stroke(.white.opacity(0.6), lineWidth: 1))
            Text(cam.running ? "LIVE" : "…").font(.caption2.weight(.bold)).foregroundStyle(.white.opacity(0.9))
        }
        .padding(.horizontal, 12).padding(.vertical, 7)
        .background(.black.opacity(0.55), in: Capsule())
        .padding(14)
    }
    private func stat(_ v: String, _ unit: String, _ c: Color) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 3) {
            Text(v).font(.system(size: 15, weight: .bold, design: .rounded)).foregroundStyle(c)
            Text(unit).font(.caption2).foregroundStyle(.white.opacity(0.7))
        }
    }
}

extension CameraController { var names: [String] { detNames } }
