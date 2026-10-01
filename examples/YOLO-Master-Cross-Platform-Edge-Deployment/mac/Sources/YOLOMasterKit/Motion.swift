// Camera motion for BoT-SORT on Apple platforms, estimated with Vision instead of the Linux
// runtime's sparse optical flow + RANSAC partial affine (cpp/src/tracker.cpp estimate_motion).
//
// VNTranslationalImageRegistrationRequest registers the previous frame (floating) onto the current
// one (reference) and returns the pixel-space translation; the rotation / scale block stays the
// identity. That is the translational subset of what estimateAffinePartial2D fits: exact for pans
// and handheld drift, an approximation when the camera rolls or zooms between two frames. Vision's
// pixel space has a bottom-left origin (the y component is negated to the tracker's top-down
// convention) and the translation is quantized to whole pixels of the registered frames, so frames
// are registered at up to `maxSide` on the long side (verified on the diagonal-pan clip of the Mac
// battery: tx = +3, ty = +2 per frame come back as +3, +2).
// YM_MOTION_DEBUG=1 prints every estimate to stderr (used to verify the axis convention on the
// synthetic diagonal-pan clip of the Mac test script).
import Foundation
import CoreGraphics
import Vision

public final class VisionCameraMotion: CameraMotionEstimator {
    private var previous: CGImage? = nil
    private var previousScale: CGFloat = 1
    private let maxSide: CGFloat
    private let debug = ProcessInfo.processInfo.environment["YM_MOTION_DEBUG"] != nil
    public private(set) var frames = 0, estimated = 0

    /// `maxSide`: frames larger than this on their long side are registered downscaled.
    public init(maxSide: CGFloat = 1280) { self.maxSide = maxSide }

    public func reset() { previous = nil; frames = 0; estimated = 0 }

    public func estimate(_ frame: CGImage) -> CameraMotion? {
        frames += 1
        let scale = min(1, maxSide / CGFloat(max(frame.width, frame.height)))
        let small = scale < 1 ? resizeExact(frame, Int((CGFloat(frame.width) * scale).rounded()), Int((CGFloat(frame.height) * scale).rounded())) : frame
        defer { previous = small; previousScale = scale }
        guard let prev = previous, prev.width == small.width, prev.height == small.height else { return nil }
        let request = VNTranslationalImageRegistrationRequest(targetedCGImage: prev, options: [:])
        let handler = VNImageRequestHandler(cgImage: small, options: [:])
        guard (try? handler.perform([request])) != nil, let obs = request.results?.first else { return nil }
        let t = obs.alignmentTransform            // pixel space of the registered (downscaled) frames
        let tx = Double(t.tx) / Double(scale), ty = -Double(t.ty) / Double(scale)   // y-up -> top-down
        if debug { FileHandle.standardError.write("[motion] frame \(frames) tx=\(tx) ty=\(ty)\n".data(using: .utf8)!) }
        // reject wild solutions (a registration failure returns huge or NaN offsets)
        guard tx.isFinite, ty.isFinite, abs(tx) < Double(frame.width) / 2, abs(ty) < Double(frame.height) / 2 else { return nil }
        estimated += 1
        return CameraMotion(r00: 1, r01: 0, r10: 0, r11: 1, tx: tx, ty: ty)
    }
}
