// Multi-object tracking over the portable core (BoT-SORT / ByteTrack, the Linux runtime's
// tracker byte for byte). Feed post-NMS detections one frame at a time, in order; get the same
// detections back tagged with track ids (coasting tracks come back as detections of their own,
// score = last matched score, no mask coefficients).
//
// BoT-SORT's camera motion compensation is a caller concern: hand `update` the affine that maps
// the previous frame onto the current one (`CameraMotion`, estimated on macOS with Vision, see
// Motion.swift in a later slice); nil = no compensation this frame, which is plain ByteTrack
// association on an xywh Kalman filter.
import Foundation
import CoreGraphics
import YOLOMasterCore

public enum TrackerKind: String, CaseIterable, Sendable {
    case botSort = "botsort", byteTrack = "bytetrack"
}

/// Affine (rotation / scale 2x2 + translation) from the previous frame to the current one, in
/// original-image pixels. The identity means "the camera did not move".
public struct CameraMotion: Sendable {
    public var r00: Double = 1, r01: Double = 0, r10: Double = 0, r11: Double = 1, tx: Double = 0, ty: Double = 0
    public init() {}
    public init(r00: Double, r01: Double, r10: Double, r11: Double, tx: Double, ty: Double) {
        self.r00 = r00; self.r01 = r01; self.r10 = r10; self.r11 = r11; self.tx = tx; self.ty = ty
    }
    public static let identity = CameraMotion()
}

/// ultralytics botsort.yaml / bytetrack.yaml defaults (tracker-only, no ReID).
public struct TrackerConfig: Sendable {
    public var kind: TrackerKind = .botSort
    public var trackHighThresh: Float = 0.25     // first association
    public var trackLowThresh: Float = 0.10      // second association (low-score detections)
    public var newTrackThresh: Float = 0.25      // start a track from an unmatched detection
    public var matchThresh: Float = 0.80         // IoU cost gate of the first association
    public var trackBuffer: Int = 30             // frames a lost track is kept (scaled by fps / 30)
    public var fuseScore: Bool = true            // cost = 1 - iou * score in the first association
    public var fps: Double = 30
    public init(kind: TrackerKind = .botSort, trackBuffer: Int = 30, fps: Double = 30) {
        self.kind = kind; self.trackBuffer = trackBuffer; self.fps = fps
    }
    /// Detector confidence floor to run at when tracking: the second association wants the low-score
    /// candidates the detector would otherwise drop (the Linux CLI does the same).
    public var detectorFloor: Float { trackLowThresh }
}

public final class Tracker {
    public let config: TrackerConfig
    private let handle: OpaquePointer

    public init(_ config: TrackerConfig) {
        self.config = config
        var c = YmTrackerConfig()
        ym_tracker_default_config(&c)
        c.botsort = config.kind == .botSort ? 1 : 0
        c.track_high_thresh = config.trackHighThresh; c.track_low_thresh = config.trackLowThresh
        c.new_track_thresh = config.newTrackThresh; c.match_thresh = config.matchThresh
        c.track_buffer = Int32(config.trackBuffer); c.fuse_score = config.fuseScore ? 1 : 0; c.fps = config.fps
        handle = ym_tracker_new(&c)
    }
    public convenience init(kind: TrackerKind, fps: Double = 30, trackBuffer: Int = 30) {
        self.init(TrackerConfig(kind: kind, trackBuffer: trackBuffer, fps: fps))
    }
    deinit { ym_tracker_free(handle) }

    public var frameCount: Int { Int(ym_tracker_frame_count(handle)) }
    public func reset() { ym_tracker_reset(handle) }

    /// One frame. Returns the confirmed tracks as detections carrying `trackId`; a track matched to an
    /// input detection keeps that detection's mask coefficients, a coasting one has none.
    public func update(_ dets: [Detection], motion: CameraMotion? = nil) -> [Detection] {
        let inputs = dets.map { d in
            YmTrackInput(box: YmBox(x: Float(d.rect.minX), y: Float(d.rect.minY),
                                    width: Float(d.rect.width), height: Float(d.rect.height)),
                         conf: d.score, class_id: Int32(d.cls), mask_coeffs: nil, n_mask_coeffs: 0)
        }
        var out: UnsafePointer<YmTrack>? = nil
        var n: Int32 = 0
        if let m = motion {
            var ym = YmMotion(r00: m.r00, r01: m.r01, r10: m.r10, r11: m.r11, tx: m.tx, ty: m.ty)
            inputs.withUnsafeBufferPointer { ym_tracker_update(handle, $0.baseAddress, Int32(dets.count), &ym, &out, &n) }
        } else {
            inputs.withUnsafeBufferPointer { ym_tracker_update(handle, $0.baseAddress, Int32(dets.count), nil, &out, &n) }
        }
        guard let o = out, n > 0 else { return [] }
        return (0..<Int(n)).map { i in
            let t = o[i]
            let rect = CGRect(x: CGFloat(t.box.x), y: CGFloat(t.box.y), width: CGFloat(t.box.width), height: CGFloat(t.box.height))
            let coeffs = t.det_index >= 0 && Int(t.det_index) < dets.count ? dets[Int(t.det_index)].maskCoeffs : []
            return Detection(cls: Int(t.class_id), score: t.conf, rect: rect, maskCoeffs: coeffs, trackId: Int(t.id))
        }
    }
}

/// Estimates the camera motion between consecutive frames for BoT-SORT. Called once per frame in
/// order; returns nil when nothing could be estimated (the tracker then skips compensation).
public protocol CameraMotionEstimator: AnyObject {
    func estimate(_ frame: CGImage) -> CameraMotion?
    func reset()
}
