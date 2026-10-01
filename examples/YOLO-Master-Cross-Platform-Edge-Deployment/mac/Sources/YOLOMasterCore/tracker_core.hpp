// Multi-object tracking core: BoT-SORT (Kalman on xywh, ByteTrack two-stage association, camera
// motion applied from a caller-supplied affine) and plain ByteTrack (Kalman on xyah). No appearance
// model: association is IoU (optionally fused with the detection score), the tracker-only
// configuration of ultralytics' botsort.yaml / bytetrack.yaml defaults.
//
// Portable core: plain C++17, no OpenCV. The camera motion itself (sparse optical flow on the
// Linux runtime, Vision on macOS) is estimated by the host and handed in as a Motion per frame;
// cpp/include/tracker.hpp wraps this with the OpenCV GMC and cv::Rect2f types. Shared verbatim
// between the CMake build (target yolomaster_ccore) and the Swift package (YOLOMasterCore).
#pragma once
#include <memory>
#include <vector>

namespace yolomaster::track {

enum class TrackState { New, Tracked, Lost, Removed };

namespace core {

struct Box { float x = 0, y = 0, width = 0, height = 0; };   // top-left + size, original-image px

struct TrackInput {                      // one post-NMS detection
    Box box;
    float conf = 0.f;
    int class_id = 0;
    std::vector<float> mask_coeffs;      // carried through to the track (seg models); may be empty
};

struct Motion { double R[2][2] = {{1, 0}, {0, 1}}; double tx = 0, ty = 0; };   // previous -> current frame

struct CoreConfig {
    bool botsort = true;                 // true: xywh Kalman (BoT-SORT); false: xyah (ByteTrack)
    float track_high_thresh = 0.25f;     // first association
    float track_low_thresh  = 0.10f;     // second association (low-score detections)
    float new_track_thresh  = 0.25f;     // start a track from an unmatched detection
    float match_thresh      = 0.80f;     // IoU cost gate of the first association
    int   track_buffer      = 30;        // frames a lost track is kept (scaled by fps / 30)
    bool  fuse_score        = true;      // cost = 1 - iou * score in the first association
    double fps              = 30.0;
};

struct CoreTrack {
    int id = 0;
    int class_id = 0;
    float conf = 0.f;
    Box box;                             // current estimate in original-image px
    int age = 0;                         // frames since activation
    int hits = 0;                        // matched detections
    int time_since_update = 0;
    TrackState state = TrackState::New;
    std::vector<float> mask_coeffs;      // from the matched detection (kept while coasting)
    int det_index = -1;                  // index into the update() input (-1 = coasting)
};

class CoreTracker {
public:
    explicit CoreTracker(const CoreConfig& cfg);
    ~CoreTracker();
    CoreTracker(const CoreTracker&) = delete;
    CoreTracker& operator=(const CoreTracker&) = delete;
    // One frame: detections -> confirmed tracks. `motion` (BoT-SORT only) rotates / translates every
    // predicted state before association; nullptr = no compensation this frame.
    std::vector<CoreTrack> update(const std::vector<TrackInput>& dets, const Motion* motion = nullptr);
    void reset();
    int frame_count() const;
    const CoreConfig& config() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

double box_iou(const Box& a, const Box& b);

} // namespace core
} // namespace yolomaster::track
