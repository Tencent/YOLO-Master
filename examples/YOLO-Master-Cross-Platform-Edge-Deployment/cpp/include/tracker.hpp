// Multi-object tracking on top of the detector output: BoT-SORT (Kalman on xywh, ByteTrack
// two-stage association, sparse-optical-flow camera motion compensation) and plain ByteTrack
// (Kalman on xyah, no compensation). No appearance model: association is IoU (optionally fused
// with the detection score), exactly the tracker-only configuration of ultralytics' botsort.yaml
// and bytetrack.yaml defaults, so ids are comparable with `yolo track`.
//
// The Kalman / association core is the portable mac/Sources/YOLOMasterCore/tracker_core.* (shared
// with the Swift package); this header adds the OpenCV types, the optical-flow camera motion
// estimate and the drawing.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "tracker_core.hpp"
#include "yolomaster.hpp"

namespace yolomaster::track {

struct Track {
    int id = 0;
    int class_id = 0;
    float conf = 0.f;
    cv::Rect2f box;            // current estimate in original-image px
    int age = 0;               // frames since activation
    int hits = 0;              // matched detections
    int time_since_update = 0;
    TrackState state = TrackState::New;
    std::vector<float> mask_coeffs;   // carried from the matched detection (seg models)
    int det_index = -1;               // index into the update() input this track was matched to (-1 = coasting)
};

struct TrackerConfig {
    enum class Kind { BotSort, ByteTrack };
    Kind kind = Kind::BotSort;
    float track_high_thresh = 0.25f;   // first association
    float track_low_thresh  = 0.10f;   // second association (low-score detections)
    float new_track_thresh  = 0.25f;   // start a track from an unmatched detection
    float match_thresh      = 0.80f;   // IoU cost gate of the first association
    int   track_buffer      = 30;      // frames a lost track is kept (scaled by fps / 30)
    bool  fuse_score        = true;    // cost = 1 - iou * score in the first association
    bool  gmc               = true;    // BoT-SORT camera motion compensation (needs a frame + HAVE_GMC)
    double fps              = 30.0;
};
bool parse_tracker_kind(const std::string& s, TrackerConfig::Kind& out);   // "botsort" | "bytetrack"
const char* tracker_kind_name(TrackerConfig::Kind k);
bool gmc_available();   // compiled with OpenCV video + calib3d

class Tracker {
public:
    explicit Tracker(const TrackerConfig& cfg);
    ~Tracker();
    Tracker(const Tracker&) = delete;
    Tracker& operator=(const Tracker&) = delete;
    // One frame: detections (post-NMS, original px) -> confirmed tracks. `frame` (BGR) enables the
    // camera motion compensation of BoT-SORT; pass nullptr to skip it.
    std::vector<Track> update(const std::vector<Detection>& dets, const cv::Mat* frame = nullptr);
    void reset();
    int frame_count() const;
    const TrackerConfig& config() const;
    bool gmc_active() const;   // compensation actually applied on the last frame
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Draw boxes with "id:class conf" labels (same palette rule as draw()).
void draw_tracks(cv::Mat& img, const std::vector<Track>& tracks, const Config& cfg);

} // namespace yolomaster::track
