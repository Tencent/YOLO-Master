// OpenCV-side wrapper of the portable tracking core (mac/Sources/YOLOMasterCore/tracker_core.*):
// Detection <-> TrackInput / Track conversion, the sparse-optical-flow camera motion estimate of
// BoT-SORT (the core only applies the affine it is handed), and the annotated drawing.
#include "tracker.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#ifdef HAVE_GMC
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#endif

namespace yolomaster::track {

bool parse_tracker_kind(const std::string& s, TrackerConfig::Kind& out) {
    std::string t;
    for (char c : s) t += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (t == "botsort" || t == "bot-sort") { out = TrackerConfig::Kind::BotSort; return true; }
    if (t == "bytetrack" || t == "byte") { out = TrackerConfig::Kind::ByteTrack; return true; }
    return false;
}
const char* tracker_kind_name(TrackerConfig::Kind k) { return k == TrackerConfig::Kind::BotSort ? "botsort" : "bytetrack"; }
bool gmc_available() {
#ifdef HAVE_GMC
    return true;
#else
    return false;
#endif
}

static core::CoreConfig to_core(const TrackerConfig& c) {
    core::CoreConfig k;
    k.botsort = (c.kind == TrackerConfig::Kind::BotSort);
    k.track_high_thresh = c.track_high_thresh; k.track_low_thresh = c.track_low_thresh;
    k.new_track_thresh = c.new_track_thresh; k.match_thresh = c.match_thresh;
    k.track_buffer = c.track_buffer; k.fuse_score = c.fuse_score; k.fps = c.fps;
    return k;
}

struct Tracker::Impl {
    TrackerConfig cfg;
    core::CoreTracker core;
    bool gmc_last = false;
#ifdef HAVE_GMC
    cv::Mat prev_gray;
    std::vector<cv::Point2f> prev_pts;
    static constexpr int kDownscale = 2;
#endif

    explicit Impl(const TrackerConfig& c) : cfg(c), core(to_core(c)) {}

    // BoT-SORT sparse optical flow GMC: affine (rotation + uniform scale + translation) between the
    // previous and the current frame from tracked corners; identity when nothing can be estimated.
    // Called every frame the compensation is on, so prev_gray / prev_pts always track the last frame.
    bool estimate_motion(const cv::Mat& frame, core::Motion& m) {
        m = core::Motion{};
#ifdef HAVE_GMC
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, gray, cv::Size(), 1.0 / kDownscale, 1.0 / kDownscale, cv::INTER_LINEAR);
        bool ok = false;
        if (!prev_gray.empty() && prev_gray.size() == gray.size() && prev_pts.size() >= 8) {
            std::vector<cv::Point2f> cur;
            std::vector<unsigned char> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, cur, status, err);
            std::vector<cv::Point2f> a, b;
            for (size_t i = 0; i < cur.size(); ++i) if (status[i]) { a.push_back(prev_pts[i]); b.push_back(cur[i]); }
            if (a.size() >= 8) {
                std::vector<unsigned char> inl;
                const cv::Mat H = cv::estimateAffinePartial2D(a, b, inl, cv::RANSAC);
                if (!H.empty() && H.rows == 2 && H.cols == 3) {
                    m.R[0][0] = H.at<double>(0, 0); m.R[0][1] = H.at<double>(0, 1);
                    m.R[1][0] = H.at<double>(1, 0); m.R[1][1] = H.at<double>(1, 1);
                    m.tx = H.at<double>(0, 2) * kDownscale; m.ty = H.at<double>(1, 2) * kDownscale;
                    ok = true;
                }
            }
        }
        prev_gray = gray;
        prev_pts.clear();
        cv::goodFeaturesToTrack(gray, prev_pts, 1000, 0.01, 1, cv::noArray(), 3);
        return ok;
#else
        (void)frame;
        return false;
#endif
    }

    std::vector<Track> update(const std::vector<Detection>& dets, const cv::Mat* frame) {
        std::vector<core::TrackInput> in;
        in.reserve(dets.size());
        for (const auto& d : dets) {
            core::TrackInput t;
            t.box = core::Box{d.box.x, d.box.y, d.box.width, d.box.height};
            t.conf = d.conf; t.class_id = d.class_id; t.mask_coeffs = d.mask_coeffs;
            in.push_back(std::move(t));
        }
        core::Motion motion;
        gmc_last = false;
        if (cfg.kind == TrackerConfig::Kind::BotSort && cfg.gmc && frame && !frame->empty())
            gmc_last = estimate_motion(*frame, motion);
        std::vector<core::CoreTrack> ct = core.update(in, gmc_last ? &motion : nullptr);
        std::vector<Track> out;
        out.reserve(ct.size());
        for (auto& t : ct) {
            Track o;
            o.id = t.id; o.class_id = t.class_id; o.conf = t.conf;
            o.box = cv::Rect2f(t.box.x, t.box.y, t.box.width, t.box.height);
            o.age = t.age; o.hits = t.hits; o.time_since_update = t.time_since_update;
            o.state = t.state; o.mask_coeffs = std::move(t.mask_coeffs); o.det_index = t.det_index;
            out.push_back(std::move(o));
        }
        return out;
    }
};

Tracker::Tracker(const TrackerConfig& cfg) : impl_(std::make_unique<Impl>(cfg)) {}
Tracker::~Tracker() = default;
std::vector<Track> Tracker::update(const std::vector<Detection>& dets, const cv::Mat* frame) { return impl_->update(dets, frame); }
void Tracker::reset() { const TrackerConfig c = impl_->cfg; impl_ = std::make_unique<Impl>(c); }
int Tracker::frame_count() const { return impl_->core.frame_count(); }
const TrackerConfig& Tracker::config() const { return impl_->cfg; }
bool Tracker::gmc_active() const { return impl_->gmc_last; }

void draw_tracks(cv::Mat& img, const std::vector<Track>& tracks, const Config& cfg) {
    for (const auto& t : tracks) {
        const int k = t.id;
        const cv::Scalar col((k * 37) % 255, (k * 91) % 255, (k * 173) % 255);
        cv::rectangle(img, t.box, col, 2);
        const std::string name = t.class_id < cfg.num_classes() ? cfg.class_names[t.class_id] : std::to_string(t.class_id);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "#%d %s %.2f", t.id, name.c_str(), t.conf);
        int base = 0;
        const cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        const cv::Point org(static_cast<int>(t.box.x), std::max(0, static_cast<int>(t.box.y) - 4));
        cv::rectangle(img, cv::Rect(org.x, org.y - ts.height - 4, ts.width + 4, ts.height + 6), col, cv::FILLED);
        cv::putText(img, buf, cv::Point(org.x + 2, org.y - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

} // namespace yolomaster::track
