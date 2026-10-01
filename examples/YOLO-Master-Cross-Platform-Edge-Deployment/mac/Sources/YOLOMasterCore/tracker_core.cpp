#include "tracker_core.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace yolomaster::track::core {

// ---------------------------------------------------------------------------------------------
// Constant-velocity Kalman filter, 8 states: (x, y, a|w, h, vx, vy, va|vw, vh). BoT-SORT keeps the
// box as centre + width + height (xywh), ByteTrack as centre + aspect + height (xyah). Noise scales
// with the box height (and width for xywh), the usual std_weight_position / velocity constants.
// ---------------------------------------------------------------------------------------------
namespace {
constexpr int NS = 8, NM = 4;
using Mean = std::array<double, NS>;
using Cov  = std::array<double, NS * NS>;
constexpr double kStdPos = 1.0 / 20.0, kStdVel = 1.0 / 160.0;

struct KF {
    bool xywh = true;
    Mean mean{};
    Cov cov{};

    std::array<double, NS> std_pos_scale(const Mean& m) const {
        // per-state noise std, position block then velocity block
        const double h = m[3];
        const double w = xywh ? m[2] : m[3];
        std::array<double, NS> s{};
        s[0] = 2 * kStdPos * w; s[1] = 2 * kStdPos * h;
        s[2] = xywh ? 2 * kStdPos * w : 1e-2; s[3] = 2 * kStdPos * h;
        s[4] = 10 * kStdVel * w; s[5] = 10 * kStdVel * h;
        s[6] = xywh ? 10 * kStdVel * w : 1e-5; s[7] = 10 * kStdVel * h;
        return s;
    }
    void initiate(const std::array<double, NM>& z) {
        mean.fill(0.0);
        for (int i = 0; i < NM; ++i) mean[i] = z[i];
        cov.fill(0.0);
        const auto s = std_pos_scale(mean);
        for (int i = 0; i < NS; ++i) cov[i * NS + i] = s[i] * s[i];
    }
    void predict() {
        // process noise
        const double h = mean[3], w = xywh ? mean[2] : mean[3];
        std::array<double, NS> q{};
        q[0] = kStdPos * w; q[1] = kStdPos * h; q[2] = xywh ? kStdPos * w : 1e-2; q[3] = kStdPos * h;
        q[4] = kStdVel * w; q[5] = kStdVel * h; q[6] = xywh ? kStdVel * w : 1e-5; q[7] = kStdVel * h;
        // mean = F mean (F = I with dt=1 on the velocity coupling)
        for (int i = 0; i < NM; ++i) mean[i] += mean[i + NM];
        // cov = F P F^T + Q
        Cov p = cov;
        auto P = [&](int r, int c) -> double& { return p[r * NS + c]; };
        // F P: rows 0..3 += rows 4..7
        for (int i = 0; i < NM; ++i) for (int c = 0; c < NS; ++c) P(i, c) += P(i + NM, c);
        // (F P) F^T: cols 0..3 += cols 4..7
        for (int r = 0; r < NS; ++r) for (int i = 0; i < NM; ++i) P(r, i) += P(r, i + NM);
        for (int i = 0; i < NS; ++i) P(i, i) += q[i] * q[i];
        cov = p;
    }
    void update(const std::array<double, NM>& z) {
        // S = H P H^T + R  (H selects the first four states); K = P H^T S^-1
        const double h = mean[3], w = xywh ? mean[2] : mean[3];
        std::array<double, NM> r{};
        r[0] = kStdPos * w; r[1] = kStdPos * h; r[2] = xywh ? kStdPos * w : 1e-1; r[3] = kStdPos * h;
        double S[NM][NM];
        for (int i = 0; i < NM; ++i) for (int j = 0; j < NM; ++j) S[i][j] = cov[i * NS + j] + (i == j ? r[i] * r[i] : 0.0);
        // invert S (4x4) by Gauss-Jordan
        double inv[NM][NM] = {};
        for (int i = 0; i < NM; ++i) inv[i][i] = 1.0;
        for (int c = 0; c < NM; ++c) {
            int piv = c;
            for (int rr = c + 1; rr < NM; ++rr) if (std::fabs(S[rr][c]) > std::fabs(S[piv][c])) piv = rr;
            if (std::fabs(S[piv][c]) < 1e-12) return;   // degenerate: skip the update
            if (piv != c) { std::swap(S[piv], S[c]); std::swap(inv[piv], inv[c]); }
            const double d = S[c][c];
            for (int j = 0; j < NM; ++j) { S[c][j] /= d; inv[c][j] /= d; }
            for (int rr = 0; rr < NM; ++rr) if (rr != c) {
                const double f = S[rr][c];
                for (int j = 0; j < NM; ++j) { S[rr][j] -= f * S[c][j]; inv[rr][j] -= f * inv[c][j]; }
            }
        }
        double K[NS][NM];
        for (int i = 0; i < NS; ++i) for (int j = 0; j < NM; ++j) {
            double s = 0; for (int k = 0; k < NM; ++k) s += cov[i * NS + k] * inv[k][j];
            K[i][j] = s;
        }
        std::array<double, NM> y{};
        for (int i = 0; i < NM; ++i) y[i] = z[i] - mean[i];
        for (int i = 0; i < NS; ++i) { double s = 0; for (int j = 0; j < NM; ++j) s += K[i][j] * y[j]; mean[i] += s; }
        // P = (I - K H) P
        Cov p{};
        for (int i = 0; i < NS; ++i) for (int c = 0; c < NS; ++c) {
            double s = cov[i * NS + c];
            for (int j = 0; j < NM; ++j) s -= K[i][j] * cov[j * NS + c];
            p[i * NS + c] = s;
        }
        cov = p;
    }
    Box box() const {
        if (xywh) return Box{static_cast<float>(mean[0] - mean[2] / 2), static_cast<float>(mean[1] - mean[3] / 2),
                   static_cast<float>(mean[2]), static_cast<float>(mean[3])};
        const double w = mean[2] * mean[3];
        return Box{static_cast<float>(mean[0] - w / 2), static_cast<float>(mean[1] - mean[3] / 2),
                   static_cast<float>(w), static_cast<float>(mean[3])};
    }
    std::array<double, NM> measure(const Box& b) const {
        const double cx = b.x + b.width / 2.0, cy = b.y + b.height / 2.0;
        if (xywh) return {cx, cy, b.width, b.height};
        return {cx, cy, b.height > 0 ? b.width / static_cast<double>(b.height) : 0.0, b.height};
    }
    // BoT-SORT multi_gmc: rotate/scale the full state by the affine 2x2 block, translate the position
    void apply_affine(const double R[2][2], double tx, double ty) {
        Mean m = mean;
        for (int blk = 0; blk < 4; ++blk) {          // (x,y) (w,h)|(a,h) (vx,vy) (vw,vh): rotate each pair
            const int i = blk * 2;
            if (!xywh && blk == 1) continue;         // aspect ratio + height are not a vector
            if (!xywh && blk == 3) continue;
            const double a = mean[i], b = mean[i + 1];
            m[i] = R[0][0] * a + R[0][1] * b;
            m[i + 1] = R[1][0] * a + R[1][1] * b;
        }
        m[0] += tx; m[1] += ty;
        mean = m;
        // P = R8 P R8^T with R8 = blockdiag(R, R, R, R) (xyah: identity on the skipped blocks)
        double R8[NS][NS] = {};
        for (int blk = 0; blk < 4; ++blk) {
            const int i = blk * 2;
            const bool rot = xywh || (blk == 0 || blk == 2);
            R8[i][i] = rot ? R[0][0] : 1; R8[i][i + 1] = rot ? R[0][1] : 0;
            R8[i + 1][i] = rot ? R[1][0] : 0; R8[i + 1][i + 1] = rot ? R[1][1] : 1;
        }
        Cov t{}, p{};
        for (int i = 0; i < NS; ++i) for (int c = 0; c < NS; ++c) { double s = 0; for (int k = 0; k < NS; ++k) s += R8[i][k] * cov[k * NS + c]; t[i * NS + c] = s; }
        for (int i = 0; i < NS; ++i) for (int c = 0; c < NS; ++c) { double s = 0; for (int k = 0; k < NS; ++k) s += t[i * NS + k] * R8[c][k]; p[i * NS + c] = s; }
        cov = p;
    }
};

struct STrack {
    int id = 0;
    KF kf;
    int class_id = 0;
    float score = 0.f;
    TrackState state = TrackState::New;
    bool activated = false;
    int frame_id = 0, start_frame = 0, tracklet_len = 0;
    std::vector<float> mask_coeffs;
    int det_index = -1;
};

static double iou_impl(const Box& a, const Box& b) {
    const double ix = std::max(0.f, std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x));
    const double iy = std::max(0.f, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
    const double inter = ix * iy;
    const double u = static_cast<double>(a.width) * a.height + static_cast<double>(b.width) * b.height - inter;
    return u > 0 ? inter / u : 0.0;
}

// Hungarian assignment (Kuhn-Munkres, O(n^3)) on a rectangular cost matrix; entries above `thresh`
// are treated as forbidden. Returns matches (row, col) plus the unmatched rows and columns.
struct Assignment { std::vector<std::pair<int, int>> matches; std::vector<int> u_rows, u_cols; };
Assignment linear_assignment(const std::vector<std::vector<double>>& cost, int ncols, double thresh) {
    Assignment out;
    const int R = static_cast<int>(cost.size()), C = ncols;   // ncols: an empty matrix still has columns
    if (R == 0 || C == 0) {
        for (int r = 0; r < R; ++r) out.u_rows.push_back(r);
        for (int c = 0; c < C; ++c) out.u_cols.push_back(c);
        return out;
    }
    const int n = std::max(R, C);
    const double BIG = 1e6;
    std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, BIG));
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) a[r + 1][c + 1] = cost[r][c] > thresh ? BIG : cost[r][c];
    std::vector<double> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i; int j0 = 0;
        std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
        std::vector<char> used(n + 1, 0);
        do {
            used[j0] = 1; const int i0 = p[j0]; double delta = std::numeric_limits<double>::infinity(); int j1 = 0;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                const double cur = a[i0][j] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) { if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta; }
            j0 = j1;
        } while (p[j0] != 0);
        do { const int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    std::vector<char> rm(R, 0), cm(C, 0);
    for (int j = 1; j <= n; ++j) {
        const int i = p[j];
        if (i >= 1 && i <= R && j <= C && cost[i - 1][j - 1] <= thresh) { out.matches.emplace_back(i - 1, j - 1); rm[i - 1] = cm[j - 1] = 1; }
    }
    for (int r = 0; r < R; ++r) if (!rm[r]) out.u_rows.push_back(r);
    for (int c = 0; c < C; ++c) if (!cm[c]) out.u_cols.push_back(c);
    return out;
}
} // namespace

double box_iou(const Box& a, const Box& b) { return iou_impl(a, b); }

struct CoreTracker::Impl {
    CoreConfig cfg;
    std::vector<STrack> tracked, lost, removed;
    int frame_id = 0, next_id = 1, max_time_lost = 30;

    explicit Impl(const CoreConfig& c) : cfg(c) {
        max_time_lost = static_cast<int>(std::lround(c.fps / 30.0 * c.track_buffer));
        if (max_time_lost < 1) max_time_lost = 1;
    }

    static std::vector<std::vector<double>> cost_matrix(const std::vector<STrack*>& trks, const std::vector<const TrackInput*>& dets,
                                                        bool fuse) {
        std::vector<std::vector<double>> c(trks.size(), std::vector<double>(dets.size(), 1.0));
        for (size_t i = 0; i < trks.size(); ++i) {
            const Box tb = trks[i]->kf.box();
            for (size_t j = 0; j < dets.size(); ++j) {
                double s = iou_impl(tb, dets[j]->box);
                if (fuse) s *= dets[j]->conf;
                c[i][j] = 1.0 - s;
            }
        }
        return c;
    }

    void activate(STrack& t, const TrackInput& d, int det_index) {
        t.id = next_id++;
        t.kf.xywh = cfg.botsort;
        t.kf.initiate(t.kf.measure(d.box));
        t.class_id = d.class_id; t.score = d.conf; t.mask_coeffs = d.mask_coeffs; t.det_index = det_index;
        t.tracklet_len = 0; t.state = TrackState::Tracked;
        t.activated = (frame_id == 1);
        t.frame_id = t.start_frame = frame_id;
    }
    void update_track(STrack& t, const TrackInput& d, int det_index, bool reactivate) {
        t.kf.update(t.kf.measure(d.box));
        t.class_id = d.class_id; t.score = d.conf; t.mask_coeffs = d.mask_coeffs; t.det_index = det_index;
        t.tracklet_len = reactivate ? 0 : t.tracklet_len + 1;
        t.state = TrackState::Tracked; t.activated = true; t.frame_id = frame_id;
    }

    std::vector<CoreTrack> update(const std::vector<TrackInput>& dets, const Motion* motion) {
        ++frame_id;
        std::vector<const TrackInput*> high, low;
        std::vector<int> high_idx, low_idx;
        for (size_t i = 0; i < dets.size(); ++i) {
            if (dets[i].conf >= cfg.track_high_thresh) { high.push_back(&dets[i]); high_idx.push_back(static_cast<int>(i)); }
            else if (dets[i].conf > cfg.track_low_thresh) { low.push_back(&dets[i]); low_idx.push_back(static_cast<int>(i)); }
        }
        // split confirmed / unconfirmed
        std::vector<STrack*> unconfirmed, confirmed;
        for (auto& t : tracked) (t.activated ? confirmed : unconfirmed).push_back(&t);
        std::vector<STrack*> pool = confirmed;
        for (auto& t : lost) pool.push_back(&t);
        for (auto* t : pool) { t->kf.predict(); t->det_index = -1; }
        for (auto* t : unconfirmed) { t->kf.predict(); t->det_index = -1; }
        // camera motion compensation (BoT-SORT): the host estimated it, apply it to every prediction
        if (cfg.botsort && motion) {
            for (auto* t : pool) t->kf.apply_affine(motion->R, motion->tx, motion->ty);
            for (auto* t : unconfirmed) t->kf.apply_affine(motion->R, motion->tx, motion->ty);
        }
        std::vector<STrack> activated, refind, lost_now, removed_now;
        // ---- first association: confirmed + lost vs high-score detections ----
        auto A = linear_assignment(cost_matrix(pool, high, cfg.fuse_score), static_cast<int>(high.size()), cfg.match_thresh);
        for (const auto& [ti, di] : A.matches) {
            STrack* t = pool[ti];
            const bool was_lost = t->state == TrackState::Lost;
            update_track(*t, *high[di], high_idx[di], was_lost);
            (was_lost ? refind : activated).push_back(*t);
        }
        // ---- second association: still-tracked leftovers vs low-score detections (IoU only) ----
        std::vector<STrack*> r_tracked;
        for (int ti : A.u_rows) if (pool[ti]->state == TrackState::Tracked) r_tracked.push_back(pool[ti]);
        auto B = linear_assignment(cost_matrix(r_tracked, low, false), static_cast<int>(low.size()), 0.5);
        for (const auto& [ti, di] : B.matches) {
            STrack* t = r_tracked[ti];
            update_track(*t, *low[di], low_idx[di], false);
            activated.push_back(*t);
        }
        for (int ti : B.u_rows) { STrack* t = r_tracked[ti]; t->state = TrackState::Lost; lost_now.push_back(*t); }
        // ---- unconfirmed tracks vs the remaining high-score detections ----
        std::vector<const TrackInput*> rem_high; std::vector<int> rem_idx;
        for (int di : A.u_cols) { rem_high.push_back(high[di]); rem_idx.push_back(high_idx[di]); }
        auto Cc = linear_assignment(cost_matrix(unconfirmed, rem_high, cfg.fuse_score), static_cast<int>(rem_high.size()), 0.7);
        for (const auto& [ti, di] : Cc.matches) {
            STrack* t = unconfirmed[ti];
            update_track(*t, *rem_high[di], rem_idx[di], false);
            activated.push_back(*t);
        }
        for (int ti : Cc.u_rows) { STrack* t = unconfirmed[ti]; t->state = TrackState::Removed; removed_now.push_back(*t); }
        // ---- new tracks ----
        for (int di : Cc.u_cols) {
            if (rem_high[di]->conf < cfg.new_track_thresh) continue;
            STrack t;
            activate(t, *rem_high[di], rem_idx[di]);
            activated.push_back(t);
        }
        // ---- age out lost tracks ----
        std::vector<STrack> lost_keep;
        for (auto& t : lost) {
            if (frame_id - t.frame_id > max_time_lost) { t.state = TrackState::Removed; removed_now.push_back(t); }
            else if (t.state == TrackState::Lost) lost_keep.push_back(t);   // (re-found ones moved to tracked below)
        }
        // rebuild the pools
        std::vector<STrack> new_tracked;
        for (auto& t : activated) new_tracked.push_back(t);
        for (auto& t : refind) new_tracked.push_back(t);
        // tracked entries that were neither matched nor lost this frame do not exist (every tracked
        // track is either matched, lost or removed above), so new_tracked is complete
        std::vector<STrack> new_lost;
        for (auto& t : lost_keep) {
            bool refound = false;
            for (const auto& r : refind) if (r.id == t.id) { refound = true; break; }
            if (!refound) new_lost.push_back(t);
        }
        for (auto& t : lost_now) new_lost.push_back(t);
        tracked.swap(new_tracked);
        lost.swap(new_lost);
        for (auto& t : removed_now) removed.push_back(t);
        if (removed.size() > 1000) removed.erase(removed.begin(), removed.begin() + 500);
        // ---- output: activated tracks ----
        std::vector<CoreTrack> out;
        for (const auto& t : tracked) {
            if (!t.activated) continue;
            CoreTrack o;
            o.id = t.id; o.class_id = t.class_id; o.conf = t.score; o.box = t.kf.box();
            o.age = frame_id - t.start_frame; o.hits = t.tracklet_len + 1; o.time_since_update = frame_id - t.frame_id;
            o.state = t.state; o.mask_coeffs = t.mask_coeffs; o.det_index = t.det_index;
            out.push_back(std::move(o));
        }
        return out;
    }
};

CoreTracker::CoreTracker(const CoreConfig& cfg) : impl_(std::make_unique<Impl>(cfg)) {}
CoreTracker::~CoreTracker() = default;
std::vector<CoreTrack> CoreTracker::update(const std::vector<TrackInput>& dets, const Motion* motion) { return impl_->update(dets, motion); }
void CoreTracker::reset() { const CoreConfig c = impl_->cfg; impl_ = std::make_unique<Impl>(c); }
int CoreTracker::frame_count() const { return impl_->frame_id; }
const CoreConfig& CoreTracker::config() const { return impl_->cfg; }

} // namespace yolomaster::track::core
