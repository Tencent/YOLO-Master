#include "app.hpp"
#include "imgui.h"
#include "about.hpp"
#include "backend_factory.hpp"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace yolomaster;

namespace gui {

// 10-color palette (RGB 0-1), indexed cls%10 - identical to the Mac runner.
static const float kPalette[10][3] = {
    {0.98f,0.26f,0.30f},{0.20f,0.71f,0.98f},{0.16f,0.85f,0.52f},{0.99f,0.79f,0.12f},
    {0.72f,0.40f,0.98f},{0.99f,0.55f,0.18f},{0.10f,0.83f,0.80f},{0.98f,0.36f,0.66f},
    {0.55f,0.82f,0.28f},{0.40f,0.52f,0.98f},
};
static ImU32 col_of(int cls, float a = 1.f) {
    const float* c = kPalette[((cls % 10) + 10) % 10];
    return ImGui::GetColorU32(ImVec4(c[0], c[1], c[2], a));
}
// black or white text depending on the box color's luminance (readable on any hue).
static ImU32 text_on(int cls) {
    const float* c = kPalette[((cls % 10) + 10) % 10];
    const float lum = 0.299f * c[0] + 0.587f * c[1] + 0.114f * c[2];
    return lum > 0.6f ? IM_COL32(0,0,0,255) : IM_COL32(255,255,255,255);
}

App::~App() { cancel_export(); close_camera(); close_video(); close_folder(); }   // join all background threads

SliceConfig App::slice_config() const {
    SliceConfig sc;
    sc.mode = slice_mode_;
    sc.tile_size = tile_size_req_;
    sc.keep_global_masks = slice_masks_;
    return sc;
}

std::string App::gpu_device_str() const {
    return device_ == Device::GPU ? "gpu" : "cpu";   // make_backend maps "gpu" per backend
}

void App::load_model(const Platform& plat) {
    const bool cam = is_cam_;
    const bool vid = is_video_;
    const bool fold = !folder_imgs_.empty();
    const std::string vpath = video_path_, fdir = folder_path_;
    stop_worker();                                  // stop any camera worker holding the old backend
    if (vid)  { vinfer_cancel_ = true; if (vinfer_.joinable()) vinfer_.join(); vinfer_cancel_ = false; }
    if (fold) { finfer_cancel_ = true; if (finfer_.joinable()) finfer_.join(); finfer_cancel_ = false; }
    be_.reset();
    dets_.clear();
    be_err_.clear();
    model_is_seg_ = false;
    const char* names[] = {"auto", "onnx", "ncnn", "mnn"};
    std::string resolved, err;
    auto be = make_backend(model_path_, names[backend_sel_], threads_, gpu_device_str(), resolved, err);
    if (!be) { be_err_ = err; return; }
    be_ = std::move(be);
    be_name_ = resolved;
    be_ep_ = be_->active_ep;
    be_note_ = be_->ep_note;

    // resolve config: model metadata > visdrone fallback (mirrors the CLI).
    if (!be_->meta_names.empty()) cfg_.class_names = be_->meta_names;
    else                          cfg_.class_names = visdrone_classes();
    cfg_.imgsz = be_->fixed_imgsz > 0 ? be_->fixed_imgsz
               : (be_->meta_imgsz > 0 ? be_->meta_imgsz : 640);
    cfg_.max_det = 300;
    cfg_.multi_label = false;
    reset_zoom();
    if (cam)                     { start_worker(); submit_job(img_bgr_); }   // resume live feed
    else if (vid && !vpath.empty()) open_video(vpath, plat);                 // re-infer the clip
    else if (fold && !fdir.empty()) load_folder(fdir, plat);                 // re-infer the folder
    else                         need_reinfer_ = !img_bgr_.empty();
}

// Load the bundled default model so the app is useful the moment it opens (mirrors the Mac
// runner shipping v0.1-seg-N). Searched next to the exe first (shipped layout), then up the
// tree at the repo's models/ (dev builds run from gui/build/Release).
void App::load_default_model(const Platform& plat) {
    tried_default_ = true;
    namespace fs = std::filesystem;
    const char* names[] = { "v0.1-seg-n.onnx", "esmoe_n_visdrone_sim.onnx" };
    const fs::path exe(plat.exe_dir.empty() ? "." : plat.exe_dir);
    const fs::path roots[] = { exe / "models", exe / ".." / ".." / ".." / "models" };
    std::error_code ec;
    for (const auto& r : roots)
        for (const char* n : names) {
            const fs::path p = r / n;
            if (fs::exists(p, ec)) {
                std::snprintf(model_path_, sizeof(model_path_), "%s",
                              p.lexically_normal().string().c_str());
                load_model(plat);
                return;
            }
        }
}

void App::open_media(const std::string& path, const Platform& plat) {
    switch (classify_source(path)) {     // autodetect image / video / folder by extension
        case SourceKind::Video: open_video(path, plat); break;
        case SourceKind::Dir:   load_folder(path, plat); break;
        case SourceKind::Image:
            load_image(path, plat); break;   // load_image exits folder/video/camera mode
        default: load_err_ = "unsupported file type: " + path;
    }
}

void App::load_image(const std::string& path, const Platform& plat) {
    close_camera(&plat);
    close_video();                       // leave video/camera/folder mode if we were in it
    close_folder(&plat);
    load_err_.clear();
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) { load_err_ = "cannot read image: " + path; return; }
    img_bgr_ = bgr;
    img_path_ = path;
    cv::Mat rgba;
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    if (!plat.upload(rgba, img_tex_)) { load_err_ = "GPU texture upload failed"; return; }
    reset_zoom();
    need_reinfer_ = (be_ != nullptr);
}

void App::close_folder(const Platform* plat) {
    finfer_cancel_ = true;
    if (finfer_.joinable()) finfer_.join();
    finfer_cancel_ = false;
    if (plat) free_thumbs(*plat);   // release thumbnail textures for the old folder
    else thumbs_.clear();
    folder_imgs_.clear(); fcache_.clear();
    cur_idx_ = -1; folder_ready_ = false; fprogress_ = 0; finfer_done_ = false;
    fmean_ms_ = fmin_ms_ = fmax_ms_ = fwall_s_ = 0; fcount_ = 0;
}

// Background: forward-pass EVERY image once, caching per-image candidates (+ proto for seg)
// and timings (per-image model ms, plus folder mean/min/max and wall clock).
void App::folder_preinfer(std::vector<std::string> paths, Config c, SliceConfig sc) {
    using clk = std::chrono::steady_clock;
    const auto t_start = clk::now();
    double sum = 0, lo = 0, hi = 0; int n = 0;
    for (int i = 0; i < (int)paths.size() && !finfer_cancel_; ++i) {
        cv::Mat bgr = cv::imread(paths[i], cv::IMREAD_COLOR);
        if (!bgr.empty()) {
            try {
                double model_ms = 0.0;
                if (sc.mode != SliceMode::Off) {
                    // postcondition restores be_->candidates/lb/proto -> snapshot below unchanged
                    const SliceOutput so = sliced_candidates(*be_, bgr, c, sc, kConfFloor,
                                                             &finfer_cancel_);
                    if (so.cancelled) break;
                    model_is_seg_ = so.model_is_seg;
                    tstats_.add(so.tiles_run, so.tiles_total, so.tile_size_used,
                                so.used_fallback, so.capped);
                    model_ms = so.infer_ms;
                } else {
                    be_->infer(bgr, c);
                    model_ms = be_->infer_ms;
                }
                FolderItem it;
                it.cands = be_->candidates; it.lb = be_->cand_lb;
                it.ow = be_->cand_orig_w;   it.oh = be_->cand_orig_h;
                if (be_->is_seg()) { it.proto = be_->proto; it.pc = be_->proto_c;
                                     it.ph = be_->proto_h; it.pw = be_->proto_w; }
                // A sliced forward consists of the global pass plus all
                // selected tiles; model_ms is the aggregate value returned by
                // sliced_candidates rather than the last tile's value.
                it.ms = model_ms;
                it.done = true;
                fcache_[i] = std::move(it);
                sum += model_ms; ++n;
                if (n == 1 || model_ms < lo) lo = model_ms;
                if (n == 1 || model_ms > hi) hi = model_ms;
                fmean_ms_ = sum / n;              // publish live so the UI can show progress stats
                fmin_ms_ = lo; fmax_ms_ = hi; fcount_ = n;
                fwall_s_ = std::chrono::duration<double>(clk::now() - t_start).count();
            } catch (...) {}
        }
        fprogress_ = i + 1;
    }
    fwall_s_ = std::chrono::duration<double>(clk::now() - t_start).count();
    finfer_done_ = true;
}

void App::load_folder(const std::string& dir, const Platform& plat) {
    close_camera(&plat);
    close_video();
    close_folder(&plat);
    load_err_.clear();
    folder_imgs_ = gather_images(dir, 0);   // sorted image paths
    std::string stem_error;
    if (!validate_unique_stems(folder_imgs_, stem_error)) {
        folder_imgs_.clear();
        cur_idx_ = -1;
        load_err_ = stem_error;
        return;
    }
    folder_path_ = dir;
    if (folder_imgs_.empty()) { cur_idx_ = -1; load_err_ = "no images in: " + dir; return; }
    if (!be_) { folder_imgs_.clear(); load_err_ = "load a model first"; return; }
    fcache_.assign(folder_imgs_.size(), FolderItem{});
    folder_ready_ = false; fprogress_ = 0; finfer_done_ = false;
    tstats_ = TileStats{};
    cur_idx_ = 0; scroll_to_cur_ = true;
    show_folder_item(0, plat);              // show first image now; boxes appear when pre-infer finishes
    Config c = cfg_;
    c.conf_thresh = kConfFloor; c.iou_thresh = iou_; c.stretch = (prep_ == Preprocess::Stretch);
    finfer_ = std::thread(&App::folder_preinfer, this, folder_imgs_, c, slice_config());
}

// re-NMS + mask for a cached image (no decode) - used on conf/IoU change
void App::overlay_folder_item(int idx, const Platform& plat) {
    if (idx < 0 || idx >= (int)fcache_.size() || !fcache_[idx].done) {
        dets_.clear(); class_counts_.clear(); has_mask_ = false; return;
    }
    const FolderItem& it = fcache_[idx];
    seg_model_ = !it.proto.empty();
    if (seg_model_) model_is_seg_ = true;
    inf_ms_ = it.ms; pre_ms_ = 0; post_ms_ = 0;   // this image's model-only time
    Config c = cfg_; c.conf_thresh = conf_; c.iou_thresh = iou_;
    dets_ = nms_and_cap(it.cands, c, it.ow, it.oh);
    class_counts_.assign(cfg_.num_classes(), 0);
    for (const auto& d : dets_)
        if (d.class_id >= 0 && d.class_id < (int)class_counts_.size()) class_counts_[d.class_id]++;
    if (seg_model_) build_overlay(dets_, it.proto, it.pc, it.ph, it.pw, it.lb, it.ow, it.oh, plat);
    else has_mask_ = false;
}

// decode the image pixels + show cached overlay (if the pre-infer has finished)
void App::show_folder_item(int idx, const Platform& plat) {
    if (idx < 0 || idx >= (int)folder_imgs_.size()) return;
    cv::Mat bgr = cv::imread(folder_imgs_[idx], cv::IMREAD_COLOR);
    if (bgr.empty()) { load_err_ = "cannot read: " + folder_imgs_[idx]; return; }
    img_bgr_ = bgr; img_path_ = folder_imgs_[idx];
    cv::Mat rgba; cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    plat.upload(rgba, img_tex_);
    reset_zoom();
    if (folder_ready_) overlay_folder_item(idx, plat);
    else { dets_.clear(); has_mask_ = false; }
}

void App::select_index(int i, const Platform& plat) {
    if (folder_imgs_.empty()) return;
    cur_idx_ = std::clamp(i, 0, (int)folder_imgs_.size() - 1);
    scroll_to_cur_ = true;
    show_folder_item(cur_idx_, plat);
}

void App::close_video() {
    vinfer_cancel_ = true;                // stop any in-flight pre-inference
    if (vinfer_.joinable()) vinfer_.join();
    vinfer_cancel_ = false;
    if (cap_.isOpened()) cap_.release();
    is_video_ = false; playing_ = false; video_ready_ = false;
    frame_idx_ = 0; total_frames_ = 0; play_accum_ = 0.0;
    vcache_.clear(); vproto_.clear(); vprogress_ = 0; vinfer_done_ = false;
}

void App::show_frame(const cv::Mat& frame, const Platform& plat) {
    img_bgr_ = frame.clone();            // clone: cap_ reuses its internal buffer on the next read
    cv::Mat rgba;
    cv::cvtColor(img_bgr_, rgba, cv::COLOR_BGR2RGBA);
    plat.upload(rgba, img_tex_);
}

// Background: forward-pass EVERY frame once, caching per-frame candidates (+ proto for seg).
// Runs on its own thread + its own capture; the UI shows a progress bar meanwhile.
void App::video_preinfer(std::string path, Config c) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) { vinfer_done_ = true; return; }
    double sum = 0; int cnt = 0;
    cv::Mat f;
    while (!vinfer_cancel_ && cap.read(f) && !f.empty()) {
        try { be_->infer(f, c); } catch (...) { break; }
        vcache_.push_back(be_->candidates);
        if (be_->is_seg()) {
            if (vproto_.size() < vcache_.size() - 1) vproto_.resize(vcache_.size() - 1);
            vproto_.push_back(be_->proto);
            vproto_c_ = be_->proto_c; vproto_h_ = be_->proto_h; vproto_w_ = be_->proto_w;
        }
        vlb_ = be_->cand_lb; vorig_w_ = be_->cand_orig_w; vorig_h_ = be_->cand_orig_h;
        sum += be_->infer_ms; cnt++;
        vprogress_ = (int)vcache_.size();
    }
    total_frames_ = (int)vcache_.size();
    vinfer_ms_ = cnt ? sum / cnt : 0.0;
    vinfer_done_ = true;
}

void App::open_video(const std::string& path, const Platform& plat) {
    close_camera(&plat);
    close_video();
    close_folder(&plat);                         // leave folder mode
    load_err_.clear();
    if (!be_) { load_err_ = "load a model first"; return; }
    if (!cap_.open(path)) { load_err_ = "cannot open video: " + path; return; }
    is_video_ = true; playing_ = false; play_accum_ = 0.0; frame_idx_ = 0;
    video_ready_ = false; vprogress_ = 0; vinfer_done_ = false;
    reset_zoom();
    video_path_ = path;
    total_frames_ = (int)cap_.get(cv::CAP_PROP_FRAME_COUNT);
    const double fps = cap_.get(cv::CAP_PROP_FPS);
    video_fps_ = (fps > 1.0 && fps < 1000.0) ? fps : 30.0;
    cv::Mat f0;                                  // show frame 0 immediately (no boxes yet)
    if (cap_.read(f0) && !f0.empty()) show_frame(f0, plat);
    cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
    Config c = cfg_;                             // cache to the conf floor; per-frame NMS is cheap
    c.conf_thresh = kConfFloor; c.iou_thresh = iou_; c.stretch = (prep_ == Preprocess::Stretch);
    vinfer_ = std::thread(&App::video_preinfer, this, path, c);
}

// re-NMS + mask for a cached frame's candidates (no decode) - used on conf/IoU change
void App::overlay_from_cache(int idx, const Platform& plat) {
    if (idx < 0 || idx >= (int)vcache_.size()) return;
    seg_model_ = !vproto_.empty();
    Config c = cfg_; c.conf_thresh = conf_; c.iou_thresh = iou_;
    dets_ = nms_and_cap(vcache_[idx], c, vorig_w_, vorig_h_);
    class_counts_.assign(cfg_.num_classes(), 0);
    for (const auto& d : dets_)
        if (d.class_id >= 0 && d.class_id < (int)class_counts_.size()) class_counts_[d.class_id]++;
    if (seg_model_ && idx < (int)vproto_.size())
        build_overlay(dets_, vproto_[idx], vproto_c_, vproto_h_, vproto_w_, vlb_, vorig_w_, vorig_h_, plat);
    else has_mask_ = false;
}

// Show frame `idx`: display its pixels + cached overlay. Forward jumps grab-skip the
// intermediate frames (decode-skip, no upload) so playback stays real-time when the UI
// can't render every frame; backward jumps seek.
void App::render_video_frame(int idx, const Platform& plat) {
    if (!cap_.isOpened() || total_frames_ <= 0) return;
    idx = std::clamp(idx, 0, total_frames_ - 1);
    if (idx != frame_idx_) {
        if (idx > frame_idx_ + 1) {                      // forward skip: grab (no decode-to-Mat)
            for (int i = frame_idx_ + 1; i < idx; ++i)
                if (!cap_.grab()) { cap_.set(cv::CAP_PROP_POS_FRAMES, (double)idx); break; }
        } else if (idx < frame_idx_ + 1) {               // backward / wrap: seek
            cap_.set(cv::CAP_PROP_POS_FRAMES, (double)idx);
        }
        cv::Mat f;
        if (!cap_.read(f) || f.empty()) {
            cap_.set(cv::CAP_PROP_POS_FRAMES, (double)idx);
            if (!cap_.read(f) || f.empty()) return;
        }
        frame_idx_ = idx;
        show_frame(f, plat);
    }
    overlay_from_cache(idx, plat);
    inf_ms_ = vinfer_ms_;
}

void App::seek_video(int idx, const Platform& plat) {
    if (video_ready_) render_video_frame(idx, plat);
}

// ---------------- webcam ----------------
void App::open_camera(const Platform& plat) {
    load_err_.clear();
    close_video();
    close_folder(&plat);
    if (!be_) { load_err_ = "load a model first"; return; }
    if (!cam_.open(0)) { load_err_ = "cannot open webcam (device 0)"; return; }
    cam_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);      // 720p is plenty; we downscale to imgsz anyway
    cam_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    is_cam_ = true; cam_fps_ema_ = cam_ms_ema_ = 0.0;
    reset_zoom();
    start_worker();
}

// Blank the preview and drop any stale detections/masks (so stopping a source
// leaves an empty canvas rather than a frozen last frame).
void App::clear_preview(const Platform* plat) {
    if (plat && plat->release) { plat->release(img_tex_); plat->release(mask_tex_); }
    else { img_tex_ = Texture{}; mask_tex_ = Texture{}; }
    img_bgr_.release();
    img_path_.clear();
    dets_.clear();
    class_counts_.clear();
    has_mask_ = false; seg_model_ = false;
    inf_ms_ = pre_ms_ = post_ms_ = 0;
    reset_zoom();
}

void App::close_camera(const Platform* plat) {
    const bool was_cam = is_cam_;
    stop_worker();
    if (cam_.isOpened()) cam_.release();
    is_cam_ = false;
    cam_fps_ema_ = cam_ms_ema_ = 0.0;
    if (was_cam) clear_preview(plat);   // no frozen last frame
}

// ---------------- async inference worker (video + webcam) ----------------
void App::start_worker() {
    if (worker_.joinable()) return;
    worker_quit_ = false; job_pending_ = false; worker_busy_ = false;
    async_mode_ = true;
    worker_ = std::thread(&App::worker_loop, this);
}

void App::stop_worker() {
    async_mode_ = false;
    if (!worker_.joinable()) return;
    { std::lock_guard<std::mutex> lk(job_mtx_); worker_quit_ = true; }
    job_cv_.notify_all();
    worker_.join();
    { std::lock_guard<std::mutex> lk(res_mtx_); ares_dirty_ = false; }
    worker_busy_ = false; job_pending_ = false;
}

void App::submit_job(const cv::Mat& frame) {
    if (worker_busy_ || frame.empty()) return;      // drop-late: skip while the worker is inferring
    {
        std::lock_guard<std::mutex> lk(job_mtx_);
        job_frame_ = frame.clone();
        job_cfg_ = cfg_;
        job_cfg_.conf_thresh = conf_;
        job_cfg_.iou_thresh  = iou_;
        job_cfg_.stretch     = (prep_ == Preprocess::Stretch);
        job_pending_ = true;
    }
    job_cv_.notify_one();
}

void App::worker_loop() {
    for (;;) {
        cv::Mat frame; Config c;
        {
            std::unique_lock<std::mutex> lk(job_mtx_);
            job_cv_.wait(lk, [&] { return job_pending_ || worker_quit_; });
            if (worker_quit_) return;
            frame = job_frame_; c = job_cfg_; job_pending_ = false;
        }
        if (!be_ || frame.empty()) continue;
        worker_busy_ = true;
        Config cc = c; cc.conf_thresh = std::min(c.conf_thresh, kConfFloor);   // cache to the floor
        bool ok = true; double inf = 0;
        try { be_->infer(frame, cc); inf = be_->infer_ms; }
        catch (...) { ok = false; }
        if (ok) {
            std::lock_guard<std::mutex> lk(res_mtx_);
            ares_.cands  = be_->candidates;
            ares_.dets   = nms_and_cap(be_->candidates, c, be_->cand_orig_w, be_->cand_orig_h);
            ares_.proto  = be_->proto;
            ares_.pc = be_->proto_c; ares_.ph = be_->proto_h; ares_.pw = be_->proto_w;
            ares_.lb = be_->cand_lb; ares_.ow = be_->cand_orig_w; ares_.oh = be_->cand_orig_h;
            ares_.inf_ms = inf; ares_.seg = be_->is_seg();
            ares_dirty_ = true;
        }
        worker_busy_ = false;
    }
}

void App::consume_async(const Platform& plat) {
    AsyncResult r;
    {
        std::lock_guard<std::mutex> lk(res_mtx_);
        if (!ares_dirty_) return;
        r = ares_;                 // copy out (proto included) so drawing is lock-free
        ares_dirty_ = false;
    }
    dets_ = std::move(r.dets);
    inf_ms_ = r.inf_ms; pre_ms_ = 0; post_ms_ = 0;
    seg_model_ = r.seg;
    if (is_cam_ && r.inf_ms > 0)
        cam_ms_ema_ = (cam_ms_ema_ <= 0) ? r.inf_ms : cam_ms_ema_ * 0.8 + r.inf_ms * 0.2;
    class_counts_.assign(cfg_.num_classes(), 0);
    for (const auto& d : dets_)
        if (d.class_id >= 0 && d.class_id < (int)class_counts_.size()) class_counts_[d.class_id]++;
    if (r.seg) build_overlay(dets_, r.proto, r.pc, r.ph, r.pw, r.lb, r.ow, r.oh, plat);
    else has_mask_ = false;
}

void App::run_inference() {
    if (!be_ || img_bgr_.empty()) return;
    // forward once with a low conf floor so the cached candidates cover the whole
    // slider range; nms_and_cap() then applies the real (display) conf cheaply.
    Config c = cfg_;
    c.conf_thresh = std::min(conf_, kConfFloor);
    c.iou_thresh  = iou_;
    c.stretch     = (prep_ == Preprocess::Stretch);
    cfg_.stretch  = c.stretch;   // keep display cfg in sync (recompute_nms uses cfg_)
    try {
        if (slice_mode_ != SliceMode::Off) {
            // sliced_candidates restores the backend's cached state to describe the merged
            // run, so recompute_nms/rebuild_overlay below work unchanged.
            sstats_ = sliced_candidates(*be_, img_bgr_, c, slice_config(), kConfFloor);
            sliced_run_ = true;
            pre_ms_ = sstats_.pre_ms;
            inf_ms_ = sstats_.infer_ms;
            post_ms_ = sstats_.post_ms;   // sums of all forwards
        } else {
            dets_ = be_->infer(img_bgr_, c);
            sliced_run_ = false;
            pre_ms_ = be_->pre_ms; inf_ms_ = be_->infer_ms; post_ms_ = be_->post_ms;
        }
    }
    catch (const std::exception& e) { be_err_ = std::string("inference error: ") + e.what(); return; }
    seg_model_ = be_->is_seg();
    model_is_seg_ = sliced_run_ ? sstats_.model_is_seg : seg_model_;
    need_reinfer_ = false;
    need_renms_ = true;   // apply the real conf/iou below
}

void App::recompute_nms() {
    if (!be_) return;
    seg_model_ = be_->is_seg();
    cfg_.conf_thresh = conf_;
    cfg_.iou_thresh  = iou_;
    dets_ = nms_and_cap(be_->candidates, cfg_, be_->cand_orig_w, be_->cand_orig_h);
    class_counts_.assign(cfg_.num_classes(), 0);
    for (const auto& d : dets_)
        if (d.class_id >= 0 && d.class_id < (int)class_counts_.size()) class_counts_[d.class_id]++;
    need_renms_ = false;
    need_overlay_ = true;   // dets changed -> seg overlay is stale
}

void App::build_overlay(const std::vector<Detection>& dets, const std::vector<float>& proto,
                        int pc, int ph, int pw, const LetterboxInfo& lb,
                        int ow, int oh, const Platform& plat) {
    has_mask_ = false;
    if (proto.empty() || pc <= 0 || dets.empty()) return;
    cv::Mat ov = seg_overlay(dets, proto, pc, ph, pw, lb, cfg_.imgsz, ow, oh);
    if (plat.upload(ov, mask_tex_)) has_mask_ = true;
}

void App::rebuild_overlay(const Platform& plat) {
    need_overlay_ = false;
    if (!be_) { has_mask_ = false; return; }
    build_overlay(dets_, be_->proto, be_->proto_c, be_->proto_h, be_->proto_w,
                  be_->cand_lb, be_->cand_orig_w, be_->cand_orig_h, plat);
}

void App::frame(const Platform& plat) {
    if (!tried_default_) load_default_model(plat);   // bundled model, once at startup
    if (exporting_) poll_export();                   // join a finished export thread

    // sync inference for a single still image only (video/camera/folder own the backend elsewhere)
    if (!async_mode_ && !is_video_ && folder_imgs_.empty()) {
        if (need_reinfer_) run_inference();
        if (need_renms_)   recompute_nms();
        if (need_overlay_) rebuild_overlay(plat);
    }

    // folder: pre-infer every image (progress bar), then browse instantly from cache
    if (!folder_imgs_.empty()) {
        if (!folder_ready_) {
            if (finfer_done_) {
                if (finfer_.joinable()) finfer_.join();
                folder_ready_ = true;
                overlay_folder_item(cur_idx_, plat);   // reveal boxes for the shown image
            }
        } else if (need_renms_) {
            overlay_folder_item(cur_idx_, plat);       // conf/IoU re-NMS from cache
            need_renms_ = false;
        }
    }

    // webcam: grab + display every UI frame at feed rate; infer off-thread (drops late frames)
    if (is_cam_) {
        cv::Mat f;
        if (cam_.read(f) && !f.empty()) {
            if (cam_mirror_) cv::flip(f, f, 1);
            const double dt = ImGui::GetIO().DeltaTime;
            if (dt > 0) cam_fps_ema_ = (cam_fps_ema_ <= 0) ? 1.0 / dt : cam_fps_ema_ * 0.9 + (1.0 / dt) * 0.1;
            show_frame(f, plat);
            submit_job(img_bgr_);
        }
        consume_async(plat);
        if (need_renms_) need_renms_ = false;   // camera picks up new conf/iou on the next frame
    }

    // video: pre-infer all frames (progress bar), then play the raw video from cache - NO inference
    // on the playback path, so it runs at true source fps.
    if (is_video_) {
        if (!video_ready_) {
            if (vinfer_done_) {
                if (vinfer_.joinable()) vinfer_.join();
                video_ready_ = true;
                render_video_frame(0, plat);            // first frame with its cached overlay
            }
        } else {
            if (playing_) {
                play_accum_ += ImGui::GetIO().DeltaTime;
                const double iv = 1.0 / video_fps_;
                int steps = 0;                          // how many source frames elapsed this UI tick
                while (play_accum_ >= iv) {
                    play_accum_ -= iv;
                    if (++steps > (int)video_fps_) { play_accum_ = 0.0; break; }  // cap catch-up to ~1s
                }
                if (steps > 0) {                        // advance real-time; render_video_frame skips
                    int next = frame_idx_ + steps;
                    while (next >= total_frames_) next -= total_frames_;   // loop
                    render_video_frame(next, plat);
                }
            }
            if (need_renms_) { overlay_from_cache(frame_idx_, plat); need_renms_ = false; }
        }
    }

    // keyboard shortcuts (ignored while typing in a field).
    if (!ImGui::GetIO().WantTextInput) {
        const bool kR = ImGui::IsKeyPressed(ImGuiKey_RightArrow);
        const bool kL = ImGui::IsKeyPressed(ImGuiKey_LeftArrow);
        const bool kD = ImGui::IsKeyPressed(ImGuiKey_DownArrow);
        const bool kU = ImGui::IsKeyPressed(ImGuiKey_UpArrow);
        const bool fwd = kR || kD, back = kL || kU;
        if (!folder_imgs_.empty()) {
            const int n = (int)folder_imgs_.size();
            if (finder_mode_ == FinderMode::Icons && finder_cols_ > 1) {
                // macOS Finder grid semantics: left/right stay in the row, up/down in the column
                const int col = cur_idx_ % finder_cols_;
                if      (kR && col < finder_cols_ - 1 && cur_idx_ + 1 < n) select_index(cur_idx_ + 1, plat);
                else if (kL && col > 0)                                    select_index(cur_idx_ - 1, plat);
                else if (kD && cur_idx_ + finder_cols_ < n)                select_index(cur_idx_ + finder_cols_, plat);
                else if (kU && cur_idx_ - finder_cols_ >= 0)               select_index(cur_idx_ - finder_cols_, plat);
            } else {                                   // list view: linear
                if (fwd)       select_index(cur_idx_ + 1, plat);
                else if (back) select_index(cur_idx_ - 1, plat);
            }
        } else if (is_video_) {
            if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
                playing_ = !playing_;
                if (playing_) reset_zoom();          // zoom is for stills and paused frames
            }
            if (!playing_ && fwd)  seek_video(frame_idx_ + 1, plat);   // step frames while paused
            if (!playing_ && back) seek_video(frame_idx_ - 1, plat);
        }
        // zoom shortcuts (center-anchored): Ctrl+= / Ctrl+- step, Ctrl+0 reset
        const bool zoom_ok = img_tex_.id && !is_cam_ && (!is_video_ || (video_ready_ && !playing_));
        if (zoom_ok && ImGui::GetIO().KeyCtrl) {
            auto step = [&](float f) {
                const float z = std::clamp(zoom_ * f, 1.f, kZoomMax);
                const float r = z / zoom_;
                zoom_ = z; pan_x_ *= r; pan_y_ *= r;
                if (zoom_ < 1.02f) reset_zoom();
            };
            if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd))
                step(1.25f);
            if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract))
                step(1.f / 1.25f);
            if (ImGui::IsKeyPressed(ImGuiKey_0)) reset_zoom();
        }
    }

    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("##root", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);

    const float ui = ImGui::GetFontSize() / 17.0f;   // ~DPI scale (font is 17*dpi on Windows)
    ImGui::BeginChild("sidebar", ImVec2(300 * ui, 0), true);
    draw_sidebar(plat);
    ImGui::EndChild();

    if (!folder_imgs_.empty()) {
        ImGui::SameLine();
        // icon view: size the panel for exactly 3 columns; list view stays narrow
        const ImGuiStyle& st = ImGui::GetStyle();
        const float fw = (finder_mode_ == FinderMode::Icons)
                       ? 3.f * icon_size_ + 2.f * st.ItemSpacing.x + 2.f * st.WindowPadding.x
                         + st.ScrollbarSize + 6.f
                       : 250.f * ui;
        ImGui::BeginChild("filelist", ImVec2(fw, 0), true);
        draw_filelist(plat);
        ImGui::EndChild();
    }

    ImGui::SameLine();
    // NoScrollWithMouse: the wheel zooms the preview instead of scrolling the child
    ImGui::BeginChild("preview", ImVec2(0, 0), true, ImGuiWindowFlags_NoScrollWithMouse);
    draw_preview(plat);
    ImGui::EndChild();

    ImGui::End();

    if (show_about_) draw_about(plat);
}

// A rounded, padded, bordered "card" section - mirrors the macOS runner's grouped panels.
static void begin_card(const char* title) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.155f, 0.166f, 0.198f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border,  ImVec4(1.0f, 1.0f, 1.0f, 0.05f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 11.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14, 12));
    ImGui::BeginChild(title, ImVec2(0, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.56f, 0.61f, 0.68f, 1.0f));   // section header
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    ImGui::Spacing();
}
static void end_card() {
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
    ImGui::Dummy(ImVec2(0, 5));   // gap between cards
}
// small dim caption above a full-width control
static void field_label(const char* t) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.60f, 0.64f, 0.70f, 1.0f));
    ImGui::TextUnformatted(t);
    ImGui::PopStyleColor();
}

void App::draw_sidebar(const Platform& plat) {
    const float ui = ImGui::GetFontSize() / 17.0f;   // DPI scale for fixed pixel offsets
    // ---- header ----
    ImFont* hf = (ImFont*)plat.heading_font;
    if (hf) ImGui::PushFont(hf);
    ImGui::TextUnformatted("YOLO-Master");
    if (hf) ImGui::PopFont();
    const float titleH = ImGui::GetItemRectSize().y;
    // right-align the About button, vertically centred against the (taller) title
    const float btnW = ImGui::CalcTextSize("About").x + ImGui::GetStyle().FramePadding.x * 2.f;
    finder_hdr_pad_ = std::max(0.f, (titleH - ImGui::GetFrameHeight()) * 0.5f);
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - btnW);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + finder_hdr_pad_);
    if (ImGui::Button("About", ImVec2(btnW, 0))) show_about_ = true;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("About / Acknowledgements / License");
    ImGui::TextDisabled("Windows runner (GUI) - ONNX / ncnn / MNN");
    ImGui::Dummy(ImVec2(0, 6));

    // Lock the controls that define/redo the whole run: while the webcam is live, and while a
    // folder/video pre-inference pass is still going (matches the Mac runner's "busy" rule).
    // APPEARANCE and DETECTION (conf/IoU) stay live throughout - both are cheap re-renders.
    const bool busy = (is_video_ && !video_ready_) ||
                      (!folder_imgs_.empty() && !folder_ready_);
    const bool lock = is_cam_ || busy || exporting_;   // exports read the caches -> freeze them

    // ---- MODEL (auto-loads; no Load button) ----
    begin_card("MODEL");
    ImGui::BeginDisabled(lock);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextWithHint("##model", ".onnx / ncnn dir / .mnn", model_path_, sizeof(model_path_),
                                 ImGuiInputTextFlags_EnterReturnsTrue))
        load_model(plat);
    if (ImGui::Button("Browse...", ImVec2(-1, 0))) {
        std::string p = plat.open_file("Select model", "Models\0*.onnx;*.mnn;*.param\0All\0*.*\0");
        if (!p.empty()) { std::snprintf(model_path_, sizeof(model_path_), "%s", p.c_str()); load_model(plat); }
    }
    field_label("Backend");
    const char* backends[] = {"auto", "onnx", "ncnn", "mnn"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##backend", &backend_sel_, backends, IM_ARRAYSIZE(backends)) && model_path_[0])
        load_model(plat);
    if (be_) {
        ImGui::TextColored(ImVec4(0.5f,0.85f,0.4f,1), "%s  \xc2\xb7  %s  \xc2\xb7  nc %d  \xc2\xb7  %dpx",
                           be_name_.c_str(), be_ep_.c_str(), cfg_.num_classes(), cfg_.imgsz);
        if (!be_note_.empty())
            ImGui::TextWrapped("%s", be_note_.c_str());
    } else if (!be_err_.empty()) {
        ImGui::TextColored(ImVec4(0.98f,0.4f,0.4f,1), "%s", be_err_.c_str());
    }
    ImGui::EndDisabled();
    end_card();

    // ---- SOURCE ----
    begin_card("SOURCE");
    ImGui::BeginDisabled(lock);   // can't switch source mid-stream; Stop Camera stays enabled
    if (ImGui::Button("Open image / video...", ImVec2(-1, 0))) {   // autodetect by extension
        std::string p = plat.open_file("Open image or video",
            "Media\0*.jpg;*.jpeg;*.png;*.bmp;*.mp4;*.avi;*.mov;*.mkv\0All\0*.*\0");
        if (!p.empty()) open_media(p, plat);
    }
    if (ImGui::Button("Open folder...", ImVec2(-1, 0))) {
        std::string d = plat.open_folder("Select image folder");
        if (!d.empty()) load_folder(d, plat);
    }
    ImGui::EndDisabled();
    if (is_cam_) {                       // always live: the way out of camera mode
        if (ImGui::Button("Stop Camera", ImVec2(-1, 0))) close_camera(&plat);
        ImGui::Checkbox("Mirror", &cam_mirror_);
    } else if (busy) {                   // always live: the way out of a long pre-inference
        if (ImGui::Button("Cancel", ImVec2(-1, 0))) {
            if (is_video_) close_video();
            else           close_folder(&plat);
            clear_preview(&plat);
        }
    } else if (ImGui::Button("Live Webcam", ImVec2(-1, 0))) {
        open_camera(plat);
    }
    if (!img_bgr_.empty())
        ImGui::TextDisabled("%d x %d%s", img_bgr_.cols, img_bgr_.rows, is_cam_ ? "   live" : "");
    if (!load_err_.empty())
        ImGui::TextColored(ImVec4(0.98f,0.4f,0.4f,1), "%s", load_err_.c_str());
    end_card();

    // ---- PREPROCESS ----
    begin_card("PREPROCESS");
    ImGui::BeginDisabled(lock);
    field_label("Input fit");
    int pp = (int)prep_;
    const char* pps[] = {"letterbox", "stretch"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##prep", &pp, pps, IM_ARRAYSIZE(pps)) && (Preprocess)pp != prep_) {
        prep_ = (Preprocess)pp;
        if (is_video_)                  open_video(video_path_, plat);   // re-infer the clip
        else if (!folder_imgs_.empty()) load_folder(folder_path_, plat); // re-infer the folder
        else if (!is_cam_)              need_reinfer_ = true;            // camera: next frame
    }
    ImGui::EndDisabled();
    end_card();

    // ---- SLICING (images + folders; video/webcam stay single-pass) ----
    begin_card("SLICING");
    {
        const bool na = is_video_ || is_cam_;
        ImGui::BeginDisabled(lock || na);
        field_label("Mode");
        int sm = (int)slice_mode_;
        const char* sms[] = {"Off", "Dense", "Sparse SAHI"};
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##slicemode", &sm, sms, IM_ARRAYSIZE(sms)) && (SliceMode)sm != slice_mode_) {
            slice_mode_ = (SliceMode)sm;
            if (!folder_imgs_.empty()) load_folder(folder_path_, plat);   // re-infer the folder
            else                       need_reinfer_ = !img_bgr_.empty();
        }
        if (slice_mode_ != SliceMode::Off) {
            // per-source tile-size bound [model imgsz, max(imgsz, shortSide/4)]
            int ceiling = cfg_.imgsz;
            if (!folder_imgs_.empty()) {
                for (const auto& it : fcache_)
                    if (it.done) ceiling = std::max(ceiling,
                        clamped_tile_size(1 << 20, cfg_.imgsz, it.ow, it.oh));
            } else if (!img_bgr_.empty()) {
                ceiling = clamped_tile_size(1 << 20, cfg_.imgsz, img_bgr_.cols, img_bgr_.rows);
            }
            field_label("Tile size");
            if (ceiling > cfg_.imgsz) {
                int ts = tile_size_req_ > 0 ? std::clamp(tile_size_req_, cfg_.imgsz, ceiling)
                                            : cfg_.imgsz;
                ImGui::SetNextItemWidth(-1);
                ImGui::SliderInt("##tilesize", &ts, cfg_.imgsz, ceiling, "%d px");
                // commit on release: every change re-runs the whole source
                if (ImGui::IsItemDeactivatedAfterEdit() && ts != tile_size_req_) {
                    tile_size_req_ = ts;
                    if (!folder_imgs_.empty()) load_folder(folder_path_, plat);
                    else                       need_reinfer_ = !img_bgr_.empty();
                }
            } else {
                ImGui::BeginDisabled(true);
                int ts = cfg_.imgsz;
                ImGui::SetNextItemWidth(-1);
                ImGui::SliderInt("##tilesize", &ts, cfg_.imgsz, cfg_.imgsz, "%d px");
                ImGui::EndDisabled();
                ImGui::TextWrapped("The input dimensions are too small for larger tiles. "
                                   "Slicing runs at the model input (%d px).", cfg_.imgsz);
            }
            if (model_is_seg_ || slice_masks_) {
                bool km = slice_masks_;
                if (ImGui::Checkbox("Masks (global pass)", &km) && km != slice_masks_) {
                    slice_masks_ = km;
                    if (!folder_imgs_.empty()) load_folder(folder_path_, plat);
                    else                       need_reinfer_ = !img_bgr_.empty();
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Keep the global pass's segmentation masks in sliced runs.\n"
                                      "Tile detections are always boxes only.");
            }
        }
        ImGui::EndDisabled();
        if (na) ImGui::TextDisabled("Images and folders only");
    }
    end_card();

    // ---- DETECTION ----
    begin_card("DETECTION");   // always live: conf/IoU/NMS are cheap re-NMS, even on a running camera
    field_label("Confidence");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderFloat("##conf", &conf_, 0.05f, 0.95f, "%.2f")) need_renms_ = true;
    field_label("IoU");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderFloat("##iou", &iou_, 0.10f, 0.90f, "%.2f")) need_renms_ = true;
    field_label("NMS");
    int nm = cfg_.nms_mode == NmsMode::ClusterWeighted ? 1 : 0;
    const char* nms[] = {"Standard", "Cluster-Weighted"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##nmsmode", &nm, nms, IM_ARRAYSIZE(nms))) {
        cfg_.nms_mode = nm ? NmsMode::ClusterWeighted : NmsMode::Standard;
        need_renms_ = true;
    }
    if (cfg_.nms_mode == NmsMode::ClusterWeighted) {
        field_label("Sigma");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderFloat("##sigma", &cfg_.cw_sigma, 0.01f, 0.50f, "%.2f"))
            need_renms_ = true;
    }
    end_card();

    // ---- APPEARANCE ----
    begin_card("APPEARANCE");
    if (seg_model_) {
        field_label("Overlay");
        int ov = (int)overlay_;
        const char* ovs[] = {"both", "masks", "boxes"};
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##overlay", &ov, ovs, IM_ARRAYSIZE(ovs))) overlay_ = (Overlay)ov;
    }
    if (!(seg_model_ && overlay_ == Overlay::Masks)) {   // box style is irrelevant in masks-only
        field_label("Box style");
        int st = (int)style_;
        const char* styles[] = {"hud", "solid", "neon"};
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##style", &st, styles, IM_ARRAYSIZE(styles))) style_ = (BoxStyle)st;
    }
    field_label("Labels");
    int lm = (int)labels_;
    const char* lms[] = {"full", "min", "off"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##labels", &lm, lms, IM_ARRAYSIZE(lms))) labels_ = (LabelMode)lm;
    end_card();

    // ---- EXPORT (rendered frames + annotation labels; not for the live webcam) ----
    if (!is_cam_ && be_ && !img_bgr_.empty()) draw_export_card(plat);

    // ---- DEVICE ----
    begin_card("DEVICE");
    ImGui::BeginDisabled(lock);
    field_label("Compute");
    int dv = (int)device_;
    const char* devs[] = {"CPU", "GPU"};
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##device", &dv, devs, IM_ARRAYSIZE(devs)) && (Device)dv != device_) {
        device_ = (Device)dv;
        if (be_ || model_path_[0]) load_model(plat);
    }
    ImGui::EndDisabled();
    end_card();

    // ---- INFERENCE ----
    begin_card("INFERENCE");
    if (be_ && !img_bgr_.empty()) {
        const bool folder = !folder_imgs_.empty();
        // "This image / frame" model-only time
        ImGui::Text(folder ? "This image" : "Model");
        ImGui::SameLine(110 * ui); ImGui::TextDisabled("%.1f ms", inf_ms_);
        if (folder) {
            // folder-wide: mean model ms + throughput (images/sec), plus min/max and wall time
            ImGui::Text("Folder avg"); ImGui::SameLine(110 * ui);
            ImGui::TextDisabled("%.1f ms  \xc2\xb7  %.1f img/s", fmean_ms_,
                                fmean_ms_ > 0 ? 1000.0 / fmean_ms_ : 0.0);
            ImGui::Text("Throughput"); ImGui::SameLine(110 * ui);
            ImGui::TextDisabled("%.1f img/s  (wall)", fwall_s_ > 0 ? fcount_ / fwall_s_ : 0.0);
            if (fcount_ > 1) {
                ImGui::Text("Min / max"); ImGui::SameLine(110 * ui);
                ImGui::TextDisabled("%.1f / %.1f ms", fmin_ms_, fmax_ms_);
                ImGui::Text("Total"); ImGui::SameLine(110 * ui);
                ImGui::TextDisabled("%d imgs  \xc2\xb7  %.2f s", fcount_, fwall_s_);
            }
            if (slice_mode_ != SliceMode::Off && tstats_.tiles_total > 0) {
                ImGui::Text("Slicing"); ImGui::SameLine(110 * ui);
                ImGui::TextDisabled("%s", slice_mode_ == SliceMode::Dense ? "Dense" : "Sparse SAHI");
                ImGui::Text("Tile size"); ImGui::SameLine(110 * ui);
                ImGui::TextDisabled("%s px", tstats_.tile_size_label().c_str());
                ImGui::Text("Tiles"); ImGui::SameLine(110 * ui);
                ImGui::TextDisabled("%d / %d", tstats_.tiles_run, tstats_.tiles_total);
                if (tstats_.fallbacks > 0 || tstats_.capped > 0) {
                    ImGui::Text("Fallbacks"); ImGui::SameLine(110 * ui);
                    ImGui::TextDisabled("%d  \xc2\xb7  capped %d", tstats_.fallbacks, tstats_.capped);
                }
            }
        } else {
            const double total = pre_ms_ + inf_ms_ + post_ms_;
            ImGui::Text("Overall"); ImGui::SameLine(110 * ui);
            ImGui::TextDisabled("%.1f ms  \xc2\xb7  %.0f fps", total, total > 0 ? 1000.0 / total : 0.0);
            if (sliced_run_) {
                ImGui::Text("Slicing"); ImGui::SameLine(110 * ui);
                ImGui::TextDisabled("%s", slice_mode_ == SliceMode::Dense ? "Dense" : "Sparse SAHI");
                ImGui::Text("Tile size"); ImGui::SameLine(110 * ui);
                ImGui::TextDisabled("%d px", sstats_.tile_size_used);
                ImGui::Text("Tiles"); ImGui::SameLine(110 * ui);
                ImGui::TextDisabled("%d / %d%s%s", sstats_.tiles_run, sstats_.tiles_total,
                                    sstats_.used_fallback ? "  \xc2\xb7  fallback" : "",
                                    sstats_.capped ? "  \xc2\xb7  capped" : "");
            }
        }
        ImGui::Text("Detections"); ImGui::SameLine(110 * ui); ImGui::TextDisabled("%d", (int)dets_.size());
        if (!class_counts_.empty()) {
            ImGui::Spacing();
            for (int i = 0; i < (int)class_counts_.size(); ++i) {
                if (!class_counts_[i]) continue;
                const float* c = kPalette[i % 10];
                ImGui::ColorButton("##c", ImVec4(c[0],c[1],c[2],1),
                    ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop, ImVec2(11*ui,11*ui));
                ImGui::SameLine();
                const char* nm = i < cfg_.num_classes() ? cfg_.class_names[i].c_str() : "?";
                ImGui::Text("%s", nm); ImGui::SameLine(110 * ui); ImGui::TextDisabled("%d", class_counts_[i]);
            }
        }
    } else {
        ImGui::TextDisabled("Load a model and a source to see stats.");
    }
    end_card();

    ImGui::Dummy(ImVec2(0, 2));
    ImGui::TextDisabled("(c) 2026 Thomas Li");
}

void App::free_thumbs(const Platform& plat) {
    if (plat.release) for (auto& kv : thumbs_) plat.release(kv.second);
    thumbs_.clear();
}

// Lazily decode + upload a thumbnail for image `idx`. `budget` caps decodes per frame so
// scrolling never hitches. Returns nullptr while not yet available.
Texture* App::ensure_thumb(int idx, const Platform& plat, int& budget) {
    auto it = thumbs_.find(idx);
    if (it != thumbs_.end()) return it->second.id ? &it->second : nullptr;
    if (budget <= 0) return nullptr;
    --budget;
    Texture tex;
    // REDUCED_COLOR_8 decodes at 1/8 size: far cheaper than a full decode of a large JPEG
    cv::Mat small = cv::imread(folder_imgs_[idx], cv::IMREAD_REDUCED_COLOR_8);
    if (small.empty()) small = cv::imread(folder_imgs_[idx], cv::IMREAD_REDUCED_COLOR_4);
    if (!small.empty()) {
        const int maxw = 256;
        if (small.cols > maxw) {
            const double s = (double)maxw / small.cols;
            cv::resize(small, small, cv::Size(), s, s, cv::INTER_AREA);
        }
        cv::Mat rgba; cv::cvtColor(small, rgba, cv::COLOR_BGR2RGBA);
        plat.upload(rgba, tex);
    }
    thumbs_[idx] = tex;                       // cache even on failure, so we retry only once
    return tex.id ? &thumbs_[idx] : nullptr;
}

void App::draw_finder_icons(const Platform& plat) {
    const int n = (int)folder_imgs_.size();
    const float cell = icon_size_;
    const float thumbH = cell * 0.72f;
    const float capH = ImGui::GetTextLineHeight();
    const float cellH = thumbH + 4.f + capH;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const int cols = std::max(1, (int)((ImGui::GetContentRegionAvail().x + spacing) / (cell + spacing)));
    finder_cols_ = cols;                       // published for grid arrow-key navigation
    ImDrawList* dl = ImGui::GetWindowDrawList();
    int budget = 3;                            // decode at most 3 thumbnails per frame

    for (int i = 0; i < n; ++i) {
        if (i % cols) ImGui::SameLine();
        ImGui::PushID(i);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const bool sel = (i == cur_idx_);
        if (ImGui::InvisibleButton("##cell", ImVec2(cell, cellH)) && !sel) select_index(i, plat);
        const bool hov = ImGui::IsItemHovered();
        if (hov) ImGui::SetTooltip("%s", folder_imgs_[i].substr(folder_imgs_[i].find_last_of("/\\") + 1).c_str());

        // cell background: selected = accent, hovered = subtle
        if (sel)      dl->AddRectFilled(p, ImVec2(p.x + cell, p.y + cellH), IM_COL32(52, 110, 168, 210), 7.f);
        else if (hov) dl->AddRectFilled(p, ImVec2(p.x + cell, p.y + cellH), IM_COL32(255, 255, 255, 18), 7.f);

        // thumbnail, aspect-fit inside the thumb area
        const ImVec2 tp(p.x + 4, p.y + 4);
        const ImVec2 ts(cell - 8, thumbH - 4);
        Texture* th = ensure_thumb(i, plat, budget);
        if (th) {
            const float s = std::min(ts.x / th->w, ts.y / th->h);
            const ImVec2 d(th->w * s, th->h * s);
            const ImVec2 o(tp.x + (ts.x - d.x) * 0.5f, tp.y + (ts.y - d.y) * 0.5f);
            dl->AddImageRounded((ImTextureID)th->id, o, ImVec2(o.x + d.x, o.y + d.y),
                                ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 4.f);
        } else {   // placeholder while decoding
            dl->AddRectFilled(tp, ImVec2(tp.x + ts.x, tp.y + ts.y), IM_COL32(38, 42, 50, 255), 4.f);
        }

        // filename caption, clipped to the cell
        const std::string nm = folder_imgs_[i].substr(folder_imgs_[i].find_last_of("/\\") + 1);
        const ImVec2 cp(p.x + 4, p.y + thumbH + 2);
        dl->PushClipRect(ImVec2(p.x + 2, cp.y), ImVec2(p.x + cell - 2, cp.y + capH), true);
        dl->AddText(cp, sel ? IM_COL32(255, 255, 255, 255) : IM_COL32(168, 175, 185, 255), nm.c_str());
        dl->PopClipRect();

        if (sel && scroll_to_cur_) { ImGui::SetScrollHereY(0.5f); scroll_to_cur_ = false; }
        ImGui::PopID();
    }
}

void App::draw_finder_list(const Platform& plat) {
    int budget = 3;
    const float rowH = ImGui::GetTextLineHeight() * 1.9f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (int i = 0; i < (int)folder_imgs_.size(); ++i) {
        ImGui::PushID(i);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        const bool sel = (i == cur_idx_);
        if (ImGui::InvisibleButton("##row", ImVec2(w, rowH)) && !sel) select_index(i, plat);
        const bool hov = ImGui::IsItemHovered();
        if (sel)      dl->AddRectFilled(p, ImVec2(p.x + w, p.y + rowH), IM_COL32(52, 110, 168, 210), 6.f);
        else if (hov) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + rowH), IM_COL32(255, 255, 255, 18), 6.f);

        // small thumbnail on the left
        const float th = rowH - 8, tw = th * 1.4f;
        const ImVec2 tp(p.x + 4, p.y + 4);
        Texture* t = ensure_thumb(i, plat, budget);
        if (t) {
            const float s = std::min(tw / t->w, th / t->h);
            const ImVec2 d(t->w * s, t->h * s);
            const ImVec2 o(tp.x + (tw - d.x) * 0.5f, tp.y + (th - d.y) * 0.5f);
            dl->AddImageRounded((ImTextureID)t->id, o, ImVec2(o.x + d.x, o.y + d.y),
                                ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 3.f);
        } else {
            dl->AddRectFilled(tp, ImVec2(tp.x + tw, tp.y + th), IM_COL32(38, 42, 50, 255), 3.f);
        }

        const std::string nm = folder_imgs_[i].substr(folder_imgs_[i].find_last_of("/\\") + 1);
        const float tx = p.x + tw + 12;
        dl->PushClipRect(ImVec2(tx, p.y), ImVec2(p.x + w - 4, p.y + rowH), true);
        dl->AddText(ImVec2(tx, p.y + (rowH - ImGui::GetTextLineHeight()) * 0.5f),
                    sel ? IM_COL32(255, 255, 255, 255) : IM_COL32(200, 206, 214, 255), nm.c_str());
        dl->PopClipRect();

        if (sel && scroll_to_cur_) { ImGui::SetScrollHereY(0.5f); scroll_to_cur_ = false; }
        ImGui::PopID();
    }
}

void App::draw_filelist(const Platform& plat) {
    const float ui = ImGui::GetFontSize() / 17.0f;
    // ---- header: view toggle + count (or progress) + icon-size slider ----
    if (!folder_ready_) {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + finder_hdr_pad_);   // level with About
        const int done = fprogress_.load(), tot = (int)folder_imgs_.size();
        ImGui::Text("Inferring %d / %d", done, tot);
        ImGui::ProgressBar(tot ? (float)done / tot : 0.f, ImVec2(-1, 0));
    } else {
        // nudge down so this row sits level with the sidebar's About button
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + finder_hdr_pad_);
        const bool icons = finder_mode_ == FinderMode::Icons;
        if (ImGui::Button(icons ? "List" : "Icons"))
            finder_mode_ = icons ? FinderMode::List : FinderMode::Icons;
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%d images", (int)folder_imgs_.size());
        if (icons) {
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##iconsize", &icon_size_, 64.f, 200.f, "size %.0f");
        }
    }
    ImGui::Separator();
    ImGui::BeginChild("files", ImVec2(0, 0), false);
    if (finder_mode_ == FinderMode::Icons) draw_finder_icons(plat);
    else                                   draw_finder_list(plat);
    ImGui::EndChild();
}

// Draw one detection onto `dl`, mapped orig-px -> screen. `draw_shape=false` skips the box
// outline and draws only the label (masks-only overlay mode).
static void draw_box(ImDrawList* dl, const Detection& d, BoxStyle style, LabelMode labels,
                     const Config& cfg, ImVec2 origin, float scale, bool draw_shape = true) {
    const float x0 = origin.x + d.box.x * scale;
    const float y0 = origin.y + d.box.y * scale;
    const float x1 = x0 + d.box.width  * scale;
    const float y1 = y0 + d.box.height * scale;
    const ImU32 col = col_of(d.class_id);

    if (!draw_shape) {
        // no box outline; fall through to the label block below
    } else if (style == BoxStyle::Solid) {
        dl->AddRect(ImVec2(x0,y0), ImVec2(x1,y1), col, 2.f, 0, 2.f);
    } else if (style == BoxStyle::Neon) {
        dl->AddRect(ImVec2(x0-1,y0-1), ImVec2(x1+1,y1+1), col_of(d.class_id, 0.35f), 3.f, 0, 5.f);
        dl->AddRect(ImVec2(x0,y0), ImVec2(x1,y1), col, 3.f, 0, 2.f);
    } else { // Hud: thin full border + thick corner brackets
        dl->AddRect(ImVec2(x0,y0), ImVec2(x1,y1), col_of(d.class_id, 0.55f), 0.f, 0, 1.2f);  // thin sides
        const float len = std::min({ (x1-x0), (y1-y0), 22.f }) * 0.35f + 4.f;
        const float t = 2.5f;
        // TL
        dl->AddLine(ImVec2(x0,y0), ImVec2(x0+len,y0), col, t);
        dl->AddLine(ImVec2(x0,y0), ImVec2(x0,y0+len), col, t);
        // TR
        dl->AddLine(ImVec2(x1,y0), ImVec2(x1-len,y0), col, t);
        dl->AddLine(ImVec2(x1,y0), ImVec2(x1,y0+len), col, t);
        // BL
        dl->AddLine(ImVec2(x0,y1), ImVec2(x0+len,y1), col, t);
        dl->AddLine(ImVec2(x0,y1), ImVec2(x0,y1-len), col, t);
        // BR
        dl->AddLine(ImVec2(x1,y1), ImVec2(x1-len,y1), col, t);
        dl->AddLine(ImVec2(x1,y1), ImVec2(x1,y1-len), col, t);
    }

    if (labels == LabelMode::Off) return;
    char buf[96];
    const char* nm = d.class_id < cfg.num_classes() ? cfg.class_names[d.class_id].c_str() : "?";
    if (labels == LabelMode::Full) std::snprintf(buf, sizeof(buf), "%s %.2f", nm, d.conf);
    else                           std::snprintf(buf, sizeof(buf), "%s", nm);
    const ImVec2 ts = ImGui::CalcTextSize(buf);
    const float padX = ts.y * 0.5f, padY = ts.y * 0.22f;   // roomy chip padding
    const float chipH = ts.y + 2 * padY, chipW = ts.x + 2 * padX;
    const float rounding = chipH * 0.34f;                  // rounded pill
    float lx = x0, ly = y0 - chipH - 1.f;                  // sit just above the box top edge
    if (ly < origin.y) ly = y0 + 1.f;                      // flip inside if it would clip off-top
    const float a = (labels == LabelMode::Min) ? 0.72f : 0.86f;
    dl->AddRectFilled(ImVec2(lx, ly), ImVec2(lx + chipW, ly + chipH), col_of(d.class_id, a), rounding);
    dl->AddText(ImVec2(lx + padX, ly + padY), text_on(d.class_id), buf);
}

void App::draw_transport(const Platform& plat) {
    if (ImGui::Button(playing_ ? "Pause" : "Play ")) {
        playing_ = !playing_;
        if (playing_) reset_zoom();
    }
    ImGui::SameLine();
    int f = frame_idx_;
    ImGui::SetNextItemWidth(-150);
    if (ImGui::SliderInt("##seek", &f, 0, std::max(0, total_frames_ - 1), "frame %d")) {
        playing_ = false; seek_video(f, plat);
    }
    ImGui::SameLine();
    ImGui::Text("%d/%d  %.0ffps", frame_idx_ + 1, total_frames_, video_fps_);
}

void App::draw_camera_hud() {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 wp = ImGui::GetWindowPos();
    const float th = ImGui::GetTextLineHeight();
    const float padX = 12.f, padY = 7.f, gap = th * 0.9f, kern = 4.f;
    const float x0 = wp.x + 16, y0 = wp.y + 16;   // top-left inset

    char fps[16], ms[16], obj[16];
    std::snprintf(fps, sizeof(fps), "%.0f", cam_fps_ema_);
    std::snprintf(ms,  sizeof(ms),  "%.0f", cam_ms_ema_);
    std::snprintf(obj, sizeof(obj), "%d",  (int)dets_.size());
    auto w = [](const char* s) { return ImGui::CalcTextSize(s).x; };

    const float dotR = th * 0.22f, dotAdv = dotR * 2 + 7;
    const float total = dotAdv + w("LIVE") + gap
                      + w(fps) + kern + w("fps") + gap
                      + w(ms)  + kern + w("ms")  + gap
                      + w(obj) + kern + w("obj");

    // pill background
    dl->AddRectFilled(ImVec2(x0 - padX, y0 - padY),
                      ImVec2(x0 + total + padX, y0 + th + padY),
                      IM_COL32(15, 17, 20, 205), (th + 2 * padY) * 0.5f);

    const ImU32 white  = IM_COL32(236, 239, 243, 255), dim = IM_COL32(146, 152, 160, 255);
    const ImU32 green  = IM_COL32(88, 214, 128, 255),  cyan   = IM_COL32(64, 206, 218, 255);
    const ImU32 orange = IM_COL32(250, 162, 74, 255),  red    = IM_COL32(244, 76, 76, 255);

    float x = x0;
    dl->AddCircleFilled(ImVec2(x + dotR, y0 + th * 0.5f), dotR, red);
    x += dotAdv;
    dl->AddText(ImVec2(x, y0), white, "LIVE");                 x += w("LIVE") + gap;
    dl->AddText(ImVec2(x, y0), green,  fps); x += w(fps) + kern;
    dl->AddText(ImVec2(x, y0), dim,    "fps");                 x += w("fps") + gap;
    dl->AddText(ImVec2(x, y0), cyan,   ms);  x += w(ms) + kern;
    dl->AddText(ImVec2(x, y0), dim,    "ms");                  x += w("ms") + gap;
    dl->AddText(ImVec2(x, y0), orange, obj); x += w(obj) + kern;
    dl->AddText(ImVec2(x, y0), dim,    "obj");
}

void App::load_about_assets(const Platform& plat) {
    assets_loaded_ = true;   // attempt once; missing files just leave a lettered placeholder
    auto load = [&](const char* file, Texture& tex) {
        const std::string path = plat.exe_dir + "/assets/ack/" + file;
        cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty()) return;
        cv::Mat rgba;
        if (img.channels() == 4)      cv::cvtColor(img, rgba, cv::COLOR_BGRA2RGBA);
        else if (img.channels() == 3) cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);
        else if (img.channels() == 1) cv::cvtColor(img, rgba, cv::COLOR_GRAY2RGBA);
        else return;
        plat.upload(rgba, tex);
    };
    load("skywalker-lt.png", avatar_tex_);
    const char* keys[] = {"tencent", "ultralytics", "onnx", "mnn", "opencv", "imgui"};
    for (const char* k : keys) load((std::string(k) + ".png").c_str(), logos_[k]);
}

void App::draw_about(const Platform& plat) {
    if (!assets_loaded_) load_about_assets(plat);
    const float ui = ImGui::GetFontSize() / 17.0f;
    ImGui::SetNextWindowSize(ImVec2(560 * ui, 640 * ui), ImGuiCond_Appearing);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::Begin("About", &show_about_, ImGuiWindowFlags_NoCollapse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        auto link = [&](const char* label, const char* url) {
            ImGui::TextColored(ImVec4(0.37f, 0.64f, 0.88f, 1.f), "%s", label);
            if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            if (ImGui::IsItemClicked() && plat.open_url) plat.open_url(url);
        };
        auto logo_of = [&](const char* key) -> Texture* {
            if (!key) return nullptr;
            auto it = logos_.find(key);
            return (it != logos_.end() && it->second.id) ? &it->second : nullptr;
        };

        if (ImFont* hf = (ImFont*)plat.heading_font) {
            ImGui::PushFont(hf); ImGui::TextUnformatted(kAppName); ImGui::PopFont();
        } else ImGui::TextUnformatted(kAppName);
        ImGui::TextDisabled("%s", kAppSubtitle);
        ImGui::TextDisabled("Version %s", kAppVersion);
        ImGui::TextWrapped("%s", kAppTagline);

        ImGui::Dummy(ImVec2(0, 6 * ui));
        // author: round avatar + name/affiliation/link
        const float av = 64 * ui;
        if (avatar_tex_.id) {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            dl->AddImageRounded((ImTextureID)avatar_tex_.id, p, ImVec2(p.x + av, p.y + av),
                                ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, av * 0.5f);
            dl->AddCircle(ImVec2(p.x + av * 0.5f, p.y + av * 0.5f), av * 0.5f, IM_COL32(255, 255, 255, 45), 0, 1.5f);
            ImGui::Dummy(ImVec2(av + 10 * ui, av));
            ImGui::SameLine();
        }
        ImGui::BeginGroup();
        ImGui::TextUnformatted(kAuthor);
        ImGui::TextDisabled("%s", kAuthorAffil);
        link("github.com/skywalker-lt", kAuthorUrl);
        ImGui::EndGroup();
        link("Source - github.com/skywalker-lt/yolo-master-edge", kSourceUrl);

        ImGui::Dummy(ImVec2(0, 6 * ui));
        ImGui::SeparatorText("Acknowledgements");
        for (const auto& a : kAcks) {
            const float ls = 40 * ui;
            const ImVec2 p = ImGui::GetCursorScreenPos();
            Texture* lt = logo_of(a.logo);
            if (lt && lt->id) {
                dl->AddImageRounded((ImTextureID)lt->id, p, ImVec2(p.x + ls, p.y + ls),
                                    ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 7 * ui);
            } else {   // lettered placeholder tile
                dl->AddRectFilled(p, ImVec2(p.x + ls, p.y + ls), IM_COL32(58, 64, 76, 255), 7 * ui);
                const char L[2] = { a.name[0], 0 };
                const ImVec2 tz = ImGui::CalcTextSize(L);
                dl->AddText(ImVec2(p.x + (ls - tz.x) * 0.5f, p.y + (ls - tz.y) * 0.5f),
                            IM_COL32(176, 182, 192, 255), L);
            }
            ImGui::Dummy(ImVec2(ls + 10 * ui, ls));
            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::TextUnformatted(a.name);
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextWrapped("%s", a.blurb);
            ImGui::PopTextWrapPos();
            ImGui::PopStyleColor();
            link(a.repo, a.repo);
            ImGui::EndGroup();
            ImGui::Dummy(ImVec2(0, 6 * ui));
        }

        ImGui::Dummy(ImVec2(0, 4 * ui));
        if (ImGui::CollapsingHeader("License (AGPL-3.0)")) {
            ImGui::BeginChild("lic", ImVec2(0, 240 * ui), true,
                              ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::TextUnformatted(kLicenseText);
            ImGui::EndChild();
        }

        ImGui::Dummy(ImVec2(0, 4 * ui));
        if (ImGui::Button("Close", ImVec2(-1, 0))) show_about_ = false;
    }
    ImGui::End();
}

void App::draw_preview(const Platform& plat) {
    if (!img_tex_.id) {
        // empty state: centred both ways, slightly larger than body text
        const char* msg = is_video_ ? "No frame."
                                    : "Open an image, folder, or video - or start the webcam.";
        ImFont* f = ImGui::GetFont();
        const float fs = ImGui::GetFontSize() * 1.2f;
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const ImVec2 ts = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, msg);
        const ImVec2 pos(p0.x + (avail.x - ts.x) * 0.5f, p0.y + (avail.y - ts.y) * 0.5f);
        ImGui::GetWindowDrawList()->AddText(f, fs, pos,
                                           ImGui::GetColorU32(ImGuiCol_TextDisabled), msg);
        return;
    }
    const ImVec2 cur = ImGui::GetCursorScreenPos();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float ctrlH = is_video_ ? ImGui::GetFrameHeightWithSpacing() : 0.f;
    const float imgH = std::max(1.f, avail.y - ctrlH);
    float scale = std::min(avail.x / img_tex_.w, imgH / img_tex_.h);
    ImVec2 disp(img_tex_.w * scale, img_tex_.h * scale);
    ImVec2 origin(cur.x + (avail.x - disp.x) * 0.5f, cur.y + (imgH - disp.y) * 0.5f);

    // ---- zoom & pan (image always; video only while paused; never the live webcam) ----
    const bool zoom_ok = !is_cam_ && (!is_video_ || (video_ready_ && !playing_));
    ImGuiIO& io = ImGui::GetIO();
    if (zoom_ok) {
        // interaction item over the image area (created first so it owns the mouse there;
        // drawing below uses absolute draw-list coords, unaffected by the cursor)
        ImGui::SetCursorScreenPos(cur);
        ImGui::InvisibleButton("##zoomarea", ImVec2(std::max(1.f, avail.x), imgH));
        const ImVec2 C(cur.x + avail.x * 0.5f, cur.y + imgH * 0.5f);   // zoom anchor frame center
        auto clamp_pan = [&]() {
            const float mx = (zoom_ - 1.f) * avail.x * 0.5f, my = (zoom_ - 1.f) * imgH * 0.5f;
            pan_x_ = std::clamp(pan_x_, -mx, mx);
            pan_y_ = std::clamp(pan_y_, -my, my);
        };
        if (ImGui::IsItemHovered() && io.MouseWheel != 0.f) {
            // cursor-anchored wheel zoom: the pixel under the cursor stays put
            const float z = std::clamp(zoom_ * std::pow(1.25f, io.MouseWheel), 1.f, kZoomMax);
            const float r = z / zoom_;
            const float px = io.MousePos.x - C.x, py = io.MousePos.y - C.y;
            pan_x_ = px - (px - pan_x_) * r;
            pan_y_ = py - (py - pan_y_) * r;
            zoom_ = z;
            if (zoom_ < 1.02f) reset_zoom();          // snap home near 1x
            clamp_pan();
        }
        if (ImGui::IsItemActive() && zoom_ > 1.001f) {   // click-and-drag pan when zoomed
            pan_x_ += io.MouseDelta.x;
            pan_y_ += io.MouseDelta.y;
            clamp_pan();
        }
    }
    if (zoom_ > 1.001f) {   // apply about the frame center; boxes and masks follow for free
        const ImVec2 C(cur.x + avail.x * 0.5f, cur.y + imgH * 0.5f);
        origin.x = C.x + (origin.x - C.x) * zoom_ + pan_x_;
        origin.y = C.y + (origin.y - C.y) * zoom_ + pan_y_;
        scale *= zoom_;
        disp.x *= zoom_; disp.y *= zoom_;
    }

    const bool show_masks = seg_model_ && overlay_ != Overlay::Boxes && has_mask_;
    // masks-only hides the box outline but keeps the labels
    const bool box_shape  = !seg_model_ || overlay_ != Overlay::Masks;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(cur, ImVec2(cur.x + avail.x, cur.y + imgH), true);   // zoomed content stays inside
    const ImVec2 br(origin.x + disp.x, origin.y + disp.y);
    dl->AddImage((ImTextureID)img_tex_.id, origin, br);
    if (show_masks)   // seg overlay is same dims as the image -> same rect, alpha-blended by ImGui
        dl->AddImage((ImTextureID)mask_tex_.id, origin, br);
    for (const auto& d : dets_)
        draw_box(dl, d, style_, labels_, cfg_, origin, scale, box_shape);
    dl->PopClipRect();

    if (zoom_ > 1.001f) {   // zoom badge, top-right pill
        char zb[48];
        std::snprintf(zb, sizeof(zb), "%d%% - Ctrl+0 to reset", (int)std::lround(zoom_ * 100));
        const ImVec2 ts = ImGui::CalcTextSize(zb);
        const float padX = 8.f, padY = 3.f;
        const ImVec2 p1(cur.x + avail.x - 10.f, cur.y + 10.f);
        const ImVec2 p0(p1.x - ts.x - 2 * padX, p1.y);
        dl->AddRectFilled(p0, ImVec2(p1.x, p1.y + ts.y + 2 * padY),
                          IM_COL32(15, 17, 20, 205), (ts.y + 2 * padY) * 0.5f);
        dl->AddText(ImVec2(p0.x + padX, p0.y + padY), IM_COL32(220, 224, 230, 255), zb);
    }

    if (is_cam_) draw_camera_hud();

    if (is_video_) {
        ImGui::SetCursorScreenPos(ImVec2(cur.x, cur.y + imgH));
        if (video_ready_) {
            draw_transport(plat);
        } else {                                    // pre-inference progress
            const int done = vprogress_.load();
            const int tot = total_frames_;
            char ov[64];
            if (tot > 0) std::snprintf(ov, sizeof(ov), "Inferring  %d / %d", done, tot);
            else         std::snprintf(ov, sizeof(ov), "Inferring frame %d", done);
            ImGui::ProgressBar(tot > 0 ? (float)done / tot : -1.0f * (float)ImGui::GetTime(),
                               ImVec2(-1, 0), ov);
        }
    }
}

// ---------------- export (rendered frames + annotation labels) ----------------

static std::string path_stem(const std::string& p) {
    return std::filesystem::path(p).stem().string();
}

// The folder picker selects an EXISTING directory, so bulk exports create an enclosed,
// source-named subfolder inside it instead of dumping files at the picked root (the Mac
// runner gets this from the save panel's name-a-folder idiom). An existing non-empty
// folder of the same name is left alone: the export gets "name-2", "name-3", ...
static std::string enclosed_dir(const std::string& base, const std::string& name) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path dir = fs::path(base) / name;
    for (int i = 2; fs::exists(dir, ec) && !fs::is_empty(dir, ec); ++i)
        dir = fs::path(base) / (name + "-" + std::to_string(i));
    fs::create_directories(dir, ec);
    return dir.string();
}

// last path component as a display/export name ("C:/data/val/" -> "val")
static std::string dir_display_name(const std::string& p) {
    std::filesystem::path q(p);
    if (q.filename().empty()) q = q.parent_path();
    const std::string n = q.filename().string();
    return n.empty() ? "folder" : n;
}

// Current run's file dialect (WYSIWYG): seg polygons only when mask data actually exists.
bool App::export_dialect() const {
    if (is_video_) return !vproto_.empty();
    if (!folder_imgs_.empty())
        return model_is_seg_ && (slice_mode_ == SliceMode::Off || slice_masks_);
    return be_ && !be_->proto.empty();
}

// Compose the annotated frame CPU-side: seg overlay (per the Overlay mode) under draw()'s
// boxes. Rendered files use the shared draw() style; the ImGui-only hud/neon looks are a
// screen affordance, not an export format.
cv::Mat App::compose_render(const cv::Mat& bgr, const std::vector<Detection>& dets,
                            const std::vector<float>& proto, int pc, int ph, int pw,
                            const LetterboxInfo& lb) const {
    cv::Mat vis = bgr.clone();
    if (!proto.empty() && pc > 0 && overlay_ != Overlay::Boxes) {
        cv::Mat ov = seg_overlay(dets, proto, pc, ph, pw, lb, cfg_.imgsz, bgr.cols, bgr.rows);
        for (int y = 0; y < vis.rows; ++y) {
            const uint8_t* o = ov.ptr<uint8_t>(y);
            uint8_t* v = vis.ptr<uint8_t>(y);
            for (int x = 0; x < vis.cols; ++x) {
                const float a = o[x * 4 + 3] / 255.f;
                if (a <= 0) continue;
                v[x * 3 + 0] = (uint8_t)(v[x * 3 + 0] * (1 - a) + o[x * 4 + 2] * a);
                v[x * 3 + 1] = (uint8_t)(v[x * 3 + 1] * (1 - a) + o[x * 4 + 1] * a);
                v[x * 3 + 2] = (uint8_t)(v[x * 3 + 2] * (1 - a) + o[x * 4 + 0] * a);
            }
        }
    }
    if (!(!proto.empty() && overlay_ == Overlay::Masks))   // masks-only: no boxes
        draw(vis, dets, cfg_);
    return vis;
}

void App::begin_export(int total) {
    if (export_.joinable()) export_.join();   // reclaim a previous (finished) thread
    eprogress_ = 0; etotal_ = total;
    export_cancel_ = false; export_done_ = false;
    exporting_ = true;
    export_msg_.clear();
}

void App::poll_export() {
    if (!export_done_) return;
    if (export_.joinable()) export_.join();
    export_done_ = false;
    exporting_ = false;
}

void App::cancel_export() {
    export_cancel_ = true;
    if (export_.joinable()) export_.join();
    export_done_ = false;
    exporting_ = false;
}

void App::export_render_current(const Platform& plat) {
    if (!plat.save_file || img_bgr_.empty()) return;
    std::string def = path_stem(img_path_.empty() ? "frame" : img_path_);
    if (is_video_) def += "_" + std::to_string(frame_idx_);
    const std::string p = plat.save_file("Save rendered image",
        "JPEG\0*.jpg;*.jpeg\0All\0*.*\0", (def + ".jpg").c_str(), "jpg");
    if (p.empty()) return;
    cv::Mat vis;
    if (is_video_ && frame_idx_ < (int)vproto_.size())
        vis = compose_render(img_bgr_, dets_, vproto_[frame_idx_], vproto_c_, vproto_h_,
                             vproto_w_, vlb_);
    else if (!folder_imgs_.empty() && cur_idx_ >= 0 && cur_idx_ < (int)fcache_.size())
        vis = compose_render(img_bgr_, dets_, fcache_[cur_idx_].proto, fcache_[cur_idx_].pc,
                             fcache_[cur_idx_].ph, fcache_[cur_idx_].pw, fcache_[cur_idx_].lb);
    else if (be_)
        vis = compose_render(img_bgr_, dets_, be_->proto, be_->proto_c, be_->proto_h,
                             be_->proto_w, be_->cand_lb);
    else
        vis = compose_render(img_bgr_, dets_, {}, 0, 0, 0, LetterboxInfo{});
    export_msg_ = write_jpg(p, vis) ? "Saved " + std::filesystem::path(p).filename().string()
                                    : "Save failed";
}

void App::export_labels_current(const Platform& plat) {
    if (!plat.save_file || !be_ || img_bgr_.empty()) return;
    const annot::Format fmt = label_fmt_ == 1 ? annot::Format::CocoJSON
                            : label_fmt_ == 2 ? annot::Format::PascalVOC
                                              : annot::Format::YoloTXT;
    const std::string stem = path_stem(img_path_.empty() ? "image" : img_path_);
    const char* ext = annot::file_extension(fmt);
    char filter[96];
    std::snprintf(filter, sizeof(filter), "%s%c*.%s%cAll%c*.*%c",
                  annot::label(fmt), 0, ext, 0, 0, 0);   // embedded NULs for the Win32 filter
    const std::string p = plat.save_file("Export labels", filter,
                                         (stem + "." + ext).c_str(), ext);
    if (p.empty()) return;
    const bool dialect = export_dialect();
    annot::Image aimg;
    aimg.name = stem;
    aimg.width = img_bgr_.cols; aimg.height = img_bgr_.rows;
    aimg.instances = annotation_instances(dets_, dialect, be_->proto, be_->proto_c,
                                          be_->proto_h, be_->proto_w, be_->cand_lb, cfg_.imgsz);
    std::string body;
    switch (fmt) {
    case annot::Format::YoloTXT:  body = annot::yolo_lines(aimg, dialect); break;
    case annot::Format::PascalVOC:
        body = annot::voc_xml(aimg, cfg_.class_names,
                              std::filesystem::path(p).parent_path().filename().string());
        break;
    default:
        body = annot::coco_json({aimg}, {std::filesystem::path(img_path_).filename().string()},
                                cfg_.class_names, dialect);
    }
    std::ofstream f(p, std::ios::binary);
    f << body;
    bool ok = (bool)f;
    if (ok && fmt == annot::Format::YoloTXT) {   // classes.txt beside the label file
        std::ofstream cf((std::filesystem::path(p).parent_path() / "classes.txt").string(),
                         std::ios::binary);
        cf << annot::classes_txt(cfg_.class_names);
        ok = (bool)cf;
    }
    export_msg_ = ok ? std::to_string(aimg.instances.size()) + " instances -> "
                       + std::filesystem::path(p).filename().string()
                     : "Export failed";
}

void App::start_folder_render_export(const std::string& dir) {
    begin_export((int)folder_imgs_.size());
    Config c = cfg_;
    c.conf_thresh = conf_; c.iou_thresh = iou_;
    export_ = std::thread([this, dir, c] {
        int n = 0;
        std::set<std::string> stems;                 // 1.jpg + 1.png must not overwrite
        for (int i = 0; i < (int)fcache_.size() && !export_cancel_; ++i) {
            eprogress_ = i + 1;
            if (!fcache_[i].done) continue;
            cv::Mat bgr = cv::imread(folder_imgs_[i], cv::IMREAD_COLOR);
            if (bgr.empty()) continue;
            const FolderItem& it = fcache_[i];
            const auto dets = nms_and_cap(it.cands, c, it.ow, it.oh);
            cv::Mat vis = compose_render(bgr, dets, it.proto, it.pc, it.ph, it.pw, it.lb);
            if (write_jpg(dir + "/" + unique_stem(stems, path_stem(folder_imgs_[i])) + ".jpg", vis)) ++n;
        }
        export_msg_ = export_cancel_ ? "Cancelled"
                                     : std::to_string(n) + " rendered images -> "
                                       + std::filesystem::path(dir).filename().string();
        export_done_ = true;
    });
}

void App::start_video_render_export(const std::string& path) {
    begin_export(total_frames_);
    Config c = cfg_;
    c.conf_thresh = conf_; c.iou_thresh = iou_;
    export_ = std::thread([this, path, c] {
        cv::VideoCapture cap(video_path_);   // own capture: never touch the playback cap_
        cv::VideoWriter vw(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           video_fps_, cv::Size(vorig_w_, vorig_h_));
        if (!cap.isOpened() || !vw.isOpened()) {
            export_msg_ = "Cannot write video";
            export_done_ = true;
            return;
        }
        cv::Mat f;
        int n = 0;
        while (!export_cancel_ && n < (int)vcache_.size() && cap.read(f) && !f.empty()) {
            const auto dets = nms_and_cap(vcache_[n], c, vorig_w_, vorig_h_);
            const bool seg = n < (int)vproto_.size();
            cv::Mat vis = seg
                ? compose_render(f, dets, vproto_[n], vproto_c_, vproto_h_, vproto_w_, vlb_)
                : compose_render(f, dets, {}, 0, 0, 0, vlb_);
            vw.write(vis);
            eprogress_ = ++n;
        }
        vw.release();
        export_msg_ = export_cancel_ ? "Cancelled"
                                     : std::to_string(n) + " frames -> "
                                       + std::filesystem::path(path).filename().string();
        export_done_ = true;
    });
}

void App::start_folder_label_export(const std::string& dir) {
    begin_export((int)folder_imgs_.size());
    Config c = cfg_;
    c.conf_thresh = conf_; c.iou_thresh = iou_;
    const annot::Format fmt = label_fmt_ == 1 ? annot::Format::CocoJSON
                            : label_fmt_ == 2 ? annot::Format::PascalVOC
                                              : annot::Format::YoloTXT;
    const bool dialect = export_dialect();
    export_ = std::thread([this, dir, c, fmt, dialect] {
        AnnotationSink sink(fmt, dir, dir + "/annotations.coco.json", c.class_names, dialect);
        int ninst = 0;
        for (int i = 0; i < (int)fcache_.size() && !export_cancel_; ++i) {
            eprogress_ = i + 1;
            if (!fcache_[i].done) continue;
            const FolderItem& it = fcache_[i];
            annot::Image aimg;
            aimg.name = path_stem(folder_imgs_[i]);
            aimg.width = it.ow; aimg.height = it.oh;
            const auto dets = nms_and_cap(it.cands, c, it.ow, it.oh);
            aimg.instances = annotation_instances(dets, dialect, it.proto, it.pc, it.ph,
                                                  it.pw, it.lb, c.imgsz);
            ninst += (int)aimg.instances.size();
            sink.add(aimg, std::filesystem::path(folder_imgs_[i]).filename().string());
        }
        const auto r = sink.finish();
        export_msg_ = export_cancel_ ? "Cancelled"
                    : !r.error.empty() ? "Export failed: " + r.error
                    : std::to_string(r.images) + " images  \xc2\xb7  " + std::to_string(ninst)
                      + " instances -> " + std::filesystem::path(dir).filename().string();
        export_done_ = true;
    });
}

void App::start_video_label_export(const std::string& root) {
    begin_export(total_frames_);
    Config c = cfg_;
    c.conf_thresh = conf_; c.iou_thresh = iou_;
    const annot::Format fmt = label_fmt_ == 1 ? annot::Format::CocoJSON
                            : label_fmt_ == 2 ? annot::Format::PascalVOC
                                              : annot::Format::YoloTXT;
    const bool dialect = export_dialect();
    const int strides[] = {1, std::max(1, (int)std::lround(video_fps_)), 5, 10, 30};
    const int stride = strides[std::clamp(sampling_sel_, 0, 4)];
    export_ = std::thread([this, root, c, fmt, dialect, stride] {
        namespace fs = std::filesystem;
        std::error_code ec;
        const std::string frames_dir = (fs::path(root) / "frames").string();
        const std::string labels_dir = fmt == annot::Format::CocoJSON
            ? root : (fs::path(root) / "labels").string();
        fs::create_directories(frames_dir, ec);
        if (fmt != annot::Format::CocoJSON) fs::create_directories(labels_dir, ec);
        AnnotationSink sink(fmt, labels_dir, (fs::path(root) / "annotations.coco.json").string(),
                            c.class_names, dialect);
        cv::VideoCapture cap(video_path_);   // own capture (precedent: video_preinfer)
        if (!cap.isOpened()) { export_msg_ = "Cannot reopen video"; export_done_ = true; return; }
        const std::string stem = path_stem(video_path_);
        cv::Mat f;
        int n = 0, nimg = 0, ninst = 0;
        while (!export_cancel_ && n < (int)vcache_.size() && cap.read(f) && !f.empty()) {
            if (n % stride == 0) {
                char fn[64];
                std::snprintf(fn, sizeof(fn), "%s_%06d", stem.c_str(), n);
                write_jpg(frames_dir + "/" + fn + ".jpg", f);
                annot::Image aimg;
                aimg.name = fn;
                aimg.width = f.cols; aimg.height = f.rows;
                const auto dets = nms_and_cap(vcache_[n], c, vorig_w_, vorig_h_);
                const bool seg = dialect && n < (int)vproto_.size();
                aimg.instances = annotation_instances(dets, dialect,
                    seg ? vproto_[n] : std::vector<float>{}, seg ? vproto_c_ : 0,
                    vproto_h_, vproto_w_, vlb_, c.imgsz);
                ninst += (int)aimg.instances.size();
                ++nimg;
                sink.add(aimg, std::string("frames/") + fn + ".jpg", n + 1);
            }
            eprogress_ = ++n;
        }
        const auto r = sink.finish();
        export_msg_ = export_cancel_ ? "Cancelled"
                    : !r.error.empty() ? "Export failed: " + r.error
                    : std::to_string(nimg) + " frames  \xc2\xb7  " + std::to_string(ninst)
                      + " instances -> " + std::filesystem::path(root).filename().string();
        export_done_ = true;
    });
}

void App::draw_export_card(const Platform& plat) {
    const bool folder = !folder_imgs_.empty();
    const bool ready = folder ? folder_ready_ : (!is_video_ || video_ready_);
    begin_card("EXPORT");
    if (exporting_) {
        const int done = eprogress_.load(), tot = etotal_.load();
        char ov[48];
        std::snprintf(ov, sizeof(ov), "Exporting  %d / %d", done, tot);
        ImGui::ProgressBar(tot > 0 ? (float)done / tot : 0.f, ImVec2(-1, 0), ov);
        if (ImGui::Button("Cancel export", ImVec2(-1, 0))) cancel_export();
        end_card();
        return;
    }
    ImGui::BeginDisabled(!ready);
    field_label("Rendered");
    if (is_video_) {
        if (ImGui::Button("This frame", ImVec2(-1, 0))) export_render_current(plat);
        if (ImGui::Button("All (annotated video)...", ImVec2(-1, 0)) && plat.save_file) {
            const std::string p = plat.save_file("Save annotated video",
                "MP4\0*.mp4\0All\0*.*\0", (path_stem(video_path_) + "_annotated.mp4").c_str(), "mp4");
            if (!p.empty()) start_video_render_export(p);
        }
    } else if (folder) {
        if (ImGui::Button("This image", ImVec2(-1, 0))) export_render_current(plat);
        if (ImGui::Button("All...", ImVec2(-1, 0)) && plat.open_folder) {
            const std::string d = plat.open_folder("Choose where to save the rendered images");
            if (!d.empty())
                start_folder_render_export(enclosed_dir(d, dir_display_name(folder_path_) + "-rendered"));
        }
    } else {
        if (ImGui::Button("Save image...", ImVec2(-1, 0))) export_render_current(plat);
    }
    ImGui::Spacing();
    field_label("Labels");
    const char* fmts[] = {"YOLO (TXT)", "COCO (JSON)", "Pascal VOC (XML)"};
    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("##labelfmt", &label_fmt_, fmts, IM_ARRAYSIZE(fmts));
    if (is_video_) {
        field_label("Frames");
        const char* samps[] = {"Every frame", "One per second", "Every 5th", "Every 10th",
                               "Every 30th"};
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("##sampling", &sampling_sel_, samps, IM_ARRAYSIZE(samps));
    }
    if (ImGui::Button("Export labels...", ImVec2(-1, 0))) {
        if (is_video_) {
            if (plat.open_folder) {
                const std::string d = plat.open_folder("Choose where to save frames + labels");
                if (!d.empty())
                    start_video_label_export(enclosed_dir(d, path_stem(video_path_) + "-labels"));
            }
        } else if (folder) {
            if (plat.open_folder) {
                const std::string d = plat.open_folder("Choose where to save the labels");
                if (!d.empty())
                    start_folder_label_export(enclosed_dir(d, dir_display_name(folder_path_) + "-labels"));
            }
        } else {
            export_labels_current(plat);
        }
    }
    ImGui::EndDisabled();
    if (!export_msg_.empty()) ImGui::TextDisabled("%s", export_msg_.c_str());
    end_card();
}

} // namespace gui
