// Portable GUI application state + logic (no Win32 / no D3D11).
// The platform layer (main_win.cpp) provides texture upload + a file-open dialog
// through the Platform struct, then calls App::frame() once per rendered frame.
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "yolomaster.hpp"
#include "slicing.hpp"
#include "annotate_export.hpp"

namespace gui {

// A GPU texture handle owned by the platform layer. `id` is an ImTextureID (an SRV
// on D3D11) stored type-erased so this header stays platform-free.
struct Texture {
    void* id  = nullptr;   // ImTextureID (an ID3D11ShaderResourceView* on D3D11)
    void* tex = nullptr;   // the underlying ID3D11Texture2D*, kept for cheap in-place updates
    int   w = 0, h = 0;
};

// Services the platform layer injects into the portable app.
struct Platform {
    // Upload/replace `rgba` (CV_8UC4) into `tex`, (re)allocating if size changed. false on failure.
    std::function<bool(const cv::Mat& rgba, Texture& tex)> upload;
    // Native open-file dialog; returns "" if cancelled. `filter` is a platform-specific spec.
    std::function<std::string(const char* title, const char* filter)> open_file;
    // Native save-file dialog; returns "" if cancelled. `def_name` seeds the file name,
    // `def_ext` (no dot) is appended when the user types none.
    std::function<std::string(const char* title, const char* filter,
                              const char* def_name, const char* def_ext)> save_file;
    // Native folder-picker; returns "" if cancelled.
    std::function<std::string(const char* title)> open_folder;
    // Open a URL in the default browser.
    std::function<void(const char* url)> open_url;
    // Free a texture's GPU resources (safe on an empty Texture).
    std::function<void(Texture& tex)> release;
    void* heading_font = nullptr;   // ImFont* (type-erased): larger face for the sidebar title
    std::string exe_dir;   // directory of the running exe (for loading bundled assets/)
};

enum class BoxStyle { Hud, Solid, Neon };
enum class LabelMode { Full, Min, Off };
enum class Preprocess { Letterbox, Stretch };
enum class Overlay { Both, Masks, Boxes };   // segmentation: what to show (masks / boxes / both)
enum class Device { CPU, GPU };              // GPU = onnx:CUDA, ncnn:Vulkan, mnn:OpenCL
enum class FinderMode { Icons, List };       // folder navigator: thumbnail grid or filename list/Vulkan

class App {
public:
    ~App();
    void frame(const Platform& plat);   // draw one ImGui frame (sidebar + preview)

private:
    // ---- backend / model ----
    char        model_path_[1024] = "";
    int         backend_sel_ = 0;       // 0 auto,1 onnx,2 ncnn,3 mnn
    int         threads_ = 4;
    std::unique_ptr<yolomaster::Backend> be_;
    std::string be_name_, be_err_, be_ep_, be_note_;
    yolomaster::Config cfg_;            // display config (real conf/iou live here)

    // ---- source image ----
    cv::Mat     img_bgr_;              // original loaded image
    Texture     img_tex_;             // uploaded to GPU once per load
    std::string img_path_, load_err_;

    // ---- folder-batch (pre-infer every image up front, then browse from cache) ----
    std::vector<std::string> folder_imgs_;   // sorted image paths (empty = single-image mode)
    int         cur_idx_ = -1;               // index into folder_imgs_ of the shown image
    std::string folder_path_;
    bool        scroll_to_cur_ = false;      // request the file list to scroll the current item into view
    FinderMode  finder_mode_ = FinderMode::Icons;
    float       icon_size_ = 108.f;          // 64..200, like the Mac runner
    int         finder_cols_ = 3;            // columns in icon view (drives grid arrow-key nav)
    float       finder_hdr_pad_ = 0.f;       // y-offset so the Icons/List button lines up with About
    std::map<int, Texture> thumbs_;          // lazily decoded thumbnails, keyed by image index
    struct FolderItem {                      // cached inference result for one image
        std::vector<yolomaster::RawDet> cands;
        std::vector<float> proto; int pc = 0, ph = 0, pw = 0;
        yolomaster::LetterboxInfo lb; int ow = 0, oh = 0;
        double ms = 0;                       // model-only inference time for this image
        bool done = false;
    };
    std::vector<FolderItem> fcache_;
    double      fmean_ms_ = 0, fmin_ms_ = 0, fmax_ms_ = 0;   // model-only, over the folder
    double      fwall_s_ = 0;                                // wall clock for the whole pass
    int         fcount_ = 0;                                 // images actually inferred
    bool        folder_ready_ = false;
    std::thread finfer_;
    std::atomic<int>  fprogress_{0};
    std::atomic<bool> finfer_done_{false}, finfer_cancel_{false};

    // ---- video (Mac model: pre-infer every frame up front, then play from cache) ----
    cv::VideoCapture cap_;
    bool        is_video_ = false;
    bool        playing_ = false;
    int         frame_idx_ = 0, total_frames_ = 0;
    double      video_fps_ = 30.0, play_accum_ = 0.0;   // play_accum_ paces playback to real time
    std::string video_path_;
    std::vector<std::vector<yolomaster::RawDet>> vcache_;   // per-frame pre-NMS candidates
    std::vector<std::vector<float>>              vproto_;   // per-frame proto (seg only)
    int         vproto_c_ = 0, vproto_h_ = 0, vproto_w_ = 0;
    yolomaster::LetterboxInfo vlb_; int vorig_w_ = 0, vorig_h_ = 0;
    bool        video_ready_ = false;                   // pre-inference finished
    std::thread vinfer_;
    std::atomic<int>  vprogress_{0};
    std::atomic<bool> vinfer_done_{false}, vinfer_cancel_{false};
    double      vinfer_ms_ = 0.0;                       // mean model ms over the clip

    // ---- webcam ----
    cv::VideoCapture cam_;
    bool        is_cam_ = false;
    bool        cam_mirror_ = true;
    double      cam_fps_ema_ = 0.0, cam_ms_ema_ = 0.0;  // EMA-smoothed feed fps / inference ms

    // ---- async inference worker (video + webcam): decode/display on main, infer off-thread ----
    std::thread worker_;
    std::mutex  job_mtx_;
    std::condition_variable job_cv_;
    cv::Mat     job_frame_;
    yolomaster::Config job_cfg_;
    bool        job_pending_ = false, worker_quit_ = false;
    std::atomic<bool> worker_busy_{false};
    std::mutex  res_mtx_;
    struct AsyncResult {
        std::vector<yolomaster::Detection> dets;
        std::vector<yolomaster::RawDet>    cands;        // for cheap re-NMS when paused
        std::vector<float> proto; int pc = 0, ph = 0, pw = 0;
        yolomaster::LetterboxInfo lb; int ow = 0, oh = 0;
        double inf_ms = 0;
        bool seg = false;
    };
    AsyncResult ares_;
    bool        ares_dirty_ = false;
    bool        async_mode_ = false;   // true while a video/webcam worker owns the backend
    bool        seg_model_ = false;    // stable is_seg() flag (safe to read from the main thread)
    bool        model_is_seg_ = false; // the loaded MODEL has a seg head (survives sliced runs
                                       // that drop masks; cleared only on load_model)

    // ---- results ("forward once, tune cheap") ----
    std::vector<yolomaster::Detection> dets_;
    Texture mask_tex_;                // segmentation overlay (RGBA), rebuilt when dets change
    bool  has_mask_ = false;
    bool  need_reinfer_ = false;      // model/source/preprocess/threads changed
    bool  need_renms_   = false;      // conf/iou changed (cheap: re-run nms on cached candidates)
    bool  need_overlay_ = false;      // dets changed -> recomposite the seg mask overlay
    double pre_ms_ = 0, inf_ms_ = 0, post_ms_ = 0;
    std::vector<int> class_counts_;

    // ---- appearance ----
    float      conf_ = 0.25f, iou_ = 0.50f;
    BoxStyle   style_ = BoxStyle::Hud;
    LabelMode  labels_ = LabelMode::Full;
    Preprocess prep_ = Preprocess::Letterbox;
    Overlay    overlay_ = Overlay::Both;
    Device     device_ = Device::CPU;

    // ---- slicing (images + folders; video/webcam stay single-pass) ----
    yolomaster::SliceMode slice_mode_ = yolomaster::SliceMode::Off;
    int  tile_size_req_ = 0;            // requested tile edge, source px (0 = model imgsz)
    int  max_det_req_ = 300;            // NMS keep cap, tiled runs only (single-pass stays 300)
    bool slice_masks_ = false;          // keep the global pass's masks in sliced runs
    bool sliced_run_ = false;           // current single-image cache came from a sliced run
    yolomaster::SliceOutput sstats_;    // per-image slicing stats (single-image mode)
    yolomaster::TileStats   tstats_;    // folder aggregate (published live by finfer thread)
    yolomaster::SliceConfig slice_config() const;

    // ---- zoom/pan (image + paused video; header stays ImGui-free -> plain floats) ----
    float zoom_ = 1.f;                  // 1..kZoomMax
    float pan_x_ = 0.f, pan_y_ = 0.f;   // pan offset in preview points
    static constexpr float kZoomMax = 8.f;
    void reset_zoom() { zoom_ = 1.f; pan_x_ = pan_y_ = 0.f; }

    // ---- annotation / rendered export ----
    int  label_fmt_ = 0;                // 0 YOLO (TXT), 1 COCO (JSON), 2 Pascal VOC (XML)
    int  sampling_sel_ = 1;             // 0 every frame, 1 one per second, 2/3/4 every 5/10/30th
    bool exporting_ = false;
    std::thread export_;
    std::atomic<int>  eprogress_{0}, etotal_{0};
    std::atomic<bool> export_done_{false}, export_cancel_{false};
    std::string export_msg_;            // completion / error line under the card

    static constexpr float kConfFloor = 0.05f;   // cache candidates down to here

    void load_model(const Platform& plat);
    void load_default_model(const Platform& plat);   // bundled model, auto-loaded on first frame
    bool tried_default_ = false;
    void load_image(const std::string& path, const Platform& plat);
    void load_folder(const std::string& dir, const Platform& plat);
    void select_index(int i, const Platform& plat);   // show folder_imgs_[i] from cache
    void folder_preinfer(std::vector<std::string> paths, yolomaster::Config c,
                         yolomaster::SliceConfig sc);  // background: infer all
    void show_folder_item(int idx, const Platform& plat);     // decode pixels + cached overlay
    void overlay_folder_item(int idx, const Platform& plat);  // re-NMS + mask from cache (no decode)
    void close_folder(const Platform* plat = nullptr);
    void open_media(const std::string& path, const Platform& plat);   // autodetect image vs video
    void open_video(const std::string& path, const Platform& plat);
    void show_frame(const cv::Mat& frame, const Platform& plat);   // upload frame texture
    void video_preinfer(std::string path, yolomaster::Config c);   // background: infer every frame
    void render_video_frame(int idx, const Platform& plat);        // decode + display + overlay from cache
    void overlay_from_cache(int idx, const Platform& plat);        // re-NMS + mask only (no decode)
    void seek_video(int idx, const Platform& plat);   // scrub to a frame
    void close_video();
    // webcam + async worker
    void open_camera(const Platform& plat);
    void close_camera(const Platform* plat = nullptr);
    void clear_preview(const Platform* plat);   // blank the preview + drop stale results
    void start_worker();
    void stop_worker();
    void worker_loop();
    void submit_job(const cv::Mat& frame);            // non-blocking; drops if worker busy
    void consume_async(const Platform& plat);         // pull latest result -> dets_ / overlay
    void build_overlay(const std::vector<yolomaster::Detection>& dets, const std::vector<float>& proto,
                       int pc, int ph, int pw, const yolomaster::LetterboxInfo& lb,
                       int ow, int oh, const Platform& plat);
    std::string gpu_device_str() const;               // backend-specific GPU token for the Device
    void run_inference();             // full forward pass -> cache candidates
    void recompute_nms();             // cheap: nms_and_cap on cached candidates
    void rebuild_overlay(const Platform& plat);   // recomposite seg masks -> mask_tex_
    void draw_sidebar(const Platform& plat);
    void draw_export_card(const Platform& plat);     // EXPORT card (rendered + labels)
    // rendered-frame composition (CPU): frame + seg overlay (per the Overlay mode) + boxes
    cv::Mat compose_render(const cv::Mat& bgr, const std::vector<yolomaster::Detection>& dets,
                           const std::vector<float>& proto, int pc, int ph, int pw,
                           const yolomaster::LetterboxInfo& lb) const;
    bool export_dialect() const;                     // seg polygons in label files right now?
    void export_render_current(const Platform& plat);   // save-file dialog + write current frame
    void export_labels_current(const Platform& plat);   // single image -> one label file
    void begin_export(int total);                    // shared thread-state setup
    void start_folder_render_export(const std::string& dir);
    void start_video_render_export(const std::string& path);
    void start_folder_label_export(const std::string& dir);
    void start_video_label_export(const std::string& root);
    void poll_export();                              // join the finished export thread
    void cancel_export();                            // request + join (Cancel button, dtor)
    void draw_filelist(const Platform& plat);   // folder-batch navigator panel
    void draw_finder_icons(const Platform& plat);
    void draw_finder_list(const Platform& plat);
    Texture* ensure_thumb(int idx, const Platform& plat, int& budget);
    void free_thumbs(const Platform& plat);
    void draw_preview(const Platform& plat);
    void draw_transport(const Platform& plat);  // video play/pause + scrubber
    void draw_camera_hud();                     // webcam fps/latency/objects overlay
    void draw_about(const Platform& plat);      // About / Acknowledgements / License window
    void load_about_assets(const Platform& plat);
    bool show_about_ = false;
    bool assets_loaded_ = false;
    Texture avatar_tex_;                             // round author avatar
    std::map<std::string, Texture> logos_;           // ack logos keyed by Ack::logo
};

} // namespace gui
