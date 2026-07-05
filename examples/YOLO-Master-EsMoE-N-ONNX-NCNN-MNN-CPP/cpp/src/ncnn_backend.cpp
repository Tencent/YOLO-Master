#include "ncnn_backend.hpp"
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <filesystem>

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

NcnnBackend::NcnnBackend(const std::string& param_path, const std::string& bin_path, int threads)
    : threads_(threads) {
    net_.opt.num_threads = threads;
    if (net_.load_param(param_path.c_str()) != 0)
        throw std::runtime_error("ncnn: failed to load param " + param_path);
    if (net_.load_model(bin_path.c_str()) != 0)
        throw std::runtime_error("ncnn: failed to load bin " + bin_path);

    // auto-read ultralytics metadata sidecar (class names + imgsz)
    const std::string dir = std::filesystem::path(param_path).parent_path().string();
    std::vector<std::string> nm; int mi = 0;
    if (meta::read_ncnn_yaml(dir + "/metadata.yaml", nm, mi)) { meta_names = nm; meta_imgsz = mi; }
    // YOLO-Master ncnn graphs bake the attention token counts at the training size,
    // so the input size is effectively fixed.
    fixed_imgsz = meta_imgsz;
}

std::vector<Detection> NcnnBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    // ---- preprocess: letterbox -> ncnn RGB /255 ----
    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = letterbox(bgr, cfg.imgsz, lb);
    ncnn::Mat in = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_BGR2RGB,
                                          padded.cols, padded.rows);
    const float mean[3] = {0.f, 0.f, 0.f};
    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean, norm);
    pre_ms = ms_since(t0);

    // ---- inference ----
    auto t1 = clk::now();
    ncnn::Extractor ex = net_.create_extractor();  // uses net_.opt.num_threads set in ctor
    ex.input(in_blob_.c_str(), in);
    ncnn::Mat out;
    ex.extract(out_blob_.c_str(), out);
    infer_ms = ms_since(t1);

    // ---- reshape to channel-major [feat_dim x num_anchors] then decode ----
    auto t2 = clk::now();
    const int feat = 4 + cfg.num_classes();
    int feat_dim, num_anchors;
    std::vector<float> buf;
    if (out.h == feat) {                       // rows = features (expected)
        feat_dim = out.h; num_anchors = out.w;
        buf.resize(static_cast<size_t>(feat_dim) * num_anchors);
        for (int f = 0; f < feat_dim; ++f)
            std::memcpy(buf.data() + static_cast<size_t>(f) * num_anchors,
                        out.row(f), num_anchors * sizeof(float));
    } else {                                   // rows = anchors -> transpose
        feat_dim = feat; num_anchors = out.h;
        buf.resize(static_cast<size_t>(feat_dim) * num_anchors);
        for (int a = 0; a < num_anchors; ++a) {
            const float* r = out.row(a);
            for (int f = 0; f < feat_dim; ++f)
                buf[static_cast<size_t>(f) * num_anchors + a] = r[f];
        }
    }
    auto dets = decode(buf.data(), feat_dim, num_anchors, cfg, lb);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
