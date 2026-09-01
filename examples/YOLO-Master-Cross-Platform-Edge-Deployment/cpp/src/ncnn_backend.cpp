#include "ncnn_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <filesystem>

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

NcnnBackend::NcnnBackend(const std::string& param_path, const std::string& bin_path, int threads,
                         bool use_vulkan)
    : threads_(threads) {
    net_.opt.num_threads = threads;
    net_.opt.use_vulkan_compute = use_vulkan;   // GPU path (the -shared prebuilt is Vulkan-enabled)
    if (use_vulkan) {                           // fp16 on the GPU: big speedup, negligible accuracy loss
        net_.opt.use_fp16_packed = true;
        net_.opt.use_fp16_storage = true;
        net_.opt.use_fp16_arithmetic = true;
    }
    active_ep = use_vulkan ? "ncnn-Vulkan" : "cpu";
    if (net_.load_param(param_path.c_str()) != 0)
        throw std::runtime_error("ncnn: failed to load param " + param_path);
    if (net_.load_model(bin_path.c_str()) != 0)
        throw std::runtime_error("ncnn: failed to load bin " + bin_path);

    // auto-read ultralytics metadata sidecar (class names + imgsz)
    const std::filesystem::path param_fs(param_path);
    const std::string dir = (param_fs.parent_path().empty()
                                 ? std::filesystem::path(".")
                                 : param_fs.parent_path()).string();
    std::vector<std::string> nm; int mi = 0;
    std::string metadata_input, metadata_output, metadata_proto;
    if (meta::read_ncnn_yaml(dir + "/metadata.yaml", nm, mi,
                             &metadata_input, &metadata_output, &metadata_proto)) {
        meta_names = nm; meta_imgsz = mi;
        if (!metadata_input.empty()) in_blob_ = metadata_input;
        if (!metadata_output.empty()) out_blob_ = metadata_output;
        if (!metadata_proto.empty()) out_proto_ = metadata_proto;
    }
    // YOLO-Master ncnn graphs bake the attention token counts at the training size,
    // so the input size is effectively fixed.
    fixed_imgsz = meta_imgsz;
}

std::vector<Detection> NcnnBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    // ---- preprocess: letterbox -> ncnn RGB /255 ----
    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = preprocess(bgr, cfg.imgsz, cfg.stretch, lb);
    ncnn::Mat in = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_BGR2RGB,
                                          padded.cols, padded.rows);
    const float mean[3] = {0.f, 0.f, 0.f};
    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean, norm);
    pre_ms = ms_since(t0);

    // ---- inference ----
    auto t1 = clk::now();
    ncnn::Extractor ex = net_.create_extractor();  // uses net_.opt.num_threads set in ctor
    if (ex.input(in_blob_.c_str(), in) != 0)
        throw std::runtime_error("ncnn: failed to set input blob '" + in_blob_ + "'");
    ncnn::Mat out, pm;
    if (ex.extract(out_blob_.c_str(), out) != 0 || out.empty())
        throw std::runtime_error("ncnn: failed to extract detection blob '" + out_blob_ + "'");
    // A detection graph has no proto output.  Treat a missing optional proto
    // as expected, but never hide a failed detection extraction.
    if (!out_proto_.empty()) (void)ex.extract(out_proto_.c_str(), pm);
    infer_ms = ms_since(t1);

    // ---- reshape to channel-major [feat_dim x num_anchors] then decode ----
    // feat << anchors always (e.g. 14/116 vs 8400), so the smaller axis is the feature dim.
    auto t2 = clk::now();
    int feat_dim, num_anchors;
    // ncnn represents a leading singleton batch either as a 2-D Mat or as a
    // 3-D Mat with c=1, depending on the exporter/version.  Both layouts are
    // equivalent for the shared decoder; reject true multi-channel tensors
    // instead of flattening them with an ambiguous stride.
    if ((out.dims != 2 && out.dims != 3) || out.w <= 0 || out.h <= 0 || out.c != 1)
        throw std::runtime_error("ncnn: detection blob must be a non-empty 2-D/3-D float matrix with singleton batch");
    if (out.elemsize != sizeof(float))
        throw std::runtime_error("ncnn: detection blob must use float32 elements");
    std::vector<float> buf;
    if (out.h <= out.w) {                      // rows = features (expected, channel-major)
        feat_dim = out.h; num_anchors = out.w;
        buf.resize(static_cast<size_t>(feat_dim) * num_anchors);
        for (int f = 0; f < feat_dim; ++f)
            std::memcpy(buf.data() + static_cast<size_t>(f) * num_anchors,
                        out.row(f), num_anchors * sizeof(float));
    } else {                                   // rows = anchors -> transpose
        feat_dim = out.w; num_anchors = out.h;
        buf.resize(static_cast<size_t>(feat_dim) * num_anchors);
        for (int a = 0; a < num_anchors; ++a) {
            const float* r = out.row(a);
            for (int f = 0; f < feat_dim; ++f)
                buf[static_cast<size_t>(f) * num_anchors + a] = r[f];
        }
    }
    if (!std::all_of(buf.begin(), buf.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("ncnn: detection blob contains NaN or Inf");
    const int expected_features = 4 + cfg.num_classes();
    bool has_objectness = false;
    int expected_masks = 0;
    if (!pm.empty()) {
        if (pm.dims != 3 || pm.c <= 0 || pm.h <= 0 || pm.w <= 0 ||
            pm.elemsize != sizeof(float))
            throw std::runtime_error("ncnn: prototype blob must be a non-empty float32 3-D tensor");
        expected_masks = pm.c;
        if (feat_dim == expected_features + expected_masks) {
            has_objectness = false;
        } else if (feat_dim == expected_features + 1 + expected_masks) {
            has_objectness = true;
        } else {
            throw std::runtime_error("ncnn: detection/prototype feature counts are incompatible");
        }
    } else if (feat_dim == expected_features + 1) {
        has_objectness = true;
    } else if (feat_dim != expected_features) {
        throw std::runtime_error(
            "ncnn: detection blob has extra feature channels but no compatible prototype blob");
    }
    candidates = decode_candidates(buf.data(), feat_dim, num_anchors, cfg, lb,
                                   has_objectness);
    cand_orig_w = lb.orig_w; cand_orig_h = lb.orig_h; cand_lb = lb;
    proto.clear(); proto_c = proto_h = proto_w = 0;
    if (!pm.empty()) {                         // segmentation proto [c=nm, h=mh, w=mw]
        proto_c = pm.c; proto_h = pm.h; proto_w = pm.w;
        const size_t plane = static_cast<size_t>(proto_h) * proto_w;
        proto.resize(static_cast<size_t>(proto_c) * plane);
        for (int c = 0; c < proto_c; ++c)
            std::memcpy(proto.data() + c * plane, pm.channel(c), plane * sizeof(float));
        if (!std::all_of(proto.begin(), proto.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("ncnn: prototype blob contains NaN or Inf");
    }
    auto dets = nms_and_cap(candidates, cfg, lb.orig_w, lb.orig_h);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
