#include "ncnn_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

namespace {

// NCNN does not expose the graph's blob table through the small C++ API used
// by this example.  Parse only the stable layer header portion of the param
// file so that arbitrary pnnx blob names can be validated without relying on
// the historical in0/out0/out1 convention.  Parameters after the top-blob
// list are intentionally ignored.
struct GraphBlobs {
    std::set<std::string> all;
    std::set<std::string> produced;
    std::vector<std::string> inputs;
    std::vector<std::string> tops;
    std::set<std::string> bottoms;
    bool parsed = false;
};

static GraphBlobs inspect_param_graph(const std::string& path) {
    GraphBlobs graph;
    std::ifstream file(std::filesystem::u8path(path));
    if (!file) return graph;

    std::string line;
    bool header_seen = false;
    while (std::getline(file, line)) {
        // NCNN param files are ASCII.  Keep parsing tolerant of CRLF and
        // comments so files produced on Windows have the same semantics.
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        line = line.substr(first);
        if (!header_seen) {
            // The first non-comment line is the magic/version and layer count
            // (for example ``7767517`` or ``7767518 235 237``).
            header_seen = true;
            continue;
        }
        std::istringstream row(line);
        std::string type, layer_name;
        int bottom_count = 0, top_count = 0;
        if (!(row >> type >> layer_name >> bottom_count >> top_count)) continue;
        (void)layer_name;  // layer names are not needed for endpoint resolution
        if (bottom_count < 0 || top_count <= 0 || bottom_count > 100000 || top_count > 100000)
            continue;
        std::vector<std::string> bottoms(static_cast<size_t>(bottom_count));
        std::vector<std::string> tops(static_cast<size_t>(top_count));
        bool complete = true;
        for (auto& name : bottoms) {
            if (!(row >> name) || name.empty()) { complete = false; break; }
        }
        if (!complete) continue;
        for (auto& name : tops) {
            if (!(row >> name) || name.empty()) { complete = false; break; }
        }
        if (!complete) continue;
        for (const auto& name : bottoms) {
            graph.all.insert(name);
            graph.bottoms.insert(name);
        }
        for (const auto& name : tops) {
            graph.all.insert(name);
            graph.produced.insert(name);
            graph.tops.push_back(name);
        }
        if (type == "Input") {
            graph.inputs.insert(graph.inputs.end(), tops.begin(), tops.end());
        }
    }
    // Preserve first appearance while removing duplicate tops.  A terminal
    // blob is one which is never consumed as a later layer bottom.
    std::set<std::string> seen;
    std::vector<std::string> unique_tops;
    for (const auto& name : graph.tops) {
        if (seen.insert(name).second && !graph.bottoms.count(name)) unique_tops.push_back(name);
    }
    graph.tops.swap(unique_tops);
    if (graph.inputs.empty()) {
        // A few hand-written NCNN graphs omit an explicit Input layer. Their
        // external inputs are bottom blobs that are never produced by another
        // layer; infer this only when the set is unambiguous.
        std::set<std::string> external;
        for (const auto& name : graph.bottoms) {
            if (!graph.produced.count(name)) external.insert(name);
        }
        graph.inputs.assign(external.begin(), external.end());
    }
    graph.parsed = header_seen && !graph.all.empty();
    return graph;
}

static bool contains(const std::set<std::string>& values, const std::string& value) {
    return !value.empty() && values.find(value) != values.end();
}

static bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

} // namespace

NcnnBackend::NcnnBackend(const std::string& param_path, const std::string& bin_path, int threads,
                         bool use_vulkan)
    : threads_(threads) {
    if (threads <= 0) throw std::invalid_argument("ncnn: thread count must be positive");
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
    const std::filesystem::path param_fs = std::filesystem::u8path(param_path);
    const std::filesystem::path metadata_fs =
        (param_fs.parent_path().empty() ? std::filesystem::path(".") : param_fs.parent_path()) /
        "metadata.yaml";
    std::vector<std::string> nm; int mi = 0;
    std::string metadata_input, metadata_output, metadata_proto;
    // A per-model sidecar avoids collisions when a release directory contains
    // more than one NCNN graph; retain the historical shared metadata.yaml as
    // a fallback for existing exports.
    std::filesystem::path per_model_metadata = param_fs;
    per_model_metadata.replace_extension(".metadata.yaml");
    const std::vector<std::filesystem::path> metadata_paths = {
        per_model_metadata, metadata_fs
    };
    bool has_metadata = false;
    for (const auto& metadata_path : metadata_paths) {
        if (meta::read_ncnn_yaml(metadata_path.u8string(), nm, mi,
                                 &metadata_input, &metadata_output, &metadata_proto)) {
            has_metadata = true;
            break;
        }
    }
    if (has_metadata) {
        if (!metadata_input.empty() && metadata_input == metadata_output) {
            throw std::runtime_error("ncnn metadata input_blob and output_blob must differ");
        }
        if (!metadata_proto.empty() &&
            (metadata_proto == metadata_input || metadata_proto == metadata_output)) {
            throw std::runtime_error("ncnn metadata proto_blob must be distinct from input/output blobs");
        }
        meta_names = nm; meta_imgsz = mi;
        if (!metadata_input.empty()) in_blob_ = metadata_input;
        if (!metadata_output.empty()) out_blob_ = metadata_output;
        if (!metadata_proto.empty()) {
            out_proto_ = metadata_proto;
            proto_required_ = true;
        }
    }

    const GraphBlobs graph = inspect_param_graph(param_path);
    if (graph.parsed) {
        // Explicit sidecar names are part of the model ABI.  Reject a stale
        // or hand-edited sidecar early instead of running the wrong tensor.
        if (!metadata_input.empty() && !contains(graph.all, metadata_input)) {
            throw std::runtime_error("ncnn metadata input_blob '" + metadata_input +
                                     "' is not present in " + param_path);
        }
        if (!metadata_input.empty() && !graph.inputs.empty() &&
            !contains(graph.inputs, metadata_input)) {
            throw std::runtime_error("ncnn metadata input_blob '" + metadata_input +
                                     "' is not an Input blob in " + param_path);
        }
        if (!metadata_output.empty() && !contains(graph.all, metadata_output)) {
            throw std::runtime_error("ncnn metadata output_blob '" + metadata_output +
                                     "' is not present in " + param_path);
        }
        if (!metadata_output.empty() && !graph.tops.empty() &&
            !contains(graph.tops, metadata_output)) {
            throw std::runtime_error("ncnn metadata output_blob '" + metadata_output +
                                     "' is not a terminal blob in " + param_path);
        }
        if (!metadata_proto.empty() && !contains(graph.all, metadata_proto)) {
            throw std::runtime_error("ncnn metadata proto_blob '" + metadata_proto +
                                     "' is not present in " + param_path);
        }
        if (!metadata_proto.empty() && !graph.tops.empty() &&
            !contains(graph.tops, metadata_proto)) {
            throw std::runtime_error("ncnn metadata proto_blob '" + metadata_proto +
                                     "' is not a terminal blob in " + param_path);
        }

        // When no sidecar is available, resolve the graph's actual endpoints.
        // A unique endpoint is safe to infer; multiple endpoints require the
        // sidecar because their tensor roles cannot be determined from names.
        if (metadata_input.empty()) {
            if (contains(graph.inputs, "in0")) {
                in_blob_ = "in0";
            } else if (graph.inputs.size() == 1) {
                in_blob_ = graph.inputs.front();
            } else if (graph.inputs.empty() && contains(graph.all, "in0")) {
                in_blob_ = "in0";
            } else if (graph.inputs.size() > 1) {
                throw std::runtime_error("ncnn graph has multiple input blobs; provide metadata.yaml input_blob");
            }
        }
        if (metadata_output.empty()) {
            if (contains(graph.tops, "out0")) {
                out_blob_ = "out0";
            } else if (graph.tops.size() == 1) {
                out_blob_ = graph.tops.front();
            } else if (graph.tops.empty() && contains(graph.all, "out0")) {
                out_blob_ = "out0";
            } else if (graph.tops.size() > 1) {
                throw std::runtime_error("ncnn graph has multiple terminal blobs; provide metadata.yaml output_blob");
            }
        }
        if (metadata_proto.empty()) {
            // The conventional out1 is an optional segmentation prototype.
            // Do not probe arbitrary terminal tensors: a detection graph with
            // several auxiliary outputs must declare their roles explicitly.
            if (contains(graph.tops, "out1") && out_blob_ != "out1") out_proto_ = "out1";
            else if (graph.tops.size() <= 1) out_proto_.clear();
            else out_proto_.clear();
        }
    } else if (!has_metadata) {
        ep_note = "NCNN param graph metadata unavailable; using legacy in0/out0 blob names";
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
    // A detection graph has no proto output.  The conventional out1 fallback
    // is optional, but a proto explicitly declared by metadata is part of the
    // segmentation ABI and must be present; otherwise silently dropping masks
    // would make a cross-backend comparison invalid.
    if (!out_proto_.empty()) {
        const int proto_status = ex.extract(out_proto_.c_str(), pm);
        if (proto_required_ && (proto_status != 0 || pm.empty())) {
            throw std::runtime_error("ncnn: required prototype blob '" + out_proto_ +
                                     "' could not be extracted");
        }
    }
    infer_ms = ms_since(t1);

    // ---- reshape to channel-major [feat_dim x num_anchors] then decode ----
    // Prefer the orientation whose feature axis agrees with the model class
    // count (and, for segmentation, the prototype channel count).  The old
    // ``smaller axis = features`` heuristic is only a final tie-breaker; it
    // mis-decoded valid low-anchor or transposed exports.
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
    const int expected_features = 4 + cfg.num_classes();
    int expected_masks = 0;
    if (!pm.empty()) {
        if (pm.dims != 3 || pm.c <= 0 || pm.h <= 0 || pm.w <= 0 ||
            pm.elemsize != sizeof(float))
            throw std::runtime_error("ncnn: prototype blob must be a non-empty float32 3-D tensor");
        expected_masks = pm.c;
    }
    const auto feature_count_matches = [&](int value) {
        if (value == expected_features || value == expected_features + 1) return true;
        return expected_masks > 0 &&
               (value == expected_features + expected_masks ||
                value == expected_features + 1 + expected_masks);
    };
    const bool rows_are_features = feature_count_matches(out.h);
    const bool cols_are_features = feature_count_matches(out.w);
    bool feature_rows;
    if (rows_are_features != cols_are_features) {
        feature_rows = rows_are_features;
    } else if (rows_are_features) {
        // Both axes can only match for a deliberately tiny synthetic graph;
        // retain deterministic compatibility with legacy exports.
        feature_rows = out.h <= out.w;
    } else {
        throw std::runtime_error(
            "ncnn: detection blob shape is incompatible with class count " +
            std::to_string(cfg.num_classes()) + " (expected feature axis " +
            std::to_string(expected_features) + ")");
    }
    std::vector<float> buf;
    if (feature_rows) {                        // rows = features (channel-major)
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
    bool has_objectness = false;
    if (!pm.empty()) {
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
