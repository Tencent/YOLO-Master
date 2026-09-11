// yolomaster_edge - universal, adaptive YOLO-Master edge runner.
// Runtime model loading (no baked-in weights), backend/classes/imgsz auto-detected
// from the model, versatile --source (image / dir / list / video / dataset.yaml).
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "yolomaster.hpp"
#include "slicing.hpp"
#include "annotate_export.hpp"
#include "backend_factory.hpp"
#include "CLI11.hpp"
#include "stb_image.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace yolomaster;
namespace fs = std::filesystem;

// image I/O via stb (avoids OpenCV imgcodecs -> GDAL/DB/poppler dependency closure).
// On Windows open through _wfopen so UTF-8 paths are not routed through the
// process ANSI code page.
#ifdef _WIN32
static std::wstring utf8_to_wide(const std::string& value) {
    if (value.empty()) return {};
    const int needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                           static_cast<int>(value.size()), nullptr, 0);
    if (needed <= 0) throw std::runtime_error("image path is not valid UTF-8");
    std::wstring result(static_cast<size_t>(needed), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                            static_cast<int>(value.size()), result.data(), needed) != needed)
        throw std::runtime_error("failed to convert image path to UTF-16");
    return result;
}
#endif
static cv::Mat imread_bgr(const std::string& path) {
    int w, h, n;
    unsigned char* d = nullptr;
#ifdef _WIN32
    std::wstring wide;
    try { wide = utf8_to_wide(path); }
    catch (...) { return cv::Mat(); }
    FILE* file = _wfopen(wide.c_str(), L"rb");
#else
    FILE* file = std::fopen(path.c_str(), "rb");
#endif
    if (!file) return cv::Mat();
    d = stbi_load_from_file(file, &w, &h, &n, 3);   // force 3-channel RGB
    std::fclose(file);
    if (!d) return cv::Mat();
    cv::Mat bgr;
    cv::cvtColor(cv::Mat(h, w, CV_8UC3, d), bgr, cv::COLOR_RGB2BGR);
    stbi_image_free(d);
    return bgr;
}
struct BenchmarkRow {
    std::string image;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
    int detections = 0;
};

static std::string csv_escape(const std::string& value) {
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '\"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += '\"';
    return escaped;
}

static double percentile(std::vector<double> values, double pct) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double rank = (pct / 100.0) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    if (lo == hi) return values[lo];
    const double weight = rank - static_cast<double>(lo);
    return values[lo] * (1.0 - weight) + values[hi] * weight;
}

static std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (unsigned char c : value) {
        switch (c) {
        case '\\': escaped += "\\\\"; break;
        case '"': escaped += "\\\""; break;
        case '\b': escaped += "\\b"; break;
        case '\f': escaped += "\\f"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                escaped += buf;
            } else {
                escaped += static_cast<char>(c);
            }
        }
    }
    return escaped;
}

static std::string host_os() {
#ifdef _WIN32
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

static std::string host_arch() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#else
    return "unknown";
#endif
}

static std::string compiler_id() {
#if defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#else
    return "unknown";
#endif
}

static std::string trim_copy(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::string cpu_model() {
#ifdef _WIN32
    if (const char* value = std::getenv("PROCESSOR_IDENTIFIER")) {
        if (*value) return value;
    }
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim_copy(line.substr(0, colon));
        if (key == "model name" || key == "Hardware" || key == "Processor") {
            const std::string value = trim_copy(line.substr(colon + 1));
            if (!value.empty()) return value;
        }
    }
#elif defined(__APPLE__)
    if (const char* value = std::getenv("HOSTTYPE")) {
        if (*value) return value;
    }
#endif
    return "unknown";
}

static std::string build_date() {
    return std::string(__DATE__) + " " + __TIME__;
}

static bool write_benchmark_json(const std::string& path, const std::vector<BenchmarkRow>& rows,
                                 int warmup, int runs, int threads, const std::string& model,
                                 const std::string& source, const std::string& backend,
                                 const std::string& execution_provider, const std::string& profile,
                                 const Config& cfg, const std::string& csv_path,
                                 long frames, long failed_frames, double wall_sec) {
    const fs::path output_path = fs::u8path(path);
    if (const fs::path parent = output_path.parent_path(); !parent.empty()) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) return false;
    }
    std::ofstream out(output_path);
    if (!out) return false;
    std::vector<double> prep, infer, post, totals;
    prep.reserve(rows.size());
    infer.reserve(rows.size());
    post.reserve(rows.size());
    totals.reserve(rows.size());
    for (const auto& row : rows) {
        prep.push_back(row.preprocess_ms);
        infer.push_back(row.inference_ms);
        post.push_back(row.postprocess_ms);
        totals.push_back(row.total_ms);
    }
    const double mean = totals.empty()
        ? 0.0
        : std::accumulate(totals.begin(), totals.end(), 0.0) / totals.size();
    const auto stats_json = [](const std::vector<double>& values) {
        const double avg = values.empty()
            ? 0.0
            : std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        std::ostringstream s;
        s << std::setprecision(10)
          << "{\"count\":" << values.size()
          << ",\"mean_ms\":" << avg
          << ",\"p50_ms\":" << percentile(values, 50.0)
          << ",\"p95_ms\":" << percentile(values, 95.0)
          << ",\"p99_ms\":" << percentile(values, 99.0)
          << ",\"fps\":" << (avg > 0.0 ? 1000.0 / avg : 0.0) << "}";
        return s.str();
    };
    out << std::setprecision(10);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"status\": \"" << (failed_frames == 0 ? "completed" : "partial") << "\",\n"
        << "  \"model\": \"" << json_escape(model) << "\",\n"
        << "  \"source\": \"" << json_escape(source) << "\",\n"
        << "  \"protocol\": {\n"
        << "    \"backend\": \"" << json_escape(backend) << "\",\n"
        << "    \"execution_provider\": \"" << json_escape(execution_provider) << "\",\n"
        << "    \"profile\": \"" << json_escape(profile) << "\",\n"
        << "    \"imgsz\": " << cfg.imgsz << ",\n"
        << "    \"conf\": " << cfg.conf_thresh << ",\n"
        << "    \"iou\": " << cfg.iou_thresh << ",\n"
        << "    \"small_conf\": " << cfg.small_conf_thresh << ",\n"
        << "    \"small_area\": " << cfg.small_area << ",\n"
        << "    \"max_det\": " << cfg.max_det << ",\n"
        << "    \"multi_label\": " << (cfg.multi_label ? "true" : "false") << ",\n"
        << "    \"letterbox\": " << (cfg.stretch ? "false" : "true") << ",\n"
        << "    \"nms_mode\": \""
        << (cfg.nms_mode == NmsMode::ClusterWeighted ? "cluster_weighted" : "standard") << "\",\n"
        << "    \"cw_sigma\": " << cfg.cw_sigma << ",\n"
        << "    \"class_count\": " << cfg.num_classes() << ",\n"
        << "    \"warmup\": " << warmup << ",\n"
        << "    \"runs\": " << runs << ",\n"
        << "    \"threads\": " << threads << "\n"
        << "  },\n"
        << "  \"host\": {\n"
        << "    \"os\": \"" << host_os() << "\",\n"
        << "    \"architecture\": \"" << host_arch() << "\",\n"
        << "    \"compiler\": \"" << json_escape(compiler_id()) << "\",\n"
        << "    \"cpu\": \"" << json_escape(cpu_model()) << "\",\n"
        << "    \"logical_cpus\": " << std::thread::hardware_concurrency() << ",\n"
        << "    \"build_date\": \"" << build_date() << "\"\n"
        << "  },\n"
        << "  \"summary\": {\n"
        << "    \"frames\": " << frames << ",\n"
        << "    \"timed_images\": " << rows.size() << ",\n"
        << "    \"failed_inputs\": " << failed_frames << ",\n"
        << "    \"mean_ms\": " << mean << ",\n"
        << "    \"p50_ms\": " << percentile(totals, 50.0) << ",\n"
        << "    \"p95_ms\": " << percentile(totals, 95.0) << ",\n"
        << "    \"p99_ms\": " << percentile(totals, 99.0) << ",\n"
        << "    \"fps\": " << (mean > 0.0 ? 1000.0 / mean : 0.0) << ",\n"
        << "    \"wall_seconds\": " << wall_sec << ",\n"
        << "    \"timing_ms\": {\n"
        << "      \"preprocess\": " << stats_json(prep) << ",\n"
        << "      \"inference\": " << stats_json(infer) << ",\n"
        << "      \"postprocess\": " << stats_json(post) << ",\n"
        << "      \"total\": " << stats_json(totals) << "\n"
        << "    }\n"
        << "  },\n"
        << "  \"timing_csv\": ";
    if (csv_path.empty()) out << "null\n";
    else out << "\"" << json_escape(csv_path) << "\"\n";
    out << "}\n";
    return out.good();
}

static bool write_benchmark_csv(const std::string& path, const std::vector<BenchmarkRow>& rows,
                                int warmup, int runs, int threads, int imgsz,
                                float conf, float iou) {
    std::ofstream out(fs::u8path(path));
    if (!out) return false;
    out << "image,preprocess_ms,inference_ms,postprocess_ms,total_ms,detections,mean_ms,p50_ms,p95_ms,p99_ms,fps\n";
    std::vector<double> totals;
    totals.reserve(rows.size());
    for (const auto& row : rows) {
        out << csv_escape(row.image) << ',' << row.preprocess_ms << ',' << row.inference_ms << ','
            << row.postprocess_ms << ',' << row.total_ms << ',' << row.detections << ",,,,,\n";
        totals.push_back(row.total_ms);
    }
    const double sum = std::accumulate(totals.begin(), totals.end(), 0.0);
    const double mean = totals.empty() ? 0.0 : sum / static_cast<double>(totals.size());
    // Keep aggregate rows parseable by the shared CSV helper: non-numeric
    // metadata belongs in the run header on stdout, not in a timing column.
    (void)warmup;
    (void)runs;
    (void)threads;
    (void)imgsz;
    (void)conf;
    (void)iou;
    out << "#summary,,,,,," << mean << ',' << percentile(totals, 50.0) << ','
        << percentile(totals, 95.0) << ',' << percentile(totals, 99.0) << ','
        << (mean > 0.0 ? 1000.0 / mean : 0.0) << '\n';
    return out.good();
}

int main(int argc, char** argv) {
    CLI::App app{"yolomaster_edge - universal YOLO-Master edge runner (ONNX / ncnn / MNN / TensorRT)"};
    std::string model, source, backend = "auto", classes_opt = "auto", outdir = "runs_edge";
    std::string profile = "default";
    std::string device = "cpu", savetxt, csv_path, benchmark_json_path;
    int imgsz = 0, threads = 4, limit = 0, max_det = 300;
    int warmup = 0, runs = 1;
    float conf = 0.25f, iou = 0.50f;
    float small_conf = -1.0f, small_area = 32.0f * 32.0f;
    bool no_save = false, quiet = false, multilabel = false, single_label = false, stretch = false;
    std::string slicing = "off", label_format = "yolo", sampling = "1s", export_labels;
    int tile_size = 0;
    bool slicing_masks = false, cw_nms = false;
    float sigma = 0.1f;

    app.add_option("-m,--model", model,
                   "model: .onnx, .mnn, .engine/.trt, ncnn directory, or .param file")->required();
    app.add_option("-s,--source", source, "image / directory / .txt or .list image list / video / dataset.yaml")->required();
    app.add_option("-b,--backend", backend, "auto|onnx|ncnn|mnn|trt")->default_str("auto");
    app.add_option("-d,--device", device,
                   "backend-dependent: cpu, cuda, vulkan, opencl, trt, or coreml")->default_str("cpu");
    app.add_option(
        "--profile", profile,
        "post-processing profile: default|visdrone|sku110k (thresholds are overridable)")
        ->default_str("default");
    app.add_option("--classes", classes_opt, "auto|visdrone|sku110k (auto = from model metadata)")->default_str("auto");
    auto* imgsz_opt = app.add_option("--imgsz", imgsz, "inference size (0 = from model / profile)");
    auto* conf_opt = app.add_option("--conf", conf, "confidence threshold")->capture_default_str();
    auto* iou_opt = app.add_option("--iou", iou, "NMS IoU threshold")->capture_default_str();
    auto* maxdet_opt = app.add_option("--max-det", max_det, "max detections per image after NMS")
                           ->capture_default_str();
    app.add_option(
        "--small-conf", small_conf,
        "optional lower confidence for boxes below --small-area (-1 disables)")
        ->capture_default_str();
    app.add_option(
        "--small-area", small_area,
        "original-image area threshold for --small-conf (pixels^2)")
        ->capture_default_str();
    app.add_option("--threads", threads, "CPU threads")->capture_default_str();
    app.add_option("--limit", limit, "cap #inputs (0 = all)");
    app.add_option("--warmup", warmup, "untimed warm-up inferences per first input (benchmark only)")
        ->capture_default_str();
    app.add_option("--runs", runs, "timed repetitions per input (benchmark only)")->capture_default_str();
    app.add_option("--csv", csv_path, "write per-image benchmark CSV (enables timing summary)");
    app.add_option("--benchmark-json", benchmark_json_path,
                   "write benchmark protocol/host/summary JSON sidecar (enables timing summary)");
    app.add_option("--out", outdir, "output dir for annotated results")->capture_default_str();
    app.add_option("--save-txt", savetxt, "dir to write per-image predictions ('class conf x1 y1 x2 y2')");
    auto* multilabel_opt = app.add_flag(
        "--multi-label", multilabel,
        "one detection per class >= conf per anchor (matches ultralytics val mAP)");
    auto* singlelabel_opt = app.add_flag(
        "--single-label", single_label,
        "diagnostic argmax-per-anchor decoding (mutually exclusive with --multi-label)");
    app.add_flag("--stretch", stretch, "preprocess by stretching to square instead of aspect-preserving letterbox");
    app.add_flag("--no-save", no_save, "do not write annotated outputs");
    app.add_flag("--quiet", quiet, "suppress per-image logs");
    app.add_option("--slicing", slicing, "off|dense|sparse: global pass + tile passes (Sparse SAHI); images/dirs only")->default_str("off");
    app.add_option("--tile-size", tile_size, "requested tile edge in source px (0 = model imgsz); clamped per image to [imgsz, max(imgsz, shortSide/4)]");
    app.add_flag("--slicing-masks", slicing_masks, "keep the global pass's masks (+proto) in sliced runs (seg models)");
    app.add_flag("--cw-nms", cw_nms, "Cluster-Weighted NMS: refine survivor boxes by their cluster's weighted average");
    app.add_option("--sigma", sigma, "CW-NMS weight falloff (0.01-0.5)")->capture_default_str();
    app.add_option("--export-labels", export_labels, "dir to write annotation labels (WYSIWYG at the current conf/iou/nms settings)");
    app.add_option("--label-format", label_format, "yolo|coco|voc")->default_str("yolo");
    app.add_option("--sampling", sampling, "video label export: all|1s|N (every Nth frame)")->default_str("1s");
    CLI11_PARSE(app, argc, argv);

    if (multilabel_opt->count() > 0 && singlelabel_opt->count() > 0) {
        std::cerr << "--multi-label and --single-label are mutually exclusive\n";
        return 2;
    }
    if (singlelabel_opt->count() > 0) multilabel = false;

    profile = lower_ascii(profile);
    classes_opt = lower_ascii(classes_opt);
    if (classes_opt == "sku") classes_opt = "sku110k";
    if (classes_opt != "auto" && classes_opt != "visdrone" && classes_opt != "sku110k") {
        std::cerr << "unknown --classes: " << classes_opt
                  << " (expected auto, visdrone, or sku110k)\n";
        return 2;
    }

    // The generic runner keeps conservative defaults, while the explicit
    // vertical profiles reproduce the Issue #51 evaluation recipe.  Only
    // values omitted by the caller are filled in, so deployment-specific
    // thresholds remain possible and are visible in the run header.
    // Canonical VisDrone defaults: imgsz = 640, conf = 0.001f,
    // iou = 0.70f, multi_label = true.
    if (profile == "visdrone") {
        if (imgsz_opt->count() == 0) imgsz = 640;
        if (conf_opt->count() == 0) conf = 0.001f;
        if (iou_opt->count() == 0) iou = 0.70f;
        if (maxdet_opt->count() == 0) max_det = 300;
        if (multilabel_opt->count() == 0 && singlelabel_opt->count() == 0) multilabel = true;
    } else if (profile == "sku110k") {
        if (imgsz_opt->count() == 0) imgsz = 1280;
        if (conf_opt->count() == 0) conf = 0.25f;
        if (iou_opt->count() == 0) iou = 0.60f;
        if (maxdet_opt->count() == 0) max_det = 300;
        // Keep the profile metadata identical to the Python evaluator.  SKU-110K
        // has one class, so this does not change the decoded boxes, but it makes
        // the protocol explicit and prevents a cross-backend manifest mismatch.
        if (multilabel_opt->count() == 0 && singlelabel_opt->count() == 0) multilabel = true;
    } else if (profile != "default") {
        std::cerr << "unknown --profile: " << profile << " (expected default, visdrone, or sku110k)\n";
        return 2;
    }
    if (!std::isfinite(conf) || conf < 0.f || conf > 1.f ||
        !std::isfinite(iou) || iou < 0.f || iou > 1.f ||
        !std::isfinite(small_conf) || small_conf < -1.f || small_conf > 1.f ||
        !std::isfinite(small_area) || small_area < 0.f ||
        max_det <= 0 || threads <= 0 || warmup < 0 || runs <= 0 || limit < 0) {
        std::cerr << "conf/iou must be in [0,1], small-conf in [-1,1], small-area "
                     "non-negative, max-det/threads/runs positive, and warmup/limit "
                     "non-negative\n";
        return 2;
    }

    SliceMode slice_mode = SliceMode::Off;
    if (slicing == "dense") slice_mode = SliceMode::Dense;
    else if (slicing == "sparse") slice_mode = SliceMode::Sparse;
    else if (slicing != "off") { std::cerr << "unknown --slicing mode: " << slicing << "\n"; return 2; }
    annot::Format lfmt = annot::Format::YoloTXT;
    if (label_format == "coco") lfmt = annot::Format::CocoJSON;
    else if (label_format == "voc") lfmt = annot::Format::PascalVOC;
    else if (label_format != "yolo") { std::cerr << "unknown --label-format: " << label_format << "\n"; return 2; }

    // ---- construct backend ----
    std::unique_ptr<Backend> be;
    std::string resolved_backend, backend_error;
    be = make_backend(model, backend, threads, device, resolved_backend, backend_error);
    if (!be) {
        std::cerr << backend_error << "\n";
        return 3;
    }
    backend = resolved_backend;

    // ---- resolve config: --flag > model metadata > default ----
    Config cfg;
    cfg.conf_thresh = conf;
    cfg.iou_thresh = iou;
    cfg.small_conf_thresh = small_conf;
    cfg.small_area = small_area;
    cfg.max_det = max_det;
    cfg.multi_label = multilabel;
    cfg.stretch = stretch;
    cfg.nms_mode = cw_nms ? NmsMode::ClusterWeighted : NmsMode::Standard;
    cfg.cw_sigma = std::min(0.5f, std::max(0.01f, sigma));
    int want = imgsz > 0 ? imgsz : (be->meta_imgsz > 0 ? be->meta_imgsz : 640);
    if (be->fixed_imgsz > 0 && want != be->fixed_imgsz) {
        // A caller-supplied size, or a canonical evaluation profile, is part
        // of the protocol and must not be silently rewritten to fit a model.
        // Generic runs with no explicit size may still adopt a model's static
        // input as a convenience.
        const bool canonical_profile = profile == "visdrone" || profile == "sku110k";
        if (imgsz_opt->count() > 0 || canonical_profile) {
            std::cerr << "model requires fixed imgsz=" << be->fixed_imgsz
                      << "; requested imgsz=" << want << " is incompatible\n";
            return 2;
        }
        std::cout << "[model] using fixed imgsz=" << be->fixed_imgsz
                  << " (model constraint; no explicit --imgsz supplied)\n";
        want = be->fixed_imgsz;
    }
    cfg.imgsz = want;
    std::string classes_src;
    const std::vector<std::string>* profile_names = nullptr;
    if (profile == "visdrone") profile_names = &visdrone_classes();
    else if (profile == "sku110k") profile_names = &sku110k_classes();

    if (profile_names) {
        // An explicit profile defines both thresholds and the class ABI.  Do
        // not let ``--classes auto`` silently select (for example) COCO-80
        // metadata from a VisDrone run; that would decode the output with the
        // wrong feature count and invalidate the metric.
        if (!be->meta_names.empty() && be->meta_names.size() != profile_names->size()) {
            std::cerr << "model metadata declares " << be->meta_names.size()
                      << " classes, but --profile " << profile << " requires "
                      << profile_names->size() << "; use a matching model or --profile default\n";
            return 2;
        }
        if (classes_opt != "auto" && classes_opt != profile) {
            std::cerr << "--profile " << profile << " conflicts with --classes " << classes_opt << "\n";
            return 2;
        }
        cfg.class_names = *profile_names;
        classes_src = "profile:" + profile;
    } else if (classes_opt == "visdrone") {
        cfg.class_names = visdrone_classes(); classes_src = "flag:visdrone";
    } else if (classes_opt == "sku110k") {
        cfg.class_names = sku110k_classes(); classes_src = "flag:sku110k";
    } else if (!be->meta_names.empty()) {
        cfg.class_names = be->meta_names; classes_src = "model-metadata";
    } else {
        cfg.class_names = visdrone_classes(); classes_src = "fallback:visdrone";
    }

    std::cout << "[model] " << model << "  backend=" << backend << "  ep=" << be->active_ep
              << "  profile=" << profile
              << "  imgsz=" << cfg.imgsz << "  nc=" << cfg.num_classes() << " (" << classes_src << ")"
              << "  conf=" << cfg.conf_thresh << "  iou=" << cfg.iou_thresh
              << "  small_conf=" << cfg.small_conf_thresh
              << "  small_area=" << cfg.small_area
              << "  max_det=" << cfg.max_det << "  multi_label=" << (cfg.multi_label ? "true" : "false")
              << "  threads=" << threads;
    if (!csv_path.empty() || !benchmark_json_path.empty()) {
        std::cout << "  warmup=" << warmup << "  runs=" << runs;
        if (!csv_path.empty()) std::cout << "  csv=" << csv_path;
        if (!benchmark_json_path.empty()) std::cout << "  benchmark_json=" << benchmark_json_path;
    }
    std::cout << "\n";

    if (!no_save) { std::error_code ec; fs::create_directories(outdir, ec); }
    if (!savetxt.empty()) { std::error_code ec; fs::create_directories(savetxt, ec); }

    // ---- run over the source ----
    const SourceKind kind = classify_source(source);
    if (slice_mode != SliceMode::Off && kind == SourceKind::Video) {
        std::cerr << "[warn] slicing applies to images and folders only - video runs single-pass\n";
        slice_mode = SliceMode::Off;
    }
    SliceConfig sconf;
    sconf.mode = slice_mode;
    sconf.tile_size = tile_size;
    sconf.keep_global_masks = slicing_masks;
    TileStats tstats;

    // Label export: the sink is created lazily after the first forward, when the run's
    // dialect is known (WYSIWYG: seg polygons only when the backend actually carries mask
    // data for this run - sliced without --slicing-masks degrades to boxes).
    std::unique_ptr<AnnotationSink> sink;
    std::string labels_dir = export_labels, frames_dir;
    if (!export_labels.empty()) {
        std::error_code ec;
        if (kind == SourceKind::Video) {
            frames_dir = (fs::path(export_labels) / "frames").string();
            fs::create_directories(frames_dir, ec);
            if (lfmt != annot::Format::CocoJSON) {
                labels_dir = (fs::path(export_labels) / "labels").string();
                fs::create_directories(labels_dir, ec);
            }
        } else fs::create_directories(export_labels, ec);
    }
    auto ensure_sink = [&]() -> AnnotationSink& {
        if (!sink) {
            const bool dialect = be->is_seg();
            sink = std::make_unique<AnnotationSink>(
                lfmt, labels_dir, (fs::path(export_labels) / "annotations.coco.json").string(),
                cfg.class_names, dialect);
        }
        return *sink;
    };

    auto t_start = std::chrono::high_resolution_clock::now();
    long frames = 0, failed_frames = 0, total_dets = 0;
    double sum_pre = 0, sum_inf = 0, sum_post = 0;
    std::vector<BenchmarkRow> benchmark_rows;
    const bool benchmark_requested = !csv_path.empty() || !benchmark_json_path.empty();
    const int timed_runs = benchmark_requested ? runs : 1;
    bool warmed_up = false;
    std::vector<std::string> failures;

    // Video sources: annotated output becomes ONE mp4 (per-frame jpgs would overwrite each
    // other - "11.mp4#930" stems to "11"), and --save-txt gets frame-indexed names.
    const bool video_mode = (kind == SourceKind::Video);
    std::set<std::string> out_stems, txt_stems;    // collision guards: 1.jpg + 1.png in one dir
    double src_fps = 30.0;
#ifdef HAVE_VIDEOIO
    cv::VideoWriter vwriter;                       // lazily opened on the first saved frame
    std::string vwriter_path;
#endif

    // coco_file/coco_id: the COCO doc's file_name (may carry "frames/") and explicit image
    // id (0 = sequence). export=false skips label emission (non-sampled video frames).
    auto run_one = [&](const cv::Mat& img, const std::string& tag,
                       bool do_export = true, const std::string& coco_file = "", int coco_id = 0) -> bool {
        auto record_failure = [&](const std::string& reason) {
            ++failed_frames;
            failures.push_back(tag + ": " + reason);
        };
        if (img.empty()) {
            std::cerr << "  [skip] unreadable: " << tag << "\n";
            record_failure("unreadable image");
            return false;
        }
        if (!warmed_up && benchmark_requested && warmup > 0) {
            try {
                for (int i = 0; i < warmup; ++i) {
                    if (slice_mode != SliceMode::Off) {
                        (void)sliced_candidates(*be, img, cfg, sconf);
                    } else {
                        (void)be->infer(img, cfg);
                    }
                }
                warmed_up = true;
            } catch (const std::exception& e) {
                std::cerr << "  [skip] warm-up failed on " << tag << ": " << e.what() << "\n";
                record_failure(std::string("warm-up error: ") + e.what());
                return false;
            }
        }

        std::vector<Detection> dets;
        std::string slice_note;
        double image_pre = 0.0, image_inf = 0.0, image_post = 0.0;
        try {
            for (int repeat = 0; repeat < timed_runs; ++repeat) {
                if (slice_mode != SliceMode::Off) {
                    const SliceOutput so = sliced_candidates(*be, img, cfg, sconf);
                    dets = nms_and_cap(be->candidates, cfg, img.cols, img.rows);
                    tstats.add(so.tiles_run, so.tiles_total, so.tile_size_used,
                               so.used_fallback, so.capped);
                    slice_note = "  tiles=" + std::to_string(so.tiles_run) + "/"
                               + std::to_string(so.tiles_total) + " @" + std::to_string(so.tile_size_used) + "px"
                               + (so.used_fallback ? " [fallback]" : "") + (so.capped ? " [capped]" : "");
                    // sliced_candidates aggregates all model forwards, including
                    // preprocessing and postprocessing for each tile.  Read the
                    // stage sums from SliceOutput so the CSV row describes the
                    // complete image rather than the final tile only.
                    image_pre += so.pre_ms;
                    image_inf += so.infer_ms;
                    image_post += so.post_ms;
                } else {
                    dets = be->infer(img, cfg);
                    image_pre += be->pre_ms;
                    image_inf += be->infer_ms;
                    image_post += be->post_ms;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "  [skip] inference error on " << tag << ": " << e.what() << "\n";
            record_failure(std::string("inference error: ") + e.what());
            return false;
        }
        image_pre /= timed_runs;
        image_inf /= timed_runs;
        image_post /= timed_runs;
        if (benchmark_requested) {
            benchmark_rows.push_back({tag, image_pre, image_inf, image_post,
                                      image_pre + image_inf + image_post,
                                      static_cast<int>(dets.size())});
        }
        if (!export_labels.empty() && do_export) {
            try {
                AnnotationSink& s = ensure_sink();
                annot::Image aimg;
                aimg.name = fs::path(tag).stem().string();
                if (coco_id > 0) aimg.name = fs::path(coco_file).stem().string();
                aimg.width = img.cols; aimg.height = img.rows;
                aimg.instances = annotation_instances(dets, be->is_seg(), be->proto, be->proto_c,
                                                      be->proto_h, be->proto_w, be->cand_lb, cfg.imgsz);
                s.add(aimg, coco_file.empty() ? fs::path(tag).filename().string() : coco_file, coco_id);
            } catch (const std::exception& e) {
                std::cerr << "  [skip] label export error on " << tag << ": " << e.what() << "\n";
                record_failure(std::string("label export error: ") + e.what());
                return false;
            }
        }
        frames++; total_dets += static_cast<long>(dets.size());
        sum_pre += image_pre; sum_inf += image_inf; sum_post += image_post;
        if (!quiet)
            std::cout << "  " << tag << "  dets=" << dets.size()
                      << "  infer=" << image_inf << "ms" << slice_note << "\n";
        if (!no_save) {
            cv::Mat vis = img.clone();
            if (be->is_seg()) {                       // alpha-composite segmentation masks under the boxes
                cv::Mat ov = seg_overlay(dets, be->proto, be->proto_c, be->proto_h, be->proto_w,
                                         be->cand_lb, cfg.imgsz, img.cols, img.rows);
                for (int y = 0; y < vis.rows; ++y) {
                    const uint8_t* o = ov.ptr<uint8_t>(y);
                    uint8_t* v = vis.ptr<uint8_t>(y);
                    for (int x = 0; x < vis.cols; ++x) {
                        const float a = o[x * 4 + 3] / 255.f;
                        if (a <= 0) continue;
                        v[x * 3 + 0] = (uint8_t)(v[x * 3 + 0] * (1 - a) + o[x * 4 + 2] * a);  // B<-B
                        v[x * 3 + 1] = (uint8_t)(v[x * 3 + 1] * (1 - a) + o[x * 4 + 1] * a);  // G<-G
                        v[x * 3 + 2] = (uint8_t)(v[x * 3 + 2] * (1 - a) + o[x * 4 + 0] * a);  // R<-R
                    }
                }
            }
            draw(vis, dets, cfg);
#ifdef HAVE_VIDEOIO
            if (video_mode) {                         // one annotated mp4, not overwriting jpgs
                if (!vwriter.isOpened()) {
                    vwriter_path = (fs::path(outdir) /
                        (fs::path(source).stem().string() + "_annotated.mp4")).string();
                    vwriter.open(vwriter_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                 src_fps, vis.size());
                    if (!vwriter.isOpened())
                        std::cerr << "  [warn] cannot open " << vwriter_path << " for writing\n";
                }
                if (vwriter.isOpened()) vwriter.write(vis);
                else {
                    record_failure("annotated video writer is not open");
                    return false;
                }
            } else {
#endif
                const std::string output_path = (fs::path(outdir) /
                    (unique_stem(out_stems, fs::path(tag).stem().string()) + ".jpg")).string();
                if (!write_jpg(output_path, vis)) {
                    std::cerr << "  [skip] failed to write annotated image: " << output_path << "\n";
                    record_failure("annotated image write failed");
                    return false;
                }
#ifdef HAVE_VIDEOIO
            }
#endif
        }
        if (!savetxt.empty()) {                       // 'class conf x1 y1 x2 y2' (pixel xyxy)
            std::string tstem = fs::path(tag).stem().string();
            if (video_mode && coco_id > 0) {          // frame-unique name (stem collides at "11")
                char b[64];
                std::snprintf(b, sizeof(b), "%s_%06d", fs::path(source).stem().string().c_str(),
                              coco_id - 1);
                tstem = b;
            }
            const std::string txt_path = (fs::path(savetxt) /
                (unique_stem(txt_stems, tstem) + ".txt")).string();
            std::ofstream f(fs::u8path(txt_path));
            if (!f) {
                std::cerr << "  [skip] failed to write predictions: " << txt_path << "\n";
                record_failure("prediction write failed");
                return false;
            }
            for (const auto& d : dets)
                f << d.class_id << ' ' << d.conf << ' ' << d.box.x << ' ' << d.box.y << ' '
                  << (d.box.x + d.box.width) << ' ' << (d.box.y + d.box.height) << '\n';
            if (!f.good()) {
                std::cerr << "  [skip] failed while writing predictions: " << txt_path << "\n";
                record_failure("prediction write failed");
                return false;
            }
        }
        return true;
    };

    if (kind == SourceKind::Video) {
#ifdef HAVE_VIDEOIO
        cv::VideoCapture cap(source);
        if (!cap.isOpened()) { std::cerr << "cannot open video: " << source << "\n"; return 4; }
        const double fps_probe = cap.get(cv::CAP_PROP_FPS);
        src_fps = (fps_probe > 1.0 && fps_probe < 1000.0) ? fps_probe : 30.0;
        // label-export sampling stride: all=1, 1s=round(fps), N=every Nth
        int stride = 1;
        if (!export_labels.empty()) {
            if (sampling == "1s") {
                const double fps = cap.get(cv::CAP_PROP_FPS);
                stride = std::max(1, static_cast<int>(std::lround(fps > 0 ? fps : 30)));
            } else if (sampling != "all") {
                try { stride = std::max(1, std::stoi(sampling)); }
                catch (...) { std::cerr << "unknown --sampling: " << sampling << "\n"; return 2; }
            }
        }
        const std::string vstem = fs::path(source).stem().string();
        cv::Mat frame; long idx = 0;
        while (cap.read(frame)) {
            if (limit > 0 && idx >= limit) break;
            const bool sampled = !export_labels.empty() && idx % stride == 0;
            std::string coco_file;
            if (sampled) {
                char fn[64];
                std::snprintf(fn, sizeof(fn), "%s_%06ld.jpg", vstem.c_str(), idx);
                const std::string frame_path = (fs::path(frames_dir) / fn).string();
                if (!write_jpg(frame_path, frame)) {
                    std::cerr << "  [skip] failed to write sampled frame: " << frame_path << "\n";
                    ++failed_frames;
                    failures.push_back(source + "#" + std::to_string(idx) + ": sampled frame write failed");
                    ++idx;
                    continue;
                }
                coco_file = std::string("frames/") + fn;
            }
            run_one(frame, source + "#" + std::to_string(idx), sampled, coco_file,
                    static_cast<int>(idx) + 1);
            ++idx;
        }
#else
        std::cerr << "video source not supported in this portable build; use image/dir/list/dataset\n";
        return 4;
#endif
    } else {
        std::vector<std::string> imgs;
        try {
            imgs = gather_images(source, limit);
        } catch (const std::exception& e) {
            std::cerr << "cannot resolve source: " << e.what() << "\n";
            return 4;
        }
        if (imgs.empty()) { std::cerr << "no inputs resolved from source: " << source << "\n"; return 4; }
        std::string stem_error;
        if (!validate_unique_stems(imgs, stem_error)) {
            std::cerr << stem_error << "\n";
            return 4;
        }
        for (const auto& p : imgs) run_one(imread_bgr(p), p);
    }

    if (frames == 0) { std::cerr << "no frames processed\n"; return 5; }
    const double wall = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
    const double avg = (sum_pre + sum_inf + sum_post) / frames;
    std::cout << "\n[summary] frames=" << frames << "  total_dets=" << total_dets
              << "  avg/frame: pre=" << sum_pre / frames << " infer=" << sum_inf / frames
              << " post=" << sum_post / frames << " total=" << avg << "ms"
              << "  model-FPS=" << 1000.0 / avg << "  wall=" << wall << "s";
    if (cfg.nms_mode == NmsMode::ClusterWeighted)
        std::cout << "  nms=cw(sigma=" << cfg.cw_sigma << ")";
    std::cout << "\n";
    if (slice_mode != SliceMode::Off)
        std::cout << "[slicing] mode=" << slicing << "  tiles=" << tstats.tiles_run << "/"
                  << tstats.tiles_total << "  size=" << tstats.tile_size_label()
                  << "  fallbacks=" << tstats.fallbacks << "  capped=" << tstats.capped << "\n";
    if (sink) {
        const AnnotationSink::Result r = sink->finish();
        if (!r.error.empty()) {
            std::cerr << "[labels] export failed: " << r.error << "\n";
            ++failed_frames;
            failures.push_back("annotation sink: " + r.error);
        }
        else std::cout << "[labels] " << annot::label(lfmt) << "  images=" << r.images
                       << "  instances=" << r.instances << " -> " << export_labels << "/\n";
    }
    if (!no_save) {
#ifdef HAVE_VIDEOIO
        if (video_mode && vwriter.isOpened()) {
            vwriter.release();
            std::cout << "[saved] annotated video -> " << vwriter_path << "\n";
        } else
#endif
        std::cout << "[saved] annotated -> " << outdir << "/\n";
    }
    if (!csv_path.empty()) {
        if (!write_benchmark_csv(csv_path, benchmark_rows, warmup, runs, threads, cfg.imgsz,
                                 cfg.conf_thresh, cfg.iou_thresh)) {
            std::cerr << "[benchmark] failed to write CSV: " << csv_path << "\n";
            ++failed_frames;
            failures.push_back("benchmark CSV write failed: " + csv_path);
        }
    }
    if (!benchmark_json_path.empty()) {
        if (!write_benchmark_json(benchmark_json_path, benchmark_rows, warmup, runs, threads,
                                  model, source, backend, be->active_ep, profile, cfg,
                                  csv_path, frames, failed_frames, wall)) {
            std::cerr << "[benchmark] failed to write JSON sidecar: " << benchmark_json_path << "\n";
            ++failed_frames;
            failures.push_back("benchmark JSON write failed: " + benchmark_json_path);
        }
    }
    if (benchmark_requested && !quiet) {
        std::vector<double> totals;
        totals.reserve(benchmark_rows.size());
        for (const auto& row : benchmark_rows) totals.push_back(row.total_ms);
        const double total_mean = totals.empty()
            ? 0.0
            : std::accumulate(totals.begin(), totals.end(), 0.0) / totals.size();
        std::cout << "[benchmark] images=" << benchmark_rows.size()
                  << "  mean=" << total_mean << "ms"
                  << "  p50=" << percentile(totals, 50.0)
                  << "  p95=" << percentile(totals, 95.0)
                  << "  p99=" << percentile(totals, 99.0)
                  << "  fps=" << (total_mean > 0.0 ? 1000.0 / total_mean : 0.0);
        if (!csv_path.empty()) std::cout << "  csv=" << csv_path;
        if (!benchmark_json_path.empty()) std::cout << "  json=" << benchmark_json_path;
        std::cout << "\n";
    }
    if (failed_frames > 0) {
        std::cerr << "[summary] failed=" << failed_frames << " of "
                  << (frames + failed_frames) << " input(s)\n";
        const size_t shown = std::min<size_t>(failures.size(), 8);
        for (size_t i = 0; i < shown; ++i) std::cerr << "  [failed] " << failures[i] << "\n";
        if (failures.size() > shown)
            std::cerr << "  [failed] ... " << (failures.size() - shown) << " more\n";
        return 6;
    }
    return 0;
}
