// yolomaster_edge - universal, adaptive YOLO-Master edge runner.
// Runtime model loading (no baked-in weights), backend/classes/imgsz auto-detected
// from the model, versatile --source (image / dir / video / dataset.yaml).
#include "yolomaster.hpp"
#ifdef USE_ORT
#include "ort_backend.hpp"
#endif
#ifdef USE_NCNN
#include "ncnn_backend.hpp"
#endif
#include "CLI11.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

using namespace yolomaster;
namespace fs = std::filesystem;

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// image I/O via stb (avoids OpenCV imgcodecs -> GDAL/DB/poppler dependency closure)
static cv::Mat imread_bgr(const std::string& path) {
    int w, h, n;
    unsigned char* d = stbi_load(path.c_str(), &w, &h, &n, 3);   // force 3-channel RGB
    if (!d) return cv::Mat();
    cv::Mat bgr;
    cv::cvtColor(cv::Mat(h, w, CV_8UC3, d), bgr, cv::COLOR_RGB2BGR);
    stbi_image_free(d);
    return bgr;
}
static bool imwrite_jpg(const std::string& path, const cv::Mat& bgr) {
    cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();
    return stbi_write_jpg(path.c_str(), rgb.cols, rgb.rows, 3, rgb.data, 90) != 0;
}

int main(int argc, char** argv) {
    CLI::App app{"yolomaster_edge - universal YOLO-Master edge runner (ONNX / ncnn)"};
    std::string model, source, backend = "auto", classes_opt = "auto", outdir = "runs_edge";
    std::string device = "cpu", savetxt;
    int imgsz = 0, threads = 4, limit = 0, max_det = 300;
    float conf = 0.25f, iou = 0.50f;
    bool no_save = false, quiet = false, multilabel = false;

    app.add_option("-m,--model", model, "model: .onnx file, or ncnn dir / .param")->required();
    app.add_option("-s,--source", source, "image / directory / video / dataset.yaml")->required();
    app.add_option("-b,--backend", backend, "auto|onnx|ncnn")->default_str("auto");
    app.add_option("-d,--device", device, "cpu|cuda (onnx backend; falls back to cpu)")->default_str("cpu");
    app.add_option("--classes", classes_opt, "auto|visdrone|sku (auto = from model metadata)")->default_str("auto");
    app.add_option("--imgsz", imgsz, "inference size (0 = from model / 640)");
    app.add_option("--conf", conf, "confidence threshold")->capture_default_str();
    app.add_option("--iou", iou, "NMS IoU threshold")->capture_default_str();
    app.add_option("--max-det", max_det, "max detections per image after NMS")->capture_default_str();
    app.add_option("--threads", threads, "CPU threads")->capture_default_str();
    app.add_option("--limit", limit, "cap #inputs (0 = all)");
    app.add_option("--out", outdir, "output dir for annotated results")->capture_default_str();
    app.add_option("--save-txt", savetxt, "dir to write per-image predictions ('class conf x1 y1 x2 y2')");
    app.add_flag("--multi-label", multilabel, "one detection per class>conf per anchor (matches ultralytics val mAP)");
    app.add_flag("--no-save", no_save, "do not write annotated outputs");
    app.add_flag("--quiet", quiet, "suppress per-image logs");
    CLI11_PARSE(app, argc, argv);

    // ---- backend auto-detect from the model path ----
    if (backend == "auto") {
        std::error_code ec;
        if (fs::is_directory(model, ec) || ends_with(model, ".param")) backend = "ncnn";
        else if (ends_with(model, ".onnx")) backend = "onnx";
        else { std::cerr << "cannot infer backend from '" << model << "'; pass --backend\n"; return 2; }
    }

    // ---- construct backend ----
    std::unique_ptr<Backend> be;
    try {
        if (backend == "onnx") {
#ifdef USE_ORT
            be = std::make_unique<OrtBackend>(model, threads, device);
#else
            std::cerr << "built without ONNXRuntime backend\n"; return 2;
#endif
        } else if (backend == "ncnn") {
#ifdef USE_NCNN
            std::string param = model, bin;
            std::error_code ec;
            if (fs::is_directory(model, ec)) {
                param = (fs::path(model) / "model.ncnn.param").string();
                bin = (fs::path(model) / "model.ncnn.bin").string();
            } else bin = param.substr(0, param.rfind('.')) + ".bin";
            be = std::make_unique<NcnnBackend>(param, bin, threads);
#else
            std::cerr << "built without ncnn backend\n"; return 2;
#endif
        } else { std::cerr << "unknown backend: " << backend << "\n"; return 2; }
    } catch (const std::exception& e) {
        std::cerr << "backend init failed: " << e.what() << "\n"; return 3;
    }

    // ---- resolve config: --flag > model metadata > default ----
    Config cfg;
    cfg.conf_thresh = conf;
    cfg.iou_thresh = iou;
    cfg.max_det = max_det;
    cfg.multi_label = multilabel;
    int want = imgsz > 0 ? imgsz : (be->meta_imgsz > 0 ? be->meta_imgsz : 640);
    if (be->fixed_imgsz > 0 && want != be->fixed_imgsz) {
        std::cerr << "[warn] model requires fixed imgsz=" << be->fixed_imgsz
                  << "; overriding requested imgsz=" << want << "\n";
        want = be->fixed_imgsz;
    }
    cfg.imgsz = want;
    std::string classes_src;
    if (classes_opt == "visdrone") { cfg.class_names = visdrone_classes(); classes_src = "flag:visdrone"; }
    else if (classes_opt == "sku" || classes_opt == "sku110k") { cfg.class_names = sku110k_classes(); classes_src = "flag:sku"; }
    else if (!be->meta_names.empty()) { cfg.class_names = be->meta_names; classes_src = "model-metadata"; }
    else { cfg.class_names = visdrone_classes(); classes_src = "fallback:visdrone"; }

    std::cout << "[model] " << model << "  backend=" << backend << "  ep=" << be->active_ep
              << "  imgsz=" << cfg.imgsz << "  nc=" << cfg.num_classes() << " (" << classes_src << ")"
              << "  conf=" << cfg.conf_thresh << "  iou=" << cfg.iou_thresh << "  max_det=" << cfg.max_det << "\n";

    if (!no_save) { std::error_code ec; fs::create_directories(outdir, ec); }
    if (!savetxt.empty()) { std::error_code ec; fs::create_directories(savetxt, ec); }

    // ---- run over the source ----
    const SourceKind kind = classify_source(source);
    auto t_start = std::chrono::high_resolution_clock::now();
    long frames = 0, total_dets = 0;
    double sum_pre = 0, sum_inf = 0, sum_post = 0;

    auto run_one = [&](const cv::Mat& img, const std::string& tag) {
        if (img.empty()) { std::cerr << "  [skip] unreadable: " << tag << "\n"; return; }
        std::vector<Detection> dets;
        try {
            dets = be->infer(img, cfg);
        } catch (const std::exception& e) {
            std::cerr << "  [skip] inference error on " << tag << ": " << e.what() << "\n";
            return;
        }
        frames++; total_dets += static_cast<long>(dets.size());
        sum_pre += be->pre_ms; sum_inf += be->infer_ms; sum_post += be->post_ms;
        if (!quiet)
            std::cout << "  " << tag << "  dets=" << dets.size()
                      << "  infer=" << be->infer_ms << "ms\n";
        if (!no_save) {
            cv::Mat vis = img.clone();
            draw(vis, dets, cfg);
            imwrite_jpg((fs::path(outdir) / (fs::path(tag).stem().string() + ".jpg")).string(), vis);
        }
        if (!savetxt.empty()) {                       // 'class conf x1 y1 x2 y2' (pixel xyxy)
            std::ofstream f((fs::path(savetxt) / (fs::path(tag).stem().string() + ".txt")).string());
            for (const auto& d : dets)
                f << d.class_id << ' ' << d.conf << ' ' << d.box.x << ' ' << d.box.y << ' '
                  << (d.box.x + d.box.width) << ' ' << (d.box.y + d.box.height) << '\n';
        }
    };

    if (kind == SourceKind::Video) {
#ifdef HAVE_VIDEOIO
        cv::VideoCapture cap(source);
        if (!cap.isOpened()) { std::cerr << "cannot open video: " << source << "\n"; return 4; }
        cv::Mat frame; long idx = 0;
        while (cap.read(frame)) {
            if (limit > 0 && idx >= limit) break;
            run_one(frame, source + "#" + std::to_string(idx));
            ++idx;
        }
#else
        std::cerr << "video source not supported in this portable build; use image/dir/dataset\n";
        return 4;
#endif
    } else {
        auto imgs = gather_images(source, limit);
        if (imgs.empty()) { std::cerr << "no inputs resolved from source: " << source << "\n"; return 4; }
        for (const auto& p : imgs) run_one(imread_bgr(p), p);
    }

    if (frames == 0) { std::cerr << "no frames processed\n"; return 5; }
    const double wall = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
    const double avg = (sum_pre + sum_inf + sum_post) / frames;
    std::cout << "\n[summary] frames=" << frames << "  total_dets=" << total_dets
              << "  avg/frame: pre=" << sum_pre / frames << " infer=" << sum_inf / frames
              << " post=" << sum_post / frames << " total=" << avg << "ms"
              << "  model-FPS=" << 1000.0 / avg << "  wall=" << wall << "s\n";
    if (!no_save) std::cout << "[saved] annotated -> " << outdir << "/\n";
    return 0;
}
