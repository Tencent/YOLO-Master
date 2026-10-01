// yolomaster_edge - universal, adaptive YOLO-Master edge runner.
// Runtime model loading (no baked-in weights), backend/classes/imgsz auto-detected
// from the model, versatile --source (image / dir / video / dataset.yaml).
#include "yolomaster.hpp"
#include "slicing.hpp"
#include "bench.hpp"
#include "tracker.hpp"
#include "map_metrics.hpp"
#include "annotate_export.hpp"
#ifdef USE_ORT
#include "ort_backend.hpp"
#endif
#ifdef USE_NCNN
#include "ncnn_backend.hpp"
#endif
#ifdef USE_MNN
#include "mnn_backend.hpp"
#endif
#ifdef USE_TRT
#include "trt_backend.hpp"
#endif
#include "CLI11.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <regex>
#include <sstream>
#include <unistd.h>
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

// --help after CLI11 laid it out: one blank line between options, and on a terminal (NO_COLOR
// honoured) the everyday options in cyan and the section headers bold. The escapes are added
// after layout, so the columns stay aligned.
static std::string colorize_help(const std::string &help) {
    const bool color = isatty(STDOUT_FILENO) && !std::getenv("NO_COLOR");
    static const char *common[] = {"--model", "--source", "--backend", "--device", "--precision", "--conf", "--iou",
                                   "--out", "--no-save", "--quiet", "--bench", "--accuracy", "--track", "--help"};
    static const std::regex head(R"(^(\s+)(-[A-Za-z],\s+)?(--[a-z-]+)(.*)$)");
    std::string out; out.reserve(help.size() + 1024);
    std::istringstream in(help); std::string line;
    bool in_options = false, first = true;
    while (std::getline(in, line)) {
        std::smatch m;
        if (std::regex_match(line, m, head)) {
            if (in_options && !first) out += '\n';   // a little air between options
            first = false;
            if (color) {
                const std::string name = m[3];
                bool hot = false;
                for (const char *c : common) if (name == c) { hot = true; break; }
                if (hot) line = m[1].str() + "\033[36m" + m[2].str() + name + "\033[0m" + m[4].str();
            }
        } else if (!line.empty() && line.back() == ':' && line.find(' ') == std::string::npos) {
            in_options = true; first = true;
            if (color) line = "\033[1m" + line + "\033[0m";   // section headers (OPTIONS:)
        }
        out += line; out += '\n';
    }
    return out;
}

int main(int argc, char** argv) {
    CLI::App app{"yolomaster_edge - universal YOLO-Master edge runner (ONNX / ncnn / MNN)"};
    // help layout: a wider name column so "--precision TEXT [auto]" and friends keep their
    // description on the same line with a gap, and a wider paragraph before wrapping
    app.get_formatter()->column_width(44);
    app.get_formatter()->right_column_width(72);
    app.set_version_flag("--version", std::string(YM_VERSION) + " (" + YM_GIT_COMMIT + ")");
    std::string model, source, backend = "auto", classes_opt = "auto", outdir = "runs_edge";
    std::string device = "cpu", savetxt;
    int imgsz = 0, threads = 4, limit = 0, max_det = 300, warmup = 0;
    float conf = 0.25f, iou = 0.50f;
    bool no_save = false, quiet = false, multilabel = false, stretch = false;
    std::string slicing = "off", label_format = "yolo", sampling = "1s", export_labels;
    int tile_size = 0;
    bool slicing_masks = false, cw_nms = false;
    float sigma = 0.1f;
    std::string precision_s = "auto";
    std::string bench_mode = "off", bench_json, accuracy;
    std::string track_mode = "off";
    int track_buffer = 30;
    bool cpu_preproc = false, cuda_graph = false;
    int bench_iters = 50, bench_warmup = 10;
    double bench_minutes = 2.0;

    app.add_option("-m,--model", model, "model: .onnx file, or ncnn dir / .param")->required();
    app.add_option("-s,--source", source, "image / directory / video / dataset.yaml")->required();
    app.add_option("-b,--backend", backend, "auto|onnx|ncnn|mnn")->default_str("auto");
    app.add_option("-d,--device", device, "cpu|cuda|trt|coreml (onnx backend; trt=TensorRT EP, coreml=Apple CoreML EP)")->default_str("cpu");
    app.add_option("--classes", classes_opt, "auto|visdrone|sku (auto = from model metadata)")->default_str("auto");
    app.add_option("--imgsz", imgsz, "inference size (0 = from model / 640)");
    app.add_option("--conf", conf, "confidence threshold")->capture_default_str();
    app.add_option("--iou", iou, "NMS IoU threshold")->capture_default_str();
    app.add_option("--max-det", max_det, "max detections per image after NMS (tiled runs only; requires --slicing)")->capture_default_str();
    app.add_option("--threads", threads, "CPU threads")->capture_default_str();
    app.add_option("--precision", precision_s,
                   "ncnn numeric precision: auto|fp32|fp16|int8 (auto = fp16 for fp16-safe models on armv8.2 CPUs, "
                   "fp32 pinned for emulated-router mixture graphs; int8 = load the <name>-int8_ncnn sibling)")
        ->default_str("auto");
    app.add_option("--limit", limit, "cap #inputs (0 = all)");
    app.add_option("--warmup", warmup, "run the first input N extra times before timing starts (0 = off)");
    app.add_option("--out", outdir, "output dir for annotated results")->capture_default_str();
    app.add_option("--save-txt", savetxt, "dir to write per-image predictions ('class conf x1 y1 x2 y2')");
    app.add_flag("--multi-label", multilabel, "one detection per class>conf per anchor (matches ultralytics val mAP)");
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
    app.add_option("--bench", bench_mode, "off|cold|sustained: benchmark mode (yolomaster-bench/v1 JSON): gray-probe run "
                   "(cold: --bench-warmup + --bench-iters; sustained: a --bench-minutes loop) plus per-stage stats of the "
                   "dataset pass; images/dirs/dataset.yaml only")->default_str("off");
    app.add_option("--bench-iters", bench_iters, "timed probe forwards in the cold run")->capture_default_str();
    app.add_option("--bench-warmup", bench_warmup, "untimed probe forwards before the timed run")->capture_default_str();
    app.add_option("--bench-minutes", bench_minutes, "sustained mode: loop duration")->capture_default_str();
    app.add_option("--bench-json", bench_json, "write the bench result JSON here (default <out>/bench.json)");
    app.add_option("--accuracy", accuracy, "score the source with the in-process mAP: a YOLO labels dir, or 'auto' to map "
                   ".../images/... to .../labels/... (dataset.yaml sources). Runs a second pass at the val protocol "
                   "(conf 0.001, iou 0.7, multi-label, max_det 300); implies --bench cold");
    app.add_option("--track", track_mode, "off|botsort|bytetrack: multi-object tracking on video sources (ids drawn, "
                   "--save-txt gains a 7th column track_id); botsort adds camera motion compensation")->default_str("off");
    app.add_option("--track-buffer", track_buffer, "frames a lost track is kept before its id retires")->capture_default_str();
    app.add_flag("--cpu-preproc", cpu_preproc, "TensorRT / ORT-CUDA: preprocess on the CPU instead of the CUDA kernel (parity runs)");
    app.add_flag("--cuda-graph", cuda_graph, "TensorRT: capture the per-frame stream work into a CUDA graph and replay it");
    try {
        app.parse(argc, argv);
    } catch (const CLI::CallForHelp &) {
        std::cout << colorize_help(app.help());
        return 0;
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }
    track::TrackerConfig tcfg;
    const bool track_on = track_mode != "off";
    if (track_on && !track::parse_tracker_kind(track_mode, tcfg.kind)) {
        std::cerr << "unknown --track mode: " << track_mode << " (off|botsort|bytetrack)\n"; return 2;
    }
    tcfg.track_buffer = track_buffer;
    if (bench_mode != "off" && bench_mode != "cold" && bench_mode != "sustained") {
        std::cerr << "unknown --bench mode: " << bench_mode << " (off|cold|sustained)\n"; return 2;
    }
    if (!accuracy.empty() && bench_mode == "off") bench_mode = "cold";
    const bool bench_on = bench_mode != "off";

    Precision precision = Precision::Auto;
    if (!parse_precision(precision_s, precision)) {
        std::cerr << "unknown --precision: " << precision_s << " (auto|fp32|fp16|int8)\n"; return 2;
    }

    SliceMode slice_mode = SliceMode::Off;
    if (slicing == "dense") slice_mode = SliceMode::Dense;
    else if (slicing == "sparse") slice_mode = SliceMode::Sparse;
    else if (slicing != "off") { std::cerr << "unknown --slicing mode: " << slicing << "\n"; return 2; }
    // the adjustable cap is a tiled-inference feature (merged tile pools can legitimately
    // exceed 300 dets); single-pass runs keep the ultralytics default
    if (app.count("--max-det") && slice_mode == SliceMode::Off) {
        std::cerr << "--max-det is only available with tiled inference (add --slicing dense|sparse)\n";
        return 2;
    }
    annot::Format lfmt = annot::Format::YoloTXT;
    if (label_format == "coco") lfmt = annot::Format::CocoJSON;
    else if (label_format == "voc") lfmt = annot::Format::PascalVOC;
    else if (label_format != "yolo") { std::cerr << "unknown --label-format: " << label_format << "\n"; return 2; }

    // ---- backend auto-detect from the model path ----
    if (backend == "auto") {
        std::error_code ec;
        if (fs::is_directory(model, ec) || ends_with(model, ".param")) backend = "ncnn";
        else if (ends_with(model, ".onnx")) {
#if defined(USE_ORT)
            backend = "onnx";
#elif defined(USE_TRT)
            backend = "trt";     // a TensorRT-only build compiles the .onnx into an engine
#else
            backend = "onnx";
#endif
        }
        else if (ends_with(model, ".mnn")) backend = "mnn";
        else if (ends_with(model, ".engine") || ends_with(model, ".trt")) backend = "trt";
        else { std::cerr << "cannot infer backend from '" << model << "'; pass --backend\n"; return 2; }
    }

    // ---- construct backend ----
    std::unique_ptr<Backend> be;
    try {
        if (backend == "onnx") {
#ifdef USE_ORT
            OrtOptions oo; oo.device = device; oo.threads = threads; oo.gpu_preproc = !cpu_preproc;
            be = std::make_unique<OrtBackend>(model, oo);
#else
            std::cerr << "built without ONNXRuntime backend\n"; return 2;
#endif
        } else if (backend == "ncnn") {
#ifdef USE_NCNN
            // --precision int8 selects the pre-quantized "<name>-int8_ncnn" sibling; a missing
            // sibling is a hard error so an int8 number can never come from a float model.
            const std::string m = (precision == Precision::Int8) ? meta::ncnn_int8_sibling(model) : model;
            std::string param = m, bin;
            std::error_code ec;
            if (fs::is_directory(m, ec)) {
                param = (fs::path(m) / "model.ncnn.param").string();
                bin = (fs::path(m) / "model.ncnn.bin").string();
            } else bin = param.substr(0, param.rfind('.')) + ".bin";
            if (!fs::exists(param, ec)) {
                std::cerr << (precision == Precision::Int8 ? "int8 model not found: " : "ncnn model not found: ")
                          << param << "\n";
                return 2;
            }
            be = std::make_unique<NcnnBackend>(param, bin, threads, false, precision);
#else
            std::cerr << "built without ncnn backend\n"; return 2;
#endif
        } else if (backend == "mnn") {
#ifdef USE_MNN
            be = std::make_unique<MnnBackend>(model, threads, device == "cuda" ? "cuda" : "cpu", precision);
#else
            std::cerr << "built without MNN backend (rebuild with -DUSE_MNN=ON)\n"; return 2;
#endif
        } else if (backend == "trt") {
#ifdef USE_TRT
            // .engine loads as-is; .onnx is built (fp32, or fp16 with --precision fp16) and cached
            TrtOptions topt;
            topt.fp16 = (precision == Precision::Fp16);
            topt.gpu_preproc = !cpu_preproc; topt.cuda_graph = cuda_graph;
            be = std::make_unique<TrtBackend>(model, topt);
#else
            std::cerr << "built without TensorRT backend (rebuild with -DUSE_TRT=ON)\n"; return 2;
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
    cfg.stretch = stretch;
    cfg.nms_mode = cw_nms ? NmsMode::ClusterWeighted : NmsMode::Standard;
    cfg.cw_sigma = std::min(0.5f, std::max(0.01f, sigma));
    int want = imgsz > 0 ? imgsz : (be->meta_imgsz > 0 ? be->meta_imgsz : 640);
    if (be->fixed_imgsz > 0 && want != be->fixed_imgsz) {
        std::cerr << "[warn] model requires fixed imgsz=" << be->fixed_imgsz
                  << "; overriding requested imgsz=" << want << "\n";
        want = be->fixed_imgsz;
    }
    cfg.imgsz = want;
    // tracking wants the low-score detections for its second association: unless the user pinned
    // --conf, the detector floor drops to the tracker's low threshold (ultralytics does the same)
    if (track_on && !app.count("--conf")) cfg.conf_thresh = tcfg.track_low_thresh;
    std::string classes_src;
    if (classes_opt == "visdrone") { cfg.class_names = visdrone_classes(); classes_src = "flag:visdrone"; }
    else if (classes_opt == "sku" || classes_opt == "sku110k") { cfg.class_names = sku110k_classes(); classes_src = "flag:sku"; }
    else if (!be->meta_names.empty()) { cfg.class_names = be->meta_names; classes_src = "model-metadata"; }
    else { cfg.class_names = visdrone_classes(); classes_src = "fallback:visdrone"; }

    std::cout << "[model] " << model << "  backend=" << backend << "  ep=" << be->active_ep
              << "  imgsz=" << cfg.imgsz << "  nc=" << cfg.num_classes() << " (" << classes_src << ")"
              << "  conf=" << cfg.conf_thresh << "  iou=" << cfg.iou_thresh << "  max_det=" << cfg.max_det
              << (be->ep_note.empty() ? "" : "  note=" + be->ep_note) << "\n";

    if (!no_save) { std::error_code ec; fs::create_directories(outdir, ec); }
    if (!savetxt.empty()) { std::error_code ec; fs::create_directories(savetxt, ec); }

    // ---- run over the source ----
    const SourceKind kind = classify_source(source);
    std::vector<std::string> imgs;                 // image sources (bench / accuracy re-use the list)
    if (bench_on && kind == SourceKind::Video) {
        std::cerr << "--bench / --accuracy apply to images, directories and dataset.yaml sources only\n"; return 2;
    }
    if (track_on && kind != SourceKind::Video) {
        std::cerr << "--track applies to video sources only\n"; return 2;
    }
    std::unique_ptr<track::Tracker> tracker;   // created once the capture fps is known
    std::vector<int> track_ids;                // parallel to `dets` inside run_one when tracking
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
    long frames = 0, total_dets = 0;
    double sum_pre = 0, sum_inf = 0, sum_post = 0;
    bench::Samples samples;                        // per-frame stage samples (bench JSON)
    bench::BenchResult bres;

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
                       bool do_export = true, const std::string& coco_file = "", int coco_id = 0) {
        if (img.empty()) { std::cerr << "  [skip] unreadable: " << tag << "\n"; return; }
        std::vector<Detection> dets;
        std::string slice_note;
        try {
            if (slice_mode != SliceMode::Off) {
                const SliceOutput so = sliced_candidates(*be, img, cfg, sconf);
                dets = nms_and_cap(be->candidates, cfg, img.cols, img.rows);
                tstats.add(so.tiles_run, so.tiles_total, so.tile_size_used,
                           so.used_fallback, so.capped);
                slice_note = "  tiles=" + std::to_string(so.tiles_run) + "/"
                           + std::to_string(so.tiles_total) + " @" + std::to_string(so.tile_size_used) + "px"
                           + (so.used_fallback ? " [fallback]" : "") + (so.capped ? " [capped]" : "");
            } else {
                dets = be->infer(img, cfg);
            }
        } catch (const std::exception& e) {
            std::cerr << "  [skip] inference error on " << tag << ": " << e.what() << "\n";
            return;
        }
        std::vector<track::Track> tracks;
        track_ids.clear();
        if (tracker) {
            tracks = tracker->update(dets, &img);
            std::vector<Detection> tracked;
            tracked.reserve(tracks.size());
            for (const auto& t : tracks) {
                Detection d;
                d.class_id = t.class_id; d.conf = t.conf; d.box = t.box; d.mask_coeffs = t.mask_coeffs;
                d.cand_index = t.det_index >= 0 && t.det_index < static_cast<int>(dets.size()) ? dets[t.det_index].cand_index : -1;
                tracked.push_back(std::move(d));
                track_ids.push_back(t.id);
            }
            dets.swap(tracked);
        }
        if (!export_labels.empty() && do_export) {
            AnnotationSink& s = ensure_sink();
            annot::Image aimg;
            aimg.name = fs::path(tag).stem().string();
            if (coco_id > 0) aimg.name = fs::path(coco_file).stem().string();
            aimg.width = img.cols; aimg.height = img.rows;
            aimg.instances = annotation_instances(dets, be->is_seg(), be->proto, be->proto_c,
                                                  be->proto_h, be->proto_w, be->cand_lb, cfg.imgsz);
            s.add(aimg, coco_file.empty() ? fs::path(tag).filename().string() : coco_file, coco_id);
        }
        frames++; total_dets += static_cast<long>(dets.size());
        sum_pre += be->pre_ms; sum_inf += be->infer_ms; sum_post += be->post_ms;
        samples.add(*be, dets.size());
        if (!quiet)
            std::cout << "  " << tag << "  dets=" << dets.size()
                      << (tracker ? "  tracks=" + std::to_string(tracks.size()) : std::string())
                      << "  infer=" << be->infer_ms << "ms" << slice_note << "\n";
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
            if (tracker) track::draw_tracks(vis, tracks, cfg); else draw(vis, dets, cfg);
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
            } else
#endif
            imwrite_jpg((fs::path(outdir) /
                (unique_stem(out_stems, fs::path(tag).stem().string()) + ".jpg")).string(), vis);
        }
        if (!savetxt.empty()) {                       // 'class conf x1 y1 x2 y2' (pixel xyxy)
            std::string tstem = fs::path(tag).stem().string();
            if (video_mode && coco_id > 0) {          // frame-unique name (stem collides at "11")
                char b[64];
                std::snprintf(b, sizeof(b), "%s_%06d", fs::path(source).stem().string().c_str(),
                              coco_id - 1);
                tstem = b;
            }
            std::ofstream f((fs::path(savetxt) /
                (unique_stem(txt_stems, tstem) + ".txt")).string());
            for (size_t i = 0; i < dets.size(); ++i) {
                const auto& d = dets[i];
                f << d.class_id << ' ' << d.conf << ' ' << d.box.x << ' ' << d.box.y << ' '
                  << (d.box.x + d.box.width) << ' ' << (d.box.y + d.box.height);
                if (tracker && i < track_ids.size()) f << ' ' << track_ids[i];
                f << '\n';
            }
        }
    };

    if (kind == SourceKind::Video) {
#ifdef HAVE_VIDEOIO
        cv::VideoCapture cap(source);
        if (!cap.isOpened()) { std::cerr << "cannot open video: " << source << "\n"; return 4; }
        const double fps_probe = cap.get(cv::CAP_PROP_FPS);
        src_fps = (fps_probe > 1.0 && fps_probe < 1000.0) ? fps_probe : 30.0;
        if (track_on) {
            tcfg.fps = src_fps;
            tracker = std::make_unique<track::Tracker>(tcfg);
            std::cout << "[track] " << track::tracker_kind_name(tcfg.kind) << "  buffer=" << tcfg.track_buffer
                      << " frames  gmc=" << ((tcfg.kind == track::TrackerConfig::Kind::BotSort && track::gmc_available()) ? "on" : "off")
                      << "  conf_floor=" << cfg.conf_thresh << "\n";
        }
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
                write_jpg((fs::path(frames_dir) / fn).string(), frame);
                coco_file = std::string("frames/") + fn;
            }
            run_one(frame, source + "#" + std::to_string(idx), sampled, coco_file,
                    static_cast<int>(idx) + 1);
            ++idx;
        }
#else
        std::cerr << "video source not supported in this portable build; use image/dir/dataset\n";
        return 4;
#endif
    } else {
        imgs = gather_images(source, limit);
        if (imgs.empty()) { std::cerr << "no inputs resolved from source: " << source << "\n"; return 4; }
        if (bench_on) {     // the probe run doubles as the warm-up of the dataset pass
            bres.has_cold = true;
            bres.cold = bench::cold_run(*be, cfg, bench_warmup, bench_iters);
            std::cout << "[bench] cold probe " << bres.cold.probe_mode << ": infer median=" << bres.cold.infer_ms.median
                      << "ms p90=" << bres.cold.infer_ms.p90 << " min=" << bres.cold.infer_ms.min
                      << " (n=" << bres.cold.infer_ms.n << ")\n";
        } else if (warmup > 0) {   // untimed forwards on the first input (lazy allocations, cuDNN autotune, TRT context)
            cv::Mat w = imread_bgr(imgs.front());
            for (int i = 0; i < warmup && !w.empty(); ++i) { try { (void)be->infer(w, cfg); } catch (...) { break; } }
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
    if (bench_on) {
        if (bench_mode == "sustained") {
            bres.has_sustained = true;
            bres.sustained = bench::sustained_loop(*be, cfg, bench_warmup, bench_minutes, bench_iters);
            std::cout << "[bench] sustained " << bres.sustained.duration_s << "s " << bres.sustained.probe_mode
                      << ": cold median=" << bres.sustained.cold_median_ms << "ms sustained median="
                      << bres.sustained.sustained_median_ms << "ms throttle=" << bres.sustained.throttle_pct << "%\n";
        }
        if (!accuracy.empty()) {
            // second pass at the val protocol; boxes and confs rounded the way --save-txt prints them so
            // the in-process number equals scoring the txt dump with scripts/eval_map*.py
            Config cv = cfg;
            cv.conf_thresh = 0.001f; cv.iou_thresh = 0.7f; cv.multi_label = true; cv.max_det = 300;
            const std::string labels_dir = (accuracy == "auto") ? std::string() : accuracy;
            std::vector<metrics::ImageEval> evals;
            evals.reserve(imgs.size());
            std::vector<double> acc_infer;
            for (const auto& p : imgs) {
                cv::Mat img = imread_bgr(p);
                if (img.empty()) continue;
                std::vector<Detection> dets;
                try {
                    if (slice_mode != SliceMode::Off) {
                        (void)sliced_candidates(*be, img, cv, sconf);
                        dets = nms_and_cap(be->candidates, cv, img.cols, img.rows);
                    } else dets = be->infer(img, cv);
                } catch (const std::exception& e) {
                    std::cerr << "  [skip] accuracy pass error on " << p << ": " << e.what() << "\n"; continue;
                }
                acc_infer.push_back(be->infer_ms);
                metrics::ImageEval ev;
                metrics::load_yolo_labels(metrics::label_path_for(p, labels_dir), img.cols, img.rows, ev.gts);
                ev.preds = metrics::from_detections(dets, /*txt_rounding=*/true);
                evals.push_back(std::move(ev));
            }
            bres.accuracy.present = true;
            bres.accuracy.conf = cv.conf_thresh; bres.accuracy.iou = cv.iou_thresh;
            bres.accuracy.max_det = cv.max_det; bres.accuracy.multi_label = cv.multi_label;
            bres.accuracy.labels = labels_dir.empty() ? "auto" : labels_dir;
            bres.accuracy.map = metrics::evaluate(evals);
            bres.accuracy.infer_ms = bench::reduce(acc_infer);
            std::printf("[accuracy] images=%d  mAP50=%.4f  mAP50-95=%.4f\n", bres.accuracy.map.images,
                        bres.accuracy.map.map50, bres.accuracy.map.map5095);
        }
        bres.tool = "cli";
        bres.timestamp = bench::timestamp_utc();
        bres.model = bench::model_info(*be, model, backend, precision_s, cfg);
        bres.env = bench::collect_env(*be, threads);
        bres.protocol.mode = bench_mode;
        bres.protocol.conf = cfg.conf_thresh; bres.protocol.iou = cfg.iou_thresh;
        bres.protocol.max_det = cfg.max_det; bres.protocol.multi_label = cfg.multi_label;
        bres.protocol.slicing = slicing; bres.protocol.tile_size = tile_size;
        bres.protocol.warmup = bench_warmup; bres.protocol.iters = bench_iters; bres.protocol.minutes = bench_minutes;
        bres.protocol.probe_mode = bres.has_cold ? bres.cold.probe_mode : bres.sustained.probe_mode;
        bres.protocol.dataset = fs::path(source).stem().string();
        bres.protocol.image_count = static_cast<int>(imgs.size());
        bres.protocol.image_list_sha256 = bench::image_list_sha256(imgs);
        bres.has_dataset = true;
        bres.dataset.frames = frames; bres.dataset.total_dets = total_dets;
        bres.dataset.pre_ms = bench::reduce(samples.pre); bres.dataset.infer_ms = bench::reduce(samples.infer);
        bres.dataset.post_ms = bench::reduce(samples.post); bres.dataset.total_ms = bench::reduce(samples.total);
        bres.dataset.model_fps = 1000.0 / avg; bres.dataset.wall_s = wall;
        const std::string jpath = bench_json.empty() ? (fs::path(outdir) / "bench.json").string() : bench_json;
        { std::error_code ec; fs::create_directories(fs::path(jpath).parent_path(), ec); }
        std::ofstream jf(jpath);
        jf << bench::to_json(bres).dump(2) << "\n";
        std::cout << "[bench] json -> " << jpath << "\n";
    }
    if (sink) {
        const AnnotationSink::Result r = sink->finish();
        if (!r.error.empty()) std::cerr << "[labels] export failed: " << r.error << "\n";
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
    return 0;
}
