#include "mnn_backend.hpp"
#include <MNN/MNNForwardType.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <cstdlib>
#include "json.hpp"

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

MnnBackend::MnnBackend(const std::string& model_path, int threads, const std::string& forward, Precision precision)
    : threads_(threads) {
    interp_ = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(model_path.c_str()),
        [](MNN::Interpreter* p) { if (p) MNN::Interpreter::destroy(p); });
    if (!interp_) throw std::runtime_error("MNN: failed to load " + model_path);

    const MNNForwardType fwd = forward == "opencl" ? MNN_FORWARD_OPENCL
                             : forward == "vulkan" ? MNN_FORWARD_VULKAN
                             : forward == "cuda"   ? MNN_FORWARD_CUDA
                                                   : MNN_FORWARD_CPU;
    const bool gpu = (fwd != MNN_FORWARD_CPU);
    const std::filesystem::path mp(model_path);
    auto note = [this](const std::string& s) { if (!ep_note.empty()) ep_note += "; "; ep_note += s; };

    // ---- fp16 policy (same contract as the ncnn backend: never a silent zero-detection run) ----
    // Precision_Low = fp16 compute on GPU backends. The emulated MoE routers (TopK / softmax gates
    // with 1e30-class masks) overflow in fp16, so a model whose metadata says fp16_safe: false runs
    // fp16 only through a multi-path session that keeps its routing segments in fp32
    // ("<model>.paths.json", written by scripts/server/mnn_router_paths.py); without that sidecar
    // the request is downgraded to fp32 and explained in ep_note.
    int yaml_fp16_safe = -1;
    {
        std::string v;
        const std::string per_model = (mp.parent_path() / (mp.stem().string() + ".metadata.yaml")).string();
        const std::string shared    = (mp.parent_path() / "metadata.yaml").string();
        if (meta::read_ncnn_yaml_scalar(per_model, "fp16_safe", v) || meta::read_ncnn_yaml_scalar(shared, "fp16_safe", v))
            yaml_fp16_safe = (v.find("rue") != std::string::npos) ? 1 : 0;
    }
    bool want_fp16 = gpu && precision == Precision::Fp16;
    std::string paths_file;
    for (const std::string cand : {model_path + ".paths.json",
                                   (mp.parent_path() / (mp.stem().string() + ".paths.json")).string(),
                                   (mp.parent_path() / "model.paths.json").string()})
        if (std::filesystem::exists(cand)) { paths_file = cand; break; }
    // The multi-path session is EXPERIMENTAL and opt-in (YOLOMASTER_MNN_MULTIPATH=1): with MNN 3.6.1
    // the Tensor-mode paths produced wrong outputs on CPU (299 vs 5581 detections on 20 VisDrone
    // images, with and without saveTensors), so by default an fp16 request on a routed model is
    // downgraded to fp32 and explained, never run as a silent zero-detection session.
    bool mixed = false;
    const bool multipath_env = std::getenv("YOLOMASTER_MNN_MULTIPATH") != nullptr;
    if (want_fp16 && yaml_fp16_safe == 0) {
        if (!paths_file.empty() && multipath_env) mixed = true;
        else { want_fp16 = false; note("fp16 refused: routed model (metadata fp16_safe: false); running fp32"); }
    }
    if (!mixed && !paths_file.empty() && multipath_env) mixed = true;   // structural test on any device

    bcs_.clear(); scs_.clear();
    auto make_bc = [&](bool low) {
        MNN::BackendConfig bc;
        bc.precision = low ? MNN::BackendConfig::Precision_Low : MNN::BackendConfig::Precision_High;
        bc.power     = MNN::BackendConfig::Power_High;
        return bc;
    };
    if (mixed) {
        nlohmann::json doc;
        try { std::ifstream f(paths_file); f >> doc; } catch (const std::exception& e) {
            throw std::runtime_error("MNN: unreadable paths sidecar " + paths_file + ": " + e.what());
        }
        const auto& segs = doc.at("segments");
        bcs_.reserve(segs.size()); scs_.reserve(segs.size());
        int n32 = 0;
        for (const auto& sg : segs) {
            const bool low = want_fp16 && sg.at("precision").get<std::string>() == "fp16";
            if (!low) ++n32;
            bcs_.push_back(make_bc(low));
            MNN::ScheduleConfig sc;
            sc.numThread = threads;
            sc.type = fwd;
            sc.backupType = MNN_FORWARD_CPU;
            sc.path.mode = MNN::ScheduleConfig::Path::Tensor;
            for (const auto& t : sg.at("inputs"))  sc.path.inputs.push_back(t.get<std::string>());
            for (const auto& t : sg.at("outputs")) sc.path.outputs.push_back(t.get<std::string>());
            sc.saveTensors = sc.path.outputs;   // cross-pipeline tensors must survive until the next path reads them
            scs_.push_back(sc);
        }
        for (size_t i = 0; i < scs_.size(); ++i) scs_[i].backendConfig = &bcs_[i];
        session_ = interp_->createMultiPathSession(scs_);
        if (!session_) throw std::runtime_error("MNN: createMultiPathSession failed for " + model_path + " with " + paths_file);
        note(std::to_string(n32) + " routing segments fp32, " + std::to_string(segs.size() - n32) + " segments fp16");
    } else {
        bcs_.push_back(make_bc(want_fp16));
        MNN::ScheduleConfig sc;
        sc.numThread = threads;
        sc.type = fwd;
        sc.backupType = MNN_FORWARD_CPU;   // fall back to CPU if the GPU backend is unavailable at runtime
        sc.backendConfig = &bcs_.back();
        scs_.push_back(sc);
        session_ = interp_->createSession(scs_.back());
        if (!session_) throw std::runtime_error("MNN: createSession failed for " + model_path);
    }
    input_  = interp_->getSessionInput(session_, nullptr);    // first input
    output_ = interp_->getSessionOutput(session_, nullptr);   // first output
    if (!input_ || !output_) throw std::runtime_error("MNN: could not resolve input/output tensor");
    active_ep = forward == "opencl" ? "MNN-OpenCL"
              : forward == "vulkan" ? "MNN-Vulkan"
              : forward == "cuda"   ? "MNN-CUDA" : "MNN-CPU";
    if (gpu) active_ep += want_fp16 ? (mixed ? "-fp16-mixed" : "-fp16") : "-fp32";
    else if (mixed) active_ep += "-multipath";

    // YOLO-Master graphs bake the attention token counts at the training size -> fixed input.
    auto ishape = input_->shape();   // NCHW, e.g. {1,3,640,640}
    if (ishape.size() == 4 && ishape[2] == ishape[3] && ishape[2] > 0) {
        fixed_imgsz = ishape[2];
        meta_imgsz  = ishape[2];
    }
    // MNN has no built-in metadata map -> read an optional class-name sidecar. Prefer a per-model
    // "<model>.metadata.yaml" (so several .mnn can share a dir); fall back to "metadata.yaml".
    const std::string per_model = (mp.parent_path() / (mp.stem().string() + ".metadata.yaml")).string();
    const std::string shared    = (mp.parent_path() / "metadata.yaml").string();
    std::vector<std::string> nm; int mi = 0;
    if (meta::read_ncnn_yaml(per_model, nm, mi, end2end_) || meta::read_ncnn_yaml(shared, nm, mi, end2end_)) {
        meta_names = nm;
        if (mi > 0) { meta_imgsz = mi; if (fixed_imgsz == 0) fixed_imgsz = mi; }
    }
}

MnnBackend::~MnnBackend() {
    if (interp_ && session_) interp_->releaseSession(session_);   // interp_ freed by shared_ptr deleter
}

std::vector<Detection> MnnBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    // ---- preprocess: letterbox -> NCHW float RGB /255 (identical to ORT) ----
    auto t0 = clk::now();
    LetterboxInfo lb;
    const int sz = cfg.imgsz;
    std::vector<float>& blob = blob_;
    blob.resize(static_cast<size_t>(3) * sz * sz);
    preprocess_nchw(bgr, sz, cfg.stretch, blob.data(), lb);
    // resize the session input if it doesn't already match imgsz (handles fixed & flexible graphs)
    auto ishape = input_->shape();
    if (ishape.size() != 4 || ishape[2] != sz || ishape[3] != sz) {
        interp_->resizeTensor(input_, std::vector<int>{1, 3, sz, sz});
        interp_->resizeSession(session_);
        output_ = interp_->getSessionOutput(session_, nullptr);
    }
    pre_ms = ms_since(t0);

    // ---- inference: copy blob into the input tensor (NCHW/Caffe), run ----
    auto t1 = clk::now();
    {
        MNN::Tensor host(input_, MNN::Tensor::CAFFE);   // NCHW host tensor shaped like input_
        std::memcpy(host.host<float>(), blob.data(), blob.size() * sizeof(float));
        input_->copyFromHostTensor(&host);
    }
    interp_->runSession(session_);
    infer_ms = ms_since(t1);

    // ---- postprocess: detection = rank-3 output [1,feat,anchors]; proto (seg) = rank-4 [1,nm,mh,mw] ----
    auto t2 = clk::now();
    auto all = interp_->getSessionOutputAll(session_);
    MNN::Tensor* detT = nullptr; MNN::Tensor* protoT = nullptr;
    for (auto& kv : all) {
        if (kv.second->shape().size() == 4) protoT = kv.second;
        else                                detT   = kv.second;
    }
    if (!detT) detT = output_;                          // detection-only safety
    MNN::Tensor detHost(detT, MNN::Tensor::CAFFE);
    detT->copyToHostTensor(&detHost);
    const auto os = detHost.shape();
    const float* raw = detHost.host<float>();
    // end2end [1,num_det,6] MUST be caught before the smaller-axis transpose heuristic,
    // which would otherwise silently mangle it into feat=6/anchors=num_det garbage.
    if (os.size() == 3 && (end2end_ || looks_end2end(os[1], os[2]))) {
        candidates = decode_end2end(raw, os[1], cfg, lb);
    } else {
        int feat_dim = 0, num_anchors = 0;
        const float* dec = raw;
        std::vector<float> buf;
        if (os.size() == 3) {
            if (os[1] <= os[2]) { feat_dim = os[1]; num_anchors = os[2]; }   // channel-major (expected)
            else {                                                          // [1,anchors,feat] -> transpose
                feat_dim = os[2]; num_anchors = os[1];
                buf.resize(static_cast<size_t>(feat_dim) * num_anchors);
                for (int a = 0; a < num_anchors; ++a)
                    for (int f = 0; f < feat_dim; ++f)
                        buf[static_cast<size_t>(f) * num_anchors + a] = raw[static_cast<size_t>(a) * feat_dim + f];
                dec = buf.data();
            }
        }
        candidates = decode_candidates(dec, feat_dim, num_anchors, cfg, lb);
    }
    cand_orig_w = lb.orig_w; cand_orig_h = lb.orig_h; cand_lb = lb;
    proto.clear(); proto_c = proto_h = proto_w = 0;
    if (protoT) {                                       // segmentation proto
        MNN::Tensor protoHost(protoT, MNN::Tensor::CAFFE);
        protoT->copyToHostTensor(&protoHost);
        const auto ps = protoHost.shape();              // {1, nm, mh, mw}
        proto_c = (int)ps[1]; proto_h = (int)ps[2]; proto_w = (int)ps[3];
        const float* pp = protoHost.host<float>();
        proto.assign(pp, pp + (size_t)proto_c * proto_h * proto_w);
    }
    auto dets = nms_and_cap(candidates, cfg, lb.orig_w, lb.orig_h);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
