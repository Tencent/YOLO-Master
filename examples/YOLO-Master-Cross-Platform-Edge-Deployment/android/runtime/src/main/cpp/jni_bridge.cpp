// JNI bridge for the YOLO-Master runtime on Android (ncnn, and ONNX Runtime when USE_ORT).
//
// It wraps exactly one seam from the shared C++ core: Backend::forward_raw / infer(cv::Mat,
// Config). Nothing from the CLI driver (main.cpp), videoio, or the filesystem source layer is
// pulled in. Two runtimes share the Handle (unique_ptr<Backend>): NcnnBackend (CPU / Vulkan) and
// OrtBackend (CPU EP, or the QNN EP on the Hexagon NPU on arm64 with the QNN libs packaged).
// Robustness rules encoded here:
//   * Precision is PER MODEL (policy lives in NcnnBackend, ncnn_backend.cpp): AUTO runs fp16 on
//     armv8.2 CPUs for fp16-safe (dense) models and pins fp32 for the emulated-router mixture
//     graphs, whose export constants (1e-9 / 1e30) are unrepresentable in fp16 and would zero
//     the routing. The decision is read from the .param fingerprint, never from the caller.
//     An explicit FP16 the model cannot honour is downgraded and explained in backendNote.
//   * INT8 loads the pre-quantized "<name>-int8_ncnn" sibling (mixed per-layer int8). It is a
//     CPU story (no ncnn Vulkan int8 kernels) and a missing sibling is a hard init failure -
//     an int8 number must never silently come from a float model.
//   * Vulkan is verified available (get_gpu_count) before it is requested; otherwise we
//     transparently fall back to the CPU choice. activeBackend reports the REAL resolved
//     precision from the backend (ncnn-CPU-fp16 / -fp32 / -int8+fp16 / -int8+fp32 / ncnn-Vulkan
//     / ncnn-Vulkan-fp32), not the requested flags.
//   * Every entry point is try/catch -> typed error string, never a native crash.
//
// Two call shapes coexist:
//   * The harness shape (nativeInfer / nativeSegOverlay): one call = forward + NMS, the raw
//     candidates stay cached inside the Backend.
//   * The app shape (the iOS Kit contract): nativeForwardRaw* hands back a native RawOutput
//     handle - the candidates and the seg proto are std::move'd OUT of the Backend (zero copy,
//     native heap, not charged to the Java heap the Photo bitmaps need) - and nativeRawDecode /
//     nativeRawMaskOverlay work on that handle alone, so the Photo screen can retune conf/IoU
//     on a cached raw with no forward and even after the model (and Vulkan) is released.
//     Both shapes go through the same C++ nms_and_cap / seg_overlay: one NMS, one mask math.
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolomaster.hpp"
#include "ncnn_backend.hpp"   // pulls ncnn net.h -> platform.h (defines NCNN_VULKAN)
#include "cpu.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif
#ifdef USE_ORT
#include "ort_backend.hpp"
#include <dlfcn.h>
#endif

using namespace yolomaster;

#define LOG_TAG "YMNcnn"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// Kotlin Runtime.native / Unit.native codes (dev.yolomaster.ncnn.Types.kt): the JNI ABI, keep stable.
enum RuntimeCode : int { kRuntimeNcnn = 0, kRuntimeOnnx = 1 };
enum UnitCode : int { kUnitCpu = 0, kUnitGpu = 1, kUnitNpu = 2 };
// nativeCapabilities() bits.
enum CapBits : int { kCapNcnn = 1, kCapOrt = 2, kCapQnn = 4 };

struct Handle {
    std::unique_ptr<Backend> be;   // NcnnBackend or OrtBackend; everything below the init is runtime-agnostic
    int runtime = kRuntimeNcnn;
    Config cfg;
    std::string activeBackend;   // the backend's real active_ep (resolved precision / EP), not the request
    std::string note;            // backend ep_note: why a request was downgraded ("" if none)
    Precision precision = Precision::Auto;
    bool vulkan = false;
    bool seg = false;            // metadata.yaml `task: segment` (missing key = detect); no forward needed
    int nodesTotal = 0, nodesOnCpu = 0;   // ORT graph placement at init (0/0 for ncnn)
};

// One forward's raw result, owned by Kotlin (dev.yolomaster.ncnn.RawOutput) until
// nativeRawRelease. Candidates are decoded to original-image px with score >= confFloor;
// the seg proto is plane-major [proto_c][proto_h][proto_w] (empty on detection models).
struct RawOutput {
    std::vector<RawDet> candidates;
    LetterboxInfo lb;
    int orig_w = 0, orig_h = 0, imgsz = 0;
    std::vector<float> proto;
    int proto_c = 0, proto_h = 0, proto_w = 0;
    double pre_ms = 0, infer_ms = 0, decode_ms = 0;
    bool is_seg() const { return proto_c > 0 && !proto.empty(); }
};

// android.graphics.Bitmap plumbing, resolved once in JNI_OnLoad (FindClass from a non-main
// thread would go through the wrong class loader; caching the ids is also the cheap path
// for the 30 fps mask overlay).
jclass g_bitmap_cls = nullptr;
jmethodID g_bitmap_create = nullptr;   // static Bitmap.createBitmap(int, int, Bitmap.Config)
jobject g_bitmap_argb8888 = nullptr;   // Bitmap.Config.ARGB_8888

// The ncnn Vulkan instance is process-global; ref-count it across handles.
std::mutex g_gpu_mu;
int g_gpu_refs = 0;
thread_local std::string g_last_error;

// Try to make a usable Vulkan GPU available. Returns false (and stays CPU) when ncnn
// was built without Vulkan, no device exists, or instance creation fails.
bool acquire_gpu() {
#if NCNN_VULKAN
    std::lock_guard<std::mutex> lk(g_gpu_mu);
    if (g_gpu_refs == 0 && ncnn::create_gpu_instance() != 0) return false;
    if (ncnn::get_gpu_count() <= 0) {
        if (g_gpu_refs == 0) ncnn::destroy_gpu_instance();
        return false;
    }
    ++g_gpu_refs;
    return true;
#else
    return false;
#endif
}

void release_gpu() {
#if NCNN_VULKAN
    std::lock_guard<std::mutex> lk(g_gpu_mu);
    if (g_gpu_refs > 0 && --g_gpu_refs == 0) ncnn::destroy_gpu_instance();
#endif
}

std::string jstr(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return r;
}

// Android ARGB_8888 Bitmap -> cv::Mat BGR (deep-copied out before unlock). Empty on failure.
cv::Mat bitmap_to_bgr(JNIEnv* env, jobject bitmap) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return {};
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return {};
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return {};
    cv::Mat bgr;
    try {
        cv::Mat rgba((int)info.height, (int)info.width, CV_8UC4, pixels, info.stride);
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);   // copies out of the locked buffer
    } catch (...) { /* fall through to unlock */ }
    AndroidBitmap_unlockPixels(env, bitmap);
    return bgr;
}

// Run forward_raw at `confFloor` and move the result out of the Backend into a fresh RawOutput.
// Throws on backend failure (callers catch into g_last_error).
RawOutput* forward_to_raw(Handle& h, const cv::Mat& bgr, float confFloor) {
    Config cfg = h.cfg;                  // the model's imgsz/names; only the floor differs
    cfg.conf_thresh = confFloor;
    Backend& be = *h.be;
    be.forward_raw(bgr, cfg, /*decode=*/true);
    auto r = std::make_unique<RawOutput>();
    r->candidates = std::move(be.candidates);
    r->lb = be.cand_lb;
    r->orig_w = be.cand_orig_w;
    r->orig_h = be.cand_orig_h;
    r->imgsz = cfg.imgsz;
    r->proto = std::move(be.proto);
    r->proto_c = be.proto_c; r->proto_h = be.proto_h; r->proto_w = be.proto_w;
    r->pre_ms = be.pre_ms; r->infer_ms = be.infer_ms; r->decode_ms = be.post_ms;
    // Leave the backend's cache consistently EMPTY (moved-from vectors are empty; the dims
    // must follow) so the harness-shape nativeSegOverlay reports "nothing cached" rather than
    // reading a proto that is no longer there.
    be.candidates.clear();
    be.proto.clear();
    be.proto_c = be.proto_h = be.proto_w = 0;
    return r.release();
}

// Runtime capability bits, probed once. QNN = the ORT build carries the QNN EP (arm64 AAR) AND
// libQnnHtp.so is loadable from the app's native lib dir (packaged by extractOrt). The skels
// are loaded later by the DSP through libcdsprpc.so; that failure shows up as an init note /
// zero HTP placement, not here.
int capabilities() {
    static int caps = -1;
    if (caps >= 0) return caps;
    int c = kCapNcnn;
#ifdef USE_ORT
    c |= kCapOrt;
#if defined(__aarch64__)
    void* h = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);   // kept open: the EP dlopens it again by name
    if (h) c |= kCapQnn;
    else LOGI("QNN unavailable: %s", dlerror());
#endif
#endif
    caps = c;
    LOGI("capabilities=%d (ncnn=%d ort=%d qnn=%d)", c, (c & kCapNcnn) != 0, (c & kCapOrt) != 0, (c & kCapQnn) != 0);
    return caps;
}

// "key=value;key=value" -> value for key ("" when absent). The nativeInit2 options string.
std::string opt_value(const std::string& options, const std::string& key) {
    size_t pos = 0;
    while (pos < options.size()) {
        size_t end = options.find(';', pos);
        if (end == std::string::npos) end = options.size();
        const std::string kv = options.substr(pos, end - pos);
        const size_t eq = kv.find('=');
        if (eq != std::string::npos && kv.substr(0, eq) == key) return kv.substr(eq + 1);
        pos = end + 1;
    }
    return "";
}

// Get-or-create the ARGB_8888 bitmap the mask overlay is written into. `reuse` is taken only
// when it matches exactly (dims + format); otherwise a new one is created. nullptr on failure.
jobject overlay_bitmap(JNIEnv* env, jobject reuse, int w, int h) {
    if (reuse) {
        AndroidBitmapInfo info;
        if (AndroidBitmap_getInfo(env, reuse, &info) == ANDROID_BITMAP_RESULT_SUCCESS &&
            info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 && (int)info.width == w && (int)info.height == h)
            return reuse;
    }
    if (!g_bitmap_cls || !g_bitmap_create || !g_bitmap_argb8888) return nullptr;
    jobject bmp = env->CallStaticObjectMethod(g_bitmap_cls, g_bitmap_create, (jint)w, (jint)h, g_bitmap_argb8888);
    if (env->ExceptionCheck()) { env->ExceptionClear(); return nullptr; }
    return bmp;
}

}  // namespace

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
    jclass bmp = env->FindClass("android/graphics/Bitmap");
    jclass cfg = env->FindClass("android/graphics/Bitmap$Config");
    if (!bmp || !cfg) { env->ExceptionClear(); return JNI_VERSION_1_6; }   // overlay creation degrades to an error, init still works
    jfieldID f = env->GetStaticFieldID(cfg, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    jobject argb = f ? env->GetStaticObjectField(cfg, f) : nullptr;
    jmethodID create = env->GetStaticMethodID(bmp, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    if (!argb || !create) { env->ExceptionClear(); return JNI_VERSION_1_6; }
    g_bitmap_cls = static_cast<jclass>(env->NewGlobalRef(bmp));
    g_bitmap_argb8888 = env->NewGlobalRef(argb);
    g_bitmap_create = create;
    return JNI_VERSION_1_6;
}

extern "C" {

// Shared init. `runtime` / `unit` are the Kotlin codes; `options` is "perf=<htp mode>;strict=<0|1>".
static jlong init_impl(JNIEnv* env, jstring jModelDir, int runtime, int unit, jint threads, jint precisionCode,
                       jstring jCacheDir, jstring jOptions) {
    g_last_error.clear();
    // precisionCode is the Kotlin Precision.native value (== C++ Precision): 0 auto, 1 fp32, 2 fp16, 3 int8.
    const Precision precision = (precisionCode >= 0 && precisionCode <= 3)
                                    ? static_cast<Precision>(precisionCode) : Precision::Auto;
    std::string dir = jstr(env, jModelDir);
    const std::string cacheDir = jstr(env, jCacheDir);
    const std::string options = jstr(env, jOptions);
    const int th = threads > 0 ? (int)threads : std::max(1, ncnn::get_big_cpu_count());
    std::string initNote;   // bridge-level downgrades (no GPU / no NPU), prepended to the backend's note

    auto h = std::make_unique<Handle>();
    h->precision = precision;
    h->runtime = runtime;

    if (runtime == kRuntimeOnnx) {
#ifdef USE_ORT
        // ONNX file choice: INT8 means a QDQ sibling (A16W8 preferred, A8W8 second) and a missing
        // sibling is a hard failure - an int8 number must never silently come from a float model.
        std::string onnx;
        if (precision == Precision::Int8) {
            for (const char* cand : {"/model-a16w8.onnx", "/model-a8w8.onnx"})
                if (std::ifstream(dir + cand).good()) { onnx = dir + cand; break; }
            if (onnx.empty()) {
                g_last_error = "int8 ONNX model not found: " + dir + "/model-a16w8.onnx (or -a8w8)";
                LOGE("%s", g_last_error.c_str());
                return 0;
            }
        } else {
            onnx = dir + "/model.onnx";
            if (!std::ifstream(onnx).good()) {
                g_last_error = "ONNX model not found: " + onnx;
                LOGE("%s", g_last_error.c_str());
                return 0;
            }
        }
        OrtOptions o;
        o.threads = th;
        o.log_tag = "YMOrt";
        if (unit == kUnitNpu) {
            if (capabilities() & kCapQnn) o.device = "qnn";
            else { o.device = "cpu"; initNote = "NPU requested but no QNN runtime on this device; using the CPU EP"; }
        } else if (unit == kUnitGpu) {
            o.device = "cpu"; initNote = "ONNX has no GPU unit on Android; using the CPU EP";
        } else {
            o.device = "cpu";
        }
        const std::string perf = opt_value(options, "perf");
        if (!perf.empty()) o.htp_perf = perf;
        o.strict_htp = opt_value(options, "strict") == "1";
        if (!cacheDir.empty() && o.device == "qnn") {
            // <cacheDir>/<onnx stem>_ctx.onnx: the pre-compiled HTP context (Detector owns the
            // dir + its SoC/ORT/model stamp, and wipes it on mismatch).
            const size_t slash = onnx.find_last_of('/');
            std::string stem = onnx.substr(slash == std::string::npos ? 0 : slash + 1);
            stem = stem.substr(0, stem.size() - 5);   // ".onnx"
            o.ctx_cache_path = cacheDir + "/" + stem + "_ctx.onnx";
        }
        if (o.strict_htp && o.device != "qnn") {
            g_last_error = "strict NPU requested but " + (initNote.empty() ? std::string("the NPU is not selectable") : initNote);
            LOGE("%s", g_last_error.c_str());
            return 0;
        }
        try {
            auto ort = std::make_unique<OrtBackend>(onnx, o);
            h->nodesTotal = ort->nodes_total();
            h->nodesOnCpu = ort->nodes_on_cpu();
            {   // segmentation: the ONNX export's own `task`, corroborated by the shared metadata.yaml
                std::string task;
                h->seg = ort->task().find("segment") != std::string::npos ||
                         (meta::read_ncnn_yaml_scalar(dir + "/metadata.yaml", "task", task) &&
                          task.find("segment") != std::string::npos);
            }
            h->be = std::move(ort);
        } catch (const std::exception& e) {
            g_last_error = e.what();
            LOGE("ort init failed: %s", e.what());
            return 0;
        }
#else
        g_last_error = "this build has no ONNX Runtime backend";
        LOGE("%s", g_last_error.c_str());
        return 0;
#endif
    } else {
        if (precision == Precision::Int8) dir = meta::ncnn_int8_sibling(dir);   // "<name>-int8_ncnn"
        const std::string param = dir + "/model.ncnn.param";
        const std::string bin = dir + "/model.ncnn.bin";
        if (precision == Precision::Int8 && !std::ifstream(param).good()) {
            g_last_error = "int8 model dir not found: " + dir;   // hard fail: never a silent float fallback
            LOGE("%s", g_last_error.c_str());
            return 0;
        }
        const bool useVulkan = unit == kUnitGpu;
        if (unit == kUnitNpu) initNote = "ncnn has no NPU path; using the CPU";
        bool haveVk = false;
        if (useVulkan && precision == Precision::Int8) {
            LOGI("Vulkan requested with int8: forcing CPU (no ncnn Vulkan int8 kernels)");
        } else if (useVulkan) {
            haveVk = acquire_gpu();
            if (!haveVk) LOGI("Vulkan requested but unavailable; using CPU");
        }
        try {
            auto nb = std::make_unique<NcnnBackend>(param, bin, th, haveVk, precision);
            // The backend may decline Vulkan for this model; keep the process-global GPU refcount honest.
            const bool vkActive = nb->active_ep.rfind("ncnn-Vulkan", 0) == 0;
            if (haveVk && !vkActive) { release_gpu(); haveVk = false; }
            h->vulkan = haveVk;
            {   // `task:` is a top-level scalar of the ultralytics sidecar; the mixture exports omit it (= detect)
                std::string task;
                h->seg = meta::read_ncnn_yaml_scalar(dir + "/metadata.yaml", "task", task) &&
                         task.find("segment") != std::string::npos;
            }
            h->be = std::move(nb);
        } catch (const std::exception& e) {
            g_last_error = e.what();
            LOGE("init failed: %s", e.what());
            if (haveVk) release_gpu();
            return 0;
        }
    }

    // Build the inference Config from the model's own metadata (mirrors main.cpp): the ncnn graph
    // bakes attention token counts at the training imgsz and the ONNX export is static, so it is fixed.
    Config& c = h->cfg;
    c.imgsz = h->be->fixed_imgsz > 0 ? h->be->fixed_imgsz
              : (h->be->meta_imgsz > 0 ? h->be->meta_imgsz : 640);
    c.class_names = h->be->meta_names;  // may be empty -> labels fall back to the class index
    h->activeBackend = h->be->active_ep;   // the REAL resolved precision / EP, not the requested flags
    h->note = initNote.empty() ? h->be->ep_note
              : (h->be->ep_note.empty() ? initNote : initNote + "; " + h->be->ep_note);
    LOGI("init ok: %s runtime=%s requested=%s imgsz=%d classes=%zu threads=%d placement=%d/%d%s%s",
         h->activeBackend.c_str(), h->be->runtime_name(), precision_name(precision), c.imgsz,
         c.class_names.size(), th, h->nodesTotal - h->nodesOnCpu, h->nodesTotal,
         h->note.empty() ? "" : " note=", h->note.c_str());
    return reinterpret_cast<jlong>(h.release());
}

// The original ncnn entry point: runtime = ncnn, unit = GPU when useVulkan else CPU.
JNIEXPORT jlong JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeInit(JNIEnv* env, jobject, jstring jModelDir,
                                                   jboolean useVulkan, jint threads, jint precisionCode) {
    return init_impl(env, jModelDir, kRuntimeNcnn, useVulkan ? kUnitGpu : kUnitCpu, threads, precisionCode,
                     nullptr, nullptr);
}

// runtime: 0 ncnn / 1 ONNX; unit: 0 CPU / 1 GPU / 2 NPU; cacheDir: EPContext cache dir ("" = none);
// options: "perf=burst|sustained_high_performance;strict=0|1".
JNIEXPORT jlong JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeInit2(JNIEnv* env, jobject, jstring jModelDir, jint runtime,
                                                    jint unit, jint threads, jint precisionCode,
                                                    jstring jCacheDir, jstring jOptions) {
    return init_impl(env, jModelDir, runtime == kRuntimeOnnx ? kRuntimeOnnx : kRuntimeNcnn,
                     (unit >= kUnitCpu && unit <= kUnitNpu) ? (int)unit : kUnitCpu, threads, precisionCode,
                     jCacheDir, jOptions);
}

// Capability bits: 1 ncnn, 2 ONNX Runtime, 4 QNN (Hexagon NPU) runtime present. Static.
JNIEXPORT jint JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeCapabilities(JNIEnv*, jclass) {
    return capabilities();
}

// [nodesTotal, nodesOnCpu] of the ORT graph placement at init ([0, 0] for ncnn / unknown).
JNIEXPORT jintArray JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativePlacement(JNIEnv* env, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    jint v[2] = {h ? h->nodesTotal : 0, h ? h->nodesOnCpu : 0};
    jintArray arr = env->NewIntArray(2);
    if (arr) env->SetIntArrayRegion(arr, 0, 2, v);
    return arr;
}

JNIEXPORT void JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeSetConfig(JNIEnv*, jobject, jlong handle, jfloat conf,
                                                        jfloat iou, jint maxDet) {
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h) return;
    h->cfg.conf_thresh = conf;
    h->cfg.iou_thresh = iou;
    if (maxDet > 0) h->cfg.max_det = maxDet;
}

// Returns a flat float[]: [n, then n*6 = (x1,y1,x2,y2,conf,cls) per detection]. null on error.
JNIEXPORT jfloatArray JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeInfer(JNIEnv* env, jobject, jlong handle,
                                                    jobject bitmap) {
    g_last_error.clear();
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h || !h->be) { g_last_error = "null handle"; return nullptr; }
    cv::Mat bgr = bitmap_to_bgr(env, bitmap);
    if (bgr.empty()) { g_last_error = "bitmap must be ARGB_8888 and lockable"; return nullptr; }

    std::vector<Detection> dets;
    try {
        dets = h->be->infer(bgr, h->cfg);
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }

    const int n = (int)dets.size();
    std::vector<float> buf(1 + (size_t)n * 6);
    buf[0] = (float)n;
    for (int i = 0; i < n; ++i) {
        const auto& d = dets[i];
        float* p = &buf[1 + (size_t)i * 6];
        p[0] = d.box.x;
        p[1] = d.box.y;
        p[2] = d.box.x + d.box.width;
        p[3] = d.box.y + d.box.height;
        p[4] = d.conf;
        p[5] = (float)d.class_id;
    }
    jfloatArray arr = env->NewFloatArray((jsize)buf.size());
    if (!arr) { g_last_error = "oom"; return nullptr; }
    env->SetFloatArrayRegion(arr, 0, (jsize)buf.size(), buf.data());
    return arr;
}

// Segmentation overlay built from the LAST infer's cached candidates/proto (no new forward,
// "forward once, tune cheap"). Writes [w,h] into dimsOut and returns an RGBA byte[] (w*h*4).
// null if the model is not segmentation or nothing has been inferred yet.
JNIEXPORT jbyteArray JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeSegOverlay(JNIEnv* env, jobject, jlong handle,
                                                         jintArray dimsOut) {
    g_last_error.clear();
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h || !h->be) { g_last_error = "null handle"; return nullptr; }
    auto& be = *h->be;
    if (!be.is_seg() || be.candidates.empty()) { g_last_error = "no segmentation output cached"; return nullptr; }

    cv::Mat rgba;
    try {
        std::vector<Detection> dets = nms_and_cap(be.candidates, h->cfg, be.cand_orig_w, be.cand_orig_h);
        rgba = seg_overlay(dets, be.proto, be.proto_c, be.proto_h, be.proto_w, be.cand_lb, h->cfg.imgsz,
                           be.cand_orig_w, be.cand_orig_h);
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
    if (rgba.empty()) { g_last_error = "empty overlay"; return nullptr; }

    cv::Mat cont = rgba.isContinuous() ? rgba : rgba.clone();
    const jsize len = (jsize)(cont.total() * cont.elemSize());
    jbyteArray out = env->NewByteArray(len);
    if (!out) { g_last_error = "oom"; return nullptr; }
    env->SetByteArrayRegion(out, 0, len, reinterpret_cast<const jbyte*>(cont.data));
    if (dimsOut && env->GetArrayLength(dimsOut) >= 2) {
        jint d[2] = {be.cand_orig_w, be.cand_orig_h};
        env->SetIntArrayRegion(dimsOut, 0, 2, d);
    }
    return out;
}

JNIEXPORT jstring JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeActiveBackend(JNIEnv* env, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    return env->NewStringUTF(h ? h->activeBackend.c_str() : "");
}

JNIEXPORT jstring JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeBackendNote(JNIEnv* env, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    return env->NewStringUTF(h ? h->note.c_str() : "");
}

JNIEXPORT jobjectArray JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeMetaNames(JNIEnv* env, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    static const std::vector<std::string> kEmpty;
    const std::vector<std::string>& names = h ? h->cfg.class_names : kEmpty;
    jclass strCls = env->FindClass("java/lang/String");
    jobjectArray arr = env->NewObjectArray((jsize)names.size(), strCls, nullptr);
    for (jsize i = 0; i < (jsize)names.size(); ++i) {
        jstring s = env->NewStringUTF(names[i].c_str());
        env->SetObjectArrayElement(arr, i, s);
        env->DeleteLocalRef(s);
    }
    return arr;
}

JNIEXPORT jstring JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeLastError(JNIEnv* env, jobject) {
    return env->NewStringUTF(g_last_error.c_str());
}

// Same thread-local error string, reachable from a RawOutput (static: no runtime instance).
JNIEXPORT jstring JNICALL
Java_dev_yolomaster_ncnn_RawOutput_nativeLastError(JNIEnv* env, jclass) {
    return env->NewStringUTF(g_last_error.c_str());
}

JNIEXPORT void JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeRelease(JNIEnv*, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h) return;
    const bool vk = h->vulkan;
    delete h;
    if (vk) release_gpu();
}

// ======================= app shape: RawOutput / decode / mask overlay =======================

// Forward on an ARGB_8888 Bitmap -> RawOutput handle (0 on error, see nativeLastError).
JNIEXPORT jlong JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeForwardRaw(JNIEnv* env, jobject, jlong handle,
                                                         jobject bitmap, jfloat confFloor) {
    g_last_error.clear();
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h || !h->be) { g_last_error = "null handle"; return 0; }
    cv::Mat bgr = bitmap_to_bgr(env, bitmap);
    if (bgr.empty()) { g_last_error = "bitmap must be ARGB_8888 and lockable"; return 0; }
    try {
        return reinterpret_cast<jlong>(forward_to_raw(*h, bgr, confFloor));
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    }
}

// Forward on a direct RGBA_8888 buffer (CameraX ImageAnalysis planes[0]): `rowStride` bytes per
// row, optional crop (cropW/cropH <= 0 = whole frame) and a 0/90/180/270 clockwise rotation
// applied AFTER the crop, so orig_w/orig_h - and every candidate - are in the upright frame the
// preview shows. The RGBA->BGR conversion is the single copy; the buffer is not retained, so the
// caller may close the ImageProxy as soon as this returns.
JNIEXPORT jlong JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeForwardRawRgba(JNIEnv* env, jobject, jlong handle,
                                                             jobject buffer, jint w, jint h_, jint rowStride,
                                                             jint cropX, jint cropY, jint cropW, jint cropH,
                                                             jint rotDeg, jfloat confFloor) {
    g_last_error.clear();
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h || !h->be) { g_last_error = "null handle"; return 0; }
    auto* ptr = static_cast<uint8_t*>(buffer ? env->GetDirectBufferAddress(buffer) : nullptr);
    if (!ptr) { g_last_error = "buffer must be a direct ByteBuffer"; return 0; }
    if (w <= 0 || h_ <= 0 || rowStride < w * 4) { g_last_error = "bad frame geometry"; return 0; }
    const jlong cap = env->GetDirectBufferCapacity(buffer);
    if (cap >= 0 && cap < (jlong)(h_ - 1) * rowStride + (jlong)w * 4) { g_last_error = "buffer smaller than the frame"; return 0; }
    int rot = ((rotDeg % 360) + 360) % 360;
    if (rot % 90 != 0) { g_last_error = "rotation must be a multiple of 90"; return 0; }
    try {
        cv::Mat rgba(h_, w, CV_8UC4, ptr, (size_t)rowStride);
        if (cropW > 0 && cropH > 0) {
            const cv::Rect crop = cv::Rect(cropX, cropY, cropW, cropH) & cv::Rect(0, 0, w, h_);
            if (crop.width <= 0 || crop.height <= 0) { g_last_error = "crop outside the frame"; return 0; }
            rgba = rgba(crop);
        }
        cv::Mat bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);     // the one copy, out of the camera buffer
        if (rot != 0) {                                   // separate output: a non-square transpose must not alias
            cv::Mat upright;
            cv::rotate(bgr, upright, rot == 90 ? cv::ROTATE_90_CLOCKWISE
                                    : rot == 180 ? cv::ROTATE_180 : cv::ROTATE_90_COUNTERCLOCKWISE);
            bgr = upright;
        }
        return reinterpret_cast<jlong>(forward_to_raw(*h, bgr, confFloor));
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    }
}

// Kernel-only timing (the bench path): preprocess + extractor, no decode, no NMS. Returns the
// extractor wall time in ms, or a negative value on error.
JNIEXPORT jdouble JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeInferOnly(JNIEnv* env, jobject, jlong handle, jobject bitmap) {
    g_last_error.clear();
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h || !h->be) { g_last_error = "null handle"; return -1.0; }
    cv::Mat bgr = bitmap_to_bgr(env, bitmap);
    if (bgr.empty()) { g_last_error = "bitmap must be ARGB_8888 and lockable"; return -1.0; }
    try {
        h->be->forward_raw(bgr, h->cfg, /*decode=*/false);
        return h->be->infer_ms;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -1.0;
    }
}

// [pre_ms, infer_ms, post_ms] of the last forward on this handle (post = decode, plus NMS after infer()).
JNIEXPORT jdoubleArray JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeLastTimings(JNIEnv* env, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    jdouble t[3] = {0, 0, 0};
    if (h && h->be) { t[0] = h->be->pre_ms; t[1] = h->be->infer_ms; t[2] = h->be->post_ms; }
    jdoubleArray arr = env->NewDoubleArray(3);
    if (arr) env->SetDoubleArrayRegion(arr, 0, 3, t);
    return arr;
}

JNIEXPORT jboolean JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeIsSeg(JNIEnv*, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    return (h && h->seg) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeImgsz(JNIEnv*, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    return h ? h->cfg.imgsz : 0;
}

// ncnn::set_cpu_powersave: 0 = all cores, 1 = little cores, 2 = big cores. Process-global and
// applied to the CALLING thread + the OpenMP team ncnn spawns from it, so the app calls it on
// its inference thread before init (the affinity an app process lacks by default). Returns
// ncnn's status (0 = ok). Static: it does not need a loaded model.
JNIEXPORT jint JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeSetPowersave(JNIEnv*, jclass, jint mode) {
    g_last_error.clear();
    if (mode < 0 || mode > 2) { g_last_error = "powersave mode must be 0, 1 or 2"; return -1; }
    return ncnn::set_cpu_powersave((int)mode);
}

// The 10-class palette as 30 floats [r0,g0,b0, r1,g1,b1, ...] (0..1), the same class_color()
// the CLI draw / seg_overlay / iOS Kit use, so the Kotlin overlay tints match the masks.
JNIEXPORT jfloatArray JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativePalette(JNIEnv* env, jclass) {
    float p[30];
    for (int i = 0; i < 10; ++i) { const float* c = class_color(i); p[i * 3] = c[0]; p[i * 3 + 1] = c[1]; p[i * 3 + 2] = c[2]; }
    jfloatArray arr = env->NewFloatArray(30);
    if (arr) env->SetFloatArrayRegion(arr, 0, 30, p);
    return arr;
}

// [origW, origH, preMs, inferMs, decodeMs, nCand, isSeg] for a raw handle.
JNIEXPORT jdoubleArray JNICALL
Java_dev_yolomaster_ncnn_RawOutput_nativeRawInfo(JNIEnv* env, jclass, jlong raw) {
    auto* r = reinterpret_cast<RawOutput*>(raw);
    jdouble v[7] = {0, 0, 0, 0, 0, 0, 0};
    if (r) {
        v[0] = r->orig_w; v[1] = r->orig_h; v[2] = r->pre_ms; v[3] = r->infer_ms; v[4] = r->decode_ms;
        v[5] = (double)r->candidates.size(); v[6] = r->is_seg() ? 1.0 : 0.0;
    }
    jdoubleArray arr = env->NewDoubleArray(7);
    if (arr) env->SetDoubleArrayRegion(arr, 0, 7, v);
    return arr;
}

// NMS on a cached raw: [n, then n*7 = (x1,y1,x2,y2,conf,cls,candIdx)]. Same nms_and_cap as
// infer(), so decode(conf, iou) on a raw == infer() at that conf/iou. null on error.
JNIEXPORT jfloatArray JNICALL
Java_dev_yolomaster_ncnn_RawOutput_nativeRawDecode(JNIEnv* env, jclass, jlong raw, jfloat conf,
                                                        jfloat iou, jint maxDet) {
    g_last_error.clear();
    auto* r = reinterpret_cast<RawOutput*>(raw);
    if (!r) { g_last_error = "null raw"; return nullptr; }
    std::vector<Detection> dets;
    try {
        Config cfg;
        cfg.imgsz = r->imgsz;
        cfg.conf_thresh = conf;
        cfg.iou_thresh = iou;
        if (maxDet > 0) cfg.max_det = maxDet;
        dets = nms_and_cap(r->candidates, cfg, r->orig_w, r->orig_h);
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
    const int n = (int)dets.size();
    std::vector<float> buf(1 + (size_t)n * 7);
    buf[0] = (float)n;
    for (int i = 0; i < n; ++i) {
        const auto& d = dets[i];
        float* p = &buf[1 + (size_t)i * 7];
        p[0] = d.box.x;
        p[1] = d.box.y;
        p[2] = d.box.x + d.box.width;
        p[3] = d.box.y + d.box.height;
        p[4] = d.conf;
        p[5] = (float)d.class_id;
        p[6] = (float)d.cand_index;
    }
    jfloatArray arr = env->NewFloatArray((jsize)buf.size());
    if (!arr) { g_last_error = "oom"; return nullptr; }
    env->SetFloatArrayRegion(arr, 0, (jsize)buf.size(), buf.data());
    return arr;
}

// Mask overlay for the candidates named by `candIdx` (the candIdx column of nativeRawDecode),
// rendered at orig * min(1, maxSide / max(orig)) (maxSide <= 0 = original size) into a
// PREMULTIPLIED ARGB_8888 Bitmap - Android composites premultiplied, so straight RGBA from
// seg_overlay would render the tints too bright. `reuse` (same dims, ARGB_8888, mutable) is
// written in place to avoid per-frame allocation in Live. null when the raw is not a
// segmentation output or on error (nativeLastError says which).
JNIEXPORT jobject JNICALL
Java_dev_yolomaster_ncnn_RawOutput_nativeRawMaskOverlay(JNIEnv* env, jclass, jlong raw, jintArray candIdx,
                                                             jint maxSide, jint alpha, jobject reuse) {
    g_last_error.clear();
    auto* r = reinterpret_cast<RawOutput*>(raw);
    if (!r) { g_last_error = "null raw"; return nullptr; }
    if (!r->is_seg()) { g_last_error = "not a segmentation raw"; return nullptr; }
    if (r->orig_w <= 0 || r->orig_h <= 0) { g_last_error = "empty raw"; return nullptr; }

    const int longest = std::max(r->orig_w, r->orig_h);
    const double s = (maxSide > 0 && longest > maxSide) ? (double)maxSide / longest : 1.0;
    const int out_w = std::max(1, (int)std::lround(r->orig_w * s));
    const int out_h = std::max(1, (int)std::lround(r->orig_h * s));

    cv::Mat rgba;
    try {
        // Rebuild the Detection subset exactly as nms_and_cap would have (box clipped to the frame)
        // from the candidate indices, mask_coeffs included - nothing crosses JNI but the ints.
        std::vector<Detection> dets;
        const jsize n = candIdx ? env->GetArrayLength(candIdx) : 0;
        std::vector<jint> ids((size_t)n);
        if (n > 0) env->GetIntArrayRegion(candIdx, 0, n, ids.data());
        dets.reserve((size_t)n);
        const cv::Rect2d frame(0, 0, r->orig_w, r->orig_h);
        for (jint id : ids) {
            if (id < 0 || (size_t)id >= r->candidates.size()) continue;
            const RawDet& c = r->candidates[(size_t)id];
            const cv::Rect2d b = cv::Rect2d(c.box.x, c.box.y, c.box.width, c.box.height) & frame;
            if (b.width <= 0 || b.height <= 0) continue;
            Detection d;
            d.class_id = c.cls; d.conf = c.score; d.cand_index = id;
            d.box = cv::Rect2f((float)b.x, (float)b.y, (float)b.width, (float)b.height);
            d.mask_coeffs = c.mask_coeffs;
            dets.push_back(std::move(d));
        }
        const int a = std::clamp((int)alpha, 0, 255);
        rgba = seg_overlay(dets, r->proto, r->proto_c, r->proto_h, r->proto_w, r->lb, r->imgsz,
                           r->orig_w, r->orig_h, out_w, out_h, a);
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
    if (rgba.empty()) { g_last_error = "empty overlay"; return nullptr; }

    jobject bmp = overlay_bitmap(env, reuse, out_w, out_h);
    if (!bmp) { g_last_error = "could not create the overlay bitmap"; return nullptr; }
    AndroidBitmapInfo info;
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bmp, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        AndroidBitmap_lockPixels(env, bmp, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        g_last_error = "overlay bitmap not lockable";
        return nullptr;
    }
    for (int y = 0; y < out_h; ++y) {
        const uint8_t* src = rgba.ptr<uint8_t>(y);
        uint8_t* dst = static_cast<uint8_t*>(pixels) + (size_t)y * info.stride;
        for (int x = 0; x < out_w; ++x, src += 4, dst += 4) {
            const unsigned a = src[3];
            dst[0] = (uint8_t)((src[0] * a + 127) / 255);   // premultiply: RGB * A / 255
            dst[1] = (uint8_t)((src[1] * a + 127) / 255);
            dst[2] = (uint8_t)((src[2] * a + 127) / 255);
            dst[3] = (uint8_t)a;
        }
    }
    AndroidBitmap_unlockPixels(env, bmp);
    return bmp;
}

JNIEXPORT void JNICALL
Java_dev_yolomaster_ncnn_RawOutput_nativeRawRelease(JNIEnv*, jclass, jlong raw) {
    delete reinterpret_cast<RawOutput*>(raw);
}

}  // extern "C"
