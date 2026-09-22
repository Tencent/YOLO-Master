package dev.yolomaster.ncnn

import android.graphics.Bitmap
import android.graphics.Rect
import java.nio.ByteBuffer
import kotlin.math.abs

/**
 * On-device YOLO-Master inference on ncnn or ONNX Runtime.
 *
 * Wraps the shared C++ core (the same letterbox -> forward -> decode -> NMS path as the
 * desktop runners). Detection and segmentation; ncnn on the CPU by default with an opt-in
 * Vulkan fast path, or ONNX Runtime on the CPU EP / the Hexagon NPU (QNN EP) when the device
 * has it ([hasQnn]). Not thread-safe: use one instance per thread, or serialize calls.
 *
 * CPU precision is decided PER MODEL by native code (see [Precision]): fp16-safe (dense)
 * models run fp16 on armv8.2 CPUs; the emulated-router mixture models are pinned fp32
 * because their export constants (1e-9 / 1e30) are unrepresentable in fp16 and would zero
 * the routing. Pass [Precision.INT8] to load the pre-quantized `<name>-int8_ncnn` sibling.
 * [activeBackend] always reports what actually resolved; [backendNote] says why a request
 * was downgraded.
 *
 * Usage (harness shape: one call = forward + NMS):
 * ```
 * YoloMasterNcnn().use { rt ->
 *     rt.init(modelDir, useVulkan = false)
 *     rt.setConfig(conf = 0.25f, iou = 0.45f)
 *     val dets = rt.infer(bitmap)
 * }
 * ```
 *
 * App shape (the iOS Kit contract - forward once, tune cheap):
 * ```
 * YoloMasterNcnn.setPowersave(2)                      // on the inference thread, before init
 * rt.init(modelDir, useVulkan = true, threads = 2)
 * rt.forwardRaw(bitmap).use { raw ->                  // or forwardRaw(ByteBuffer, ...) from CameraX
 *     val dets = raw.decode(conf = 0.25f, iou = 0.45f)
 *     val mask = raw.maskOverlay(dets, maxSide = 640)  // null for detection models
 * }
 * ```
 * Thread policy: the runtime is NOT thread-safe - own it from one inference thread (a
 * single-thread executor), call [setPowersave] there before [init] so the ncnn OpenMP team is
 * pinned with the caller, and pass `threads = 2` on phones with two prime cores (`threads = 0`
 * = ncnn's big-core count). [RawOutput]s outlive the runtime: decode/mask need no model handle.
 */
class YoloMasterNcnn : AutoCloseable {

    private var handle: Long = 0L
    private var names: Array<String> = emptyArray()

    val isLoaded: Boolean get() = handle != 0L

    /**
     * The precision / execution provider that actually resolved, reported by native code (never
     * the requested flag): ncnn -> "ncnn-CPU-fp32", "ncnn-CPU-fp16", "ncnn-CPU-int8+fp32",
     * "ncnn-CPU-int8+fp16", "ncnn-Vulkan" (GPU fp16), "ncnn-Vulkan-fp32" (GPU, fp16 declined by
     * the model); ONNX -> "ort-CPU-fp32|a16w8|a8w8", "ort-QNN-htp-fp16|a16w8|a8w8" (NPU, plus
     * "-mixed" when some nodes fell back to the CPU EP); "none" when not loaded.
     */
    val activeBackend: String get() = if (handle != 0L) nativeActiveBackend(handle) else "none"

    /** The runtime this instance was loaded with ([Runtime.NCNN] until [init] says otherwise). */
    var runtime: Runtime = Runtime.NCNN
        private set

    /** True when the model runs on the Hexagon NPU (the ONNX QNN EP), fully or in part. */
    val onNpu: Boolean get() = activeBackend.startsWith("ort-QNN")

    /** ONNX graph placement at init ("N/M HTP" for the HUD); 0/0 for ncnn. */
    val placement: Placement
        get() {
            if (handle == 0L) return Placement(0, 0)
            val p = nativePlacement(handle)
            return Placement(p[0], p[1])
        }

    /** Why a requested precision was downgraded (e.g. "fp32 pinned: emulated-router ..."); "" if it was honoured. */
    val backendNote: String get() = if (handle != 0L) nativeBackendNote(handle) else ""

    /** Message from the last failed native call ("" if none). */
    val lastError: String get() = nativeLastError()

    /** True for segmentation models (metadata.yaml `task: segment`; a missing key means detect). Needs no forward. */
    val isSeg: Boolean get() = handle != 0L && nativeIsSeg(handle)

    /** Class names from the model's metadata (may be empty: labels then fall back to the class index). */
    val classNames: List<String> get() = names.asList()

    /** The model's fixed input size (the ncnn graph bakes it), 0 when not loaded. */
    val imgsz: Int get() = if (handle != 0L) nativeImgsz(handle) else 0

    /** Per-stage times of the last forward on this runtime (zeros before the first one). */
    val lastTimings: Timings
        get() {
            if (handle == 0L) return Timings(0.0, 0.0, 0.0)
            val t = nativeLastTimings(handle)
            return Timings(t[0], t[1], t[2])
        }

    /**
     * Load an ncnn model directory containing `model.ncnn.param`, `model.ncnn.bin`, and
     * `metadata.yaml`. If [useVulkan] is true but no usable GPU exists, it transparently
     * falls back to the CPU choice (check [activeBackend]). [precision] selects the CPU
     * numeric policy (see [Precision]); [Precision.INT8] loads the `<name>-int8_ncnn` sibling
     * and returns false (with [lastError]) if it does not exist. Returns true on success.
     */
    fun init(
        modelDir: String,
        useVulkan: Boolean = false,
        threads: Int = 0,
        precision: Precision = Precision.AUTO,
    ): Boolean = init(modelDir, Runtime.NCNN, if (useVulkan) Unit.GPU else Unit.CPU, threads, precision)

    /**
     * Load a model directory on [runtime] / [unit]. ncnn needs `model.ncnn.param/.bin` +
     * `metadata.yaml`; ONNX needs `model.onnx` beside them (a QDQ `model-a16w8.onnx` /
     * `model-a8w8.onnx` sibling for [Precision.INT8], missing = failure, like the ncnn int8 rule).
     *
     * A [unit] the runtime cannot honour on this device (ncnn/NPU, ONNX/GPU, ONNX/NPU without
     * [hasQnn]) falls back to the CPU with a [backendNote] - except when [strictNpu] is set: then
     * the NPU must take the WHOLE graph (`session.disable_cpu_ep_fallback`) or init fails with the
     * reason in [lastError]. That is the verification switch, not a Live setting.
     *
     * [cacheDir] (ONNX/NPU) holds the pre-compiled HTP context `<stem>_ctx.onnx`: generated at the
     * first init (slow: seconds of graph finalization), opened directly afterwards. The caller
     * owns the dir and must wipe it when the SoC, ORT or the model file changes. [perfMode] is the
     * QNN `htp_performance_mode` ("burst" for Live, "sustained_high_performance" for a bench).
     * Returns true on success.
     */
    fun init(
        modelDir: String,
        runtime: Runtime,
        unit: Unit,
        threads: Int = 0,
        precision: Precision = Precision.AUTO,
        cacheDir: String = "",
        perfMode: String = "burst",
        strictNpu: Boolean = false,
    ): Boolean {
        close()
        val options = "perf=$perfMode;strict=${if (strictNpu) 1 else 0}"
        handle = nativeInit2(modelDir, runtime.native, unit.native, threads, precision.native, cacheDir, options)
        if (handle != 0L) {
            names = nativeMetaNames(handle)
            this.runtime = runtime
        }
        return handle != 0L
    }

    /**
     * Robust policy for one model. Loads the fp32 CPU reference first and uses its detection
     * count on [probe] as the yardstick; then loads the requested [precision] and keeps it only if
     * it still detects and agrees with the reference (+/-1 for fp16; for int8 a garbage band of
     * non-zero and within 35% of the count - a single image's int8 count legitimately jitters by
     * 20% (seg on the COCO probe: 15 vs 12) while the model's accuracy is certified on the full val
     * set, so this check only catches a dead int8 graph); then tries Vulkan the same way (skipped
     * for INT8, which is CPU-only). fp16 underflow (empty result) and int8 garbage are caught here
     * on unknown SoCs.
     * Ends loaded either way; INT8 with no sibling returns false.
     */
    fun initBest(
        modelDir: String,
        probe: Bitmap,
        threads: Int = 0,
        precision: Precision = Precision.AUTO,
    ): Boolean {
        if (!init(modelDir, useVulkan = false, threads = threads, precision = Precision.FP32)) return false
        val ref = infer(probe).size
        if (precision != Precision.FP32) {
            if (!init(modelDir, useVulkan = false, threads = threads, precision = precision)) {
                if (precision == Precision.INT8) return false   // no int8 sibling: hard failure
                init(modelDir, useVulkan = false, threads = threads, precision = Precision.FP32)
            } else if (activeBackend.endsWith("fp16") || activeBackend.contains("int8")) {
                val n = infer(probe).size
                val tol = if (activeBackend.contains("int8")) maxOf(2, (ref * 35) / 100) else 1
                if (n == 0 || abs(n - ref) > tol) {
                    init(modelDir, useVulkan = false, threads = threads, precision = Precision.FP32)
                }
            }
        }
        if (precision == Precision.INT8) return isLoaded
        if (!init(modelDir, useVulkan = true, threads = threads, precision = precision)) {
            return init(modelDir, useVulkan = false, threads = threads, precision = precision)
        }
        if (!activeBackend.startsWith("ncnn-Vulkan")) return true // no GPU: already on the CPU choice
        val gpu = infer(probe)
        val agrees = gpu.isNotEmpty() && abs(gpu.size - ref) <= 1
        if (!agrees) init(modelDir, useVulkan = false, threads = threads, precision = precision)
        return isLoaded
    }

    /** Tune confidence / IoU / max detections. Cheap: reuses the cached forward pass. */
    fun setConfig(conf: Float = 0.25f, iou: Float = 0.45f, maxDet: Int = 300) {
        check(handle != 0L) { "runtime not loaded" }
        nativeSetConfig(handle, conf, iou, maxDet)
    }

    /** Run detection. Throws if the runtime is not loaded or inference fails. */
    fun infer(bitmap: Bitmap): List<Detection> {
        check(handle != 0L) { "runtime not loaded" }
        val src = bitmap.ensureArgb8888()
        val flat = nativeInfer(handle, src) ?: throw RuntimeException("infer failed: $lastError")
        return decode(flat)
    }

    /**
     * Run detection and, for segmentation models, also composite the mask overlay from the
     * same forward pass. For detection models the overlay is empty.
     */
    fun inferSeg(bitmap: Bitmap): SegResult {
        val dets = infer(bitmap) // runs the forward and caches candidates+proto natively
        val dims = IntArray(2)
        val rgba = nativeSegOverlay(handle, dims) ?: return SegResult(dets, ByteArray(0), 0, 0)
        return SegResult(dets, rgba, dims[0], dims[1])
    }

    /**
     * Forward only (no NMS) on an ARGB_8888 bitmap, keeping every candidate with score >=
     * [confFloor]: the raw is then tuned with [RawOutput.decode] / [RawOutput.maskOverlay] at any
     * conf >= confFloor without another forward. The caller owns the returned raw ([RawOutput.close]).
     */
    fun forwardRaw(bitmap: Bitmap, confFloor: Float = 0.05f): RawOutput {
        check(handle != 0L) { "runtime not loaded" }
        val src = bitmap.ensureArgb8888()
        val ptr = nativeForwardRaw(handle, src, confFloor)
        if (ptr == 0L) throw RuntimeException("forwardRaw failed: $lastError")
        return RawOutput(ptr, names)
    }

    /**
     * Forward only on a DIRECT RGBA_8888 buffer - the CameraX `ImageAnalysis` frame
     * (`planes[0].buffer`, `planes[0].rowStride`, `cropRect`, `imageInfo.rotationDegrees`) - so a
     * live loop never allocates a Bitmap. [crop] (null = whole frame) is applied first, then
     * [rotationDegrees] (0/90/180/270 clockwise), so the raw's `origW/origH` and every box are in
     * the upright frame the preview shows. The buffer is copied once natively and not retained:
     * close the `ImageProxy` right after this returns.
     */
    fun forwardRaw(
        rgba: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        crop: Rect?,
        rotationDegrees: Int,
        confFloor: Float = 0.05f,
    ): RawOutput {
        check(handle != 0L) { "runtime not loaded" }
        require(rgba.isDirect) { "rgba must be a direct ByteBuffer" }
        val ptr = nativeForwardRawRgba(
            handle, rgba, width, height, rowStride,
            crop?.left ?: 0, crop?.top ?: 0, crop?.width() ?: 0, crop?.height() ?: 0,
            rotationDegrees, confFloor,
        )
        if (ptr == 0L) throw RuntimeException("forwardRaw failed: $lastError")
        return RawOutput(ptr, names)
    }

    /**
     * Kernel-only timing for benchmarks: letterbox + ncnn extractor, no decode, no NMS. Returns
     * the extractor time in ms (the iOS `inferOnly` number). Clears the harness-shape cache.
     */
    fun inferOnly(bitmap: Bitmap): Double {
        check(handle != 0L) { "runtime not loaded" }
        val ms = nativeInferOnly(handle, bitmap.ensureArgb8888())
        if (ms < 0) throw RuntimeException("inferOnly failed: $lastError")
        return ms
    }

    private fun decode(flat: FloatArray): List<Detection> {
        if (flat.isEmpty()) return emptyList()
        val n = flat[0].toInt()
        val out = ArrayList<Detection>(n)
        for (i in 0 until n) {
            val p = 1 + i * 6
            val cls = flat[p + 5].toInt()
            out += Detection(
                x1 = flat[p], y1 = flat[p + 1], x2 = flat[p + 2], y2 = flat[p + 3],
                score = flat[p + 4], classId = cls,
                label = names.getOrElse(cls) { cls.toString() },
            )
        }
        return out
    }

    override fun close() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0L
            names = emptyArray()
            runtime = Runtime.NCNN
        }
    }

    private fun Bitmap.ensureArgb8888(): Bitmap =
        if (config == Bitmap.Config.ARGB_8888) this else copy(Bitmap.Config.ARGB_8888, false)

    private external fun nativeInit(modelDir: String, useVulkan: Boolean, threads: Int, precision: Int): Long
    private external fun nativeInit2(
        modelDir: String, runtime: Int, unit: Int, threads: Int, precision: Int, cacheDir: String, options: String,
    ): Long
    private external fun nativePlacement(handle: Long): IntArray
    private external fun nativeSetConfig(handle: Long, conf: Float, iou: Float, maxDet: Int)
    private external fun nativeInfer(handle: Long, bitmap: Bitmap): FloatArray?
    private external fun nativeSegOverlay(handle: Long, dimsOut: IntArray): ByteArray?
    private external fun nativeActiveBackend(handle: Long): String
    private external fun nativeBackendNote(handle: Long): String
    private external fun nativeMetaNames(handle: Long): Array<String>
    private external fun nativeLastError(): String
    private external fun nativeRelease(handle: Long)
    private external fun nativeForwardRaw(handle: Long, bitmap: Bitmap, confFloor: Float): Long
    private external fun nativeForwardRawRgba(
        handle: Long, buffer: ByteBuffer, width: Int, height: Int, rowStride: Int,
        cropX: Int, cropY: Int, cropW: Int, cropH: Int, rotationDegrees: Int, confFloor: Float,
    ): Long
    private external fun nativeInferOnly(handle: Long, bitmap: Bitmap): Double
    private external fun nativeLastTimings(handle: Long): DoubleArray
    private external fun nativeIsSeg(handle: Long): Boolean
    private external fun nativeImgsz(handle: Long): Int

    companion object {
        init { System.loadLibrary("yolomaster_ncnn") }

        /**
         * `ncnn::set_cpu_powersave(mode)`: 0 = all cores, 1 = little cores only, 2 = big cores
         * only. Process-global; it pins the CALLING thread and the OpenMP team ncnn spawns from
         * it, so call it on the inference thread before [init] (an app process has no affinity
         * by default and ran 2-4x slower than the shell bench at >= 2 threads). Returns true when
         * ncnn accepted the mode.
         */
        fun setPowersave(mode: Int): Boolean = nativeSetPowersave(mode) == 0

        /**
         * Runtime capability bits of THIS build on THIS device, probed once: 1 = ncnn, 2 = ONNX
         * Runtime compiled in, 4 = the QNN (Hexagon NPU) runtime is loadable. The app builds its
         * runtime / unit menus from these instead of guessing from the SoC name.
         */
        val capabilities: Int by lazy { nativeCapabilities() }

        /** ONNX Runtime is part of this build (both ABIs of the current packaging). */
        val hasOrt: Boolean get() = (capabilities and 2) != 0

        /** The Hexagon NPU can be requested ([Runtime.ONNX] + [Unit.NPU]); false on x86_64 and non-Qualcomm SoCs. */
        val hasQnn: Boolean get() = (capabilities and 4) != 0

        /**
         * The 10-color class palette (index = classId % 10) as 30 floats `[r, g, b] * 10` in
         * 0..1 - the same table the native mask overlay and the CLI draw use.
         */
        val palette: FloatArray by lazy { nativePalette() }

        /** Color of [classId] as an opaque ARGB int, from [palette]. */
        fun classColor(classId: Int): Int {
            val i = ((classId % 10) + 10) % 10
            val r = (palette[i * 3] * 255f + 0.5f).toInt()
            val g = (palette[i * 3 + 1] * 255f + 0.5f).toInt()
            val b = (palette[i * 3 + 2] * 255f + 0.5f).toInt()
            return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        // Static natives (no model handle). The raw handle natives live on RawOutput itself.
        @JvmStatic private external fun nativeSetPowersave(mode: Int): Int
        @JvmStatic private external fun nativePalette(): FloatArray
        @JvmStatic private external fun nativeCapabilities(): Int
    }
}
