package dev.yolomaster.ncnn

/**
 * Numeric precision policy for the ncnn CPU path. Mirrors `Precision` in the C++ core; the
 * integer codes are the JNI ABI and must stay stable.
 *
 *  - [AUTO]: fp16 when the model is fp16-safe (the dense models) on an armv8.2 CPU; fp32 pinned
 *    for the emulated-router mixture models, whose export constants (1e-9 / 1e30) are
 *    unrepresentable in fp16 and would zero the routing. The decision is read from the model
 *    itself (its `.param` fingerprint), not from the caller.
 *  - [FP32] / [FP16]: explicit. A request the model cannot honour is downgraded and explained in
 *    [YoloMasterNcnn.backendNote]; it is never run as a silent zero-detection model.
 *  - [INT8]: load the pre-quantized sibling directory `<name>-int8_ncnn` (mixed per-layer int8
 *    from scripts/quantize_ncnn_int8.py; the sensitive layers stay float). CPU only. Loading fails
 *    if the sibling is absent - an int8 number must never silently come from a float model.
 */
enum class Precision(val native: Int) { AUTO(0), FP32(1), FP16(2), INT8(3) }

/**
 * Inference runtime. [NCNN] loads `model.ncnn.param/.bin`; [ONNX] loads `model.onnx` (or the
 * `model-a16w8.onnx` / `model-a8w8.onnx` QDQ sibling for [Precision.INT8]) through ONNX Runtime,
 * whose QNN execution provider is the only road to the Hexagon NPU on Snapdragon phones.
 * The integer codes are the JNI ABI and must stay stable; [label] is the UI spelling.
 */
enum class Runtime(val native: Int, val label: String) { NCNN(0, "ncnn"), ONNX(1, "ONNX") }

/**
 * Compute unit. What each runtime can honour: ncnn -> [CPU] | [GPU] (Vulkan), ONNX -> [CPU] |
 * [NPU] (QNN / HTP, when [YoloMasterNcnn.hasQnn]). A unit the runtime cannot serve is downgraded
 * to [CPU] with a [YoloMasterNcnn.backendNote], never a failure - the strict NPU switch of
 * [YoloMasterNcnn.init] is the one exception, by design. Codes are the JNI ABI.
 *
 * Note: this shadows `kotlin.Unit` inside this package; refer to `kotlin.Unit` explicitly there.
 */
enum class Unit(val native: Int, val label: String) { CPU(0, "CPU"), GPU(1, "GPU"), NPU(2, "NPU") }

/**
 * ONNX Runtime graph placement at init: [total] nodes after graph optimization, [onCpu] of them
 * assigned to the CPU execution provider instead of the requested accelerator (0/0 for ncnn or
 * when ORT reported nothing parseable). [onAccelerator] is the HUD number ("N/M HTP").
 */
data class Placement(val total: Int, val onCpu: Int) {
    val onAccelerator: Int get() = total - onCpu
    val isFull: Boolean get() = total > 0 && onCpu == 0
}

/**
 * One detection in original-image pixel coordinates. [candIndex] is the index into the
 * [RawOutput] candidate pool this detection was selected from (-1 for detections that did not
 * come from [RawOutput.decode]); [RawOutput.maskOverlay] uses it to render masks for exactly
 * the chosen subset without any mask data crossing JNI.
 */
data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
    val classId: Int,
    val label: String,
    val candIndex: Int = -1,
)

/**
 * Per-stage wall times of the last forward on a runtime, in ms: [preMs] = letterbox + tensor
 * fill, [inferMs] = the ncnn extractor alone (the iOS `inferOnly` number), [postMs] = candidate
 * decode (plus NMS after [YoloMasterNcnn.infer]; decode only after [YoloMasterNcnn.forwardRaw]).
 */
data class Timings(val preMs: Double, val inferMs: Double, val postMs: Double)

/**
 * Segmentation result: the detections plus a composited RGBA overlay the size of the
 * original image ([maskWidth] x [maskHeight], 4 bytes/pixel, RGBA order). [maskRgba] is
 * empty for non-segmentation models.
 */
data class SegResult(
    val detections: List<Detection>,
    val maskRgba: ByteArray,
    val maskWidth: Int,
    val maskHeight: Int,
) {
    override fun equals(other: Any?): Boolean =
        this === other || (other is SegResult &&
            detections == other.detections && maskWidth == other.maskWidth &&
            maskHeight == other.maskHeight && maskRgba.contentEquals(other.maskRgba))

    override fun hashCode(): Int =
        (detections.hashCode() * 31 + maskWidth) * 31 + maskHeight
}
