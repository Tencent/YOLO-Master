package dev.yolomaster.ncnn

import android.graphics.Bitmap
import android.util.Log

/**
 * The raw result of ONE forward pass (the iOS Kit `RawOutput` shape): the pre-NMS candidates
 * (score >= the `confFloor` passed to [YoloMasterNcnn.forwardRaw]) and, for segmentation
 * models, the mask prototypes, both kept on the native heap. "Forward once, tune cheap":
 * [decode] re-runs NMS at any conf/IoU and [maskOverlay] re-renders masks for a chosen subset,
 * neither needing the model - the runtime (and its Vulkan device) may already be closed.
 *
 * Owns native memory: a seg raw holds ~3.3 MB of proto plus up to ~1.6 MB of candidates, a
 * detection raw <= 0.4 MB. Call [close] (idempotent) when done - `use {}` in a camera loop, a
 * list closed on new run / clear / `onCleared()` in a photo screen. A raw dropped without
 * [close] is freed by the finalizer with a leak warning under the `YMNcnn` tag; do not rely on it.
 *
 * Not thread-safe for concurrent decode + close; decode/maskOverlay from different threads on a
 * live raw are fine (read-only on the native side).
 */
class RawOutput internal constructor(
    ptr: Long,
    private val names: Array<String>,
) : AutoCloseable {

    @Volatile
    private var ptr: Long = ptr

    /** Width / height of the frame the candidates are expressed in (after any crop + rotation). */
    val origW: Int
    val origH: Int
    /** Letterbox + tensor fill time, ms. */
    val preMs: Double
    /** ncnn extractor time alone, ms (== `inferOnly`). */
    val inferMs: Double
    /** Candidate decode time, ms (NMS is not included; it happens in [decode]). */
    val decodeMs: Double
    /** Number of candidates kept at the forward's confFloor. */
    val candidateCount: Int
    /** True when this raw carries mask prototypes (a segmentation model). */
    val isSeg: Boolean

    init {
        val info = nativeRawInfo(ptr)
        origW = info[0].toInt()
        origH = info[1].toInt()
        preMs = info[2]
        inferMs = info[3]
        decodeMs = info[4]
        candidateCount = info[5].toInt()
        isSeg = info[6] != 0.0
    }

    val isClosed: Boolean get() = ptr == 0L

    /**
     * Per-class NMS at [conf] / [iou], capped at [maxDet] - the same C++ `nms_and_cap` that
     * [YoloMasterNcnn.infer] runs, so the boxes are identical for equal thresholds. [conf] below
     * the forward's confFloor cannot bring back candidates that were never kept.
     */
    fun decode(conf: Float = 0.25f, iou: Float = 0.45f, maxDet: Int = 300): List<Detection> {
        val p = ptr
        check(p != 0L) { "RawOutput is closed" }
        val flat = nativeRawDecode(p, conf, iou, maxDet)
            ?: throw RuntimeException("decode failed: ${nativeLastError()}")
        val n = flat[0].toInt()
        val out = ArrayList<Detection>(n)
        for (i in 0 until n) {
            val q = 1 + i * 7
            val cls = flat[q + 5].toInt()
            out += Detection(
                x1 = flat[q], y1 = flat[q + 1], x2 = flat[q + 2], y2 = flat[q + 3],
                score = flat[q + 4], classId = cls,
                label = names.getOrElse(cls) { cls.toString() },
                candIndex = flat[q + 6].toInt(),
            )
        }
        return out
    }

    /**
     * Composite the masks of [dets] (which must come from [decode] on THIS raw: their
     * [Detection.candIndex] selects the candidate) into a premultiplied ARGB_8888 bitmap sized
     * `orig * min(1, maxSide / max(origW, origH))` ([maxSide] <= 0 = original size), class-tinted
     * with the shared palette at [alpha] (0..255) and clipped to each box - exactly the overlay
     * the CLI / iOS draw. [reuse] of the same size and config is written in place (no per-frame
     * allocation in a live loop). Returns null for detection models.
     */
    fun maskOverlay(dets: List<Detection>, maxSide: Int = 0, alpha: Int = 165, reuse: Bitmap? = null): Bitmap? {
        val p = ptr
        check(p != 0L) { "RawOutput is closed" }
        if (!isSeg) return null
        val idx = IntArray(dets.size) { dets[it].candIndex }
        val target = reuse?.takeIf { it.isMutable && it.config == Bitmap.Config.ARGB_8888 }
        return nativeRawMaskOverlay(p, idx, maxSide, alpha, target)
            ?: throw RuntimeException("maskOverlay failed: ${nativeLastError()}")
    }

    /** Free the native memory. Safe to call more than once. */
    override fun close() {
        val p = ptr
        if (p != 0L) {
            ptr = 0L
            nativeRawRelease(p)
        }
    }

    /** Leak net: frees a raw that was never closed and says so, so the leak gets fixed upstream. */
    @Suppress("ProtectedInFinal")
    protected fun finalize() {
        if (ptr != 0L) {
            Log.w("YMNcnn", "RawOutput leaked (${origW}x${origH}, seg=$isSeg): closed by the finalizer")
            close()
        }
    }

    companion object {
        // Same library as YoloMasterNcnn (idempotent); the raw handle natives are static so a
        // raw never needs a runtime instance.
        init { System.loadLibrary("yolomaster_ncnn") }

        @JvmStatic private external fun nativeRawInfo(raw: Long): DoubleArray
        @JvmStatic private external fun nativeRawDecode(raw: Long, conf: Float, iou: Float, maxDet: Int): FloatArray?
        @JvmStatic private external fun nativeRawMaskOverlay(
            raw: Long, candIdx: IntArray, maxSide: Int, alpha: Int, reuse: Bitmap?,
        ): Bitmap?
        @JvmStatic private external fun nativeRawRelease(raw: Long)
        @JvmStatic private external fun nativeLastError(): String
    }
}
