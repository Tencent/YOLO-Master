package dev.yolomaster.app.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import dev.yolomaster.app.model.BundledModel
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.system.Prefs
import dev.yolomaster.ncnn.Detection
import dev.yolomaster.ncnn.Placement
import dev.yolomaster.ncnn.Precision
import dev.yolomaster.ncnn.RawOutput
import dev.yolomaster.ncnn.Runtime
import dev.yolomaster.ncnn.YoloMasterNcnn
import dev.yolomaster.ncnn.Unit as NcnnUnit
import java.io.File

/**
 * The iOS `Detector` shape over the ncnn / ONNX Runtime runtime: one instance per screen, opened
 * on the inference thread, closed on suspend. `forward` returns a [RawOutput] the caller owns
 * (`decode`/`maskOverlay` run on it without the detector), which is what lets the Photo tab
 * retune conf/IoU with no re-inference.
 *
 * Thread policy (S26 finding): pin to the prime cores and use 2 threads; without affinity the
 * OpenMP team migrates and multi-threaded inference ran 2-4x slower in an app process.
 */
class Detector private constructor(
    private val rt: YoloMasterNcnn,
    val model: BundledModel,
    val runtime: Runtime,
    val compute: ComputeChoice,
    val threads: Int,
    /** First-forward latency after load (Vulkan pipeline build / HTP graph finalization), for the bench/HUD. */
    val warmupMs: Double,
) : AutoCloseable {
    val isSeg: Boolean = rt.isSeg
    val classNames: List<String> = rt.classNames
    val imgsz: Int = rt.imgsz
    val activeBackend: String get() = rt.activeBackend
    val backendNote: String get() = rt.backendNote
    /** True when the backend really runs on Vulkan (the runtime may have declined GPU). */
    val onGpu: Boolean get() = activeBackend.startsWith("ncnn-Vulkan")
    /** True when the backend really runs on the Hexagon NPU (QNN EP), fully or in part. */
    val onNpu: Boolean get() = activeBackend.startsWith("ort-QNN")
    /** ONNX graph placement ("N/M HTP" in the HUD); 0/0 for ncnn. */
    val placement: Placement get() = rt.placement

    /** Full forward on a bitmap (Photo tab, bench stage pass). */
    fun forward(bitmap: Bitmap, confFloor: Float = 0.05f): RawOutput = rt.forwardRaw(bitmap, confFloor)

    /**
     * Full forward straight from a CameraX RGBA_8888 frame: no Bitmap, one RGBA->BGR copy, the
     * rotation is applied natively so boxes come back in upright-frame pixels.
     */
    fun forward(image: ImageProxy, confFloor: Float = 0.05f): RawOutput {
        val plane = image.planes[0]
        val crop: Rect = image.cropRect
        return rt.forwardRaw(
            plane.buffer, image.width, image.height, plane.rowStride,
            crop, image.imageInfo.rotationDegrees, confFloor,
        )
    }

    /** Pure model time in ms on a bitmap (the bench headline, iOS `inferOnly`). */
    fun inferOnly(bitmap: Bitmap): Double = rt.inferOnly(bitmap)

    /** Per-stage timings of the last forward (pre / infer / post). */
    val lastTimings get() = rt.lastTimings

    override fun close() = rt.close()

    companion object {
        private const val TAG = "Detector"
        /**
         * 0 = every core, no pinning. Measured on the S26 (`ncnn_bench --powersave 0`, seg-N fp16):
         * 2 pinned prime cores 87 ms, all 8 cores 76 ms, so the spread wins by 13% on this SoC.
         */
        const val DEFAULT_THREADS = 0

        /** The ORT version the EPContext cache was compiled by (part of the cache stamp; bump with the AAR). */
        const val ORT_VERSION = "1.29.0"

        /**
         * Load [model] on [runtime] / [compute]. ncnn: GPU = Vulkan (fp16 for fp16-safe models, fp32
         * pinned for router-emulated ones); INT8 models always run on CPU. ONNX: NPU = the QNN EP
         * (falls back to the CPU EP with a [backendNote] when the device has no HTP), CPU = the CPU
         * EP; [ctx] supplies the EPContext cache dir and the Settings (perf mode, prefer-quantized).
         * Runs one gray warm-up forward so the first real frame does not pay the Vulkan pipeline
         * compile or the HTP graph finalization. Call off the main thread. Returns null (with the
         * reason in [lastError]) when the model cannot be loaded.
         */
        fun open(
            model: BundledModel, runtime: Runtime, compute: ComputeChoice, threads: Int = DEFAULT_THREADS,
            warmup: Boolean = true, ctx: Context? = null,
        ): Detector? {
            // threads == 0: every core, no pinning (lets the scheduler dodge the camera HAL's threads);
            // otherwise pin the OpenMP team to the big cores as the S26 bench showed is optimal.
            val nThreads = if (threads <= 0) java.lang.Runtime.getRuntime().availableProcessors() else threads
            val rt = YoloMasterNcnn()
            val ok = if (runtime == Runtime.ONNX) {
                // WHY no setPowersave: it pins the calling thread + ncnn's OpenMP team; ORT owns its own
                // thread pool and the HTP runs on the DSP, so the ncnn affinity would only mislead.
                val unit = if (compute == ComputeChoice.NPU) NcnnUnit.NPU else NcnnUnit.CPU
                val quant = model.onnxQuantFile?.takeIf { ctx != null && Prefs.preferQuant(ctx) && it.isFile }
                val precision = if (quant != null) Precision.INT8 else Precision.AUTO
                val cache = ctx?.let { contextCacheDir(it, model, quant ?: File(model.dir, BundledModel.ONNX)) }
                val perf = when (ctx?.let { Prefs.ortPerfMode(it) } ?: Prefs.PERF_BURST) {
                    Prefs.PERF_SUSTAINED -> "sustained_high_performance"
                    else -> "burst"
                }
                rt.init(model.dir.absolutePath, Runtime.ONNX, unit, nThreads, precision, cacheDir = cache?.absolutePath ?: "", perfMode = perf)
            } else {
                YoloMasterNcnn.setPowersave(if (threads <= 0) 0 else 2)
                val useVulkan = compute == ComputeChoice.GPU && !model.cpuOnly
                val precision = if (model.isInt8) Precision.INT8 else Precision.AUTO
                rt.init(model.dir.absolutePath, useVulkan = useVulkan, threads = nThreads, precision = precision)
            }
            if (!ok) {
                lastError = rt.lastError
                Log.w(TAG, "init failed for ${model.id} on ${runtime.label}/${compute.label}: $lastError")
                rt.close()
                return null
            }
            var warm = 0.0
            if (warmup) {
                val gray = grayProbe(rt.imgsz.takeIf { it > 0 } ?: 640)
                val t0 = SystemClock.elapsedRealtimeNanos()
                try { rt.forwardRaw(gray).close() } catch (t: Throwable) { Log.w(TAG, "warm-up failed: ${rt.lastError}") }
                warm = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                gray.recycle()
                val p = rt.placement
                Log.i(TAG, "opened ${model.id} on ${rt.activeBackend} threads=$threads warmup=${"%.1f".format(warm)} ms" +
                    (if (runtime == Runtime.ONNX) " placement=${p.onAccelerator}/${p.total}" else "") + " note='${rt.backendNote}'")
            }
            return Detector(rt, model, runtime, compute, nThreads, warm)
        }

        /**
         * `filesDir/ort_ctx/<model.id>/`: the pre-compiled HTP context (`<stem>_ctx.onnx`) the runtime
         * writes at the first NPU init and opens directly afterwards. A context binary is only valid
         * for the SoC, the ORT/QNN build and the exact graph it was compiled from, so the dir carries a
         * stamp `"<SOC_MODEL>|ort=<ver>|<onnx length:lastModified>"`; a mismatch wipes the dir before
         * init and the runtime regenerates (slow once, behind the loading card).
         */
        fun contextCacheDir(ctx: Context, model: BundledModel, onnx: File): File {
            val dir = File(ctx.filesDir, "ort_ctx/${model.id}")
            val stampFile = File(dir, ".stamp")
            val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE
            val stamp = "$soc|ort=$ORT_VERSION|${onnx.length()}:${onnx.lastModified()}"
            val current = try { stampFile.takeIf { it.isFile }?.readText() } catch (_: Throwable) { null }
            if (current != stamp) {
                if (dir.exists()) { Log.i(TAG, "ort_ctx stamp changed for ${model.id}: '$current' -> '$stamp', wiping"); dir.deleteRecursively() }
                dir.mkdirs()
                try { stampFile.writeText(stamp) } catch (t: Throwable) { Log.w(TAG, "cannot write ort_ctx stamp: ${t.message}") }
            }
            return dir
        }

        /** Thread-local like the runtime's own lastError; read right after a failed open. */
        @Volatile var lastError: String = ""
            private set

        /** The iOS bench input: a 640x640 mid-gray (0.45) image. */
        fun grayProbe(size: Int = 640): Bitmap {
            val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
            val g = (0.45f * 255).toInt()
            bmp.eraseColor(android.graphics.Color.rgb(g, g, g))
            return bmp
        }
    }
}

/** Convenience: decode a raw with the app's defaults. */
fun RawOutput.decodeDets(conf: Float, iou: Float, maxDet: Int = 300): List<Detection> = decode(conf, iou, maxDet)
