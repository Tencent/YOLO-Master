package dev.yolomaster.ncnn

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import java.io.File
import kotlin.math.abs

/**
 * On-device end-to-end latency bench per (model, precision, thread count) on the CPU plus one
 * Vulkan (GPU) row per float model, logging ONE parseable line per configuration:
 *
 *   YM_LAT model=.. precision=.. backend=.. abi=.. threads=.. n=.. median_ms=.. p90_ms=.. dets=.. fp32_dets=.. note=".."
 *
 * Vulkan rows (`useVulkan=true`, `threads=gpu`, `backend=ncnn-Vulkan[-fp32]`) carry one extra field,
 * `first_ms=`: the first inference after init, which includes the ncnn pipeline / shader compile
 * that the warmup otherwise absorbs (the app's "Loading the model to GPU" card covers it). They are
 * SKIPPED on a device without a usable Vulkan GPU (the runtime falls back to the CPU choice).
 *
 * Capture with `adb logcat -s ParityTest | grep YM_LAT`. This is the app-level number (bitmap ->
 * letterbox -> ncnn -> decode -> NMS); `cpp/tools/ncnn_bench` gives the kernel-only number.
 * Rows whose assets are not staged are SKIPPED (Assume), so the matrix stays green when only the
 * default models are present. Latency is informational: the only assertions are that every mode
 * detects, INT8 rows actually run int8, INT8 is not garbage (non-zero and within 35% of the
 * fp32 detection count; a single image's int8 count jitters by up to ~20% on a certified model, so
 * the accuracy claim is the full-val certification in results/int8_cert/, never this count), and a
 * Vulkan row on a dense model agrees with the CPU fp32 detection count to within +/-1.
 *
 * Run: ./gradlew :runtime:connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=dev.yolomaster.ncnn.LatencyBenchTest
 */
@RunWith(Parameterized::class)
class LatencyBenchTest(private val model: String, private val precision: Precision, private val useVulkan: Boolean) {

    companion object {
        private const val TAG = "ParityTest"
        private val fp32Counts = HashMap<String, Int>()   // model -> fp32 dets on the probe (reference)

        @JvmStatic
        @Parameterized.Parameters(name = "{0}/{1}/vulkan={2}")
        fun rows(): List<Array<Any>> {
            val dense = listOf("v0.1-seg-n_ncnn", "esmoe_n_visdrone_ncnn", "p03_v01n_ncnn")
            val out = ArrayList<Array<Any>>()
            for (m in dense) for (p in listOf(Precision.FP32, Precision.AUTO, Precision.INT8)) out += arrayOf<Any>(m, p, false)
            out += arrayOf<Any>("moa-n_ncnn", Precision.AUTO, false)   // mixture: pinned fp32, no int8 sibling
            // Vulkan (GPU) rows: AUTO resolves to ncnn-Vulkan (fp16 shaders) on the dense models and to
            // ncnn-Vulkan-fp32 on the router-emulated mixture; one row each, threads are irrelevant.
            for (m in dense) out += arrayOf<Any>(m, Precision.AUTO, true)
            out += arrayOf<Any>("moa-n_ncnn", Precision.AUTO, true)
            return out
        }
    }

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val arm64 = Build.SUPPORTED_ABIS.firstOrNull() == "arm64-v8a"

    /** Copy an asset model dir into filesDir (false if the asset is absent). */
    private fun stage(name: String): String? {
        val out = File(ctx.filesDir, "models/$name")
        if (File(out, "model.ncnn.param").exists()) return out.absolutePath
        val files = listOf("model.ncnn.param", "model.ncnn.bin", "metadata.yaml")
        val present = try { ctx.assets.list("models/$name")?.toSet() ?: emptySet() } catch (_: Exception) { emptySet<String>() }
        if (!present.containsAll(files)) return null
        out.mkdirs()
        for (f in files) ctx.assets.open("models/$name/$f").use { i -> File(out, f).outputStream().use { i.copyTo(it) } }
        return out.absolutePath
    }

    /**
     * Domain-matched probe: VisDrone-trained models (esmoe_n_visdrone, the mixture models) get
     * `probe_visdrone.jpg`, COCO models get `probe.jpg`. A COCO living-room scene legitimately yields
     * zero detections from an aerial detector at conf 0.25, which is not a runtime failure. Falls
     * back to `probe.jpg` when the VisDrone asset is not staged.
     */
    private fun probe(model: String): Bitmap {
        val visdrone = model.contains("visdrone") || model.startsWith("moa") || model.startsWith("mot") || model.startsWith("molora")
        val name = if (visdrone && (ctx.assets.list("")?.contains("probe_visdrone.jpg") == true)) "probe_visdrone.jpg" else "probe.jpg"
        return ctx.assets.open(name).use { BitmapFactory.decodeStream(it) }.copy(Bitmap.Config.ARGB_8888, false)
    }

    private fun fp32Reference(dir: String, img: Bitmap): Int = fp32Counts.getOrPut(model) {
        YoloMasterNcnn().use { rt ->
            assertTrue("fp32 reference init: ${rt.lastError}", rt.init(dir, useVulkan = false, precision = Precision.FP32))
            rt.setConfig(conf = 0.25f, iou = 0.45f)
            rt.infer(img).size
        }
    }

    @Test
    fun bench() {
        val dir = stage(model)
        assumeTrue("model $model not staged", dir != null)
        if (precision == Precision.INT8) {
            // the runtime resolves INT8 to the "-int8_ncnn" sibling next to the float dir in filesDir
            assumeTrue("int8 sibling for $model not staged", stage(model.removeSuffix("_ncnn") + "-int8_ncnn") != null)
        }
        val img = probe(model)
        val ref = fp32Reference(dir!!, img)
        // x86_64 emulator: keep it short (fp16 flags are inert there anyway); arm64: the real sweep.
        val n = if (arm64) 50 else 5
        val warm = if (arm64) 10 else 2
        if (useVulkan) { benchVulkan(dir, img, ref, n, warm); return }
        val threadCounts = if (arm64) listOf(1, 2, 4, 0) else listOf(1, 0)   // 0 = runtime default (big cores)

        for (t in threadCounts) {
            YoloMasterNcnn().use { rt ->
                assertTrue("init $model/$precision threads=$t: ${rt.lastError}",
                           rt.init(dir, useVulkan = false, threads = t, precision = precision))
                rt.setConfig(conf = 0.25f, iou = 0.45f)
                repeat(warm) { rt.infer(img) }
                val times = DoubleArray(n)
                var dets = 0
                for (i in 0 until n) {
                    val t0 = System.nanoTime()
                    dets = rt.infer(img).size
                    times[i] = (System.nanoTime() - t0) / 1e6
                }
                times.sort()
                val median = times[n / 2]
                val p90 = times[minOf(n - 1, (n * 0.9).toInt())]
                val threadsLabel = if (t == 0) "big" else t.toString()
                Log.i(TAG, "YM_LAT model=$model precision=$precision backend=${rt.activeBackend} " +
                           "abi=${Build.SUPPORTED_ABIS.firstOrNull()} threads=$threadsLabel n=$n " +
                           "median_ms=%.2f p90_ms=%.2f dets=$dets fp32_dets=$ref note=\"${rt.backendNote}\"".format(median, p90))
                assertTrue("$model/$precision must detect (threads=$t)", dets > 0)
                when (precision) {
                    Precision.FP32 -> assertTrue(rt.activeBackend == "ncnn-CPU-fp32")
                    Precision.INT8 -> {
                        assertTrue("INT8 row must run int8: ${rt.activeBackend}", rt.activeBackend.startsWith("ncnn-CPU-int8+"))
                        val tol = maxOf(2, (ref * 35) / 100)
                        assertTrue("INT8 dets $dets vs fp32 $ref (tol $tol)", abs(dets - ref) <= tol)
                    }
                    else -> if (model.startsWith("moa-n")) assertTrue(rt.activeBackend == "ncnn-CPU-fp32")
                }
            }
        }
    }

    /**
     * One Vulkan row: `threads=gpu`, `backend=` the resolved `activeBackend` (`ncnn-Vulkan` or
     * `ncnn-Vulkan-fp32`), `first_ms=` the first inference after init (pipeline / shader compile).
     * Skipped (Assume) when the runtime resolved to a CPU backend, i.e. the device has no usable
     * Vulkan GPU. Dense models must agree with the CPU fp32 count to within +/-1 (fp16 shaders on a
     * certified fp16-safe model); the router-emulated mixture must have been pinned to fp32 shaders.
     */
    private fun benchVulkan(dir: String, img: Bitmap, ref: Int, n: Int, warm: Int) {
        YoloMasterNcnn().use { rt ->
            assertTrue("init $model/$precision vulkan: ${rt.lastError}",
                       rt.init(dir, useVulkan = true, threads = 0, precision = precision))
            assumeTrue("no Vulkan GPU on this device (resolved ${rt.activeBackend}): ${rt.backendNote}",
                       rt.activeBackend.startsWith("ncnn-Vulkan"))
            rt.setConfig(conf = 0.25f, iou = 0.45f)
            val tFirst = System.nanoTime()
            rt.infer(img)
            val firstMs = (System.nanoTime() - tFirst) / 1e6
            repeat(warm) { rt.infer(img) }
            val times = DoubleArray(n)
            var dets = 0
            for (i in 0 until n) {
                val t0 = System.nanoTime()
                dets = rt.infer(img).size
                times[i] = (System.nanoTime() - t0) / 1e6
            }
            times.sort()
            val median = times[n / 2]
            val p90 = times[minOf(n - 1, (n * 0.9).toInt())]
            Log.i(TAG, "YM_LAT model=$model precision=$precision backend=${rt.activeBackend} " +
                       "abi=${Build.SUPPORTED_ABIS.firstOrNull()} threads=gpu n=$n " +
                       "median_ms=%.2f p90_ms=%.2f dets=$dets fp32_dets=$ref first_ms=%.2f note=\"${rt.backendNote}\"".format(median, p90, firstMs))
            assertTrue("$model/$precision vulkan must detect", dets > 0)
            if (model.startsWith("moa-n")) {
                assertTrue("mixture must run fp32 shaders on Vulkan: ${rt.activeBackend}", rt.activeBackend == "ncnn-Vulkan-fp32")
            } else {
                assertTrue("Vulkan dets $dets vs CPU fp32 $ref (tol 1)", abs(dets - ref) <= 1)
            }
        }
    }
}
