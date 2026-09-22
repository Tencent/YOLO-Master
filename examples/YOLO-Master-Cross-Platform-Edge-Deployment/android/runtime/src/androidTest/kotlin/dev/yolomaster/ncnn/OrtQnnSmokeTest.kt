package dev.yolomaster.ncnn

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import java.io.File
import kotlin.math.abs

/**
 * M0 go/no-go gate for ONNX Runtime + the QNN execution provider (Hexagon NPU): runs the ONNX
 * export of a model on the CPU EP and on the NPU, strict (the HTP must take the WHOLE graph or
 * init fails - that failure is a finding, not a test failure) and permissive (CPU fallback for the
 * nodes the HTP refuses), and logs ONE parseable line per row:
 *
 *   YM_QNN model=.. unit=CPU|NPU strict=0|1 backend=.. placement=<onHtp>/<total> first_ms=..
 *          median_ms=.. p90_ms=.. dets=.. ref_dets=.. parity=ok|off:<n> vs <ref>|n/a ctx_kb=.. note=".."
 *
 * plus `YM_CAPS caps=.. hasOrt=.. hasQnn=..` once. `first_ms` is the first infer after init: on
 * the NPU it carries the HTP graph finalization the app hides behind its loading card (the
 * EPContext cache dir is wiped per row so every NPU row measures the cold path and leaves its
 * `<model>_ctx.onnx` behind, whose size is `ctx_kb`). Timings are `infer()` (bitmap -> letterbox
 * -> forward -> decode -> NMS), median / p90 of 30 after 5 warm-ups. Go = (near-)full placement,
 * a steady median well under ncnn's ~33 ms for seg-N, dets within +/-1 of the ONNX/CPU row.
 *
 * Rows: (CPU), (NPU strict), (NPU) for EVERY staged model dir that carries a `model.onnx`
 * (`android/scripts/stage_models.sh` copies it beside the ncnn files: yolo11n, v0.1-seg-n, v0.1-n,
 * esmoe_n_visdrone today), enumerated from the test APK's assets. VisDrone-domain models get
 * `probe_visdrone.jpg`, COCO models `probe.jpg` (an aerial detector legitimately finds nothing in
 * a living room). NPU rows are SKIPPED (Assume) when the build/device has no QNN runtime, so the
 * class stays green on the x86_64 emulator. Capture with `adb logcat -s OrtQnnSmokeTest`.
 *
 * Run: ./gradlew :runtime:connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=dev.yolomaster.ncnn.OrtQnnSmokeTest
 */
@RunWith(Parameterized::class)
class OrtQnnSmokeTest(private val model: String, private val unit: Unit, private val strict: Boolean) {

    companion object {
        private const val TAG = "OrtQnnSmokeTest"
        private const val WARMUP = 5
        private const val TIMED = 30
        private val refDets = HashMap<String, Int>()   // model -> ONNX/CPU dets on the probe (reference)

        /** Staged model dirs with a `model.onnx`, stock yolo11n first (the known-good gate model). */
        private fun onnxModels(): List<String> {
            val assets = InstrumentationRegistry.getInstrumentation().targetContext.assets
            val dirs = try { assets.list("models")?.toList() ?: emptyList() } catch (_: Exception) { emptyList<String>() }
            val withOnnx = dirs.filter { d ->
                try { assets.list("models/$d")?.contains("model.onnx") == true } catch (_: Exception) { false }
            }
            return withOnnx.sortedWith(compareBy({ it != "yolo11n_ncnn" }, { it }))
        }

        @JvmStatic
        @Parameterized.Parameters(name = "{0}/{1}/strict={2}")
        fun rows(): List<Array<Any>> {
            val out = ArrayList<Array<Any>>()
            for (m in onnxModels()) {
                out += arrayOf<Any>(m, Unit.CPU, false)   // the reference row: must run first (ref_dets)
                out += arrayOf<Any>(m, Unit.NPU, true)
                out += arrayOf<Any>(m, Unit.NPU, false)
            }
            return out
        }

        @JvmStatic
        @BeforeClass
        fun logCapabilities() {
            Log.i(TAG, "YM_CAPS caps=${YoloMasterNcnn.capabilities} hasOrt=${YoloMasterNcnn.hasOrt} " +
                "hasQnn=${YoloMasterNcnn.hasQnn} abi=${Build.SUPPORTED_ABIS.firstOrNull()} soc=${socModel()}")
        }

        private fun socModel(): String =
            if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE
    }

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val id = model.removeSuffix("_ncnn")

    /** Copy the ONNX (+ metadata.yaml) of an asset model dir into filesDir (null if not staged). */
    private fun stage(name: String): String? {
        val out = File(ctx.filesDir, "models/$name")
        val present = try { ctx.assets.list("models/$name")?.toSet() ?: emptySet() } catch (_: Exception) { emptySet<String>() }
        if (!present.contains("model.onnx")) return null
        out.mkdirs()
        for (f in listOf("model.onnx", "metadata.yaml")) {
            if (!present.contains(f)) continue
            val dst = File(out, f)
            if (dst.exists() && dst.length() > 0) continue
            ctx.assets.open("models/$name/$f").use { i -> dst.outputStream().use { i.copyTo(it) } }
        }
        return out.absolutePath
    }

    /** Domain-matched probe (the LatencyBenchTest rule): VisDrone models get `probe_visdrone.jpg` when staged. */
    private fun probe(): Bitmap {
        val visdrone = model.contains("visdrone") || model.startsWith("moa") || model.startsWith("mot")
        val want = if (visdrone) "probe_visdrone.jpg" else "probe.jpg"
        val have = ctx.assets.list("")?.contains(want) == true
        return ctx.assets.open(if (have) want else "probe.jpg").use { BitmapFactory.decodeStream(it) }
            .copy(Bitmap.Config.ARGB_8888, false)
    }

    /** Per-row EPContext cache dir, wiped so every NPU row measures the cold (finalization) path. */
    private fun cacheDir(): File {
        val d = File(ctx.cacheDir, "ort_ctx/$model/strict${if (strict) 1 else 0}")
        d.deleteRecursively()
        d.mkdirs()
        return d
    }

    private fun line(backend: String, placement: String, firstMs: Double, medianMs: Double, p90Ms: Double,
                     dets: Int, parity: String, ctxKb: Long, note: String): String {
        val times = "first_ms=%.1f median_ms=%.1f p90_ms=%.1f".format(firstMs, medianMs, p90Ms)
        return "YM_QNN model=$id unit=${unit.label} strict=${if (strict) 1 else 0} backend=$backend " +
            "placement=$placement $times dets=$dets ref_dets=${refDets[model] ?: -1} parity=$parity " +
            "ctx_kb=$ctxKb note=\"$note\""
    }

    @Test
    fun row() {
        val dir = stage(model)
        assumeTrue("$model/model.onnx is not staged", dir != null)
        assumeTrue("this build has no ONNX Runtime", YoloMasterNcnn.hasOrt)
        if (unit == Unit.NPU) assumeTrue("no QNN runtime on this device/ABI", YoloMasterNcnn.hasQnn)
        val img = probe()
        val cache = cacheDir()

        YoloMasterNcnn().use { rt ->
            val ok = rt.init(
                dir!!, Runtime.ONNX, unit, threads = 2, precision = Precision.AUTO,
                cacheDir = cache.path, perfMode = "burst", strictNpu = strict,
            )
            if (!ok) {
                val reason = rt.lastError
                if (unit == Unit.NPU && strict) {
                    // The strict row failing IS the finding: the HTP refused part of the graph (the
                    // reason names the nodes) or the QNN runtime did not come up (libs / driver).
                    Log.i(TAG, line("none", "0/0", Double.NaN, Double.NaN, Double.NaN, 0, "n/a", 0, "strict=fail: $reason"))
                    return
                }
                fail("init $id on ${unit.label}: $reason")
            }
            rt.setConfig(conf = 0.25f, iou = 0.45f)

            var t = System.nanoTime()
            var dets = rt.infer(img)
            val firstMs = (System.nanoTime() - t) / 1e6
            repeat(WARMUP) { rt.infer(img) }
            val times = DoubleArray(TIMED) {
                t = System.nanoTime()
                dets = rt.infer(img)
                (System.nanoTime() - t) / 1e6
            }
            times.sort()
            val median = times[TIMED / 2]
            val p90 = times[(TIMED * 9) / 10]

            val n = dets.size
            if (unit == Unit.CPU) refDets[model] = n
            val ref = refDets[model]
            val parity = when {
                unit != Unit.NPU || ref == null -> "n/a"
                abs(n - ref) <= 1 -> "ok"
                else -> "off:$n vs $ref"
            }
            val p = rt.placement
            val ctxKb = (cache.listFiles()?.filter { it.name.endsWith("_ctx.onnx") }?.sumOf { it.length() } ?: 0L) / 1024
            Log.i(TAG, line(rt.activeBackend, "${p.onAccelerator}/${p.total}", firstMs, median, p90, n, parity, ctxKb, rt.backendNote))

            assertTrue("$id on ${unit.label} (${rt.activeBackend}) must detect something", n > 0)
            // Parity with the CPU row is recorded (parity=), not enforced: an fp16 HTP run may
            // legitimately move one borderline box across conf 0.25, and the accuracy claim is the
            // smoke-set mAP certification of the later milestone, never a single probe count.
        }
    }
}
