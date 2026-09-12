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
import java.util.Locale

/**
 * M3 mAP certification dump: runs every staged model that carries a `model.onnx` over the smoke
 * image set at ultralytics val settings (conf 0.001 / IoU 0.7 / maxDet 300) on
 *
 *   (ncnn, CPU, AUTO)        the shipped reference (what the app runs today)
 *   (ONNX, CPU, AUTO)        the float graph on ORT's CPU EP (== the desktop reference)
 *   (ONNX, NPU, AUTO)        the float graph on the Hexagon HTP as fp16
 *   (ONNX, NPU, INT8)        the QDQ sibling (`model-a16w8.onnx`, else `model-a8w8.onnx`) on the HTP,
 *                            only when one is staged
 *
 * and writes one `<stem>.txt` per image with `class conf x1 y1 x2 y2` rows in ORIGINAL image
 * pixels ([Detection] already is) under
 * `getExternalFilesDir("dump")/<modelId>/<runtime>-<unit>-<precision>/`, i.e. exactly what
 * `scripts/eval_map.py` scores. Pull the dir and score it on the pod with
 * `scripts/score_device_dumps.sh` (gate: mAP50-95 within 1.0 pt of the ncnn/ONNX CPU rows, as
 * `tests/certify_ncnn_int8.py`). One parseable line per row:
 *
 *   YM_DUMP model=.. runtime=.. unit=.. precision=.. backend=.. placement=<onHtp>/<total>
 *           images=.. dets=.. median_ms=.. p90_ms=.. dir=..
 *
 * Images come from `getExternalFilesDir("smoke")` (`/sdcard/Android/data/dev.yolomaster.ncnn.test/
 * files/smoke/` jpgs, pushed by `results/int8_bench/device/run_s26_dump.sh`); VisDrone-domain
 * models (`esmoe`, `moa`, `mot`) read `smoke_visdrone/` instead, so the aerial detector is not
 * scored on living rooms. A row whose image dir is empty or missing is SKIPPED (Assume), as is an
 * NPU row on a build/device without QNN and an INT8 row without a staged QDQ sibling: the class
 * stays green on the emulator and on a phone that never got the push.
 *
 * Latency here is `infer()` (bitmap -> letterbox -> forward -> decode -> NMS at conf 0.001, which
 * costs more NMS than the app's 0.25) and is reported for context only; the speed numbers are
 * LatencyBenchTest / OrtQnnSmokeTest. Capture with `adb logcat -s OrtDumpTest`.
 *
 * Run: ./gradlew :runtime:connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=dev.yolomaster.ncnn.OrtDumpTest
 */
@RunWith(Parameterized::class)
class OrtDumpTest(
    private val model: String,
    private val runtime: Runtime,
    private val unit: Unit,
    private val precision: Precision,
) {

    companion object {
        private const val TAG = "OrtDumpTest"
        private const val THREADS = 2
        private const val CONF = 0.001f
        private const val IOU = 0.7f
        private const val MAX_DET = 300
        private val QUANT_FILES = listOf("model-a16w8.onnx", "model-a8w8.onnx")

        private fun assetList(path: String): Set<String> {
            val assets = InstrumentationRegistry.getInstrumentation().targetContext.assets
            return try { assets.list(path)?.toSet() ?: emptySet() } catch (_: Exception) { emptySet() }
        }

        /** Staged model dirs with a `model.onnx`, yolo11n first (the known-good gate model). */
        private fun onnxModels(): List<String> =
            assetList("models").filter { "model.onnx" in assetList("models/$it") }
                .sortedWith(compareBy({ it != "yolo11n_ncnn" }, { it }))

        @JvmStatic
        @Parameterized.Parameters(name = "{0}/{1}/{2}/{3}")
        fun rows(): List<Array<Any>> {
            val out = ArrayList<Array<Any>>()
            for (m in onnxModels()) {
                val files = assetList("models/$m")
                out += arrayOf<Any>(m, Runtime.NCNN, Unit.CPU, Precision.AUTO)
                out += arrayOf<Any>(m, Runtime.ONNX, Unit.CPU, Precision.AUTO)
                out += arrayOf<Any>(m, Runtime.ONNX, Unit.NPU, Precision.AUTO)
                if (QUANT_FILES.any { it in files }) out += arrayOf<Any>(m, Runtime.ONNX, Unit.NPU, Precision.INT8)
            }
            return out
        }

        @JvmStatic
        @BeforeClass
        fun logCapabilities() {
            val ctx = InstrumentationRegistry.getInstrumentation().targetContext
            Log.i(TAG, "YM_DUMP_CAPS caps=${YoloMasterNcnn.capabilities} hasOrt=${YoloMasterNcnn.hasOrt} " +
                "hasQnn=${YoloMasterNcnn.hasQnn} abi=${Build.SUPPORTED_ABIS.firstOrNull()} " +
                "smoke=${ctx.getExternalFilesDir("smoke")?.absolutePath} " +
                "smoke_visdrone=${ctx.getExternalFilesDir("smoke_visdrone")?.absolutePath} " +
                "dump=${ctx.getExternalFilesDir("dump")?.absolutePath}")
        }
    }

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val id = model.removeSuffix("_ncnn")
    private val visdrone = model.contains("visdrone") || model.startsWith("moa") || model.startsWith("mot")
    private val rowName = "${runtime.label}-${unit.label}-${precision.name}"

    /** Copy the asset model dir into filesDir (ncnn pair + metadata + every ONNX sibling present). */
    private fun stage(name: String): String {
        val out = File(ctx.filesDir, "models/$name")
        out.mkdirs()
        val present = assetList("models/$name")
        for (f in listOf("model.ncnn.param", "model.ncnn.bin", "metadata.yaml", "model.onnx") + QUANT_FILES) {
            if (f !in present) continue
            val dst = File(out, f)
            if (dst.exists() && dst.length() > 0) continue
            ctx.assets.open("models/$name/$f").use { i -> dst.outputStream().use { i.copyTo(it) } }
        }
        return out.absolutePath
    }

    /** Sorted jpgs of the smoke dir for this model's domain (empty when nothing was pushed). */
    private fun smokeImages(): List<File> {
        val dir = ctx.getExternalFilesDir(if (visdrone) "smoke_visdrone" else "smoke") ?: return emptyList()
        return dir.listFiles { f -> f.isFile && f.name.lowercase().endsWith(".jpg") }
            ?.sortedBy { it.name } ?: emptyList()
    }

    /** Per-row EPContext cache dir (kept across runs: the second run of a row opens the compiled context). */
    private fun cacheDir(): File = File(ctx.cacheDir, "ort_ctx_dump/$model/$rowName").apply { mkdirs() }

    private fun decode(f: File): Bitmap =
        BitmapFactory.decodeFile(f.absolutePath)?.copy(Bitmap.Config.ARGB_8888, false)
            ?: throw IllegalStateException("cannot decode ${f.name}")

    @Test
    fun row() {
        val images = smokeImages()
        assumeTrue("no smoke images for $id (push them, see run_s26_dump.sh)", images.isNotEmpty())
        if (runtime == Runtime.ONNX) assumeTrue("this build has no ONNX Runtime", YoloMasterNcnn.hasOrt)
        if (runtime == Runtime.ONNX && unit == Unit.NPU) assumeTrue("no QNN runtime on this device/ABI", YoloMasterNcnn.hasQnn)
        val dir = stage(model)
        if (precision == Precision.INT8) {
            assumeTrue("no QDQ sibling staged for $id", QUANT_FILES.any { File(dir, it).length() > 0 })
        }
        val outDir = File(ctx.getExternalFilesDir("dump"), "$id/$rowName")
        outDir.deleteRecursively()
        outDir.mkdirs()

        YoloMasterNcnn().use { rt ->
            val ok = rt.init(
                dir, runtime, unit, threads = THREADS, precision = precision,
                cacheDir = cacheDir().path, perfMode = "burst",
            )
            if (!ok) fail("init $id on $rowName: ${rt.lastError}")
            rt.setConfig(conf = CONF, iou = IOU, maxDet = MAX_DET)
            if (runtime == Runtime.ONNX && unit == Unit.NPU) {
                // A "NPU" row that silently fell back to the CPU would certify the wrong thing.
                assertTrue("$id/$rowName landed on ${rt.activeBackend}: ${rt.backendNote}",
                    rt.activeBackend.contains("QNN", ignoreCase = true))
            }
            if (precision == Precision.INT8) {
                assertTrue("$id/$rowName is not a QDQ graph: ${rt.activeBackend}",
                    rt.activeBackend.contains("a16w8") || rt.activeBackend.contains("a8w8"))
            }

            rt.infer(decode(images[0]))   // warm-up: HTP finalization / first-run allocations
            val times = DoubleArray(images.size)
            var total = 0
            for ((k, f) in images.withIndex()) {
                val bmp = decode(f)
                val t = System.nanoTime()
                val dets = rt.infer(bmp)
                times[k] = (System.nanoTime() - t) / 1e6
                bmp.recycle()
                total += dets.size
                File(outDir, f.nameWithoutExtension + ".txt").bufferedWriter().use { w ->
                    for (d in dets) {
                        w.write(String.format(Locale.US, "%d %.6f %.3f %.3f %.3f %.3f\n",
                            d.classId, d.score, d.x1, d.y1, d.x2, d.y2))
                    }
                }
            }
            times.sort()
            val p = rt.placement
            Log.i(TAG, "YM_DUMP model=$id runtime=${runtime.label} unit=${unit.label} precision=${precision.name} " +
                "backend=${rt.activeBackend} placement=${p.onAccelerator}/${p.total} images=${images.size} " +
                "dets=$total " + String.format(Locale.US, "median_ms=%.1f p90_ms=%.1f ", times[times.size / 2], times[(times.size * 9) / 10]) +
                "dir=${outDir.absolutePath} " +
                "note=\"${rt.backendNote}\"")
            assertTrue("$id/$rowName wrote no detections over ${images.size} images", total > 0)
        }
    }
}
