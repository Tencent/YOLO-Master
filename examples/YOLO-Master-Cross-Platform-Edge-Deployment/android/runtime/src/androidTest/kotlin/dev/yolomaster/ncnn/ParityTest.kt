package dev.yolomaster.ncnn

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

/**
 * On-device runtime harness. Proves the ncnn backend before any app UI exists.
 *
 * Prerequisites (run `android/scripts/stage_models.sh` first, then rebuild):
 *   - assets/models/v0.1-seg-n_ncnn/{model.ncnn.param,model.ncnn.bin,metadata.yaml}
 *   - assets/models/moa-n_ncnn/{...}
 *   - assets/probe.jpg   (any scene with detectable objects)
 *
 * Run on a connected arm64 device:  ./gradlew :runtime:connectedAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class ParityTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val tag = "ParityTest"

    private fun stage(model: String): String {
        val out = File(ctx.filesDir, "models/$model")
        if (File(out, "model.ncnn.param").exists()) return out.absolutePath
        out.mkdirs()
        for (f in listOf("model.ncnn.param", "model.ncnn.bin", "metadata.yaml")) {
            ctx.assets.open("models/$model/$f").use { input ->
                File(out, f).outputStream().use { input.copyTo(it) }
            }
        }
        return out.absolutePath
    }

    /** `probe.jpg` (COCO scene) by default; VisDrone-domain models pass `probe_visdrone.jpg` (falls back if absent). */
    private fun probe(name: String = "probe.jpg"): Bitmap {
        val have = ctx.assets.list("")?.contains(name) == true
        return ctx.assets.open(if (have) name else "probe.jpg").use { BitmapFactory.decodeStream(it) }
            .copy(Bitmap.Config.ARGB_8888, false)
    }

    /** ncnn's fp16 kernels are the armv8.2 path; on the x86_64 emulator AUTO honestly resolves to fp32. */
    private val arm64 = Build.SUPPORTED_ABIS.firstOrNull() == "arm64-v8a"
    private fun expectedAutoCpu() = if (arm64) "ncnn-CPU-fp16" else "ncnn-CPU-fp32"

    /**
     * Default seg-N model: explicit FP32 reports CPU-fp32 and detects (the original fp16-underflow
     * regression guard, now guarding the explicit fp32 path), and AUTO resolves to fp16 on arm64
     * (fp32 on x86_64) and still detects - the new fp16 guard for a dense model.
     */
    @Test
    fun default_model_detects_on_cpu_fp32_and_auto() {
        val img = probe()
        val dir = stage("v0.1-seg-n_ncnn")
        val fp32Count = YoloMasterNcnn().use { rt ->
            assertTrue("init CPU: ${rt.lastError}", rt.init(dir, useVulkan = false, precision = Precision.FP32))
            assertEquals("ncnn-CPU-fp32", rt.activeBackend)
            rt.setConfig(conf = 0.25f, iou = 0.45f)
            val t0 = System.nanoTime()
            val dets = rt.infer(img)
            val ms = (System.nanoTime() - t0) / 1e6
            Log.i(tag, "v0.1-seg-N cpu-fp32: ${dets.size} dets in %.1f ms".format(ms))
            assertTrue("CPU-fp32 must produce detections (fp16 regression?)", dets.isNotEmpty())
            dets.size
        }
        YoloMasterNcnn().use { rt ->
            assertTrue("init AUTO: ${rt.lastError}", rt.init(dir, useVulkan = false, precision = Precision.AUTO))
            assertEquals("AUTO on a dense model", expectedAutoCpu(), rt.activeBackend)
            assertEquals("AUTO must not need a downgrade note on a dense model", "", rt.backendNote)
            rt.setConfig(conf = 0.25f, iou = 0.45f)
            val t0 = System.nanoTime()
            val dets = rt.infer(img)
            val ms = (System.nanoTime() - t0) / 1e6
            Log.i(tag, "v0.1-seg-N ${rt.activeBackend}: ${dets.size} dets in %.1f ms (fp32: $fp32Count)".format(ms))
            assertTrue("AUTO/fp16 must produce detections (fp16 underflow?)", dets.isNotEmpty())
            assertTrue("AUTO/fp16 diverged from fp32 count", abs(dets.size - fp32Count) <= 1)
        }
    }

    /**
     * An emulated-router MoE model on ARM: AUTO must pin fp32 by fingerprint (with a note), and an
     * explicit FP16 request must be refused (still fp32, note set) rather than silently zero-det.
     */
    @Test
    fun mixture_router_runs_on_arm_pinned_fp32() {
        val img = probe("probe_visdrone.jpg")   // moa-n is a VisDrone-domain model
        val dir = stage("moa-n_ncnn")
        YoloMasterNcnn().use { rt ->
            assertTrue("init moa-n: ${rt.lastError}", rt.init(dir, useVulkan = false))
            assertEquals("mixture must be pinned fp32", "ncnn-CPU-fp32", rt.activeBackend)
            assertTrue("pin must be explained: '${rt.backendNote}'", rt.backendNote.contains("fp32 pinned"))
            val dets = rt.infer(img)
            Log.i(tag, "moa-n ${rt.activeBackend}: ${dets.size} dets  note=${rt.backendNote}")
            assertTrue("emulated MoE router must detect on ARM", dets.isNotEmpty())
        }
        YoloMasterNcnn().use { rt ->
            assertTrue(rt.init(dir, useVulkan = false, precision = Precision.FP16))
            assertEquals("FP16 on a mixture must be refused", "ncnn-CPU-fp32", rt.activeBackend)
            assertTrue("refusal must be explained", rt.backendNote.contains("refused"))
            assertTrue("refused-fp16 path must still detect", rt.infer(img).isNotEmpty())
        }
    }

    /** Vulkan either agrees with CPU (within +/-1 detection) or cleanly falls back to the CPU choice. */
    @Test
    fun vulkan_matches_cpu_or_falls_back() {
        val img = probe()
        val dir = stage("v0.1-seg-n_ncnn")
        val cpuCount = YoloMasterNcnn().use { rt ->
            assertTrue(rt.init(dir, useVulkan = false)); rt.infer(img).size
        }
        YoloMasterNcnn().use { rt ->
            assertTrue(rt.init(dir, useVulkan = true))
            if (rt.activeBackend.startsWith("ncnn-Vulkan")) {
                val gpu = rt.infer(img)
                Log.i(tag, "${rt.activeBackend}: ${gpu.size} vs cpu: $cpuCount")
                assertTrue("Vulkan diverged from CPU", gpu.isNotEmpty() && abs(gpu.size - cpuCount) <= 1)
            } else {
                assertEquals("no GPU must fall back to the CPU choice", expectedAutoCpu(), rt.activeBackend)
            }
        }
    }

    /** A missing model directory fails cleanly with an error, never a native crash. */
    @Test
    fun missing_model_fails_cleanly() {
        YoloMasterNcnn().use { rt ->
            assertFalse(rt.init(File(ctx.filesDir, "does-not-exist").absolutePath, useVulkan = false))
            assertTrue("expected a load error message", rt.lastError.isNotEmpty())
        }
    }

    /**
     * INT8 on a model with no "<name>-int8_ncnn" sibling staged must fail (never silently load
     * the float model): an int8 number must always come from an int8 model.
     */
    @Test
    fun int8_without_sibling_fails_cleanly() {
        val dir = stage("moa-n_ncnn")   // no moa-n-int8_ncnn is ever staged
        YoloMasterNcnn().use { rt ->
            assertFalse("INT8 must not fall back to float", rt.init(dir, useVulkan = false, precision = Precision.INT8))
            assertTrue("error must name int8: '${rt.lastError}'", rt.lastError.contains("int8"))
        }
    }
}
