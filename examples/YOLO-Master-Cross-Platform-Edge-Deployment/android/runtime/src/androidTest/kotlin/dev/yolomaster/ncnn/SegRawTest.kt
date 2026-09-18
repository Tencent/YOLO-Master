package dev.yolomaster.ncnn

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.abs

/**
 * The app-shape contract (forwardRaw -> RawOutput.decode / maskOverlay) against the harness
 * shape (infer / inferSeg) on the staged `v0.1-seg-n_ncnn` model, CPU and Vulkan, plus the
 * Bitmap-free RGBA input path and the thread-policy call. Runs BEFORE any UI work: it is the
 * early detector for "Vulkan + proto extraction" and "premultiplied overlay" mistakes.
 *
 * Prerequisites: `android/scripts/stage_models.sh` (assets/models/v0.1-seg-n_ncnn) and
 * assets/probe.jpg. Run: ./gradlew :runtime:connectedAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class SegRawTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val tag = "SegRawTest"
    private val model = "v0.1-seg-n_ncnn"

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

    private fun probe(): Bitmap =
        ctx.assets.open("probe.jpg").use { BitmapFactory.decodeStream(it) }.copy(Bitmap.Config.ARGB_8888, false)

    /** Box-for-box comparison (same order: both come from the same score-descending NMS). */
    private fun assertSameBoxes(what: String, a: List<Detection>, b: List<Detection>, tol: Float = 1e-3f) {
        assertEquals("$what: detection count", a.size, b.size)
        for (i in a.indices) {
            val p = a[i]; val q = b[i]
            assertEquals("$what[$i]: class", p.classId, q.classId)
            assertTrue("$what[$i]: score ${p.score} vs ${q.score}", abs(p.score - q.score) <= tol)
            for ((u, v) in listOf(p.x1 to q.x1, p.y1 to q.y1, p.x2 to q.x2, p.y2 to q.y2))
                assertTrue("$what[$i]: box $u vs $v", abs(u - v) <= tol)
        }
    }

    /** Count pixels with a non-zero alpha in a premultiplied ARGB_8888 overlay. */
    private fun opaquePixels(bmp: Bitmap): Int {
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        return px.count { (it ushr 24) != 0 }
    }

    /** forwardRaw + decode(0.25, 0.45) == infer() at the same thresholds; the mask overlay is real. */
    @Test
    fun raw_decode_matches_infer_and_mask_is_nonempty_on_cpu() {
        val img = probe()
        val dir = stage(model)
        YoloMasterNcnn().use { rt ->
            assertTrue("init: ${rt.lastError}", rt.init(dir, useVulkan = false))
            assertTrue("seg model must report isSeg without a forward", rt.isSeg)
            assertEquals(640, rt.imgsz)
            assertEquals(80, rt.classNames.size)
            rt.setConfig(conf = 0.25f, iou = 0.45f)
            val ref = rt.infer(img)
            assertTrue("reference must detect", ref.isNotEmpty())
            val t = rt.lastTimings
            assertTrue("timings must be filled: $t", t.inferMs > 0 && t.preMs > 0)

            rt.forwardRaw(img, confFloor = 0.05f).use { raw ->
                assertTrue(raw.isSeg)
                assertEquals(img.width, raw.origW); assertEquals(img.height, raw.origH)
                assertTrue("candidates at the floor must exceed the NMS survivors", raw.candidateCount >= ref.size)
                val dets = raw.decode(conf = 0.25f, iou = 0.45f)
                Log.i(tag, "cpu ${rt.activeBackend}: infer=${ref.size} decode=${dets.size} cand=${raw.candidateCount} " +
                    "pre=%.1f inf=%.1f dec=%.1f ms".format(raw.preMs, raw.inferMs, raw.decodeMs))
                assertSameBoxes("cpu decode vs infer", ref, dets)
                assertTrue("every decoded det carries its candidate index", dets.all { it.candIndex >= 0 })

                val overlay = raw.maskOverlay(dets, maxSide = 640)
                assertNotNull("seg raw must produce an overlay", overlay)
                overlay!!
                assertEquals(Bitmap.Config.ARGB_8888, overlay.config)
                assertTrue("overlay must be premultiplied", overlay.isPremultiplied)
                assertTrue("overlay long side capped at 640", maxOf(overlay.width, overlay.height) <= 640)
                val n = opaquePixels(overlay)
                Log.i(tag, "overlay ${overlay.width}x${overlay.height}: $n mask pixels")
                assertTrue("overlay must have non-zero alpha pixels", n > 0)

                // Reuse path: same bitmap object comes back, still with mask pixels.
                val again = raw.maskOverlay(dets, maxSide = 640, reuse = overlay)
                assertTrue("reuse must write in place", again === overlay)
                assertTrue(opaquePixels(overlay) > 0)

                // A tighter conf on the same raw is a strict subset (no forward involved).
                val strict = raw.decode(conf = 0.5f, iou = 0.45f)
                assertTrue("higher conf must not add detections", strict.size <= dets.size)
                assertTrue(strict.all { it.score >= 0.5f })
            }
        }
    }

    /** The same contract on Vulkan when a GPU exists: det count within +/-1 of CPU, mask non-empty. */
    @Test
    fun raw_decode_and_mask_on_vulkan_match_cpu() {
        val img = probe()
        val dir = stage(model)
        val cpu = YoloMasterNcnn().use { rt ->
            assertTrue(rt.init(dir, useVulkan = false))
            rt.forwardRaw(img).use { it.decode(0.25f, 0.45f) }
        }
        YoloMasterNcnn().use { rt ->
            assertTrue("init vulkan: ${rt.lastError}", rt.init(dir, useVulkan = true))
            if (!rt.activeBackend.startsWith("ncnn-Vulkan")) {
                Log.i(tag, "no usable Vulkan device (${rt.activeBackend}); GPU leg skipped")
                return
            }
            rt.setConfig(conf = 0.25f, iou = 0.45f)
            val ref = rt.infer(img)
            rt.forwardRaw(img).use { raw ->
                val gpu = raw.decode(0.25f, 0.45f)
                Log.i(tag, "${rt.activeBackend}: decode=${gpu.size} infer=${ref.size} cpu=${cpu.size} inf=%.1f ms".format(raw.inferMs))
                assertSameBoxes("vulkan decode vs infer", ref, gpu)
                assertTrue("Vulkan diverged from CPU", gpu.isNotEmpty() && abs(gpu.size - cpu.size) <= 1)
                val overlay = raw.maskOverlay(gpu, maxSide = 640)
                assertNotNull("Vulkan seg raw must carry the proto", overlay)
                assertTrue("Vulkan overlay must have mask pixels", opaquePixels(overlay!!) > 0)
            }
        }
    }

    /** The Bitmap-free RGBA path (CameraX shape) equals the Bitmap path on the same pixels. */
    @Test
    fun rgba_buffer_path_matches_bitmap_path() {
        val img = probe()
        val dir = stage(model)
        YoloMasterNcnn().use { rt ->
            assertTrue(rt.init(dir, useVulkan = false))
            val fromBitmap = rt.forwardRaw(img).use { it.decode(0.25f, 0.45f) }

            // ARGB_8888 pixels are RGBA bytes in memory, exactly the ImageAnalysis RGBA_8888 layout.
            val buf = ByteBuffer.allocateDirect(img.rowBytes * img.height)
            img.copyPixelsToBuffer(buf)
            buf.rewind()
            rt.forwardRaw(buf, img.width, img.height, img.rowBytes, crop = null, rotationDegrees = 0).use { raw ->
                assertEquals(img.width, raw.origW); assertEquals(img.height, raw.origH)
                assertSameBoxes("rgba vs bitmap", fromBitmap, raw.decode(0.25f, 0.45f))
            }
            // A 90-degree rotation reports upright (swapped) frame dimensions.
            rt.forwardRaw(buf, img.width, img.height, img.rowBytes, crop = null, rotationDegrees = 90).use { raw ->
                assertEquals(img.height, raw.origW); assertEquals(img.width, raw.origH)
            }
        }
    }

    /** Pinning the inference thread to the big cores (the app's thread policy) changes nothing. */
    @Test
    fun powersave_does_not_change_detections() {
        val img = probe()
        val dir = stage(model)
        val ref = YoloMasterNcnn().use { rt ->
            assertTrue(rt.init(dir, useVulkan = false)); rt.setConfig(0.25f, 0.45f); rt.infer(img)
        }
        assertTrue("set_cpu_powersave(2) must be accepted", YoloMasterNcnn.setPowersave(2))
        try {
            YoloMasterNcnn().use { rt ->
                assertTrue(rt.init(dir, useVulkan = false, threads = 2)); rt.setConfig(0.25f, 0.45f)
                val dets = rt.infer(img)
                Log.i(tag, "powersave=2 threads=2: ${dets.size} dets, inferOnly=%.1f ms".format(rt.inferOnly(img)))
                assertSameBoxes("powersave", ref, dets)
            }
        } finally {
            YoloMasterNcnn.setPowersave(0)   // leave the process as found for the other test classes
        }
    }
}
