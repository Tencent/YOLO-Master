package dev.yolomaster.app.model

import dev.yolomaster.app.detect.DefaultPolicy
import dev.yolomaster.app.ui.bench.BenchHistory
import dev.yolomaster.app.ui.bench.BenchResult
import dev.yolomaster.app.ui.bench.BenchStats
import dev.yolomaster.ncnn.Runtime
import kotlinx.serialization.encodeToString
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The runtime x unit rules of the catalog, the bench cell enumeration, the measured-default
 * policy and the history schema - pure logic, exercised with capability bits instead of the
 * native probe (no native library in a JVM test).
 */
class CatalogRulesTest {
    private val dir = File("/nonexistent")
    private val both = BundledModel("v0.1-seg-n_ncnn", dir, isSeg = true, isCustom = false, hasNcnn = true, hasOnnx = true)
    private val ncnnOnly = BundledModel("moa-n_ncnn", dir, isSeg = false, isCustom = false)
    private val int8 = BundledModel("v0.1-seg-n-int8_ncnn", dir, isSeg = true, isCustom = false)
    private val onnxOnly = BundledModel("custom_ncnn", dir, isSeg = false, isCustom = true, hasNcnn = false, hasOnnx = true, onnxQuant = "a16w8")
    private val all = Caps.NCNN or Caps.ORT or Caps.QNN
    private val noQnn = Caps.NCNN or Caps.ORT
    private val ncnnBuild = Caps.NCNN

    @Test fun ncnnUnitsFollowTheSettingsToggleAndInt8() {
        assertEquals(listOf(ComputeChoice.GPU, ComputeChoice.CPU), ComputeChoice.available(Runtime.NCNN, true, both, all))
        assertEquals(listOf(ComputeChoice.GPU), ComputeChoice.available(Runtime.NCNN, false, both, all))
        assertEquals(listOf(ComputeChoice.CPU), ComputeChoice.available(Runtime.NCNN, false, int8, all))
        assertEquals(listOf(ComputeChoice.GPU, ComputeChoice.CPU), ComputeChoice.available(Runtime.NCNN, true, null, ncnnBuild))
    }

    @Test fun onnxUnitsFollowTheQnnCapability() {
        assertEquals(listOf(ComputeChoice.NPU, ComputeChoice.CPU), ComputeChoice.available(Runtime.ONNX, true, both, all))
        // the CPU EP is ORT's own fallback: never hidden, whatever the toggle says
        assertEquals(listOf(ComputeChoice.NPU, ComputeChoice.CPU), ComputeChoice.available(Runtime.ONNX, false, both, all))
        assertEquals(listOf(ComputeChoice.CPU), ComputeChoice.available(Runtime.ONNX, true, both, noQnn))
    }

    @Test fun modelRuntimesComeFromTheFilesAndTheBuild() {
        assertEquals(listOf(Runtime.NCNN, Runtime.ONNX), both.runtimes)
        assertEquals(listOf(Runtime.NCNN), ncnnOnly.runtimes)
        assertEquals(listOf(Runtime.ONNX), onnxOnly.runtimes)
        assertEquals("a16w8", onnxOnly.onnxQuant); assertNull(both.onnxQuant)
        assertEquals(File(dir, "model-a16w8.onnx"), onnxOnly.onnxQuantFile)
        // a build without ONNX Runtime never offers ONNX, and never offers nothing
        assertEquals(listOf(Runtime.NCNN), runtimesAvailable(both, ncnnBuild))
        assertEquals(listOf(Runtime.NCNN), runtimesAvailable(onnxOnly, ncnnBuild))
        assertEquals(listOf(Runtime.NCNN, Runtime.ONNX), runtimesAvailable(both, noQnn))
        assertEquals(listOf(Runtime.NCNN), runtimesAvailable(null, all))
    }

    @Test fun benchCellsPerModel() {
        assertEquals(
            listOf(Runtime.NCNN to ComputeChoice.GPU, Runtime.NCNN to ComputeChoice.CPU, Runtime.ONNX to ComputeChoice.NPU, Runtime.ONNX to ComputeChoice.CPU),
            BenchStats.cellsFor(both, all),
        )
        assertEquals(listOf(Runtime.NCNN to ComputeChoice.GPU, Runtime.NCNN to ComputeChoice.CPU, Runtime.ONNX to ComputeChoice.CPU), BenchStats.cellsFor(both, noQnn))
        assertEquals(listOf(Runtime.NCNN to ComputeChoice.GPU, Runtime.NCNN to ComputeChoice.CPU), BenchStats.cellsFor(both, ncnnBuild))
        assertEquals(listOf(Runtime.NCNN to ComputeChoice.CPU), BenchStats.cellsFor(int8, all))
        assertEquals(listOf(Runtime.ONNX to ComputeChoice.NPU, Runtime.ONNX to ComputeChoice.CPU), BenchStats.cellsFor(onnxOnly, all))
        assertEquals("ONNX·NPU", BenchStats.cellLabel(Runtime.ONNX, ComputeChoice.NPU))
        assertEquals(60, BenchStats.maxSustainedMinutes(ComputeChoice.NPU))
    }

    @Test fun parityToleratesTheFp16HtpDropsButNotDeadGraphs() {
        // S26 M0: seg-N 11 vs 15 passes (27%), EsMoE-N 19 vs 43 fails (56%), det exact passes
        assertTrue(DefaultPolicy.parityOk(11, 15))
        assertFalse(DefaultPolicy.parityOk(19, 43))
        assertTrue(DefaultPolicy.parityOk(7, 7))
        assertTrue(DefaultPolicy.parityOk(1, 2)); assertTrue(DefaultPolicy.parityOk(3, 2))   // floor of 1
        assertFalse(DefaultPolicy.parityOk(0, 5)); assertFalse(DefaultPolicy.parityOk(0, 0))
    }

    @Test fun decideKeepsNcnnUnlessTheNpuIsFasterAndAgrees() {
        val seg = DefaultPolicy.decide(33.0, 15, 19.6, 11)
        assertEquals(Runtime.ONNX to ComputeChoice.NPU, seg.choice); assertEquals("ONNX·NPU", seg.cell)
        val esmoe = DefaultPolicy.decide(30.0, 43, 12.3, 19)
        assertEquals(Runtime.NCNN to ComputeChoice.CPU, esmoe.choice); assertTrue(esmoe.note.contains("parity off"))
        val slow = DefaultPolicy.decide(20.0, 10, 25.0, 10)
        assertEquals(Runtime.NCNN to ComputeChoice.CPU, slow.choice); assertTrue(slow.note.contains("slower"))
        val noNpu = DefaultPolicy.decide(20.0, 10, Double.NaN, -1)
        assertEquals(Runtime.NCNN to ComputeChoice.CPU, noNpu.choice); assertEquals("ncnn only", noNpu.note)
        // ncnn failed to open but the NPU runs: the NPU is the only working candidate
        val ncnnDead = DefaultPolicy.decide(Double.NaN, -1, 10.0, 12)
        assertEquals(Runtime.NCNN, ncnnDead.runtime)   // parity needs the reference: stays on the certified path
    }

    @Test fun measuredVerdictRoundTripsWithNaN() {
        val m = DefaultPolicy.decide(33.0, 15, Double.NaN, -1).copy(modelId = "esmoe_n_visdrone_ncnn")
        val back = DefaultPolicy.decode(DefaultPolicy.encode(m))!!
        assertEquals(m.modelId, back.modelId); assertEquals(m.choice, back.choice)
        assertEquals(33.0, back.ncnnCpuMs, 0.0); assertTrue(back.onnxNpuMs.isNaN())
        assertNull(DefaultPolicy.decode("not json"))
        assertEquals("--", DefaultPolicy.fmt(Double.NaN)); assertEquals("19.6", DefaultPolicy.fmt(19.62))
    }

    @Test fun probeAndClamp() {
        assertEquals("probe_visdrone.jpg", DefaultPolicy.probeName("esmoe_n_visdrone_ncnn"))
        assertEquals("probe_visdrone.jpg", DefaultPolicy.probeName("moa-n_ncnn"))
        assertEquals("probe.jpg", DefaultPolicy.probeName("v0.1-seg-n_ncnn"))
        // a stored ONNX·NPU pick on a build without ORT lands on ncnn's first unit
        assertEquals(Runtime.NCNN to ComputeChoice.GPU, DefaultPolicy.clamp(Runtime.ONNX to ComputeChoice.NPU, both, true, ncnnBuild))
        // an ncnn·CPU pick with CPU hidden by the toggle lands on GPU
        assertEquals(Runtime.NCNN to ComputeChoice.GPU, DefaultPolicy.clamp(Runtime.NCNN to ComputeChoice.CPU, both, false, all))
        // ONNX·NPU on a QNN-less device lands on the CPU EP
        assertEquals(Runtime.ONNX to ComputeChoice.CPU, DefaultPolicy.clamp(Runtime.ONNX to ComputeChoice.NPU, both, true, noQnn))
        assertEquals(Runtime.ONNX to ComputeChoice.NPU, DefaultPolicy.clamp(Runtime.ONNX to ComputeChoice.NPU, both, true, all))
    }

    @Test fun historySavedBeforeTheOnnxRuntimeStillDecodes() {
        val old = """{"modelId":"v0.1-seg-n_ncnn","compute":"GPU","coldMedian":12.0,"coldP90":14.0,"coldMin":11.0}"""
        val r = BenchHistory.json.decodeFromString<BenchResult>(old)
        assertEquals("ncnn", r.runtime); assertEquals("ncnn·GPU", r.cell)
        val npu = r.copy(runtime = "ONNX", compute = "NPU")
        assertEquals("ONNX·NPU", npu.cell)
        assertTrue(BenchHistory.json.encodeToString(npu).contains("\"runtime\":\"ONNX\""))
    }
}
