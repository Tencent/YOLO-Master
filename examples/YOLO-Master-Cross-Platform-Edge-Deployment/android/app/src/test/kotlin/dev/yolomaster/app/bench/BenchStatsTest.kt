package dev.yolomaster.app.bench

import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.ui.bench.BenchHistory
import dev.yolomaster.app.ui.bench.BenchResult
import dev.yolomaster.app.ui.bench.BenchRun
import dev.yolomaster.app.ui.bench.BenchStats
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The pure bench statistics and writers, checked against the Swift formulas they mirror. */
class BenchStatsTest {
    private val samples = listOf(12.0, 10.0, 11.0, 30.0, 13.0, 15.0, 14.0, 9.0, 20.0, 16.0)   // 10 values
    private val sorted = samples.sorted()                                                    // 9 10 11 12 13 14 15 16 20 30

    @Test fun medianIsUpperMiddleOfSorted() {
        // ms[count / 2] = index 5 -> 14 (BenchView.swift:641)
        assertEquals(14.0, BenchStats.median(sorted), 0.0)
        assertEquals(0.0, BenchStats.median(emptyList()), 0.0)
    }

    @Test fun p90IsFloorOfNinetyPercentIndex() {
        // ms[min(int(10 * 0.9), 9)] = index 9 -> 30 (BenchView.swift:642)
        assertEquals(30.0, BenchStats.p90(sorted), 0.0)
        // 5 samples: index int(4.5) = 4 -> the last
        assertEquals(50.0, BenchStats.p90(listOf(10.0, 20.0, 30.0, 40.0, 50.0)), 0.0)
        assertEquals(7.0, BenchStats.p90(listOf(7.0)), 0.0)
    }

    @Test fun minIsFirstOfSorted() {
        assertEquals(9.0, BenchStats.min(sorted), 0.0)
        assertEquals(0.0, BenchStats.min(emptyList()), 0.0)
    }

    @Test fun lastQuarterMedianTakesSlowestQuarter() {
        // suffix(10 / 4 = 2) = [20, 30] -> tail[1] = 30 (BenchView.swift:712-714)
        assertEquals(30.0, BenchStats.lastQuarterMedian(sorted, 1.0), 0.0)
        // fewer than 4 samples: at least one (the slowest)
        assertEquals(3.0, BenchStats.lastQuarterMedian(listOf(1.0, 2.0, 3.0), 1.0), 0.0)
        // no samples -> the cold fallback
        assertEquals(42.0, BenchStats.lastQuarterMedian(emptyList(), 42.0), 0.0)
    }

    @Test fun throttlePercent() {
        assertEquals(25.0, BenchStats.throttlePct(20.0, 25.0), 1e-9)
        assertEquals(-10.0, BenchStats.throttlePct(20.0, 18.0), 1e-9)
        assertEquals(0.0, BenchStats.throttlePct(0.0, 18.0), 0.0)
    }

    @Test fun liveMedianUsesLastThirty() {
        val all = List(100) { 100.0 } + List(30) { it.toDouble() }   // last 30 = 0..29
        assertEquals(15.0, BenchStats.liveMedian(all), 0.0)
    }

    @Test fun bucketedAveragesIntoAtMostNBuckets() {
        val arr = List(250) { it.toDouble() }
        val b = BenchStats.bucketed(arr, 100)   // bucket size 2 -> 125 buckets (iOS keeps the remainder logic)
        assertEquals(125, b.size)
        assertEquals(0.5, b[0], 0.0)
        assertTrue(BenchStats.bucketed(emptyList(), 100).isEmpty())
        assertEquals(listOf(1.0, 2.0), BenchStats.bucketed(listOf(1.0, 2.0), 100))
    }

    @Test fun runNames() {
        assertEquals("v0.1-s... · GPU · 3min", BenchStats.runName(true, "v0.1-s...", "GPU", 3, 1))
        assertEquals("run · CPU · 1min", BenchStats.runName(true, null, "CPU", 1, 1))
        assertEquals("Sweep · 7 models", BenchStats.runName(false, null, "GPU", 3, 7))
    }

    @Test fun sustainedCapsAndPresets() {
        assertEquals(3, BenchStats.maxSustainedMinutes(ComputeChoice.CPU))
        assertEquals(60, BenchStats.maxSustainedMinutes(ComputeChoice.GPU))
        assertEquals(60, BenchStats.maxSustainedMinutes(ComputeChoice.NPU))
        assertEquals(listOf(3, 5, 10, 20), BenchStats.presets(60))
        assertEquals(emptyList<Int>(), BenchStats.presets(3))
    }

    @Test fun durationText() {
        assertEquals("45s", BenchHistory.durationText(45))
        assertEquals("3m 12s", BenchHistory.durationText(192))
        assertEquals("30m", BenchHistory.durationText(1800))
        assertEquals("0s", BenchHistory.durationText(0))
    }

    @Test fun resultsCsvHeaderAndRow() {
        val r = BenchResult(
            modelId = "v0.1-seg-n_ncnn", compute = "GPU", coldMedian = 12.25, coldP90 = 14.0, coldMin = 11.0,
            preMs = 1.5, infMs = 12.0, decMs = 0.75, sustainedMedian = 15.5, throttlePct = 25.25,
        )
        val lines = BenchHistory.resultsCSV(listOf(r)).trimEnd().lines()
        assertEquals("model,runtime,compute,cold_median_ms,cold_p90_ms,cold_min_ms,pre_ms,inf_ms,dec_ms,fps_equiv,sustained_ms,throttle_pct", lines[0])
        assertEquals("YOLO-Master-v0.1-seg-n,ncnn,GPU,12.25,14.00,11.00,1.50,12.00,0.75,81.6,15.50,25.3", lines[1])
        // a cold-only result leaves the sustained columns empty
        val cold = BenchHistory.resultsCSV(listOf(r.copy(sustainedMedian = null, throttlePct = null))).trimEnd().lines()[1]
        assertTrue(cold.endsWith(",81.6,,"))
        // an NPU cell carries its runtime
        val npu = BenchHistory.resultsCSV(listOf(r.copy(runtime = "ONNX", compute = "NPU"))).trimEnd().lines()[1]
        assertTrue(npu.startsWith("YOLO-Master-v0.1-seg-n,ONNX,NPU,"))
    }

    @Test fun runsCsvHeaderAndRow() {
        val r = BenchResult(modelId = "p03_v01n-int8_ncnn", compute = "CPU", coldMedian = 20.0, coldP90 = 22.0, coldMin = 19.0)
        val run = BenchRun(
            id = "abc", name = "Sweep \"x\"", dateMs = 0L, mode = "Cold Sweep", results = listOf(r),
            thermalStart = 0, thermalEnd = 1, thermalPeak = 2,
        )
        val lines = BenchHistory.runsCSV(listOf(run)).trimEnd().lines()
        assertEquals("run,date,mode,model,runtime,compute,cold_median_ms,cold_p90_ms,fps_equiv,sustained_ms,throttle_pct,thermal_start,thermal_end,thermal_peak", lines[0])
        assertEquals("\"Sweep \"\"x\"\"\",1970-01-01T00:00:00Z,Cold Sweep,YOLO-Master-p03_v01n-int8,ncnn,CPU,20.00,22.00,50.0,,,0,1,2", lines[1])
        assertEquals(r, run.fastest)
    }
}
