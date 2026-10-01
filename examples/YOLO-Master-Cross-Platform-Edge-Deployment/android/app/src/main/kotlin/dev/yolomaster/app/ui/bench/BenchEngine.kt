package dev.yolomaster.app.ui.bench

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import dev.yolomaster.app.detect.Detector
import dev.yolomaster.app.model.BundledModel
import dev.yolomaster.app.model.Caps
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.ncnn.Runtime
import kotlinx.coroutines.delay
import kotlin.math.max
import kotlin.math.min

/*
 * The bench loops of `BenchView.swift` (runCold 614-655, runSustained 657-734), split into pure
 * statistics (`BenchStats`, unit-tested) and the detector-driving engine. Methodology (iOS header):
 * flagships throttle, so a real-time claim needs BOTH a cold median and a sustained number. The
 * headline metric is `inferOnly` (pure extractor time); each result also carries one forward +
 * decode pass for the pre/inf/dec breakdown. The gray input yields ~no detections, so decode reads
 * the empty-scene floor - inference is the number that matters.
 */

/** "Cold run" | "Sustained" (`BenchView.Mode`, `BenchView.swift:104-107`). */
enum class BenchMode(val label: String) { Cold("Cold run"), Sustained("Sustained") }

/** `BenchView.Phase` (`BenchView.swift:108-114`). */
sealed class BenchPhase {
    data object Idle : BenchPhase()
    data class Loading(val model: String, val unit: String) : BenchPhase()
    data class Benchmarking(val model: String, val unit: String, val done: Int, val total: Int) : BenchPhase()
    data class Sustained(val model: String, val unit: String, val elapsedS: Int, val totalS: Int) : BenchPhase()
    data object Done : BenchPhase()
}

/**
 * Run-control flags the bench loop polls (`BenchControl`, `BenchView.swift:37-40`): word-sized
 * volatile reads, eventual consistency is fine. Lets a run be PAUSED (resumable) rather than only
 * cancelled, and auto-paused when the tab is hidden so benchmarking never competes with Live/Photo.
 */
class BenchControl {
    @Volatile var paused = false
    @Volatile var cancelled = false
}

/** Pure statistics shared by the engine, the cards and the unit tests. No Android types. */
object BenchStats {
    /** `ms[ms.count / 2]` on a sorted list (`BenchView.swift:641`). */
    fun median(sorted: List<Double>): Double = if (sorted.isEmpty()) 0.0 else sorted[sorted.size / 2]

    /** `ms[min(Int(count * 0.9), count - 1)]` (`BenchView.swift:642`). */
    fun p90(sorted: List<Double>): Double =
        if (sorted.isEmpty()) 0.0 else sorted[min((sorted.size * 0.9).toInt(), sorted.size - 1)]

    fun min(sorted: List<Double>): Double = sorted.firstOrNull() ?: 0.0

    /**
     * The "last-quarter median" (`BenchView.swift:712-714`): the median of the slowest quarter of
     * ALL samples (at least one), i.e. what the SoC settles to once it throttles. [fallback] is
     * the cold median, returned when there were no samples at all.
     */
    fun lastQuarterMedian(sorted: List<Double>, fallback: Double): Double {
        if (sorted.isEmpty()) return fallback
        val tail = sorted.takeLast(max(sorted.size / 4, 1))
        return tail[tail.size / 2]
    }

    /** `(sust - cold) / cold * 100`, 0 when there is no baseline (`BenchView.swift:722`). */
    fun throttlePct(cold: Double, sustained: Double): Double = if (cold > 0) (sustained - cold) / cold * 100.0 else 0.0

    /** Median of the last [last] inferences = the "live" readout (`BenchView.swift:702-703`). */
    fun liveMedian(all: List<Double>, last: Int = 30): Double = median(all.takeLast(last).sorted())

    /** Average [arr] into up to [n] buckets to smooth per-inference jitter (`BenchView.swift:487-496`). */
    fun bucketed(arr: List<Double>, n: Int): List<Double> {
        if (arr.isEmpty()) return emptyList()
        val bs = max(1, arr.size / n)
        val out = ArrayList<Double>(arr.size / bs + 1)
        var i = 0
        while (i < arr.size) {
            val end = min(i + bs, arr.size)
            var s = 0.0
            for (k in i until end) s += arr[k]
            out += s / (end - i)
            i = end
        }
        return out
    }

    /** Auto run names (`BenchView.swift:570-572`): "<shortID> · <unit> · <n>min" / "Cold run · <n> models". */
    fun runName(sustained: Boolean, shortID: String?, unit: String, minutes: Int, modelCount: Int): String =
        if (sustained) "${shortID ?: "run"} · $unit · ${minutes}min" else "Cold run · $modelCount models"

    /** CPU inference is slow and hot: sustained CPU stress is capped at 3 minutes (`BenchView.swift:309`); GPU and NPU get 60. */
    fun maxSustainedMinutes(unit: ComputeChoice): Int = if (unit == ComputeChoice.CPU) 3 else 60

    /** Quick presets 3/5/10/20 filtered `< max` (`BenchView.swift:290`). */
    fun presets(maxMinutes: Int): List<Int> = listOf(3, 5, 10, 20).filter { it < maxMinutes }

    /**
     * The runtime x unit cells a model is swept on, accelerator first within each runtime:
     * ncnn·GPU, ncnn·CPU (INT8 CPU-only siblings skip GPU), then ONNX·NPU (when the device has the
     * QNN runtime) and ONNX·CPU for models that ship a `model.onnx` on a build with ONNX Runtime.
     */
    fun cellsFor(model: BundledModel, caps: Int): List<Pair<Runtime, ComputeChoice>> = buildList {
        if (model.hasNcnn) {
            if (!model.cpuOnly) add(Runtime.NCNN to ComputeChoice.GPU)
            add(Runtime.NCNN to ComputeChoice.CPU)
        }
        if (model.hasOnnx && Caps.hasOrt(caps)) {
            if (Caps.hasQnn(caps)) add(Runtime.ONNX to ComputeChoice.NPU)
            add(Runtime.ONNX to ComputeChoice.CPU)
        }
    }

    /** The label of one cell: "ncnn·GPU", "ONNX·NPU" (matches [BenchResult.cell]). */
    fun cellLabel(runtime: Runtime, compute: ComputeChoice): String = "${runtime.label}·${compute.label}"
}

/** What the engine reports while it runs; every call may come from the bench thread. */
interface BenchSink {
    fun phase(p: BenchPhase)
    /** A cold-run cell finished (the view appends it and taps a light haptic). */
    fun cell(r: BenchResult)
    /** Sustained: the cold baseline is in; the graph starts. */
    fun baseline(coldMedian: Double)
    /** Sustained: one publish (<= every 100 ms) of the rolling window. */
    fun tick(spark: List<Double>, thermal: List<Int>, liveMs: Double, elapsedS: Int)
    /** Current thermal level 0..3, sampled per publish. */
    fun thermalLevel(): Int
}

/** Outcome of a sustained run. Cancelled runs are never saved. */
sealed class SustainedOutcome {
    data class Completed(val result: BenchResult, val durationSec: Int) : SustainedOutcome()
    data object Cancelled : SustainedOutcome()
    data class Failed(val message: String) : SustainedOutcome()
}

/**
 * Drives a [Detector] through the two protocols. Call on ONE dedicated thread: `Detector.open`
 * pins the calling thread to the big cores and the affinity only holds for that thread.
 */
class BenchEngine(
    /** App context for the ONNX side of [Detector.open] (EPContext cache dir, perf-mode / quant Settings); null = ncnn-only defaults. */
    private val ctx: Context? = null,
    /** Runtime capability bits; decides which ONNX cells exist. */
    private val caps: Int = Caps.device,
) {
    /** Suspend while paused (polled every 120 ms, `BenchView.swift:560-565`); false = cancelled. */
    private suspend fun pauseGate(control: BenchControl): Boolean {
        while (control.paused && !control.cancelled) delay(120)
        return !control.cancelled
    }

    /**
     * Cold run (`BenchView.swift:614-655`): every model x its runtime·unit cells, warmup untimed
     * then [iters] timed `inferOnly` calls on the gray probe. Returns true when it ran to completion.
     */
    suspend fun runCold(models: List<BundledModel>, warmup: Int, iters: Int, control: BenchControl, sink: BenchSink): Boolean {
        val img = Detector.grayProbe(640)
        try {
            val total = models.sumOf { BenchStats.cellsFor(it, caps).size }
            var done = 0
            for (m in models) for ((r, c) in BenchStats.cellsFor(m, caps)) {
                val cell = BenchStats.cellLabel(r, c)
                if (control.cancelled) return false
                if (!pauseGate(control)) return false
                sink.phase(BenchPhase.Loading(m.fullName, cell))
                val det = Detector.open(m, r, c, ctx = ctx)
                if (det == null) { Log.w(TAG, "skip ${m.id}@$cell: ${Detector.lastError}"); done++; continue }
                try {
                    // WHY: the runtime may decline Vulkan (no device, driver blacklist) or the QNN EP
                    // (no HTP, libs missing) and silently run on the CPU; a CPU number labelled "GPU"
                    // or "NPU" would poison the accelerator-vs-CPU verdict.
                    if (c == ComputeChoice.GPU && !det.onGpu) { Log.w(TAG, "skip ${m.id}@$cell: backend ${det.activeBackend}"); done++; continue }
                    if (c == ComputeChoice.NPU && !det.onNpu) { Log.w(TAG, "skip ${m.id}@$cell: backend ${det.activeBackend}"); done++; continue }
                    sink.phase(BenchPhase.Benchmarking(m.fullName, cell, done, total))
                    val ms = timedSamples(det, img, warmup, iters, control) ?: return false
                    if (ms.isEmpty()) { done++; continue }
                    val (pre, inf, dec) = stagePass(det, img)
                    val res = BenchResult(
                        modelId = m.id, runtime = r.label, compute = c.label,
                        coldMedian = BenchStats.median(ms), coldP90 = BenchStats.p90(ms), coldMin = BenchStats.min(ms),
                        preMs = pre, infMs = inf, decMs = dec,
                    )
                    done++
                    sink.phase(BenchPhase.Benchmarking(m.fullName, cell, done, total))
                    sink.cell(res)
                } finally { det.close() }
            }
            return true
        } finally { img.recycle() }
    }

    /**
     * Sustained (`BenchView.swift:657-734`): cold baseline (same warmup + iters), then a loop until
     * [minutes] of un-paused time: publish <= every 100 ms, rolling 60 s window bucketed to 100
     * points, live = median of the last 30, thermal sampled per publish, final = last-quarter median.
     */
    suspend fun runSustained(
        m: BundledModel, r: Runtime, c: ComputeChoice, minutes: Int, warmup: Int, iters: Int, control: BenchControl, sink: BenchSink,
    ): SustainedOutcome {
        val img = Detector.grayProbe(640)
        val totalS = minutes * 60
        val cell = BenchStats.cellLabel(r, c)
        try {
            sink.phase(BenchPhase.Loading(m.fullName, cell))
            val det = Detector.open(m, r, c, ctx = ctx) ?: return SustainedOutcome.Failed("Could not load ${m.fullName}: ${Detector.lastError}")
            try {
                if (c == ComputeChoice.GPU && !det.onGpu) return SustainedOutcome.Failed("GPU unavailable: ${det.backendNote.ifEmpty { det.activeBackend }}")
                if (c == ComputeChoice.NPU && !det.onNpu) return SustainedOutcome.Failed("NPU unavailable: ${det.backendNote.ifEmpty { det.activeBackend }}")
                // cold baseline
                val cold = timedSamples(det, img, warmup, iters, control) ?: return SustainedOutcome.Cancelled
                val coldMed = BenchStats.median(cold)
                sink.baseline(coldMed)
                // sustained loop with a rolling 1-minute display window
                var t0 = SystemClock.elapsedRealtime()
                val all = ArrayList<Double>()               // every inference ms
                val atimes = ArrayList<Long>()              // its elapsed time (ms since t0)
                val thermT = ArrayList<Long>()              // thermal sample times
                val thermL = ArrayList<Int>()               // thermal sample levels
                var lastPub = 0L
                var winStart = 0
                val totalMs = totalS * 1000L
                while (SystemClock.elapsedRealtime() - t0 < totalMs) {
                    if (control.cancelled) return SustainedOutcome.Cancelled
                    if (control.paused) {
                        val ps = SystemClock.elapsedRealtime()
                        if (!pauseGate(control)) return SustainedOutcome.Cancelled
                        t0 += SystemClock.elapsedRealtime() - ps   // exclude paused time from elapsed
                    }
                    val st = SystemClock.elapsedRealtime() - t0
                    val t = try { det.inferOnly(img) } catch (e: Throwable) { Log.w(TAG, "inferOnly failed: ${e.message}"); -1.0 }
                    if (t >= 0) { all += t; atimes += st }
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastPub >= 100) {
                        lastPub = now
                        val elapsed = now - t0
                        thermT += elapsed; thermL += sink.thermalLevel()
                        // rolling 1-minute window
                        val cutoff = elapsed - 60_000
                        while (winStart < atimes.size && atimes[winStart] < cutoff) winStart++
                        val winMs = all.subList(winStart, all.size)
                        val winTherm = ArrayList<Int>()
                        for (i in thermT.indices) if (thermT[i] >= cutoff) winTherm += thermL[i]
                        sink.tick(BenchStats.bucketed(winMs, 100), winTherm, BenchStats.liveMedian(all), (elapsed / 1000).toInt())
                        sink.phase(BenchPhase.Sustained(m.fullName, cell, (elapsed / 1000).toInt(), totalS))
                    }
                }
                val durationSec = ((SystemClock.elapsedRealtime() - t0) / 1000).toInt()
                val sortedMs = all.sorted()
                val sust = BenchStats.lastQuarterMedian(sortedMs, coldMed)
                val (pre, inf, dec) = stagePass(det, img)
                val res = BenchResult(
                    modelId = m.id, runtime = r.label, compute = c.label,
                    coldMedian = coldMed, coldP90 = BenchStats.p90(sortedMs), coldMin = BenchStats.min(sortedMs),
                    preMs = pre, infMs = inf, decMs = dec,
                    sustainedMedian = sust, throttlePct = BenchStats.throttlePct(coldMed, sust),
                    sparkline = BenchStats.bucketed(all, 120),    // smooth full-run trend
                    thermal = thermL.toList(),
                )
                return SustainedOutcome.Completed(res, durationSec)
            } finally { det.close() }
        } finally { img.recycle() }
    }

    /**
     * [warmup] untimed then [iters] timed `inferOnly` calls, sorted; null when cancelled. A pause
     * that lands mid-cell restarts the cell: the SoC cools while paused and the resumed samples
     * would no longer be a cold median.
     */
    private suspend fun timedSamples(det: Detector, img: Bitmap, warmup: Int, iters: Int, control: BenchControl): List<Double>? {
        while (true) {
            for (i in 0 until warmup) {
                if (control.cancelled) return null
                try { det.inferOnly(img) } catch (_: Throwable) {}
            }
            val ms = ArrayList<Double>(iters)
            var interrupted = false
            for (i in 0 until iters) {
                if (control.cancelled) return null
                if (control.paused) { if (!pauseGate(control)) return null; interrupted = true; break }
                try { ms += det.inferOnly(img) } catch (_: Throwable) {}
            }
            if (!interrupted) return ms.sorted()
        }
    }

    /**
     * One forward + decode pass -> pre / inf / dec stage ms (`stageBreakdown`, `BenchView.swift:599-612`).
     * pre and inf come from the runtime's own stage clocks; dec = candidate decode + NMS wall time.
     */
    private fun stagePass(det: Detector, img: Bitmap): Triple<Double, Double, Double> {
        val raw = try { det.forward(img) } catch (t: Throwable) { Log.w(TAG, "stage pass failed: ${t.message}"); return Triple(0.0, 0.0, 0.0) }
        try {
            val timings = det.lastTimings
            val t1 = System.nanoTime()
            raw.decode(0.25f, 0.45f)
            val nms = (System.nanoTime() - t1) / 1e6
            return Triple(timings.preMs, timings.inferMs, raw.decodeMs + nms)
        } finally { raw.close() }
    }

    private companion object { const val TAG = "BenchEngine" }
}
