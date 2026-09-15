package dev.yolomaster.app.ui.bench

import android.content.Context
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.model.Naming
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/** One model x runtime x unit cell of a cold sweep (or the cold baseline of a sustained run). */
@Serializable
data class BenchResult(
    val modelId: String,
    val compute: String,            // "GPU" | "CPU" | "NPU"
    val coldMedian: Double,
    val coldP90: Double,
    val coldMin: Double,
    val preMs: Double = 0.0,
    val infMs: Double = 0.0,
    val decMs: Double = 0.0,
    val sustainedMedian: Double? = null,
    val throttlePct: Double? = null,
    val sparkline: List<Double> = emptyList(),
    val thermal: List<Int> = emptyList(),
    /** "ncnn" | "ONNX". Defaulted so history saved before the ONNX runtime (ncnn-only) still decodes. */
    val runtime: String = "ncnn",
) {
    val fullName: String get() = Naming.fullName(modelId)
    val shortID: String get() = Naming.shortID(modelId)
    val fps: Double get() = if (coldMedian > 0) 1000.0 / coldMedian else 0.0
    /** The cell label everywhere a unit used to stand alone: "ncnn·GPU", "ONNX·NPU". */
    val cell: String get() = "$runtime·$compute"
}

/** A saved run (`BenchHistory`, `BenchView.swift:59-81`). */
@Serializable
data class BenchRun(
    val id: String,
    var name: String,
    val dateMs: Long,
    val mode: String,               // "Cold Sweep" | "Sustained"
    val durationSec: Int = 0,
    val results: List<BenchResult>,
    val thermalStart: Int = 0,
    val thermalEnd: Int = 0,
    val thermalPeak: Int = 0,
) {
    val fastest: BenchResult? get() = results.minByOrNull { it.coldMedian }
}

/** JSON at `filesDir/bench_history.json`, newest first. */
class BenchHistory(ctx: Context) {
    private val file = File(ctx.filesDir, "bench_history.json")
    private val _runs = MutableStateFlow(load())
    val runs: StateFlow<List<BenchRun>> = _runs

    private fun load(): List<BenchRun> = try {
        if (file.isFile) json.decodeFromString<List<BenchRun>>(file.readText()) else emptyList()
    } catch (_: Throwable) { emptyList() }

    private fun persist() { try { file.writeText(json.encodeToString(_runs.value)) } catch (_: Throwable) {} }

    fun add(run: BenchRun) { _runs.value = listOf(run) + _runs.value; persist() }
    fun rename(id: String, name: String) { _runs.value = _runs.value.map { if (it.id == id) it.copy(name = name) else it }; persist() }
    fun delete(ids: Set<String>) { _runs.value = _runs.value.filterNot { it.id in ids }; persist() }
    fun clear() { _runs.value = emptyList(); persist() }

    companion object {
        /** The history codec: unknown keys ignored (forward), defaults written (a file always carries `runtime`). */
        val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

        /** `buildCSV` (`BenchView.swift:738-766`) plus the `runtime` column after `model`. */
        fun resultsCSV(results: List<BenchResult>): String = buildString {
            append("model,runtime,compute,cold_median_ms,cold_p90_ms,cold_min_ms,pre_ms,inf_ms,dec_ms,fps_equiv,sustained_ms,throttle_pct\n")
            for (r in results) {
                append(String.format(Locale.US, "%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%s,%s\n",
                    r.fullName, r.runtime, r.compute, r.coldMedian, r.coldP90, r.coldMin, r.preMs, r.infMs, r.decMs, r.fps,
                    r.sustainedMedian?.let { String.format(Locale.US, "%.2f", it) } ?: "",
                    r.throttlePct?.let { String.format(Locale.US, "%.1f", it) } ?: ""))
            }
        }

        /** `runsCSV` (`BenchView.swift:1018-1038`) plus the `runtime` column after `model`. */
        fun runsCSV(runs: List<BenchRun>): String = buildString {
            append("run,date,mode,model,runtime,compute,cold_median_ms,cold_p90_ms,fps_equiv,sustained_ms,throttle_pct,thermal_start,thermal_end,thermal_peak\n")
            val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply { timeZone = TimeZone.getTimeZone("UTC") }
            for (run in runs) for (r in run.results) {
                append(String.format(Locale.US, "\"%s\",%s,%s,%s,%s,%s,%.2f,%.2f,%.1f,%s,%s,%d,%d,%d\n",
                    run.name.replace("\"", "\"\""), iso.format(Date(run.dateMs)), run.mode, r.fullName, r.runtime, r.compute,
                    r.coldMedian, r.coldP90, r.fps,
                    r.sustainedMedian?.let { String.format(Locale.US, "%.2f", it) } ?: "",
                    r.throttlePct?.let { String.format(Locale.US, "%.1f", it) } ?: "",
                    run.thermalStart, run.thermalEnd, run.thermalPeak))
            }
        }

        /** `durationText` (`BenchView.swift:93-97`): "45s", "3m 12s", "30m". */
        fun durationText(sec: Int): String = when {
            sec < 60 -> "${sec}s"
            sec % 60 == 0 -> "${sec / 60}m"
            else -> "${sec / 60}m ${sec % 60}s"
        }

        fun unitOf(c: ComputeChoice): String = c.label
    }
}
