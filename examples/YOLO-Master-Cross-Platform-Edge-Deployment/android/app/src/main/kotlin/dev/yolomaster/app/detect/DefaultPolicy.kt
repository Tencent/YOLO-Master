package dev.yolomaster.app.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.util.Log
import dev.yolomaster.app.model.BundledModel
import dev.yolomaster.app.model.Caps
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.model.runtimesAvailable
import dev.yolomaster.app.system.Prefs
import dev.yolomaster.ncnn.Runtime
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * The measured default (decision of 2026-09-08): which runtime x unit a model opens on when the
 * user has not picked one by hand. Runs ONCE per model id and app version: ncnn-CPU (fp16 AUTO)
 * against ONNX-NPU when the device has the Hexagon runtime and the model ships an ONNX graph,
 * on the bundled probe image, 3 warm + 5 timed forwards each; the faster candidate that still
 * detects wins. The verdict and both medians are stored in [Prefs]; Settings lists and resets them.
 *
 * Parity rule: the NPU runs fp32 graphs as fp16 and DROPS borderline boxes (S26 M0: seg-N 11 vs
 * 15, EsMoE-N 19 vs 43 at conf 0.25) while the det family is exact; a candidate counts only when
 * it detects at least one object and its count is within 35% of the ncnn-CPU count. So seg-N goes
 * to the NPU (27% off, 19.6 vs 33 ms) and EsMoE-N stays on ncnn (56% off) with the numbers on
 * record - the accuracy claim itself is the smoke-set mAP certification, never a probe count.
 */
object DefaultPolicy {
    private const val TAG = "DefaultPolicy"
    private const val PREFIX = "default:"
    private const val WARM = 3
    private const val TIMED = 5
    /** A candidate's det count may differ from the ncnn-CPU reference by this fraction (at least 1). */
    const val PARITY_TOLERANCE = 0.35

    /** One stored verdict. Medians are NaN when that candidate was not run (or did not load). */
    @Serializable
    data class Measured(
        val modelId: String,
        val runtime: Runtime,
        val compute: ComputeChoice,
        val ncnnCpuMs: Double = Double.NaN,
        val onnxNpuMs: Double = Double.NaN,
        val ncnnDets: Int = -1,
        val npuDets: Int = -1,
        /** Why the verdict is what it is ("NPU 1.7x faster", "NPU parity off: 19 vs 43", "ncnn only", ...). */
        val note: String = "",
    ) {
        val choice: Pair<Runtime, ComputeChoice> get() = runtime to compute
        val cell: String get() = "${runtime.label}·${compute.label}"
    }

    // NaN marks a candidate that did not run: the codec must let it through (Json rejects it by default)
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true; allowSpecialFloatingPointValues = true }

    /** The store codec (a Prefs string per model); [decode] is null for a corrupt or foreign entry. */
    fun encode(m: Measured): String = json.encodeToString(m)
    fun decode(s: String): Measured? = try { json.decodeFromString<Measured>(s) } catch (_: Throwable) { null }

    // ---- pure rules (unit-tested) ----------------------------------------------------------------

    /** Domain-matched probe: aerial models get the VisDrone frame (a living room legitimately yields nothing). */
    fun probeName(modelId: String): String {
        val id = modelId.lowercase()
        return if (id.contains("visdrone") || id.startsWith("moa") || id.startsWith("mot") || id.contains("mixture")) "probe_visdrone.jpg" else "probe.jpg"
    }

    /** The parity gate: [n] detections against the ncnn-CPU reference [ref]. */
    fun parityOk(n: Int, ref: Int): Boolean =
        n >= 1 && abs(n - ref) <= max(1, (ref * PARITY_TOLERANCE).roundToInt())

    /**
     * The verdict from the two measurements: the NPU wins when it loaded, passes parity and is
     * faster; ncnn-CPU otherwise (also when ncnn itself failed to load - it is the certified path
     * and its failure is reported at open time, not hidden behind a default).
     */
    fun decide(ncnnMs: Double, ncnnDets: Int, npuMs: Double, npuDets: Int): Measured {
        val npuRan = !npuMs.isNaN()
        val parity = npuRan && parityOk(npuDets, ncnnDets)
        val faster = npuRan && (ncnnMs.isNaN() || npuMs < ncnnMs)
        val note = when {
            !npuRan -> "ncnn only"
            ncnnDets < 0 -> "ncnn did not load: no reference"
            !parity -> "NPU parity off: $npuDets vs $ncnnDets dets"
            !faster -> "NPU slower: ${fmt(npuMs)} vs ${fmt(ncnnMs)} ms"
            else -> "NPU ${"%.1f".format(ncnnMs / npuMs)}x faster"
        }
        val npuWins = parity && faster
        return Measured(
            modelId = "", runtime = if (npuWins) Runtime.ONNX else Runtime.NCNN,
            compute = if (npuWins) ComputeChoice.NPU else ComputeChoice.CPU,
            ncnnCpuMs = ncnnMs, onnxNpuMs = npuMs, ncnnDets = ncnnDets, npuDets = npuDets, note = note,
        )
    }

    /** Keep a runtime x unit inside what the pickers offer for [model] (the Settings toggle, the device, the files). */
    fun clamp(choice: Pair<Runtime, ComputeChoice>, model: BundledModel?, allowCPU: Boolean, caps: Int): Pair<Runtime, ComputeChoice> {
        val runtimes = runtimesAvailable(model, caps)
        val r = if (choice.first in runtimes) choice.first else runtimes.first()
        val units = ComputeChoice.available(r, allowCPU, model, caps)
        val c = if (choice.second in units) choice.second else units.first()
        return r to c
    }

    // ---- store ----------------------------------------------------------------------------------------

    private fun key(ctx: Context, modelId: String) = "$PREFIX${versionCode(ctx)}:$modelId"

    private fun versionCode(ctx: Context): Long = try {
        val pi = ctx.packageManager.getPackageInfo(ctx.packageName, 0)
        if (android.os.Build.VERSION.SDK_INT >= 28) pi.longVersionCode else @Suppress("DEPRECATION") pi.versionCode.toLong()
    } catch (_: Throwable) { 0L }

    /** The stored verdict for [model] under this app version, or null (never measured / reset). */
    fun cached(ctx: Context, model: BundledModel): Measured? =
        Prefs.get(ctx).getString(key(ctx, model.id), null)?.let { decode(it) }

    /** Every stored verdict of this app version, catalog order by model id. */
    fun all(ctx: Context): List<Measured> {
        val prefix = "$PREFIX${versionCode(ctx)}:"
        return Prefs.get(ctx).all.entries
            .filter { it.key.startsWith(prefix) }
            .mapNotNull { (_, v) -> (v as? String)?.let { decode(it) } }
            .sortedBy { it.modelId.lowercase() }
    }

    fun reset(ctx: Context, modelId: String) = Prefs.get(ctx).edit().remove(key(ctx, modelId)).apply()

    /** Drop every verdict of every version (older versions' keys are dead weight anyway). */
    fun resetAll(ctx: Context) {
        val p = Prefs.get(ctx)
        val e = p.edit()
        p.all.keys.filter { it.startsWith(PREFIX) }.forEach { e.remove(it) }
        e.apply()
    }

    // ---- resolution -------------------------------------------------------------------------------

    /**
     * What [model] should open on WITHOUT measuring: the user's remembered pick for it, else the
     * cached verdict, else null (call [resolve] off the main thread). Clamped to the pickers.
     */
    fun resolveCached(ctx: Context, model: BundledModel, allowCPU: Boolean, caps: Int = Caps.device): Pair<Runtime, ComputeChoice>? {
        val choice = Prefs.choiceFor(ctx, model.id) ?: cached(ctx, model)?.choice ?: return null
        return clamp(choice, model, allowCPU, caps)
    }

    /** [resolveCached], measuring first when nothing is stored. Blocking: seconds on the first NPU init. */
    fun resolve(ctx: Context, model: BundledModel, allowCPU: Boolean, caps: Int = Caps.device): Pair<Runtime, ComputeChoice> =
        resolveCached(ctx, model, allowCPU, caps) ?: clamp(pick(ctx, model, caps), model, allowCPU, caps)

    /**
     * The measured default for [model] (cached per model id + app version). Call on the inference
     * thread: it opens detectors, and the ncnn open pins the caller. Serialized: two tabs asking
     * for the same model at once measure once.
     */
    @Synchronized
    fun pick(ctx: Context, model: BundledModel, caps: Int = Caps.device): Pair<Runtime, ComputeChoice> {
        cached(ctx, model)?.let { return it.choice }
        val m = measure(ctx, model, caps)
        Prefs.get(ctx).edit().putString(key(ctx, model.id), encode(m)).apply()
        Log.i(TAG, "measured default for ${model.id}: ${m.cell} (ncnn-CPU ${fmt(m.ncnnCpuMs)} ms / ${m.ncnnDets} dets, ONNX-NPU ${fmt(m.onnxNpuMs)} ms / ${m.npuDets} dets): ${m.note}")
        return m.choice
    }

    private fun measure(ctx: Context, model: BundledModel, caps: Int): Measured {
        val npuCandidate = Caps.hasQnn(caps) && model.hasOnnx && Runtime.ONNX in runtimesAvailable(model, caps)
        if (!npuCandidate || !model.hasNcnn) {
            // WHY no mini-bench: with a single candidate there is nothing to compare, and an
            // ncnn-only model would pay a 1-2 s open for a number nobody acts on. ncnn-CPU is the
            // certified path (and on the S26 fp16 CPU beats Vulkan, 33 vs 50 ms for seg-N).
            val only = if (model.hasNcnn) Runtime.NCNN to ComputeChoice.CPU else Runtime.ONNX to (if (Caps.hasQnn(caps)) ComputeChoice.NPU else ComputeChoice.CPU)
            return Measured(model.id, only.first, only.second, note = if (model.hasNcnn) "ncnn only" else "ONNX only")
        }
        val probe = loadProbe(ctx, model)
            ?: return Measured(model.id, Runtime.NCNN, ComputeChoice.CPU, note = "no probe image bundled")
        try {
            val (ncnnMs, ncnnDets) = time(model, Runtime.NCNN, ComputeChoice.CPU, probe, ctx)
            val (npuMs, npuDets) = time(model, Runtime.ONNX, ComputeChoice.NPU, probe, ctx)
            return decide(ncnnMs, ncnnDets, npuMs, npuDets).copy(modelId = model.id)
        } finally { probe.recycle() }
    }

    /** Median forward wall time of [TIMED] frames after [WARM], and the det count of the last one; NaN/-1 when the open failed. */
    private fun time(model: BundledModel, runtime: Runtime, compute: ComputeChoice, probe: Bitmap, ctx: Context): Pair<Double, Int> {
        val det = Detector.open(model, runtime, compute, ctx = ctx) ?: run {
            Log.w(TAG, "${model.id} on ${runtime.label}/${compute.label} did not open: ${Detector.lastError}")
            return Double.NaN to -1
        }
        try {
            // an NPU request that fell back to the CPU EP is not an NPU number: report it as "did not run"
            if (compute == ComputeChoice.NPU && !det.onNpu) { Log.w(TAG, "${model.id}: NPU declined (${det.activeBackend})"); return Double.NaN to -1 }
            repeat(WARM) { det.forward(probe).close() }
            val ms = DoubleArray(TIMED)
            var dets = 0
            for (i in 0 until TIMED) {
                val t0 = SystemClock.elapsedRealtimeNanos()
                val raw = det.forward(probe)
                ms[i] = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                if (i == TIMED - 1) dets = raw.decode(0.25f, 0.45f, 300).size
                raw.close()
            }
            ms.sort()
            return ms[TIMED / 2] to dets
        } catch (t: Throwable) {
            Log.w(TAG, "${model.id} on ${runtime.label}/${compute.label} failed while measuring: ${t.message}")
            return Double.NaN to -1
        } finally { det.close() }
    }

    private fun loadProbe(ctx: Context, model: BundledModel): Bitmap? {
        val want = probeName(model.id)
        for (name in listOf(want, "probe.jpg")) {
            try {
                return ctx.assets.open(name).use { BitmapFactory.decodeStream(it) }?.let { b ->
                    if (b.config == Bitmap.Config.ARGB_8888) b else b.copy(Bitmap.Config.ARGB_8888, false).also { b.recycle() }
                }
            } catch (_: Throwable) { /* not bundled: try the next */ }
        }
        Log.w(TAG, "no probe image in assets ($want): defaulting ${model.id} to ncnn-CPU unmeasured")
        return null
    }

    fun fmt(ms: Double): String = if (ms.isNaN()) "--" else String.format(java.util.Locale.US, "%.1f", ms)
}
