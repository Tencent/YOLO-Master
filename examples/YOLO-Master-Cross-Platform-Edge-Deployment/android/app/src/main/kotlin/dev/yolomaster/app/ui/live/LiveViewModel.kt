package dev.yolomaster.app.ui.live

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import android.util.Size
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.yolomaster.app.YoloMasterApp
import dev.yolomaster.app.detect.DefaultPolicy
import dev.yolomaster.app.detect.Detector
import dev.yolomaster.app.model.BundledModel
import dev.yolomaster.app.model.Caps
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.system.GallerySaver
import dev.yolomaster.app.system.Prefs
import dev.yolomaster.app.system.ThermalMonitor
import dev.yolomaster.app.ui.common.Tuning
import dev.yolomaster.app.ui.overlay.Annotate
import dev.yolomaster.app.ui.overlay.SegOverlayMode
import dev.yolomaster.ncnn.Detection
import dev.yolomaster.ncnn.Placement
import dev.yolomaster.ncnn.Runtime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.roundToInt

/** One rendered frame's worth of overlay data (published conflated; stale frames are dropped). */
data class FrameResult(
    val dets: List<Detection> = emptyList(),
    val mask: Bitmap? = null,
    val frameSize: Size = Size(1280, 720),
    val pre: Double = 0.0, val inf: Double = 0.0, val dec: Double = 0.0, val maskMs: Double = 0.0,
    val loopHz: Double = 0.0,
    val seq: Long = 0,
)

/** UI-facing state of the Live tab (`LiveView.swift` @State). */
data class LiveUi(
    val models: List<BundledModel> = emptyList(),
    /** Runtime capability bits of this build/device (probed once, off main); 0 until known. */
    val caps: Int = 0,
    val initializing: Boolean = true,
    val selected: BundledModel? = null,
    val runtime: Runtime = Runtime.NCNN,
    val compute: ComputeChoice = ComputeChoice.GPU,
    /** The measured-default mini-bench is running for [selected] (the pickers settle when it lands). */
    val measuring: Boolean = false,
    val wantRun: Boolean = true,
    val running: Boolean = false,
    val loadingModel: Boolean = false,
    val isSeg: Boolean = false,
    val classNames: List<String> = emptyList(),
    val backend: String = "",
    val backendNote: String = "",
    /** ONNX graph placement of the loaded model (0/0 for ncnn). */
    val placement: Placement = Placement(0, 0),
    val capturing: Boolean = false,
    val loadError: String? = null,
)

class LiveViewModel(app: Application) : AndroidViewModel(app), ImageAnalysis.Analyzer {
    private val catalog = YoloMasterApp.from(app).catalog
    val tuning = Tuning()
    val thermal = ThermalMonitor(app)
    val camera = CameraController(app)

    private val _ui = MutableStateFlow(LiveUi(runtime = Prefs.runtime(app)))
    val ui: StateFlow<LiveUi> = _ui
    private val _frame = MutableStateFlow(FrameResult())
    val frame: StateFlow<FrameResult> = _frame

    // inference-thread state
    @Volatile private var detector: Detector? = null
    private val running = AtomicBoolean(false)
    private val maskRing = arrayOfNulls<Bitmap>(3)
    private var ringIdx = 0
    private var seq = 0L
    private var loopWindowStart = 0L
    private var loopFrames = 0
    private var loopHz = 0.0
    private val shutterArmed = AtomicBoolean(true)

    /**
     * ADPF performance hint session (API 31+): tells the power HAL the inference thread has a
     * 33 ms deadline and reports each frame's real duration. This is the sanctioned way to lift
     * a vendor "camera scenario" CPU cap (S26: prime cores held at 1.4 GHz while cool).
     */
    private var hint: android.os.PerformanceHintManager.Session? = null
    private fun openHint() {
        if (android.os.Build.VERSION.SDK_INT < 31) return
        try {
            val phm = getApplication<Application>().getSystemService(android.os.PerformanceHintManager::class.java) ?: return
            hint?.close()
            // the whole process' threads share the session: the OpenMP workers are unnamed and
            // cannot be told apart from /proc, and boosting the group is what we want anyway
            val tids = java.io.File("/proc/self/task").list()?.mapNotNull { it.toIntOrNull() }?.toIntArray() ?: intArrayOf(android.os.Process.myTid())
            hint = phm.createHintSession(tids, 33_000_000L)
            Log.i(TAG, "ADPF hint session: ${if (hint != null) "created for ${tids.size} threads" else "unavailable"}")
        } catch (t: Throwable) { Log.w(TAG, "ADPF hint failed: ${t.message}") }
    }
    private fun closeHint() { try { hint?.close() } catch (_: Throwable) {}; hint = null }

    init {
        thermal.start()
        viewModelScope.launch {
            // the capability probe dlopens the QNN runtime: off main, together with the asset copy
            val (models, caps) = withContext(Dispatchers.IO) { catalog.discover() to Caps.device }
            _ui.update { it.copy(models = models, caps = caps, initializing = false) }
            BundledModel.preferred(models)?.let { applySelection(it) }
            // Settings toggle: CPU hidden -> ncnn snaps to GPU (LiveView.swift:204-206); ONNX has no GPU
            if (!Prefs.allowCPU(app) && _ui.value.runtime == Runtime.NCNN && _ui.value.compute == ComputeChoice.CPU && _ui.value.selected?.cpuOnly != true) {
                _ui.update { it.copy(compute = ComputeChoice.GPU) }
            }
            if (_ui.value.wantRun) startLoop()
        }
    }

    // ---- selection ----------------------------------------------------------------------------

    /**
     * Select [m] and settle its runtime x unit: the user's remembered pick for it, else the cached
     * measured default, else measure now on the inference thread (the pickers keep the previous
     * value and the loading card says "(measuring)" until the verdict lands, 1-2 s per candidate).
     */
    private fun applySelection(m: BundledModel) {
        val app = getApplication<Application>()
        val allow = Prefs.allowCPU(app)
        val caps = _ui.value.caps
        val cached = DefaultPolicy.resolveCached(app, m, allow, caps)
        if (cached != null) { _ui.update { it.copy(selected = m, runtime = cached.first, compute = cached.second) }; return }
        _ui.update { it.copy(selected = m, measuring = true) }
        camera.analysisExecutor.execute {
            val pick = try { DefaultPolicy.resolve(app, m, allow, caps) } catch (t: Throwable) {
                Log.w(TAG, "measured default failed for ${m.id}: ${t.message}"); Runtime.NCNN to ComputeChoice.CPU
            }
            _ui.update { if (it.selected?.id == m.id) it.copy(runtime = pick.first, compute = pick.second, measuring = false) else it.copy(measuring = false) }
        }
    }

    fun selectModel(m: BundledModel) = applySelection(m)

    /** A hand pick is remembered per model and beats the measured default from then on. */
    fun selectRuntime(r: Runtime) {
        val app = getApplication<Application>()
        val u = _ui.value
        val units = ComputeChoice.available(r, Prefs.allowCPU(app), u.selected, u.caps)
        val c = if (u.compute in units) u.compute else units.first()
        Prefs.setRuntime(app, r)
        u.selected?.let { Prefs.setChoice(app, it.id, r, c) }
        _ui.update { it.copy(runtime = r, compute = c) }
    }

    fun selectCompute(c: ComputeChoice) {
        val u = _ui.value
        u.selected?.let { Prefs.setChoice(getApplication(), it.id, u.runtime, c) }
        _ui.update { it.copy(compute = c) }
    }

    fun refreshModels() {
        viewModelScope.launch {
            val models = withContext(Dispatchers.IO) { catalog.discover(force = true) }
            _ui.update { it.copy(models = models) }
            val keep = _ui.value.selected?.let { s -> models.firstOrNull { it.id == s.id } } ?: BundledModel.preferred(models)
            if (keep != null) applySelection(keep) else _ui.update { it.copy(selected = null) }
        }
    }

    // ---- run control ----------------------------------------------------------------------------

    fun togglePlay() {
        if (_ui.value.running || _ui.value.loadingModel) { _ui.update { it.copy(wantRun = false) }; suspendLoop() }
        else { _ui.update { it.copy(wantRun = true) }; startLoop() }
    }

    /** Loads the model on the inference thread; frames are only processed once it is loaded. */
    fun startLoop() {
        val model = _ui.value.selected ?: return
        if (running.get() || _ui.value.loadingModel) return
        _ui.update { it.copy(loadingModel = true, loadError = null) }
        camera.analysisExecutor.execute {
            detector?.close(); detector = null
            // a pending measured-default task ran ahead of this one on the same executor, so the
            // runtime / unit read here are the settled ones
            val u = _ui.value
            val det = Detector.open(model, u.runtime, u.compute, threads = tuning.threads, ctx = getApplication())
            if (det == null) {
                _ui.update { it.copy(loadingModel = false, running = false, wantRun = false, loadError = Detector.lastError) }
                return@execute
            }
            detector = det
            seq = 0; loopFrames = 0; loopWindowStart = 0
            openHint()
            running.set(true)
            _ui.update {
                it.copy(
                    loadingModel = false, running = true, isSeg = det.isSeg, classNames = det.classNames,
                    backend = det.activeBackend, backendNote = det.backendNote, placement = det.placement,
                )
            }
        }
    }

    /** `suspendLoop()`: stop, wipe the overlay and every stat; the thermal reading stays. */
    fun suspendLoop() {
        running.set(false)
        camera.analysisExecutor.execute { closeHint(); detector?.close(); detector = null }
        _ui.update { it.copy(running = false, loadingModel = false) }
        _frame.value = FrameResult(frameSize = _frame.value.frameSize)
    }

    /** Threads picker changed: reload the model with the new thread policy if running. */
    fun applyThreads() { if (running.get() || _ui.value.loadingModel) { suspendLoop(); startLoop() } }

    // ---- diagnostics: thermal headroom + prime-core clock, sampled once a second ----------------
    /**
     * [clusters]: one entry per CPU cluster (descending max clock), "cur/max MHz".
     * [inferCore]: the CPU the "ym-infer" thread last ran on (so a thread that never reaches the
     * prime cores is visible), -1 when unknown.
     */
    data class Diag(val headroom: Float = -1f, val primeMHz: Int = -1, val clusters: List<String> = emptyList(), val inferCore: Int = -1, val primeCores: String = "")
    private val _diag = MutableStateFlow(Diag())
    val diag: StateFlow<Diag> = _diag
    private val pm = app.getSystemService(android.content.Context.POWER_SERVICE) as android.os.PowerManager
    private fun readInt(path: String): Int = try { val f = java.io.File(path); if (f.canRead()) f.readText().trim().toInt() else -1 } catch (_: Throwable) { -1 }
    private val diagJob = viewModelScope.launch(Dispatchers.IO) {
        val cpus = java.lang.Runtime.getRuntime().availableProcessors()
        // cluster = group of cpus sharing cpuinfo_max_freq, represented by its first cpu
        val maxOf = (0 until cpus).map { it to readInt("/sys/devices/system/cpu/cpu$it/cpufreq/cpuinfo_max_freq") }
        val clusters = maxOf.filter { it.second > 0 }.groupBy { it.second }.entries.sortedByDescending { it.key }
            .map { e -> Triple(e.value.first().first, e.key / 1000, e.value.map { it.first }) }
        val primeCores = clusters.firstOrNull()?.third?.joinToString(",") ?: ""
        val primeCpu = clusters.firstOrNull()?.first ?: (cpus - 1)
        while (true) {
            val head = if (android.os.Build.VERSION.SDK_INT >= 30) try { pm.getThermalHeadroom(0) } catch (_: Throwable) { Float.NaN } else Float.NaN
            val cur = clusters.map { (cpu, maxMHz, _) -> "${readInt("/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq") / 1000}/$maxMHz" }
            val prime = readInt("/sys/devices/system/cpu/cpu$primeCpu/cpufreq/scaling_cur_freq") / 1000
            // where is the inference thread running? /proc/self/task/<tid>/stat field 39 = last cpu
            var core = -1
            try {
                java.io.File("/proc/self/task").listFiles()?.forEach { t ->
                    if (core < 0 && java.io.File(t, "comm").readText().trim() == "ym-infer") {
                        val stat = java.io.File(t, "stat").readText()
                        val fields = stat.substring(stat.lastIndexOf(')') + 2).split(" ")
                        core = fields.getOrNull(36)?.toIntOrNull() ?: -1   // field 39 overall = index 36 after the comm
                    }
                }
            } catch (_: Throwable) {}
            _diag.value = Diag(if (head.isNaN()) -1f else head, prime, cur, core, primeCores)
            kotlinx.coroutines.delay(1000)
        }
    }

    /** Tab hidden / app backgrounded: stop but keep the intent. */
    fun onHidden() { if (running.get() || _ui.value.loadingModel) suspendLoop() }
    fun onShown() { if (_ui.value.wantRun && !running.get()) startLoop() }

    // ---- the loop: runs inside analyze() on "ym-infer" -----------------------------------------

    override fun analyze(image: ImageProxy) {
        val det = detector
        if (!running.get() || det == null) { image.close(); return }
        val t0 = SystemClock.elapsedRealtimeNanos()
        camera.noteFrame(image)
        try {
            val raw = det.forward(image)
            image.close()
            val t1 = SystemClock.elapsedRealtimeNanos()
            val conf = tuning.conf; val iou = tuning.iou
            val dets = raw.decode(conf, iou, 300)
            val t2 = SystemClock.elapsedRealtimeNanos()
            var mask: Bitmap? = null
            if (det.isSeg && tuning.segOverlay != SegOverlayMode.Boxes && dets.isNotEmpty()) {
                val slot = ringIdx; ringIdx = (ringIdx + 1) % maskRing.size
                mask = raw.maskOverlay(dets.take(100), maxSide = 640, reuse = maskRing[slot])
                if (mask != null) maskRing[slot] = mask
            }
            val t3 = SystemClock.elapsedRealtimeNanos()
            val fwdWall = (t1 - t0) / 1e6
            val pre = max(0.0, fwdWall - raw.inferMs - raw.decodeMs)
            val inf = raw.inferMs
            val dec = raw.decodeMs + (t2 - t1) / 1e6
            val maskMs = (t3 - t2) / 1e6
            val size = Size(raw.origW, raw.origH)
            raw.close()
            // loop rate over a 1 s window
            val now = SystemClock.elapsedRealtime()
            if (loopWindowStart == 0L) loopWindowStart = now
            loopFrames++
            if (now - loopWindowStart >= 1000) { loopHz = loopFrames * 1000.0 / (now - loopWindowStart); loopFrames = 0; loopWindowStart = now }
            if (running.get()) _frame.value = FrameResult(dets, mask, size, pre, inf, dec, maskMs, loopHz, ++seq)
        } catch (t: Throwable) {
            Log.w(TAG, "frame failed: ${t.message}")
            try { image.close() } catch (_: Throwable) {}
        }
        // ~30 fps pacing: the next delivered frame is then the freshest.
        val spentNs = SystemClock.elapsedRealtimeNanos() - t0
        if (android.os.Build.VERSION.SDK_INT >= 31) try { hint?.reportActualWorkDuration(spentNs) } catch (_: Throwable) {}
        val spent = spentNs / 1_000_000
        if (spent < 33) SystemClock.sleep(33 - spent)
    }

    // ---- shutter ----------------------------------------------------------------------------------

    /**
     * `captureFrame()`: full-res still cropped to the preview aspect, overlay baked, saved to the
     * gallery. [onDone] gets true on success. Re-arms no sooner than 1.0 s after the tap.
     */
    fun capture(onDone: (Boolean) -> Unit) {
        if (!shutterArmed.compareAndSet(true, false)) return
        val t0 = SystemClock.elapsedRealtime()
        val snap = _frame.value
        val ui = _ui.value
        _ui.value = ui.copy(capturing = true)
        val drawBoxes = !(ui.isSeg && tuning.segOverlay == SegOverlayMode.Masks)
        val drawMask = ui.isSeg && tuning.segOverlay != SegOverlayMode.Boxes
        val style = tuning.style.kitStyle
        val names = ui.classNames
        camera.capturePhoto { proxy ->
            viewModelScope.launch(Dispatchers.Default) {
                var ok = false
                try {
                    val bmp = proxy?.use { decodeCropped(it, snap.frameSize) }
                    if (bmp != null) {
                        val s = bmp.height.toFloat() / snap.frameSize.height
                        val scaled = snap.dets.map { d -> d.copy(x1 = d.x1 * s, y1 = d.y1 * s, x2 = d.x2 * s, y2 = d.y2 * s) }
                        val canvas = Canvas(bmp)
                        if (drawMask && snap.mask != null) {
                            canvas.drawBitmap(snap.mask, null, Rect(0, 0, bmp.width, bmp.height), Paint(Paint.FILTER_BITMAP_FLAG))
                        }
                        Annotate.draw(canvas, bmp.width, scaled, names, style, drawBoxes)
                        ok = GallerySaver.save(getApplication(), bmp)
                        bmp.recycle()
                    }
                } catch (t: Throwable) { Log.w(TAG, "capture compose failed", t) }
                // re-arm >= 1.0 s after the tap (the shutter sound is about that long)
                val wait = 1000 - (SystemClock.elapsedRealtime() - t0)
                if (wait > 0) kotlinx.coroutines.delay(wait)
                _ui.update { it.copy(capturing = false) }
                shutterArmed.set(true)
                withContext(Dispatchers.Main) { onDone(ok) }
            }
        }
    }

    /** Decode the JPEG still, rotated upright, center-cropped to the analysis frame's aspect. */
    private fun decodeCropped(proxy: ImageProxy, frame: Size): Bitmap? {
        val buf = proxy.planes[0].buffer
        val bytes = ByteArray(buf.remaining()).also { buf.get(it) }
        val rot = proxy.imageInfo.rotationDegrees
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        val w = bounds.outWidth; val h = bounds.outHeight
        if (w <= 0 || h <= 0) return null
        // crop in the UNROTATED image so that after rotation it has the upright frame's aspect
        val uprightAspect = frame.width.toFloat() / frame.height
        val cropAspect = if (rot == 90 || rot == 270) 1f / uprightAspect else uprightAspect
        var cw = w; var ch = (w / cropAspect).roundToInt()
        if (ch > h) { ch = h; cw = (h * cropAspect).roundToInt() }
        val rect = Rect((w - cw) / 2, (h - ch) / 2, (w - cw) / 2 + cw, (h - ch) / 2 + ch)
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val decoder = BitmapRegionDecoder.newInstance(bytes, 0, bytes.size, false) ?: return null
        var region = try { decoder.decodeRegion(rect, opts) } catch (oom: OutOfMemoryError) {
            decoder.decodeRegion(rect, BitmapFactory.Options().apply { inSampleSize = 2; inPreferredConfig = Bitmap.Config.ARGB_8888 })
        } finally { decoder.recycle() }
        region ?: return null
        if (rot != 0) {
            val m = Matrix().apply { postRotate(rot.toFloat()) }
            val r = Bitmap.createBitmap(region, 0, 0, region.width, region.height, m, true)
            region.recycle(); region = r
        }
        return if (region.isMutable) region else region.copy(Bitmap.Config.ARGB_8888, true).also { region.recycle() }
    }

    override fun onCleared() {
        running.set(false)
        camera.analysisExecutor.execute { closeHint(); detector?.close(); detector = null }
        camera.stop()
        thermal.stop()
    }

    companion object { private const val TAG = "LiveVM" }
}
