package dev.yolomaster.app.ui.photo

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.yolomaster.app.YoloMasterApp
import dev.yolomaster.app.detect.DefaultPolicy
import dev.yolomaster.app.detect.Detector
import dev.yolomaster.app.model.BundledModel
import dev.yolomaster.app.model.Caps
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.system.GallerySaver
import dev.yolomaster.app.system.ImageLoader
import dev.yolomaster.app.system.Prefs
import dev.yolomaster.app.ui.common.Tuning
import dev.yolomaster.app.ui.overlay.Annotate
import dev.yolomaster.app.ui.overlay.SegOverlayMode
import dev.yolomaster.ncnn.Detection
import dev.yolomaster.ncnn.Placement
import dev.yolomaster.ncnn.RawOutput
import dev.yolomaster.ncnn.Runtime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import kotlin.math.max

enum class Phase { Idle, Loading, LoadingModel, Inference }
enum class ViewMode { Gallery, Pager }

/** One picked image: a thumbnail for the grid, the mask as PNG bytes, stats, and the cached raw. */
class PhotoItem(
    val uri: Uri,
    val name: String,
    val type: String,
    val width: Int,
    val height: Int,
    val thumb: Bitmap,                 // 384 px RGB_565
) {
    @Volatile var dets: List<Detection> = emptyList()
    @Volatile var maskPng: ByteArray? = null
    @Volatile var maskW: Int = 0
    @Volatile var maskH: Int = 0
    @Volatile var pre = 0.0; @Volatile var inf = 0.0; @Volatile var dec = 0.0; @Volatile var mask = 0.0
    @Volatile var done = false
    @Volatile var raw: RawOutput? = null
    fun maskBitmap(): Bitmap? = maskPng?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
}

data class PhotoUi(
    val models: List<BundledModel> = emptyList(),
    /** Runtime capability bits of this build/device (probed once, off main); 0 until known. */
    val caps: Int = 0,
    val selected: BundledModel? = null,
    val runtime: Runtime = Runtime.NCNN,
    val compute: ComputeChoice = ComputeChoice.GPU,
    /** The measured-default mini-bench is running for [selected]; `run()` waits for its verdict. */
    val measuring: Boolean = false,
    val items: List<PhotoItem> = emptyList(),
    val version: Int = 0,               // bumped whenever per-item results change
    val phase: Phase = Phase.Idle,
    val progress: Int = 0,
    val progressTotal: Int = 0,
    val viewMode: ViewMode = ViewMode.Gallery,
    val page: Int = 0,
    val isSeg: Boolean = false,
    val classNames: List<String> = emptyList(),
    /** What the last run resolved to (HUD rows): backend string and, for ONNX, the graph placement. */
    val backend: String = "",
    val placement: Placement = Placement(0, 0),
    val statPre: Double = 0.0, val statInf: Double = 0.0, val statDec: Double = 0.0, val statMask: Double = 0.0,
    val throughput: Double = 0.0,
    val wallSeconds: Double = 0.0,
    val exporting: Boolean = false,
    val error: String = "",
) {
    val isRunning: Boolean get() = phase != Phase.Idle
}

/**
 * `PhotoTestView.swift` state + `load` / `run` / `retune` / `export`. Raws are cached (native)
 * for batches of at most 40 (20 for segmentation) so the conf/IoU sliders re-decode instead of
 * re-running the model.
 */
class PhotoViewModel(app: Application) : AndroidViewModel(app) {
    private val catalog = YoloMasterApp.from(app).catalog
    val tuning = Tuning()
    private val _ui = MutableStateFlow(PhotoUi(runtime = Prefs.runtime(app)))
    val ui: StateFlow<PhotoUi> = _ui
    private val infer = Executors.newSingleThreadExecutor { r -> Thread(r, "ym-photo") }
    private var runJob: Job? = null
    private var retuneGen = 0

    init {
        viewModelScope.launch {
            // the capability probe dlopens the QNN runtime: off main, together with the asset copy
            val (models, caps) = withContext(Dispatchers.IO) { catalog.discover() to Caps.device }
            _ui.update { it.copy(models = models, caps = caps) }
            BundledModel.preferred(models)?.let { applySelection(it) }
        }
    }

    fun refreshModels() = viewModelScope.launch {
        val models = withContext(Dispatchers.IO) { catalog.discover(force = true) }
        _ui.update { it.copy(models = models) }
        val keep = _ui.value.selected?.let { s -> models.firstOrNull { it.id == s.id } } ?: BundledModel.preferred(models)
        if (keep != null) applySelection(keep) else _ui.update { it.copy(selected = null) }
    }

    /**
     * Select [m] and settle its runtime x unit: the user's remembered pick, else the cached measured
     * default, else measure now off main (the progress card says "(measuring)"); a loaded batch
     * re-runs once the verdict is in.
     */
    private fun applySelection(m: BundledModel) {
        val app = getApplication<Application>()
        val allow = Prefs.allowCPU(app)
        val caps = _ui.value.caps
        val cached = DefaultPolicy.resolveCached(app, m, allow, caps)
        if (cached != null) {
            _ui.update { it.copy(selected = m, runtime = cached.first, compute = cached.second) }
            if (_ui.value.items.isNotEmpty()) run()
            return
        }
        runJob?.cancel()
        _ui.update { it.copy(selected = m, measuring = true, phase = Phase.Idle) }
        viewModelScope.launch(Dispatchers.IO) {
            val pick = try { DefaultPolicy.resolve(app, m, allow, caps) } catch (t: Throwable) {
                Log.w(TAG, "measured default failed for ${m.id}: ${t.message}"); Runtime.NCNN to ComputeChoice.CPU
            }
            _ui.update { if (it.selected?.id == m.id) it.copy(runtime = pick.first, compute = pick.second, measuring = false) else it.copy(measuring = false) }
            if (_ui.value.selected?.id == m.id && _ui.value.items.isNotEmpty()) run()
        }
    }

    fun selectModel(m: BundledModel) = applySelection(m)

    /** A hand pick is remembered per model and beats the measured default from then on. */
    fun selectRuntime(r: Runtime) {
        val app = getApplication<Application>()
        val u = _ui.value
        if (r == u.runtime) return
        val units = ComputeChoice.available(r, Prefs.allowCPU(app), u.selected, u.caps)
        val c = if (u.compute in units) u.compute else units.first()
        Prefs.setRuntime(app, r)
        u.selected?.let { Prefs.setChoice(app, it.id, r, c) }
        _ui.update { it.copy(runtime = r, compute = c) }
        if (_ui.value.items.isNotEmpty()) run()
    }

    fun selectCompute(c: ComputeChoice) {
        val u = _ui.value
        if (c == u.compute) return
        u.selected?.let { Prefs.setChoice(getApplication(), it.id, u.runtime, c) }
        _ui.update { it.copy(compute = c) }
        if (_ui.value.items.isNotEmpty()) run()
    }

    fun setViewMode(m: ViewMode) { _ui.update { it.copy(viewMode = m) } }
    fun setPage(p: Int) { _ui.update { it.copy(page = p) } }

    // ---- load ---------------------------------------------------------------------------------------

    fun load(uris: List<Uri>) {
        if (uris.isEmpty()) return
        runJob?.cancel()
        clearItems()
        _ui.update { it.copy(phase = Phase.Loading, progress = 0, progressTotal = uris.size, error = "") }
        runJob = viewModelScope.launch(Dispatchers.IO) {
            val items = ArrayList<PhotoItem>()
            uris.forEachIndexed { i, uri ->
                val li = ImageLoader.load(getApplication(), uri, i, maxSide = 2048)
                if (li != null) {
                    items += PhotoItem(uri, li.name, li.type, li.bitmap.width, li.bitmap.height, thumbnail(li.bitmap))
                    li.bitmap.recycle()
                }
                _ui.update { it.copy(progress = i + 1) }
            }
            if (items.isEmpty()) { _ui.update { it.copy(phase = Phase.Idle, error = "no loadable photos") }; return@launch }
            _ui.update { it.copy(items = items, page = 0, viewMode = if (items.size > 1) ViewMode.Gallery else ViewMode.Pager, version = it.version + 1, phase = Phase.Idle) }
            run()
        }
    }

    private fun thumbnail(src: Bitmap): Bitmap {
        val s = 384f / max(src.width, src.height)
        val w = max(1, (src.width * s).toInt()); val h = max(1, (src.height * s).toInt())
        val t = Bitmap.createScaledBitmap(src, w, h, true)
        return if (t.config == Bitmap.Config.RGB_565) t else t.copy(Bitmap.Config.RGB_565, false).also { if (t !== src) t.recycle() }
    }

    private fun clearItems() {
        for (it in _ui.value.items) { it.raw?.close(); it.raw = null }
        _ui.update { it.copy(items = emptyList(), page = 0, statInf = 0.0, throughput = 0.0) }
    }

    // ---- run -----------------------------------------------------------------------------------------

    /** Full pass over every item; publishes incrementally so the gallery fills in live. */
    fun run() {
        val model = _ui.value.selected ?: return
        val items = _ui.value.items
        if (items.isEmpty()) return
        // the measured default is still deciding the runtime x unit: its completion re-enters run()
        if (_ui.value.measuring) return
        runJob?.cancel()
        runJob = viewModelScope.launch(Dispatchers.IO) {
            _ui.update { it.copy(phase = Phase.LoadingModel, error = "") }
            val u = _ui.value
            val det = Detector.open(model, u.runtime, u.compute, ctx = getApplication())
            if (det == null) { _ui.update { it.copy(phase = Phase.Idle, error = "ERROR: ${Detector.lastError}") }; return@launch }
            try {
                val cacheRaws = items.size <= (if (det.isSeg) 20 else 40)
                _ui.update {
                    it.copy(
                        phase = Phase.Inference, progress = 0, progressTotal = items.size, isSeg = det.isSeg, classNames = det.classNames,
                        backend = det.activeBackend, placement = det.placement,
                    )
                }
                val t0 = SystemClock.elapsedRealtime()
                var sPre = 0.0; var sInf = 0.0; var sDec = 0.0; var sMask = 0.0
                items.forEachIndexed { i, item ->
                    if (!isActive) return@forEachIndexed
                    val li = ImageLoader.load(getApplication(), item.uri, i, maxSide = 2048) ?: return@forEachIndexed
                    val ta = SystemClock.elapsedRealtimeNanos()
                    val raw = det.forward(li.bitmap)
                    val tb = SystemClock.elapsedRealtimeNanos()
                    val dets = raw.decode(tuning.conf, tuning.iou, 300)
                    val tc = SystemClock.elapsedRealtimeNanos()
                    var maskMs = 0.0
                    if (det.isSeg) {
                        val m = raw.maskOverlay(dets, maxSide = 1024)
                        maskMs = (SystemClock.elapsedRealtimeNanos() - tc) / 1e6
                        storeMask(item, m)
                    }
                    item.pre = max(0.0, (tb - ta) / 1e6 - raw.inferMs - raw.decodeMs)
                    item.inf = raw.inferMs
                    item.dec = raw.decodeMs + (tc - tb) / 1e6
                    item.mask = maskMs
                    item.dets = dets
                    item.done = true
                    item.raw?.close(); item.raw = if (cacheRaws) raw else { raw.close(); null }
                    li.bitmap.recycle()
                    sPre += item.pre; sInf += item.inf; sDec += item.dec; sMask += item.mask
                    _ui.update { it.copy(progress = i + 1, version = it.version + 1) }
                }
                val wall = (SystemClock.elapsedRealtime() - t0) / 1000.0
                val n = items.count { it.done }.coerceAtLeast(1)
                _ui.update {
                    it.copy(
                        phase = Phase.Idle, statPre = sPre / n, statInf = sInf / n, statDec = sDec / n, statMask = sMask / n,
                        throughput = if (wall > 0) n / wall else 0.0, wallSeconds = wall, version = it.version + 1,
                    )
                }
            } catch (t: Throwable) {
                Log.w(TAG, "run failed", t)
                _ui.update { it.copy(phase = Phase.Idle, error = "ERROR: ${t.message}") }
            } finally { det.close() }
        }
    }

    private fun storeMask(item: PhotoItem, m: Bitmap?) {
        if (m == null) { item.maskPng = null; item.maskW = 0; item.maskH = 0; return }
        val bos = ByteArrayOutputStream()
        m.compress(Bitmap.CompressFormat.PNG, 100, bos)
        item.maskPng = bos.toByteArray(); item.maskW = m.width; item.maskH = m.height
        m.recycle()
    }

    private val kotlinx.coroutines.CoroutineScope.isActive: Boolean get() = coroutineContext[Job]?.isActive != false

    // ---- retune ---------------------------------------------------------------------------------------

    /** Sliders moved: re-decode cached raws only (no re-inference); masks re-render off-main. */
    fun retune() {
        val items = _ui.value.items
        if (items.isEmpty() || _ui.value.isRunning) return
        val gen = ++retuneGen
        val conf = tuning.conf; val iou = tuning.iou
        infer.execute {
            for (item in items) {
                if (gen != retuneGen) return@execute
                val raw = item.raw ?: continue
                try {
                    val dets = raw.decode(conf, iou, 300)
                    item.dets = dets
                    if (_ui.value.isSeg) storeMask(item, raw.maskOverlay(dets, maxSide = 1024))
                } catch (t: Throwable) { Log.w(TAG, "retune failed", t) }
            }
            if (gen == retuneGen) _ui.update { it.copy(version = it.version + 1) }
        }
    }

    // ---- export ---------------------------------------------------------------------------------------

    /** Exports the current page (pager) or every image (gallery); returns the count saved or -1. */
    fun export(indices: List<Int>, onDone: (Int) -> Unit) {
        if (_ui.value.exporting) return
        _ui.update { it.copy(exporting = true) }
        val drawMask = _ui.value.isSeg && tuning.segOverlay != SegOverlayMode.Boxes
        val drawBoxes = !(_ui.value.isSeg && tuning.segOverlay == SegOverlayMode.Masks)
        val style = tuning.style.kitStyle
        val names = _ui.value.classNames
        viewModelScope.launch(Dispatchers.IO) {
            var saved = 0
            var failed = false
            for (i in indices) {
                val item = _ui.value.items.getOrNull(i) ?: continue
                try {
                    val li = ImageLoader.load(getApplication(), item.uri, i, maxSide = 2048) ?: continue
                    val bmp = if (li.bitmap.isMutable) li.bitmap else li.bitmap.copy(Bitmap.Config.ARGB_8888, true).also { li.bitmap.recycle() }
                    val c = Canvas(bmp)
                    if (drawMask) item.maskBitmap()?.let { m -> c.drawBitmap(m, null, Rect(0, 0, bmp.width, bmp.height), Paint(Paint.FILTER_BITMAP_FLAG)); m.recycle() }
                    Annotate.draw(c, bmp.width, item.dets, names, style, drawBoxes)
                    if (GallerySaver.save(getApplication(), bmp, "YM_${item.name}")) saved++ else failed = true
                    bmp.recycle()
                } catch (t: Throwable) { failed = true; Log.w(TAG, "export failed", t) }
            }
            _ui.update { it.copy(exporting = false) }
            withContext(Dispatchers.Main) { onDone(if (failed && saved == 0) -1 else saved) }
        }
    }

    override fun onCleared() {
        runJob?.cancel()
        for (it in _ui.value.items) { it.raw?.close() }
        infer.shutdown()
    }

    companion object { private const val TAG = "PhotoVM" }
}
