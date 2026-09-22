package dev.yolomaster.app.ui.live

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.CameraCaptureSession
import android.os.Build
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** What the Live UI observes from the camera. */
data class CamState(
    val bound: Boolean = false,
    val zoom: Float = 1f,
    val minZoom: Float = 1f,
    val maxZoom: Float = 1f,
    val lensStops: List<Float> = listOf(1f),
    val torchOn: Boolean = false,
    val hasTorch: Boolean = false,
    val cameraHz: Double = 0.0,
    val frameAgeMs: Double = 0.0,
    val ageReliable: Boolean = true,
    val frameSize: Size = Size(1280, 720),   // upright analysis frame size
)

/**
 * `CameraController.swift` on CameraX: Preview + ImageAnalysis (RGBA, keep-only-latest) +
 * ImageCapture bound as one group sharing the PreviewView's viewport; native lens stops from
 * the physical cameras; pinch zoom; tap-to-focus with a 5 s revert; torch; sensor delivery rate
 * and frame-age diagnostics.
 */
@OptIn(ExperimentalCamera2Interop::class)
class CameraController(private val context: Context) {
    val previewView: PreviewView = PreviewView(context).apply {
        scaleType = PreviewView.ScaleType.FILL_CENTER   // == AVLayerVideoGravity.resizeAspectFill
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
    }
    private val _state = MutableStateFlow(CamState())
    val state: StateFlow<CamState> = _state

    /** Single inference thread: `analyze()` runs the model, so frames are dropped, never queued. */
    val analysisExecutor: Executor = Executors.newSingleThreadExecutor { r -> Thread(r, "ym-infer").apply { priority = Thread.MAX_PRIORITY - 1 } }
    private val mainExecutor: Executor get() = ContextCompat.getMainExecutor(context)

    private var provider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null
    private var analysis: ImageAnalysis? = null
    private var zoomBias = 1f
    private var torchIntent = false
    /** Lighter streams for the clamp experiment (set before start/rebind). */
    @Volatile var lite: Boolean = false
    private var lastOwner: LifecycleOwner? = null
    private var lastAnalyzer: ImageAnalysis.Analyzer? = null

    // delivery-rate window (sensor frames, counted in the capture callback)
    @Volatile private var sensorFrames = 0
    @Volatile private var windowStart = 0L
    private var ageEma = -1.0
    private var timestampsRealtime = true

    /** Bind the use cases; [analyzer] receives every delivered frame on [analysisExecutor]. */
    fun start(owner: LifecycleOwner, analyzer: ImageAnalysis.Analyzer) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val p = try { future.get() } catch (t: Throwable) { Log.e(TAG, "camera provider", t); return@addListener }
            provider = p
            bind(p, owner, analyzer)
        }, mainExecutor)
    }

    fun stop() {
        try { provider?.unbindAll() } catch (_: Throwable) {}
        camera = null; imageCapture = null; analysis = null
        _state.value = _state.value.copy(bound = false, cameraHz = 0.0, frameAgeMs = 0.0)
    }

    @SuppressLint("RestrictedApi")
    /** Re-bind with the current [lite] setting (used by the tuning panel switch). */
    fun rebind() { val p = provider ?: return; val o = lastOwner ?: return; val a = lastAnalyzer ?: return; mainExecutor.execute { bind(p, o, a) } }

    private fun bind(p: ProcessCameraProvider, owner: LifecycleOwner, analyzer: ImageAnalysis.Analyzer) {
        lastOwner = owner; lastAnalyzer = analyzer
        p.unbindAll()
        val ratio16x9 = AspectRatioStrategy(AspectRatio.RATIO_16_9, AspectRatioStrategy.FALLBACK_RULE_AUTO)
        val analysisSize = if (lite) Size(640, 480) else Size(1280, 720)
        val fpsRange = if (lite) Range(15, 30) else Range(30, 30)
        val ratioStrategy = if (lite) AspectRatioStrategy(AspectRatio.RATIO_4_3, AspectRatioStrategy.FALLBACK_RULE_AUTO) else ratio16x9
        val previewBuilder = Preview.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder().setAspectRatioStrategy(ratio16x9)
                    .setResolutionStrategy(ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build(),
            )
        // The iOS app turns video HDR off for latency; here the equivalents are the HAL's per-frame
        // stabilization / noise reduction / edge enhancement, which burn CPU+GPU the model needs.
        Camera2Interop.Extender(previewBuilder)
            .setCaptureRequestOption(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
            .setCaptureRequestOption(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_FAST)
            .setCaptureRequestOption(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_FAST)
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange)
        val preview = previewBuilder.build()
        val analysisBuilder = ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder().setAspectRatioStrategy(ratioStrategy)
                    .setResolutionStrategy(ResolutionStrategy(analysisSize, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build(),
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setTargetRotation(previewView.display?.rotation ?: android.view.Surface.ROTATION_0)
        // Lock 30 fps and count sensor frames (dropped frames never reach analyze()).
        Camera2Interop.Extender(analysisBuilder)
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange)
            .setCaptureRequestOption(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
            .setCaptureRequestOption(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_FAST)
            .setCaptureRequestOption(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_FAST)
            .setSessionCaptureCallback(object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                    val now = SystemClock.elapsedRealtime()
                    if (windowStart == 0L) windowStart = now
                    sensorFrames++
                    val dt = now - windowStart
                    if (dt >= 1000) {
                        val hz = sensorFrames * 1000.0 / dt
                        sensorFrames = 0; windowStart = now
                        _state.value = _state.value.copy(cameraHz = hz)
                    }
                }
            })
        val analysisUse = analysisBuilder.build().also { it.setAnalyzer(analysisExecutor, analyzer) }

        val capture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setResolutionSelector(
                ResolutionSelector.Builder().setAspectRatioStrategy(ratio16x9)
                    .setResolutionFilter { sizes, _ ->
                        // largest 16:9-ish size at or under 25 MP (iOS caps at 25 MP too)
                        val ok = sizes.filter { it.width.toLong() * it.height <= 25_000_000L }
                        ok.sortedByDescending { it.width.toLong() * it.height }.ifEmpty { sizes }
                    }.build(),
            ).build()

        val group = UseCaseGroup.Builder().addUseCase(preview).addUseCase(analysisUse).addUseCase(capture)
            .also { b -> previewView.viewPort?.let { b.setViewPort(it) } }.build()
        try {
            val cam = p.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, group)
            camera = cam; imageCapture = capture; analysis = analysisUse
            preview.setSurfaceProvider(previewView.surfaceProvider)
            readCharacteristics(cam)
            cam.cameraInfo.zoomState.observe(owner) { z ->
                _state.value = _state.value.copy(zoom = z.zoomRatio, minZoom = z.minZoomRatio, maxZoom = z.maxZoomRatio)
            }
            cam.cameraInfo.torchState.observe(owner) { t -> _state.value = _state.value.copy(torchOn = t == androidx.camera.core.TorchState.ON) }
            if (torchIntent) cam.cameraControl.enableTorch(true)
            _state.value = _state.value.copy(bound = true, hasTorch = cam.cameraInfo.hasFlashUnit())
        } catch (t: Throwable) {
            Log.e(TAG, "bind failed", t)
        }
    }

    /** Lens stops from the physical cameras' focal lengths (`CameraController.swift:165-176`). */
    private fun readCharacteristics(cam: Camera) {
        val stops = ArrayList<Float>()
        try {
            val info = Camera2CameraInfo.from(cam.cameraInfo)
            val cm = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val logical = cm.getCameraCharacteristics(info.cameraId)
            timestampsRealtime = logical.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
            val zs = cam.cameraInfo.zoomState.value
            val minZ = zs?.minZoomRatio ?: 1f; val maxZ = zs?.maxZoomRatio ?: 1f
            if (Build.VERSION.SDK_INT >= 28) {
                val ids = logical.physicalCameraIds
                val fovs = ArrayList<Pair<String, Float>>()
                for (id in ids) {
                    val c = cm.getCameraCharacteristics(id)
                    val fl = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: continue
                    val sz = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: continue
                    if (c.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_BACK) continue
                    fovs += id to (sz.width / fl)
                }
                Log.i(TAG, "logical=${info.cameraId} physical=$ids fovs=$fovs")
                if (fovs.isNotEmpty()) {
                    val logicalFl = logical.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
                    val logicalSz = logical.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    val wideFov = if (logicalFl != null && logicalSz != null) logicalSz.width / logicalFl else fovs.map { it.second }.sorted()[fovs.size / 2]
                    for ((_, fov) in fovs) {
                        val stop = ((wideFov / fov) * 10f).roundToInt() / 10f
                        if (stop in minZ..maxZ) stops += stop
                    }
                }
            }
            if (minZ < 0.95f && stops.none { it < 0.95f }) stops += (minZ * 10f).roundToInt() / 10f
            stops += 1f
            _state.value = _state.value.copy(
                lensStops = stops.filter { it <= maxZ + 1e-3f }.map { (it * 10f).roundToInt() / 10f }.distinct().sorted(),
                ageReliable = timestampsRealtime,
            )
        } catch (t: Throwable) {
            Log.w(TAG, "characteristics", t)
            _state.value = _state.value.copy(lensStops = listOf(1f))
        }
    }

    /** Pinch/lens zoom; clamp to `[min, min(max, 16)]` like iOS. Returns the applied ratio. */
    fun setZoom(ratio: Float): Float {
        val s = _state.value
        val r = ratio.coerceIn(s.minZoom, min(s.maxZoom, 16f))
        camera?.cameraControl?.setZoomRatio(r)
        return r
    }

    /** Tap-to-focus: AF+AE at the tap; CameraX reverts to continuous after 5 s (iOS parity). */
    fun focus(x: Float, y: Float) {
        val cam = camera ?: return
        val point = previewView.meteringPointFactory.createPoint(x, y)
        val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
            .setAutoCancelDuration(5, TimeUnit.SECONDS).build()
        cam.cameraControl.startFocusAndMetering(action)
    }

    fun setTorch(on: Boolean) { torchIntent = on; camera?.cameraControl?.enableTorch(on) }

    /** Full-resolution still; the callback receives a JPEG [ImageProxy] (close it) or null. */
    fun capturePhoto(cb: (ImageProxy?) -> Unit) {
        val ic = imageCapture ?: return cb(null)
        ic.takePicture(mainExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) = cb(image)
            override fun onError(exception: ImageCaptureException) { Log.w(TAG, "capture failed", exception); cb(null) }
        })
    }

    /** Called by the analyzer for each delivered frame: frame-age EMA (0.9/0.1) and size. */
    fun noteFrame(image: ImageProxy) {
        val age = (SystemClock.elapsedRealtimeNanos() - image.imageInfo.timestamp) / 1e6
        ageEma = if (ageEma < 0) age else 0.9 * ageEma + 0.1 * age
        val rot = image.imageInfo.rotationDegrees
        val w = image.cropRect.width(); val h = image.cropRect.height()
        val upright = if (rot == 90 || rot == 270) Size(h, w) else Size(w, h)
        val s = _state.value
        if (abs(s.frameAgeMs - ageEma) > 0.5 || s.frameSize != upright) _state.value = s.copy(frameAgeMs = max(0.0, ageEma), frameSize = upright)
    }

    /** `displayZoom`: the ratio shown in the lens capsule (CameraX ratios are already display-relative). */
    val displayZoom: Float get() = _state.value.zoom / zoomBias

    companion object {
        private const val TAG = "CameraController"
        /** `zoomLabel` (`LiveView.swift:248-251`): "1x", "2x", "0.5x", "2.4x". */
        fun zoomLabel(z: Float): String {
            val r = (z * 10f).roundToInt() / 10f
            return if (abs(r - r.roundToInt()) < 1e-3f) "${r.roundToInt()}x" else "${r}x"
        }
    }
}
