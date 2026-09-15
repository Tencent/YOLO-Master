package dev.yolomaster.app.ui.live

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FlashlightOff
import androidx.compose.material.icons.filled.FlashlightOn
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.model.runtimesAvailable
import dev.yolomaster.app.system.Haptics
import dev.yolomaster.app.system.Prefs
import dev.yolomaster.app.system.ShutterSound
import dev.yolomaster.app.system.rememberBoolPref
import dev.yolomaster.app.ui.common.BorderedIconButton
import dev.yolomaster.app.ui.common.ComputeMenu
import dev.yolomaster.app.ui.common.LocalHazeState
import dev.yolomaster.app.ui.common.ModelMenu
import dev.yolomaster.app.ui.common.ProminentIconButton
import dev.yolomaster.app.ui.common.RuntimeMenu
import dev.yolomaster.app.ui.common.TuningPanel
import dev.yolomaster.ncnn.Runtime
import dev.yolomaster.app.ui.common.hazeBackdrop
import dev.yolomaster.app.ui.common.materialCard
import dev.yolomaster.app.ui.common.rememberHazeState
import dev.yolomaster.app.ui.common.tabular
import dev.yolomaster.app.ui.hud.DialMode
import dev.yolomaster.app.ui.hud.HudStats
import dev.yolomaster.app.ui.hud.StatsHud
import dev.yolomaster.app.ui.overlay.ScreenOverlay
import dev.yolomaster.app.ui.overlay.SegOverlayMode
import dev.yolomaster.app.ui.theme.IosRed
import dev.yolomaster.app.ui.theme.IosType
import dev.yolomaster.app.ui.theme.IosYellow
import dev.yolomaster.app.ui.theme.LocalIosColors
import androidx.compose.runtime.CompositionLocalProvider
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.max

/**
 * `LiveView.swift`: preview + async overlay in one coordinate space, focus square, shutter
 * flash, loading/initializing cards, lens capsule, shutter, HUD, and the floating top bar.
 */
@Composable
fun LiveScreen(vm: LiveViewModel = viewModel()) {
    val ios = LocalIosColors.current
    val ctx = LocalContext.current
    val view = LocalView.current
    val density = LocalDensity.current.density
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    val haptics = remember { Haptics(view) }
    val shutter = remember { ShutterSound() }
    val haze = rememberHazeState()

    val ui by vm.ui.collectAsStateWithLifecycle()
    val frame by vm.frame.collectAsStateWithLifecycle()
    val cam by vm.camera.state.collectAsStateWithLifecycle()
    val thermal by vm.thermal.state.collectAsStateWithLifecycle()
    val diag by vm.diag.collectAsStateWithLifecycle()
    val allowCPU by rememberBoolPref(Prefs.ALLOW_CPU, true)

    var hasCamera by remember { mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasCamera = it }
    var showTuning by remember { mutableStateOf(false) }
    var tapPoint by remember { mutableStateOf<Offset?>(null) }
    var flash by remember { mutableStateOf(false) }
    var lensExpanded by remember { mutableStateOf(false) }
    var lensRect by remember { mutableStateOf<androidx.compose.ui.geometry.Rect?>(null) }
    var lensTimer by remember { mutableStateOf<Job?>(null) }
    var longPressFired by remember { mutableStateOf(false) }
    var zoomBase by remember { mutableStateOf(1f) }
    var showHud by remember { mutableStateOf(vm.tuning.showHUD) }

    // Settings toggle: CPU hidden -> ncnn snaps back to GPU (LiveView.swift:204-206); ONNX has no GPU, its CPU EP stays.
    LaunchedEffect(allowCPU) { if (!allowCPU && ui.runtime == Runtime.NCNN && ui.compute == ComputeChoice.CPU && ui.selected?.cpuOnly != true) vm.selectCompute(ComputeChoice.GPU) }
    LaunchedEffect(Unit) { if (!hasCamera) permission.launch(Manifest.permission.CAMERA) }

    // Camera + loop lifecycle: bind while visible, suspend when hidden (onDisappear / scenePhase).
    DisposableEffect(hasCamera, lifecycleOwner) {
        val observer = LifecycleEventObserver { _, e ->
            when (e) {
                Lifecycle.Event.ON_START -> { if (hasCamera) { vm.camera.start(lifecycleOwner, vm); vm.onShown() } }
                Lifecycle.Event.ON_STOP -> { vm.onHidden(); vm.camera.stop() }
                else -> {}
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        if (hasCamera && lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) { vm.camera.start(lifecycleOwner, vm); vm.onShown() }
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer); vm.onHidden(); vm.camera.stop(); shutter.release() }
    }

    fun bumpLensTimer() {
        lensTimer?.cancel()
        lensTimer = scope.launch { delay(3000); lensExpanded = false }
    }

    // Live: NO real-time backdrop blur. Re-blurring the 30 fps camera feed behind three cards
    // runs on the same GPU as ncnn Vulkan and was the prime suspect for 4x slower inference on
    // the S26; the cards use the translucent scrim here (blur stays on the static tabs).
    CompositionLocalProvider(LocalHazeState provides null) {
        Box(Modifier.fillMaxSize().background(Color.Black)) {
            if (!hasCamera) {
                EmptyState("Camera access required", Icons.Filled.Videocam, null)
            } else {
                // preview + overlay share one full-bleed container so boxes never shear
                Box(
                    Modifier.fillMaxSize()
                        .pointerInput(cam.minZoom, cam.maxZoom) {
                            detectTransformGestures { _, _, zoom, _ ->
                                if (zoom != 1f) { zoomBase = vm.camera.setZoom(zoomBase * zoom) }
                            }
                        }
                        .pointerInput(lensRect) {
                            detectTapGestures { p ->
                                val dead = lensRect?.let { androidx.compose.ui.geometry.Rect(it.left - 28 * density, it.top - 28 * density, it.right + 28 * density, it.bottom + 28 * density) }
                                if (dead != null && dead.contains(p)) return@detectTapGestures
                                vm.camera.focus(p.x, p.y)
                                tapPoint = p
                                scope.launch { delay(900); tapPoint = null }
                            }
                        },
                ) {
                    AndroidView(factory = { vm.camera.previewView }, modifier = Modifier.fillMaxSize())
                    LaunchedEffect(cam.zoom) { zoomBase = cam.zoom }
                    // overlay: resizeAspectFill mapping (LiveView.swift:381-404)
                    val drawBoxes = !(ui.isSeg && vm.tuning.segOverlay == SegOverlayMode.Masks)
                    val style = vm.tuning.style
                    Canvas(Modifier.fillMaxSize()) {
                        val fw = frame.frameSize.width.toFloat(); val fh = frame.frameSize.height.toFloat()
                        if (fw <= 0 || fh <= 0) return@Canvas
                        val scale = max(size.width / fw, size.height / fh)
                        val ox = (size.width - fw * scale) / 2f
                        val oy = (size.height - fh * scale) / 2f
                        drawIntoCanvas { c ->
                            val nc = c.nativeCanvas
                            frame.mask?.let { m ->
                                if (!m.isRecycled) nc.drawBitmap(m, null, android.graphics.RectF(ox, oy, ox + fw * scale, oy + fh * scale), maskPaint)
                            }
                            if (drawBoxes) ScreenOverlay.draw(nc, frame.dets, ui.classNames, style, scale, ox, oy, density)
                        }
                    }
                }
                // focus square
                tapPoint?.let { p ->
                    Box(
                        Modifier.offset { androidx.compose.ui.unit.IntOffset((p.x - 36 * density).toInt(), (p.y - 36 * density).toInt()) }
                            .size(72.dp).border(1.5.dp, IosYellow, RoundedCornerShape(3.dp)),
                    )
                }
                // shutter flash
                val flashAlpha by animateFloatAsState(if (flash) 1f else 0f, tween(if (flash) 0 else 200), label = "flash")
                if (flashAlpha > 0f) Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = flashAlpha)))

                // centered cards
                when {
                    ui.initializing -> CenterCard { Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) { CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp, color = ios.label); Text("Initializing...", style = IosType.caption, color = ios.label) } }
                    ui.models.isEmpty() -> EmptyState("No models bundled", Icons.Filled.Videocam, "Copy ncnn model folders into android/app/src/main/assets/models/, rebuild.")
                    ui.loadingModel || ui.measuring -> CenterCard { Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) { CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp, color = ios.label); Text("Loading the model to ${ui.compute.label}${if (ui.measuring) " (measuring)" else ""}", style = IosType.caption, color = ios.label) } }
                    ui.loadError != null -> CenterCard { Text("ERROR: ${ui.loadError}", style = IosType.caption, color = IosRed) }
                }

                // bottom stack: lens, shutter, HUD
                Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
                    LensControl(
                        stops = cam.lensStops, zoom = vm.camera.displayZoom, expanded = lensExpanded,
                        onExpand = { lensExpanded = true; bumpLensTimer() },
                        onStop = { f -> vm.camera.setZoom(f); zoomBase = f; bumpLensTimer() },
                        modifier = Modifier.padding(bottom = 6.dp).onGloballyPositioned { c ->
                            val pos = c.positionInRoot(); lensRect = androidx.compose.ui.geometry.Rect(pos.x, pos.y, pos.x + c.size.width, pos.y + c.size.height)
                        },
                    )
                    ShutterButton(capturing = ui.capturing, modifier = Modifier.padding(bottom = 8.dp)) {
                        shutter.play(); haptics.medium()
                        flash = true; scope.launch { delay(80); flash = false }
                        vm.capture { }
                    }
                    if (showHud) {
                        val e2e = frame.pre + frame.inf + frame.dec + frame.maskMs
                        val active = ui.running && !ui.loadingModel && frame.seq > 0
                        StatsHud(
                            HudStats(
                                active = active, fps = if (active) 1000.0 / max(e2e, 0.1) else 0.0, mode = DialMode.FPS, fpsLabel = "FPS",
                                pre = frame.pre, inf = frame.inf, dec = frame.dec, mask = frame.maskMs, isSeg = ui.isSeg, dets = frame.dets.size,
                                // headroom >= 0.9 = the SoC is clamping clocks (S26: prime cores at ~1.45 of
                                // 4.74 GHz with the camera open): show it as the red tachometer state.
                                thermalLevel = if (diag.headroom >= 0.9f) 3 else thermal.level, thermalKnown = thermal.known || diag.headroom >= 0f,
                                extras = if (active) listOf(
                                    "camera" to String.format("%.1f fps", cam.cameraHz),
                                    "loop" to String.format("%.1f fps", frame.loopHz),
                                    "frame age" to if (cam.ageReliable) String.format("%.0f ms", cam.frameAgeMs) else "--",
                                    "runtime" to ui.runtime.label,
                                    "backend" to ui.backend,
                                ) + (if (ui.runtime == Runtime.ONNX) listOf("placement" to "${ui.placement.onAccelerator}/${ui.placement.total} HTP") else emptyList()) + listOf(
                                    "threads" to "${vm.tuning.threads.let { if (it == 0) "all" else it }}",
                                    "headroom" to if (diag.headroom >= 0f) String.format("%.2f", diag.headroom) else "--",
                                    "prime clk" to if (diag.primeMHz > 0) "${diag.primeMHz} MHz" else "--",
                                    "throttled" to if (diag.headroom >= 0.9f) "yes" else "no",
                                    "clusters" to diag.clusters.joinToString(" "),
                                    "infer core" to if (diag.inferCore >= 0) "cpu${diag.inferCore} (prime: ${diag.primeCores})" else "--",
                                ) else listOf(
                                    // paused: keep the SoC diagnostics visible so the camera-alone clamp can be seen
                                    "camera" to String.format("%.1f fps", cam.cameraHz),
                                    "headroom" to if (diag.headroom >= 0f) String.format("%.2f", diag.headroom) else "--",
                                    "prime clk" to if (diag.primeMHz > 0) "${diag.primeMHz} MHz" else "--",
                                    "throttled" to if (diag.headroom >= 0.9f) "yes" else "no",
                                    "clusters" to diag.clusters.joinToString(" "),
                                ),
                            ),
                            modifier = Modifier.padding(bottom = 20.dp),
                        )
                    }
                }
            }

            // top bar (safe-area inset; no top padding so it lines up with Photo)
            Column(Modifier.align(Alignment.TopCenter).statusBarsPadding().fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(
                    Modifier.padding(horizontal = 16.dp).materialCard(12.dp).padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    val pickersOn = !ui.wantRun && !ui.measuring
                    ModelMenu(ui.models, ui.selected, enabled = pickersOn) { vm.selectModel(it) }
                    // runtime left of the unit; hidden while the model has a single runtime (nothing to pick)
                    val runtimes = runtimesAvailable(ui.selected, ui.caps)
                    if (runtimes.size > 1) RuntimeMenu(runtimes, ui.runtime, enabled = pickersOn) { vm.selectRuntime(it) }
                    ComputeMenu(ComputeChoice.available(ui.runtime, allowCPU, ui.selected, ui.caps), ui.compute, enabled = pickersOn) { vm.selectCompute(it) }
                    Spacer(Modifier.weight(1f))
                    BorderedIconButton(
                        icon = if (cam.torchOn) Icons.Filled.FlashlightOn else Icons.Filled.FlashlightOff,
                        onClick = { haptics.light(); vm.camera.setTorch(!cam.torchOn) },
                        tint = if (cam.torchOn) IosYellow else null, enabled = cam.hasTorch,
                    )
                    BorderedIconButton(icon = Icons.Filled.Tune, onClick = { showTuning = !showTuning })
                    // play/pause: a plain combined-click surface (a Material Button would swallow the
                    // tap before an outer gesture box saw it). Long press >= 0.4 s = heavy haptic, tap swallowed.
                    val playEnabled = ui.selected != null
                    val playTint = if (ui.wantRun) IosRed else ios.accent
                    Box(
                        Modifier.height(34.dp).widthIn(min = 44.dp).clip(RoundedCornerShape(8.dp))
                            .background(if (playEnabled) playTint else ios.quaternaryFill)
                            .combinedClickable(
                                enabled = playEnabled,
                                onLongClick = { longPressFired = true; haptics.heavy(); scope.launch { delay(600); longPressFired = false } },
                                onClick = { if (!longPressFired) { haptics.medium(); vm.togglePlay() } },
                            )
                            .padding(horizontal = 10.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(
                            if (ui.wantRun) Icons.Filled.Pause else Icons.Filled.PlayArrow, null,
                            tint = if (playEnabled) Color.White else ios.tertiaryLabel, modifier = Modifier.size(18.dp),
                        )
                    }
                }
                AnimatedVisibility(showTuning, enter = fadeIn(), exit = fadeOut()) {
                    TuningPanel(vm.tuning, isSeg = ui.isSeg, onChange = { showHud = vm.tuning.showHUD }, modifier = Modifier.padding(horizontal = 16.dp), showThreads = true, onThreads = { vm.applyThreads() }, onCamLite = { vm.camera.lite = it; vm.camera.rebind() })
                }
            }
        }
    }
}

private val maskPaint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)

@Composable
private fun CenterCard(content: @Composable () -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Box(Modifier.materialCard(12.dp).padding(14.dp)) { content() }
    }
}

@Composable
private fun EmptyState(title: String, icon: androidx.compose.ui.graphics.vector.ImageVector, description: String?) {
    val ios = LocalIosColors.current
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(32.dp)) {
            Icon(icon, null, tint = ios.secondaryLabel, modifier = Modifier.size(44.dp))
            Text(title, style = IosType.title3Bold, color = ios.label)
            if (description != null) Text(description, style = IosType.footnote, color = ios.secondaryLabel, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        }
    }
}

/** `lensControl` (`LiveView.swift:261-308`): collapsed = live zoom label; expanded = one button per stop. */
@Composable
private fun LensControl(stops: List<Float>, zoom: Float, expanded: Boolean, onExpand: () -> Unit, onStop: (Float) -> Unit, modifier: Modifier = Modifier) {
    val ios = LocalIosColors.current
    val active = stops.lastOrNull { it <= zoom + 0.05f } ?: stops.firstOrNull() ?: 1f
    Row(
        modifier.materialCard(CircleShape).padding(horizontal = if (expanded) 8.dp else 0.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically,
    ) {
        if (!expanded) {
            Text(
                CameraController.zoomLabel(zoom), style = IosType.captionBold.tabular, color = ios.label,
                modifier = Modifier.clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) { onExpand() }.padding(horizontal = 10.dp, vertical = 5.dp),
            )
        } else {
            for (f in stops) {
                val isActive = f == active
                Text(
                    if (isActive) CameraController.zoomLabel(zoom) else CameraController.zoomLabel(f),
                    style = IosType.captionBold.tabular, color = if (isActive) IosYellow else ios.label,
                    modifier = Modifier.widthIn(min = 34.dp).clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) { onStop(f) }.padding(vertical = 6.dp),
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                )
            }
        }
    }
}

/** The shutter (`LiveView.swift:160-176`): 58 ring, 46 disc, press scale 0.85. */
@Composable
private fun ShutterButton(capturing: Boolean, modifier: Modifier = Modifier, onClick: () -> Unit) {
    var pressed by remember { mutableStateOf(false) }
    val s by animateFloatAsState(if (pressed) 0.85f else 1f, tween(120), label = "shutter")
    Box(
        modifier.size(58.dp).scale(s)
            .pointerInput(capturing) {
                detectTapGestures(onPress = { pressed = true; tryAwaitRelease(); pressed = false }, onTap = { if (!capturing) onClick() })
            },
        contentAlignment = Alignment.Center,
    ) {
        Box(Modifier.size(58.dp).border(3.dp, Color.White.copy(alpha = 0.9f), CircleShape))
        if (capturing) CircularProgressIndicator(Modifier.size(28.dp), color = Color.White, strokeWidth = 2.5.dp)
        else Box(Modifier.size(46.dp).clip(CircleShape).background(Color.White))
    }
}
