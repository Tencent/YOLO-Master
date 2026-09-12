package dev.yolomaster.app.ui.photo

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.CropPortrait
import androidx.compose.material.icons.filled.GridOn
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.model.runtimesAvailable
import dev.yolomaster.app.system.Haptics
import dev.yolomaster.app.system.ImageLoader
import dev.yolomaster.app.system.Prefs
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
import dev.yolomaster.app.ui.hud.ExportToast
import dev.yolomaster.app.ui.hud.HudStats
import dev.yolomaster.app.ui.hud.StatsHud
import dev.yolomaster.app.ui.hud.Toast
import dev.yolomaster.app.ui.overlay.ScreenOverlay
import dev.yolomaster.app.ui.overlay.SegOverlayMode
import dev.yolomaster.app.ui.theme.IosRed
import dev.yolomaster.app.ui.theme.IosType
import dev.yolomaster.app.ui.theme.LocalIosColors
import dev.yolomaster.app.system.ThermalMonitor
import androidx.compose.runtime.DisposableEffect
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.min

/** `PhotoTestView.swift`: picker, gallery / pager, per-image and batch HUD, export toast. */
@Composable
fun PhotoScreen(vm: PhotoViewModel = viewModel()) {
    val ios = LocalIosColors.current
    val ctx = LocalContext.current
    val view = LocalView.current
    val haptics = remember { Haptics(view) }
    val haze = rememberHazeState()
    val ui by vm.ui.collectAsStateWithLifecycle()
    val allowCPU by rememberBoolPref(Prefs.ALLOW_CPU, true)
    val thermal = remember { ThermalMonitor(ctx) }
    val thermalState by thermal.state.collectAsStateWithLifecycle()
    DisposableEffect(Unit) { thermal.start(); onDispose { thermal.stop() } }

    var showTuning by remember { mutableStateOf(false) }
    var showHud by remember { mutableStateOf(vm.tuning.showHUD) }
    var toast by remember { mutableStateOf<Toast?>(null) }
    var toastGen by remember { mutableStateOf(0) }

    // Settings toggle: CPU hidden -> ncnn snaps back to GPU; ONNX has no GPU, its CPU EP stays.
    LaunchedEffect(allowCPU) { if (!allowCPU && ui.runtime == Runtime.NCNN && ui.compute == ComputeChoice.CPU && ui.selected?.cpuOnly != true) vm.selectCompute(ComputeChoice.GPU) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(100)) { uris ->
        if (uris.isNotEmpty()) vm.load(uris.take(100))
    }

    fun showToast(msg: String, ok: Boolean) { toastGen++; toast = Toast(msg, ok, gen = toastGen); if (ok) haptics.success() else haptics.error() }

    CompositionLocalProvider(LocalHazeState provides haze) {
        Box(Modifier.fillMaxSize().background(ios.systemBackground)) {
            Column(Modifier.fillMaxSize().statusBarsPadding()) {
                Spacer(Modifier.height(54.dp))
                // only the photo content is a blur source; the cards below it are haze children
                Box(Modifier.weight(1f).fillMaxWidth().hazeBackdrop(haze)) {
                    when {
                        ui.items.isEmpty() -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Icon(Icons.Filled.PhotoLibrary, null, tint = ios.secondaryLabel, modifier = Modifier.size(44.dp))
                                Text("No photos", style = IosType.title3Bold, color = ios.label)
                                Text("Pick up to 100 images to run the detector on.", style = IosType.footnote, color = ios.secondaryLabel, textAlign = TextAlign.Center)
                            }
                        }
                        ui.viewMode == ViewMode.Gallery -> Gallery(vm, ui) { i -> vm.setPage(i); vm.setViewMode(ViewMode.Pager) }
                        else -> Pager(vm, ui) { vm.setViewMode(ViewMode.Gallery) }
                    }
                }
                if (ui.phase != Phase.Idle || ui.measuring) ProgressCard(ui, Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
                AnimatedVisibility(showTuning) {
                    TuningPanel(vm.tuning, isSeg = ui.isSeg, onChange = { showHud = vm.tuning.showHUD; vm.retune() }, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
                }
                if (showHud) {
                    val perImage = ui.viewMode == ViewMode.Pager && ui.items.getOrNull(ui.page)?.done == true
                    val item = ui.items.getOrNull(ui.page)
                    val stats = if (perImage && item != null) {
                        val e2e = item.pre + item.inf + item.dec + item.mask
                        HudStats(
                            active = true, fps = e2e, mode = DialMode.MS, fpsLabel = "ms",
                            pre = item.pre, inf = item.inf, dec = item.dec, mask = item.mask, isSeg = ui.isSeg, dets = item.dets.size,
                            thermalLevel = thermalState.level, thermalKnown = thermalState.known,
                            extras = listOf(
                                "image" to "${ui.page + 1}/${ui.items.size}", "size" to "${item.width}x${item.height}",
                                "file" to item.name, "type" to item.type, "e2e" to String.format("%.1f ms", e2e), "dets" to "${item.dets.size}",
                            ),
                        )
                    } else HudStats(
                        active = ui.statInf > 0, fps = ui.throughput, mode = DialMode.FPS, fpsLabel = "FPS",
                        pre = ui.statPre, inf = ui.statInf, dec = ui.statDec, mask = ui.statMask, isSeg = ui.isSeg,
                        dets = item?.dets?.size ?: 0, thermalLevel = thermalState.level, thermalKnown = thermalState.known,
                        extras = if (ui.statInf > 0) listOf(
                            "images" to "${ui.items.size}", "wall" to String.format("%.2f s", ui.wallSeconds),
                            "total dets" to "${ui.items.sumOf { it.dets.size }}", "avg e2e" to String.format("%.1f ms", ui.statPre + ui.statInf + ui.statDec + ui.statMask),
                            "runtime" to ui.runtime.label, "backend" to ui.backend,
                        ) + (if (ui.runtime == Runtime.ONNX) listOf("placement" to "${ui.placement.onAccelerator}/${ui.placement.total} HTP") else emptyList())
                        else emptyList(),
                    )
                    StatsHud(stats, fullWidth = true, modifier = Modifier.padding(horizontal = 8.dp).padding(bottom = 20.dp))
                }
                if (ui.error.isNotEmpty()) Text(ui.error, style = IosType.caption2, color = IosRed, maxLines = 3, modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp))
            }

            // top bar
            Column(Modifier.align(Alignment.TopCenter).statusBarsPadding().fillMaxWidth()) {
                Row(
                    Modifier.padding(horizontal = 8.dp).materialCard(12.dp).padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    val pickersOn = !ui.isRunning && !ui.measuring
                    ModelMenu(ui.models, ui.selected, enabled = pickersOn) { vm.selectModel(it) }
                    // runtime left of the unit; hidden while the model has a single runtime (nothing to pick)
                    val runtimes = runtimesAvailable(ui.selected, ui.caps)
                    if (runtimes.size > 1) RuntimeMenu(runtimes, ui.runtime, enabled = pickersOn) { vm.selectRuntime(it) }
                    ComputeMenu(ComputeChoice.available(ui.runtime, allowCPU, ui.selected, ui.caps), ui.compute, enabled = pickersOn) { vm.selectCompute(it) }
                    Spacer(Modifier.weight(1f))
                    if (ui.items.isNotEmpty()) {
                        BorderedIconButton(icon = if (ui.viewMode == ViewMode.Gallery) Icons.Filled.CropPortrait else Icons.Filled.GridOn, onClick = {
                            haptics.light(); vm.setViewMode(if (ui.viewMode == ViewMode.Gallery) ViewMode.Pager else ViewMode.Gallery)
                        })
                        BorderedIconButton(icon = Icons.Filled.SaveAlt, enabled = !ui.exporting && !ui.isRunning, onClick = {
                            haptics.light()
                            val idx = if (ui.viewMode == ViewMode.Pager) listOf(ui.page) else ui.items.indices.toList()
                            vm.export(idx) { n -> if (n < 0) showToast("Could not save to Photos", false) else showToast(if (n == 1) "Saved 1 image to Photos" else "Saved $n images to Photos", true) }
                        }, content = if (ui.exporting) ({ CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp) }) else null)
                    }
                    BorderedIconButton(icon = Icons.Filled.Tune, onClick = { showTuning = !showTuning })
                    ProminentIconButton(icon = Icons.Filled.AddPhotoAlternate, enabled = !ui.isRunning, onClick = {
                        picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                    })
                }
            }

            toast?.let { t ->
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    AnimatedVisibility(visible = true, enter = scaleIn(initialScale = 0.82f) + fadeIn(), exit = scaleOut(targetScale = 0.82f) + fadeOut()) {
                        ExportToast(t) { toast = null }
                    }
                }
            }
        }
    }
}

@Composable
private fun ProgressCard(ui: PhotoUi, modifier: Modifier = Modifier) {
    val ios = LocalIosColors.current
    Row(modifier.fillMaxWidth().materialCard(12.dp).padding(10.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        if (ui.phase == Phase.LoadingModel || ui.measuring) {
            CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp, color = ios.label)
            Text("Loading the model to ${ui.compute.label}${if (ui.measuring) " (measuring)" else ""}", style = IosType.caption, color = ios.label)
            Spacer(Modifier.weight(1f))
        } else {
            Text(if (ui.phase == Phase.Loading) "Loading / iCloud" else "Inference", style = IosType.caption, color = ios.label)
            LinearProgressIndicator(progress = { if (ui.progressTotal > 0) ui.progress.toFloat() / ui.progressTotal else 0f }, modifier = Modifier.weight(1f), color = ios.accent, trackColor = ios.quaternaryFill)
            Text("${ui.progress}/${ui.progressTotal}", style = IosType.caption.tabular, color = ios.label)
        }
    }
}

/** 3-up square grid (`PhotoTestView.swift:162-171, 189-223`): aspect-fill image + mask + boxes scaled by width. */
@Composable
private fun Gallery(vm: PhotoViewModel, ui: PhotoUi, onOpen: (Int) -> Unit) {
    val density = LocalDensity.current.density
    LazyVerticalGrid(columns = GridCells.Fixed(3), verticalArrangement = Arrangement.spacedBy(4.dp), horizontalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.fillMaxSize().padding(horizontal = 8.dp)) {
        itemsIndexed(ui.items, key = { i, it -> "${it.uri}#$i" }) { i, item ->
            val version = ui.version
            var mask by remember(item, version) { mutableStateOf<Bitmap?>(null) }
            LaunchedEffect(item, version) { if (vm.tuning.segOverlay != SegOverlayMode.Boxes) mask = withContext(Dispatchers.IO) { item.maskBitmap() } }
            Box(Modifier.aspectRatio(1f).clip(RoundedCornerShape(8.dp)).clickable { onOpen(i) }) {
                Image(item.thumb.asImageBitmap(), null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                mask?.let { Image(it.asImageBitmap(), null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
                val drawBoxes = !(ui.isSeg && vm.tuning.segOverlay == SegOverlayMode.Masks)
                if (drawBoxes) Canvas(Modifier.fillMaxSize()) {
                    // aspect-fill square crop: same mapping as the Image's ContentScale.Crop
                    val scale = max(size.width / item.width, size.height / item.height)
                    val ox = (size.width - item.width * scale) / 2f; val oy = (size.height - item.height * scale) / 2f
                    drawIntoCanvas { c -> ScreenOverlay.draw(c.nativeCanvas, item.dets, ui.classNames, vm.tuning.style, scale, ox, oy, density) }
                }
            }
        }
    }
}

/** Pager (`PhotoTestView.swift:172-186, 316-340`) with Photos-app zoom (1..5x, double-tap, pinch-out exits). */
@Composable
private fun Pager(vm: PhotoViewModel, ui: PhotoUi, onExit: () -> Unit) {
    val ctx = LocalContext.current
    val density = LocalDensity.current.density
    val state = rememberPagerState(initialPage = ui.page.coerceIn(0, max(0, ui.items.size - 1))) { ui.items.size }
    LaunchedEffect(state.currentPage) { vm.setPage(state.currentPage) }
    HorizontalPager(state = state, modifier = Modifier.fillMaxSize(), beyondBoundsPageCount = 1) { page ->
        val item = ui.items[page]
        val version = ui.version
        var full by remember(item) { mutableStateOf<Bitmap?>(null) }
        var mask by remember(item, version) { mutableStateOf<Bitmap?>(null) }
        LaunchedEffect(item) { full = withContext(Dispatchers.IO) { ImageLoader.load(ctx, item.uri, page, maxSide = 2048)?.bitmap } }
        LaunchedEffect(item, version) { mask = if (vm.tuning.segOverlay != SegOverlayMode.Boxes) withContext(Dispatchers.IO) { item.maskBitmap() } else null }
        var zoom by remember(item) { mutableFloatStateOf(1f) }
        var pan by remember(item) { mutableStateOf(Offset.Zero) }
        Box(
            Modifier.fillMaxSize().padding(horizontal = 8.dp)
                .pointerInput(item) {
                    detectTransformGestures { centroid, dragAmount, gestureZoom, _ ->
                        val nz = (zoom * gestureZoom)
                        if (nz < 0.85f && zoom <= 1f) { onExit(); return@detectTransformGestures }
                        zoom = nz.coerceIn(1f, 5f)
                        pan = if (zoom > 1f) pan + dragAmount else Offset.Zero
                    }
                }
                .pointerInput(item) {
                    detectTapGestures(onDoubleTap = { p ->
                        if (zoom > 1.3f) { zoom = 1f; pan = Offset.Zero } else { zoom = 2.5f; pan = Offset((size.width / 2f - p.x) * 1.5f, (size.height / 2f - p.y) * 1.5f) }
                    })
                },
            contentAlignment = Alignment.Center,
        ) {
            val bmp = full ?: item.thumb
            Box(
                Modifier.aspectRatio(item.width.toFloat() / item.height).fillMaxWidth().clip(RoundedCornerShape(12.dp))
                    .graphicsLayer { scaleX = zoom; scaleY = zoom; translationX = pan.x; translationY = pan.y; transformOrigin = TransformOrigin.Center },
            ) {
                Image(bmp.asImageBitmap(), null, contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize())
                mask?.let { Image(it.asImageBitmap(), null, contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize()) }
                val drawBoxes = !(ui.isSeg && vm.tuning.segOverlay == SegOverlayMode.Masks)
                if (drawBoxes) Canvas(Modifier.fillMaxSize()) {
                    val scale = min(size.width / item.width, size.height / item.height)
                    val ox = (size.width - item.width * scale) / 2f; val oy = (size.height - item.height * scale) / 2f
                    drawIntoCanvas { c -> ScreenOverlay.draw(c.nativeCanvas, item.dets, ui.classNames, vm.tuning.style, scale, ox, oy, density) }
                }
            }
        }
    }
}
