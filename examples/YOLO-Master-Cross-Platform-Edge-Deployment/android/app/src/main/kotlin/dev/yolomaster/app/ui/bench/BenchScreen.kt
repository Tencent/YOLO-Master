package dev.yolomaster.app.ui.bench

import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.LocalFireDepartment
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PauseCircle
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Remove
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.system.Haptics
import dev.yolomaster.app.ui.common.BorderedIconButton
import dev.yolomaster.app.ui.common.LocalHazeState
import dev.yolomaster.app.ui.common.ModelMenu
import dev.yolomaster.app.ui.common.PresetButton
import dev.yolomaster.app.ui.common.ProminentIconButton
import dev.yolomaster.app.ui.common.Segmented
import dev.yolomaster.app.ui.common.hazeBackdrop
import dev.yolomaster.app.ui.common.materialCard
import dev.yolomaster.app.ui.common.rememberHazeState
import dev.yolomaster.app.ui.common.tabular
import dev.yolomaster.app.ui.hud.DetailBar
import dev.yolomaster.app.ui.hud.ExportToast
import dev.yolomaster.app.ui.hud.FpsDial
import dev.yolomaster.app.ui.hud.HudColors
import dev.yolomaster.app.ui.hud.Sparkline
import dev.yolomaster.app.ui.hud.StageIcons
import dev.yolomaster.app.ui.hud.ThermalBar
import dev.yolomaster.app.ui.hud.ThermalTach
import dev.yolomaster.app.ui.hud.Toast
import dev.yolomaster.app.ui.theme.IosBlue
import dev.yolomaster.app.ui.theme.IosGreen
import dev.yolomaster.app.ui.theme.IosOrange
import dev.yolomaster.app.ui.theme.IosRed
import dev.yolomaster.app.ui.theme.IosType
import dev.yolomaster.app.ui.theme.LocalIosColors
import kotlin.math.max
import kotlin.math.roundToInt

/*
 * `BenchView.swift`: the on-device benchmark in the HUD design language - tachometer dials,
 * colour-ramp bars, the thermal tach, a live throttle sparkline, haptics and the export toast.
 */

/** Unit icons (`benchUnitIcon`, `BenchView.swift:84-90`): GPU = layered squares, NPU = bolt, CPU = chip. */
internal fun unitIcon(compute: String): ImageVector = when (compute) {
    ComputeChoice.GPU.label -> Icons.Filled.Layers
    ComputeChoice.NPU.label -> Icons.Filled.Bolt
    else -> Icons.Filled.Memory
}

internal fun fmt1(v: Double): String = String.format(java.util.Locale.US, "%.1f", v)

@Composable
fun BenchScreen(vm: BenchViewModel = viewModel()) {
    val ios = LocalIosColors.current
    val ctx = LocalContext.current
    val view = LocalView.current
    val density = LocalDensity.current
    val haptics = remember { Haptics(view) }
    val haze = rememberHazeState()

    val ui by vm.ui.collectAsStateWithLifecycle()
    val thermal by vm.thermal.state.collectAsStateWithLifecycle()

    var showHistory by remember { mutableStateOf(false) }
    var toast by remember { mutableStateOf<Toast?>(null) }
    var toastGen by remember { mutableIntStateOf(0) }
    var barHeightPx by remember { mutableIntStateOf(0) }

    // The bench never competes with another tab's inference: hidden (ON_STOP) or disposed (tab
    // switch) -> auto-pause (`onDisappear` / `scenePhase`, BenchView.swift:199-200).
    LifecycleEventEffect(Lifecycle.Event.ON_STOP) { vm.onHidden() }
    LifecycleEventEffect(Lifecycle.Event.ON_START) { vm.onShown() }
    DisposableEffect(Unit) { onDispose { vm.onHidden() } }
    LaunchedEffect(Unit) {
        vm.events.collect { e ->
            when (e) {
                BenchEvent.Started -> haptics.medium()
                BenchEvent.Cell -> haptics.light()
                BenchEvent.Finished -> haptics.success()
                is BenchEvent.Failed -> haptics.error()
            }
        }
    }

    fun exportCSV() {
        val ok = BenchShare.shareCsv(ctx, "benchmark.csv", vm.resultsCSV())
        haptics.light(); if (ok) haptics.success() else haptics.error()
        toast = Toast(if (ok) "benchmark.csv ready" else "Export failed", ok, autoDismissMs = 1800, gen = ++toastGen)
    }

    CompositionLocalProvider(LocalHazeState provides haze) {
        Box(Modifier.fillMaxSize()) {
            // the scrolling list is the backdrop the floating top bar blurs over
            Box(Modifier.fillMaxSize().hazeBackdrop(haze)) {
                if (ui.results.isEmpty() && !ui.running) {
                    EmptyBench()
                } else {
                    // cards inside the backdrop use the tinted scrim: a haze child cannot blur itself
                    CompositionLocalProvider(LocalHazeState provides null) {
                        Column(
                            Modifier.fillMaxSize().verticalScroll(rememberScrollState())
                                .padding(top = with(density) { barHeightPx.toDp() } + 10.dp, bottom = 24.dp)
                                .padding(horizontal = 10.dp),
                            verticalArrangement = Arrangement.spacedBy(10.dp),
                        ) {
                            if (ui.running) ProgressCard(ui, thermal.level, HudColors.thermalColor(thermal.level, thermal.known))
                            if (ui.mode == BenchMode.Sustained && ui.sparkSamples.size > 1) SustainedGraphCard(ui)
                            if (ui.mode == BenchMode.Cold) ui.fastest?.let { HeroCard(it) }
                            for (mid in ui.modelsWithResults) ModelCard(mid, ui) { vm.toggleExpanded(mid) }
                        }
                    }
                }
            }
            // top bar (`safeAreaInset(edge: .top)`): padding 8, radius 12, horizontal 10
            Column(
                Modifier.fillMaxWidth().statusBarsPadding().padding(horizontal = 10.dp)
                    .onSizeChanged { barHeightPx = it.height },
            ) {
                TopBar(ui, haptics, vm, onHistory = { showHistory = true }, onExport = ::exportCSV)
                ui.error?.let { Text(it, style = IosType.caption2, color = IosRed, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)) }
            }
            toast?.let { t ->
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { ExportToast(t) { if (toast?.gen == t.gen) toast = null } }
            }
        }
    }

    if (showHistory) HistorySheet(vm.history, onDismiss = { showHistory = false })
}

/** `ContentUnavailableView("No benchmarks yet", ...)`, vertically centred like the Photo tab. */
@Composable
private fun EmptyBench() {
    val ios = LocalIosColors.current
    Column(Modifier.fillMaxSize().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Icon(Icons.Filled.Speed, null, tint = ios.secondaryLabel, modifier = Modifier.size(48.dp))
        Spacer(Modifier.height(10.dp))
        Text("No benchmarks yet", style = IosType.title3Bold, color = ios.label)
        Spacer(Modifier.height(4.dp))
        Text("Run a cold pass across every model, runtime and compute unit.", style = IosType.subheadline, color = ios.secondaryLabel)
    }
}

// ---- top bar ---------------------------------------------------------------------------------

@Composable
private fun TopBar(ui: BenchUi, haptics: Haptics, vm: BenchViewModel, onHistory: () -> Unit, onExport: () -> Unit) {
    Column(Modifier.fillMaxWidth().materialCard(12.dp).padding(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Segmented(BenchMode.entries, ui.mode, { it.label }, enabled = !ui.running, modifier = Modifier.weight(1f)) {
                if (it != ui.mode) { haptics.light(); vm.setMode(it) }
            }
            BorderedIconButton(Icons.Filled.Tune, onClick = { vm.toggleAdvanced() })
            BorderedIconButton(Icons.Filled.History, onClick = onHistory)
            if (ui.results.isNotEmpty()) BorderedIconButton(Icons.Filled.Share, onClick = onExport, enabled = !ui.running)
            if (ui.running) {
                ProminentIconButton(
                    if (ui.paused) Icons.Filled.PlayArrow else Icons.Filled.Pause,
                    onClick = { haptics.light(); vm.togglePause() },
                    tint = if (ui.paused) null else IosOrange, minWidth = 48.dp,
                )
                BorderedIconButton(Icons.Filled.Close, onClick = { vm.stop() }, tint = IosRed)
            } else {
                ProminentIconButton(Icons.Filled.PlayArrow, onClick = { vm.start() }, enabled = ui.canStart, minWidth = 48.dp)
            }
        }
        // sustained target + duration are ALWAYS visible in Sustained mode; the sliders button
        // holds the advanced iters/warmup (BenchView.swift:254-257)
        if (ui.mode == BenchMode.Sustained) SustainedConfig(ui, haptics, vm)
        if (ui.showAdvanced) AdvancedCard(ui, vm)
    }
}

/** A nested card inside the top bar (iOS nests a second material; a haze child cannot nest). */
@Composable
private fun Modifier.innerCard(): Modifier {
    val ios = LocalIosColors.current
    return this.background(ios.quaternaryFill.copy(alpha = 0.12f), RoundedCornerShape(12.dp)).padding(10.dp)
}

/** `sustainedConfig` (`BenchView.swift:265-306`). */
@Composable
private fun SustainedConfig(ui: BenchUi, haptics: Haptics, vm: BenchViewModel) {
    val enabled = !ui.running
    Column(Modifier.fillMaxWidth().innerCard(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ModelMenu(ui.models, ui.selectedModel, enabled = enabled) { vm.selectModel(it) }
            // runtime first (hidden for single-runtime models), then its units. CPU is ALWAYS offered
            // here: the bench measures every unit regardless of the Settings toggle
            if (ui.runtimeChoices.size > 1) {
                Segmented(ui.runtimeChoices, ui.selectedRuntime, { it.label }, enabled = enabled, modifier = Modifier.weight(1f)) {
                    if (it != ui.selectedRuntime) { haptics.light(); vm.selectRuntime(it) }
                }
            }
            Segmented(ui.computeChoices, ui.selectedCompute, { it.label }, enabled = enabled, modifier = Modifier.weight(1f)) {
                if (it != ui.selectedCompute) { haptics.light(); vm.selectCompute(it) }
            }
        }
        // manual duration selection: a fine stepper plus quick presets
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            StepperRow(
                "Duration: ${ui.minutes} min", enabled = enabled, modifier = Modifier.weight(1f),
                canDec = ui.minutes > 1, canInc = ui.minutes < ui.maxMinutes,
                onDec = { haptics.light(); vm.setMinutes(ui.minutes - 1) }, onInc = { haptics.light(); vm.setMinutes(ui.minutes + 1) },
            )
            for (p in BenchStats.presets(ui.maxMinutes)) {
                PresetButton("$p", selected = ui.minutes == p, enabled = enabled) { if (ui.minutes != p) { haptics.light(); vm.setMinutes(p) } }
            }
        }
    }
}

/** `advancedSettings` (`BenchView.swift:311-329`). */
@Composable
private fun AdvancedCard(ui: BenchUi, vm: BenchViewModel) {
    val ios = LocalIosColors.current
    val enabled = !ui.running
    Column(Modifier.fillMaxWidth().innerCard(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        StepperRow(
            "Timed iters: ${ui.iters}", enabled = enabled, canDec = ui.iters > 20, canInc = ui.iters < 200,
            onDec = { vm.setIters(ui.iters - 10) }, onInc = { vm.setIters(ui.iters + 10) },
        )
        StepperRow(
            "Warmup: ${ui.warmup}", enabled = enabled, canDec = ui.warmup > 0, canInc = ui.warmup < 50,
            onDec = { vm.setWarmup(ui.warmup - 5) }, onInc = { vm.setWarmup(ui.warmup + 5) },
        )
        Text(
            if (ui.mode == BenchMode.Sustained) "In Sustained mode these set the cold baseline the throttle is measured against."
            else "Per model x unit cell in the cold run.",
            style = IosType.caption2, color = ios.secondaryLabel, modifier = Modifier.fillMaxWidth(),
        )
    }
}

/** iOS `Stepper`: label + the "-" / "+" pair. */
@Composable
private fun StepperRow(
    label: String, enabled: Boolean, canDec: Boolean, canInc: Boolean, onDec: () -> Unit, onInc: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val ios = LocalIosColors.current
    Row(modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = IosType.caption.tabular, color = if (enabled) ios.label else ios.tertiaryLabel, modifier = Modifier.weight(1f), maxLines = 1)
        Row(horizontalArrangement = Arrangement.spacedBy(2.dp)) {
            BorderedIconButton(Icons.Filled.Remove, onClick = onDec, enabled = enabled && canDec, minWidth = 34.dp, tint = ios.secondaryLabel)
            BorderedIconButton(Icons.Filled.Add, onClick = onInc, enabled = enabled && canInc, minWidth = 34.dp, tint = ios.secondaryLabel)
        }
    }
}

// ---- cards -----------------------------------------------------------------------------------

/** `progressCard` (`BenchView.swift:333-353`): "Paused" row, phase content, ThermalTach 46 trailing. */
@Composable
private fun ProgressCard(ui: BenchUi, thermalLevel: Int, thermalColor: androidx.compose.ui.graphics.Color) {
    val ios = LocalIosColors.current
    Column(Modifier.fillMaxWidth().materialCard(14.dp).padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (ui.paused) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.PauseCircle, null, tint = IosOrange, modifier = Modifier.size(14.dp))
                Spacer(Modifier.width(4.dp))
                Text("Paused", style = IosType.captionBold, color = IosOrange)
            }
        }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            // leading flexible column: the progress bar fills THIS width only, never under the dial
            Box(Modifier.weight(1f)) {
                when (val p = ui.phase) {
                    is BenchPhase.Loading -> Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp, color = ios.label)
                        Text("Loading ${p.model} to ${p.unit}", style = IosType.caption, color = ios.label)
                    }
                    is BenchPhase.Benchmarking -> Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("Benchmarking ${p.model} @ ${p.unit}", style = IosType.caption, color = ios.label)
                        LinearProgressIndicator(progress = { p.done.toFloat() / max(p.total, 1) }, modifier = Modifier.fillMaxWidth(), color = ios.accent, trackColor = ios.quaternaryFill)
                        Text("${p.done}/${p.total}", style = IosType.caption2.tabular, color = ios.secondaryLabel)
                    }
                    is BenchPhase.Sustained -> Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("Sustained ${p.model} @ ${p.unit}", style = IosType.caption, color = ios.label)
                        LinearProgressIndicator(progress = { p.elapsedS.toFloat() / max(p.totalS, 1) }, modifier = Modifier.fillMaxWidth(), color = ios.accent, trackColor = ios.quaternaryFill)
                        Text("${p.elapsedS / 60}m ${p.elapsedS % 60}s / ${p.totalS / 60}m", style = IosType.caption2.tabular, color = ios.secondaryLabel)
                    }
                    else -> {}
                }
            }
            ThermalTach(level = thermalLevel, color = thermalColor, size = 46.dp)
        }
    }
}

/**
 * `sustainedGraphCard` (`BenchView.swift:358-380`): rolling 1-minute window while running, the
 * full-run trend once finished, the thermal timeline, "cold x -> live|final y ms" + duration.
 */
@Composable
private fun SustainedGraphCard(ui: BenchUi) {
    val ios = LocalIosColors.current
    Column(Modifier.fillMaxWidth().materialCard(14.dp).padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Sparkline(ui.sparkSamples, ui.sparkBaseline, IosBlue, Modifier.fillMaxWidth().height(64.dp))
        if (ui.sparkThermal.isNotEmpty()) ThermalBar(ui.sparkThermal, Modifier.fillMaxWidth())
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            ui.sparkBaseline?.let { Text("cold ${fmt1(it)}", style = IosType.caption2, color = ios.secondaryLabel) }
            Text("→ ${if (ui.running) "live" else "final"} ${fmt1(ui.liveMs)} ms", style = IosType.caption2.tabular, color = ios.label)
            Spacer(Modifier.weight(1f))
            if (ui.runDuration > 0) Text(BenchHistory.durationText(ui.runDuration), style = IosType.caption2.tabular, color = ios.secondaryLabel)
        }
    }
}

/** `heroCard` (`BenchView.swift:407-428`): 60-pt ms dial 30..100, "FASTEST", fullName, unit · ms · FPS. */
@Composable
private fun HeroCard(r: BenchResult) {
    val ios = LocalIosColors.current
    Row(Modifier.fillMaxWidth().materialCard(14.dp).padding(14.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        FpsDial(
            value = r.coldMedian.coerceIn(30.0, 100.0), range = 30.0..100.0, tint = HudColors.msColor(r.coldMedian), active = true,
            unitLabel = "ms", valueText = "${r.coldMedian.roundToInt()}", size = 60.dp,
        )
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text("FASTEST", style = IosType.caption2, color = ios.secondaryLabel)
            Text(r.fullName, style = IosType.headline, color = ios.label, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text("${r.cell}  ·  ${fmt1(r.coldMedian)} ms  ·  ${r.fps.toInt()} FPS", style = IosType.caption.tabular, color = ios.secondaryLabel)
        }
    }
}

/** `modelCard` (`BenchView.swift:430-477`): title + "fastest <unit>" badge + chevron; bars per unit; expandable stage rows. */
@Composable
private fun ModelCard(mid: String, ui: BenchUi, onToggle: () -> Unit) {
    val ios = LocalIosColors.current
    val rows = ui.results.filter { it.modelId == mid }.sortedBy { it.coldMedian }
    val title = rows.firstOrNull()?.fullName ?: mid
    val expanded = mid in ui.expanded
    Column(
        Modifier.fillMaxWidth().materialCard(14.dp)
            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null, onClick = onToggle)
            .animateContentSize().padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, style = IosType.subheadlineBold, color = ios.label, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
            rows.firstOrNull()?.let { f ->
                Text(
                    "fastest ${f.cell}", style = IosType.caption2, color = ios.label,
                    modifier = Modifier.background(IosGreen.copy(alpha = 0.2f), CircleShape).padding(horizontal = 6.dp, vertical = 2.dp),
                )
            }
            Icon(if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore, null, tint = ios.tertiaryLabel, modifier = Modifier.size(14.dp))
        }
        for (r in rows) {
            DetailBar(unitIcon(r.compute), r.cell, r.coldMedian, fullScale = 50.0, color = HudColors.msColor(r.coldMedian), value = fmt1(r.coldMedian))
            val s = r.sustainedMedian; val tp = r.throttlePct
            if (s != null && tp != null) {
                DetailBar(Icons.Filled.LocalFireDepartment, "sustained", s, fullScale = 50.0, color = HudColors.msColor(s), value = "+${tp.toInt()}%")
            }
        }
        if (expanded) {
            Divider(color = ios.separator, thickness = 0.5.dp)
            for (r in rows) {
                Text("${r.cell} · ${r.fps.toInt()} FPS · p90 ${fmt1(r.coldP90)}", style = IosType.caption2, color = ios.secondaryLabel)
                DetailBar(StageIcons.preprocess, "preprocess", r.preMs)
                DetailBar(StageIcons.inference, "inference", r.infMs)
                DetailBar(StageIcons.decode, "decode", r.decMs)
                DetailBar(StageIcons.endToEnd, "end to end", r.preMs + r.infMs + r.decMs, fullScale = 100.0, greenUntil = 50.0)
            }
        }
    }
}
