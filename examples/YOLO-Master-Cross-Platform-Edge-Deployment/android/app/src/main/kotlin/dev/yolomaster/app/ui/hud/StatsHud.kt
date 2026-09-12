package dev.yolomaster.app.ui.hud

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.spring
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.layout.heightIn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AspectRatio
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.SelectAll
import androidx.compose.material.icons.filled.PersonPin
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.yolomaster.app.ui.common.materialCard
import dev.yolomaster.app.ui.common.tabular
import dev.yolomaster.app.ui.theme.IosType
import dev.yolomaster.app.ui.theme.LocalIosColors
import kotlin.math.roundToInt

/** Icons of the stage rows (iOS SF symbols mapped to Material). */
object StageIcons {
    val preprocess: ImageVector = Icons.Filled.AspectRatio
    val inference: ImageVector = Icons.Filled.Memory
    val decode: ImageVector = Icons.Filled.SelectAll
    val mask: ImageVector = Icons.Filled.PersonPin
    val postprocess: ImageVector = Icons.Filled.Settings
    val endToEnd: ImageVector = Icons.Filled.Timer
}

enum class DialMode { FPS, MS }

/** Everything the HUD shows; the screens fill it per frame / per image / per batch. */
data class HudStats(
    val active: Boolean = false,
    val fps: Double = 0.0,            // dial value (fps or ms depending on [mode])
    val mode: DialMode = DialMode.FPS,
    val fpsLabel: String = "FPS",
    val pre: Double = 0.0,
    val inf: Double = 0.0,
    val dec: Double = 0.0,
    val mask: Double = 0.0,
    val isSeg: Boolean = false,
    val dets: Int = 0,
    val thermalLevel: Int = 0,
    val thermalKnown: Boolean = true,
    val extras: List<Pair<String, String>> = emptyList(),
)

/**
 * `StatsHUD` (`StatsHUD.swift:55-199`): material card, padding 10, radius 14, tap toggles the
 * expanded section. In Live the card keeps the collapsed row's width when expanded
 * (`fullWidth = false`); Photo stretches it.
 */
@Composable
fun StatsHud(stats: HudStats, fullWidth: Boolean = false, modifier: Modifier = Modifier) {
    val ios = LocalIosColors.current
    var expanded by remember { mutableStateOf(false) }
    var rowWidth by remember { mutableStateOf<Dp?>(null) }
    val density = LocalDensity.current

    val dialTint = if (stats.mode == DialMode.FPS) HudColors.fpsColor(stats.fps) else HudColors.msColor(stats.fps)
    val dialRange = if (stats.mode == DialMode.FPS) 0.0..30.0 else 30.0..100.0
    val dialValue = if (stats.mode == DialMode.FPS) stats.fps.coerceIn(0.0, 30.0) else stats.fps.coerceIn(30.0, 100.0)

    Column(
        modifier
            .then(if (fullWidth) Modifier.fillMaxWidth() else Modifier)
            .materialCard(14.dp)
            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) { expanded = !expanded }
            .padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(
            Modifier
                .then(if (fullWidth) Modifier.fillMaxWidth() else Modifier)
                .onSizeChanged { if (!fullWidth) rowWidth = with(density) { it.width.toDp() } },
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            FpsDial(
                value = dialValue, range = dialRange, tint = dialTint, active = stats.active,
                unitLabel = stats.fpsLabel,
                valueText = if (stats.active) "${stats.fps.roundToInt()}" else "N/A",
            )
            ThermalTach(level = stats.thermalLevel, color = HudColors.thermalColor(stats.thermalLevel, stats.thermalKnown))
            if (!expanded) {
                Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                    StageBar(StageIcons.preprocess, stats.pre)
                    StageBar(StageIcons.inference, stats.inf)
                    StageBar(StageIcons.decode, stats.dec)
                    if (stats.isSeg) StageBar(StageIcons.mask, stats.mask)
                }
            }
            if (fullWidth) androidx.compose.foundation.layout.Spacer(Modifier.weight(1f))
            DetsCount(stats.dets)
        }
        AnimatedVisibility(
            visible = expanded,
            enter = expandVertically(spring(stiffness = 300f)) + fadeIn(),
            exit = shrinkVertically(spring(stiffness = 300f)) + fadeOut(),
        ) {
            val w = rowWidth
            Column(
                Modifier.then(if (!fullWidth && w != null) Modifier.width(w) else Modifier.fillMaxWidth()),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                HorizontalDivider(color = ios.separator)
                Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                    DetailBar(StageIcons.preprocess, "preprocess", stats.pre)
                    DetailBar(StageIcons.inference, "inference", stats.inf)
                    DetailBar(StageIcons.decode, "decode", stats.dec)
                    if (stats.isSeg) DetailBar(StageIcons.mask, "mask", stats.mask)
                    DetailBar(StageIcons.postprocess, "postprocess", stats.dec + stats.mask)
                    DetailBar(StageIcons.endToEnd, "end to end", stats.pre + stats.inf + stats.dec + stats.mask, fullScale = 100.0, greenUntil = 50.0)
                }
                if (stats.extras.isNotEmpty()) {
                    HorizontalDivider(color = ios.separator)
                    LazyVerticalGrid(
                        columns = GridCells.Fixed(2),
                        verticalArrangement = Arrangement.spacedBy(3.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        userScrollEnabled = false,
                        modifier = Modifier.heightIn(max = 400.dp),
                    ) {
                        items(stats.extras) { (k, v) ->
                            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                Text(k, style = IosType.caption2, color = ios.secondaryLabel)
                                Text(v, style = IosType.caption2.tabular, color = ios.label)
                            }
                        }
                    }
                }
            }
        }
    }
}
