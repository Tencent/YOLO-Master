package dev.yolomaster.app.ui.common

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.UnfoldMore
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.yolomaster.app.model.BundledModel
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.ui.overlay.BoxStyle
import dev.yolomaster.app.ui.overlay.SegOverlayMode
import dev.yolomaster.app.ui.theme.IosType
import dev.yolomaster.app.ui.theme.LocalIosColors
import dev.yolomaster.ncnn.Runtime

/*
 * The controls shared by the three working tabs: the model Menu (shortID + chevron), the runtime
 * and compute dropdowns, iOS `.bordered` / `.borderedProminent` icon buttons and the tuning panel.
 */

/** iOS `Menu` with a `Picker("Model")` inside; label = `shortID` + up/down chevron. */
@Composable
fun ModelMenu(models: List<BundledModel>, selected: BundledModel?, enabled: Boolean, onSelect: (BundledModel) -> Unit) {
    val ios = LocalIosColors.current
    var open by remember { mutableStateOf(false) }
    Box {
        TextButton(onClick = { open = true }, enabled = enabled, contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)) {
            Text(selected?.shortID ?: "Model", style = IosType.body, color = if (enabled) ios.accent else ios.tertiaryLabel, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Icon(Icons.Filled.UnfoldMore, null, tint = if (enabled) ios.accent else ios.tertiaryLabel, modifier = Modifier.size(14.dp))
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            for (m in models) DropdownMenuItem(
                text = { Text(m.fullName, style = IosType.body, color = if (m == selected) ios.accent else ios.label) },
                onClick = { open = false; onSelect(m) },
            )
        }
    }
}

/**
 * A menu-style `Picker` over a short list: label + up/down chevron, one dropdown item per choice.
 * The button is inert when there is a single choice (nothing to pick; the label still shows what
 * is in force). The compute and runtime pickers are thin wrappers.
 */
@Composable
fun <T> PickerMenu(choices: List<T>, selected: T, label: (T) -> String, enabled: Boolean, onSelect: (T) -> Unit) {
    val ios = LocalIosColors.current
    var open by remember { mutableStateOf(false) }
    Box {
        TextButton(onClick = { open = true }, enabled = enabled && choices.size > 1, contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)) {
            Text(label(selected), style = IosType.body, color = if (enabled) ios.accent else ios.tertiaryLabel, maxLines = 1)
            Icon(Icons.Filled.UnfoldMore, null, tint = if (enabled) ios.accent else ios.tertiaryLabel, modifier = Modifier.size(14.dp))
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            for (c in choices) DropdownMenuItem(
                text = { Text(label(c), style = IosType.body, color = if (c == selected) ios.accent else ios.label) },
                onClick = { open = false; onSelect(c) },
            )
        }
    }
}

/** The compute `Picker` (menu style): GPU / CPU / NPU as the runtime offers, CPU only when allowed or forced. */
@Composable
fun ComputeMenu(choices: List<ComputeChoice>, selected: ComputeChoice, enabled: Boolean, onSelect: (ComputeChoice) -> Unit) =
    PickerMenu(choices, selected, { it.label }, enabled, onSelect)

/** The runtime `Picker` (ncnn / ONNX); callers hide it when the model has a single runtime. */
@Composable
fun RuntimeMenu(choices: List<Runtime>, selected: Runtime, enabled: Boolean, onSelect: (Runtime) -> Unit) =
    PickerMenu(choices, selected, { it.label }, enabled, onSelect)

/** iOS `.buttonStyle(.bordered)`: tinted translucent fill, no outline. */
@Composable
fun BorderedIconButton(
    icon: ImageVector, onClick: () -> Unit, modifier: Modifier = Modifier, tint: Color? = null,
    enabled: Boolean = true, minWidth: Dp = 44.dp, content: (@Composable () -> Unit)? = null,
) {
    val ios = LocalIosColors.current
    val t = tint ?: ios.accent
    Button(
        onClick = onClick, enabled = enabled, modifier = modifier.height(34.dp).widthIn(min = minWidth),
        shape = RoundedCornerShape(8.dp),
        colors = ButtonDefaults.buttonColors(containerColor = t.copy(alpha = 0.15f), contentColor = t, disabledContainerColor = ios.quaternaryFill, disabledContentColor = ios.tertiaryLabel),
        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
        elevation = null,
    ) {
        if (content != null) content() else Icon(icon, null, modifier = Modifier.size(18.dp))
    }
}

/** iOS `.buttonStyle(.borderedProminent)`: filled with the tint, white glyph. */
@Composable
fun ProminentIconButton(
    icon: ImageVector, onClick: () -> Unit, modifier: Modifier = Modifier, tint: Color? = null,
    enabled: Boolean = true, minWidth: Dp = 44.dp, content: (@Composable () -> Unit)? = null,
) {
    val ios = LocalIosColors.current
    Button(
        onClick = onClick, enabled = enabled, modifier = modifier.height(34.dp).widthIn(min = minWidth),
        shape = RoundedCornerShape(8.dp),
        colors = ButtonDefaults.buttonColors(containerColor = tint ?: ios.accent, contentColor = Color.White, disabledContainerColor = ios.quaternaryFill, disabledContentColor = ios.tertiaryLabel),
        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
        elevation = null,
    ) {
        if (content != null) content() else Icon(icon, null, modifier = Modifier.size(18.dp))
    }
}

/** A `.pickerStyle(.segmented)` row. */
@Composable
fun <T> Segmented(options: List<T>, selected: T, label: (T) -> String, enabled: Boolean = true, modifier: Modifier = Modifier, onSelect: (T) -> Unit) {
    val ios = LocalIosColors.current
    SingleChoiceSegmentedButtonRow(modifier.height(30.dp)) {
        options.forEachIndexed { i, o ->
            SegmentedButton(
                selected = o == selected, onClick = { onSelect(o) }, enabled = enabled,
                shape = SegmentedButtonDefaults.itemShape(i, options.size),
                colors = SegmentedButtonDefaults.colors(activeContainerColor = ios.accent.copy(alpha = 0.18f), activeContentColor = ios.label, inactiveContentColor = ios.label),
                border = BorderStroke(0.5.dp, ios.separator),
                icon = {},
            ) { Text(label(o), style = IosType.caption, maxLines = 1, textAlign = TextAlign.Center) }
        }
    }
}

/**
 * The plain `Tuning` object of `StatsHUD.swift:455-459`: read by the detection loop without a
 * main-thread hop; values take effect on the next frame.
 */
class Tuning {
    @Volatile var conf: Float = 0.25f
    @Volatile var iou: Float = 0.5f
    @Volatile var style: BoxStyle = BoxStyle.Chip
    @Volatile var segOverlay: SegOverlayMode = SegOverlayMode.Both
    @Volatile var showHUD: Boolean = true
    /** ncnn threads for the Live loop: 1 / 2 / 4, or 0 = every core (no big-core pinning). */
    @Volatile var threads: Int = 0   // all cores (S26: 76 vs 87 ms pinned)
    /** Lighter camera config (640x480 analysis, 15-30 fps) to test whether the camera pipeline triggers the SoC clamp. */
    @Volatile var camLite: Boolean = false
}

/** Labels of the Live threads picker; 0 = all cores. */
val threadChoices = listOf(1, 2, 4, 0)
fun threadLabel(t: Int) = if (t == 0) "all" else "$t"

/** `TuningPanel` (`StatsHUD.swift:409-451`): padding 10, radius 12. */
@Composable
fun TuningPanel(tuning: Tuning, isSeg: Boolean, onChange: () -> Unit = {}, modifier: Modifier = Modifier, showThreads: Boolean = false, onThreads: (Int) -> Unit = {}, onCamLite: (Boolean) -> Unit = {}) {
    val ios = LocalIosColors.current
    var conf by remember { mutableFloatStateOf(tuning.conf) }
    var iou by remember { mutableFloatStateOf(tuning.iou) }
    var style by remember { mutableStateOf(tuning.style) }
    var seg by remember { mutableStateOf(tuning.segOverlay) }
    var hud by remember { mutableStateOf(tuning.showHUD) }
    Column(modifier.materialCard(12.dp).padding(10.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("Show stats HUD", style = IosType.caption, color = ios.label, modifier = Modifier.weight(1f))
            Switch(checked = hud, onCheckedChange = { hud = it; tuning.showHUD = it; onChange() }, modifier = Modifier.height(24.dp))
        }
        Segmented(BoxStyle.entries, style, { it.label }, modifier = Modifier.fillMaxWidth()) { style = it; tuning.style = it; onChange() }
        if (isSeg) Segmented(SegOverlayMode.entries, seg, { it.label }, modifier = Modifier.fillMaxWidth()) { seg = it; tuning.segOverlay = it; onChange() }
        SliderRow("conf", conf, 0.05f..0.9f) { conf = it; tuning.conf = it; onChange() }
        SliderRow("IoU", iou, 0.1f..0.9f) { iou = it; tuning.iou = it; onChange() }
        if (showThreads) {
            var threads by remember { mutableStateOf(tuning.threads) }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text("threads", style = IosType.caption, color = ios.label, modifier = Modifier.width(54.dp))
                Segmented(threadChoices, threads, { threadLabel(it) }, modifier = Modifier.weight(1f)) { threads = it; tuning.threads = it; onThreads(it) }
            }
            var lite by remember { mutableStateOf(tuning.camLite) }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text("cam lite (640x480, 15-30 fps)", style = IosType.caption, color = ios.label, modifier = Modifier.weight(1f))
                Switch(checked = lite, onCheckedChange = { lite = it; tuning.camLite = it; onCamLite(it) }, modifier = Modifier.height(24.dp))
            }
        }
    }
}

@Composable
private fun SliderRow(name: String, value: Float, range: ClosedFloatingPointRange<Float>, onValue: (Float) -> Unit) {
    val ios = LocalIosColors.current
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(name, style = IosType.caption, color = ios.label, modifier = Modifier.width(34.dp))
        Slider(
            value = value, onValueChange = onValue, valueRange = range, modifier = Modifier.weight(1f).height(24.dp),
            colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = ios.accent, inactiveTrackColor = ios.quaternaryFill),
        )
        Text(String.format("%.2f", value), style = IosType.caption.tabular, color = ios.label, modifier = Modifier.width(34.dp), textAlign = TextAlign.End)
    }
}

/** A small bare-number preset button (`.bordered`, `.caption2`). */
@Composable
fun PresetButton(text: String, selected: Boolean, enabled: Boolean = true, onClick: () -> Unit) {
    val ios = LocalIosColors.current
    val t = if (selected) ios.accent else ios.secondaryLabel
    OutlinedButton(
        onClick = onClick, enabled = enabled, modifier = Modifier.height(26.dp), shape = RoundedCornerShape(7.dp),
        border = null, contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
        colors = ButtonDefaults.outlinedButtonColors(containerColor = t.copy(alpha = 0.15f), contentColor = t),
    ) { Text(text, style = IosType.caption2) }
}
