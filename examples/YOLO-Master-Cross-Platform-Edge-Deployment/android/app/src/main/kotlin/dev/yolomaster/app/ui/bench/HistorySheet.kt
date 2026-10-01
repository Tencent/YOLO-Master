package dev.yolomaster.app.ui.bench

import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.SwapVert
import androidx.compose.material.icons.outlined.Circle
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.yolomaster.app.system.Haptics
import dev.yolomaster.app.ui.common.materialCard
import dev.yolomaster.app.ui.common.tabular
import dev.yolomaster.app.ui.hud.DetailBar
import dev.yolomaster.app.ui.hud.ExportToast
import dev.yolomaster.app.ui.hud.HudColors
import dev.yolomaster.app.ui.hud.Sparkline
import dev.yolomaster.app.ui.hud.ThermalBar
import dev.yolomaster.app.ui.hud.Toast
import dev.yolomaster.app.ui.theme.IosBlue
import dev.yolomaster.app.ui.theme.IosOrange
import dev.yolomaster.app.ui.theme.IosRed
import dev.yolomaster.app.ui.theme.IosType
import dev.yolomaster.app.ui.theme.LocalIosColors
import java.text.DateFormat
import java.util.Date
import kotlin.math.max

/*
 * `HistoryView` (`BenchView.swift:814-1052`): the permanent benchmark history - searchable,
 * sortable, renameable, exportable. Rows are simplified by default and EXPAND on tap to reveal
 * the sustained throttle sparkline + thermal bar and the comparison bars.
 */

private enum class SortKey(val label: String) { Time("Time"), Name("Name") }

/** The iOS `.sheet { HistoryView }`: a full-height modal sheet titled "History". */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun HistorySheet(history: BenchHistory, onDismiss: () -> Unit) {
    val ios = LocalIosColors.current
    val ctx = LocalContext.current
    val view = LocalView.current
    val haptics = remember { Haptics(view) }
    val runs by history.runs.collectAsStateWithLifecycle()
    val sheet = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    var query by remember { mutableStateOf("") }
    var sort by remember { mutableStateOf(SortKey.Time) }
    var sortOpen by remember { mutableStateOf(false) }
    var editing by remember { mutableStateOf(false) }
    var selection by remember { mutableStateOf<Set<String>>(emptySet()) }
    var expanded by remember { mutableStateOf<Set<String>>(emptySet()) }
    var fpsRuns by remember { mutableStateOf<Set<String>>(emptySet()) }   // cards showing FPS instead of latency
    var renaming by remember { mutableStateOf<BenchRun?>(null) }
    var newName by remember { mutableStateOf("") }
    var toast by remember { mutableStateOf<Toast?>(null) }
    var toastGen by remember { mutableIntStateOf(0) }

    // `shown` (BenchView.swift:831-846): filter by name / mode / any model name, then sort
    val shown = remember(runs, query, sort) {
        var r = runs
        if (query.isNotBlank()) {
            val q = query.trim()
            r = r.filter { run ->
                run.name.contains(q, ignoreCase = true) || run.mode.contains(q, ignoreCase = true) ||
                    run.results.any { it.fullName.contains(q, ignoreCase = true) }
            }
        }
        when (sort) {
            SortKey.Time -> r.sortedByDescending { it.dateMs }
            SortKey.Name -> r.sortedWith(compareBy(String.CASE_INSENSITIVE_ORDER) { it.name })
        }
    }

    /** `exportRuns` (`BenchView.swift:998-1016`): explicit runs, else the selection, else everything shown. */
    fun exportRuns(explicit: List<BenchRun>? = null) {
        val list = explicit ?: if (selection.isEmpty()) shown else runs.filter { it.id in selection }
        if (list.isEmpty()) return
        val ok = BenchShare.shareCsv(ctx, "bench_history.csv", BenchHistory.runsCSV(list))
        haptics.light(); if (ok) haptics.success() else haptics.error()
        toast = Toast(if (ok) "${list.size} run${if (list.size == 1) "" else "s"} exported" else "Export failed", ok, autoDismissMs = 1800, gen = ++toastGen)
    }

    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = sheet, containerColor = ios.groupedBackground, dragHandle = null) {
        Box(Modifier.fillMaxWidth().fillMaxHeight(0.94f)) {
            Column(Modifier.fillMaxSize()) {
                // toolbar: sort (leading) | "History" | export · Edit/Done · Done (trailing)
                Box(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp)) {
                    Box(Modifier.align(Alignment.CenterStart)) {
                        IconButton(onClick = { sortOpen = true }) { Icon(Icons.Filled.SwapVert, "Sort", tint = ios.accent) }
                        DropdownMenu(expanded = sortOpen, onDismissRequest = { sortOpen = false }) {
                            for (k in SortKey.entries) DropdownMenuItem(
                                text = { Text(k.label, style = IosType.body, color = ios.label) },
                                leadingIcon = { if (k == sort) Icon(Icons.Filled.Check, null, tint = ios.accent, modifier = Modifier.size(16.dp)) },
                                onClick = { sort = k; sortOpen = false },
                            )
                        }
                    }
                    Text("History", style = IosType.headline, color = ios.label, modifier = Modifier.align(Alignment.Center))
                    Row(Modifier.align(Alignment.CenterEnd), verticalAlignment = Alignment.CenterVertically) {
                        if (runs.isNotEmpty()) {
                            if (editing && selection.isNotEmpty()) {
                                IconButton(onClick = { history.delete(selection); selection = emptySet() }) { Icon(Icons.Filled.Delete, "Delete", tint = IosRed) }
                            }
                            IconButton(onClick = { exportRuns() }) { Icon(Icons.Filled.Share, "Export", tint = ios.accent) }
                            TextButton(onClick = { editing = !editing; if (!editing) selection = emptySet() }, contentPadding = PaddingValues(horizontal = 6.dp)) {
                                Text(if (editing) "Done" else "Edit", style = IosType.body, color = ios.accent)
                            }
                        }
                        TextButton(onClick = onDismiss, contentPadding = PaddingValues(horizontal = 6.dp)) {
                            Text("Done", style = IosType.headline, color = ios.accent)
                        }
                    }
                }
                if (runs.isEmpty()) {
                    Column(Modifier.fillMaxSize().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
                        Icon(Icons.Filled.History, null, tint = ios.secondaryLabel, modifier = Modifier.size(48.dp))
                        Spacer(Modifier.height(10.dp))
                        Text("No saved runs", style = IosType.title3Bold, color = ios.label)
                        Spacer(Modifier.height(4.dp))
                        Text("Completed cold and sustained runs are saved here.", style = IosType.subheadline, color = ios.secondaryLabel)
                    }
                } else {
                    OutlinedTextField(
                        value = query, onValueChange = { query = it }, singleLine = true,
                        placeholder = { Text("Search runs or models", style = IosType.subheadline, color = ios.tertiaryLabel) },
                        leadingIcon = { Icon(Icons.Filled.Search, null, tint = ios.secondaryLabel, modifier = Modifier.size(18.dp)) },
                        textStyle = IosType.subheadline, shape = RoundedCornerShape(10.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Color.Transparent, unfocusedBorderColor = Color.Transparent,
                            focusedContainerColor = ios.quaternaryFill, unfocusedContainerColor = ios.quaternaryFill,
                        ),
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
                    )
                    LazyColumn(contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        items(shown, key = { it.id }) { run ->
                            HistoryCard(
                                run = run, editing = editing, selected = run.id in selection, open = run.id in expanded, asFPS = run.id in fpsRuns,
                                onTap = {
                                    if (editing) selection = if (run.id in selection) selection - run.id else selection + run.id
                                    else expanded = if (run.id in expanded) expanded - run.id else expanded + run.id
                                },
                                onToggleFPS = { fpsRuns = if (run.id in fpsRuns) fpsRuns - run.id else fpsRuns + run.id },
                                onRename = { renaming = run; newName = run.name },
                                onExport = { exportRuns(listOf(run)) },
                                onDelete = { history.delete(setOf(run.id)) },
                            )
                        }
                    }
                }
            }
            toast?.let { t ->
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { ExportToast(t) { if (toast?.gen == t.gen) toast = null } }
            }
        }
    }

    // `.alert("Rename run")` (BenchView.swift:891-899)
    renaming?.let { run ->
        AlertDialog(
            onDismissRequest = { renaming = null },
            title = { Text("Rename run", style = IosType.headline) },
            text = {
                OutlinedTextField(value = newName, onValueChange = { newName = it }, singleLine = true, label = { Text("Name") }, modifier = Modifier.fillMaxWidth())
            },
            confirmButton = {
                TextButton(onClick = { if (newName.isNotBlank()) history.rename(run.id, newName.trim()); renaming = null }) { Text("Save") }
            },
            dismissButton = { TextButton(onClick = { renaming = null }) { Text("Cancel") } },
        )
    }
}

/** `card(run)` (`BenchView.swift:903-996`): fixed header + summary, detail block grows in below. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun HistoryCard(
    run: BenchRun, editing: Boolean, selected: Boolean, open: Boolean, asFPS: Boolean,
    onTap: () -> Unit, onToggleFPS: () -> Unit, onRename: () -> Unit, onExport: () -> Unit, onDelete: () -> Unit,
) {
    val ios = LocalIosColors.current
    var menu by remember { mutableStateOf(false) }
    val sustained = run.mode == BenchMode.Sustained.label
    val spark = run.results.firstOrNull { it.sparkline.size > 1 }?.sparkline
    val timeline = run.results.firstOrNull { it.thermal.isNotEmpty() }?.thermal
    Column(
        Modifier.fillMaxWidth().materialCard(14.dp)
            .border(2.dp, if (selected) ios.accent else Color.Transparent, RoundedCornerShape(14.dp))
            .combinedClickable(interactionSource = remember { MutableInteractionSource() }, indication = null, onClick = onTap, onLongClick = { menu = true })
            .animateContentSize().padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (editing) {
                Icon(
                    if (selected) Icons.Filled.CheckCircle else Icons.Outlined.Circle, null,
                    tint = if (selected) ios.accent else ios.secondaryLabel, modifier = Modifier.size(20.dp),
                )
            }
            Text(run.name, style = IosType.subheadlineBold, color = ios.label, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
            Text(
                run.mode, style = IosType.caption2, color = ios.label,
                modifier = Modifier.background((if (sustained) IosOrange else IosBlue).copy(alpha = 0.2f), CircleShape).padding(horizontal = 6.dp, vertical = 2.dp),
            )
            if (!editing) {
                Icon(Icons.Filled.ExpandMore, null, tint = ios.tertiaryLabel, modifier = Modifier.size(14.dp).rotate(if (open) 180f else 0f))
                // the iOS context menu (Rename / Export / Delete): long-press or the dots
                Box {
                    Icon(
                        Icons.Filled.MoreVert, "More", tint = ios.tertiaryLabel,
                        modifier = Modifier.size(18.dp).clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) { menu = true },
                    )
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        DropdownMenuItem(text = { Text("Rename", style = IosType.body, color = ios.label) }, leadingIcon = { Icon(Icons.Filled.Edit, null, tint = ios.label) }, onClick = { menu = false; onRename() })
                        DropdownMenuItem(text = { Text("Export", style = IosType.body, color = ios.label) }, leadingIcon = { Icon(Icons.Filled.Share, null, tint = ios.label) }, onClick = { menu = false; onExport() })
                        DropdownMenuItem(text = { Text("Delete", style = IosType.body, color = IosRed) }, leadingIcon = { Icon(Icons.Filled.Delete, null, tint = IosRed) }, onClick = { menu = false; onDelete() })
                    }
                }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(run.dateMs)), style = IosType.caption2, color = ios.secondaryLabel)
            if (sustained && run.durationSec > 0) Text("· ${BenchHistory.durationText(run.durationSec)}", style = IosType.caption2.tabular, color = ios.secondaryLabel)
        }
        run.fastest?.let { f ->
            Text("fastest ${f.shortID} @ ${f.cell} · ${fmt1(f.coldMedian)} ms · ${f.fps.toInt()} FPS", style = IosType.caption.tabular, color = ios.secondaryLabel)
        }
        if (open) {
            Column(Modifier.padding(top = 2.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                if (sustained && spark != null) Sparkline(spark, null, IosBlue, Modifier.fillMaxWidth().height(44.dp))
                if (sustained && !timeline.isNullOrEmpty()) {
                    ThermalBar(timeline, Modifier.fillMaxWidth().padding(top = 8.dp))
                    run.fastest?.throttlePct?.let { tp -> Text("throttle +${tp.toInt()}%", style = IosType.caption2.tabular, color = IosOrange) }
                }
                // comparison bars grouped by model (fastest unit first within a model), scale = the slowest cold median
                val scale = max(run.results.maxOfOrNull { it.coldMedian } ?: 50.0, 1.0)
                val sorted = run.results.sortedWith(compareBy<BenchResult> { it.modelId }.thenBy { it.coldMedian })
                for (r in sorted) {
                    DetailBar(
                        unitIcon(r.compute), "${r.shortID} ${r.cell}", r.coldMedian, fullScale = scale, color = HudColors.msColor(r.coldMedian),
                        value = if (asFPS) "${r.fps.toInt()} fps" else "${fmt1(r.coldMedian)} ms", valueWidth = 56.dp,
                        modifier = Modifier.clickable(interactionSource = remember { MutableInteractionSource() }, indication = null, onClick = onToggleFPS),
                    )
                }
            }
        }
    }
}
