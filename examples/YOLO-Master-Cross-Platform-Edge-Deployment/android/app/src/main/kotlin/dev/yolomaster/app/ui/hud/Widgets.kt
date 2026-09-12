package dev.yolomaster.app.ui.hud

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Thermostat
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.yolomaster.app.ui.common.tabular
import dev.yolomaster.app.ui.theme.IosType
import dev.yolomaster.app.ui.theme.LocalIosColors
import kotlin.math.max
import kotlin.math.min

/*
 * The hand-drawn HUD widgets of `StatsHUD.swift`, geometry reproduced from the Swift source.
 */

/**
 * `ThermalTach` (`StatsHUD.swift:233-263`): four arc segments on a 270-degree ring starting at
 * 135 degrees (bottom-left), gap at the bottom, thermometer glyph in the middle.
 */
@Composable
fun ThermalTach(level: Int, color: Color, size: Dp = 56.dp, modifier: Modifier = Modifier) {
    val ios = LocalIosColors.current
    val track = ios.quaternaryFill
    val animated by animateColorAsState(color, tween(300), label = "tachColor")
    Box(modifier.size(size), contentAlignment = Alignment.Center) {
        Canvas(Modifier.size(size)) {
            val span = 0.75f; val gap = 0.055f; val segLen = (span - 3 * gap) / 4f
            val stroke = size.toPx() * 0.107f
            val d = size.toPx() - 2.dp.toPx()
            val topLeft = Offset((size.toPx() - d) / 2f, (size.toPx() - d) / 2f)
            for (i in 0 until 4) {
                val from = i * (segLen + gap)
                val startDeg = 135f + from * 360f
                val sweepDeg = segLen * 360f
                drawArc(
                    color = if (i <= level) animated else track,
                    startAngle = startDeg, sweepAngle = sweepDeg, useCenter = false,
                    topLeft = topLeft, size = Size(d, d),
                    style = Stroke(width = stroke, cap = StrokeCap.Round),
                )
            }
        }
        Icon(
            Icons.Filled.Thermostat, contentDescription = null, tint = animated,
            modifier = Modifier.size(size * 0.43f).offset(y = size * 0.054f),
        )
    }
}

/**
 * The iOS `Gauge(.accessoryCircular)`: a 270-degree ring (135 -> 405 degrees), tinted progress
 * arc over a faint track, the value centred, the unit label at the bottom of the ring.
 */
@Composable
fun FpsDial(
    value: Double, range: ClosedFloatingPointRange<Double>, tint: Color, active: Boolean,
    unitLabel: String, valueText: String, size: Dp = 56.dp, modifier: Modifier = Modifier,
) {
    val ios = LocalIosColors.current
    val frac = if (!active) 0f else (((value - range.start) / (range.endInclusive - range.start)).coerceIn(0.0, 1.0)).toFloat()
    val animFrac by animateFloatAsState(frac, tween(250), label = "dialFrac")
    val animTint by animateColorAsState(if (active) tint else HudColors.inactiveDial, tween(250), label = "dialTint")
    Box(modifier.size(size), contentAlignment = Alignment.Center) {
        Canvas(Modifier.size(size)) {
            val stroke = size.toPx() * 0.107f
            val d = size.toPx() - stroke
            val tl = Offset((size.toPx() - d) / 2f, (size.toPx() - d) / 2f)
            drawArc(ios.quaternaryFill, 135f, 270f, false, tl, Size(d, d), style = Stroke(stroke, cap = StrokeCap.Round))
            if (animFrac > 0f) drawArc(animTint, 135f, 270f * animFrac, false, tl, Size(d, d), style = Stroke(stroke, cap = StrokeCap.Round))
        }
        Text(
            valueText,
            style = (if (active) IosType.title3Rounded else IosType.footnoteRounded).tabular,
            color = if (active) ios.label else ios.secondaryLabel,
            modifier = Modifier.offset(y = 1.dp),
            maxLines = 1,
        )
        Text(
            unitLabel, style = IosType.caption2.copy(fontSize = 8.sp, lineHeight = 9.sp), color = ios.secondaryLabel,
            modifier = Modifier.align(Alignment.BottomCenter).offset(y = (-1).dp),
        )
    }
}

/** Capsule track + fill, height 6, minimum fill 3 (`StageBar`/`DetailBar`). */
@Composable
private fun CapsuleBar(fraction: Float, color: Color, modifier: Modifier = Modifier) {
    val ios = LocalIosColors.current
    val f by animateFloatAsState(fraction.coerceIn(0f, 1f), tween(150), label = "bar")
    val c by animateColorAsState(color, tween(150), label = "barColor")
    Box(modifier.height(6.dp).clip(CircleShape).background(ios.quaternaryFill)) {
        Box(
            Modifier.fillMaxHeight().fillMaxWidth(fraction = max(f, 0.0001f)).clip(CircleShape).background(c),
        )
    }
}

/** Compact collapsed row: icon 24 wide, 90x6 bar, `%4.1f` value 36 wide (`StatsHUD.swift:201-227`). */
@Composable
fun StageBar(icon: ImageVector, ms: Double, greenUntil: Double = 20.0, fullScale: Double = 50.0) {
    val ios = LocalIosColors.current
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Box(Modifier.width(24.dp), contentAlignment = Alignment.Center) {
            Icon(icon, null, tint = ios.secondaryLabel, modifier = Modifier.size(13.dp))
        }
        val fillPx = max(3.0, min(ms / fullScale, 1.0) * 90.0)
        CapsuleBar((fillPx / 90.0).toFloat(), HudColors.stageColor(ms, greenUntil), Modifier.width(90.dp))
        Text(String.format("%4.1f", ms), style = IosType.caption2.tabular, color = ios.label, modifier = Modifier.width(36.dp), textAlign = TextAlign.End, maxLines = 1)
    }
}

/**
 * Expanded row: icon 20 + name 78 wide + flexible (or fixed) bar + value 40 wide
 * (`StatsHUD.swift:266-302`).
 */
@Composable
fun DetailBar(
    icon: ImageVector, name: String, ms: Double, fullScale: Double = 50.0, greenUntil: Double = 20.0,
    color: Color? = null, value: String? = null, barWidth: Dp? = null, valueWidth: Dp = 40.dp,
    modifier: Modifier = Modifier,
) {
    val ios = LocalIosColors.current
    Row(modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Box(Modifier.width(20.dp), contentAlignment = Alignment.Center) {
            Icon(icon, null, tint = ios.secondaryLabel, modifier = Modifier.size(13.dp))
        }
        Text(name, style = IosType.caption2, color = ios.label, modifier = Modifier.width(78.dp), maxLines = 1, overflow = TextOverflow.Ellipsis)
        val frac = min(ms / fullScale, 1.0).toFloat()
        if (barWidth != null) {
            CapsuleBar(max(frac, (3.dp / barWidth)), color ?: HudColors.stageColor(ms, greenUntil), Modifier.width(barWidth))
            Spacer(Modifier.weight(1f, fill = true))
        } else {
            CapsuleBar(max(frac, 0.03f), color ?: HudColors.stageColor(ms, greenUntil), Modifier.weight(1f))
        }
        Text(
            value ?: String.format("%5.1f", ms), style = IosType.caption2.tabular, color = ios.label,
            modifier = Modifier.width(valueWidth), textAlign = TextAlign.End, maxLines = 1,
        )
    }
}

/** `SparklineView` (`StatsHUD.swift:307-343`): polyline lw 2, dashed baseline, 4 pt inset. */
@Composable
fun Sparkline(samples: List<Double>, baseline: Double?, color: Color, modifier: Modifier = Modifier) {
    val ios = LocalIosColors.current
    Canvas(modifier) {
        if (samples.size < 2) return@Canvas
        val pad = 4.dp.toPx()
        var lo = samples.min(); var hi = samples.max()
        if (baseline != null) { lo = min(lo, baseline); hi = max(hi, baseline) }
        if (hi - lo < 1e-6) { hi = lo + 1.0 }
        val h = size.height - 2 * pad
        fun y(v: Double) = (pad + h * (1.0 - (v - lo) / (hi - lo))).toFloat()
        val dx = size.width / (samples.size - 1)
        if (baseline != null) {
            val yb = y(baseline)
            drawLine(
                ios.secondaryLabel.copy(alpha = 0.5f), Offset(0f, yb), Offset(size.width, yb), strokeWidth = 1.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(3.dp.toPx(), 3.dp.toPx())),
            )
        }
        val path = androidx.compose.ui.graphics.Path()
        samples.forEachIndexed { i, v -> if (i == 0) path.moveTo(0f, y(v)) else path.lineTo(i * dx, y(v)) }
        drawPath(path, color, style = Stroke(2.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
    }
}

/** `ThermalBar` (`BenchView.swift:785-808`): run-length coloured strip, height 8, capsule clipped. */
@Composable
fun ThermalBar(levels: List<Int>, modifier: Modifier = Modifier) {
    val ios = LocalIosColors.current
    Row(modifier.height(8.dp).clip(CircleShape)) {
        if (levels.isEmpty()) {
            Box(Modifier.fillMaxWidth().fillMaxHeight().background(ios.quaternaryFill))
        } else {
            // run-length encode
            val runs = ArrayList<Pair<Int, Int>>()
            for (l in levels) { if (runs.isNotEmpty() && runs.last().first == l) runs[runs.lastIndex] = l to runs.last().second + 1 else runs += l to 1 }
            for ((lvl, n) in runs) {
                Box(Modifier.weight(n.toFloat()).fillMaxHeight().background(HudColors.thermalColor(lvl)))
            }
        }
    }
}

/** The green detection count with its "dets" caption, 40 wide. */
@Composable
fun DetsCount(n: Int) {
    val ios = LocalIosColors.current
    Column(Modifier.width(40.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(0.dp)) {
        Text("$n", style = IosType.title3Rounded.tabular, color = dev.yolomaster.app.ui.theme.IosGreen, maxLines = 1)
        Text("dets", style = IosType.caption2, color = ios.secondaryLabel)
    }
}

internal val BoldWeight = FontWeight.Bold
