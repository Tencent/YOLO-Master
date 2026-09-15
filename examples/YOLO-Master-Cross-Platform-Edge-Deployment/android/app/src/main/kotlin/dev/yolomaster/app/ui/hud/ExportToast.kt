package dev.yolomaster.app.ui.hud

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathMeasure
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import dev.yolomaster.app.ui.common.materialCard
import dev.yolomaster.app.ui.theme.IosGreen
import dev.yolomaster.app.ui.theme.IosRed
import dev.yolomaster.app.ui.theme.IosType
import dev.yolomaster.app.ui.theme.LocalIosColors

/** One toast request; the screen holds `ToastState?` and clears it on dismiss. */
data class Toast(val message: String, val success: Boolean, val autoDismissMs: Long = if (success) 1600 else 2400, val gen: Int = 0)

/**
 * `ExportToast` (`StatsHUD.swift:347-385`): padding 20, radius 16, a 46-pt ring and a 20-pt tick
 * (or cross) that draw themselves in over 0.45 s, caption below. Tap dismisses early.
 */
@Composable
fun ExportToast(toast: Toast, onDismiss: () -> Unit) {
    val ios = LocalIosColors.current
    var drawn by remember(toast.gen) { mutableStateOf(false) }
    val progress by animateFloatAsState(if (drawn) 1f else 0f, tween(450, delayMillis = 50), label = "toastDraw")
    LaunchedEffect(toast.gen) {
        drawn = true
        kotlinx.coroutines.delay(toast.autoDismissMs)
        onDismiss()
    }
    val color = if (toast.success) IosGreen else IosRed
    Column(
        Modifier
            .materialCard(16.dp)
            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) { onDismiss() }
            .padding(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Box(Modifier.size(46.dp), contentAlignment = Alignment.Center) {
            Canvas(Modifier.size(46.dp)) {
                val stroke = 3.dp.toPx()
                val d = size.minDimension - stroke
                drawArc(color, -90f, 360f * progress, false, Offset(stroke / 2, stroke / 2), Size(d, d), style = Stroke(stroke, cap = StrokeCap.Round))
            }
            Canvas(Modifier.size(20.dp)) {
                val w = size.width; val h = size.height
                // One path per contour: Compose's PathMeasure walks a single contour.
                val contours: List<Path> = if (toast.success) listOf(
                    // TickShape: (minX, midY + h*0.08) -> (minX + w*0.36, maxY - h*0.08) -> (maxX, minY + h*0.1)
                    Path().apply { moveTo(0f, h / 2 + h * 0.08f); lineTo(w * 0.36f, h - h * 0.08f); lineTo(w, h * 0.1f) },
                ) else listOf(
                    Path().apply { moveTo(0f, 0f); lineTo(w, h) },
                    Path().apply { moveTo(w, 0f); lineTo(0f, h) },
                )
                for (c in contours) {
                    val pm = PathMeasure().apply { setPath(c, false) }
                    val partial = Path()
                    pm.getSegment(0f, pm.length * progress, partial, true)
                    drawPath(partial, color, style = Stroke(3.5.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
                }
            }
        }
        Text(toast.message, style = IosType.caption, color = ios.label)
    }
}
