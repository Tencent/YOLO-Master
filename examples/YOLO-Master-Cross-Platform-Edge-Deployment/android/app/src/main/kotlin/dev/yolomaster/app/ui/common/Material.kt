package dev.yolomaster.app.ui.common

import androidx.compose.foundation.background
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.chrisbanes.haze.HazeState
import dev.chrisbanes.haze.HazeStyle
import dev.chrisbanes.haze.haze
import dev.chrisbanes.haze.hazeChild
import dev.yolomaster.app.ui.theme.LocalIosColors

/**
 * The iOS `.ultraThinMaterial`: a blurred, lightly tinted backdrop. Haze blurs whatever is drawn
 * under the `Modifier.hazeBackdrop()` (camera preview, photo grid, bench list) behind each
 * `Modifier.materialCard()`; on devices without RenderEffect it degrades to the tinted scrim.
 */
val LocalHazeState = compositionLocalOf<HazeState?> { null }

@Composable
fun rememberHazeState(): HazeState = remember { HazeState() }

/** Put on the content that sits BEHIND the floating cards. */
fun Modifier.hazeBackdrop(state: HazeState?): Modifier = if (state == null) this else this.haze(state)

@Composable
private fun materialStyle(): HazeStyle {
    val ios = LocalIosColors.current
    val tint = if (ios.isDark) Color(0xFF1C1C1E).copy(alpha = 0.55f) else Color(0xFFF7F7F9).copy(alpha = 0.55f)
    return HazeStyle(tint = tint, blurRadius = 20.dp, noiseFactor = 0.03f)
}

/** A floating card with the material look; `shape` follows the iOS corner radii (8/12/14/16). */
@Composable
fun Modifier.materialCard(shape: Shape = RoundedCornerShape(12.dp)): Modifier {
    val state = LocalHazeState.current
    val style = materialStyle()
    return if (state != null) this.clip(shape).hazeChild(state, shape, style)
    else this.clip(shape).background(style.tint)
}

@Composable
fun Modifier.materialCard(radius: Dp): Modifier = materialCard(RoundedCornerShape(radius))
