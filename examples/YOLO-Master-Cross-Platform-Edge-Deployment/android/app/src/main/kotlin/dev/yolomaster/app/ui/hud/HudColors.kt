package dev.yolomaster.app.ui.hud

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import dev.yolomaster.app.ui.theme.IosBlue
import dev.yolomaster.app.ui.theme.IosGreen
import dev.yolomaster.app.ui.theme.IosOrange
import dev.yolomaster.app.ui.theme.IosRed
import dev.yolomaster.app.ui.theme.RampGreen
import dev.yolomaster.app.ui.theme.RampOrange
import dev.yolomaster.app.ui.theme.RampPurple
import dev.yolomaster.app.ui.theme.RampRed

/** Colour ramps of `StatsHUD.swift:9-53`, reproduced band for band. */
object HudColors {
    /** Pure bands, no blending: <10 red, <20 orange, <29.5 green, >=29.5 purple (model outruns the camera). */
    fun fpsColor(fps: Double): Color = when {
        fps < 10 -> RampRed
        fps < 20 -> RampOrange
        fps < 29.5 -> RampGreen
        else -> RampPurple
    }

    /** <30 purple, <50 green, <100 orange, else red. */
    fun msColor(ms: Double): Color = when {
        ms < 30 -> RampPurple
        ms < 50 -> RampGreen
        ms < 100 -> RampOrange
        else -> RampRed
    }

    /** Continuous green -> orange over `(ms - greenUntil) / 15`, clamped. */
    fun stageColor(ms: Double, greenUntil: Double = 20.0): Color {
        val t = ((ms - greenUntil) / 15.0).coerceIn(0.0, 1.0).toFloat()
        return lerp(RampGreen, RampOrange, t)
    }

    /** Thermal level -> colour: 0 blue (nominal), 1 green (fair), 2 orange (serious), 3 red (critical). */
    fun thermalColor(level: Int, known: Boolean = true): Color = when {
        !known -> Color.Gray
        level <= 0 -> IosBlue
        level == 1 -> IosGreen
        level == 2 -> IosOrange
        else -> IosRed
    }

    val inactiveDial: Color = Color.Gray.copy(alpha = 0.5f)
}
