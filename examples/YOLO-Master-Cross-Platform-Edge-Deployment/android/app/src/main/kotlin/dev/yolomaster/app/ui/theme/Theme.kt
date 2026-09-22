package dev.yolomaster.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.PlatformTextStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.LineHeightStyle
import androidx.compose.ui.unit.sp

/*
 * Visual tokens mirroring the iOS app (SwiftUI semantic colors + SF text styles). Dynamic color
 * is deliberately OFF: the accent is iOS system blue and the class palette is fixed, so the two
 * apps read identically.
 */

/** iOS system blue (light / dark). */
val AccentLight = Color(0xFF007AFF)
val AccentDark = Color(0xFF0A84FF)

/** StatsHUD ramp bases (`StatsHUD.swift:9-12`). */
val RampRed = Color(0xFFF54236)
val RampOrange = Color(0xFFFF9400)
val RampGreen = Color(0xFF33D64A)
val RampPurple = Color(0xFFB052DE)

/** iOS explicit colors used by the app. */
val IosYellow = Color(0xFFFFCC00)
val IosRed = Color(0xFFFF3B30)
val IosOrange = Color(0xFFFF9500)
val IosGreen = Color(0xFF34C759)
val IosBlue = Color(0xFF007AFF)

/**
 * iOS semantic text/fill colors that Material3 has no exact slot for. `.secondary`, `.tertiary`
 * and `.quaternary` are label opacities on the primary color; `secondarySystemBackground` is the
 * grouped-list card fill.
 */
class IosColors(
    val label: Color,
    val secondaryLabel: Color,
    val tertiaryLabel: Color,
    val quaternaryFill: Color,
    val systemBackground: Color,
    val secondarySystemBackground: Color,
    val groupedBackground: Color,
    val separator: Color,
    val accent: Color,
    val isDark: Boolean,
)

val LocalIosColors = staticCompositionLocalOf {
    IosColors(
        label = Color.Black, secondaryLabel = Color(0x99000000), tertiaryLabel = Color(0x4D000000),
        quaternaryFill = Color(0x2E000000), systemBackground = Color.White,
        secondarySystemBackground = Color(0xFFF2F2F7), groupedBackground = Color(0xFFF2F2F7),
        separator = Color(0x4A3C3C43), accent = AccentLight, isDark = false,
    )
}

private val LightIos = IosColors(
    label = Color.Black,
    secondaryLabel = Color(0x993C3C43),
    tertiaryLabel = Color(0x4D3C3C43),
    quaternaryFill = Color(0x2E787880),
    systemBackground = Color.White,
    secondarySystemBackground = Color(0xFFF2F2F7),
    groupedBackground = Color(0xFFF2F2F7),
    separator = Color(0x4A3C3C43),
    accent = AccentLight,
    isDark = false,
)

private val DarkIos = IosColors(
    label = Color.White,
    secondaryLabel = Color(0x99EBEBF5),
    tertiaryLabel = Color(0x4DEBEBF5),
    quaternaryFill = Color(0x2E787880),
    systemBackground = Color.Black,
    secondarySystemBackground = Color(0xFF1C1C1E),
    groupedBackground = Color.Black,
    separator = Color(0xA6545458),
    accent = AccentDark,
    isDark = true,
)

private fun schemeFor(ios: IosColors): ColorScheme = if (ios.isDark) darkColorScheme(
    primary = ios.accent, onPrimary = Color.White,
    background = ios.systemBackground, onBackground = ios.label,
    surface = ios.systemBackground, onSurface = ios.label,
    surfaceVariant = ios.secondarySystemBackground, onSurfaceVariant = ios.secondaryLabel,
    outline = ios.separator, error = IosRed,
) else lightColorScheme(
    primary = ios.accent, onPrimary = Color.White,
    background = ios.systemBackground, onBackground = ios.label,
    surface = ios.systemBackground, onSurface = ios.label,
    surfaceVariant = ios.secondarySystemBackground, onSurfaceVariant = ios.secondaryLabel,
    outline = ios.separator, error = IosRed,
)

/** iOS text styles at their default (Large) sizes. */
object IosType {
    private fun style(size: Int, weight: FontWeight = FontWeight.Normal, family: FontFamily = FontFamily.Default, lineHeight: Int = size + 5) =
        TextStyle(
            fontSize = size.sp, fontWeight = weight, fontFamily = family, lineHeight = lineHeight.sp,
            platformStyle = PlatformTextStyle(includeFontPadding = false),
            lineHeightStyle = LineHeightStyle(LineHeightStyle.Alignment.Center, LineHeightStyle.Trim.None),
        )
    val largeTitle = style(34, FontWeight.Bold, lineHeight = 41)
    val title3 = style(20, lineHeight = 25)
    val title3Bold = style(20, FontWeight.Bold, lineHeight = 25)
    val headline = style(17, FontWeight.SemiBold, lineHeight = 22)
    val body = style(17, lineHeight = 22)
    val callout = style(16, lineHeight = 21)
    val calloutSemibold = style(16, FontWeight.SemiBold, lineHeight = 21)
    val subheadline = style(15, lineHeight = 20)
    val subheadlineBold = style(15, FontWeight.Bold, lineHeight = 20)
    val footnote = style(13, lineHeight = 18)
    val footnoteBold = style(13, FontWeight.Bold, lineHeight = 18)
    val footnoteSemibold = style(13, FontWeight.SemiBold, lineHeight = 18)
    val caption = style(12, lineHeight = 16)
    val captionSemibold = style(12, FontWeight.SemiBold, lineHeight = 16)
    val captionBold = style(12, FontWeight.Bold, lineHeight = 16)
    val caption2 = style(11, lineHeight = 13)
    val caption2Bold = style(11, FontWeight.Bold, lineHeight = 13)
    /** `.system(.title3, design: .rounded).bold()` — Roboto has no rounded face; bold stands in. */
    val title3Rounded = style(20, FontWeight.Bold, lineHeight = 25)
    val footnoteRounded = style(13, FontWeight.Bold, lineHeight = 18)
    val mono11 = style(11, family = FontFamily.Monospace, lineHeight = 14)
    val mono9Bold = style(9, FontWeight.Bold, FontFamily.Monospace, lineHeight = 11)
    val label9Medium = style(9, FontWeight.Medium, lineHeight = 11)
}

private val AppTypography = Typography(
    bodyLarge = IosType.body,
    bodyMedium = IosType.callout,
    bodySmall = IosType.footnote,
    titleLarge = IosType.title3Bold,
    titleMedium = IosType.headline,
    titleSmall = IosType.subheadlineBold,
    labelLarge = IosType.captionSemibold,
    labelMedium = IosType.caption,
    labelSmall = IosType.caption2,
)

@Composable
fun YoloMasterTheme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    val ios = if (darkTheme) DarkIos else LightIos
    androidx.compose.runtime.CompositionLocalProvider(LocalIosColors provides ios) {
        MaterialTheme(colorScheme = schemeFor(ios), typography = AppTypography, content = content)
    }
}
