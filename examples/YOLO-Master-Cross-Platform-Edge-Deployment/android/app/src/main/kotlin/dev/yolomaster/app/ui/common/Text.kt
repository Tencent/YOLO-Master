package dev.yolomaster.app.ui.common

import androidx.compose.ui.text.TextStyle

/** iOS `.monospacedDigit()`: tabular figures so live numbers don't jitter. */
val TextStyle.tabular: TextStyle get() = copy(fontFeatureSettings = "tnum")
