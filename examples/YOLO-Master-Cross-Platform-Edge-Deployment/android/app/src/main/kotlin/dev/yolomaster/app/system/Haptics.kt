package dev.yolomaster.app.system

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.HapticFeedbackConstants
import android.view.View

/**
 * The iOS haptics map (`UIImpactFeedbackGenerator` light/medium/heavy,
 * `UINotificationFeedbackGenerator` success/error) on Android primitives.
 */
class Haptics(private val view: View) {
    private val vibrator: Vibrator? by lazy {
        val ctx = view.context
        if (Build.VERSION.SDK_INT >= 31) (ctx.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)?.defaultVibrator
        else @Suppress("DEPRECATION") ctx.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    }

    fun light() { view.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP) }
    fun medium() { view.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK) }

    fun heavy() {
        if (Build.VERSION.SDK_INT >= 29) {
            vibrator?.vibrate(VibrationEffect.createPredefined(VibrationEffect.EFFECT_HEAVY_CLICK))
        } else view.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
    }

    /** iOS notification haptic: a short double pulse for success, a longer triple for error. */
    fun success() {
        if (Build.VERSION.SDK_INT >= 26) vibrator?.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 18, 60, 30), -1))
        else medium()
    }

    fun error() {
        if (Build.VERSION.SDK_INT >= 26) vibrator?.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 40, 50, 40, 50, 60), -1))
        else heavy()
    }
}
