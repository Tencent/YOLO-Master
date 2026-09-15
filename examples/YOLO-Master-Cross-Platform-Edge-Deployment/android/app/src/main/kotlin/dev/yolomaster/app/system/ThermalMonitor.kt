package dev.yolomaster.app.system

import android.content.Context
import android.os.Build
import android.os.PowerManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/** iOS `ProcessInfo.thermalState` levels: 0 nominal, 1 fair, 2 serious, 3 critical. */
data class ThermalState(val level: Int, val known: Boolean)

/**
 * `PowerManager.addThermalStatusListener` (API 29+). Below 29 the level stays 0 and is flagged
 * unknown so the tachometer renders gray, like iOS's `@unknown default`.
 */
class ThermalMonitor(context: Context) {
    private val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private val _state = MutableStateFlow(ThermalState(0, known = Build.VERSION.SDK_INT >= 29))
    val state: StateFlow<ThermalState> = _state

    private val listener = if (Build.VERSION.SDK_INT >= 29) PowerManager.OnThermalStatusChangedListener { s -> _state.value = ThermalState(map(s), true) } else null

    fun start() {
        if (Build.VERSION.SDK_INT >= 29 && listener != null) {
            _state.value = ThermalState(map(pm.currentThermalStatus), true)
            pm.addThermalStatusListener(listener)
        }
    }

    fun stop() {
        if (Build.VERSION.SDK_INT >= 29 && listener != null) pm.removeThermalStatusListener(listener)
    }

    companion object {
        /** NONE/LIGHT -> nominal, MODERATE -> fair, SEVERE -> serious, CRITICAL+ -> critical. */
        fun map(status: Int): Int = when {
            status <= PowerManager.THERMAL_STATUS_LIGHT -> 0
            status == PowerManager.THERMAL_STATUS_MODERATE -> 1
            status == PowerManager.THERMAL_STATUS_SEVERE -> 2
            else -> 3
        }
    }
}
