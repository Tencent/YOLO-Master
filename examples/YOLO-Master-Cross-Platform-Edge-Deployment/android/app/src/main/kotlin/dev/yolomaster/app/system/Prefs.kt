package dev.yolomaster.app.system

import android.content.Context
import android.content.SharedPreferences
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.ncnn.Runtime

/** `@AppStorage` equivalents. Single SharedPreferences file, read synchronously (tiny). */
object Prefs {
    private const val FILE = "yolomaster"
    const val ALLOW_CPU = "allowCPU"
    /** The last runtime the user picked anywhere (the initial value of every tab's picker). */
    const val RUNTIME = "runtime"
    /** QNN `htp_performance_mode` for the NPU: "burst" (Live) | "sustained" (long runs). */
    const val ORT_PERF_MODE = "ortPerfMode"
    /** Load the QDQ `model-a16w8.onnx` / `model-a8w8.onnx` sibling on the NPU when the model has one. */
    const val ORT_PREFER_QUANT = "ortPreferQuant"
    private const val CHOICE_PREFIX = "choice:"

    fun get(ctx: Context): SharedPreferences = ctx.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    /** iOS default is false (Core ML CPU crashes); on Android CPU is the certified path -> true. */
    fun allowCPU(ctx: Context): Boolean = get(ctx).getBoolean(ALLOW_CPU, true)
    fun setAllowCPU(ctx: Context, v: Boolean) = get(ctx).edit().putBoolean(ALLOW_CPU, v).apply()

    fun runtime(ctx: Context): Runtime = runtimeOf(get(ctx).getString(RUNTIME, null)) ?: Runtime.NCNN
    fun setRuntime(ctx: Context, r: Runtime) = get(ctx).edit().putString(RUNTIME, r.name).apply()

    /** "burst" | "sustained"; [Detector] maps it to the QNN option string. */
    fun ortPerfMode(ctx: Context): String = get(ctx).getString(ORT_PERF_MODE, PERF_BURST) ?: PERF_BURST
    fun preferQuant(ctx: Context): Boolean = get(ctx).getBoolean(ORT_PREFER_QUANT, false)

    /**
     * The user's explicit runtime x unit pick for one model (made through the Live / Photo pickers).
     * It wins over the measured default for as long as it exists; null = never picked by hand.
     */
    fun choiceFor(ctx: Context, modelId: String): Pair<Runtime, ComputeChoice>? {
        val s = get(ctx).getString(CHOICE_PREFIX + modelId, null) ?: return null
        val (r, c) = s.split('|').let { if (it.size == 2) it[0] to it[1] else return null }
        val runtime = runtimeOf(r) ?: return null
        val compute = ComputeChoice.entries.firstOrNull { it.name == c } ?: return null
        return runtime to compute
    }

    fun setChoice(ctx: Context, modelId: String, r: Runtime, c: ComputeChoice) =
        get(ctx).edit().putString(CHOICE_PREFIX + modelId, "${r.name}|${c.name}").apply()

    fun clearChoice(ctx: Context, modelId: String) = get(ctx).edit().remove(CHOICE_PREFIX + modelId).apply()

    /** Forget every hand pick (Settings > Measured defaults > Reset): the measured defaults rule again. */
    fun clearAllChoices(ctx: Context) {
        val p = get(ctx)
        val e = p.edit()
        p.all.keys.filter { it.startsWith(CHOICE_PREFIX) }.forEach { e.remove(it) }
        e.apply()
    }

    const val PERF_BURST = "burst"
    const val PERF_SUSTAINED = "sustained"
    val perfModes = listOf(PERF_BURST, PERF_SUSTAINED)

    private fun runtimeOf(name: String?): Runtime? = Runtime.entries.firstOrNull { it.name == name }
}

/** A Compose state mirrored into SharedPreferences, updated when the key changes elsewhere. */
@Composable
fun rememberBoolPref(key: String, default: Boolean): MutableState<Boolean> {
    val ctx = LocalContext.current
    val prefs = remember { Prefs.get(ctx) }
    val state = remember { mutableStateOf(prefs.getBoolean(key, default)) }
    androidx.compose.runtime.DisposableEffect(key) {
        val l = SharedPreferences.OnSharedPreferenceChangeListener { p, k -> if (k == key) state.value = p.getBoolean(key, default) }
        prefs.registerOnSharedPreferenceChangeListener(l)
        onDispose { prefs.unregisterOnSharedPreferenceChangeListener(l) }
    }
    return object : MutableState<Boolean> by state {
        override var value: Boolean
            get() = state.value
            set(v) { state.value = v; prefs.edit().putBoolean(key, v).apply() }
    }
}

/** The String twin of [rememberBoolPref] (the ONNX perf-mode segmented control). */
@Composable
fun rememberStringPref(key: String, default: String): MutableState<String> {
    val ctx = LocalContext.current
    val prefs = remember { Prefs.get(ctx) }
    val state = remember { mutableStateOf(prefs.getString(key, default) ?: default) }
    androidx.compose.runtime.DisposableEffect(key) {
        val l = SharedPreferences.OnSharedPreferenceChangeListener { p, k -> if (k == key) state.value = p.getString(key, default) ?: default }
        prefs.registerOnSharedPreferenceChangeListener(l)
        onDispose { prefs.unregisterOnSharedPreferenceChangeListener(l) }
    }
    return object : MutableState<String> by state {
        override var value: String
            get() = state.value
            set(v) { state.value = v; prefs.edit().putString(key, v).apply() }
    }
}
