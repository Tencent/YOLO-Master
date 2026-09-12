package dev.yolomaster.app

import android.os.PowerManager
import dev.yolomaster.app.model.Naming
import dev.yolomaster.app.model.classLabel
import dev.yolomaster.app.model.cocoNames
import dev.yolomaster.app.system.ThermalMonitor
import dev.yolomaster.app.ui.hud.HudColors
import dev.yolomaster.app.ui.overlay.Palette
import dev.yolomaster.app.ui.theme.RampGreen
import dev.yolomaster.app.ui.theme.RampOrange
import dev.yolomaster.app.ui.theme.RampPurple
import dev.yolomaster.app.ui.theme.RampRed
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Pure-logic checks of the pieces that must match the iOS app exactly. */
class UiKitTest {
    @Test fun naming_matches_models_swift() {
        assertEquals("YOLO-Master-v0.1-seg-n", Naming.fullName("v0.1-seg-n_ncnn"))
        assertEquals("v0.1-s...", Naming.shortID("v0.1-seg-n_ncnn"))
        assertEquals("YOLO-Master-p03", Naming.fullName("yolo-master-p03_ncnn"))   // prefix never doubled
        assertEquals("p03", Naming.shortID("YOLO_Master_p03"))
        assertEquals("moa-n", Naming.shortID("moa-n_ncnn"))
        assertEquals("YOLO-Master-v0.1-seg-n-int8", Naming.fullName("v0.1-seg-n-int8_ncnn"))
    }

    @Test fun coco_fallback_for_digit_names() {
        assertEquals(80, cocoNames.size)
        assertEquals("person", classLabel(listOf("0", "1"), 0))
        assertEquals("car", classLabel(emptyList(), 2))
        assertEquals("pedestrian", classLabel(listOf("pedestrian", "people"), 0))
        assertEquals("99", classLabel(emptyList(), 99))
    }

    @Test fun fps_bands_are_pure() {
        assertEquals(RampRed, HudColors.fpsColor(9.9)); assertEquals(RampOrange, HudColors.fpsColor(10.0))
        assertEquals(RampGreen, HudColors.fpsColor(20.0)); assertEquals(RampPurple, HudColors.fpsColor(29.5))
        assertEquals(RampPurple, HudColors.msColor(29.9)); assertEquals(RampGreen, HudColors.msColor(30.0))
        assertEquals(RampOrange, HudColors.msColor(50.0)); assertEquals(RampRed, HudColors.msColor(100.0))
    }

    @Test fun stage_color_lerps_green_to_orange_over_15ms() {
        assertEquals(RampGreen, HudColors.stageColor(20.0))
        assertEquals(RampOrange, HudColors.stageColor(35.0))
        assertEquals(RampGreen, HudColors.stageColor(50.0, greenUntil = 50.0))
        val mid = HudColors.stageColor(27.5)
        assertTrue(mid != RampGreen && mid != RampOrange)
    }

    @Test fun palette_matches_the_kit() {
        assertEquals(0xFFFA424D.toInt(), Palette.color(0)); assertEquals(0xFF6685FA.toInt(), Palette.color(9))
        assertEquals(Palette.color(3), Palette.color(13)); assertEquals(Palette.color(9), Palette.color(-1))
        assertTrue(Palette.textIsDark(2) && Palette.textIsDark(3) && Palette.textIsDark(6) && Palette.textIsDark(8))
        assertTrue(!Palette.textIsDark(0))
    }

    @Test fun thermal_mapping() {
        assertEquals(0, ThermalMonitor.map(PowerManager.THERMAL_STATUS_NONE))
        assertEquals(0, ThermalMonitor.map(PowerManager.THERMAL_STATUS_LIGHT))
        assertEquals(1, ThermalMonitor.map(PowerManager.THERMAL_STATUS_MODERATE))
        assertEquals(2, ThermalMonitor.map(PowerManager.THERMAL_STATUS_SEVERE))
        assertEquals(3, ThermalMonitor.map(PowerManager.THERMAL_STATUS_CRITICAL))
        assertEquals(3, ThermalMonitor.map(PowerManager.THERMAL_STATUS_SHUTDOWN))
    }
}
