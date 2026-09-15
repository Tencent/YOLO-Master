package dev.yolomaster.app.ui.overlay

import android.graphics.BlurMaskFilter
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import dev.yolomaster.app.model.classLabel
import dev.yolomaster.ncnn.Detection
import kotlin.math.max
import kotlin.math.min

/** The five on-screen box styles (`Overlay.swift:7-18`); Chip is the default. */
enum class BoxStyle(val label: String) {
    Solid("Solid"), Neon("Neon"), Hud("HUD"), Chip("Chip"), Minimal("Minimal");

    /** The Kit style used when baking exports (`Overlay.swift:37-43`). */
    val kitStyle: KitStyle get() = when (this) { Neon -> KitStyle.Neon; Hud, Minimal -> KitStyle.Hud; Solid, Chip -> KitStyle.Solid }
}

/** `Annotate.swift` `BoxStyle`. */
enum class KitStyle { Hud, Solid, Neon }

/** `SegOverlay` (`Annotate.swift:10`): what a segmentation model draws. */
enum class SegOverlayMode(val label: String) { Masks("Masks"), Boxes("Boxes"), Both("Both") }

/** The shared 10-colour class palette (`Overlay.swift:25-31` == `common.cpp:235`). */
object Palette {
    private val rgb = arrayOf(
        floatArrayOf(0.98f, 0.26f, 0.30f), floatArrayOf(0.20f, 0.71f, 0.98f), floatArrayOf(0.16f, 0.85f, 0.52f),
        floatArrayOf(0.99f, 0.79f, 0.12f), floatArrayOf(0.72f, 0.40f, 0.98f), floatArrayOf(0.99f, 0.55f, 0.18f),
        floatArrayOf(0.10f, 0.83f, 0.80f), floatArrayOf(0.98f, 0.36f, 0.66f), floatArrayOf(0.55f, 0.82f, 0.28f),
        floatArrayOf(0.40f, 0.52f, 0.98f),
    )
    // packed by hand (not android.graphics.Color) so the palette is usable in plain JVM unit tests
    val colors: IntArray = IntArray(10) { i ->
        (0xFF shl 24) or ((rgb[i][0] * 255 + 0.5f).toInt() shl 16) or ((rgb[i][1] * 255 + 0.5f).toInt() shl 8) or (rgb[i][2] * 255 + 0.5f).toInt()
    }
    /** Classes whose chip is bright enough for black text (`Overlay.swift:32`). */
    private val darkText = setOf(2, 3, 6, 8)

    fun index(cls: Int): Int = ((cls % 10) + 10) % 10
    fun color(cls: Int): Int = colors[index(cls)]
    fun textIsDark(cls: Int): Boolean = index(cls) in darkText
    fun composeColor(cls: Int): androidx.compose.ui.graphics.Color = androidx.compose.ui.graphics.Color(color(cls))
}

private fun withAlpha(c: Int, a: Float): Int = Color.argb((a * 255).toInt().coerceIn(0, 255), Color.red(c), Color.green(c), Color.blue(c))

/**
 * On-screen renderer (`DetOverlay.draw`, `Overlay.swift:46-132`). Coordinates: detections are
 * in original-frame pixels; `scale`/`ox`/`oy` map them onto the canvas. Sizes are in canvas
 * pixels after multiplying the iOS point constants by [density].
 */
object ScreenOverlay {
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG)
    private val glow = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val textBounds = Rect()

    fun draw(
        canvas: Canvas, dets: List<Detection>, names: List<String>, style: BoxStyle,
        scale: Float, ox: Float, oy: Float, density: Float,
    ) {
        for (d in dets) {
            val r = RectF(ox + d.x1 * scale, oy + d.y1 * scale, ox + d.x2 * scale, oy + d.y2 * scale)
            if (r.width() < 1f || r.height() < 1f) continue
            val col = Palette.color(d.classId)
            val title = "${classLabel(names, d.classId)} ${(d.score * 100).toInt()}%"
            val corner = min(min(r.width(), r.height()) * 0.14f, 10f * density)
            when (style) {
                BoxStyle.Solid -> {
                    stroke.color = col; stroke.strokeWidth = 2.4f * density
                    canvas.drawRoundRect(r, corner, corner, stroke)
                    chipLabel(canvas, r, title, col, d.classId, density)
                }
                BoxStyle.Chip -> {
                    stroke.color = col; stroke.strokeWidth = 2f * density
                    val c3 = 3f * density
                    canvas.drawRoundRect(r, c3, c3, stroke)
                    chipLabel(canvas, r, title, col, d.classId, density)
                }
                BoxStyle.Neon -> {
                    glow.color = withAlpha(col, 0.95f); glow.strokeWidth = 2.6f * density
                    glow.maskFilter = BlurMaskFilter(8f * density, BlurMaskFilter.Blur.NORMAL)
                    canvas.drawRoundRect(r, corner, corner, glow)
                    stroke.color = col; stroke.strokeWidth = 1.2f * density
                    canvas.drawRoundRect(r, corner, corner, stroke)
                    text.typeface = Typeface.DEFAULT_BOLD; text.textSize = 11f * density; text.color = col
                    text.setShadowLayer(5f * density, 0f, 0f, withAlpha(col, 0.9f))
                    val y = max(r.top - 9f * density, 6f * density)
                    canvas.drawText(title, r.left + 2f * density, y + text.textSize * 0.35f, text)
                    text.clearShadowLayer()
                }
                BoxStyle.Hud -> {
                    fill.color = withAlpha(col, 0.08f); canvas.drawRect(r, fill)
                    stroke.color = withAlpha(col, 0.35f); stroke.strokeWidth = 1f * density; canvas.drawRect(r, stroke)
                    val arm = min(min(r.width(), r.height()) * 0.28f, 26f * density)
                    stroke.color = col; stroke.strokeWidth = 2.8f * density
                    val p = Path()
                    // four corner brackets
                    p.moveTo(r.left, r.top + arm); p.lineTo(r.left, r.top); p.lineTo(r.left + arm, r.top)
                    p.moveTo(r.right - arm, r.top); p.lineTo(r.right, r.top); p.lineTo(r.right, r.top + arm)
                    p.moveTo(r.right, r.bottom - arm); p.lineTo(r.right, r.bottom); p.lineTo(r.right - arm, r.bottom)
                    p.moveTo(r.left + arm, r.bottom); p.lineTo(r.left, r.bottom); p.lineTo(r.left, r.bottom - arm)
                    canvas.drawPath(p, stroke)
                    text.typeface = Typeface.MONOSPACE; text.textSize = 9f * density; text.color = col
                    text.isFakeBoldText = true
                    val y = max(r.top - 9f * density, 6f * density)
                    canvas.drawText(title.uppercase(), r.left + 3f * density, y + text.textSize * 0.35f, text)
                    text.isFakeBoldText = false
                }
                BoxStyle.Minimal -> {
                    stroke.color = withAlpha(Color.WHITE, 0.9f); stroke.strokeWidth = 1f * density
                    canvas.drawRect(r, stroke)
                    text.typeface = Typeface.DEFAULT; text.textSize = 9f * density; text.color = withAlpha(Color.WHITE, 0.9f)
                    val y = max(r.top - 8f * density, 5f * density)
                    canvas.drawText(title, r.left + 2f * density, y + text.textSize * 0.35f, text)
                }
            }
        }
    }

    /** `chipLabel` (`Overlay.swift:123-132`): opaque class-coloured plate above the box. */
    private fun chipLabel(canvas: Canvas, r: RectF, title: String, col: Int, cls: Int, density: Float) {
        text.typeface = Typeface.DEFAULT_BOLD; text.textSize = 11f * density
        text.getTextBounds(title, 0, title.length, textBounds)
        val textW = text.measureText(title); val textH = text.textSize
        val top = max(r.top - textH - 4f * density, 2f * density)
        val chip = RectF(r.left, top, r.left + textW + 8f * density, top + textH + 3f * density)
        fill.color = col
        val c3 = 3f * density
        canvas.drawRoundRect(chip, c3, c3, fill)
        text.color = if (Palette.textIsDark(cls)) Color.BLACK else Color.WHITE
        val baseline = chip.centerY() - (text.descent() + text.ascent()) / 2f
        canvas.drawText(title, chip.left + 4f * density, baseline, text)
    }
}

/**
 * Baked-export renderer (`Annotate.swift:29-104`): scale-aware constants in IMAGE pixels, the
 * two-space `"name  0.87"` label, luminance-picked text colour. Draws into the image itself.
 */
object Annotate {
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { typeface = Typeface.create("sans-serif", Typeface.BOLD) }
    private val glow = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }

    fun draw(canvas: Canvas, imageWidth: Int, dets: List<Detection>, names: List<String>, style: KitStyle, drawBoxes: Boolean = true) {
        if (!drawBoxes) return
        val lw = max(2f, imageWidth / 640f)
        val baseFont = max(12f, imageWidth / 95f)
        for (d in dets) {
            val r = RectF(d.x1, d.y1, d.x2, d.y2)
            if (r.width() < 1f || r.height() < 1f) continue
            val col = Palette.color(d.classId)
            val rad = min(min(r.width(), r.height()) * 0.14f, lw * 5f)
            when (style) {
                KitStyle.Solid -> { stroke.color = col; stroke.strokeWidth = lw * 1.2f; canvas.drawRoundRect(r, rad, rad, stroke) }
                KitStyle.Neon -> {
                    glow.color = withAlpha(col, 0.95f); glow.strokeWidth = lw * 1.3f
                    glow.maskFilter = BlurMaskFilter(lw * 5f, BlurMaskFilter.Blur.NORMAL)
                    canvas.drawRoundRect(r, rad, rad, glow); canvas.drawRoundRect(r, rad, rad, glow)
                    stroke.color = col; stroke.strokeWidth = lw * 1.3f; canvas.drawRoundRect(r, rad, rad, stroke)
                }
                KitStyle.Hud -> {
                    fill.color = withAlpha(col, 0.08f); canvas.drawRect(r, fill)
                    stroke.color = withAlpha(col, 0.35f); stroke.strokeWidth = lw * 0.6f; canvas.drawRect(r, stroke)
                    val arm = min(min(r.width(), r.height()) * 0.28f, lw * 22f)
                    stroke.color = col; stroke.strokeWidth = lw * 1.4f
                    val p = Path()
                    p.moveTo(r.left, r.top + arm); p.lineTo(r.left, r.top); p.lineTo(r.left + arm, r.top)
                    p.moveTo(r.right - arm, r.top); p.lineTo(r.right, r.top); p.lineTo(r.right, r.top + arm)
                    p.moveTo(r.right, r.bottom - arm); p.lineTo(r.right, r.bottom); p.lineTo(r.right - arm, r.bottom)
                    p.moveTo(r.left + arm, r.bottom); p.lineTo(r.left, r.bottom); p.lineTo(r.left, r.bottom - arm)
                    canvas.drawPath(p, stroke)
                }
            }
            // label plate
            val label = "${classLabel(names, d.classId)}  ${String.format("%.2f", d.score)}"
            text.textSize = baseFont
            val padX = baseFont * 0.5f
            val chipH = baseFont + 6f
            val tw = text.measureText(label)
            var top = r.top - chipH
            if (top < 0f) top = r.top  // flip inside when it would leave the image
            val chip = RectF(r.left, top, min(r.left + tw + 2 * padX, canvas.width.toFloat()), top + chipH)
            fill.color = withAlpha(col, 0.72f)
            val cr = chipH * 0.28f
            canvas.drawRoundRect(chip, cr, cr, fill)
            val lum = 0.299 * Color.red(col) / 255.0 + 0.587 * Color.green(col) / 255.0 + 0.114 * Color.blue(col) / 255.0
            text.color = if (lum > 0.62) Color.rgb(13, 13, 13) else Color.WHITE
            val baseline = chip.centerY() - (text.descent() + text.ascent()) / 2f
            canvas.drawText(label, chip.left + padX, baseline, text)
        }
    }
}
