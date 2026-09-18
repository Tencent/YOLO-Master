package dev.yolomaster.app.system

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.provider.OpenableColumns
import androidx.exifinterface.media.ExifInterface
import kotlin.math.max

/** A decoded, upright, size-capped image plus the picker's display name and type. */
data class LoadedImage(val bitmap: Bitmap, val name: String, val type: String)

/**
 * The Photo tab's decoder (`PhotoTestView.swift:344-402`): EXIF orientation applied, long side
 * capped at [maxSide] (2048 on iOS), display name from the provider (`IMG_n` fallback), type =
 * the extension uppercased.
 */
object ImageLoader {
    fun load(ctx: Context, uri: Uri, index: Int, maxSide: Int = 2048): LoadedImage? {
        val (name, ext) = displayName(ctx, uri, index)
        val bmp = try {
            if (Build.VERSION.SDK_INT >= 28) decodeModern(ctx, uri, maxSide) else decodeLegacy(ctx, uri, maxSide)
        } catch (t: Throwable) { null } ?: return null
        return LoadedImage(bmp, name, ext)
    }

    private fun displayName(ctx: Context, uri: Uri, index: Int): Pair<String, String> {
        var display: String? = null
        try {
            ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) display = c.getString(0)
            }
        } catch (_: Throwable) {}
        val d = display
        if (d.isNullOrBlank()) {
            val ext = ctx.contentResolver.getType(uri)?.substringAfter('/')?.uppercase() ?: "-"
            return "IMG_${index + 1}" to ext
        }
        val dot = d.lastIndexOf('.')
        return if (dot > 0) d.substring(0, dot) to d.substring(dot + 1).uppercase() else d to "-"
    }

    private fun decodeModern(ctx: Context, uri: Uri, maxSide: Int): Bitmap {
        val src = ImageDecoder.createSource(ctx.contentResolver, uri)
        return ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
            val w = info.size.width; val h = info.size.height
            val longSide = max(w, h)
            if (longSide > maxSide) {
                val s = maxSide.toFloat() / longSide
                decoder.setTargetSize((w * s).toInt().coerceAtLeast(1), (h * s).toInt().coerceAtLeast(1))
            }
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.isMutableRequired = false
        }.let { if (it.config == Bitmap.Config.ARGB_8888) it else it.copy(Bitmap.Config.ARGB_8888, false) }
    }

    private fun decodeLegacy(ctx: Context, uri: Uri, maxSide: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) } ?: return null
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888 }
        var bmp = ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) } ?: return null
        val longSide = max(bmp.width, bmp.height)
        if (longSide > maxSide) {
            val s = maxSide.toFloat() / longSide
            bmp = Bitmap.createScaledBitmap(bmp, (bmp.width * s).toInt(), (bmp.height * s).toInt(), true)
        }
        val orientation = try {
            ctx.contentResolver.openInputStream(uri)?.use { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
        } catch (_: Throwable) { null } ?: ExifInterface.ORIENTATION_NORMAL
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
            else -> return bmp
        }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }
}
