package dev.yolomaster.app.system

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaScannerConnection
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Saves an annotated frame to the user's gallery under `Pictures/YOLO-Master` (the iOS app
 * writes to the Photos library). MediaStore + IS_PENDING on API 29+, a plain file plus a media
 * scan below (WRITE_EXTERNAL_STORAGE must have been granted by the caller there).
 */
object GallerySaver {
    private const val ALBUM = "YOLO-Master"

    /** Returns true on success. JPEG quality 92. */
    fun save(ctx: Context, bitmap: Bitmap, stem: String = "YM"): Boolean { return try {
        val name = "${stem}_${SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())}.jpg"
        if (Build.VERSION.SDK_INT >= 29) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/" + ALBUM)
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val resolver = ctx.contentResolver
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return false
            resolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 92, it) } ?: return false
            values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            true
        } else {
            @Suppress("DEPRECATION")
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), ALBUM)
            dir.mkdirs()
            val f = File(dir, name)
            f.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 92, it) }
            MediaScannerConnection.scanFile(ctx, arrayOf(f.absolutePath), arrayOf("image/jpeg"), null)
            true
        }
    } catch (t: Throwable) {
        android.util.Log.w("GallerySaver", "save failed", t); false
    } }
}
