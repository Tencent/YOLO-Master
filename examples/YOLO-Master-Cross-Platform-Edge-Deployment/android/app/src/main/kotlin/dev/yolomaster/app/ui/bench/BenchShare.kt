package dev.yolomaster.app.ui.bench

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.core.content.FileProvider
import java.io.File

/**
 * The iOS `share(url)` (`BenchView.swift:768-779`, `UIActivityViewController`): write the CSV to
 * `cacheDir/share/<name>` (the one path `res/xml/file_paths.xml` exposes) and open the system
 * share sheet through the app's `FileProvider`.
 */
object BenchShare {
    /** Returns false when the file could not be written or no app can receive it. */
    fun shareCsv(ctx: Context, fileName: String, csv: String): Boolean = try {
        val dir = File(ctx.cacheDir, "share").apply { mkdirs() }
        val file = File(dir, fileName)
        file.writeText(csv)
        val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/csv"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, fileName)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        val chooser = Intent.createChooser(send, fileName)
        if (ctx !is Activity) chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ctx.startActivity(chooser)
        true
    } catch (t: Throwable) {
        Log.w("BenchShare", "share failed: ${t.message}")
        false
    }
}
