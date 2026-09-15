package dev.yolomaster.app.model

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import dev.yolomaster.ncnn.Runtime
import dev.yolomaster.ncnn.YoloMasterNcnn
import java.io.File
import java.util.zip.ZipInputStream

/** Runtime capability bits (`YoloMasterNcnn.capabilities`), pure so the catalog rules unit-test without the native lib. */
object Caps {
    const val NCNN = 1
    const val ORT = 2
    const val QNN = 4
    fun hasOrt(caps: Int): Boolean = (caps and ORT) != 0
    fun hasQnn(caps: Int): Boolean = (caps and QNN) != 0
    /** The live bits of this build on this device (loads the native library on first use). */
    val device: Int get() = YoloMasterNcnn.capabilities
}

/** Which compute unit a model may run on; the app shows this in the compute picker. */
enum class ComputeChoice(val label: String) {
    GPU("GPU"), CPU("CPU"), NPU("NPU");

    companion object {
        /**
         * The units [runtime] can honour for [model] on a device with [caps], in picker order
         * (the accelerator first). ncnn: GPU + CPU, CPU hidden behind the Settings toggle as on
         * iOS, and INT8 siblings CPU-only regardless. ONNX: the Hexagon NPU (QNN) when the device
         * has it, plus the CPU EP - always listed, since it is the fallback ORT itself uses and
         * hiding it would leave a QNN-less phone with an empty picker.
         */
        fun available(runtime: Runtime, allowCPU: Boolean, model: BundledModel?, caps: Int): List<ComputeChoice> = when (runtime) {
            Runtime.NCNN -> when {
                model?.cpuOnly == true -> listOf(CPU)
                allowCPU -> listOf(GPU, CPU)
                else -> listOf(GPU)
            }
            Runtime.ONNX -> if (Caps.hasQnn(caps)) listOf(NPU, CPU) else listOf(CPU)
        }

        /** The ncnn rule with the device's capabilities (the pre-ONNX signature). */
        fun available(allowCPU: Boolean, model: BundledModel?): List<ComputeChoice> =
            available(Runtime.NCNN, allowCPU, model, Caps.device)
    }
}

/**
 * The runtimes [model] can be loaded on by THIS build: its own file pairs ([BundledModel.runtimes])
 * minus ONNX when ONNX Runtime is not compiled in. ncnn first: it is the certified path.
 */
fun runtimesAvailable(model: BundledModel?, caps: Int): List<Runtime> =
    (model?.runtimes ?: listOf(Runtime.NCNN)).filter { it != Runtime.ONNX || Caps.hasOrt(caps) }.ifEmpty { listOf(Runtime.NCNN) }

/**
 * One model directory: `metadata.yaml` plus `model.ncnn.param` + `model.ncnn.bin` ([hasNcnn])
 * and / or `model.onnx` ([hasOnnx], with an optional QDQ sibling [onnxQuant]). `id` is the
 * directory name, which is also the runtime's identity for INT8 siblings.
 */
data class BundledModel(
    val id: String,
    val dir: File,
    val isSeg: Boolean,
    val isCustom: Boolean,
    val hasNcnn: Boolean = true,
    val hasOnnx: Boolean = false,
    /** "a16w8" | "a8w8" when `model-<quant>.onnx` exists beside `model.onnx` (the NPU's preferred numerics); null otherwise. */
    val onnxQuant: String? = null,
) {
    val isInt8: Boolean get() = id.endsWith("-int8_ncnn")
    /** ncnn has no Vulkan int8 kernels: INT8 entries force CPU (the runtime would too). */
    val cpuOnly: Boolean get() = isInt8
    val fullName: String get() = Naming.fullName(id)
    val shortID: String get() = Naming.shortID(id)
    /** The runtimes this directory has files for, ncnn first. */
    val runtimes: List<Runtime> get() = listOfNotNull(Runtime.NCNN.takeIf { hasNcnn }, Runtime.ONNX.takeIf { hasOnnx })
    /** The ONNX file [dev.yolomaster.ncnn.Precision.INT8] would load (null when there is none). */
    val onnxQuantFile: File? get() = onnxQuant?.let { File(dir, "model-$it.onnx") }

    companion object {
        /**
         * iOS default: the first model whose stem mentions "seg", else the first. Reduced-input
         * variants ("-416", "-320") are opt-in fast entries (seg-N@416 costs ~6 pt box mAP on the
         * COCO smoke), so they never win the default even though they sort first.
         */
        fun preferred(models: List<BundledModel>): BundledModel? {
            val reduced = Regex("-(416|320|256)(_|$)")
            return models.firstOrNull { it.id.contains("seg", ignoreCase = true) && !reduced.containsMatchIn(it.id) }
                ?: models.firstOrNull { it.id.contains("seg", ignoreCase = true) }
                ?: models.firstOrNull { !reduced.containsMatchIn(it.id) }
                ?: models.firstOrNull()
        }

        /** Catalog rule: a directory is a model when it has the ncnn pair or an ONNX graph. */
        fun isModelDir(dir: File): Boolean = hasNcnnPair(dir) || File(dir, ONNX).isFile
        internal fun hasNcnnPair(dir: File): Boolean = File(dir, NCNN_PARAM).isFile && File(dir, NCNN_BIN).isFile

        const val NCNN_PARAM = "model.ncnn.param"
        const val NCNN_BIN = "model.ncnn.bin"
        const val ONNX = "model.onnx"
        /** The QDQ siblings in the order the runtime tries them for INT8 (`jni_bridge.cpp`). */
        val ONNX_QUANTS = listOf("a16w8", "a8w8")
    }
}

/**
 * Discovery of bundled (APK assets, copied once to `filesDir/models`) and custom
 * (`filesDir/CustomModels`) models. Mirrors `BundledModel.discover()` in `Models.swift`: the
 * list is enumeration only; nothing is loaded until a tab selects a model.
 */
class ModelCatalog(private val ctx: Context) {
    private val bundledRoot = File(ctx.filesDir, "models")
    private val customRoot = File(ctx.filesDir, "CustomModels")
    private val stamp = File(bundledRoot, ".assets-version")

    /** Bump when the bundled asset set changes so the copy is refreshed. */
    private val assetsVersion = "6"   // 6: model.onnx staged beside the ncnn pair (yolo11n, seg-N, v0.1-N, EsMoE-N)

    @Volatile private var cached: List<BundledModel>? = null

    /** Blocking: first call copies ~120 MB of assets. Call off the main thread. */
    @Synchronized
    fun discover(force: Boolean = false): List<BundledModel> {
        cached?.takeIf { !force }?.let { return it }
        ensureAssetsCopied()
        val out = ArrayList<BundledModel>()
        listDirs(bundledRoot).forEach { out += describe(it, custom = false) }
        listDirs(customRoot).forEach { out += describe(it, custom = true) }
        val list = out.distinctBy { it.id }.sortedBy { it.id.lowercase() }
        cached = list
        return list
    }

    fun invalidate() { cached = null }

    private fun listDirs(root: File): List<File> =
        root.listFiles()?.filter { it.isDirectory && BundledModel.isModelDir(it) }?.sortedBy { it.name } ?: emptyList()

    private fun describe(dir: File, custom: Boolean): BundledModel = BundledModel(
        id = dir.name, dir = dir, isSeg = readTask(dir) == "segment", isCustom = custom,
        hasNcnn = BundledModel.hasNcnnPair(dir),
        hasOnnx = File(dir, BundledModel.ONNX).isFile,
        onnxQuant = BundledModel.ONNX_QUANTS.firstOrNull { File(dir, "model-$it.onnx").isFile },
    )

    private fun readTask(dir: File): String? {
        val yaml = File(dir, "metadata.yaml")
        if (!yaml.isFile) return null
        yaml.useLines { lines ->
            for (l in lines) {
                val t = l.trim()
                if (t.startsWith("task:")) return t.removePrefix("task:").trim().trim('"', '\'')
            }
        }
        return null
    }

    private fun ensureAssetsCopied() {
        if (stamp.isFile && stamp.readText() == assetsVersion) return
        val am = ctx.assets
        val names = am.list("models") ?: emptyArray()
        bundledRoot.mkdirs()
        for (name in names) {
            val files = am.list("models/$name") ?: continue
            if (BundledModel.NCNN_PARAM !in files && BundledModel.ONNX !in files) continue
            val dst = File(bundledRoot, name)
            dst.mkdirs()
            for (f in files) {
                am.open("models/$name/$f").use { i -> File(dst, f).outputStream().use { o -> i.copyTo(o) } }
            }
        }
        stamp.writeText(assetsVersion)
    }

    // ---- custom models (Settings > Custom models) ------------------------------------------

    /** Import a picked directory tree (SAF). Returns the new model or throws with a reason. */
    fun importTree(uri: Uri): BundledModel {
        val doc = DocumentFile.fromTreeUri(ctx, uri) ?: throw IllegalArgumentException("Cannot open folder")
        val name = sanitize(doc.name ?: "custom")
        val dst = File(customRoot, if (name.endsWith("_ncnn")) name else "${name}_ncnn")
        dst.mkdirs()
        for (f in doc.listFiles()) {
            val n = f.name ?: continue
            if (n in MODEL_FILES) {
                ctx.contentResolver.openInputStream(f.uri)?.use { i -> File(dst, n).outputStream().use { i.copyTo(it) } }
            }
        }
        if (!BundledModel.isModelDir(dst)) { dst.deleteRecursively(); throw IllegalArgumentException("Folder must contain $REQUIRED_MSG") }
        invalidate()
        return describe(dst, custom = true)
    }

    /** Import a .zip whose entries (any top folder) contain the model files. */
    fun importZip(uri: Uri, displayName: String?): BundledModel {
        val name = sanitize((displayName ?: "custom").removeSuffix(".zip"))
        val dst = File(customRoot, if (name.endsWith("_ncnn")) name else "${name}_ncnn")
        dst.mkdirs()
        ctx.contentResolver.openInputStream(uri)?.use { raw ->
            ZipInputStream(raw).use { zip ->
                var e = zip.nextEntry
                while (e != null) {
                    val base = e.name.substringAfterLast('/')
                    if (!e.isDirectory && base in MODEL_FILES) {
                        File(dst, base).outputStream().use { zip.copyTo(it) }
                    }
                    zip.closeEntry(); e = zip.nextEntry
                }
            }
        } ?: throw IllegalArgumentException("Cannot open zip")
        if (!BundledModel.isModelDir(dst)) { dst.deleteRecursively(); throw IllegalArgumentException("Zip must contain $REQUIRED_MSG") }
        invalidate()
        return describe(dst, custom = true)
    }

    fun deleteCustom(model: BundledModel) {
        if (model.isCustom) { model.dir.deleteRecursively(); invalidate() }
    }

    private fun sanitize(s: String) = s.replace(Regex("[^A-Za-z0-9._-]"), "_").ifEmpty { "custom" }

    private companion object {
        /** Everything an import keeps: the ncnn pair, the ONNX graph and its QDQ siblings, the metadata. */
        val MODEL_FILES = setOf(BundledModel.NCNN_PARAM, BundledModel.NCNN_BIN, BundledModel.ONNX, "metadata.yaml") +
            BundledModel.ONNX_QUANTS.map { "model-$it.onnx" }
        const val REQUIRED_MSG = "model.ncnn.param + model.ncnn.bin, or model.onnx"
    }
}
