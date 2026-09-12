package dev.yolomaster.app.model

/**
 * Display-name rules shared with the iOS app (`Models.swift:12-24`): every model is shown as
 * `YOLO-Master-<stem>`, and the collapsed picker label is the first 6 characters of the stem
 * (the shared prefix carries no information in a narrow bar).
 */
object Naming {
    private val prefixes = listOf("yolo-master-", "yolo-master_", "yolo_master_", "yolomaster-")

    /** The model id without an `_ncnn` suffix and without a doubled "YOLO-Master" prefix. */
    fun stem(id: String): String {
        var s = id.removeSuffix("_ncnn")
        val lower = s.lowercase()
        for (p in prefixes) if (lower.startsWith(p)) { s = s.substring(p.length); break }
        return s
    }

    fun fullName(id: String): String = "YOLO-Master-" + stem(id)

    fun shortID(id: String): String {
        val s = stem(id)
        return if (s.length > 6) s.take(6) + "..." else s
    }
}

/** COCO-80 fallback names (`ios/App/Overlay.swift:22`) for models whose metadata carries digits. */
val cocoNames: List<String> = listOf(
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)

/** Resolve a class label: model names first, COCO-80 when the model only carries digits. */
fun classLabel(names: List<String>, cls: Int): String {
    val n = names.getOrNull(cls)
    if (n != null && n.isNotEmpty() && n.toIntOrNull() == null) return n
    return cocoNames.getOrNull(cls) ?: cls.toString()
}
