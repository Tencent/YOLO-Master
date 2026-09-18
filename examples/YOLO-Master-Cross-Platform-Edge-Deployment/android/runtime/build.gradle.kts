import java.util.Properties

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

// Per-machine SDK roots for the native build (ncnn Android + opencv-mobile).
// Copy android/sdk-paths.example.properties -> android/sdk-paths.properties (gitignored).
val sdkPaths = Properties().apply {
    val f = rootProject.file("sdk-paths.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}
val ncnnRoot: String = sdkPaths.getProperty("NCNN_ANDROID_ROOT", System.getenv("NCNN_ANDROID_ROOT") ?: "")
val opencvRoot: String = sdkPaths.getProperty("OPENCV_ANDROID_ROOT", System.getenv("OPENCV_ANDROID_ROOT") ?: "")

// ---- ONNX Runtime + Qualcomm QNN (Hexagon NPU) native payload ----
// The ORT backend is a second runtime next to ncnn (Workstream B of the ORT/QNN plan). Nothing
// from these AARs goes on the Java classpath: the ORT AAR's manifest would merge INTERNET /
// ACCESS_NETWORK_STATE and a telemetry ContentProvider into the app, and we use the C API only
// (libonnxruntime4j_jni.so is dropped). A private, non-transitive configuration downloads the
// three AARs and `extractOrt` unpacks exactly the files the native build and the APK need:
//   * onnxruntime-android-qnn  -> headers/ + jni/arm64-v8a/libonnxruntime.so (QNN EP built in)
//   * onnxruntime-android      -> jni/x86_64/libonnxruntime.so only (ort-CPU on the emulator)
//   * qnn-runtime (QTI)        -> libQnnHtp.so, libQnnSystem.so, the Hexagon skel/stub pair(s)
//                                 for the target SoC (-PymQnnArchs, default V81 = Snapdragon
//                                 8 Elite Gen 5 / Samsung S26) and libQnnHtpPrepare.so (84 MB,
//                                 on-device graph compile; -PymShipPrepare=false once EPContext
//                                 caches are shipped, milestone M4).
// The Qualcomm libraries are under QTI's own proprietary license (LICENSE.pdf / NOTICE.txt in
// the AAR): fine for these test builds, to be reviewed before any public release.
val ymQnnArchs: List<String> = (findProperty("ymQnnArchs") as String? ?: "V81")
    .split(",").map { it.trim() }.filter { it.isNotEmpty() }
val ymShipPrepare: Boolean = (findProperty("ymShipPrepare") as String? ?: "true").toBoolean()
val ortDir = layout.buildDirectory.dir("ort")

val ortNative: Configuration by configurations.creating {
    isTransitive = false     // the QNN AAR's POM pulls qnn-runtime itself; we pin it explicitly
    isCanBeConsumed = false
    isCanBeResolved = true
}

dependencies {
    ortNative("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.29.0@aar")
    ortNative("com.microsoft.onnxruntime:onnxruntime-android:1.29.0@aar")
    ortNative("com.qualcomm.qti:qnn-runtime:2.42.0@aar")
}

/**
 * build/ort/headers and build/ort/jni/<abi>/libNAME.so: the ORT include dir + per-ABI shared libraries.
 * Sync (not Copy) so a changed arch list or a dropped Prepare never leaves stale libs behind.
 */
val extractOrt by tasks.registering(Sync::class) {
    description = "Unpack the ONNX Runtime headers and the ORT/QNN shared libraries into build/ort"
    val qnnAar = ortNative.filter { it.name.startsWith("onnxruntime-android-qnn-") }
    val plainAar = ortNative.filter { it.name.startsWith("onnxruntime-android-1") }
    val qtiAar = ortNative.filter { it.name.startsWith("qnn-runtime-") }
    // Hexagon skel/stub pairs are per DSP architecture; ship only what the target phone needs
    // (each foreign pair is ~18 MB of dead weight). The *Skel.so is loaded by the DSP through
    // libcdsprpc.so from the app's native lib dir, which ORT points ADSP_LIBRARY_PATH at.
    val qtiLibs = mutableListOf("jni/arm64-v8a/libQnnHtp.so", "jni/arm64-v8a/libQnnSystem.so")
    for (arch in ymQnnArchs) qtiLibs += "jni/arm64-v8a/libQnnHtp${arch}*.so"
    if (ymShipPrepare) qtiLibs += "jni/arm64-v8a/libQnnHtpPrepare.so"
    inputs.property("ymQnnArchs", ymQnnArchs)
    inputs.property("ymShipPrepare", ymShipPrepare)
    from({ qnnAar.map { zipTree(it) } }) { include("headers/**", "jni/arm64-v8a/libonnxruntime.so") }
    from({ plainAar.map { zipTree(it) } }) { include("jni/x86_64/libonnxruntime.so") }
    from({ qtiAar.map { zipTree(it) } }) { include(qtiLibs) }
    into(ortDir)
}

android {
    namespace = "dev.yolomaster.ncnn"
    compileSdk = 34
    ndkVersion = "29.0.14206865" // r29: opencv-mobile 4.13 links __kmpc_dispatch_deinit, absent from the r26-r28 OpenMP runtime

    defaultConfig {
        minSdk = 24 // libc++fs + Vulkan 1.1
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DNCNN_ANDROID_ROOT=$ncnnRoot",
                    "-DOPENCV_ANDROID_ROOT=$opencvRoot",
                    "-DORT_ANDROID_ROOT=${ortDir.get().asFile.absolutePath}",
                    "-DANDROID_STL=c++_shared",
                )
                cppFlags += "-std=c++17"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    sourceSets["main"].java.srcDir("src/main/kotlin")
    sourceSets["androidTest"].java.srcDir("src/androidTest/kotlin")
    // ORT + QNN shared libraries packaged next to libyolomaster_ncnn.so (arm64-v8a: ORT-QNN +
    // Qualcomm libs; x86_64: the plain ORT for ort-CPU). libonnxruntime.so is also on the CMake
    // link line, which AGP packages too: pick one copy instead of failing on the duplicate.
    sourceSets["main"].jniLibs.srcDir(ortDir.map { it.dir("jni") })
    packaging {
        jniLibs {
            pickFirsts += "**/libonnxruntime.so"
            // The Hexagon FastRPC loader opens the skel/stub libraries by file path from the app's
            // native lib dir; zip-mapped (non-extracted) libs make HTP init fail silently and the
            // QNN EP then places zero nodes. This applies to the androidTest APK (the M0 gate) too.
            useLegacyPackaging = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

// The CMake configure/build reads headers + libonnxruntime.so, the jniLibs merge reads jni/.
// configureEach (not matching{}) so AGP's tasks are not realized before android {} is evaluated.
tasks.configureEach {
    if (name.startsWith("configureCMake") || name.startsWith("buildCMake") || name.endsWith("JniLibFolders"))
        dependsOn(extractOrt)
}

dependencies {
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
}
