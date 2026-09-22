import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.serialization")
}

// Release APKs are signed with a local dev keystore so a build from this machine installs over
// the previous one on a test phone (`adb install -r`). Path/passwords come from
// android/local.properties (gitignored) or env; absent -> the AGP debug key.
val localProps = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}
fun prop(name: String): String? = localProps.getProperty(name) ?: System.getenv(name)
val devKeystore = prop("YM_KEYSTORE")

android {
    namespace = "dev.yolomaster.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "dev.yolomaster.app"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.1.0"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    signingConfigs {
        if (devKeystore != null) {
            create("dev") {
                storeFile = file(devKeystore)
                storePassword = prop("YM_KEYSTORE_PASSWORD") ?: "yolomaster"
                keyAlias = prop("YM_KEY_ALIAS") ?: "yolomaster"
                keyPassword = prop("YM_KEY_PASSWORD") ?: "yolomaster"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = if (devKeystore != null) signingConfigs.getByName("dev")
                            else signingConfigs.getByName("debug")
        }
    }

    buildFeatures { compose = true }
    // Kotlin 1.9.24 <-> Compose compiler 1.5.14 (the Kotlin-2.0 compose plugin is not an option).
    composeOptions { kotlinCompilerExtensionVersion = "1.5.14" }

    androidResources {
        // Model weights are read straight from the APK on first launch; don't deflate them.
        noCompress += listOf("bin", "param", "onnx")
        // Quantizer leftovers must never ship even if stage_models.sh is bypassed.
        ignoreAssetsPattern = "!__pycache__:!quant:!.DS_Store:!*.npz:!*.py:!quant_manifest.json"
    }

    packaging {
        // Legacy packaging = the .so files are extracted to the native lib dir at install time.
        // The Hexagon skels (libQnnHtpV81Skel.so) are loaded by the DSP side through
        // libcdsprpc.so from that dir (ORT points ADSP_LIBRARY_PATH at it): they must be real
        // files, not pages inside the APK.
        jniLibs { useLegacyPackaging = true }
        resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" }
    }

    sourceSets["main"].java.srcDir("src/main/kotlin")
    sourceSets["test"].java.srcDir("src/test/kotlin")

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
        // Material3 segmented buttons / Camera2 interop / foundation pager are still opt-in APIs.
        freeCompilerArgs += listOf(
            "-opt-in=androidx.compose.material3.ExperimentalMaterial3Api",
            "-opt-in=androidx.compose.foundation.ExperimentalFoundationApi",
            "-opt-in=androidx.camera.camera2.interop.ExperimentalCamera2Interop",
        )
    }
}

dependencies {
    implementation(project(":runtime"))

    val composeBom = platform("androidx.compose:compose-bom:2024.06.00")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.3")
    implementation("androidx.core:core-ktx:1.13.1")

    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    implementation("androidx.exifinterface:exifinterface:1.3.7")
    implementation("androidx.documentfile:documentfile:1.0.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")

    // Backdrop blur for the iOS ultraThinMaterial look. 0.7.3 is the last line built against
    // Compose 1.6 / Kotlin 1.9.24; 0.9+ requires Kotlin 2.0.
    implementation("dev.chrisbanes.haze:haze:0.7.3")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
}
