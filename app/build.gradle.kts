plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.roidetection"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.roidetection"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        // Object classifier run on the ROI crop; identical in both flavors.
        buildConfigField("String", "CLASSIFIER_ASSET", "\"mobilenetv4_conv_medium_fp16w.onnx\"")
        buildConfigField("String", "CLASSIFIER_LABELS_ASSET", "\"mobilenetv4_labels.txt\"")
        // Earbuds recognition (Pair Earbuds mode); identical in both flavors.
        buildConfigField("String", "EARBUDS_MODEL_ASSET", "\"mobileclip2_s0_image_fp16w.onnx\"")

        // Phones (and Apple Silicon emulators) are ARM; x86 builds of ONNX Runtime
        // and ML Kit would add ~60 MB for PC emulators only.
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
    }
    flavorDimensions += "model"
    productFlavors {
        create("fp32") {
            dimension = "model"
            versionNameSuffix = "-fp32"
            buildConfigField("String", "MODEL_ASSET", "\"spatialnet_v4.onnx\"")
            manifestPlaceholders["appLabel"] = "ROI Detection FP32"
        }
        create("fromFp16") {
            dimension = "model"
            applicationIdSuffix = ".fromfp16"
            versionNameSuffix = "-from-fp16"
            buildConfigField("String", "MODEL_ASSET", "\"spatialnet_fastest_from_fp16.onnx\"")
            manifestPlaceholders["appLabel"] = "ROI Detection FP16 Source"
        }
    }
    sourceSets.getByName("fp32").assets.srcDir(layout.buildDirectory.dir("generated/modelAssets/fp32"))
    sourceSets.getByName("fromFp16").assets.srcDir(layout.buildDirectory.dir("generated/modelAssets/fromFp16"))
    sourceSets.getByName("main").assets.srcDir(layout.buildDirectory.dir("generated/modelAssets/classifier"))
    sourceSets.getByName("main").assets.srcDir(layout.buildDirectory.dir("generated/modelAssets/earbuds"))

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    androidResources {
        noCompress += listOf("onnx", "bin")
    }
}

val prepareFp32ModelAsset by tasks.registering(Sync::class) {
    from("../ROIModel/spatialnet_v4.onnx")
    into(layout.buildDirectory.dir("generated/modelAssets/fp32"))
}
val prepareFromFp16ModelAsset by tasks.registering(Sync::class) {
    from("../ROIModel/spatialnet_fastest_from_fp16.onnx")
    into(layout.buildDirectory.dir("generated/modelAssets/fromFp16"))
}
val prepareClassifierAssets by tasks.registering(Sync::class) {
    from("../ObjectModel/mobilenetv4_conv_medium_fp16w.onnx", "../ObjectModel/mobilenetv4_labels.txt")
    into(layout.buildDirectory.dir("generated/modelAssets/classifier"))
}
// Built by tools/earbuds/build_catalog.py into a git-ignored folder (its thumbnails come
// from web photos); without it the app still builds and Pair Earbuds says it is unavailable.
val prepareEarbudsAssets by tasks.registering(Sync::class) {
    from("../ObjectModel/mobileclip2_s0_image_fp16w.onnx")
    from("../tools/earbuds/out") {
        include("earbuds_catalog.json", "earbuds_classifier.bin", "earbuds_embeddings.bin", "thumbs/*.jpg")
        into("earbuds")
    }
    into(layout.buildDirectory.dir("generated/modelAssets/earbuds"))
}
// Asset merging and lint both read the generated asset folders.
fun Task.readsAssets() = (name.startsWith("merge") && name.endsWith("Assets")) || name.contains("lint", ignoreCase = true)
tasks.matching { it.readsAssets() }
    .configureEach { dependsOn(prepareClassifierAssets, prepareEarbudsAssets) }
tasks.matching { it.readsAssets() && it.name.contains("Fp32") }
    .configureEach { dependsOn(prepareFp32ModelAsset) }
tasks.matching { it.readsAssets() && it.name.contains("FromFp16") }
    .configureEach { dependsOn(prepareFromFp16ModelAsset) }

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)

    // ONNX Runtime
    implementation(libs.onnxruntime.android)

    // ML Kit, bundled models (work offline from install; no Play services download)
    implementation(libs.mlkit.text.recognition)
    implementation(libs.mlkit.barcode.scanning)

    // CameraX
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)

    // Navigation Compose
    implementation(libs.androidx.navigation.compose)

    debugImplementation(libs.androidx.ui.tooling)

    testImplementation(libs.junit)
}
