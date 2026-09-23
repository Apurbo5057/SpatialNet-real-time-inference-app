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
        noCompress += "onnx"
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
tasks.matching { it.name.startsWith("mergeFp32") && it.name.endsWith("Assets") }
    .configureEach { dependsOn(prepareFp32ModelAsset) }
tasks.matching { it.name.startsWith("mergeFromFp16") && it.name.endsWith("Assets") }
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

    // CameraX
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)

    // Navigation Compose
    implementation(libs.androidx.navigation.compose)

    debugImplementation(libs.androidx.ui.tooling)
}
