package com.example.roidetection

import android.content.Context
import android.util.Log
import java.security.MessageDigest

/**
 * Identity of the model weights actually bundled in this APK.
 *
 * The digest is computed from the asset bytes at runtime instead of being
 * hardcoded, so the badge shown in the UI reports the weights in the APK.
 */
data class ModelInfo(
    val name: String,
    val digest: String,
    val sizeBytes: Int
) {
    /** e.g. "2.87 MB" */
    val size: String
        get() = "%.2f MB".format(sizeBytes / 1024.0 / 1024.0)

    /** e.g. "76710641 · 2.87 MB" */
    val detail: String
        get() = "$digest · $size"

    override fun toString(): String = "$name · $detail"

    companion object {
        private const val TAG = "ModelInfo"
        val SPATIALNET_ASSET = BuildConfig.MODEL_ASSET
        val CLASSIFIER_ASSET = BuildConfig.CLASSIFIER_ASSET
        val CLASSIFIER_LABELS_ASSET = BuildConfig.CLASSIFIER_LABELS_ASSET
        val EARBUDS_MODEL_ASSET = BuildConfig.EARBUDS_MODEL_ASSET

        private val cache = HashMap<String, ModelInfo>()

        /** Identity of [assetName], derived from bytes already in hand. */
        fun of(assetName: String, bytes: ByteArray): ModelInfo {
            val sha = MessageDigest.getInstance("SHA-256").digest(bytes)
            val digest = sha.take(4).joinToString("") { "%02x".format(it) }
            val info = ModelInfo(assetName.removeSuffix(".onnx"), digest, bytes.size)
            synchronized(cache) { cache[assetName] = info }
            return info
        }

        /**
         * Identity of [assetName], read straight from the APK. Cached, so the
         * home screen and the inference screen hash the file only once.
         */
        fun read(context: Context, assetName: String = SPATIALNET_ASSET): ModelInfo? {
            synchronized(cache) { cache[assetName]?.let { return it } }
            return try {
                of(assetName, context.assets.open(assetName).use { it.readBytes() })
            } catch (e: Exception) {
                Log.e(TAG, "Could not read $assetName: ${e.javaClass.simpleName}: ${e.message}")
                null
            }
        }
    }
}
