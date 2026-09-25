package com.example.roidetection.objectid

import com.example.roidetection.LabelScore
import kotlin.math.exp

/** Android-free pre/post-processing for the object classifier. */
object ClassifierMath {
    val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /** No normalization beyond scaling to 0..1 (MobileCLIP). */
    val NO_MEAN = floatArrayOf(0f, 0f, 0f)
    val NO_STD = floatArrayOf(1f, 1f, 1f)

    /**
     * ARGB pixels (row-major, [size] x [size]) to a normalized NCHW float
     * tensor in [out], in one pass. ImageNet mean/std unless told otherwise.
     */
    fun toNchw(
        pixels: IntArray,
        size: Int,
        out: FloatArray,
        mean: FloatArray = IMAGENET_MEAN,
        std: FloatArray = IMAGENET_STD
    ) {
        val plane = size * size
        require(pixels.size >= plane && out.size >= 3 * plane)
        for (i in 0 until plane) {
            val p = pixels[i]
            out[i] = (((p shr 16) and 0xFF) / 255f - mean[0]) / std[0]
            out[plane + i] = (((p shr 8) and 0xFF) / 255f - mean[1]) / std[1]
            out[2 * plane + i] = ((p and 0xFF) / 255f - mean[2]) / std[2]
        }
    }

    fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: return logits
        val out = FloatArray(logits.size)
        var sum = 0.0
        for (i in logits.indices) {
            val e = exp((logits[i] - maxLogit).toDouble())
            out[i] = e.toFloat()
            sum += e
        }
        for (i in out.indices) out[i] = (out[i] / sum).toFloat()
        return out
    }

    /**
     * Sums probabilities of classes that share a friendly label (e.g. the six
     * ImageNet screen classes all read "TV / monitor") and returns the best [k].
     */
    fun topLabels(probs: FloatArray, labels: List<String>, k: Int): List<LabelScore> {
        require(probs.size == labels.size) { "${probs.size} scores for ${labels.size} labels" }
        val merged = HashMap<String, Float>()
        for (i in probs.indices) merged[labels[i]] = (merged[labels[i]] ?: 0f) + probs[i]
        return merged.entries
            .sortedByDescending { it.value }
            .take(k)
            .map { LabelScore(it.key, it.value) }
    }
}
