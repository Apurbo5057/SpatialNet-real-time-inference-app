package com.example.roidetection.memory

import com.example.roidetection.LabelScore
import kotlin.math.sqrt

/**
 * Compares an ROI embedding with the user's saved objects.
 *
 * Embeddings are the classifier's pooled features, L2-normalized, so the dot
 * product is the cosine similarity. On jittered crops of real captures the same
 * object scored about 0.90 and different objects about 0.46, so the default
 * [DEFAULT_THRESHOLD] sits between the two.
 */
object ObjectMatcher {
    const val DEFAULT_THRESHOLD = 0.78f

    fun normalize(v: FloatArray): FloatArray {
        var sum = 0.0
        for (x in v) sum += x * x
        val norm = sqrt(sum).toFloat()
        if (norm == 0f) return v.copyOf()
        return FloatArray(v.size) { v[it] / norm }
    }

    fun cosine(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "embedding sizes differ: ${a.size} vs ${b.size}" }
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot
    }

    /**
     * Similarity of [embedding] to each object (its closest saved sample), keyed
     * by object id and sorted best first. Objects with no samples are skipped.
     */
    fun scores(embedding: FloatArray, objects: List<SavedObject>): List<LabelScore> =
        objects.mapNotNull { obj ->
            obj.embeddings.maxOfOrNull { cosine(embedding, it) }?.let { LabelScore(obj.id, it) }
        }.sortedByDescending { it.score }
}
