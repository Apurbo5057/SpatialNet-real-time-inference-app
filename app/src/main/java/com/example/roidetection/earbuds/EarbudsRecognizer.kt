package com.example.roidetection.earbuds

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/** Softmax(W · e + b) over the catalog models; W is row-major [classes][dim]. */
class LinearClassifier(val classes: Int, val dim: Int, private val weights: FloatArray, private val bias: FloatArray) {
    init {
        require(weights.size == classes * dim && bias.size == classes) { "classifier shape mismatch" }
    }

    fun probabilities(e: FloatArray): FloatArray {
        require(e.size == dim) { "embedding has ${e.size} values, classifier expects $dim" }
        val logits = FloatArray(classes) { c ->
            var s = bias[c]
            val off = c * dim
            for (i in 0 until dim) s += weights[off + i] * e[i]
            s
        }
        val max = logits.maxOrNull() ?: 0f
        var sum = 0.0
        val out = FloatArray(classes)
        for (c in 0 until classes) {
            val v = exp((logits[c] - max).toDouble())
            out[c] = v.toFloat()
            sum += v
        }
        for (c in 0 until classes) out[c] = (out[c] / sum).toFloat()
        return out
    }
}

/** Reference photo vectors (L2-normalized, row-major [count][dim]) used to tell earbuds from other things. */
class ReferenceSet(val count: Int, val dim: Int, private val vectors: FloatArray) {
    init {
        require(vectors.size == count * dim) { "reference shape mismatch" }
    }

    fun bestSimilarity(e: FloatArray): Float {
        require(e.size == dim)
        var best = -1f
        for (r in 0 until count) {
            var dot = 0f
            val off = r * dim
            for (i in 0 until dim) dot += vectors[off + i] * e[i]
            if (dot > best) best = dot
        }
        return best
    }
}

/** A catalog model index and how likely the crop shows it. */
data class Candidate(val index: Int, val probability: Float)

/**
 * What the crop looks like. [isEarbuds] is false when it resembles no reference photo
 * (a wall, a hand, a cup). Earbuds that are not in the catalog still look like earbuds,
 * so they pass this check; Bluetooth is what finally rules them out.
 */
data class Recognition(val isEarbuds: Boolean, val similarity: Float, val candidates: List<Candidate>)

class EarbudsRecognizer(
    private val classifier: LinearClassifier,
    private val references: ReferenceSet,
    private val minSimilarity: Float
) {
    val dim: Int get() = classifier.dim

    /** [embedding] must be L2-normalized. Returns the best [k] candidates, most likely first. */
    fun recognize(embedding: FloatArray, k: Int = 3): Recognition {
        val similarity = references.bestSimilarity(embedding)
        val probs = classifier.probabilities(embedding)
        val top = probs.indices.sortedByDescending { probs[it] }.take(k).map { Candidate(it, probs[it]) }
        return Recognition(similarity >= minSimilarity, similarity, top)
    }
}

/** Readers for the little-endian files written by tools/earbuds/build_catalog.py. */
object EarbudsBinary {

    /** int32 classes, int32 dim, float32 W[classes][dim], float32 b[classes]. */
    fun parseClassifier(bytes: ByteArray): LinearClassifier {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val classes = buf.int
        val dim = buf.int
        require(classes in 1..10_000 && dim in 1..10_000) { "bad classifier header $classes x $dim" }
        val w = FloatArray(classes * dim).also { buf.asFloatBuffer().get(it) }
        buf.position(8 + 4 * classes * dim)
        val b = FloatArray(classes).also { buf.asFloatBuffer().get(it) }
        return LinearClassifier(classes, dim, w, b)
    }

    /** int32 count, int32 dim, then per vector int32 class + float32[dim]. Class labels are not needed. */
    fun parseReferences(bytes: ByteArray): ReferenceSet {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val count = buf.int
        val dim = buf.int
        require(count in 1..1_000_000 && dim in 1..10_000) { "bad reference header $count x $dim" }
        val vectors = FloatArray(count * dim)
        for (r in 0 until count) {
            buf.int // class label
            for (i in 0 until dim) vectors[r * dim + i] = buf.float
        }
        return ReferenceSet(count, dim, vectors)
    }
}
