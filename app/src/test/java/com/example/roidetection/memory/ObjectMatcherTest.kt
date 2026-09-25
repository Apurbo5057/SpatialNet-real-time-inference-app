package com.example.roidetection.memory

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ObjectMatcherTest {

    private fun v(vararg x: Float) = ObjectMatcher.normalize(floatArrayOf(*x))

    @Test
    fun normalizeGivesUnitLength() {
        val n = v(3f, 4f)
        assertEquals(0.6f, n[0], 1e-6f)
        assertEquals(0.8f, n[1], 1e-6f)
        assertEquals(1f, ObjectMatcher.cosine(n, n), 1e-6f)
    }

    @Test
    fun zeroVectorStaysZero() {
        assertTrue(ObjectMatcher.normalize(floatArrayOf(0f, 0f)).all { it == 0f })
    }

    @Test
    fun bestSampleCountsForEachObjectAndResultsAreSorted() {
        val keys = SavedObject("k", "Keys", embeddings = listOf(v(1f, 0f, 0f), v(0f, 1f, 0f)))
        val mug = SavedObject("m", "Mug", embeddings = listOf(v(0f, 0f, 1f)))
        val scores = ObjectMatcher.scores(v(0.1f, 1f, 0f), listOf(mug, keys))
        assertEquals(listOf("k", "m"), scores.map { it.label })
        assertTrue(scores[0].score > 0.99f)
        assertEquals(0f, scores[1].score, 1e-6f)
    }

    @Test
    fun objectsWithoutSamplesAreSkipped() {
        val empty = SavedObject("e", "Empty")
        assertTrue(ObjectMatcher.scores(v(1f, 0f), listOf(empty)).isEmpty())
    }
}
