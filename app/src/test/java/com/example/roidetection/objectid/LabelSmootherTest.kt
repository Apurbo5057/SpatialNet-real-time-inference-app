package com.example.roidetection.objectid

import com.example.roidetection.BBox
import com.example.roidetection.LabelScore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class LabelSmootherTest {

    private val box = BBox(0.5f, 0.5f, 0.3f, 0.3f)
    private val tv = listOf(LabelScore("TV / monitor", 0.8f), LabelScore("Laptop", 0.1f))
    private val laptop = listOf(LabelScore("Laptop", 0.7f), LabelScore("TV / monitor", 0.1f))

    @Test
    fun firstUpdateShowsLabelImmediately() {
        val s = LabelSmoother().update(tv, box, 0L)
        assertEquals("TV / monitor", s.top?.label)
        assertEquals(0.8f, s.top!!.score, 1e-6f)
    }

    @Test
    fun singleOutlierFrameDoesNotSwitchLabel() {
        val smoother = LabelSmoother()
        smoother.update(tv, box, 0L)
        smoother.update(tv, box, 250L)
        val s = smoother.update(laptop, box, 500L)
        assertEquals("TV / monitor", s.top?.label)
    }

    @Test
    fun sustainedChangeSwitchesLabel() {
        val smoother = LabelSmoother()
        smoother.update(tv, box, 0L)
        smoother.update(laptop, box, 250L)
        val s = smoother.update(laptop, box, 500L)
        assertEquals("Laptop", s.top?.label)
    }

    @Test
    fun lowConfidenceIsUnknownButKeepsCandidates() {
        val s = LabelSmoother(showThreshold = 0.35f)
            .update(listOf(LabelScore("Lamp", 0.2f), LabelScore("Fan", 0.15f)), box, 0L)
        assertNull(s.top)
        assertEquals(listOf("Lamp", "Fan"), s.candidates.map { it.label })
    }

    @Test
    fun resetsWhenRoiJumps() {
        val smoother = LabelSmoother()
        smoother.update(tv, box, 0L)
        smoother.update(tv, box, 250L)
        val s = smoother.update(laptop, BBox(0.1f, 0.1f, 0.1f, 0.1f), 500L)
        assertEquals("Laptop", s.top?.label)
        assertEquals(0.7f, s.top!!.score, 1e-6f)
    }

    @Test
    fun resetsAfterRoiWasLost() {
        val smoother = LabelSmoother(resetGapMs = 600L)
        smoother.update(tv, box, 0L)
        val s = smoother.update(laptop, box, 1000L)
        assertEquals("Laptop", s.top?.label)
    }

    @Test
    fun candidatesAreLimitedAndSorted() {
        val many = listOf(
            LabelScore("A", 0.1f), LabelScore("B", 0.4f), LabelScore("C", 0.3f), LabelScore("D", 0.2f)
        )
        val s = LabelSmoother(maxCandidates = 3).update(many, box, 0L)
        assertEquals(listOf("B", "C", "D"), s.candidates.map { it.label })
    }
}
