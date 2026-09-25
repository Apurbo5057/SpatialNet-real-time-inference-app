package com.example.roidetection.objectid

import com.example.roidetection.BBox
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ClassificationSchedulerTest {

    private val box = BBox(0.5f, 0.5f, 0.3f, 0.3f)

    @Test
    fun firstRoiIsClassifiedAtOnce() {
        assertTrue(ClassificationScheduler().shouldClassify(box, 0L))
    }

    @Test
    fun steadyRoiIsRateLimited() {
        val s = ClassificationScheduler(intervalMs = 250L)
        s.onClassified(box, 1000L)
        assertFalse(s.shouldClassify(box, 1100L))
        assertTrue(s.shouldClassify(box, 1250L))
    }

    @Test
    fun movedRoiIsClassifiedAtOnce() {
        val s = ClassificationScheduler(intervalMs = 250L, moveIou = 0.5f)
        s.onClassified(box, 1000L)
        assertTrue(s.shouldClassify(BBox(0.7f, 0.5f, 0.3f, 0.3f), 1050L))
    }

    @Test
    fun resetForgetsLastRoi() {
        val s = ClassificationScheduler()
        s.onClassified(box, 1000L)
        s.reset()
        assertTrue(s.shouldClassify(box, 1010L))
    }
}
