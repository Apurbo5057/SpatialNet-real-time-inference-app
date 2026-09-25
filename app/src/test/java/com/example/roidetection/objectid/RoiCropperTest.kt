package com.example.roidetection.objectid

import com.example.roidetection.BBox
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RoiCropperTest {

    private fun assertInside(rect: PixelRect, w: Int, h: Int) {
        assertTrue("left ${rect.left}", rect.left >= 0)
        assertTrue("top ${rect.top}", rect.top >= 0)
        assertTrue("right ${rect.left + rect.width}", rect.left + rect.width <= w)
        assertTrue("bottom ${rect.top + rect.height}", rect.top + rect.height <= h)
    }

    @Test
    fun centeredBoxBecomesPaddedSquareAroundIt() {
        // 0.2 x 0.1 of a 480x640 portrait frame = 96 x 64 px, longer side 96
        val rect = RoiCropper.cropRect(BBox(0.5f, 0.5f, 0.2f, 0.1f), 480, 640, padding = 0.25f)!!
        assertEquals(120, rect.width)
        assertEquals(120, rect.height)
        assertEquals(240 - 60, rect.left)
        assertEquals(320 - 60, rect.top)
    }

    @Test
    fun boxAtFrameEdgeIsShiftedInside() {
        val rect = RoiCropper.cropRect(BBox(0.02f, 0.98f, 0.2f, 0.2f), 480, 640)!!
        assertInside(rect, 480, 640)
        assertEquals(0, rect.left)
        assertEquals(640 - rect.height, rect.top)
    }

    @Test
    fun boxLargerThanFrameIsCappedToShorterSide() {
        val rect = RoiCropper.cropRect(BBox(0.5f, 0.5f, 1.2f, 1.0f), 480, 640)!!
        assertEquals(480, rect.width)
        assertInside(rect, 480, 640)
    }

    @Test
    fun tinyBoxIsRejected() {
        // 0.05 x 0.04 of 480x640 = 24 x 25.6 px, below 32 px
        assertNull(RoiCropper.cropRect(BBox(0.5f, 0.5f, 0.05f, 0.04f), 480, 640))
        assertNotNull(RoiCropper.cropRect(BBox(0.5f, 0.5f, 0.1f, 0.04f), 480, 640))
    }

    @Test
    fun emptyFrameIsRejected() {
        assertNull(RoiCropper.cropRect(BBox(0.5f, 0.5f, 0.5f, 0.5f), 0, 640))
    }
}
