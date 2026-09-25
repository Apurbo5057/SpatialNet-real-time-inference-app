package com.example.roidetection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DetectionFilterTest {

    // A real-looking pointing frame: hand at the bottom, fingertip on its top edge,
    // pointing straight up (sin = -1 in image coordinates) at a target above it.
    private val hand = BBox(0.5f, 0.75f, 0.4f, 0.4f)
    private val tip = Point(0.5f, 0.56f)
    private val up = Direction(sin = -1f, cos = 0f)
    private val target = BBox(0.5f, 0.25f, 0.3f, 0.2f)

    private fun frame(
        hand: BBox? = this.hand,
        tip: Point? = this.tip,
        dir: Direction? = up,
        roi: BBox? = target,
        pointing: Boolean = true
    ) = ROIResult(
        handBBox = hand, fingertip = if (pointing) tip else null, pointingDir = if (pointing) dir else null,
        roiBBox = if (pointing) roi else null,
        isHandDetected = hand != null, isPointing = pointing && hand != null,
        isRoiDetected = pointing && roi != null && hand != null
    )

    private val empty = ROIResult()

    @Test
    fun aSingleFrameHandIsNeverShown() {
        val f = DetectionFilter()
        val out = f.update(frame(), 0.75f)
        assertFalse(out.isPointing)
        assertNull(out.result.handBBox)
        assertNull(out.result.roiBBox)
        assertNull(f.update(empty, 0.75f).result.handBBox)
    }

    @Test
    fun aSteadyPointingHandIsShownAfterThreeFrames() {
        val f = DetectionFilter()
        f.update(frame(), 0.75f)
        f.update(frame(), 0.75f)
        val out = f.update(frame(), 0.75f)
        assertTrue(out.isPointing)
        assertNotNull(out.result.handBBox)
        assertNotNull(out.result.fingertip)
        assertEquals(target, out.result.roiBBox)
    }

    @Test
    fun oneMissedFrameDoesNotDropAConfirmedHand() {
        val f = DetectionFilter()
        repeat(4) { f.update(frame(), 0.75f) }
        f.update(empty, 0.75f)
        val back = f.update(frame(), 0.75f)
        assertTrue(back.isPointing)
        assertNotNull(back.result.roiBBox)
    }

    @Test
    fun handGoesAwayAfterItLeaves() {
        val f = DetectionFilter()
        repeat(5) { f.update(frame(), 0.75f) }
        repeat(4) { f.update(empty, 0.75f) }
        val out = f.update(empty, 0.75f)
        assertFalse(out.isPointing)
        assertNull(out.result.handBBox)
    }

    @Test
    fun fingertipOffTheHandRejectsTheWholeHand() {
        val f = DetectionFilter()
        val off = frame(tip = Point(0.1f, 0.1f))
        repeat(5) { f.update(off, 0.75f) }
        val out = f.update(off, 0.75f)
        assertFalse(out.isPointing)
        assertNull(out.result.handBBox)
        assertEquals("fingertip off the hand", out.rejectedReason)
    }

    @Test
    fun targetBehindThePointingDirectionIsRejected() {
        val f = DetectionFilter()
        val behind = frame(dir = Direction(sin = 1f, cos = 0f))   // pointing down, target above
        repeat(5) { f.update(behind, 0.75f) }
        val out = f.update(behind, 0.75f)
        assertFalse(out.isPointing)
        assertNull(out.result.roiBBox)
        assertNotNull(out.result.handBBox)  // the hand itself looks fine
    }

    @Test
    fun aJumpingHandBoxStartsOver() {
        val f = DetectionFilter()
        repeat(4) { f.update(frame(), 0.75f) }
        val jumped = frame(hand = BBox(0.15f, 0.2f, 0.2f, 0.2f), tip = Point(0.15f, 0.1f), roi = BBox(0.15f, 0.03f, 0.1f, 0.05f))
        val out = f.update(jumped, 0.75f)
        assertFalse(out.isPointing)
        assertNull(out.result.handBBox)
    }

    @Test
    fun targetThatIsTheHandIsDropped() {
        val f = DetectionFilter()
        val onHand = frame(roi = BBox(0.5f, 0.75f, 0.2f, 0.2f), dir = null)
        repeat(4) { f.update(onHand, 0.75f) }
        val out = f.update(onHand, 0.75f)
        assertTrue(out.isPointing)
        assertNull(out.result.roiBBox)
    }

    @Test
    fun geometryHelpers() {
        assertEquals(0f, DetectionFilter.tipOutside(hand, tip), 1e-6f)
        assertEquals(0.5f, DetectionFilter.tipOutside(BBox(0.5f, 0.5f, 0.2f, 0.2f), Point(0.5f, 0.3f)), 1e-5f)
        assertEquals(1f, DetectionFilter.coveredFraction(BBox(0.5f, 0.5f, 0.1f, 0.1f), hand.copy(cy = 0.5f)), 1e-6f)
        assertEquals(0f, DetectionFilter.targetAngleDeg(tip, up, target, 0.75f)!!, 0.5f)
        assertEquals(180f, DetectionFilter.targetAngleDeg(tip, Direction(1f, 0f), target, 0.75f)!!, 0.5f)
        assertNull(DetectionFilter.targetAngleDeg(tip, up, BBox(0.5f, 0.56f, 0.1f, 0.1f), 0.75f))
    }
}
