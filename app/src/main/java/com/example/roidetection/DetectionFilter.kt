package com.example.roidetection

import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Removes hands, fingertips and targets that SpatialNet invents when nothing is there,
 * without retraining it. SpatialNet sometimes reads finger-shaped things (an earbud stem,
 * a cable, a pattern) as a pointing hand, and its confidences can't separate those from
 * real hands, so this checks what a real pointing hand always satisfies instead:
 *
 * Per frame (geometry):
 *  - the fingertip is on the hand box (at most [maxTipOutside] of the box size outside it),
 *  - the target lies in the pointing direction (within [maxTargetAngleDeg]),
 *  - the target is not mostly the hand itself, and the hand box has a plausible size.
 * Over time:
 *  - a hand or pointing is shown only after [enterFrames] of the last [window] frames agree,
 *    and it stays until fewer than [stayFrames] do, so a one-frame hallucination never shows;
 *  - a hand box that jumps (IoU below [minTrackIou]) starts over as a new, unconfirmed hand.
 *
 * On 156 no-hand photos, the geometry alone removed 5 of 12 fake pointing detections and
 * none of 5 real pointing captures; the frame rule targets fakes that flicker in live video.
 */
class DetectionFilter(
    private val window: Int = 5,
    private val enterFrames: Int = 3,
    private val stayFrames: Int = 2,
    private val maxTipOutside: Float = 0.10f,
    private val maxTargetAngleDeg: Float = 75f,
    private val maxTargetCoveredByHand: Float = 0.7f,
    private val minHandArea: Float = 0.01f,
    private val maxHandArea: Float = 0.85f,
    private val minTrackIou: Float = 0.15f
) {
    /** The detection to show, and whether the user is (steadily) pointing. */
    data class Output(val result: ROIResult, val isPointing: Boolean, val rejectedReason: String?)

    private val handHistory = ArrayDeque<Boolean>()
    private val pointHistory = ArrayDeque<Boolean>()
    private var lastHand: BBox? = null
    private var handShown = false
    private var pointingShown = false

    @Synchronized
    fun reset() {
        handHistory.clear(); pointHistory.clear()
        lastHand = null; handShown = false; pointingShown = false
    }

    /** [aspect] is frame width / height, so angles are measured as they look on screen. */
    @Synchronized
    fun update(raw: ROIResult, aspect: Float): Output {
        var reason: String? = null
        val hand = raw.handBBox

        // --- Per-frame geometry.
        var handOk = raw.isHandDetected && hand != null && (hand.w * hand.h) in minHandArea..maxHandArea
        if (raw.isHandDetected && !handOk) reason = "hand box size"

        var pointOk = handOk && raw.isPointing && raw.fingertip != null
        if (pointOk && tipOutside(hand!!, raw.fingertip!!) > maxTipOutside) {
            // A fingertip that is not on the hand means the hand itself is likely invented.
            pointOk = false; handOk = false; reason = "fingertip off the hand"
        }
        var roiOk = pointOk && raw.roiBBox != null
        if (roiOk) {
            val roi = raw.roiBBox!!
            val angle = raw.pointingDir?.let { targetAngleDeg(raw.fingertip!!, it, roi, aspect) }
            when {
                angle != null && angle > maxTargetAngleDeg -> {
                    pointOk = false; roiOk = false; reason = "target not in pointing direction"
                }
                coveredFraction(roi, hand!!) > maxTargetCoveredByHand -> {
                    roiOk = false; reason = "target is the hand"
                }
            }
        }

        // --- Tracking: a jumping hand box is a new hand.
        if (handOk) {
            val prev = lastHand
            if (prev != null && prev.iou(hand!!) < minTrackIou) {
                handHistory.clear(); pointHistory.clear()
                handShown = false; pointingShown = false
            }
            lastHand = hand
        }

        // --- Over time.
        handShown = vote(handHistory, handOk, handShown)
        pointingShown = vote(pointHistory, pointOk, pointingShown) && handShown
        if (!handShown && handOk) reason = reason ?: "waiting for a steady hand"

        val shown = raw.copy(
            handBBox = if (handShown && handOk) hand else null,
            fingertip = if (pointingShown && pointOk) raw.fingertip else null,
            pointingDir = if (pointingShown && pointOk) raw.pointingDir else null,
            roiBBox = if (pointingShown && roiOk) raw.roiBBox else null,
            isHandDetected = handShown && handOk,
            isPointing = pointingShown && pointOk,
            isRoiDetected = pointingShown && roiOk
        )
        return Output(shown, pointingShown, reason)
    }

    private fun vote(history: ArrayDeque<Boolean>, now: Boolean, shown: Boolean): Boolean {
        history.addLast(now)
        while (history.size > window) history.removeFirst()
        val count = history.count { it }
        return if (shown) count >= stayFrames else count >= enterFrames
    }

    companion object {
        /** Distance of the fingertip outside the hand box, in units of the box's longer side. */
        fun tipOutside(hand: BBox, tip: Point): Float {
            val c = hand.toCorners()
            val dx = max(max(c.x1 - tip.x, 0f), tip.x - c.x2)
            val dy = max(max(c.y1 - tip.y, 0f), tip.y - c.y2)
            return hypot(dx, dy) / max(max(hand.w, hand.h), 1e-6f)
        }

        /** Fraction of [roi] that the hand box covers. */
        fun coveredFraction(roi: BBox, hand: BBox): Float {
            val a = roi.toCorners(); val b = hand.toCorners()
            val iw = min(a.x2, b.x2) - max(a.x1, b.x1)
            val ih = min(a.y2, b.y2) - max(a.y1, b.y1)
            if (iw <= 0f || ih <= 0f) return 0f
            return iw * ih / max(roi.w * roi.h, 1e-6f)
        }

        /**
         * Angle between the pointing direction and the line from fingertip to target centre,
         * in screen proportions; null when the target sits on the fingertip.
         */
        fun targetAngleDeg(tip: Point, dir: Direction, roi: BBox, aspect: Float): Float? {
            val vx = (roi.cx - tip.x) * aspect
            val vy = roi.cy - tip.y
            val dx = dir.cos * aspect
            val dy = dir.sin
            val n = hypot(vx, vy) * hypot(dx, dy)
            if (hypot(roi.cx - tip.x, roi.cy - tip.y) < 0.02f || n < 1e-6f) return null
            val cos = ((vx * dx + vy * dy) / n).coerceIn(-1f, 1f)
            return Math.toDegrees(acos(cos).toDouble()).toFloat()
        }
    }
}
