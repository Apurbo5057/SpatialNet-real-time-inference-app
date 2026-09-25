package com.example.roidetection.objectid

import com.example.roidetection.BBox
import com.example.roidetection.LabelScore

/**
 * Stabilizes classifier output for one ROI "track" with an exponential moving
 * average per label, so the shown name does not flicker between classes.
 *
 * The track restarts when the ROI jumps (IoU with the previous box below
 * [resetIou]) or when no update arrived for [resetGapMs], i.e. the ROI was lost.
 */
class LabelSmoother(
    private val alpha: Float = 0.5f,
    private val resetGapMs: Long = 600L,
    private val resetIou: Float = 0.3f,
    private val showThreshold: Float = 0.35f,
    private val maxCandidates: Int = 3
) {
    data class Smoothed(val top: LabelScore?, val candidates: List<LabelScore>)

    private val scores = HashMap<String, Float>()
    private var lastBox: BBox? = null
    private var lastMs = 0L

    @Synchronized
    fun update(latest: List<LabelScore>, box: BBox, nowMs: Long): Smoothed {
        val prev = lastBox
        val newTrack = prev == null || nowMs - lastMs > resetGapMs || box.iou(prev) < resetIou
        if (newTrack) {
            scores.clear()
            for (s in latest) scores[s.label] = s.score
        } else {
            for (key in scores.keys.toList()) scores[key] = scores.getValue(key) * (1f - alpha)
            for (s in latest) scores[s.label] = (scores[s.label] ?: 0f) + alpha * s.score
            scores.values.removeAll { it < 0.01f }
        }
        lastBox = box
        lastMs = nowMs

        val candidates = scores.entries
            .sortedByDescending { it.value }
            .take(maxCandidates)
            .map { LabelScore(it.key, it.value) }
        val top = candidates.firstOrNull()?.takeIf { it.score >= showThreshold }
        return Smoothed(top, candidates)
    }

    @Synchronized
    fun reset() {
        scores.clear()
        lastBox = null
        lastMs = 0L
    }
}
