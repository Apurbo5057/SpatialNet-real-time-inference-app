package com.example.roidetection.objectid

import com.example.roidetection.BBox

/**
 * Decides when a new ROI crop is worth classifying: as soon as an ROI appears,
 * then at most every [intervalMs], or straight away when the box moves
 * (IoU with the last classified box below [moveIou]).
 */
class ClassificationScheduler(
    private val intervalMs: Long = 250L,
    private val moveIou: Float = 0.5f
) {
    private var lastBox: BBox? = null
    private var lastMs = 0L

    @Synchronized
    fun shouldClassify(box: BBox, nowMs: Long): Boolean {
        val prev = lastBox ?: return true
        return nowMs - lastMs >= intervalMs || box.iou(prev) < moveIou
    }

    @Synchronized
    fun onClassified(box: BBox, nowMs: Long) {
        lastBox = box
        lastMs = nowMs
    }

    @Synchronized
    fun reset() {
        lastBox = null
        lastMs = 0L
    }
}
