package com.example.roidetection

import kotlin.math.max
import kotlin.math.min

/**
 * Shared data classes for ROI detection models.
 */

data class ROIResult(
    val handBBox: BBox? = null,
    val fingertip: Point? = null,
    val pointingDir: Direction? = null,
    val roiBBox: BBox? = null,
    val objectness: Float = 0f,
    val handConf: Float = 0f,
    val pointConf: Float = 0f,
    val roiConf: Float = 0f,
    val isHandDetected: Boolean = false,
    val isPointing: Boolean = false,
    val isRoiDetected: Boolean = false,
    val inferenceTimeMs: Long = 0L,
    val rawOutput: FloatArray? = null
)

data class BBox(val cx: Float, val cy: Float, val w: Float, val h: Float) {
    fun toCorners(): Corners {
        val x1 = cx - w / 2f
        val y1 = cy - h / 2f
        val x2 = cx + w / 2f
        val y2 = cy + h / 2f
        return Corners(x1, y1, x2, y2)
    }
}

data class Corners(val x1: Float, val y1: Float, val x2: Float, val y2: Float) {
    fun clamp(): Corners = Corners(
        max(0f, min(1f, x1)),
        max(0f, min(1f, y1)),
        max(0f, min(1f, x2)),
        max(0f, min(1f, y2))
    )
}

data class Point(val x: Float, val y: Float)

data class Direction(val sin: Float, val cos: Float) {
    fun angleRadians(): Float = kotlin.math.atan2(sin.toDouble(), cos.toDouble()).toFloat()
}
