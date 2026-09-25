package com.example.roidetection.objectid

import com.example.roidetection.BBox
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** A crop rectangle in frame pixels. */
data class PixelRect(val left: Int, val top: Int, val width: Int, val height: Int)

/**
 * Maps a normalized ROI box to the square pixel crop the classifier sees.
 *
 * ROI coordinates are normalized against the whole rotated frame (the
 * SpatialNet input is a plain stretch of it), so pixels are just x * width.
 */
object RoiCropper {
    const val DEFAULT_PADDING = 0.15f
    const val DEFAULT_MIN_SIZE_PX = 32

    /**
     * Square crop around [box], [padding] larger than its longer side, shifted
     * to stay inside the frame. Returns null when the ROI is smaller than
     * [minSizePx] on its longer side, since the classifier cannot read it.
     */
    fun cropRect(
        box: BBox,
        frameWidth: Int,
        frameHeight: Int,
        padding: Float = DEFAULT_PADDING,
        minSizePx: Int = DEFAULT_MIN_SIZE_PX
    ): PixelRect? {
        if (frameWidth <= 0 || frameHeight <= 0) return null
        val boxW = box.w * frameWidth
        val boxH = box.h * frameHeight
        if (max(boxW, boxH) < minSizePx) return null

        val side = min(
            (max(boxW, boxH) * (1f + padding)).roundToInt(),
            min(frameWidth, frameHeight)
        )
        val cx = box.cx * frameWidth
        val cy = box.cy * frameHeight
        val left = (cx - side / 2f).roundToInt().coerceIn(0, frameWidth - side)
        val top = (cy - side / 2f).roundToInt().coerceIn(0, frameHeight - side)
        return PixelRect(left, top, side, side)
    }
}
