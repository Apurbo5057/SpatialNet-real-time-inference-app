package com.example.roidetection.ui

import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Typeface
import com.example.roidetection.CombinedResult
import kotlin.math.max

private val CHIP_READY = android.graphics.Color.rgb(255, 185, 56)      // Beam, like the target brackets
private val CHIP_PENDING = android.graphics.Color.argb(200, 18, 35, 61)  // Glass (Ink)
private val CHIP_READY_TEXT = android.graphics.Color.rgb(18, 35, 61)     // Ink

/** Typeface for text drawn on the camera canvas; set to Atkinson Hyperlegible at startup. */
object OverlayFont {
    var typeface: Typeface = Typeface.DEFAULT_BOLD
}

/** Text for the chip on the ROI box, or null when the current mode has nothing to say. */
fun objectLabelText(result: CombinedResult): String? = result.roiLabel?.text

/**
 * Draws the object label chip above the ROI box whose top-left corner is
 * ([boxLeft], [boxTop]), or just inside it when there is no room above.
 * Shared by the live overlay and Capture so both show the same thing.
 */
fun drawObjectLabelChip(
    canvas: Canvas,
    result: CombinedResult,
    boxLeft: Float,
    boxTop: Float,
    canvasWidth: Float,
    textSize: Float
) {
    val text = objectLabelText(result) ?: return
    val confident = result.roiLabel?.confident == true

    val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = if (confident) CHIP_READY_TEXT else android.graphics.Color.WHITE
        this.textSize = textSize
        typeface = OverlayFont.typeface
    }
    val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = if (confident) CHIP_READY else CHIP_PENDING
        style = Paint.Style.FILL
    }

    val metrics = textPaint.fontMetrics
    val padH = textSize * 0.4f
    val padV = textSize * 0.2f
    val chipW = textPaint.measureText(text) + 2 * padH
    val chipH = metrics.descent - metrics.ascent + 2 * padV
    val gap = 4f

    val top = if (boxTop - chipH - gap >= 0f) boxTop - chipH - gap else boxTop + gap
    val left = boxLeft.coerceIn(0f, max(0f, canvasWidth - chipW))
    val radius = chipH / 2f
    canvas.drawRoundRect(left, top, left + chipW, top + chipH, radius, radius, bgPaint)
    canvas.drawText(text, left + padH, top + padV - metrics.ascent, textPaint)
}
