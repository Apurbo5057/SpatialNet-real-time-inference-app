package com.example.roidetection.ui

import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
import com.example.roidetection.CombinedResult
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

private val BEAM = android.graphics.Color.rgb(255, 185, 56)
private val BEAM_FAINT = android.graphics.Color.argb(26, 255, 185, 56)
private val HAND = android.graphics.Color.argb(110, 255, 255, 255)

/**
 * Draws the detection overlay in the app's beam style: a quiet outline around the hand,
 * an amber beam from the fingertip to the target, amber corner brackets on the target,
 * and the answer chip. One renderer for the live view and for Capture, so the saved
 * photo always matches what was on screen.
 *
 * ([left], [top], [width], [height]) is where the frame is drawn on [canvas]; line
 * widths scale with it, so both the phone screen and the saved photo look the same.
 */
fun drawBeamOverlay(canvas: Canvas, result: CombinedResult, left: Float, top: Float, width: Float, height: Float) {
    val unit = max(width, height) / 360f
    val roi = result.roiResult
    fun x(v: Float) = left + v * width
    fun y(v: Float) = top + v * height

    val handPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = HAND; style = Paint.Style.STROKE; strokeWidth = 1.5f * unit
    }
    roi.handBBox?.toCorners()?.clamp()?.let { c ->
        val r = 6f * unit
        canvas.drawRoundRect(RectF(x(c.x1), y(c.y1), x(c.x2), y(c.y2)), r, r, handPaint)
    }
    if (!result.isPointing) return

    val target = roi.roiBBox?.toCorners()?.clamp()
    val tip = roi.fingertip

    // The beam: from the fingertip to the target's centre, or along the pointing direction.
    if (tip != null) {
        val sx = x(tip.x)
        val sy = y(tip.y)
        val (ex, ey) = when {
            target != null -> x((target.x1 + target.x2) / 2) to y((target.y1 + target.y2) / 2)
            roi.pointingDir != null -> {
                val a = roi.pointingDir.angleRadians().toDouble()
                (sx + 34f * unit * cos(a).toFloat()) to (sy + 34f * unit * sin(a).toFloat())
            }
            else -> sx to sy
        }
        if (ex != sx || ey != sy) {
            val beam = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.STROKE
                strokeWidth = 3f * unit
                strokeCap = Paint.Cap.ROUND
                shader = LinearGradient(sx, sy, ex, ey,
                    android.graphics.Color.argb(230, 255, 185, 56), android.graphics.Color.argb(40, 255, 185, 56),
                    Shader.TileMode.CLAMP)
            }
            canvas.drawLine(sx, sy, ex, ey, beam)
        }
        val dot = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = BEAM }
        canvas.drawCircle(sx, sy, 4.5f * unit, dot)
        dot.apply { color = android.graphics.Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 1.5f * unit }
        canvas.drawCircle(sx, sy, 7.5f * unit, dot)
    }

    // The target: faint amber wash and corner brackets.
    if (target != null) {
        val l = x(target.x1); val t = y(target.y1); val r = x(target.x2); val b = y(target.y2)
        canvas.drawRoundRect(RectF(l, t, r, b), 4f * unit, 4f * unit, Paint().apply { color = BEAM_FAINT })
        val len = min(r - l, b - t) * 0.24f
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = BEAM; style = Paint.Style.STROKE; strokeWidth = 3.5f * unit; strokeCap = Paint.Cap.ROUND
        }
        for ((cx, cy, dx, dy) in listOf(
            Quad(l, t, 1f, 1f), Quad(r, t, -1f, 1f), Quad(l, b, 1f, -1f), Quad(r, b, -1f, -1f)
        )) {
            canvas.drawLine(cx, cy, cx + len * dx, cy, p)
            canvas.drawLine(cx, cy, cx, cy + len * dy, p)
        }
        drawObjectLabelChip(canvas, result, l, t, left + width, 10f * unit)
    }
}

private data class Quad(val a: Float, val b: Float, val c: Float, val d: Float)
