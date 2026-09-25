package com.example.roidetection.read

import kotlin.math.max
import kotlin.math.min

/** Names an RGB colour in plain words, e.g. "Dark blue" or "Light grey". */
object ColorNamer {

    /** Average colour of the centre [fraction] of an ARGB pixel grid. */
    fun averageCenter(pixels: IntArray, width: Int, height: Int, fraction: Float = 0.3f): Int {
        val w = max(1, (width * fraction).toInt())
        val h = max(1, (height * fraction).toInt())
        val x0 = (width - w) / 2
        val y0 = (height - h) / 2
        var r = 0L; var g = 0L; var b = 0L
        for (y in y0 until y0 + h) for (x in x0 until x0 + w) {
            val p = pixels[y * width + x]
            r += (p shr 16) and 0xFF; g += (p shr 8) and 0xFF; b += p and 0xFF
        }
        val n = (w * h).toLong()
        return (0xFF shl 24) or ((r / n).toInt() shl 16) or ((g / n).toInt() shl 8) or (b / n).toInt()
    }

    fun name(rgb: Int): String {
        val r = ((rgb shr 16) and 0xFF) / 255f
        val g = ((rgb shr 8) and 0xFF) / 255f
        val b = (rgb and 0xFF) / 255f
        val maxC = max(r, max(g, b))
        val minC = min(r, min(g, b))
        val light = (maxC + minC) / 2f
        val delta = maxC - minC
        val sat = if (delta == 0f) 0f else delta / (1f - kotlin.math.abs(2f * light - 1f))

        if (light < 0.12f) return "Black"
        if (light > 0.92f && sat < 0.5f) return "White"
        if (sat < 0.15f) return when {
            light < 0.35f -> "Dark grey"
            light > 0.7f -> "Light grey"
            else -> "Grey"
        }

        val hue = when (maxC) {
            r -> 60f * (((g - b) / delta) % 6f)
            g -> 60f * ((b - r) / delta + 2f)
            else -> 60f * ((r - g) / delta + 4f)
        }.let { if (it < 0f) it + 360f else it }

        // Dark oranges read as brown to most people.
        if (hue in 15f..45f && light < 0.4f) return "Brown"

        val base = when {
            hue < 15f || hue >= 345f -> "Red"
            hue < 45f -> "Orange"
            hue < 70f -> "Yellow"
            hue < 160f -> "Green"
            hue < 200f -> "Cyan"
            hue < 255f -> "Blue"
            hue < 290f -> "Purple"
            else -> "Pink"
        }
        return when {
            light < 0.3f -> "Dark ${base.lowercase()}"
            light > 0.72f -> "Light ${base.lowercase()}"
            else -> base
        }
    }
}
