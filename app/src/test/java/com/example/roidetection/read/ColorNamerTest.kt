package com.example.roidetection.read

import org.junit.Assert.assertEquals
import org.junit.Test

class ColorNamerTest {

    private fun rgb(r: Int, g: Int, b: Int) = (0xFF shl 24) or (r shl 16) or (g shl 8) or b

    @Test
    fun namesBasicColours() {
        assertEquals("Red", ColorNamer.name(rgb(220, 30, 30)))
        assertEquals("Green", ColorNamer.name(rgb(40, 180, 60)))
        assertEquals("Blue", ColorNamer.name(rgb(30, 60, 210)))
        assertEquals("Yellow", ColorNamer.name(rgb(230, 210, 30)))
        assertEquals("Orange", ColorNamer.name(rgb(240, 140, 20)))
        assertEquals("Purple", ColorNamer.name(rgb(130, 40, 190)))
        assertEquals("Pink", ColorNamer.name(rgb(230, 60, 160)))
    }

    @Test
    fun namesNeutralsAndShades() {
        assertEquals("Black", ColorNamer.name(rgb(10, 10, 12)))
        assertEquals("White", ColorNamer.name(rgb(250, 250, 248)))
        assertEquals("Grey", ColorNamer.name(rgb(128, 128, 128)))
        assertEquals("Dark blue", ColorNamer.name(rgb(10, 20, 110)))
        assertEquals("Light blue", ColorNamer.name(rgb(170, 200, 250)))
        assertEquals("Brown", ColorNamer.name(rgb(110, 60, 20)))
    }

    @Test
    fun averagesOnlyTheCentre() {
        // 10x10 red image with a green 4x4 centre; the 30% centre (3x3) is all green.
        val px = IntArray(100) { rgb(255, 0, 0) }
        for (y in 3..6) for (x in 3..6) px[y * 10 + x] = rgb(0, 200, 0)
        assertEquals(rgb(0, 200, 0), ColorNamer.averageCenter(px, 10, 10))
    }
}
