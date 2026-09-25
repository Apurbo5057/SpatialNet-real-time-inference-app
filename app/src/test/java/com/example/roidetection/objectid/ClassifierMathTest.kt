package com.example.roidetection.objectid

import com.example.roidetection.BBox
import com.example.roidetection.LabelScore
import com.example.roidetection.ObjectResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class ClassifierMathTest {

    @Test
    fun softmaxSumsToOneAndKeepsOrder() {
        val p = ClassifierMath.softmax(floatArrayOf(1f, 3f, 2f, 1000f, -1000f))
        assertEquals(1f, p.sum(), 1e-5f)
        assertEquals(3, p.indices.maxByOrNull { p[it] })
        assertTrue(p.all { it.isFinite() })
    }

    @Test
    fun sharedLabelsAreSummed() {
        val labels = listOf("TV / monitor", "Laptop", "TV / monitor", "Cup")
        val top = ClassifierMath.topLabels(floatArrayOf(0.3f, 0.35f, 0.25f, 0.1f), labels, 2)
        assertEquals("TV / monitor", top[0].label)
        assertEquals(0.55f, top[0].score, 1e-6f)
        assertEquals(LabelScore("Laptop", 0.35f), top[1])
    }

    @Test
    fun nchwMatchesImageNetNormalization() {
        // 2x2 image: red, green, blue, white
        val px = intArrayOf(0xFFFF0000.toInt(), 0xFF00FF00.toInt(), 0xFF0000FF.toInt(), 0xFFFFFFFF.toInt())
        val out = FloatArray(12)
        ClassifierMath.toNchw(px, 2, out)
        val m = ClassifierMath.IMAGENET_MEAN
        val s = ClassifierMath.IMAGENET_STD
        assertEquals((1f - m[0]) / s[0], out[0], 1e-6f)      // R plane, red pixel
        assertEquals((0f - m[0]) / s[0], out[1], 1e-6f)      // R plane, green pixel
        assertEquals((1f - m[1]) / s[1], out[4 + 1], 1e-6f)  // G plane, green pixel
        assertEquals((1f - m[2]) / s[2], out[8 + 2], 1e-6f)  // B plane, blue pixel
        assertEquals((1f - m[2]) / s[2], out[8 + 3], 1e-6f)  // B plane, white pixel
    }

    @Test
    fun bundledLabelsFileHasOneLabelPerImageNetClass() {
        val labels = File("../ObjectModel/mobilenetv4_labels.txt").readLines().filter { it.isNotBlank() }
        assertEquals(1000, labels.size)
        assertEquals("TV / monitor", labels[851])
    }

    @Test
    fun objectResultExpiresAndDoesNotFollowAMovedBox() {
        val box = BBox(0.5f, 0.5f, 0.3f, 0.3f)
        val r = ObjectResult(LabelScore("Lamp", 0.9f), emptyList(), box, computedAtMs = 1000L, inferenceTimeMs = 20L)
        assertTrue(r.appliesTo(box, 1500L))
        assertFalse(r.appliesTo(box, 3000L))
        assertFalse(r.appliesTo(BBox(0.1f, 0.1f, 0.1f, 0.1f), 1500L))
    }

    @Test
    fun iouOfIdenticalAndDisjointBoxes() {
        val a = BBox(0.5f, 0.5f, 0.2f, 0.2f)
        assertEquals(1f, a.iou(a), 1e-6f)
        assertEquals(0f, a.iou(BBox(0.1f, 0.1f, 0.1f, 0.1f)), 0f)
        // Half-overlapping along x: inter 0.1*0.2, union 2*0.04 - 0.02
        assertEquals(0.02f / 0.06f, a.iou(BBox(0.6f, 0.5f, 0.2f, 0.2f)), 1e-5f)
    }
}
