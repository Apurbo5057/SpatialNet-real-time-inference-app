package com.example.roidetection.earbuds

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class EarbudsRecognizerTest {

    // 3 classes over 2-d embeddings: class c scores highest when e points along axis c.
    private val classifier = LinearClassifier(
        classes = 3, dim = 2,
        weights = floatArrayOf(5f, 0f, 0f, 5f, -5f, -5f),
        bias = floatArrayOf(0f, 0f, 0f)
    )
    private val references = ReferenceSet(2, 2, floatArrayOf(1f, 0f, 0f, 1f))

    @Test
    fun probabilitiesSumToOneAndPickTheRightClass() {
        val p = classifier.probabilities(floatArrayOf(1f, 0f))
        assertEquals(1f, p.sum(), 1e-5f)
        assertEquals(0, p.indices.maxByOrNull { p[it] })
    }

    @Test
    fun bestSimilarityIsTheClosestReference() {
        assertEquals(1f, references.bestSimilarity(floatArrayOf(0f, 1f)), 1e-6f)
        assertEquals(0.6f, references.bestSimilarity(floatArrayOf(0.6f, -0.8f)), 1e-6f)
    }

    @Test
    fun recognizerRanksCandidatesAndGatesNonEarbuds() {
        val r = EarbudsRecognizer(classifier, references, minSimilarity = 0.7f)
        val yes = r.recognize(floatArrayOf(0.8f, 0.6f), k = 2)
        assertTrue(yes.isEarbuds)
        assertEquals(listOf(0, 1), yes.candidates.map { it.index })
        assertTrue(yes.candidates[0].probability > yes.candidates[1].probability)

        val no = r.recognize(floatArrayOf(-0.6f, -0.8f))
        assertFalse(no.isEarbuds)
        assertEquals(3, no.candidates.size)
    }

    @Test
    fun binaryFilesRoundTrip() {
        val cls = ByteBuffer.allocate(8 + 4 * (3 * 2 + 3)).order(ByteOrder.LITTLE_ENDIAN).apply {
            putInt(3); putInt(2)
            floatArrayOf(5f, 0f, 0f, 5f, -5f, -5f, 0.1f, 0.2f, 0.3f).forEach { putFloat(it) }
        }.array()
        val parsed = EarbudsBinary.parseClassifier(cls)
        assertEquals(3, parsed.classes)
        assertEquals(2, parsed.dim)
        val p = parsed.probabilities(floatArrayOf(0f, 1f))
        assertEquals(1, p.indices.maxByOrNull { p[it] })

        val refs = ByteBuffer.allocate(8 + 2 * (4 + 4 * 2)).order(ByteOrder.LITTLE_ENDIAN).apply {
            putInt(2); putInt(2)
            putInt(0); putFloat(1f); putFloat(0f)
            putInt(1); putFloat(0f); putFloat(1f)
        }.array()
        val r = EarbudsBinary.parseReferences(refs)
        assertEquals(2, r.count)
        assertEquals(1f, r.bestSimilarity(floatArrayOf(0f, 1f)), 1e-6f)
    }

    @Test
    fun builtCatalogFilesParse() {
        val dir = File("../tools/earbuds/out")
        assumeTrue("catalog not built on this machine", File(dir, "earbuds_classifier.bin").exists())
        val cls = EarbudsBinary.parseClassifier(File(dir, "earbuds_classifier.bin").readBytes())
        val refs = EarbudsBinary.parseReferences(File(dir, "earbuds_embeddings.bin").readBytes())
        assertEquals(512, cls.dim)
        assertEquals(512, refs.dim)
        assertEquals(8, cls.classes)
        assertTrue(refs.count > 100)
    }

    private fun model(vararg patterns: String) = EarbudModel(
        "id", "brand", "model", emptyList(), emptyList(), patterns.toList(), emptyList(), emptyList(), null, false
    )

    @Test
    fun catalogNamePatternsMatchTheRightModelsOnly() {
        val realme = model("^realme Buds Air ?7$")
        assertTrue(realme.matchesName("realme Buds Air7"))
        assertTrue(realme.matchesName("realme Buds Air 7"))
        assertFalse(realme.matchesName("realme Buds Air7 Pro"))

        val soundcore = model("(?i)^soundcore R50i(?! NC).*")
        assertTrue(soundcore.matchesName("soundcore R50i"))
        assertFalse(soundcore.matchesName("soundcore R50i NC"))

        val fe = model("^Galaxy Buds FE\\b.*")
        assertTrue(fe.matchesName("Galaxy Buds FE (4F2A)"))
        assertFalse(fe.matchesName("Galaxy Buds3 FE"))

        val redmi = model("(?i)^redmi buds 5$")
        assertTrue(redmi.matchesName("REDMI Buds 5"))
        assertFalse(redmi.matchesName("Redmi Buds 5 Pro"))
        assertFalse(redmi.matchesName(null))
    }

    @Test
    fun displayNameAndSpecRows() {
        val m = EarbudModel("r", "realme", "Buds Air7", emptyList(), emptyList(), emptyList(), emptyList(), emptyList(), null, false)
        assertEquals("realme Buds Air7", m.displayName)
        val g = m.copy(brand = "Samsung", model = "Galaxy Buds FE")
        assertEquals("Samsung Galaxy Buds FE", g.displayName)
        assertEquals("JBL Wave Buds 2", m.copy(brand = "JBL", model = "JBL Wave Buds 2").displayName)

        val rows = EarbudModel.specRows(mapOf("anc" to true, "earbudWeightGrams" to 4.9, "codecs" to listOf("SBC", "AAC"), "bluetoothVersion" to null))
        assertEquals(listOf("Noise cancelling" to "Yes", "Audio codecs" to "SBC, AAC", "Weight (each earbud)" to "4.9 g"), rows)
    }
}
