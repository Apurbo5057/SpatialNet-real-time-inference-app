package com.example.roidetection.earbuds

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.roidetection.ModelInfo
import com.example.roidetection.memory.ObjectMatcher
import com.example.roidetection.objectid.ClassifierMath
import org.json.JSONArray
import org.json.JSONObject
import java.nio.FloatBuffer

/**
 * The bundled earbuds catalog (assets/earbuds/, built by tools/earbuds/build_catalog.py)
 * plus the MobileCLIP2-S0 image encoder that recognises the models in it.
 */
class EarbudsCatalog private constructor(
    val models: List<EarbudModel>,
    val recognizer: EarbudsRecognizer,
    val modelInfo: ModelInfo,
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputSize: Int
) {
    companion object {
        private const val TAG = "EarbudsCatalog"
        private const val DIR = "earbuds"

        /** Loads everything; throws if the catalog or model is missing from the APK. */
        fun load(context: Context): EarbudsCatalog {
            val start = System.currentTimeMillis()
            val assets = context.assets
            val json = JSONObject(assets.open("$DIR/earbuds_catalog.json").bufferedReader().readText())
            val models = json.getJSONArray("entries").let { arr -> (0 until arr.length()).map { parseModel(arr.getJSONObject(it)) } }
            val modelJson = json.getJSONObject("model")
            val inputSize = modelJson.optInt("inputSize", 256)
            val minSimilarity = json.getJSONObject("thresholds").getDouble("minSimilarity").toFloat()

            val classifier = EarbudsBinary.parseClassifier(assets.open("$DIR/earbuds_classifier.bin").use { it.readBytes() })
            val references = EarbudsBinary.parseReferences(assets.open("$DIR/earbuds_embeddings.bin").use { it.readBytes() })
            check(classifier.classes == models.size) { "classifier has ${classifier.classes} classes for ${models.size} models" }

            val modelAsset = ModelInfo.EARBUDS_MODEL_ASSET
            val bytes = assets.open(modelAsset).use { it.readBytes() }
            val info = ModelInfo.of(modelAsset, bytes)
            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply { addConfigEntry("session.intra_op_thread_count", "2") }
            val session = env.createSession(bytes, options)

            Log.i(TAG, "Loaded ${models.size} models, ${references.count} references, $info " +
                    "in ${System.currentTimeMillis() - start}ms")
            return EarbudsCatalog(models, EarbudsRecognizer(classifier, references, minSimilarity), info, env, session, inputSize)
        }

        private fun parseModel(j: JSONObject): EarbudModel {
            val specs = j.optJSONObject("specs")
            val raw = HashMap<String, Any?>()
            specs?.keys()?.forEach { key ->
                raw[key] = when (val v = specs.get(key)) {
                    JSONObject.NULL -> null
                    is JSONArray -> (0 until v.length()).map { v.get(it).toString() }
                    else -> v
                }
            }
            val bt = j.getJSONObject("bluetooth")
            return EarbudModel(
                id = j.getString("id"),
                brand = j.getString("brand"),
                model = j.getString("model"),
                specs = EarbudModel.specRows(raw),
                features = (raw["otherFeatures"] as? List<*>)?.map { it.toString() } ?: emptyList(),
                namePatterns = bt.optJSONArray("namePatterns").strings(),
                defaultNames = bt.optJSONArray("defaultNames").strings(),
                pairingSteps = j.optJSONArray("pairingSteps").strings(),
                thumbnailAsset = j.optString("thumbnail").ifEmpty { null }?.let { "$DIR/$it" },
                verifiedOnDevice = bt.optBoolean("verifiedOnDevice", false)
            )
        }

        private fun JSONArray?.strings(): List<String> =
            if (this == null) emptyList() else (0 until length()).map { getString(it) }
    }

    private val pixels = IntArray(inputSize * inputSize)
    private val input = FloatArray(3 * inputSize * inputSize)

    /** L2-normalized MobileCLIP2 embedding of a square crop. */
    @Synchronized
    fun embed(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        if (resized !== bitmap) resized.recycle()
        ClassifierMath.toNchw(pixels, inputSize, input, ClassifierMath.NO_MEAN, ClassifierMath.NO_STD)
        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { tensor ->
            session.run(mapOf(session.inputNames.first() to tensor)).use { results ->
                val buf = (results[0] as OnnxTensor).floatBuffer
                ObjectMatcher.normalize(FloatArray(buf.remaining()).also { buf.get(it) })
            }
        }
    }

    fun thumbnail(context: Context, model: EarbudModel): Bitmap? = model.thumbnailAsset?.let { path ->
        runCatching { context.assets.open(path).use { BitmapFactory.decodeStream(it) } }.getOrNull()
    }

    @Synchronized
    fun close() {
        session.close()
    }
}
