package com.example.roidetection.objectid

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.roidetection.LabelScore
import com.example.roidetection.ModelInfo
import com.example.roidetection.memory.ObjectMatcher
import java.nio.FloatBuffer

/** Labels and appearance vector for one crop. */
class ClassifierOutput(
    val labels: List<LabelScore>,
    /** L2-normalized pooled features, compared against saved objects; null if the model lacks them. */
    val embedding: FloatArray?
)

/**
 * MobileNetV4 (ImageNet-1k) ONNX classifier for the ROI crop.
 * Input:  "image" 1x3xSxS (NCHW), RGB, ImageNet normalized (S read from the model, 224)
 * Output: "logits" 1x1000, turned into friendly labels via the bundled labels file,
 *         and "embedding" 1x960 pooled features used by My Objects.
 */
class OnnxImageClassifier(
    private val context: Context,
    private val modelAsset: String = ModelInfo.CLASSIFIER_ASSET,
    private val labelsAsset: String = ModelInfo.CLASSIFIER_LABELS_ASSET
) {
    companion object {
        private const val TAG = "OnnxImageClassifier"
        private const val DEFAULT_INPUT_SIZE = 224
        const val TOP_K = 5
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var labels: List<String> = emptyList()
    private var inputSize = DEFAULT_INPUT_SIZE
    private lateinit var pixels: IntArray
    private lateinit var input: FloatArray

    var modelInfo: ModelInfo? = null
        private set

    @Synchronized
    fun initialize() {
        if (session != null) return
        val startTime = System.currentTimeMillis()

        val environment = OrtEnvironment.getEnvironment()
        val modelBytes = context.assets.open(modelAsset).use { it.readBytes() }
        modelInfo = ModelInfo.of(modelAsset, modelBytes)
        labels = context.assets.open(labelsAsset).bufferedReader().readLines().filter { it.isNotBlank() }

        // Two threads, so SpatialNet keeps its four.
        val options = OrtSession.SessionOptions().apply {
            addConfigEntry("session.intra_op_thread_count", "2")
        }
        val sess = environment.createSession(modelBytes, options)
        val shape = (sess.inputInfo.values.first().info as? ai.onnxruntime.TensorInfo)?.shape
        inputSize = shape?.getOrNull(2)?.toInt()?.takeIf { it > 0 } ?: DEFAULT_INPUT_SIZE
        pixels = IntArray(inputSize * inputSize)
        input = FloatArray(3 * inputSize * inputSize)

        env = environment
        session = sess
        Log.i(TAG, "Loaded $modelInfo, ${labels.size} labels, input ${inputSize}px " +
                "in ${System.currentTimeMillis() - startTime}ms")
    }

    /** Best [TOP_K] friendly labels for [bitmap], probabilities summed per label. */
    fun classify(bitmap: Bitmap): List<LabelScore> = analyze(bitmap).labels

    /** Labels plus the appearance vector, from one model run. */
    @Synchronized
    fun analyze(bitmap: Bitmap): ClassifierOutput {
        val sess = session ?: throw IllegalStateException("Classifier not initialized")
        val environment = env ?: throw IllegalStateException("ONNX env not initialized")

        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        if (resized !== bitmap) resized.recycle()
        ClassifierMath.toNchw(pixels, inputSize, input)

        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        var logits = FloatArray(0)
        var embedding: FloatArray? = null
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), shape).use { tensor ->
            sess.run(mapOf(sess.inputNames.first() to tensor)).use { results ->
                for ((name, value) in results) {
                    val buf = (value as? OnnxTensor)?.floatBuffer ?: continue
                    val arr = FloatArray(buf.remaining()).also { buf.get(it) }
                    when (name) {
                        "logits" -> logits = arr
                        "embedding" -> embedding = ObjectMatcher.normalize(arr)
                    }
                }
            }
        }
        val top = ClassifierMath.topLabels(ClassifierMath.softmax(logits), labels, TOP_K)
        return ClassifierOutput(top, embedding)
    }

    @Synchronized
    fun close() {
        session?.close()
        session = null
        env = null
        Log.i(TAG, "Classifier closed")
    }
}
