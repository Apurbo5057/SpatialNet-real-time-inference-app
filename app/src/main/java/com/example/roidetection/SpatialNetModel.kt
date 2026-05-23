package com.example.roidetection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * SpatialNet-Fastest ONNX model for ROI detection.
 * Input: 1x3x160x160 (NCHW), ImageNet normalized
 * Output: 13 values (hand_bbox, fingertip, pointing_dir, roi_bbox, objectness)
 */
class SpatialNetModel(private val context: Context) {

    companion object {
        private const val TAG = "SpatialNetModel"
        private const val MODEL_FILE = "model.onnx"
        private const val INPUT_SIZE = 160

        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.3f
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var isInitialized = false

    fun initialize() {
        if (isInitialized) return
        try {
            Log.i(TAG, "========== SpatialNet INITIALIZATION START ==========")
            val startTime = System.currentTimeMillis()

            env = OrtEnvironment.getEnvironment()

            val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            Log.i(TAG, "Model file loaded, size: ${modelBytes.size} bytes")

            val options = OrtSession.SessionOptions().apply {
                addConfigEntry("session.intra_op_thread_count", "4")
            }
            session = env?.createSession(modelBytes, options)

            // Log input/output info
            session?.inputNames?.let { Log.i(TAG, "Input names: $it") }
            session?.outputNames?.let { Log.i(TAG, "Output names: $it") }

            isInitialized = true
            val elapsed = System.currentTimeMillis() - startTime
            Log.i(TAG, "========== SpatialNet INITIALIZATION COMPLETE in ${elapsed}ms ==========")
        } catch (e: Exception) {
            Log.e(TAG, "========== SpatialNet INITIALIZATION FAILED ==========")
            Log.e(TAG, "Error: ${e.javaClass.simpleName}: ${e.message}", e)
            throw e
        }
    }

    fun predict(bitmap: Bitmap, confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD): ROIResult {
        val startTime = System.currentTimeMillis()
        Log.d(TAG, "predict() - input: ${bitmap.width}x${bitmap.height}")

        val sess = session ?: throw IllegalStateException("SpatialNet not initialized")
        val environment = env ?: throw IllegalStateException("ONNX env not initialized")

        // Preprocess: resize to 160x160, ImageNet normalize, NCHW layout
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val floatBuffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // NCHW: [C][H][W] - channel first
        for (c in 0 until 3) {
            for (h in 0 until INPUT_SIZE) {
                for (w in 0 until INPUT_SIZE) {
                    val pixel = pixels[h * INPUT_SIZE + w]
                    val channelValue = when (c) {
                        0 -> ((pixel shr 16) and 0xFF) / 255.0f  // R
                        1 -> ((pixel shr 8) and 0xFF) / 255.0f   // G
                        else -> (pixel and 0xFF) / 255.0f        // B
                    }
                    floatBuffer.put((channelValue - MEAN[c]) / STD[c])
                }
            }
        }
        floatBuffer.rewind()

        if (resized !== bitmap) resized.recycle()

        // Create input tensor [1, 3, 160, 160]
        val inputTensor = OnnxTensor.createTensor(environment, floatBuffer, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))

        // Run inference
        val inputName = sess.inputNames.iterator().next()
        val results = sess.run(mapOf(inputName to inputTensor))
        inputTensor.close()

        // Parse output
        val outputTensor = results[0] as OnnxTensor
        val outputBuffer = outputTensor.floatBuffer
        val output = FloatArray(outputBuffer.remaining())
        outputBuffer.get(output)
        outputTensor.close()
        results.close()

        val inferenceTime = System.currentTimeMillis() - startTime
        Log.d(TAG, "Inference complete in ${inferenceTime}ms, output size: ${output.size}")
        Log.d(TAG, "First 13 values: ${output.take(13)}")

        return parseOutput(output, inferenceTime, confidenceThreshold)
    }

    private fun parseOutput(output: FloatArray, inferenceTimeMs: Long, threshold: Float): ROIResult {
        if (output.size < 13) {
            Log.w(TAG, "Unexpected output size: ${output.size}, expected 13")
            return ROIResult(inferenceTimeMs = inferenceTimeMs, rawOutput = output)
        }

        val objectness = output[12]
        if (objectness < threshold) {
            return ROIResult(objectness = objectness, inferenceTimeMs = inferenceTimeMs, rawOutput = output)
        }

        return ROIResult(
            handBBox = BBox(output[0], output[1], output[2], output[3]),
            fingertip = Point(output[4], output[5]),
            pointingDir = Direction(output[6], output[7]),
            roiBBox = BBox(output[8], output[9], output[10], output[11]),
            objectness = objectness,
            inferenceTimeMs = inferenceTimeMs,
            rawOutput = output
        )
    }

    fun close() {
        session?.close()
        env?.close()
        session = null
        env = null
        isInitialized = false
        Log.i(TAG, "SpatialNet closed")
    }
}
