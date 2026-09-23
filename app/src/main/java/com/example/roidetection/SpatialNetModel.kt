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
 * SpatialNet-Fastest v3 ONNX model for ROI detection.
 * Input:  "image" 1x3x160x160 (NCHW), RGB, ImageNet normalized
 * Output: hand (cx,cy,w,h), fingertip (x,y), angle (sin,cos), roi (cx,cy,w,h)
 *         and the three cascade gates hand_conf / point_conf / roi_conf,
 *         all three of which are RAW LOGITS and need a sigmoid.
 */
class SpatialNetModel(private val context: Context) {

    companion object {
        private const val TAG = "SpatialNetModel"
        private val MODEL_FILE = ModelInfo.SPATIALNET_ASSET
        private const val INPUT_SIZE = 160

        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.3f
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var isInitialized = false

    /** Identity of the weights this session loaded; null until initialized. */
    var modelInfo: ModelInfo? = null
        private set

    fun initialize() {
        if (isInitialized) return
        try {
            Log.i(TAG, "========== SpatialNet INITIALIZATION START ==========")
            val startTime = System.currentTimeMillis()

            env = OrtEnvironment.getEnvironment()

            val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            modelInfo = ModelInfo.of(MODEL_FILE, modelBytes)
            Log.i(TAG, "Loaded model: $modelInfo")

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

        // Parse outputs by name
        var handTensor: OnnxTensor? = null
        var fingertipTensor: OnnxTensor? = null
        var angleTensor: OnnxTensor? = null
        var roiTensor: OnnxTensor? = null
        var handConfTensor: OnnxTensor? = null
        var pointConfTensor: OnnxTensor? = null
        var roiConfTensor: OnnxTensor? = null

        for ((name, value) in results) {
            val tensor = value as? OnnxTensor ?: continue
            when (name) {
                "hand" -> handTensor = tensor
                "fingertip" -> fingertipTensor = tensor
                "angle" -> angleTensor = tensor
                "roi" -> roiTensor = tensor
                "hand_conf" -> handConfTensor = tensor
                "point_conf" -> pointConfTensor = tensor
                "roi_conf" -> roiConfTensor = tensor
            }
        }

        val handArray = FloatArray(4)
        handTensor?.floatBuffer?.get(handArray)

        val fingertipArray = FloatArray(2)
        fingertipTensor?.floatBuffer?.get(fingertipArray)

        val angleArray = FloatArray(2)
        angleTensor?.floatBuffer?.get(angleArray)

        val roiArray = FloatArray(4)
        roiTensor?.floatBuffer?.get(roiArray)

        val handConfArray = FloatArray(1)
        handConfTensor?.floatBuffer?.get(handConfArray)

        val pointConfArray = FloatArray(1)
        pointConfTensor?.floatBuffer?.get(pointConfArray)

        val roiConfArray = FloatArray(1)
        roiConfTensor?.floatBuffer?.get(roiConfArray)

        results.close()

        val inferenceTime = System.currentTimeMillis() - startTime
        Log.d(TAG, "Inference complete in ${inferenceTime}ms")

        return parseOutput(
            hand = handArray,
            fingertip = fingertipArray,
            angle = angleArray,
            roi = roiArray,
            handConf = handConfArray[0],
            pointConf = pointConfArray[0],
            roiConf = roiConfArray[0],
            inferenceTimeMs = inferenceTime
        )
    }

    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + kotlin.math.exp(-x.toDouble()).toFloat())
    }

    private fun parseOutput(
        hand: FloatArray,
        fingertip: FloatArray,
        angle: FloatArray,
        roi: FloatArray,
        handConf: Float,
        pointConf: Float,
        roiConf: Float,
        inferenceTimeMs: Long
    ): ROIResult {
        val handProb = sigmoid(handConf)
        val pointProb = sigmoid(pointConf)
        val roiProb = sigmoid(roiConf)

        val isHand = handProb >= 0.5f
        val isPointing = isHand && pointProb >= 0.5f
        val isRoi = isPointing && roiProb >= 0.5f

        val handBBox = if (isHand) BBox(hand[0], hand[1], hand[2], hand[3]) else null
        val fingertipPoint = if (isPointing) Point(fingertip[0], fingertip[1]) else null
        val pointingDir = if (isPointing) Direction(angle[0], angle[1]) else null
        val roiBBox = if (isRoi) BBox(roi[0], roi[1], roi[2], roi[3]) else null

        val objectness = if (isPointing) pointProb else handProb

        return ROIResult(
            handBBox = handBBox,
            fingertip = fingertipPoint,
            pointingDir = pointingDir,
            roiBBox = roiBBox,
            objectness = objectness,
            handConf = handProb,
            pointConf = pointProb,
            roiConf = roiProb,
            isHandDetected = isHand,
            isPointing = isPointing,
            isRoiDetected = isRoi,
            inferenceTimeMs = inferenceTimeMs,
            rawOutput = floatArrayOf(
                hand[0], hand[1], hand[2], hand[3],
                fingertip[0], fingertip[1],
                angle[0], angle[1],
                roi[0], roi[1], roi[2], roi[3],
                objectness
            )
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
