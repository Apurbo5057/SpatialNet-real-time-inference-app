package com.example.roidetection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Gesture detection result from YOLO-Fastest V2.
 */
data class GestureResult(
    val isPointing: Boolean = false,
    val confidence: Float = 0f,
    val handBBox: BBox? = null,
    val inferenceTimeMs: Long = 0L,
    val diagnosticInfo: String = ""
)

/**
 * YOLO-Fastest V2 Gesture Classifier ONNX model.
 * Matches the Vercel app's modelService.js decoding exactly.
 */
class YOLOGestureModel(private val context: Context) {

    companion object {
        private const val TAG = "YOLOGestureModel"
        private const val MODEL_FILE = "yolo_fastest_gesture_binary_fp32.onnx"
        private const val INPUT_SIZE = 352
        private const val NUM_CLASSES = 2
        private const val CONF_THRESH = 0.15f
        private const val IOU_THRESH = 0.5f

        private val ANCHORS_BY_SCALE = arrayOf(
            arrayOf(floatArrayOf(23.06f, 39.50f), floatArrayOf(43.81f, 70.43f), floatArrayOf(107.09f, 166.69f)),
            arrayOf(floatArrayOf(146.80f, 246.85f), floatArrayOf(187.71f, 149.59f), floatArrayOf(224.67f, 245.59f))
        )
        private val STRIDES = intArrayOf(16, 32)
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var isInitialized = false

    // Cached output info from initialization
    private var outputInfoStr: String = ""

    fun initialize() {
        if (isInitialized) return
        try {
            Log.i(TAG, "========== YOLO Gesture INITIALIZATION START ==========")
            val startTime = System.currentTimeMillis()

            env = OrtEnvironment.getEnvironment()

            val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            Log.i(TAG, "Model file loaded, size: ${modelBytes.size} bytes")

            val options = OrtSession.SessionOptions().apply {
                addConfigEntry("session.intra_op_thread_count", "4")
            }
            session = env?.createSession(modelBytes, options)

            // Log output info
            session?.outputNames?.let { names ->
                Log.i(TAG, "Output names: $names")
                outputInfoStr = "outs=${names.size}:$names"
            }
            Log.i(TAG, "Output info: $outputInfoStr")

            isInitialized = true
            val elapsed = System.currentTimeMillis() - startTime
            Log.i(TAG, "========== YOLO Gesture INITIALIZATION COMPLETE in ${elapsed}ms ==========")
        } catch (e: Exception) {
            Log.e(TAG, "========== YOLO Gesture INITIALIZATION FAILED ==========")
            Log.e(TAG, "Error: ${e.javaClass.simpleName}: ${e.message}", e)
            throw e
        }
    }

    fun predict(bitmap: Bitmap): GestureResult {
        val startTime = System.currentTimeMillis()

        val sess = session ?: throw IllegalStateException("YOLO not initialized")
        val environment = env ?: throw IllegalStateException("ONNX env not initialized")

        // Preprocess: resize to 352x352, /255, BGR, NCHW
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val floatBuffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // NCHW with BGR channel order
        val numPixels = INPUT_SIZE * INPUT_SIZE
        for (i in 0 until numPixels) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            floatBuffer.put(0 * numPixels + i, b)
            floatBuffer.put(1 * numPixels + i, g)
            floatBuffer.put(2 * numPixels + i, r)
        }

        if (resized !== bitmap) resized.recycle()

        val inputTensor = OnnxTensor.createTensor(environment, floatBuffer, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))

        val inputName = sess.inputNames.iterator().next()
        val results = sess.run(mapOf(inputName to inputTensor))
        inputTensor.close()

        // Decode YOLO outputs
        val extractResult = extractDetections(results, sess)
        results.close()

        val inferenceTime = System.currentTimeMillis() - startTime
        val detections = extractResult.detections
        val nmsBoxes = nms(detections, IOU_THRESH)

        val pointBoxes = nmsBoxes.filter { it.classId == 1 }
        val nonPointBoxes = nmsBoxes.filter { it.classId == 0 }

        val diagInfo = "d:${detections.size} n:${nmsBoxes.size} p:${pointBoxes.size} np:${nonPointBoxes.size} | ${extractResult.matchInfo}"

        Log.d(TAG, "YOLO ${inferenceTime}ms: $diagInfo")

        if (pointBoxes.isNotEmpty()) {
            val best = pointBoxes.maxBy { it.confidence }
            return GestureResult(
                isPointing = true,
                confidence = best.confidence,
                handBBox = BBox(best.x + best.w / 2f, best.y + best.h / 2f, best.w, best.h),
                inferenceTimeMs = inferenceTime,
                diagnosticInfo = diagInfo
            )
        } else if (nonPointBoxes.isNotEmpty()) {
            val best = nonPointBoxes.maxBy { it.confidence }
            return GestureResult(
                isPointing = false,
                confidence = best.confidence,
                handBBox = BBox(best.x + best.w / 2f, best.y + best.h / 2f, best.w, best.h),
                inferenceTimeMs = inferenceTime,
                diagnosticInfo = diagInfo
            )
        } else {
            return GestureResult(inferenceTimeMs = inferenceTime, diagnosticInfo = diagInfo)
        }
    }

    private data class YOLODetection(
        val x: Float, val y: Float, val w: Float, val h: Float,
        val confidence: Float, val classId: Int
    )

    private data class ExtractResult(
        val detections: List<YOLODetection>,
        val matchInfo: String
    )

    private fun extractDetections(results: OrtSession.Result, sess: OrtSession): ExtractResult {
        val allDetections = mutableListOf<YOLODetection>()
        val matchInfo = StringBuilder()

        // Read all output tensors with their actual shapes
        // IMPORTANT: Iterate over results directly, NOT using sess.outputNames
        // (sess.outputNames keys may not match the result map keys)
        data class TensorInfo(val name: String, val data: FloatArray, val shape: LongArray, val gridSize: Int, val channels: Int)

        val outputTensors = mutableListOf<TensorInfo>()
        val outputNames = mutableListOf<String>()

        for ((name, onnxValue) in results) {
            outputNames.add(name)
            val tensor = onnxValue as? OnnxTensor ?: continue
            val buffer = tensor.floatBuffer
            val array = FloatArray(buffer.remaining())
            buffer.get(array)
            val shape = tensor.info.shape  // LongArray from ONNX Runtime

            // Find min/max for diagnostics
            var minVal = Float.MAX_VALUE
            var maxVal = -Float.MAX_VALUE
            for (v in array) {
                if (v < minVal) minVal = v
                if (v > maxVal) maxVal = v
            }

            // Determine grid size and channels from shape
            // Shape could be [1, C, H, W] or [-1, C, H, W] (dynamic batch)
            val channels = if (shape.size >= 2) shape[1].toInt() else -1
            val spatialH = if (shape.size >= 3) shape[2].toInt() else -1
            val spatialW = if (shape.size >= 4) shape[3].toInt() else -1
            // Handle dynamic dimensions (-1): use array size to infer
            val gridSize = when {
                spatialH > 0 && spatialW > 0 -> spatialH  // Use spatialH as grid
                channels > 0 && array.size > 0 -> {
                    // Infer: array.size = batch * channels * gridH * gridW
                    // Assume batch=1, gridH=gridW
                    val remaining = array.size / max(channels, 1)
                    val sqrt = kotlin.math.sqrt(remaining.toDouble()).toInt()
                    if (sqrt * sqrt == remaining) sqrt else -1
                }
                else -> -1
            }

            Log.d(TAG, "Output '$name': shape=[${shape.joinToString(",")}], inferred(ch=$channels, grid=$gridSize), min=$minVal, max=$maxVal, first10=${array.take(10)}")
            matchInfo.append("$name:sh[${shape.joinToString("x")}]g${gridSize}c${channels};")

            if (gridSize > 0 && channels > 0) {
                outputTensors.add(TensorInfo(name, array, shape, gridSize, channels))
            }
        }

        // Group tensors by grid size, then identify reg/obj/cls by channel count
        val byGridSize = outputTensors.groupBy { it.gridSize }

        for ((gridSize, tensors) in byGridSize) {
            val stride = INPUT_SIZE / gridSize
            val numAnchors = 3  // Always 3 anchors per scale
            val gridArea = gridSize * gridSize

            var regOutput: FloatArray? = null
            var objOutput: FloatArray? = null
            var clsOutput: FloatArray? = null

            for (t in tensors) {
                when {
                    t.channels == numAnchors * 4 -> regOutput = t.data
                    t.channels == numAnchors -> objOutput = t.data
                    t.channels == NUM_CLASSES -> clsOutput = t.data
                }
            }

            if (regOutput == null || objOutput == null || clsOutput == null) {
                Log.w(TAG, "Grid $gridSize: reg=${regOutput != null} obj=${objOutput != null} cls=${clsOutput != null}")
                // Try fallback: assign by channel count heuristics
                // If we have 3 tensors for this grid size, assign by channel count order
                val sorted = tensors.sortedByDescending { it.channels }
                if (sorted.size >= 3) {
                    regOutput = sorted[0].data  // Largest channels = reg (12)
                    objOutput = sorted[1].data  // Medium channels = obj (3)
                    clsOutput = sorted[2].data  // Smallest channels = cls (2)
                    Log.w(TAG, "Grid $gridSize: Using heuristic fallback assignment")
                    matchInfo.append("HEURISTIC;")
                }
                continue
            }

            // Find the right anchors for this stride
            val scaleIdx = STRIDES.indexOf(stride)
            val anchors = if (scaleIdx >= 0) ANCHORS_BY_SCALE[scaleIdx] else ANCHORS_BY_SCALE[0]

            decodeScale(regOutput!!, objOutput!!, clsOutput!!, gridSize, stride, anchors, allDetections)
            Log.d(TAG, "Grid $gridSize (stride $stride): decoded ${allDetections.size} detections so far")
        }

        return ExtractResult(allDetections, matchInfo.toString())
    }

    private fun decodeScale(
        reg: FloatArray, obj: FloatArray, cls: FloatArray,
        gridSize: Int, stride: Int, anchors: Array<FloatArray>,
        detections: MutableList<YOLODetection>
    ) {
        val numAnchors = anchors.size
        val gridArea = gridSize * gridSize

        for (gy in 0 until gridSize) {
            for (gx in 0 until gridSize) {
                val gridIdx = gy * gridSize + gx

                for (a in 0 until numAnchors) {
                    val objIdx = a * gridArea + gridIdx
                    val objScore = sigmoid(obj[objIdx])

                    if (objScore < CONF_THRESH) continue

                    var maxClassProb = 0f
                    var classId = -1
                    for (c in 0 until NUM_CLASSES) {
                        val clsIdx = c * gridArea + gridIdx
                        val clsProb = sigmoid(cls[clsIdx])
                        if (clsProb > maxClassProb) {
                            maxClassProb = clsProb
                            classId = c
                        }
                    }

                    val confidence = objScore * maxClassProb
                    if (confidence < CONF_THRESH) continue

                    val regChannelBase = a * 4
                    val tx = reg[(regChannelBase + 0) * gridArea + gridIdx]
                    val ty = reg[(regChannelBase + 1) * gridArea + gridIdx]
                    val tw = reg[(regChannelBase + 2) * gridArea + gridIdx]
                    val th = reg[(regChannelBase + 3) * gridArea + gridIdx]

                    val cx = (sigmoid(tx) + gx) * stride
                    val cy = (sigmoid(ty) + gy) * stride
                    val w = anchors[a][0] * exp(min(max(tw.toDouble(), -5.0), 5.0)).toFloat()
                    val h = anchors[a][1] * exp(min(max(th.toDouble(), -5.0), 5.0)).toFloat()

                    val normX = (cx - w / 2f) / INPUT_SIZE
                    val normY = (cy - h / 2f) / INPUT_SIZE
                    val normW = w / INPUT_SIZE
                    val normH = h / INPUT_SIZE

                    Log.d(TAG, "Det: grid=($gx,$gy) a=$a cls=$classId conf=$confidence obj=$objScore " +
                            "pixel=($cx,$cy,$w,$h)")

                    detections.add(YOLODetection(normX, normY, normW, normH, confidence, classId))
                }
            }
        }
    }

    private fun sigmoid(x: Float): Float = 1.0f / (1.0f + exp(-x.toDouble()).toFloat())

    private fun nms(detections: List<YOLODetection>, iouThreshold: Float): List<YOLODetection> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it.confidence }
        val active = BooleanArray(sorted.size) { true }
        val selected = mutableListOf<YOLODetection>()

        for (i in sorted.indices) {
            if (!active[i]) continue
            selected.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (!active[j]) continue
                if (sorted[i].classId == sorted[j].classId && iou(sorted[i], sorted[j]) > iouThreshold) {
                    active[j] = false
                }
            }
        }
        return selected
    }

    private fun iou(a: YOLODetection, b: YOLODetection): Float {
        val interX1 = max(a.x, b.x)
        val interY1 = max(a.y, b.y)
        val interX2 = min(a.x + a.w, b.x + b.w)
        val interY2 = min(a.y + a.h, b.y + b.h)

        val interArea = max(0f, interX2 - interX1) * max(0f, interY2 - interY1)
        val aArea = a.w * a.h
        val bArea = b.w * b.h
        val unionArea = aArea + bArea - interArea

        return if (unionArea > 0f) interArea / unionArea else 0f
    }

    fun close() {
        session?.close()
        env?.close()
        session = null
        env = null
        isInitialized = false
    }
}
