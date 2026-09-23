package com.example.roidetection

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The exact analysis frame and the prediction made from it.
 */
data class CombinedResult(
    val frame: Bitmap? = null,
    val roiResult: ROIResult = ROIResult(),
    val isPointing: Boolean = false,
    val totalInferenceTimeMs: Long = 0L
)

class InferenceViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "InferenceViewModel"

        // Temporal smoothing parameters (from Vercel's temporalFilter.js)
        private const val WINDOW_SIZE = 5
        private const val POINT_ENTER_THRESH = 0.30f  // Enter pointing state
    }

    private val _result = MutableStateFlow<CombinedResult>(CombinedResult())
    val result: StateFlow<CombinedResult> = _result.asStateFlow()

    private val _isModelReady = MutableStateFlow(false)
    val isModelReady: StateFlow<Boolean> = _isModelReady.asStateFlow()

    private val _modelError = MutableStateFlow<String?>(null)
    val modelError: StateFlow<String?> = _modelError.asStateFlow()

    private val _fps = MutableStateFlow(0f)
    val fps: StateFlow<Float> = _fps.asStateFlow()

    private val _isRunning = MutableStateFlow(false)
    val isRunning: StateFlow<Boolean> = _isRunning.asStateFlow()

    /** Weights the running session actually loaded, for the on-screen badge. */
    private val _modelInfo = MutableStateFlow<ModelInfo?>(null)
    val modelInfo: StateFlow<ModelInfo?> = _modelInfo.asStateFlow()

    private var spatialNetModel: SpatialNetModel? = null
    private val isProcessing = AtomicBoolean(false)

    // FPS calculation
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var firstFrameLogged = false

    // Temporal smoothing state
    private val gestureWindow = ArrayDeque<Pair<Boolean, Float>>(WINDOW_SIZE)
    private var currentPointingState = false

    fun initializeModel() {
        if (_isModelReady.value) return
        if (_modelError.value != null) return

        _modelError.value = null
        Log.i(TAG, "========== VIEWMODEL: Starting model initialization ==========")
        viewModelScope.launch(Dispatchers.IO) {
            try {
                Log.i(TAG, "Initializing SpatialNet model...")
                val sn = SpatialNetModel(getApplication<Application>())
                sn.initialize()
                spatialNetModel = sn
                _modelInfo.value = sn.modelInfo
                Log.i(TAG, "SpatialNet initialized successfully (${sn.modelInfo})")

                _isModelReady.value = true
                Log.i(TAG, "========== VIEWMODEL: Model initialized ==========")
            } catch (e: Exception) {
                _isModelReady.value = false
                _modelError.value = e.message ?: "Unknown error loading models"
                Log.e(TAG, "========== VIEWMODEL: Model initialization FAILED ==========")
                Log.e(TAG, "Error: ${e.javaClass.simpleName}: ${e.message}", e)
            }
        }
    }

    fun retryModelInitialization() {
        Log.i(TAG, "Retrying model initialization...")
        _modelError.value = null
        _isModelReady.value = false
        initializeModel()
    }

    fun setRunning(running: Boolean) {
        Log.i(TAG, "setRunning($running)")
        _isRunning.value = running
    }

    fun processFrame(imageProxy: ImageProxy) {
        if (!_isRunning.value) {
            imageProxy.close()
            return
        }
        if (isProcessing.getAndSet(true)) {
            imageProxy.close()
            return
        }

        if (!firstFrameLogged) {
            Log.i(TAG, "processFrame: First frame received! rotation=${imageProxy.imageInfo.rotationDegrees}, size=${imageProxy.width}x${imageProxy.height}")
            firstFrameLogged = true
        }

        viewModelScope.launch(Dispatchers.Default) {
            try {
                val bitmap = imageProxy.toBitmap()
                val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees.toFloat())

                val startTime = System.currentTimeMillis()

                // Run SpatialNet ROI detection
                val roiResult = spatialNetModel?.predict(rotatedBitmap) ?: ROIResult()

                val totalTime = System.currentTimeMillis() - startTime

                // Apply temporal smoothing using SpatialNet's unified outputs
                val isPointing = applyTemporalSmoothing(roiResult)

                val combined = CombinedResult(
                    frame = rotatedBitmap,
                    roiResult = roiResult,
                    isPointing = isPointing,
                    totalInferenceTimeMs = totalTime
                )
                _result.value = combined

                // FPS calculation
                frameCount++
                val now = System.currentTimeMillis()
                val elapsed = now - lastFpsTime
                if (elapsed >= 1000) {
                    _fps.value = frameCount * 1000f / elapsed
                    frameCount = 0
                    lastFpsTime = now
                }

                // The displayed frame is owned by the result until Compose releases it.
                if (rotatedBitmap !== bitmap) bitmap.recycle()
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame: ${e.javaClass.simpleName}: ${e.message}", e)
            } finally {
                isProcessing.set(false)
                imageProxy.close()
            }
        }
    }

    /**
     * Apply temporal smoothing with sliding window voting + confidence hysteresis.
     * Inspired by Vercel's temporalFilter.js and burst-mode detection logic.
     */
    private fun applyTemporalSmoothing(latestResult: ROIResult): Boolean {
        // Add to sliding window
        gestureWindow.addLast(latestResult.isPointing to latestResult.pointConf)
        if (gestureWindow.size > WINDOW_SIZE) {
            gestureWindow.removeFirst()
        }

        // Sliding window voting: if ANY frame in window detected pointing, consider it
        val anyPointingInWindow = gestureWindow.any { it.first }

        // Best pointing confidence in window
        val bestPointConf = gestureWindow
            .filter { it.first }
            .maxOfOrNull { it.second } ?: 0f

        // Best non-pointing confidence in window

        // Confidence hysteresis with window voting
        val isPointing = when {
            // Enter pointing: need high confidence OR any pointing in window with decent confidence
            !currentPointingState && anyPointingInWindow && bestPointConf >= POINT_ENTER_THRESH -> true
            // Stay pointing: lower threshold to exit (hysteresis)
            currentPointingState && !anyPointingInWindow -> false
            // Already pointing and still seeing some pointing
            currentPointingState && anyPointingInWindow -> true
            // Not pointing and no pointing in window
            else -> false
        }

        // Update state
        currentPointingState = isPointing
        Log.d(TAG, "Temporal: window=${gestureWindow.size} anyPt=$anyPointingInWindow " +
                "ptConf=$bestPointConf state=$currentPointingState")

        return isPointing
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        if (degrees == 0f) return bitmap
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    override fun onCleared() {
        super.onCleared()
        spatialNetModel?.close()
        spatialNetModel = null
        _result.value = CombinedResult()
    }
}
