package com.example.roidetection

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.roidetection.earbuds.EarbudModel
import com.example.roidetection.earbuds.EarbudsCatalog
import com.example.roidetection.earbuds.EarbudsPairer
import com.example.roidetection.memory.ObjectMatcher
import com.example.roidetection.memory.ObjectStore
import com.example.roidetection.memory.SavedObject
import com.example.roidetection.objectid.ClassificationScheduler
import com.example.roidetection.objectid.LabelSmoother
import com.example.roidetection.objectid.OnnxImageClassifier
import com.example.roidetection.objectid.PixelRect
import com.example.roidetection.objectid.RoiCropper
import com.example.roidetection.read.ColorNamer
import com.example.roidetection.read.ReadResult
import com.example.roidetection.read.TextReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

/** The app's modes, each opened from its own card on the home screen. */
enum class AppMode { IDENTIFY, MY_OBJECTS, READ, EARBUDS }

/**
 * Caption on the ROI box, mirrored by the answer panel. [confident] draws it as a
 * filled green answer; [spoken] is what the speaker button says.
 */
data class RoiLabel(val text: String, val confident: Boolean = false, val spoken: String = text)

/** Which saved object (if any) is inside an ROI box. */
data class PersonalMatch(
    val obj: SavedObject?,
    val score: Float,
    val box: BBox,
    val computedAtMs: Long
) {
    fun appliesTo(roi: BBox, nowMs: Long): Boolean = nowMs - computedAtMs <= 1500L && box.iou(roi) >= 0.3f
}

/**
 * Pair Earbuds: which catalog model is inside an ROI box. [candidates] are the most likely
 * models (best first) that the Bluetooth search looks for; [model] is null when none is
 * likely enough, and [isEarbuds] is false when the crop does not look like earbuds at all.
 */
data class EarbudsMatch(
    val model: EarbudModel?,
    val probability: Float,
    val candidates: List<EarbudModel>,
    val isEarbuds: Boolean,
    val box: BBox,
    val computedAtMs: Long
) {
    fun appliesTo(roi: BBox, nowMs: Long): Boolean = nowMs - computedAtMs <= 1500L && box.iou(roi) >= 0.3f
}

/** Progress of teaching a new object in My Objects mode. */
sealed interface TeachState {
    data object Idle : TeachState
    /** Collecting [count] of [target] samples while the user keeps pointing. */
    data class Collecting(val count: Int, val target: Int) : TeachState
    /** Samples collected; waiting for the user to name the object. */
    data class Naming(val suggestedName: String, val photo: Bitmap?) : TeachState
}

/**
 * The exact analysis frame and the prediction made from it.
 */
data class CombinedResult(
    val frame: Bitmap? = null,
    val roiResult: ROIResult = ROIResult(),
    val isPointing: Boolean = false,
    val totalInferenceTimeMs: Long = 0L,
    /** Chip drawn on the ROI box by the current mode; null draws the plain "ROI" label. */
    val roiLabel: RoiLabel? = null,
    val objectResult: ObjectResult? = null,
    /** My Objects: the saved object inside the ROI box, if recognised. */
    val matchedObject: SavedObject? = null,
    /** Pair Earbuds: the earbuds model inside the ROI box. */
    val earbudsMatch: EarbudsMatch? = null
)

class InferenceViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "InferenceViewModel"

        private const val TEACH_SAMPLES = 5

        private const val AUTO_PAIR_MIN_PROBABILITY = 0.6f
        private const val AUTO_PAIR_FRAMES = 3
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

    /** User toggle for object identification inside the ROI. */
    private val _identifyEnabled = MutableStateFlow(true)
    val identifyEnabled: StateFlow<Boolean> = _identifyEnabled.asStateFlow()

    private val _classifierInfo = MutableStateFlow<ModelInfo?>(null)
    val classifierInfo: StateFlow<ModelInfo?> = _classifierInfo.asStateFlow()

    /** Set if the classifier failed to load; ROI detection keeps working. */
    private val _classifierError = MutableStateFlow<String?>(null)
    val classifierError: StateFlow<String?> = _classifierError.asStateFlow()

    private var spatialNetModel: SpatialNetModel? = null
    private val isProcessing = AtomicBoolean(false)

    // Object identification: runs beside SpatialNet with its own busy flag, so
    // a slow classification skips ROIs instead of delaying frames.
    @Volatile private var classifier: OnnxImageClassifier? = null
    private val isClassifying = AtomicBoolean(false)
    private val scheduler = ClassificationScheduler()
    private val smoother = LabelSmoother()
    @Volatile private var latestObject: ObjectResult? = null

    // ---- Modes ----
    private val _mode = MutableStateFlow(AppMode.IDENTIFY)
    val mode: StateFlow<AppMode> = _mode.asStateFlow()

    /** Shows FPS, timings and model badges; hidden by default to keep screens simple. */
    private val _showDetails = MutableStateFlow(false)
    val showDetails: StateFlow<Boolean> = _showDetails.asStateFlow()

    // ---- My Objects ----
    private val store = ObjectStore(application)
    private val _savedObjects = MutableStateFlow<List<SavedObject>>(emptyList())
    val savedObjects: StateFlow<List<SavedObject>> = _savedObjects.asStateFlow()

    private val _teachState = MutableStateFlow<TeachState>(TeachState.Idle)
    val teachState: StateFlow<TeachState> = _teachState.asStateFlow()

    private val personalSmoother = LabelSmoother(showThreshold = ObjectMatcher.DEFAULT_THRESHOLD)
    @Volatile private var latestPersonal: PersonalMatch? = null
    private val teachSamples = ArrayList<FloatArray>()
    private val teachVotes = HashMap<String, Float>()
    private var teachPhoto: Bitmap? = null
    private val lastSeenSavedMs = HashMap<String, Long>()

    // ---- Read ----
    private var textReader: TextReader? = null
    private val readScheduler = ClassificationScheduler(intervalMs = 700L)
    private val _readResult = MutableStateFlow<ReadResult?>(null)
    val readResult: StateFlow<ReadResult?> = _readResult.asStateFlow()
    @Volatile private var readWholeViewRequested = false
    @Volatile private var latestReadBox: BBox? = null

    // ---- Pair Earbuds ----
    @Volatile private var earbudsCatalog: EarbudsCatalog? = null
    private val _earbudsReady = MutableStateFlow(false)
    val earbudsReady: StateFlow<Boolean> = _earbudsReady.asStateFlow()
    private val _earbudsError = MutableStateFlow<String?>(null)
    val earbudsError: StateFlow<String?> = _earbudsError.asStateFlow()
    private val _earbudsModelInfo = MutableStateFlow<ModelInfo?>(null)
    val earbudsModelInfo: StateFlow<ModelInfo?> = _earbudsModelInfo.asStateFlow()
    private val earbudsSmoother = LabelSmoother(showThreshold = 0.35f)
    @Volatile private var latestEarbuds: EarbudsMatch? = null
    private var earbudsLoading = false
    val pairer = EarbudsPairer(application)

    // Auto-pair: start pairing by itself once one model has been recognised confidently for
    // AUTO_PAIR_FRAMES recognitions in a row (~1 s), at most once per model per visit.
    @Volatile private var autoPairStreak = 0
    @Volatile private var autoPairStreakId: String? = null
    private val autoPairTried = HashSet<String>()

    init {
        viewModelScope.launch(Dispatchers.IO) { _savedObjects.value = store.load() }
    }

    // FPS calculation
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var firstFrameLogged = false

    // Removes invented hands and targets before anything is drawn or identified.
    private val detectionFilter = DetectionFilter()

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

                initializeClassifier()
            } catch (e: Exception) {
                _isModelReady.value = false
                _modelError.value = e.message ?: "Unknown error loading models"
                Log.e(TAG, "========== VIEWMODEL: Model initialization FAILED ==========")
                Log.e(TAG, "Error: ${e.javaClass.simpleName}: ${e.message}", e)
            }
        }
    }

    private fun initializeClassifier() {
        if (classifier != null || _classifierError.value != null) return
        try {
            val clf = OnnxImageClassifier(getApplication<Application>())
            clf.initialize()
            classifier = clf
            _classifierInfo.value = clf.modelInfo
            Log.i(TAG, "Classifier initialized (${clf.modelInfo})")
        } catch (e: Exception) {
            _classifierError.value = e.message ?: e.javaClass.simpleName
            Log.e(TAG, "Classifier initialization FAILED: ${e.javaClass.simpleName}: ${e.message}", e)
        }
    }

    fun setIdentifyEnabled(enabled: Boolean) {
        Log.i(TAG, "setIdentifyEnabled($enabled)")
        _identifyEnabled.value = enabled
        if (!enabled) {
            latestObject = null
            smoother.reset()
            scheduler.reset()
        }
    }

    fun retryModelInitialization() {
        Log.i(TAG, "Retrying model initialization...")
        _modelError.value = null
        _classifierError.value = null
        _isModelReady.value = false
        initializeModel()
    }

    fun setRunning(running: Boolean) {
        Log.i(TAG, "setRunning($running)")
        if (running && !_isRunning.value) detectionFilter.reset()
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
                val rawResult = spatialNetModel?.predict(rotatedBitmap) ?: ROIResult()

                val totalTime = System.currentTimeMillis() - startTime

                // Drop invented hands/fingertips/targets: geometry checks plus a steady-frames vote.
                val filtered = detectionFilter.update(rawResult, rotatedBitmap.width.toFloat() / rotatedBitmap.height)
                val roiResult = filtered.result
                val isPointing = filtered.isPointing
                filtered.rejectedReason?.let { Log.d(TAG, "Filtered detection: $it") }

                // The current mode looks at the ROI; its answer reaches the UI on a later frame.
                val frameTimeMs = System.currentTimeMillis()
                val roi = roiResult.roiBBox?.takeIf { isPointing }
                var roiLabel: RoiLabel? = null
                var objectResult: ObjectResult? = null
                var matchedObject: SavedObject? = null
                var earbudsMatch: EarbudsMatch? = null
                when (_mode.value) {
                    AppMode.IDENTIFY -> if (roi != null && _identifyEnabled.value && classifier != null) {
                        val crop = RoiCropper.cropRect(roi, rotatedBitmap.width, rotatedBitmap.height)
                        if (crop != null) maybeAnalyze(rotatedBitmap, roi, crop, frameTimeMs)
                        objectResult = latestObject?.takeIf { it.appliesTo(roi, frameTimeMs) }
                        roiLabel = when {
                            objectResult != null -> objectResult.top
                                ?.let { RoiLabel("${it.label} ${"%.0f".format(it.score * 100)}%", true, it.label) }
                                ?: RoiLabel("Unknown object")
                            crop == null -> RoiLabel("Move closer")
                            else -> RoiLabel("Identifying…")
                        }
                    }
                    AppMode.MY_OBJECTS -> if (roi != null && classifier != null) {
                        val crop = RoiCropper.cropRect(roi, rotatedBitmap.width, rotatedBitmap.height)
                        if (crop != null) maybeAnalyze(rotatedBitmap, roi, crop, frameTimeMs)
                        val match = latestPersonal?.takeIf { it.appliesTo(roi, frameTimeMs) }
                        // Latest name and note, in case they were edited since the match.
                        matchedObject = match?.obj?.let { m -> _savedObjects.value.firstOrNull { it.id == m.id } }
                        val teach = _teachState.value
                        roiLabel = when {
                            teach is TeachState.Collecting -> RoiLabel("Learning… ${teach.count}/${teach.target}")
                            crop == null -> RoiLabel("Move closer")
                            _savedObjects.value.isEmpty() -> RoiLabel("New object")
                            match == null -> RoiLabel("Looking…")
                            matchedObject != null -> RoiLabel(
                                matchedObject.name, true,
                                "This is your ${matchedObject.name}" +
                                        matchedObject.note.takeIf { it.isNotBlank() }?.let { ". Note: $it" }.orEmpty()
                            )
                            else -> RoiLabel("Not one of your objects")
                        }
                    }
                    AppMode.READ -> roiLabel = readStage(rotatedBitmap, roi, frameTimeMs)
                    AppMode.EARBUDS -> if (roi != null) {
                        val catalog = earbudsCatalog
                        val crop = RoiCropper.cropRect(roi, rotatedBitmap.width, rotatedBitmap.height)
                        if (catalog != null && crop != null) maybeRecognizeEarbuds(catalog, rotatedBitmap, roi, crop, frameTimeMs)
                        val match = latestEarbuds?.takeIf { it.appliesTo(roi, frameTimeMs) }
                        earbudsMatch = match
                        roiLabel = when {
                            catalog == null -> RoiLabel(if (_earbudsError.value != null) "Earbuds unavailable" else "Getting ready…")
                            crop == null -> RoiLabel("Move closer")
                            match == null -> RoiLabel("Looking…")
                            match.model != null -> RoiLabel(
                                "${match.model.displayName} ${"%.0f".format(match.probability * 100)}%",
                                true, match.model.displayName
                            )
                            !match.isEarbuds -> RoiLabel("Not earbuds")
                            else -> RoiLabel("Unknown earbuds")
                        }
                    }
                }

                val combined = CombinedResult(
                    frame = rotatedBitmap,
                    roiResult = roiResult,
                    isPointing = isPointing,
                    totalInferenceTimeMs = totalTime,
                    roiLabel = roiLabel,
                    objectResult = objectResult,
                    matchedObject = matchedObject,
                    earbudsMatch = earbudsMatch
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
     * Runs the classifier on [crop] of [frame] in the background if the scheduler
     * asks for it and no run is already in progress. One run gives both the
     * Identify labels and the My Objects appearance vector.
     */
    private fun maybeAnalyze(frame: Bitmap, roi: BBox, crop: PixelRect, frameTimeMs: Long) {
        val clf = classifier ?: return
        if (!scheduler.shouldClassify(roi, frameTimeMs)) return
        if (isClassifying.getAndSet(true)) return
        scheduler.onClassified(roi, frameTimeMs)
        val mode = _mode.value

        viewModelScope.launch(Dispatchers.Default) {
            var cropBitmap: Bitmap? = null
            try {
                // The frame stays alive in its CombinedResult, so copying from it here is safe.
                cropBitmap = Bitmap.createBitmap(frame, crop.left, crop.top, crop.width, crop.height)
                val start = System.currentTimeMillis()
                val out = clf.analyze(cropBitmap)
                val elapsed = System.currentTimeMillis() - start
                when (mode) {
                    AppMode.IDENTIFY -> onIdentified(out.labels, roi, frameTimeMs, elapsed)
                    AppMode.MY_OBJECTS -> out.embedding?.let {
                        onEmbedding(it, out.labels, cropBitmap, frame, roi, frameTimeMs)
                    }
                    AppMode.READ, AppMode.EARBUDS -> Unit
                }
                Log.d(TAG, "Analyzed ${crop.width}px crop in ${elapsed}ms ($mode): top=${out.labels.firstOrNull()}")
            } catch (e: Exception) {
                Log.e(TAG, "Error analyzing ROI: ${e.javaClass.simpleName}: ${e.message}", e)
            } finally {
                if (cropBitmap != null && cropBitmap !== frame && cropBitmap !== teachPhoto) cropBitmap.recycle()
                isClassifying.set(false)
            }
        }
    }

    private fun onIdentified(labels: List<LabelScore>, roi: BBox, frameTimeMs: Long, elapsed: Long) {
        val smoothed = smoother.update(labels, roi, frameTimeMs)
        if (!_identifyEnabled.value || _mode.value != AppMode.IDENTIFY) return
        latestObject = ObjectResult(smoothed.top, smoothed.candidates, roi, frameTimeMs, elapsed)
    }

    // ---------------- My Objects ----------------

    private fun onEmbedding(
        embedding: FloatArray,
        labels: List<LabelScore>,
        crop: Bitmap,
        frame: Bitmap,
        roi: BBox,
        frameTimeMs: Long
    ) {
        val teach = _teachState.value
        if (teach is TeachState.Collecting) {
            synchronized(teachSamples) {
                teachSamples += embedding
                labels.firstOrNull()?.let { teachVotes[it.label] = (teachVotes[it.label] ?: 0f) + it.score }
                if (teachPhoto == null) teachPhoto = crop
                val count = teachSamples.size
                _teachState.value = if (count >= teach.target) {
                    val suggestion = teachVotes.maxByOrNull { it.value }?.key ?: ""
                    TeachState.Naming(suggestion, teachPhoto)
                } else teach.copy(count = count)
            }
            return
        }
        if (_mode.value != AppMode.MY_OBJECTS) return

        val objects = _savedObjects.value
        val smoothed = personalSmoother.update(ObjectMatcher.scores(embedding, objects), roi, frameTimeMs)
        val match = smoothed.top?.let { top -> objects.firstOrNull { it.id == top.label } }
        latestPersonal = PersonalMatch(match, smoothed.top?.score ?: smoothed.candidates.firstOrNull()?.score ?: 0f,
            roi, frameTimeMs)
        if (match != null) rememberSighting(match, frame, frameTimeMs)
    }

    /** Records when and where a saved object was seen, at most once a minute per object. */
    private fun rememberSighting(obj: SavedObject, frame: Bitmap, nowMs: Long) {
        val last = lastSeenSavedMs[obj.id]
        if (last != null && nowMs - last < 60_000L) return
        lastSeenSavedMs[obj.id] = nowMs
        val scale = 640f / maxOf(frame.width, frame.height)
        val view = if (scale < 1f) {
            Bitmap.createScaledBitmap(frame, (frame.width * scale).toInt(), (frame.height * scale).toInt(), true)
        } else frame.copy(Bitmap.Config.ARGB_8888, false)
        viewModelScope.launch(Dispatchers.IO) {
            _savedObjects.value = store.markSeen(_savedObjects.value, obj.id, nowMs, view)
            view.recycle()
        }
    }

    /** Starts collecting samples of the object the user is pointing at. */
    fun startTeaching() {
        synchronized(teachSamples) {
            teachSamples.clear()
            teachVotes.clear()
            teachPhoto = null
            scheduler.reset()
            _teachState.value = TeachState.Collecting(0, TEACH_SAMPLES)
        }
    }

    fun cancelTeaching() {
        synchronized(teachSamples) {
            teachSamples.clear()
            teachVotes.clear()
            teachPhoto = null
            _teachState.value = TeachState.Idle
        }
    }

    /** Saves the collected samples under [name]; the same name adds to an existing object. */
    fun saveTaughtObject(name: String, note: String) {
        val samples: List<FloatArray>
        val photo: Bitmap?
        synchronized(teachSamples) {
            samples = teachSamples.toList()
            photo = teachPhoto
            teachSamples.clear()
            teachVotes.clear()
            teachPhoto = null
            _teachState.value = TeachState.Idle
        }
        if (samples.isEmpty() || name.isBlank()) return
        viewModelScope.launch(Dispatchers.IO) {
            _savedObjects.value = store.teach(
                _savedObjects.value, name.trim(), note.trim(), samples, photo, System.currentTimeMillis()
            )
            personalSmoother.reset()
        }
    }

    /** Renames an object and/or changes its note; its photos and samples are kept. */
    fun updateObject(id: String, name: String, note: String) {
        val obj = _savedObjects.value.firstOrNull { it.id == id } ?: return
        val newName = name.trim().ifEmpty { obj.name }
        viewModelScope.launch(Dispatchers.IO) {
            _savedObjects.value = store.update(_savedObjects.value, obj.copy(name = newName, note = note.trim()))
        }
    }

    fun deleteObject(id: String) {
        viewModelScope.launch(Dispatchers.IO) {
            _savedObjects.value = store.delete(_savedObjects.value, id)
            personalSmoother.reset()
        }
    }

    fun objectPhotoFile(name: String?) = store.file(name)

    // ---------------- Read ----------------

    /** Reads the whole camera view once, for text that is hard to point at. */
    fun readWholeView() {
        readWholeViewRequested = true
    }

    fun clearReadResult() {
        _readResult.value = null
    }

    private fun readStage(frame: Bitmap, roi: BBox?, frameTimeMs: Long): RoiLabel? {
        val whole = readWholeViewRequested
        val rect = when {
            whole -> PixelRect(0, 0, frame.width, frame.height)
            roi != null -> RoiCropper.cropRect(roi, frame.width, frame.height, padding = 0.3f)
                ?: return RoiLabel("Move closer")
            else -> return null
        }
        if (!whole && readScheduler.shouldClassify(roi!!, frameTimeMs) && !isClassifying.getAndSet(true)) {
            readScheduler.onClassified(roi, frameTimeMs)
            runRead(frame, rect, roi, frameTimeMs, wholeView = false)
        } else if (whole && !isClassifying.getAndSet(true)) {
            readWholeViewRequested = false
            runRead(frame, rect, null, frameTimeMs, wholeView = true)
        }
        if (roi == null) return null
        val latest = _readResult.value?.takeIf {
            !it.wholeView && frameTimeMs - it.computedAtMs < 3000L && latestReadBox?.let { b -> b.iou(roi) >= 0.3f } == true
        } ?: return RoiLabel("Reading…")
        return when {
            latest.codes.isNotEmpty() -> RoiLabel(latest.codes.first().kindLabel, true)
            latest.text.isNotBlank() -> RoiLabel("Text found", true)
            latest.colorName != null -> RoiLabel(latest.colorName, true)
            else -> RoiLabel("Nothing to read")
        }
    }

    private fun runRead(frame: Bitmap, rect: PixelRect, roi: BBox?, frameTimeMs: Long, wholeView: Boolean) {
        viewModelScope.launch(Dispatchers.Default) {
            var crop: Bitmap? = null
            try {
                val reader = textReader ?: TextReader().also { textReader = it }
                crop = Bitmap.createBitmap(frame, rect.left, rect.top, rect.width, rect.height)
                val start = System.currentTimeMillis()
                val (text, codes) = reader.read(crop)

                val small = Bitmap.createScaledBitmap(crop, 32, 32, true)
                val px = IntArray(32 * 32).also { small.getPixels(it, 0, 32, 0, 0, 32, 32) }
                if (small !== crop) small.recycle()
                val rgb = ColorNamer.averageCenter(px, 32, 32)

                val result = ReadResult(
                    text = text.trim(),
                    codes = codes,
                    colorName = if (wholeView) null else ColorNamer.name(rgb),
                    colorRgb = if (wholeView) null else rgb,
                    computedAtMs = frameTimeMs,
                    wholeView = wholeView
                )
                if (_mode.value == AppMode.READ) {
                    // Keep showing the last text or code while the finger only finds colour.
                    val prev = _readResult.value
                    val keepPrev = !wholeView && result.isEmpty && prev != null && !prev.isEmpty &&
                            frameTimeMs - prev.computedAtMs < 5000L
                    if (!keepPrev) _readResult.value = result
                    latestReadBox = roi
                }
                Log.d(TAG, "Read ${rect.width}x${rect.height} in ${System.currentTimeMillis() - start}ms: " +
                        "${text.length} chars, ${codes.size} codes")
            } catch (e: Exception) {
                Log.e(TAG, "Error reading ROI: ${e.javaClass.simpleName}: ${e.message}", e)
            } finally {
                if (crop != null && crop !== frame) crop.recycle()
                isClassifying.set(false)
            }
        }
    }

    // ---------------- Pair Earbuds ----------------

    private fun loadEarbudsCatalog() {
        if (earbudsCatalog != null || earbudsLoading || _earbudsError.value != null) return
        earbudsLoading = true
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val catalog = EarbudsCatalog.load(getApplication<Application>())
                earbudsCatalog = catalog
                _earbudsModelInfo.value = catalog.modelInfo
                _earbudsReady.value = true
            } catch (e: Exception) {
                _earbudsError.value = "The earbuds list is missing from this app build (${e.javaClass.simpleName})."
                Log.e(TAG, "Earbuds catalog failed to load: ${e.message}", e)
            } finally {
                earbudsLoading = false
            }
        }
    }

    /** Supported models, for the "Supported earbuds" list. */
    fun supportedEarbuds(): List<EarbudModel> = earbudsCatalog?.models ?: emptyList()

    fun earbudsThumbnail(model: EarbudModel): Bitmap? = earbudsCatalog?.thumbnail(getApplication(), model)

    private fun maybeRecognizeEarbuds(catalog: EarbudsCatalog, frame: Bitmap, roi: BBox, crop: PixelRect, frameTimeMs: Long) {
        if (!scheduler.shouldClassify(roi, frameTimeMs)) return
        if (isClassifying.getAndSet(true)) return
        scheduler.onClassified(roi, frameTimeMs)

        viewModelScope.launch(Dispatchers.Default) {
            var cropBitmap: Bitmap? = null
            try {
                cropBitmap = Bitmap.createBitmap(frame, crop.left, crop.top, crop.width, crop.height)
                val start = System.currentTimeMillis()
                val rec = catalog.recognizer.recognize(catalog.embed(cropBitmap))
                val models = catalog.models
                // Things that do not look like earbuds feed nothing, so the shown model fades out.
                val scores = if (rec.isEarbuds) rec.candidates.map { LabelScore(models[it.index].id, it.probability) } else emptyList()
                val smoothed = earbudsSmoother.update(scores, roi, frameTimeMs)
                val byId = models.associateBy { it.id }
                if (_mode.value == AppMode.EARBUDS) {
                    val top = smoothed.top?.takeIf { rec.isEarbuds && it.score >= AUTO_PAIR_MIN_PROBABILITY }
                    if (top != null && top.label == autoPairStreakId) autoPairStreak++ else autoPairStreak = if (top != null) 1 else 0
                    autoPairStreakId = top?.label
                    if (top != null && autoPairStreak >= AUTO_PAIR_FRAMES) {
                        maybeAutoPair(top.label, smoothed.candidates.mapNotNull { byId[it.label] })
                    }
                    latestEarbuds = EarbudsMatch(
                        model = smoothed.top?.let { byId[it.label] },
                        probability = smoothed.top?.score ?: 0f,
                        candidates = smoothed.candidates.mapNotNull { byId[it.label] },
                        isEarbuds = rec.isEarbuds,
                        box = roi,
                        computedAtMs = frameTimeMs
                    )
                }
                Log.d(TAG, "Earbuds ${crop.width}px in ${System.currentTimeMillis() - start}ms: " +
                        "sim=${"%.3f".format(rec.similarity)} top=${rec.candidates.firstOrNull()?.let { models[it.index].id }} " +
                        "shown=${smoothed.top}")
            } catch (e: Exception) {
                Log.e(TAG, "Error recognizing earbuds: ${e.javaClass.simpleName}: ${e.message}", e)
            } finally {
                if (cropBitmap != null && cropBitmap !== frame) cropBitmap.recycle()
                isClassifying.set(false)
            }
        }
    }

    /** Starts pairing without a tap, once per model per visit, when nothing else is going on. */
    private fun maybeAutoPair(modelId: String, candidates: List<EarbudModel>) {
        viewModelScope.launch(Dispatchers.Main) {
            if (pairer.state.value != com.example.roidetection.earbuds.PairState.Idle) return@launch
            if (!autoPairTried.add(modelId) || candidates.isEmpty()) return@launch
            Log.i(TAG, "Auto-pairing: ${candidates.map { it.id }}")
            pairer.start(candidates)
        }
    }

    // ---------------- Mode switching ----------------

    fun setMode(mode: AppMode) {
        if (_mode.value == mode) return
        Log.i(TAG, "setMode($mode)")
        _mode.value = mode
        latestObject = null
        latestPersonal = null
        smoother.reset()
        personalSmoother.reset()
        scheduler.reset()
        readScheduler.reset()
        readWholeViewRequested = false
        latestEarbuds = null
        earbudsSmoother.reset()
        pairer.reset()
        autoPairTried.clear()
        autoPairStreak = 0
        autoPairStreakId = null
        if (mode == AppMode.EARBUDS) loadEarbudsCatalog()
        _readResult.value = null
        if (_teachState.value != TeachState.Idle) cancelTeaching()
    }

    fun setShowDetails(show: Boolean) {
        _showDetails.value = show
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
        classifier?.close()
        classifier = null
        textReader?.close()
        textReader = null
        earbudsCatalog?.close()
        earbudsCatalog = null
        pairer.reset()
        _result.value = CombinedResult()
    }
}
