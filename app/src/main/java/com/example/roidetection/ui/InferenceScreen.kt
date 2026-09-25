package com.example.roidetection.ui

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import android.util.Log
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.compose.foundation.layout.navigationBarsPadding
import com.example.roidetection.AppMode
import com.example.roidetection.CombinedResult
import com.example.roidetection.InferenceViewModel
import com.example.roidetection.ModelInfo
import com.example.roidetection.TeachState
import com.example.roidetection.ui.theme.Beam
import com.example.roidetection.ui.theme.Mist
import java.util.concurrent.Executors
import android.graphics.Bitmap
import android.content.ContentValues
import android.provider.MediaStore
import android.widget.Toast
import kotlinx.coroutines.launch

private const val CAMERA_PERMISSION_REQUEST_CODE = 1001

/** Behind the camera picture and its top bar: night-navy, so the glass panels belong to it. */
private val CameraBackdrop = Color(0xFF0A1322)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InferenceScreen(
    viewModel: InferenceViewModel,
    mode: AppMode,
    onBack: () -> Unit,
    onOpenObjectList: () -> Unit = {}
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val activity = context as? Activity
    val coroutineScope = rememberCoroutineScope()

    val result by viewModel.result.collectAsState()
    val isModelReady by viewModel.isModelReady.collectAsState()
    val modelError by viewModel.modelError.collectAsState()
    val fps by viewModel.fps.collectAsState()
    val modelInfo by viewModel.modelInfo.collectAsState()
    // Falls back to the bundled asset so the badge is readable while the
    // ONNX session is still starting up.
    val bundledModelInfo = remember { ModelInfo.read(context) }
    val identifyEnabled by viewModel.identifyEnabled.collectAsState()
    val classifierInfo by viewModel.classifierInfo.collectAsState()
    val classifierError by viewModel.classifierError.collectAsState()
    val bundledClassifierInfo = remember { ModelInfo.read(context, ModelInfo.CLASSIFIER_ASSET) }
    val showDetails by viewModel.showDetails.collectAsState()
    val teachState by viewModel.teachState.collectAsState()
    val savedObjects by viewModel.savedObjects.collectAsState()
    val readResult by viewModel.readResult.collectAsState()
    var autoSpeak by remember { mutableStateOf(false) }
    val speaker = rememberSpeaker()

    LaunchedEffect(mode) { viewModel.setMode(mode) }

    val savePhoto: () -> Unit = {
        coroutineScope.launch {
            val compositeBitmap = captureWithOverlays(viewModel.result.value)
            if (compositeBitmap != null) {
                val saved = saveBitmapToGallery(context, compositeBitmap)
                compositeBitmap.recycle()
                Toast.makeText(
                    context,
                    if (saved) "Photo saved to Gallery" else "Could not save the photo",
                    Toast.LENGTH_SHORT
                ).show()
            } else {
                Toast.makeText(context, "No camera picture yet", Toast.LENGTH_SHORT).show()
            }
        }
    }

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED
        )
    }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                hasCameraPermission = ContextCompat.checkSelfPermission(
                    context, Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    LaunchedEffect(Unit) {
        viewModel.initializeModel()
    }

    LaunchedEffect(hasCameraPermission, isModelReady) {
        if (hasCameraPermission && isModelReady) {
            viewModel.setRunning(true)
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            viewModel.setRunning(false)
        }
    }

    Scaffold(
        containerColor = CameraBackdrop,
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        style = MaterialTheme.typography.titleLarge,
                        text = when (mode) {
                            AppMode.IDENTIFY -> "Identify"
                            AppMode.MY_OBJECTS -> "My Objects"
                            AppMode.READ -> "Read"
                            AppMode.EARBUDS -> "Pair Earbuds"
                        }
                    )
                },
                navigationIcon = {
                    FilledIconButton(
                        onClick = {
                            viewModel.setRunning(false)
                            onBack()
                        },
                        colors = IconButtonDefaults.filledIconButtonColors(
                            containerColor = Color(0x26FFFFFF),
                            contentColor = Color.White
                        ),
                        modifier = Modifier.padding(start = 8.dp)
                    ) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                actions = {
                    if (mode == AppMode.IDENTIFY && showDetails) Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(end = 8.dp)
                    ) {
                        Text(
                            text = "Identify",
                            color = Color.White,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium
                        )
                        Spacer(modifier = Modifier.size(6.dp))
                        Switch(
                            checked = identifyEnabled && classifierError == null,
                            onCheckedChange = { viewModel.setIdentifyEnabled(it) },
                            enabled = classifierError == null
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = CameraBackdrop,
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White
                ),
                modifier = Modifier.statusBarsPadding()
            )
        },
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            if (!hasCameraPermission) {
                CenteredMessage(
                    title = "The camera is off",
                    body = "This app needs the camera to see what you point at. Pictures stay on this phone.",
                    action = "Turn on camera",
                    onAction = {
                        activity?.let {
                            ActivityCompat.requestPermissions(it, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
                        }
                    }
                )
            } else if (modelError != null) {
                CenteredMessage(
                    title = "The app could not start",
                    body = modelError ?: "A model file could not be loaded.",
                    action = "Try again",
                    onAction = { viewModel.retryModelInitialization() }
                )
            } else if (!isModelReady) {
                Column(
                    modifier = Modifier.fillMaxSize().padding(32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(44.dp), color = Beam, strokeWidth = 3.dp)
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("Getting ready…", style = MaterialTheme.typography.bodyLarge, color = Color.White)
                }
            } else {
                CameraPreviewWithOverlay(
                    context = context,
                    viewModel = viewModel,
                    result = result,
                    // Sharper frames for small text; the other modes keep the faster default.
                    highResolution = mode == AppMode.READ || mode == AppMode.EARBUDS
                )
            }

            // Instruction, top centre
            if (hasCameraPermission && isModelReady) {
                GuidanceBanner(
                    guidanceText(result, mode),
                    modifier = Modifier.align(Alignment.TopCenter).padding(top = 12.dp, start = 16.dp, end = 16.dp)
                )
            }
            // Model identity badges (technical details only), below the instruction
            if (showDetails) Column(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(start = 12.dp, top = 64.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                ModelBadge(info = modelInfo ?: bundledModelInfo)
                if (classifierError != null) {
                    ModelBadge(
                        info = null,
                        unavailableName = "object ID unavailable",
                        unavailableDetail = classifierError ?: "",
                        nameColor = Color(0xFFFF8A80)
                    )
                } else {
                    ModelBadge(info = classifierInfo ?: bundledClassifierInfo)
                }
            }

            // Stats (details only) and the mode's answer panel, bottom
            if (hasCameraPermission && isModelReady) Column(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .navigationBarsPadding()
            ) {
                if (showDetails) Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    color = Color.Black.copy(alpha = 0.7f),
                    shape = MaterialTheme.shapes.small
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        val catLabel = getCategoryLabel(result)
                        val catColor = getCategoryColor(result)
                        Text(
                            text = "FPS: ${"%.1f".format(fps)}",
                            color = Color.White,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = inferenceTimeText(result),
                            color = Color.White,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = catLabel,
                            color = catColor,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold
                        )
                        val activeConf = when {
                            !result.roiResult.isHandDetected -> 0f
                            !result.roiResult.isPointing -> result.roiResult.handConf
                            !result.roiResult.isRoiDetected -> result.roiResult.pointConf
                            else -> result.roiResult.roiConf
                        }
                        Text(
                            text = "${"%.0f".format(activeConf * 100)}%",
                            color = Color.White,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
                when (mode) {
                    AppMode.IDENTIFY -> IdentifyPanel(result, speaker, onSavePhoto = savePhoto)
                    AppMode.MY_OBJECTS -> MyObjectsPanel(
                        result = result,
                        teachState = teachState,
                        savedCount = savedObjects.size,
                        speaker = speaker,
                        onTeach = viewModel::startTeaching,
                        onCancelTeach = viewModel::cancelTeaching,
                        onOpenList = onOpenObjectList
                    )
                    AppMode.READ -> ReadPanel(
                        readResult = readResult,
                        speaker = speaker,
                        autoSpeak = autoSpeak,
                        onAutoSpeakChange = { autoSpeak = it },
                        onReadWholeView = viewModel::readWholeView,
                        onClear = viewModel::clearReadResult
                    )
                    AppMode.EARBUDS -> EarbudsPanel(result, viewModel, speaker)
                }
            }
        }
    }

    (teachState as? TeachState.Naming)?.let { naming ->
        TeachNameDialog(
            state = naming,
            existingNames = savedObjects.map { it.name },
            onSave = viewModel::saveTaughtObject,
            onCancel = viewModel::cancelTeaching
        )
    }
}

/** Title, explanation and one amber action, centred on the dark camera backdrop. */
@Composable
private fun CenteredMessage(title: String, body: String, action: String, onAction: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(title, style = MaterialTheme.typography.headlineSmall, color = Color.White, textAlign = TextAlign.Center)
        Spacer(modifier = Modifier.height(10.dp))
        Text(body, style = MaterialTheme.typography.bodyLarge, color = Mist, textAlign = TextAlign.Center)
        Spacer(modifier = Modifier.height(24.dp))
        BeamButton(action, onClick = onAction)
    }
}

/** SpatialNet time, plus the classifier time when a label is shown, e.g. "70ms + 18ms". */
private fun inferenceTimeText(result: CombinedResult): String {
    val classifierMs = result.objectResult?.inferenceTimeMs
    return if (classifierMs != null) "${result.totalInferenceTimeMs}ms + ${classifierMs}ms"
    else "${result.totalInferenceTimeMs}ms"
}

private fun getCategoryLabel(result: CombinedResult): String {
    val roi = result.roiResult
    return when {
        !roi.isHandDetected -> "NO HAND"
        !roi.isPointing -> "NON-POINTING"
        !roi.isRoiDetected -> "POINTING-EMPTY"
        else -> "POINTING"
    }
}

private fun getCategoryColor(result: CombinedResult): Color {
    val roi = result.roiResult
    return when {
        !roi.isHandDetected -> Color(0xFF9E9E9E) // Gray
        !roi.isPointing -> Color(0xFFFF9800)     // Orange
        !roi.isRoiDetected -> Color(0xFF9C27B0)  // Purple
        else -> Color(0xFF4CAF50)                // Green
    }
}

private fun getCategoryColorAndroid(result: CombinedResult): Int {
    val roi = result.roiResult
    return when {
        !roi.isHandDetected -> android.graphics.Color.GRAY
        !roi.isPointing -> android.graphics.Color.rgb(255, 152, 0)     // Orange
        !roi.isRoiDetected -> android.graphics.Color.rgb(156, 39, 176) // Purple
        else -> android.graphics.Color.GREEN
    }
}

@Composable
private fun CameraPreviewWithOverlay(
    context: Context,
    viewModel: InferenceViewModel,
    result: CombinedResult,
    highResolution: Boolean = false
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val executor = remember { Executors.newSingleThreadExecutor() }

    val frame = result.frame

    Box(modifier = Modifier.fillMaxSize().background(CameraBackdrop)) {
        frame?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "Live camera analysis frame",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize()
            )
        }

        Canvas(
            modifier = Modifier.fillMaxSize()
        ) {
            val canvasWidth = size.width
            val canvasHeight = size.height

            if (canvasWidth <= 0f || canvasHeight <= 0f || frame == null) return@Canvas

            // ContentScale.Fit: the full frame and its annotations share one transform.
            val scale = minOf(canvasWidth / frame.width, canvasHeight / frame.height)
            val imageWidth = frame.width * scale
            val imageHeight = frame.height * scale
            val offsetX = (canvasWidth - imageWidth) / 2f
            val offsetY = (canvasHeight - imageHeight) / 2f

            drawBeamOverlay(drawContext.canvas.nativeCanvas, result, offsetX, offsetY, imageWidth, imageHeight)
        }
    }

    LaunchedEffect(cameraProviderFuture, highResolution) {
        Log.i("InferenceScreen", "Binding camera use cases...")
        try {
            val cameraProvider = cameraProviderFuture.get()
            Log.i("InferenceScreen", "CameraProvider obtained")

            val builder = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            if (highResolution) {
                builder.setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                android.util.Size(1280, 720),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                            )
                        )
                        .build()
                )
            }
            val imageAnalysis = builder.build()
                .also {
                    it.setAnalyzer(executor) { imageProxy ->
                        viewModel.processFrame(imageProxy)
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                imageAnalysis
            )
            Log.i("InferenceScreen", "Camera bound successfully")
        } catch (e: Exception) {
            Log.e("InferenceScreen", "Failed to bind camera: ${e.javaClass.simpleName}: ${e.message}", e)
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            Log.i("InferenceScreen", "Disposing camera")
            try {
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()
            } catch (_: Exception) {}
            executor.shutdown()
        }
    }
}

/**
 * Captures the latest camera frame and draws detection overlays on it.
 * This avoids the PixelCopy issue where SurfaceView content (camera preview)
 * isn't captured, resulting in black backgrounds.
 */
private fun captureWithOverlays(
    result: CombinedResult
): Bitmap? {
    val frameBitmap = result.frame ?: return null

    // Create a mutable copy to draw overlays on
    val outputBitmap = frameBitmap.copy(Bitmap.Config.ARGB_8888, true)

    val canvas = android.graphics.Canvas(outputBitmap)
    val w = outputBitmap.width.toFloat()
    val h = outputBitmap.height.toFloat()

    drawBeamOverlay(canvas, result, 0f, 0f, w, h)

    // Draw stats bar at the bottom
    val barPaint = android.graphics.Paint().apply {
        color = android.graphics.Color.argb(180, 0, 0, 0)
        style = android.graphics.Paint.Style.FILL
    }
    val barHeight = 50f
    canvas.drawRect(0f, h - barHeight, w, h, barPaint)

    val statsPaint = android.graphics.Paint().apply {
        color = android.graphics.Color.WHITE
        textSize = 28f
        isAntiAlias = true
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    val catLabel = getCategoryLabel(result)
    val activeConf = when {
        !result.roiResult.isHandDetected -> 0f
        !result.roiResult.isPointing -> result.roiResult.handConf
        !result.roiResult.isRoiDetected -> result.roiResult.pointConf
        else -> result.roiResult.roiConf
    }
    val objectText = objectLabelText(result)?.let { " | $it" } ?: ""
    val statsText = "${inferenceTimeText(result)} | $catLabel | ${"%.0f".format(activeConf * 100)}%$objectText"
    canvas.drawText(statsText, 16f, h - 14f, statsPaint)

    return outputBitmap
}

private fun saveBitmapToGallery(context: Context, bitmap: Bitmap): Boolean {
    val filename = "ROI_Screenshot_${System.currentTimeMillis()}.jpg"
    val contentValues = ContentValues().apply {
        put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
        put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/ROIDetection")
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
    }

    val resolver = context.contentResolver
    val imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues) ?: return false

    try {
        resolver.openOutputStream(imageUri).use { outputStream ->
            if (outputStream != null) {
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, outputStream)
            }
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            contentValues.clear()
            contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
            resolver.update(imageUri, contentValues, null, null)
        }
        return true
    } catch (e: Exception) {
        resolver.delete(imageUri, null, null)
        return false
    }
}
