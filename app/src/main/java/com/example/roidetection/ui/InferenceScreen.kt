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
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
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
import com.example.roidetection.CombinedResult
import com.example.roidetection.InferenceViewModel
import com.example.roidetection.ModelInfo
import java.util.concurrent.Executors
import android.graphics.Bitmap
import android.content.ContentValues
import android.provider.MediaStore
import android.widget.Toast
import androidx.compose.material3.FloatingActionButton
import kotlinx.coroutines.launch

private const val CAMERA_PERMISSION_REQUEST_CODE = 1001

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InferenceScreen(
    viewModel: InferenceViewModel,
    onBack: () -> Unit
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
        topBar = {
            TopAppBar(
                title = { Text("ROI Detection Live") },
                navigationIcon = {
                    FilledIconButton(
                        onClick = {
                            viewModel.setRunning(false)
                            onBack()
                        }
                    ) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.Black.copy(alpha = 0.6f),
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White
                ),
                modifier = Modifier.statusBarsPadding()
            )
        },
        floatingActionButton = {
            if (hasCameraPermission && isModelReady) {
                FloatingActionButton(
                    onClick = {
                        coroutineScope.launch {
                            val currentResult = viewModel.result.value
                            val compositeBitmap = captureWithOverlays(currentResult)
                            if (compositeBitmap != null) {
                                val saved = saveBitmapToGallery(context, compositeBitmap)
                                compositeBitmap.recycle()
                                if (saved) {
                                    Toast.makeText(context, "Screenshot saved to Gallery", Toast.LENGTH_SHORT).show()
                                } else {
                                    Toast.makeText(context, "Failed to save screenshot", Toast.LENGTH_SHORT).show()
                                }
                            } else {
                                Toast.makeText(context, "No camera frame available", Toast.LENGTH_SHORT).show()
                            }
                        }
                    },
                    containerColor = Color.White.copy(alpha = 0.8f),
                    contentColor = Color.Black,
                    modifier = Modifier.padding(bottom = 64.dp)
                ) {
                    Text("Capture", modifier = Modifier.padding(horizontal = 16.dp), fontWeight = FontWeight.Bold)
                }
            }
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            if (!hasCameraPermission) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Camera permission is required for inference.",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.error
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Button(
                        onClick = {
                            activity?.let {
                                ActivityCompat.requestPermissions(
                                    it,
                                    arrayOf(Manifest.permission.CAMERA),
                                    CAMERA_PERMISSION_REQUEST_CODE
                                )
                            }
                        }
                    ) {
                        Text("Grant Camera Permission")
                    }
                }
            } else if (modelError != null) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = "Model Loading Failed",
                        style = MaterialTheme.typography.titleLarge,
                        color = MaterialTheme.colorScheme.error,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = modelError ?: "Unknown error",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(24.dp))
                    Button(
                        onClick = {
                            viewModel.retryModelInitialization()
                        }
                    ) {
                        Text("Retry")
                    }
                }
            } else if (!isModelReady) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(48.dp),
                        color = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Loading model...",
                        style = MaterialTheme.typography.bodyLarge
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "SpatialNet v4",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                    )
                }
            } else {
                CameraPreviewWithOverlay(
                    context = context,
                    viewModel = viewModel,
                    result = result
                )
            }

            // Model identity badge, top-left
            ModelBadge(
                info = modelInfo ?: bundledModelInfo,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(12.dp)
            )

            // Stats overlay at bottom
            if (hasCameraPermission && isModelReady) {
                Surface(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .padding(16.dp),
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
                            text = "${result.totalInferenceTimeMs}ms",
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
            }
        }
    }
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
    result: CombinedResult
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val executor = remember { Executors.newSingleThreadExecutor() }

    val roiResult = result.roiResult
    val isPointing = result.isPointing
    val frame = result.frame

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
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

            if (isPointing) {
                // POINTING MODE: Green ROI, Blue hand, Red fingertip, Yellow arrow

                // Draw hand bounding box (blue dashed)
                roiResult.handBBox?.let { bbox ->
                    val corners = bbox.toCorners().clamp()
                    drawDetectionBox(
                        corners = corners,
                        canvasWidth = imageWidth,
                        canvasHeight = imageHeight,
                        offsetX = offsetX,
                        offsetY = offsetY,
                        color = Color(0xFF2196F3),
                        label = "Hand",
                        strokeWidth = 3f
                    )
                }

                // Draw ROI bounding box (green)
                roiResult.roiBBox?.let { bbox ->
                    val corners = bbox.toCorners().clamp()
                    drawDetectionBox(
                        corners = corners,
                        canvasWidth = imageWidth,
                        canvasHeight = imageHeight,
                        offsetX = offsetX,
                        offsetY = offsetY,
                        color = Color(0xFF4CAF50),
                        label = "ROI",
                        strokeWidth = 3f
                    )
                }

                // Draw fingertip point (red)
                roiResult.fingertip?.let { ft ->
                    val x = offsetX + ft.x * imageWidth
                    val y = offsetY + ft.y * imageHeight

                    drawCircle(
                        color = Color(0xFFF44336),
                        radius = 12f,
                        center = Offset(x, y),
                        style = Stroke(width = 3f)
                    )
                    drawCircle(
                        color = Color(0xFFF44336),
                        radius = 5f,
                        center = Offset(x, y)
                    )

                    drawContext.canvas.nativeCanvas.drawText(
                        "Fingertip",
                        x + 16f,
                        y - 8f,
                        android.graphics.Paint().apply {
                            color = android.graphics.Color.RED
                            textSize = 28f
                            isAntiAlias = true
                            typeface = android.graphics.Typeface.DEFAULT_BOLD
                        }
                    )
                }

                // Draw pointing direction arrow (yellow)
                roiResult.fingertip?.let { ft ->
                    roiResult.pointingDir?.let { dir ->
                        val startX = offsetX + ft.x * imageWidth
                        val startY = offsetY + ft.y * imageHeight
                        val arrowLength = 80f
                        val angle = dir.angleRadians()
                        val endX = startX + arrowLength * kotlin.math.cos(angle.toDouble()).toFloat()
                        val endY = startY + arrowLength * kotlin.math.sin(angle.toDouble()).toFloat()

                        drawLine(
                            color = Color(0xFFFFEB3B),
                            start = Offset(startX, startY),
                            end = Offset(endX, endY),
                            strokeWidth = 4f
                        )

                        val arrowHeadLen = 16f
                        val arrowAngle = Math.toRadians(30.0)
                        val angle1 = angle + Math.PI + arrowAngle
                        val angle2 = angle + Math.PI - arrowAngle

                        val path = Path().apply {
                            moveTo(endX, endY)
                            lineTo(
                                endX + arrowHeadLen * kotlin.math.cos(angle1).toFloat(),
                                endY + arrowHeadLen * kotlin.math.sin(angle1).toFloat()
                            )
                            moveTo(endX, endY)
                            lineTo(
                                endX + arrowHeadLen * kotlin.math.cos(angle2).toFloat(),
                                endY + arrowHeadLen * kotlin.math.sin(angle2).toFloat()
                            )
                        }
                        drawPath(
                            path = path,
                            color = Color(0xFFFFEB3B),
                            style = Stroke(width = 3f)
                        )
                    }
                }
            } else {
                // NON-POINTING MODE: Red hand box, "Non-Pointing" label
                roiResult.handBBox?.let { bbox ->
                    val corners = bbox.toCorners().clamp()
                    drawDetectionBox(
                        corners = corners,
                        canvasWidth = imageWidth,
                        canvasHeight = imageHeight,
                        offsetX = offsetX,
                        offsetY = offsetY,
                        color = Color(0xFFF44336),
                        label = "Non-Pointing",
                        strokeWidth = 3f
                    )
                }
            }
        }
    }

    LaunchedEffect(cameraProviderFuture) {
        Log.i("InferenceScreen", "Binding camera use cases...")
        try {
            val cameraProvider = cameraProviderFuture.get()
            Log.i("InferenceScreen", "CameraProvider obtained")

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
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

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawDetectionBox(
    corners: com.example.roidetection.Corners,
    canvasWidth: Float,
    canvasHeight: Float,
    offsetX: Float,
    offsetY: Float,
    color: Color,
    label: String,
    strokeWidth: Float
) {
    val x1 = offsetX + corners.x1 * canvasWidth
    val y1 = offsetY + corners.y1 * canvasHeight
    val x2 = offsetX + corners.x2 * canvasWidth
    val y2 = offsetY + corners.y2 * canvasHeight
    val boxWidth = x2 - x1
    val boxHeight = y2 - y1

    drawRect(
        color = color,
        topLeft = Offset(x1, y1),
        size = Size(boxWidth, boxHeight),
        style = Stroke(width = strokeWidth)
    )

    val cornerLen = minOf(20f, boxWidth * 0.15f, boxHeight * 0.15f)

    drawLine(color, Offset(x1, y1), Offset(x1 + cornerLen, y1), strokeWidth + 1f)
    drawLine(color, Offset(x1, y1), Offset(x1, y1 + cornerLen), strokeWidth + 1f)
    drawLine(color, Offset(x2, y1), Offset(x2 - cornerLen, y1), strokeWidth + 1f)
    drawLine(color, Offset(x2, y1), Offset(x2, y1 + cornerLen), strokeWidth + 1f)
    drawLine(color, Offset(x1, y2), Offset(x1 + cornerLen, y2), strokeWidth + 1f)
    drawLine(color, Offset(x1, y2), Offset(x1, y2 - cornerLen), strokeWidth + 1f)
    drawLine(color, Offset(x2, y2), Offset(x2 - cornerLen, y2), strokeWidth + 1f)
    drawLine(color, Offset(x2, y2), Offset(x2, y2 - cornerLen), strokeWidth + 1f)

    drawContext.canvas.nativeCanvas.drawText(
        label,
        x1 + 4f,
        y1 - 8f,
        android.graphics.Paint().apply {
            this.color = when (color) {
                Color(0xFF2196F3) -> android.graphics.Color.BLUE
                Color(0xFF4CAF50) -> android.graphics.Color.GREEN
                Color(0xFFF44336) -> android.graphics.Color.RED
                else -> android.graphics.Color.WHITE
            }
            textSize = 28f
            isAntiAlias = true
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }
    )
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

    val roiResult = result.roiResult
    val isPointing = result.isPointing

    if (isPointing) {
        // Draw hand bounding box (blue)
        roiResult.handBBox?.let { bbox ->
            val corners = bbox.toCorners().clamp()
            drawBoxOnCanvas(canvas, corners, w, h, android.graphics.Color.BLUE, "Hand", 4f)
        }

        // Draw ROI bounding box (green)
        roiResult.roiBBox?.let { bbox ->
            val corners = bbox.toCorners().clamp()
            drawBoxOnCanvas(canvas, corners, w, h, android.graphics.Color.GREEN, "ROI", 4f)
        }

        // Draw fingertip point (red)
        roiResult.fingertip?.let { ft ->
            val x = ft.x * w
            val y = ft.y * h
            val paint = android.graphics.Paint().apply {
                color = android.graphics.Color.RED
                style = android.graphics.Paint.Style.STROKE
                strokeWidth = 4f
                isAntiAlias = true
            }
            canvas.drawCircle(x, y, 14f, paint)
            paint.style = android.graphics.Paint.Style.FILL
            canvas.drawCircle(x, y, 6f, paint)

            val textPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.RED
                textSize = 32f
                isAntiAlias = true
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            }
            canvas.drawText("Fingertip", x + 18f, y - 10f, textPaint)
        }

        // Draw pointing direction arrow (yellow)
        roiResult.fingertip?.let { ft ->
            roiResult.pointingDir?.let { dir ->
                val startX = ft.x * w
                val startY = ft.y * h
                val arrowLength = 90f
                val angle = dir.angleRadians()
                val endX = startX + arrowLength * kotlin.math.cos(angle.toDouble()).toFloat()
                val endY = startY + arrowLength * kotlin.math.sin(angle.toDouble()).toFloat()

                val arrowPaint = android.graphics.Paint().apply {
                    color = android.graphics.Color.YELLOW
                    strokeWidth = 5f
                    style = android.graphics.Paint.Style.STROKE
                    isAntiAlias = true
                }
                canvas.drawLine(startX, startY, endX, endY, arrowPaint)

                // Arrowhead
                val arrowHeadLen = 18f
                val arrowAngle = Math.toRadians(30.0)
                val a1 = angle + Math.PI + arrowAngle
                val a2 = angle + Math.PI - arrowAngle
                canvas.drawLine(
                    endX, endY,
                    endX + arrowHeadLen * kotlin.math.cos(a1).toFloat(),
                    endY + arrowHeadLen * kotlin.math.sin(a1).toFloat(),
                    arrowPaint
                )
                canvas.drawLine(
                    endX, endY,
                    endX + arrowHeadLen * kotlin.math.cos(a2).toFloat(),
                    endY + arrowHeadLen * kotlin.math.sin(a2).toFloat(),
                    arrowPaint
                )
            }
        }
    } else {
        // NON-POINTING: Red hand box
        roiResult.handBBox?.let { bbox ->
            val corners = bbox.toCorners().clamp()
            drawBoxOnCanvas(canvas, corners, w, h, android.graphics.Color.RED, "Non-Pointing", 4f)
        }
    }

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
    val statsText = "${result.totalInferenceTimeMs}ms | $catLabel | ${"%.0f".format(activeConf * 100)}%"
    canvas.drawText(statsText, 16f, h - 14f, statsPaint)

    return outputBitmap
}

/**
 * Draws a labeled bounding box with corner accents on an Android Canvas.
 */
private fun drawBoxOnCanvas(
    canvas: android.graphics.Canvas,
    corners: com.example.roidetection.Corners,
    canvasW: Float,
    canvasH: Float,
    color: Int,
    label: String,
    strokeW: Float
) {
    val x1 = corners.x1 * canvasW
    val y1 = corners.y1 * canvasH
    val x2 = corners.x2 * canvasW
    val y2 = corners.y2 * canvasH

    val boxPaint = android.graphics.Paint().apply {
        this.color = color
        style = android.graphics.Paint.Style.STROKE
        this.strokeWidth = strokeW
        isAntiAlias = true
    }
    canvas.drawRect(x1, y1, x2, y2, boxPaint)

    // Corner accents
    val boxWidth = x2 - x1
    val boxHeight = y2 - y1
    val cornerLen = minOf(22f, boxWidth * 0.15f, boxHeight * 0.15f)
    val cornerPaint = android.graphics.Paint().apply {
        this.color = color
        style = android.graphics.Paint.Style.STROKE
        this.strokeWidth = strokeW + 2f
        isAntiAlias = true
    }
    canvas.drawLine(x1, y1, x1 + cornerLen, y1, cornerPaint)
    canvas.drawLine(x1, y1, x1, y1 + cornerLen, cornerPaint)
    canvas.drawLine(x2, y1, x2 - cornerLen, y1, cornerPaint)
    canvas.drawLine(x2, y1, x2, y1 + cornerLen, cornerPaint)
    canvas.drawLine(x1, y2, x1 + cornerLen, y2, cornerPaint)
    canvas.drawLine(x1, y2, x1, y2 - cornerLen, cornerPaint)
    canvas.drawLine(x2, y2, x2 - cornerLen, y2, cornerPaint)
    canvas.drawLine(x2, y2, x2, y2 - cornerLen, cornerPaint)

    // Label
    val textPaint = android.graphics.Paint().apply {
        this.color = color
        textSize = 32f
        isAntiAlias = true
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    canvas.drawText(label, x1 + 6f, y1 - 10f, textPaint)
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
