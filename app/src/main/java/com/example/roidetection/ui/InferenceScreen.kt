package com.example.roidetection.ui

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.view.ViewGroup
import android.widget.LinearLayout
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
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
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import android.util.Log
import com.example.roidetection.CombinedResult
import com.example.roidetection.InferenceViewModel
import java.util.concurrent.Executors

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

    val result by viewModel.result.collectAsState()
    val isModelReady by viewModel.isModelReady.collectAsState()
    val modelError by viewModel.modelError.collectAsState()
    val fps by viewModel.fps.collectAsState()

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
                        text = "Loading models...",
                        style = MaterialTheme.typography.bodyLarge
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "SpatialNet + YOLO Gesture",
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
                            text = if (result.isPointing) "POINTING" else "NON-POINT",
                            color = if (result.isPointing) Color(0xFF4CAF50) else Color(0xFFF44336),
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "${"%.0f".format(result.gestureResult.confidence * 100)}%",
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

@Composable
private fun CameraPreviewWithOverlay(
    context: Context,
    viewModel: InferenceViewModel,
    result: CombinedResult
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val executor = remember { Executors.newSingleThreadExecutor() }

    val previewView = remember {
        PreviewView(context).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    val roiResult = result.roiResult
    val isPointing = result.isPointing

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        Canvas(
            modifier = Modifier.fillMaxSize()
        ) {
            val canvasWidth = size.width
            val canvasHeight = size.height

            if (canvasWidth <= 0f || canvasHeight <= 0f) return@Canvas

            if (isPointing) {
                // POINTING MODE: Green ROI, Blue hand, Red fingertip, Yellow arrow

                // Draw hand bounding box (blue dashed)
                roiResult.handBBox?.let { bbox ->
                    val corners = bbox.toCorners().clamp()
                    drawDetectionBox(
                        corners = corners,
                        canvasWidth = canvasWidth,
                        canvasHeight = canvasHeight,
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
                        canvasWidth = canvasWidth,
                        canvasHeight = canvasHeight,
                        color = Color(0xFF4CAF50),
                        label = "ROI",
                        strokeWidth = 3f
                    )
                }

                // Draw fingertip point (red)
                roiResult.fingertip?.let { ft ->
                    val x = ft.x * canvasWidth
                    val y = ft.y * canvasHeight

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
                        val startX = ft.x * canvasWidth
                        val startY = ft.y * canvasHeight
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
                        canvasWidth = canvasWidth,
                        canvasHeight = canvasHeight,
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

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

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
                preview,
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
    color: Color,
    label: String,
    strokeWidth: Float
) {
    val x1 = corners.x1 * canvasWidth
    val y1 = corners.y1 * canvasHeight
    val x2 = corners.x2 * canvasWidth
    val y2 = corners.y2 * canvasHeight
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
