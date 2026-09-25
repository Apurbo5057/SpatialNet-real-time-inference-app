package com.example.roidetection.ui

import android.provider.Settings
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.example.roidetection.AppMode
import com.example.roidetection.ModelInfo
import com.example.roidetection.ui.theme.Beam
import com.example.roidetection.ui.theme.Ink
import com.example.roidetection.ui.theme.TintEarbuds
import com.example.roidetection.ui.theme.TintIdentify
import com.example.roidetection.ui.theme.TintObjects
import com.example.roidetection.ui.theme.TintRead

private data class ModeRow(
    val mode: AppMode,
    val icon: ImageVector,
    val tint: Color,
    val title: String,
    val description: String
)

private val MODES = listOf(
    ModeRow(AppMode.IDENTIFY, AppIcons.Target, TintIdentify, "Identify", "Find out what something is."),
    ModeRow(AppMode.MY_OBJECTS, AppIcons.Tag, TintObjects, "My Objects", "Teach the app your things, add notes, and see where you last saw them."),
    ModeRow(AppMode.READ, AppIcons.Page, TintRead, "Read", "Hear text read aloud, scan QR codes, or name a colour."),
    ModeRow(AppMode.EARBUDS, AppIcons.Headphones, TintEarbuds, "Pair Earbuds", "See what your earbuds are and connect them to this phone.")
)

@Composable
fun HomeScreen(
    onOpenMode: (AppMode) -> Unit,
    showDetails: Boolean,
    onShowDetailsChange: (Boolean) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .navigationBarsPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp)
    ) {
        BeamHeader(modifier = Modifier.fillMaxWidth().height(88.dp))
        Spacer(Modifier.height(12.dp))
        Text("Point & Know", style = MaterialTheme.typography.displaySmall, color = MaterialTheme.colorScheme.onBackground)
        Spacer(Modifier.height(6.dp))
        Text(
            "Point at something with one finger. The app tells you what it is.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(24.dp))

        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            MODES.forEach { ModeRowItem(it) { onOpenMode(it.mode) } }
        }

        Spacer(Modifier.height(24.dp))
        Row(verticalAlignment = Alignment.Top) {
            Icon(Icons.Filled.Lock, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(18.dp).padding(top = 2.dp))
            Spacer(Modifier.size(10.dp))
            Text(
                "Everything runs on this phone. Nothing goes online, and your camera pictures stay here. " +
                        "Pair Earbuds uses Bluetooth only to connect your earbuds.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Spacer(Modifier.height(12.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Show technical details", style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
            Switch(checked = showDetails, onCheckedChange = onShowDetailsChange)
        }
        if (showDetails) TechnicalDetails()
    }
}

@Composable
private fun ModeRowItem(row: ModeRow, onClick: () -> Unit) {
    Surface(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(22.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
    ) {
        Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(56.dp),
                contentAlignment = Alignment.Center
            ) {
                Surface(shape = RoundedCornerShape(16.dp), color = row.tint, modifier = Modifier.fillMaxSize()) {}
                Icon(row.icon, contentDescription = null, tint = Ink, modifier = Modifier.size(28.dp))
            }
            Spacer(Modifier.size(16.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(row.title, style = MaterialTheme.typography.titleLarge, color = MaterialTheme.colorScheme.onSurface)
                Spacer(Modifier.height(2.dp))
                Text(row.description, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

/**
 * The app's mark: a fingertip sends an amber beam to corner brackets around a target.
 * The beam draws itself once when the screen opens (skipped if animations are off).
 */
@Composable
private fun BeamHeader(modifier: Modifier) {
    val context = LocalContext.current
    val animationsOff = remember {
        Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
    }
    val progress = remember { Animatable(if (animationsOff) 1f else 0f) }
    LaunchedEffect(Unit) { progress.animateTo(1f, tween(900, delayMillis = 150, easing = FastOutSlowInEasing)) }
    val ink = MaterialTheme.colorScheme.onBackground

    Canvas(modifier.semantics { contentDescription = "A finger pointing at an object" }) {
        val h = size.height
        val tip = Offset(h * 0.32f, h * 0.62f)
        val target = Offset(size.width * 0.78f, h * 0.42f)
        val box = h * 0.62f
        val p = progress.value

        // Beam: fades in from the fingertip.
        val end = Offset(tip.x + (target.x - box / 2 - 10f - tip.x) * p, tip.y + (target.y - tip.y) * p)
        drawLine(
            brush = Brush.linearGradient(listOf(Beam.copy(alpha = 0.15f), Beam), start = tip, end = end),
            start = tip, end = end, strokeWidth = h * 0.07f, cap = StrokeCap.Round
        )
        // Fingertip.
        drawCircle(ink, radius = h * 0.13f, center = tip)
        drawCircle(Beam, radius = h * 0.13f, center = tip, style = Stroke(width = h * 0.045f))

        // Target object and its brackets; the brackets settle in as the beam arrives.
        val tl = Offset(target.x - box / 2, target.y - box / 2)
        drawRoundRect(ink.copy(alpha = 0.10f), topLeft = tl + Offset(box * 0.18f, box * 0.18f),
            size = Size(box * 0.64f, box * 0.64f), cornerRadius = CornerRadius(box * 0.12f))
        val gap = (1f - p) * box * 0.18f
        val len = box * 0.26f
        val w = h * 0.05f
        val corners = listOf(
            tl + Offset(-gap, -gap) to Offset(1f, 1f),
            Offset(tl.x + box + gap, tl.y - gap) to Offset(-1f, 1f),
            Offset(tl.x - gap, tl.y + box + gap) to Offset(1f, -1f),
            Offset(tl.x + box + gap, tl.y + box + gap) to Offset(-1f, -1f)
        )
        val bracket = if (p >= 0.99f) Beam else ink
        corners.forEach { (c, d) ->
            drawLine(bracket, c, c + Offset(len * d.x, 0f), strokeWidth = w, cap = StrokeCap.Round)
            drawLine(bracket, c, c + Offset(0f, len * d.y), strokeWidth = w, cap = StrokeCap.Round)
        }
    }
}

/** Model identity badges, for checking which weights this APK runs. */
@Composable
private fun TechnicalDetails() {
    val context = LocalContext.current
    val infos = remember {
        listOf(ModelInfo.SPATIALNET_ASSET, ModelInfo.CLASSIFIER_ASSET, ModelInfo.EARBUDS_MODEL_ASSET)
            .map { ModelInfo.read(context, it) }
    }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(top = 8.dp)) {
        infos.forEach { info ->
            ModelBadge(
                info = info,
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
                nameColor = MaterialTheme.colorScheme.onSurface,
                detailColor = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
