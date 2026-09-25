package com.example.roidetection.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.widget.Toast
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.List
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.roidetection.AppMode
import com.example.roidetection.CombinedResult
import com.example.roidetection.TeachState
import com.example.roidetection.read.ReadResult
import com.example.roidetection.ui.theme.Beam
import com.example.roidetection.ui.theme.Ink
import com.example.roidetection.ui.theme.Mist

/** One plain-language instruction for the top of each mode screen. */
fun guidanceText(result: CombinedResult, mode: AppMode): String {
    val roi = result.roiResult
    return when {
        !roi.isHandDetected -> "Show your hand to the camera"
        !result.isPointing -> "Point at something with one finger"
        roi.roiBBox == null -> "Point a little closer to the object"
        else -> when (mode) {
            AppMode.IDENTIFY -> "Hold still to see what it is"
            AppMode.MY_OBJECTS -> "Hold still on the object"
            AppMode.READ -> "Hold still on the text or code"
            AppMode.EARBUDS -> "Hold still on the earbuds or their case"
        }
    }
}

@Composable
fun GuidanceBanner(text: String, modifier: Modifier = Modifier) = GlassPill(text, modifier)

/**
 * The same text as the caption floating on the ROI box, so the panel never
 * disagrees with it; a hint when nothing is being pointed at.
 */
@Composable
private fun CaptionText(result: CombinedResult, hint: String, modifier: Modifier = Modifier) {
    val label = result.roiLabel
    AnswerText(label?.text ?: hint, confident = label?.confident == true, isHint = label == null, modifier = modifier)
}

// ---------------------------------------------------------------- Identify

@Composable
fun IdentifyPanel(
    result: CombinedResult,
    speaker: Speaker,
    onSavePhoto: () -> Unit
) {
    GlassPanel {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            CaptionText(result, hint = "Point at an object to hear what it is", modifier = Modifier.weight(1f))
            GlassIconButton(AppIcons.Speaker, "Say it", onClick = { result.roiLabel?.let { speaker.speak(it.spoken) } },
                enabled = result.roiLabel != null)
            GlassIconButton(AppIcons.Camera, "Save photo", onClick = onSavePhoto)
        }
    }
}

// ---------------------------------------------------------------- My Objects

@Composable
fun MyObjectsPanel(
    result: CombinedResult,
    teachState: TeachState,
    savedCount: Int,
    speaker: Speaker,
    onTeach: () -> Unit,
    onCancelTeach: () -> Unit,
    onOpenList: () -> Unit
) {
    val canTeach = result.isPointing && result.roiResult.roiBBox != null
    GlassPanel {
        if (teachState is TeachState.Collecting) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    AnswerText("Learning ${teachState.count} of ${teachState.target}", confident = false, isHint = false)
                    GlassNote("Keep pointing, and move the phone a little to show other sides.")
                }
                GlassIconButton(Icons.Filled.Close, "Cancel", onClick = onCancelTeach)
            }
            LearningBar(teachState.count, teachState.target)
            return@GlassPanel
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Column(modifier = Modifier.weight(1f)) {
                CaptionText(
                    result,
                    hint = if (savedCount == 0) "Point at one of your things, then tap Remember this"
                    else "Point at one of your things"
                )
                result.matchedObject?.note?.takeIf { it.isNotBlank() }?.let { GlassNote(it, color = Color.White, maxLines = 2) }
            }
            GlassIconButton(AppIcons.Speaker, "Say it", onClick = { result.roiLabel?.let { speaker.speak(it.spoken) } },
                enabled = result.roiLabel != null)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            BeamButton("Remember this", onClick = onTeach, enabled = canTeach, icon = Icons.Filled.Add, modifier = Modifier.weight(1f))
            GlassButton("My list ($savedCount)", onClick = onOpenList, icon = Icons.Filled.List, modifier = Modifier.weight(1f))
        }
    }
}

/** Segmented progress for the teaching samples: one amber segment per sample taken. */
@Composable
private fun LearningBar(count: Int, target: Int) {
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
        repeat(target) { i ->
            Box(
                Modifier
                    .weight(1f)
                    .heightIn(min = 6.dp)
                    .clip(RoundedCornerShape(3.dp))
                    .background(if (i < count) Beam else Color(0x33FFFFFF))
            )
        }
    }
}

/** Asks for a name (and optional note) once the object has been learned. */
@Composable
fun TeachNameDialog(
    state: TeachState.Naming,
    existingNames: List<String>,
    onSave: (name: String, note: String) -> Unit,
    onCancel: () -> Unit
) {
    var name by remember(state) { mutableStateOf(state.suggestedName) }
    var note by remember(state) { mutableStateOf("") }
    val merging = existingNames.any { it.equals(name.trim(), ignoreCase = true) }

    AlertDialog(
        onDismissRequest = onCancel,
        title = { Text("What is this?") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                state.photo?.let { PhotoThumb(it) }
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Name, e.g. My keys") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
                OutlinedTextField(
                    value = note,
                    onValueChange = { note = it },
                    label = { Text("Note (optional), e.g. Water on Sunday") },
                    modifier = Modifier.fillMaxWidth()
                )
                if (merging) {
                    Text(
                        "You already have \"${name.trim()}\". These photos will be added to it, so it gets recognised better.",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        },
        confirmButton = { Button(onClick = { onSave(name, note) }, enabled = name.isNotBlank()) { Text("Save") } },
        dismissButton = { TextButton(onClick = onCancel) { Text("Cancel") } }
    )
}

@Composable
private fun PhotoThumb(bitmap: Bitmap) {
    Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = "Photo of the object",
            contentScale = ContentScale.Crop,
            modifier = Modifier.size(128.dp).clip(RoundedCornerShape(18.dp))
        )
    }
}

// ---------------------------------------------------------------- Read

@Composable
fun ReadPanel(
    readResult: ReadResult?,
    speaker: Speaker,
    autoSpeak: Boolean,
    onAutoSpeakChange: (Boolean) -> Unit,
    onReadWholeView: () -> Unit,
    onClear: () -> Unit
) {
    val context = LocalContext.current

    // Speak new text by itself when the user asked for that.
    var lastSpoken by remember { mutableStateOf("") }
    LaunchedEffect(readResult, autoSpeak) {
        val say = readResult?.let { spokenText(it) } ?: return@LaunchedEffect
        if (autoSpeak && say != lastSpoken) {
            speaker.speak(say)
            lastSpoken = say
        }
    }

    GlassPanel {
        val code = readResult?.codes?.firstOrNull()
        Row(verticalAlignment = Alignment.Top) {
            Box(modifier = Modifier.weight(1f)) {
                when {
                    readResult == null -> AnswerText("Point at text, a QR code, or a colour", confident = false, isHint = true)
                    code != null -> Column {
                        GlassNote(code.kindLabel)
                        AnswerText(code.text, confident = true, isHint = false)
                    }
                    readResult.text.isNotBlank() -> Text(
                        readResult.text,
                        color = Color.White,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.heightIn(max = 96.dp).verticalScroll(rememberScrollState())
                    )
                    readResult.colorName != null -> Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            Modifier.size(28.dp).clip(RoundedCornerShape(8.dp))
                                .background(Color(readResult.colorRgb ?: 0))
                        )
                        Spacer(Modifier.size(10.dp))
                        AnswerText(readResult.colorName, confident = true, isHint = false)
                    }
                    else -> GlassNote("No text found. Move closer or add light.")
                }
            }
            if (readResult != null) {
                GlassIconButton(Icons.Filled.Close, "Clear", onClick = { speaker.stop(); onClear() })
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            GlassIconButton(AppIcons.Speaker, "Read aloud", onClick = { readResult?.let { speaker.speak(spokenText(it)) } },
                enabled = readResult != null)
            if (code?.url != null) {
                GlassIconButton(AppIcons.OpenLink, "Open link", onClick = { openLink(context, code.url) })
            } else {
                GlassIconButton(AppIcons.Copy, "Copy",
                    onClick = { readResult?.let { copyText(context, code?.text ?: it.text) } },
                    enabled = readResult != null && (code != null || readResult.text.isNotBlank()))
            }
            BeamButton("Read all", onClick = onReadWholeView, icon = AppIcons.Page, modifier = Modifier.weight(1f))
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            GlassNote("Read new text aloud by itself", color = Color.White, modifier = Modifier.weight(1f), maxLines = 1)
            Switch(
                checked = autoSpeak,
                onCheckedChange = onAutoSpeakChange,
                colors = SwitchDefaults.colors(
                    checkedThumbColor = Ink, checkedTrackColor = Beam,
                    uncheckedThumbColor = Mist, uncheckedTrackColor = Color(0x33FFFFFF), uncheckedBorderColor = Color(0x59FFFFFF)
                )
            )
        }
    }
}

private fun spokenText(r: ReadResult): String {
    r.codes.firstOrNull()?.let { return "${it.kindLabel}: ${it.text}" }
    if (r.text.isNotBlank()) return r.text
    return r.colorName?.let { "The colour is $it" } ?: "No text found"
}

private fun copyText(context: Context, text: String) {
    val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    cm.setPrimaryClip(ClipData.newPlainText("Read text", text))
    Toast.makeText(context, "Copied", Toast.LENGTH_SHORT).show()
}

private fun openLink(context: Context, url: String) {
    try {
        context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
    } catch (e: Exception) {
        Toast.makeText(context, "No app can open this link", Toast.LENGTH_SHORT).show()
    }
}
