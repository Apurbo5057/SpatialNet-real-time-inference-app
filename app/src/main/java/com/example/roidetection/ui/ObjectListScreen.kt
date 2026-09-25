package com.example.roidetection.ui

import android.graphics.BitmapFactory
import android.text.format.DateUtils
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Place
import androidx.compose.material3.Surface
import com.example.roidetection.ui.theme.Ink
import com.example.roidetection.ui.theme.TintObjects
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.roidetection.InferenceViewModel
import com.example.roidetection.memory.SavedObject
import java.io.File

/** The user's taught objects: photo, note, and when and where each was last seen. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ObjectListScreen(viewModel: InferenceViewModel, onBack: () -> Unit) {
    val objects by viewModel.savedObjects.collectAsState()
    var editing by remember { mutableStateOf<SavedObject?>(null) }
    var deleting by remember { mutableStateOf<SavedObject?>(null) }
    var showingWhere by remember { mutableStateOf<SavedObject?>(null) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("My list") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        if (objects.isEmpty()) {
            Column(modifier = Modifier.fillMaxSize().padding(padding).padding(24.dp)) {
                Text("No objects yet", fontSize = 22.sp, fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.size(8.dp))
                Text(
                    "Go back, point at one of your things (like your keys), and tap \"Remember this\".",
                    fontSize = 16.sp
                )
            }
            return@Scaffold
        }
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(objects.sortedByDescending { it.lastSeenAtMs ?: it.createdAtMs }, key = { it.id }) { obj ->
                ObjectCard(
                    obj = obj,
                    photo = viewModel.objectPhotoFile(obj.photoFile),
                    onWhere = { showingWhere = obj },
                    onEdit = { editing = obj },
                    onDelete = { deleting = obj }
                )
            }
        }
    }

    editing?.let { obj ->
        var name by remember(obj.id) { mutableStateOf(obj.name) }
        var note by remember(obj.id) { mutableStateOf(obj.note) }
        // Two objects with one name would be merged the next time that name is taught.
        val duplicate = objects.any { it.id != obj.id && it.name.equals(name.trim(), ignoreCase = true) }
        AlertDialog(
            onDismissRequest = { editing = null },
            title = { Text("Edit ${obj.name}") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedTextField(
                        value = name,
                        onValueChange = { name = it },
                        label = { Text("Name") },
                        singleLine = true,
                        isError = duplicate || name.isBlank(),
                        supportingText = when {
                            duplicate -> { { Text("You already have an object called \"${name.trim()}\"") } }
                            name.isBlank() -> { { Text("Please type a name") } }
                            else -> null
                        },
                        modifier = Modifier.fillMaxWidth()
                    )
                    OutlinedTextField(
                        value = note,
                        onValueChange = { note = it },
                        label = { Text("Note (optional), e.g. Water on Sunday") },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = { viewModel.updateObject(obj.id, name, note); editing = null },
                    enabled = name.isNotBlank() && !duplicate
                ) { Text("Save") }
            },
            dismissButton = { TextButton(onClick = { editing = null }) { Text("Cancel") } }
        )
    }

    deleting?.let { obj ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Forget ${obj.name}?") },
            text = { Text("The app will stop recognising it. You can teach it again later.") },
            confirmButton = { Button(onClick = { viewModel.deleteObject(obj.id); deleting = null }) { Text("Forget") } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Keep") } }
        )
    }

    showingWhere?.let { obj ->
        AlertDialog(
            onDismissRequest = { showingWhere = null },
            title = { Text("Where ${obj.name} was") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(lastSeenText(obj))
                    FilePhoto(viewModel.objectPhotoFile(obj.lastSeenPhotoFile), Modifier.fillMaxWidth(), ContentScale.Fit)
                }
            },
            confirmButton = { TextButton(onClick = { showingWhere = null }) { Text("Close") } }
        )
    }
}

@Composable
private fun ObjectCard(
    obj: SavedObject,
    photo: File?,
    onWhere: () -> Unit,
    onEdit: () -> Unit,
    onDelete: () -> Unit
) {
    Surface(
        shape = RoundedCornerShape(22.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                FilePhoto(
                    photo,
                    Modifier.size(72.dp).clip(RoundedCornerShape(12.dp)),
                    ContentScale.Crop
                )
                Spacer(modifier = Modifier.size(16.dp))
                Column {
                    Text(obj.name, fontSize = 20.sp, fontWeight = FontWeight.Bold)
                    Text(lastSeenText(obj), fontSize = 14.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            if (obj.note.isNotBlank()) Text("📝 ${obj.note}", fontSize = 16.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = onWhere, enabled = obj.lastSeenPhotoFile != null) {
                    Icon(Icons.Filled.Place, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.size(6.dp))
                    Text("Where")
                }
                OutlinedButton(onClick = onEdit) {
                    Icon(Icons.Filled.Edit, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.size(6.dp))
                    Text("Edit")
                }
                TextButton(onClick = onDelete) { Text("Forget", color = MaterialTheme.colorScheme.onSurfaceVariant) }
            }
        }
    }
}

@Composable
private fun FilePhoto(file: File?, modifier: Modifier, contentScale: ContentScale) {
    val bitmap = remember(file?.path, file?.lastModified()) {
        file?.let { BitmapFactory.decodeFile(it.path) }
    }
    if (bitmap != null) {
        Image(bitmap.asImageBitmap(), contentDescription = null, modifier = modifier, contentScale = contentScale)
    } else {
        Box(modifier = modifier.background(TintObjects), contentAlignment = Alignment.Center) {
            Icon(AppIcons.Tag, contentDescription = null, tint = Ink, modifier = Modifier.size(28.dp))
        }
    }
}

private fun lastSeenText(obj: SavedObject): String {
    val seen = obj.lastSeenAtMs ?: return "Not seen since you taught it"
    return "Last seen " + DateUtils.getRelativeTimeSpanString(seen, System.currentTimeMillis(), DateUtils.MINUTE_IN_MILLIS)
}
