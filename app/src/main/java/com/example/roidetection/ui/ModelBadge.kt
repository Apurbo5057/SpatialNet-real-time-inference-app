package com.example.roidetection.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.roidetection.ModelInfo

/**
 * Shows which model weights this build is actually running: the asset name
 * plus the first four bytes of their SHA-256 and the file size. The digest
 * is hashed from the bundled asset at runtime, so a stale install shows a
 * stale digest instead of silently claiming to be current.
 */
@Composable
fun ModelBadge(
    info: ModelInfo?,
    modifier: Modifier = Modifier,
    unavailableName: String = "model unavailable",
    unavailableDetail: String = "asset missing from APK",
    containerColor: Color = Color.Black.copy(alpha = 0.7f),
    nameColor: Color = Color(0xFF7CE38B),
    detailColor: Color = Color.White.copy(alpha = 0.75f)
) {
    Surface(
        modifier = modifier,
        color = containerColor,
        shape = MaterialTheme.shapes.small
    ) {
        Column(modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)) {
            Text(
                text = info?.name ?: unavailableName,
                color = nameColor,
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = info?.detail ?: unavailableDetail,
                color = detailColor,
                fontSize = 10.sp,
                fontWeight = FontWeight.Medium
            )
        }
    }
}
