package com.example.roidetection.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.roidetection.ui.theme.Beam
import com.example.roidetection.ui.theme.Glass
import com.example.roidetection.ui.theme.GlassLine
import com.example.roidetection.ui.theme.Ink
import com.example.roidetection.ui.theme.Mist

/*
 * Camera-screen chrome. Everything over the live picture is "glass": Ink at 72% with a
 * hairline edge, so the view stays visible. Amber is kept for the answer and the one
 * main action; everything else is white or Mist.
 */

@Composable
fun GlassPanel(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Surface(
        modifier = modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 12.dp),
        color = Glass,
        shape = RoundedCornerShape(22.dp),
        border = BorderStroke(1.dp, GlassLine)
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            content = content
        )
    }
}

/** The instruction pill at the top of camera screens. */
@Composable
fun GlassPill(text: String, modifier: Modifier = Modifier) {
    Surface(modifier = modifier, color = Glass, shape = CircleShape, border = BorderStroke(1.dp, GlassLine)) {
        Text(
            text,
            color = Color.White,
            style = MaterialTheme.typography.labelLarge,
            modifier = Modifier.padding(horizontal = 18.dp, vertical = 9.dp)
        )
    }
}

/** The one main action on a camera screen: amber with Ink text. */
@Composable
fun BeamButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    icon: ImageVector? = null
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.heightIn(min = 48.dp),
        shape = RoundedCornerShape(14.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = Beam,
            contentColor = Ink,
            disabledContainerColor = Color(0x33FFFFFF),
            disabledContentColor = Color(0x99FFFFFF)
        ),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
    ) {
        ButtonContent(text, icon)
    }
}

/** A secondary action on glass: white outline. */
@Composable
fun GlassButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    icon: ImageVector? = null
) {
    OutlinedButton(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.heightIn(min = 48.dp),
        shape = RoundedCornerShape(14.dp),
        border = BorderStroke(1.dp, if (enabled) Color(0x80FFFFFF) else GlassLine),
        colors = ButtonDefaults.outlinedButtonColors(
            contentColor = Color.White,
            disabledContentColor = Color(0x80FFFFFF)
        ),
        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 8.dp)
    ) {
        ButtonContent(text, icon)
    }
}

@Composable
private fun ButtonContent(text: String, icon: ImageVector?) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        icon?.let {
            Icon(it, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(Modifier.size(8.dp))
        }
        Text(text, style = MaterialTheme.typography.labelLarge, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

/** Round icon button on glass, e.g. speak or save photo. */
@Composable
fun GlassIconButton(icon: ImageVector, description: String, onClick: () -> Unit, enabled: Boolean = true) {
    FilledIconButton(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier.size(48.dp),
        colors = IconButtonDefaults.filledIconButtonColors(
            containerColor = Color(0x26FFFFFF),
            contentColor = Color.White,
            disabledContainerColor = Color(0x14FFFFFF),
            disabledContentColor = Color(0x59FFFFFF)
        )
    ) {
        Icon(icon, contentDescription = description, modifier = Modifier.size(22.dp))
    }
}

/**
 * The answer line, mirroring the caption on the box: amber and large when it is an
 * answer, white while working, Mist for hints.
 */
@Composable
fun AnswerText(text: String, confident: Boolean, isHint: Boolean, modifier: Modifier = Modifier) {
    Text(
        text = text,
        color = when {
            isHint -> Mist
            confident -> Beam
            else -> Color.White
        },
        style = if (isHint) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.titleLarge,
        maxLines = 2,
        overflow = TextOverflow.Ellipsis,
        modifier = modifier
    )
}

/** Secondary text on glass. */
@Composable
fun GlassNote(text: String, modifier: Modifier = Modifier, color: Color = Mist, maxLines: Int = 3) {
    Text(text, color = color, style = MaterialTheme.typography.bodyMedium, maxLines = maxLines,
        overflow = TextOverflow.Ellipsis, modifier = modifier)
}
