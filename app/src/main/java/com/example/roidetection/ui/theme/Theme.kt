package com.example.roidetection.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

private val LightColors = lightColorScheme(
    primary = Ink,
    onPrimary = Color.White,
    primaryContainer = PaperRaised,
    onPrimaryContainer = Ink,
    secondary = BeamDeep,
    onSecondary = Color.White,
    secondaryContainer = Beam,
    onSecondaryContainer = Ink,
    tertiary = InkSoft,
    background = Paper,
    onBackground = Ink,
    surface = Paper,
    onSurface = Ink,
    surfaceVariant = PaperRaised,
    onSurfaceVariant = Slate,
    surfaceContainerHigh = PaperRaised,
    surfaceContainerHighest = PaperRaised,
    outline = Mist,
    outlineVariant = Color(0xFFDDE4EE),
    error = Color(0xFFB3261E)
)

private val DarkColors = darkColorScheme(
    primary = Beam,
    onPrimary = Ink,
    primaryContainer = InkSoft,
    onPrimaryContainer = Color.White,
    secondary = Beam,
    onSecondary = Ink,
    secondaryContainer = Beam,
    onSecondaryContainer = Ink,
    tertiary = Mist,
    background = Color(0xFF0C1729),
    onBackground = Color(0xFFEAF0F7),
    surface = Color(0xFF0C1729),
    onSurface = Color(0xFFEAF0F7),
    surfaceVariant = Ink,
    onSurfaceVariant = Mist,
    surfaceContainerHigh = Ink,
    surfaceContainerHighest = InkSoft,
    outline = InkSoft,
    outlineVariant = InkSoft,
    error = Coral
)

private val AppShapes = Shapes(
    extraSmall = RoundedCornerShape(8.dp),
    small = RoundedCornerShape(12.dp),
    medium = RoundedCornerShape(18.dp),
    large = RoundedCornerShape(24.dp),
    extraLarge = RoundedCornerShape(28.dp)
)

/**
 * The app's own palette in light and dark. Android's wallpaper colours are not used,
 * so the app looks the same, and keeps its contrast, on every phone.
 */
@Composable
fun ROIDetectionTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        typography = Typography,
        shapes = AppShapes,
        content = content
    )
}
