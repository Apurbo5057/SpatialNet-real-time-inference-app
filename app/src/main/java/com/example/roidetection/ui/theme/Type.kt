package com.example.roidetection.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.example.roidetection.R

/**
 * Atkinson Hyperlegible (Braille Institute, SIL OFL): drawn so similar letters and digits
 * can't be mistaken for each other, for an app that reads things out to everyone.
 * Bundled in the APK, since the app never goes online.
 */
val Atkinson = FontFamily(
    Font(R.font.atkinson_regular, FontWeight.Normal),
    Font(R.font.atkinson_bold, FontWeight.Bold)
)

// Scale on a 1.25 ratio from 16 sp body text; bold carries hierarchy, sizes stay few.
private fun style(size: Int, line: Int, bold: Boolean = false, tracking: Double = 0.0) = TextStyle(
    fontFamily = Atkinson,
    fontWeight = if (bold) FontWeight.Bold else FontWeight.Normal,
    fontSize = size.sp,
    lineHeight = line.sp,
    letterSpacing = tracking.sp
)

val Typography = Typography(
    displaySmall = style(38, 44, bold = true, tracking = -0.6),
    headlineMedium = style(30, 36, bold = true, tracking = -0.4),
    headlineSmall = style(24, 30, bold = true, tracking = -0.2),
    titleLarge = style(20, 26, bold = true),
    titleMedium = style(18, 24, bold = true),
    titleSmall = style(16, 22, bold = true),
    bodyLarge = style(17, 25),
    bodyMedium = style(15, 22),
    bodySmall = style(13, 18),
    labelLarge = style(16, 20, bold = true),
    labelMedium = style(14, 18, bold = true),
    labelSmall = style(12, 16)
)
