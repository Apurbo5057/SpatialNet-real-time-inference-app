package com.example.roidetection.read

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.concurrent.TimeUnit

/** A QR code or barcode: its content, and a link when it holds one. */
data class CodeResult(val text: String, val url: String?, val kindLabel: String)

/** Everything Read mode found in one crop. */
data class ReadResult(
    val text: String,
    val codes: List<CodeResult>,
    val colorName: String?,
    val colorRgb: Int?,
    val computedAtMs: Long,
    val wholeView: Boolean
) {
    val isEmpty: Boolean get() = text.isBlank() && codes.isEmpty()
}

/**
 * On-device text and code reading with ML Kit's bundled models, which ship
 * inside the APK and never download anything. Latin-script text only.
 * Blocking; call from a background thread.
 */
class TextReader {
    private val textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val barcodeScanner = BarcodeScanning.getClient()

    fun read(bitmap: Bitmap): Pair<String, List<CodeResult>> {
        val image = InputImage.fromBitmap(bitmap, 0)
        val text = Tasks.await(textRecognizer.process(image), 5, TimeUnit.SECONDS)
        val codes = Tasks.await(barcodeScanner.process(image), 5, TimeUnit.SECONDS)
        val lines = text.textBlocks.joinToString("\n") { block -> block.lines.joinToString(" ") { it.text } }
        return lines to codes.mapNotNull { it.toCodeResult() }
    }

    fun close() {
        textRecognizer.close()
        barcodeScanner.close()
    }

    private fun Barcode.toCodeResult(): CodeResult? {
        val value = displayValue ?: rawValue ?: return null
        val kind = when (valueType) {
            Barcode.TYPE_URL -> "Link"
            Barcode.TYPE_WIFI -> "Wi-Fi"
            Barcode.TYPE_CONTACT_INFO -> "Contact"
            Barcode.TYPE_PHONE -> "Phone number"
            Barcode.TYPE_EMAIL -> "Email"
            Barcode.TYPE_PRODUCT, Barcode.TYPE_ISBN -> "Product code"
            else -> if (format == Barcode.FORMAT_QR_CODE) "QR code" else "Barcode"
        }
        val link = url?.url ?: value.takeIf { it.startsWith("http://") || it.startsWith("https://") }
        val shown = if (valueType == Barcode.TYPE_WIFI) "Network: ${wifi?.ssid ?: value}" else value
        return CodeResult(shown, link, kind)
    }
}
