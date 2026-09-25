package com.example.roidetection.earbuds

/** One supported earbuds model from the bundled catalog. */
data class EarbudModel(
    val id: String,
    val brand: String,
    val model: String,
    /** Plain-language spec rows, e.g. "Bluetooth" to "5.4". Unknown values are left out. */
    val specs: List<Pair<String, String>>,
    val features: List<String>,
    /** Java regexes matched against the Bluetooth name the earbuds advertise while pairing. */
    val namePatterns: List<String>,
    val defaultNames: List<String>,
    val pairingSteps: List<String>,
    val thumbnailAsset: String?,
    /** False until someone confirmed the Bluetooth name on real earbuds. */
    val verifiedOnDevice: Boolean
) {
    /** "realme" + "Buds Air7" -> "realme Buds Air7"; "Galaxy Buds FE" already starts with a sub-brand. */
    val displayName: String
        get() = if (model.startsWith(brand, ignoreCase = true)) model else "$brand $model"

    private val regexes: List<Regex> by lazy { namePatterns.mapNotNull { runCatching { Regex(it) }.getOrNull() } }

    /** True if [bluetoothName] is what this model calls itself over Bluetooth. */
    fun matchesName(bluetoothName: String?): Boolean =
        bluetoothName != null && regexes.any { it.containsMatchIn(bluetoothName) }

    companion object {
        private val SPEC_LABELS = listOf(
            "batteryEarbudHours" to "Battery (earbuds)",
            "batteryWithCaseHours" to "Battery (with case)",
            "anc" to "Noise cancelling",
            "bluetoothVersion" to "Bluetooth",
            "codecs" to "Audio codecs",
            "waterResistance" to "Water resistance",
            "earbudWeightGrams" to "Weight (each earbud)",
            "chargingPort" to "Charging port",
            "wirelessCharging" to "Wireless charging"
        )

        /** Turns raw spec values (String, Boolean, Number, List) into display rows. */
        fun specRows(raw: Map<String, Any?>): List<Pair<String, String>> = SPEC_LABELS.mapNotNull { (key, label) ->
            val text = when (val v = raw[key]) {
                null -> null
                is Boolean -> if (v) "Yes" else "No"
                is Number -> if (key == "earbudWeightGrams") "${v} g" else v.toString()
                is List<*> -> v.filterNotNull().joinToString(", ").ifBlank { null }
                else -> v.toString().ifBlank { null }
            }
            text?.let { label to it }
        }
    }
}
