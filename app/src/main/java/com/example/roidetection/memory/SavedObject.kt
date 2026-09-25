package com.example.roidetection.memory

/**
 * An object the user taught the app. [embeddings] are L2-normalized samples
 * from different moments of teaching; more samples make recognition sturdier.
 */
data class SavedObject(
    val id: String,
    val name: String,
    val note: String = "",
    val embeddings: List<FloatArray> = emptyList(),
    val createdAtMs: Long = 0L,
    val lastSeenAtMs: Long? = null,
    /** Crop of the object saved when it was taught. */
    val photoFile: String? = null,
    /** Whole camera view the last time it was recognised, to show where it was. */
    val lastSeenPhotoFile: String? = null
)
