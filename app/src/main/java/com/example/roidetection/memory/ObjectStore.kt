package com.example.roidetection.memory

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

/**
 * Saves the user's taught objects in the app's private storage:
 * `files/my_objects/objects.json` plus one JPEG per photo. Nothing leaves the phone.
 */
class ObjectStore(context: Context) {

    companion object {
        private const val TAG = "ObjectStore"
        private const val MAX_SAMPLES_PER_OBJECT = 20
    }

    private val dir = File(context.filesDir, "my_objects").apply { mkdirs() }
    private val indexFile = File(dir, "objects.json")

    @Synchronized
    fun load(): List<SavedObject> {
        if (!indexFile.exists()) return emptyList()
        return try {
            val arr = JSONArray(indexFile.readText())
            (0 until arr.length()).map { fromJson(arr.getJSONObject(it)) }
        } catch (e: Exception) {
            Log.e(TAG, "Could not read ${indexFile.name}: ${e.message}", e)
            emptyList()
        }
    }

    /**
     * Adds a new object, or adds the samples to an existing object with the same
     * name (ignoring case), so teaching the same thing again improves it.
     */
    @Synchronized
    fun teach(
        current: List<SavedObject>,
        name: String,
        note: String,
        samples: List<FloatArray>,
        photo: Bitmap?,
        nowMs: Long
    ): List<SavedObject> {
        val existing = current.firstOrNull { it.name.equals(name, ignoreCase = true) }
        val updated = if (existing != null) {
            existing.copy(
                note = note.ifBlank { existing.note },
                embeddings = (existing.embeddings + samples).takeLast(MAX_SAMPLES_PER_OBJECT),
                photoFile = existing.photoFile ?: photo?.let { savePhoto(existing.id, "photo", it) }
            )
        } else {
            val id = UUID.randomUUID().toString()
            SavedObject(
                id = id,
                name = name,
                note = note,
                embeddings = samples,
                createdAtMs = nowMs,
                photoFile = photo?.let { savePhoto(id, "photo", it) }
            )
        }
        return save(current.filter { it.id != updated.id } + updated)
    }

    @Synchronized
    fun update(current: List<SavedObject>, obj: SavedObject): List<SavedObject> =
        save(current.map { if (it.id == obj.id) obj else it })

    @Synchronized
    fun markSeen(current: List<SavedObject>, id: String, nowMs: Long, view: Bitmap?): List<SavedObject> {
        val obj = current.firstOrNull { it.id == id } ?: return current
        val photo = view?.let { savePhoto(id, "last_seen", it) } ?: obj.lastSeenPhotoFile
        return update(current, obj.copy(lastSeenAtMs = nowMs, lastSeenPhotoFile = photo))
    }

    @Synchronized
    fun delete(current: List<SavedObject>, id: String): List<SavedObject> {
        dir.listFiles { f -> f.name.startsWith(id) }?.forEach { it.delete() }
        return save(current.filter { it.id != id })
    }

    fun file(name: String?): File? = name?.let { File(dir, it) }?.takeIf { it.exists() }

    private fun savePhoto(id: String, kind: String, bitmap: Bitmap): String? = try {
        val name = "${id}_$kind.jpg"
        File(dir, name).outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 85, it) }
        name
    } catch (e: Exception) {
        Log.e(TAG, "Could not save photo: ${e.message}", e)
        null
    }

    private fun save(objects: List<SavedObject>): List<SavedObject> {
        val arr = JSONArray()
        objects.forEach { arr.put(toJson(it)) }
        val tmp = File(dir, "objects.json.tmp")
        tmp.writeText(arr.toString())
        tmp.renameTo(indexFile)
        return objects
    }

    private fun toJson(o: SavedObject) = JSONObject().apply {
        put("id", o.id)
        put("name", o.name)
        put("note", o.note)
        put("createdAtMs", o.createdAtMs)
        o.lastSeenAtMs?.let { put("lastSeenAtMs", it) }
        o.photoFile?.let { put("photoFile", it) }
        o.lastSeenPhotoFile?.let { put("lastSeenPhotoFile", it) }
        put("embeddings", JSONArray().apply {
            o.embeddings.forEach { e -> put(JSONArray().apply { e.forEach { put(it.toDouble()) } }) }
        })
    }

    private fun fromJson(j: JSONObject): SavedObject {
        val embs = j.getJSONArray("embeddings")
        return SavedObject(
            id = j.getString("id"),
            name = j.getString("name"),
            note = j.optString("note", ""),
            embeddings = (0 until embs.length()).map { i ->
                val e = embs.getJSONArray(i)
                FloatArray(e.length()) { e.getDouble(it).toFloat() }
            },
            createdAtMs = j.optLong("createdAtMs", 0L),
            lastSeenAtMs = if (j.has("lastSeenAtMs")) j.getLong("lastSeenAtMs") else null,
            photoFile = j.optString("photoFile").ifEmpty { null },
            lastSeenPhotoFile = j.optString("lastSeenPhotoFile").ifEmpty { null }
        )
    }
}
