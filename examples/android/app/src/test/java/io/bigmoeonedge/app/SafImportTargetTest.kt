package io.bigmoeonedge.app

import io.bigmoeonedge.app.data.SafImport
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/**
 * Where a SAF import lands when a file of the same name already sits in the models dir. The rule
 * under test: a size match is treated as the same file (re-picking what is already there is a
 * no-op); a size mismatch must never silently keep the stale file under the new import's name, so
 * it gets a uniquified name instead, preserving any split-shard suffix.
 */
class SafImportTargetTest {
    @Rule @JvmField val tmp = TemporaryFolder()

    @Test
    fun noExistingFile_usesThePickedName() {
        val dir = tmp.newFolder()
        val target = SafImport.resolveTarget(dir, "model.gguf", 1000L)
        assertEquals("model.gguf", target.name)
    }

    @Test
    fun sameSizeAsExisting_reusesIt() {
        val dir = tmp.newFolder()
        val existing = java.io.File(dir, "model.gguf").apply { writeBytes(ByteArray(1000)) }
        val target = SafImport.resolveTarget(dir, "model.gguf", 1000L)
        assertEquals(existing.absolutePath, target.absolutePath)
    }

    @Test
    fun differentSizeThanExisting_picksAUniqueName_ratherThanKeepingTheOldOne() {
        val dir = tmp.newFolder()
        java.io.File(dir, "model.gguf").apply { writeBytes(ByteArray(1000)) }
        val target = SafImport.resolveTarget(dir, "model.gguf", 2000L)
        assertEquals("model (2).gguf", target.name)
    }

    @Test
    fun differentSizeThanExisting_keepsSplitShardSuffixAtTheEnd() {
        val dir = tmp.newFolder()
        java.io.File(dir, "model-00001-of-00003.gguf").apply { writeBytes(ByteArray(1000)) }
        val target = SafImport.resolveTarget(dir, "model-00001-of-00003.gguf", 2000L)
        // Must still match the server's shard regex (stem)-(NNNNN)-of-(NNNNN).gguf.
        assertEquals("model (2)-00001-of-00003.gguf", target.name)
    }

    @Test
    fun severalConflicts_skipsEveryNameAlreadyTaken() {
        val dir = tmp.newFolder()
        java.io.File(dir, "model.gguf").apply { writeBytes(ByteArray(1000)) }
        java.io.File(dir, "model (2).gguf").apply { writeBytes(ByteArray(1000)) }
        val target = SafImport.resolveTarget(dir, "model.gguf", 2000L)
        assertEquals("model (3).gguf", target.name)
    }

    @Test
    fun unknownPickedSize_trustsTheNameAsBefore() {
        val dir = tmp.newFolder()
        val existing = java.io.File(dir, "model.gguf").apply { writeBytes(ByteArray(1000)) }
        val target = SafImport.resolveTarget(dir, "model.gguf", -1L)
        assertEquals(existing.absolutePath, target.absolutePath)
    }
}
