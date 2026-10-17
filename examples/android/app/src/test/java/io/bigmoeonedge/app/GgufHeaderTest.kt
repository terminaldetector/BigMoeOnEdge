package io.bigmoeonedge.app

import io.bigmoeonedge.app.data.GgufHeader
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * GgufHeader.isMoe decides whether a picked/imported .gguf is streamable by this engine, without
 * reading any tensor data. These build tiny synthetic gguf headers byte-by-byte (metadata-only —
 * no multi-GB fixture needed) covering both signals it looks for, the split-shard case where only
 * the KV signal is present, the dense negative case, and the corrupt-file fallback path.
 */
class GgufHeaderTest {
    @Rule @JvmField val tmp = TemporaryFolder()

    private fun gguf(kv: List<Triple<String, Int, Any>>, tensors: List<Pair<String, Int>>): ByteArray {
        val out = ByteArrayOutputStream()
        fun u32(v: Int) { for (i in 0 until 4) out.write((v ushr (8 * i)) and 0xff) }
        fun u64(v: Long) { for (i in 0 until 8) out.write(((v ushr (8 * i)) and 0xff).toInt()) }
        fun str(s: String) {
            val b = s.toByteArray(Charsets.UTF_8)
            u64(b.size.toLong())
            out.write(b)
        }

        out.write('G'.code); out.write('G'.code); out.write('U'.code); out.write('F'.code)
        u32(3) // version
        u64(tensors.size.toLong())
        u64(kv.size.toLong())

        // KV block: (key, type, value). type 4 = uint32 here, the only scalar width these tests need.
        for ((key, type, value) in kv) {
            str(key)
            u32(type)
            when (type) {
                4 -> u32(value as Int) // T_UINT32
                else -> error("test helper only models uint32 kv values")
            }
        }

        // Tensor-info block: name, nDims (0 here — these tests only care about the name), ggml type, offset.
        for ((name, _) in tensors) {
            str(name)
            u32(0) // nDims
            u32(0) // ggml type
            u64(0) // data offset
        }
        return out.toByteArray()
    }

    private fun fileOf(bytes: ByteArray): File = tmp.newFile().apply { writeBytes(bytes) }

    @Test
    fun denseModel_noExpertSignal_isNotMoe() {
        val bytes = gguf(
            kv = listOf(Triple("general.architecture", 4, 0)),
            tensors = listOf("blk.0.attn_q.weight" to 0, "blk.0.ffn_down.weight" to 0),
        )
        assertFalse(GgufHeader.isMoe(fileOf(bytes)))
    }

    @Test
    fun expertCountKvAboveZero_isMoe() {
        val bytes = gguf(
            kv = listOf(Triple("qwen3moe.expert_count", 4, 128)),
            tensors = emptyList(),
        )
        assertTrue(GgufHeader.isMoe(fileOf(bytes)))
    }

    @Test
    fun expertCountKvZero_isNotMoeByItself() {
        val bytes = gguf(
            kv = listOf(Triple("qwen3moe.expert_count", 4, 0)),
            tensors = emptyList(),
        )
        assertFalse(GgufHeader.isMoe(fileOf(bytes)))
    }

    @Test
    fun expertTensorNameWithNoKvSignal_isMoe() {
        // No expert_count key at all — just the tensor name, e.g. a header the KV scan missed.
        val bytes = gguf(
            kv = listOf(Triple("general.architecture", 4, 0)),
            tensors = listOf("blk.0.ffn_down_exps" to 0),
        )
        assertTrue(GgufHeader.isMoe(fileOf(bytes)))
    }

    @Test
    fun splitShardFirstFile_metadataOnlyNoExpertTensors_stillDetectedByKv() {
        // The case the class doc calls out: a split model's first shard carries the KV metadata
        // but barely any tensors, so the tensor-name signal alone would miss it.
        val bytes = gguf(
            kv = listOf(Triple("qwen3moe.expert_count", 4, 64)),
            tensors = listOf("token_embd.weight" to 0),
        )
        assertTrue(GgufHeader.isMoe(fileOf(bytes)))
    }

    @Test
    fun corruptFile_noMagic_fallsBackAndIsNotMoe() {
        val garbage = "not a gguf file at all, just some bytes".toByteArray(Charsets.US_ASCII)
        assertFalse(GgufHeader.isMoe(fileOf(garbage)))
    }

    @Test
    fun corruptFile_withExpertCountSubstring_fallbackStillCatchesIt() {
        // A header the structured parser can't walk (e.g. truncated) but whose bytes still carry
        // the marker string: the documented fallback must only ever add detections, never miss one
        // the old substring probe would have caught.
        val garbage = "GGUF\u0001\u0000\u0000\u0000garbled...qwen3moe.expert_count...more garbage"
            .toByteArray(Charsets.US_ASCII)
        assertTrue(GgufHeader.isMoe(fileOf(garbage)))
    }
}
