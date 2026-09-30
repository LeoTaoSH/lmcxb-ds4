/* Synthetic tokenizer-only GGUF for tests: 256 byte-level tokens (ids 0..255),
 * then one model family's chat markers, and no merges.  Open it with
 * ds4_engine_open_vocab_only(): no weights, no backend, no model file.
 *
 * Families are named as ds4_engine_open_vocab_only() names them:
 * "deepseek4", "deepseek41", "glm", "qwen4". */
#ifndef DS4_TEST_VOCAB_FIXTURE_H
#define DS4_TEST_VOCAB_FIXTURE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *vocab_fixture_markers(const char *family) {
    /* NUL separated, double NUL terminated */
    if (!strcmp(family, "deepseek4"))
        return "<｜begin▁of▁sentence｜>\0<｜end▁of▁sentence｜>\0<｜User｜>\0<｜Assistant｜>\0"
               "<think>\0</think>\0｜DSML｜\0";
    if (!strcmp(family, "deepseek41"))
        return "<｜begin▁of▁sentence｜>\0<｜end▁of▁sentence｜>\0<｜System｜>\0<｜User｜>\0"
               "<｜Assistant｜>\0<think>\0</think>\0｜DSML｜\0";
    if (!strcmp(family, "glm"))
        return "[gMASK]\0<|endoftext|>\0<sop>\0<|system|>\0<|user|>\0<|assistant|>\0"
               "<|observation|>\0<think>\0</think>\0<tool_call>\0</tool_call>\0"
               "<tool_response>\0</tool_response>\0<arg_key>\0</arg_key>\0"
               "<arg_value>\0</arg_value>\0";
    return "<|im_start|>\0<|im_end|>\0<|endoftext|>\0<think>\0</think>\0<tool_call>\0"
           "</tool_call>\0<tool_response>\0</tool_response>\0";
}

static void vf_u32(FILE *f, uint32_t n) { fwrite(&n, 4, 1, f); }
static void vf_u64(FILE *f, uint64_t n) { fwrite(&n, 8, 1, f); }
static void vf_str(FILE *f, const char *s, size_t n) { vf_u64(f, n); fwrite(s, n, 1, f); }

/* GPT-2 byte-to-unicode: printable bytes map to themselves, the others to
 * U+0100 upward in byte order. */
static size_t vf_byte_token(int b, char out[4]) {
    uint32_t cp = (uint32_t)b;
    if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174)) {
        uint32_t n = 0;
        for (int x = 0; x < b; x++)
            if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174)) n++;
        cp = 256 + n;
    }
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    out[0] = (char)(0xc0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3f));
    return 2;
}

/* Writes the fixture to a fresh temp file and returns its path (caller frees),
 * or NULL. */
static char *vocab_fixture_write(const char *family) {
    const char *markers = vocab_fixture_markers(family);
    int n_markers = 0;
    for (const char *p = markers; *p; p += strlen(p) + 1) n_markers++;
    char *path = strdup("/tmp/ds4-vocab-fixture-XXXXXX");
    int fd = mkstemp(path);
    if (fd < 0) { free(path); return NULL; }
    close(fd);
    FILE *f = fopen(path, "wb");
    if (!f) { free(path); return NULL; }
    const int glm = !strcmp(family, "glm");
    fwrite("GGUF", 4, 1, f);
    vf_u32(f, 3);
    vf_u64(f, 0);
    vf_u64(f, glm ? 4 : 2);
    vf_str(f, "tokenizer.ggml.tokens", 21); vf_u32(f, 9); vf_u32(f, 8);
    vf_u64(f, 256 + (uint64_t)n_markers);
    for (int b = 0; b < 256; b++) {
        char tok[4];
        vf_str(f, tok, vf_byte_token(b, tok));
    }
    for (const char *p = markers; *p; p += strlen(p) + 1) vf_str(f, p, strlen(p));
    vf_str(f, "tokenizer.ggml.merges", 21); vf_u32(f, 9); vf_u32(f, 8); vf_u64(f, 0);
    if (glm) { /* [gMASK] is the first marker, <|endoftext|> the second */
        vf_str(f, "tokenizer.ggml.bos_token_id", 27); vf_u32(f, 4); vf_u32(f, 256);
        vf_str(f, "tokenizer.ggml.eos_token_id", 27); vf_u32(f, 4); vf_u32(f, 257);
    }
    fclose(f);
    return path;
}

#endif
