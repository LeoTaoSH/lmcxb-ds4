/* Model-free coverage of corpus parsing, whole-prompt packing and resume cursors. */
#include <assert.h>
#include "../ds4.c"

static void tokenize_chars(void *ctx, const char *text, token_vec *tokens) {
    (void)ctx;
    for (; *text; text++) token_vec_push(tokens, (unsigned char)*text);
}

static void check_tokens(const ds4_imatrix_batch *batch, const char *expected) {
    assert(batch->tokens.len == (int)strlen(expected));
    for (int i = 0; i < batch->tokens.len; i++) assert(batch->tokens.v[i] == expected[i]);
}

int main(void) {
    char corpus[] = "ignored preamble\n"
        "===== DS4_IMATRIX_PROMPT empty =====\n \n"
        "===== DS4_IMATRIX_PROMPT a =====\n abc \n"
        "===== DS4_IMATRIX_PROMPT b =====\n de \n"
        "===== DS4_IMATRIX_PROMPT c =====\n fghijk \n"
        "===== DS4_IMATRIX_PROMPT d =====\n lm \n";
    char original[sizeof(corpus)];
    memcpy(original, corpus, sizeof(corpus));
    ds4_imatrix_batch b = {0};
    const size_t len = strlen(corpus);
#define READ(offset, ctx, prompts, tokens, pack) \
    imatrix_batch_read(&b, corpus, len, offset, ctx, prompts, tokens, pack, tokenize_chars, NULL)
    READ(0, 100, 0, 0, 5);
    assert(b.count == 2 && b.lengths[0] == 3 && b.lengths[1] == 2);
    check_tokens(&b, "abcde");
    assert(corpus + b.next_offset == strstr(corpus, "===== DS4_IMATRIX_PROMPT c"));
    const size_t resume = b.next_offset;
    READ(resume, 100, 0, 0, 5); /* Oversized prompts are not split to fill a pack. */
    assert(b.count == 1 && b.lengths[0] == 6);
    check_tokens(&b, "fghijk");
    READ(b.next_offset, 100, 0, 0, 5);
    assert(b.count == 1 && b.next_offset == len);
    check_tokens(&b, "lm");
    READ(len, 100, 0, 0, 5);
    assert(!b.count && b.next_offset == len);
    READ(0, 100, 0, 0, 4); /* Non-fitting lookahead must leave its cursor intact. */
    assert(b.count == 1);
    check_tokens(&b, "abc");
    READ(b.next_offset, 100, 0, 0, 4);
    check_tokens(&b, "de");
    READ(0, 100, 0, 0, 0); /* Default/explicit zero disables packing. */
    assert(b.count == 1);
    check_tokens(&b, "abc");
    READ(0, 100, 1, 0, 100);
    assert(b.count == 1);
    check_tokens(&b, "abc");
    READ(0, 100, 0, 4, 100); /* Explicit max-tokens may truncate a prompt. */
    assert(b.count == 2 && b.lengths[1] == 1);
    check_tokens(&b, "abcd");
    READ(0, 2, 0, 0, 100); /* Context truncation is per prompt, not per pack. */
    assert(b.count == 4);
    check_tokens(&b, "abdefglm");
    assert(!memcmp(corpus, original, sizeof(corpus)));
#undef READ

    char plain[] = "  unmarked text  \n";
    imatrix_batch_read(&b, plain, strlen(plain), 0, 100, 0, 0, 8192, tokenize_chars, NULL);
    assert(b.count == 1 && b.next_offset == strlen(plain));
    check_tokens(&b, "unmarked text");
    char inline_marker[] = "x ===== DS4_IMATRIX_PROMPT inline\ny";
    imatrix_batch_read(&b, inline_marker, strlen(inline_marker), 0,
        100, 0, 0, 8192, tokenize_chars, NULL);
    assert(b.count == 1);
    check_tokens(&b, inline_marker);
    char empty[] = " \n===== DS4_IMATRIX_PROMPT empty\n \n===== DS4_IMATRIX_PROMPT unfinished";
    imatrix_batch_read(&b, empty, strlen(empty), 0, 100, 0, 0, 8192, tokenize_chars, NULL);
    assert(!b.count && b.next_offset == strlen(empty));
    imatrix_batch_free(&b);
    puts("Imatrix corpus parsing, pack limits and resume cursors: PASS");
    return 0;
}
