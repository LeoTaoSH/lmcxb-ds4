/* Dedicated Metal host: test packed attention against independent prompts on
 * the real weights, without running MoE. All input projections are fixed so
 * equality here is exact, not a tolerance that could hide cross-prompt leaks. */
#include "../ds4.c"
#include <assert.h>

static void fill_input(ds4_gpu_tensor *tensor, uint32_t off, uint32_t rows,
                       uint32_t width, uint32_t seed) {
    const size_t n = (size_t)rows * width;
    float *values = xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        values[i] = ((int)(seed >> 24) - 128) / 128.0f;
    }
    assert(ds4_gpu_tensor_write(tensor, (uint64_t)off * width * sizeof(float),
                               values, n * sizeof(float)));
    free(values);
}

static void fill_attention(ds41_gpu_graph *g, uint32_t off, uint32_t rows,
                           uint32_t prompt, uint32_t layer) {
    const uint32_t seed = 1234u + prompt * 7537u + layer * 439u;
    fill_input(g->batch.norm, off, rows, DS4_N_EMBD, seed);
    fill_input(g->batch.qr, off, rows, DS4_N_LORA_Q, seed + 1);
    fill_input(g->batch.q, off, rows, DS4_N_HEAD * DS4_N_HEAD_DIM, seed + 2);
    fill_input(g->batch.kv, off, rows, DS4_N_HEAD_DIM, seed + 3);
}

static uint64_t head_hash(ds41_gpu_graph *g, uint32_t off, uint32_t rows) {
    const size_t row_bytes = (size_t)DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float);
    void *data = xmalloc(rows * row_bytes);
    assert(ds4_gpu_tensor_read(g->batch.heads, off * row_bytes, data, rows * row_bytes));
    const uint64_t hash = imatrix_checkpoint_hash(UINT64_C(1469598103934665603),
                                                 data, rows * row_bytes);
    free(data);
    return hash;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s V4.1_MODEL_GGUF\n", argv[0]);
        return 2;
    }
    ds4_engine *e = NULL;
    const ds4_engine_options opt = {.model_path = argv[1],
        .backend = DS4_BACKEND_METAL, .context_size = 32768,
        .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(54) << 30};
    assert(ds4_engine_open(&e, &opt) == 0);
    assert(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    ds41_gpu_graph g = {.table = {{.fd = -1}, {.fd = -1}}};
    assert(ds41_graph_alloc(&g, &e->model, &e->weights, argv[1], 32768, true));
    /* Singleton, odd/even compressor boundaries, >2048 raw partitions, and
     * >1024 compressed keys exercise raw/mixed/indexed attention and reuse. */
    const uint32_t lengths[] = {1, 33, 2051, 2052};
    const uint32_t layers[] = {0, 2, 3, 8, 9, 14, 15, 20, 21, 24, 25, 28, 32, 36};
    int tokens[4137];
    uint32_t ids[4137][2][DS4_ENGRAM_COLS];
    for (uint32_t t = 0; t < 4137; t++) tokens[t] = (int)(17 + t % 500);
    ds4_engram_history history = g.history;
    assert(ds41_hash_packed_tokens(&g, &history, tokens, lengths, 4));
    uint32_t offset = 0;
    for (uint32_t p = 0; p < 4; p++) {
        ds4_engram_history_reset(&history);
        assert(ds41_hash_tokens(&g, &history, tokens + offset, lengths[p], &ids[offset][0][0]));
        offset += lengths[p];
    }
    assert(memcmp(ids, g.prefill_ids, sizeof(ids)) == 0);
    puts("Packed Engram histories: exact independent-prompt hashes PASS");
    uint64_t expected[4][14];
    for (uint32_t p = 0; p < 4; p++) {
        ds41_graph_reset(&g);
        memset(ds4_gpu_tensor_contents(g.batch.selected_comp), 0xff,
               (size_t)lengths[p] * DS4_N_INDEXER_TOP_K * sizeof(int32_t));
        for (uint32_t j = 0; j < 14; j++) {
            const uint32_t il = layers[j];
            assert(metal_graph_stream_map_layer_decode(&e->model, &e->weights, il));
            fill_attention(&g, 0, lengths[p], p, il);
            assert(ds4_gpu_begin_commands());
            assert(ds41_attention_batch(&g, &e->model, &e->weights.layer[il], il, lengths[p]));
            assert(ds4_gpu_end_commands());
            expected[p][j] = head_hash(&g, 0, lengths[p]);
        }
    }
    for (uint32_t reverse = 0; reverse < 2; reverse++) {
        uint32_t packed_lengths[4];
        for (uint32_t p = 0; p < 4; p++) packed_lengths[p] = lengths[reverse ? 3 - p : p];
        ds41_graph_reset(&g);
        memset(ds4_gpu_tensor_contents(g.batch.selected_comp), 0xff,
               (size_t)4137 * DS4_N_INDEXER_TOP_K * sizeof(int32_t));
        for (uint32_t j = 0; j < 14; j++) {
            const uint32_t il = layers[j];
            assert(metal_graph_stream_map_layer_decode(&e->model, &e->weights, il));
            uint32_t off = 0;
            for (uint32_t p = 0; p < 4; p++) {
                fill_attention(&g, off, packed_lengths[p], reverse ? 3 - p : p, il);
                off += packed_lengths[p];
            }
            assert(ds4_gpu_begin_commands());
            assert(ds41_attention_packed(&g, &e->model, &e->weights.layer[il], il, packed_lengths, 4));
            assert(ds4_gpu_end_commands());
            off = 0;
            for (uint32_t p = 0; p < 4; p++) {
                const uint64_t actual = head_hash(&g, off, packed_lengths[p]);
                fprintf(stderr, "packed attention reverse=%u layer=%u prompt=%u rows=%u: %s\n",
                        reverse, il, p, packed_lengths[p],
                        actual == expected[reverse ? 3 - p : p][j] ? "exact" : "MISMATCH");
                assert(actual == expected[reverse ? 3 - p : p][j]);
                off += packed_lengths[p];
            }
        }
    }
    ds41_graph_free(&g);
    ds4_engine_close(e);
    puts("Packed attention: exact independent-prompt and permutation checks PASS");
    return 0;
}
