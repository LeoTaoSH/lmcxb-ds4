/* No model download: exercise the serial V4.1 graph's actual handoff helper. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

int main(void) {
    int rc = 1;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    const uint64_t gib = UINT64_C(1) << 30;
    ds4_gpu_config cfg = {.n_gpus = 4};
    for (int i = 0; i < 4; i++) {
        cfg.device_indices[i] = i;
        cfg.vram_bytes[i] = 48 * gib;
    }
    const int placement[] = {0, 1, 2, 3, 0};
    ds41_gpu_graph g = {.ctx = 4096, .prefill_cap = 1, .placement = placement,
        .table = {{.fd = -1}, {.fd = -1}}};
    uint8_t raw[32 * 34] = {0};
    ds4_model model = {.map = raw, .size = sizeof(raw)};
    ds4_tensor weight = {.type = DS4_TENSOR_Q8_0, .ndim = 2,
        .dim = {32, 32}, .bytes = sizeof(raw)};
    ds4_tensor tensors[] = {
        {.name = {.ptr = "blk.1.engram_embd.weight", .len = 24}, .bytes = 94 * gib},
        {.name = {.ptr = "blk.14.engram_embd.weight", .len = 25}, .bytes = 95 * gib},
        {.name = {.ptr = "blk.0.ffn_gate_exps.weight", .len = 26}, .bytes = 3 * gib},
    };
    ds4_engine engine = {.model = {.tensors = tensors, .n_tensors = 3},
        .placement_ctx_hint = 4096, .placement_session_count_hint = 1};
    size_t entries[DS4_MAX_LAYER + 2];
    float input[32], actual[2048], checkpoint[2048];
    const uint32_t ids[] = {0, 1, 2, 3};
    const uint16_t scales[] = {0x3800, 0x3c00, 0x3e00, 0x4000};
    CHECK(engine_compute_entry_bytes(&engine, entries) == 0);
    CHECK(entries[1] == 3 * gib && entries[2] == 0 && entries[15] == 0);
    CHECK(engine_per_tier_graph_overhead_bytes(&engine) >=
        ds41_graph_bytes_placed(4096, true) + 2 * gib);
    engine.model.size = 152 * gib;
    CHECK(ds41_memory_admit_for_host(&engine, ds41_graph_bytes_placed(4096, true),
        false, 256 * gib, 192 * gib));
    CHECK(!ds41_memory_admit_for_host(&engine, ds41_graph_bytes_placed(4096, true),
        false, 256 * gib, 48 * gib));
    {
        ds4_tensor layout[DS4_MAX_LAYER + 4] = {0};
        char names[DS4_MAX_LAYER + 2][48];
        for (uint32_t i = 0; i < DS4_N_LAYER + 2; i++) {
            if (i == 0) snprintf(names[i], sizeof(names[i]), "token_embd.weight");
            else if (i == DS4_N_LAYER + 1)
                snprintf(names[i], sizeof(names[i]), "output.weight");
            else snprintf(names[i], sizeof(names[i]), "blk.%u.ffn_gate_exps.weight", i - 1);
            layout[i].name.ptr = names[i];
            layout[i].name.len = strlen(names[i]);
            layout[i].bytes = i == 0 || i == DS4_N_LAYER + 1 ? gib : 15 * gib / 4;
        }
        layout[DS4_N_LAYER + 2] = tensors[0];
        layout[DS4_N_LAYER + 3] = tensors[1];
        engine.model.tensors = layout;
        engine.model.n_tensors = DS4_N_LAYER + 4;
        CHECK(engine_classify_multi_tier(&engine, &cfg) == 0 && engine.multi_tier);
        for (int i = 0; i < 4; i++) {
            CHECK(engine.gpu_admission_cfg.vram_bytes[i] == cfg.vram_bytes[i]);
            CHECK(engine.gpu_cfg.vram_bytes[i] < engine.gpu_admission_cfg.vram_bytes[i]);
        }
        unsigned tiers = 0;
        for (uint32_t i = 0; i < DS4_N_LAYER + 2; i++) {
            CHECK(engine.placement[i] >= 0 && engine.placement[i] < 4);
            tiers |= 1u << engine.placement[i];
        }
        CHECK(tiers == 15u);
        cfg.n_gpus = 3;
        CHECK(engine_classify_multi_tier(&engine, &cfg) == 0);
        bool spilled = false;
        for (uint32_t i = 0; i < DS4_N_LAYER + 2; i++)
            spilled |= engine.placement[i] == DS4_LAYER_PACK_CPU;
        CHECK(spilled);
        cfg.n_gpus = 4;
        engine.model.tensors = tensors;
        engine.model.n_tensors = 3;
    }
    CHECK(ds41_prefill_count(&g, 8192) == 1);
    CHECK(ds41_short_prefill_count(&g, &engine.weights, 8) == 0);
    CHECK(!ds41_cuda_row_batch_supported(&g, &engine.weights));
    CHECK(ds4_gpu_init_multi(&cfg));
    CHECK(ds4_gpu_managed_pipeline_supported());
    CHECK(ds4_gpu_selected_working_set_size(&cfg) > 4 * 40 * gib);
    cfg.n_gpus = 1;
    CHECK(ds4_gpu_selected_working_set_size(&cfg) <= 48 * gib);
    CHECK(ds4_gpu_register_model_map_no_copy(raw, sizeof(raw)));
    for (int tier = 0; tier < 4; tier++) {
        for (int row = 0; row < 32; row++) {
            memcpy(raw + row * 34, &scales[tier], 2);
            memset(raw + row * 34 + 2, 1, 32);
        }
        ds4_tensor_range range = {.source_offset = 0, .bytes = sizeof(raw),
            .target_device = g_gpu[tier].device_id};
        CHECK(ds4_gpu_device_cache_tensors(g_gpu[tier].device_id, &range, 1) == 0);
    }
    /* Make the fallback regression deterministic even when every probe passes. */
    g_gpu_peer_ok[0][2] = 0;
#define ALLOC(field, count) CHECK((g.field = ds4_gpu_tensor_alloc_pipeline((count) * 4u)))
    ALLOC(norm, 32);
    ALLOC(block, 32);
    ALLOC(residual, 32);
    ALLOC(compressed[0], 2048);
    ALLOC(index_cache[0], 512);
    ALLOC(selected_comp, 4);
    ALLOC(selected_kv, 2048);
#undef ALLOC
    CHECK(g.norm->owner == 2);
    CHECK((g.rows_view = calloc(1, sizeof(*g.rows_view))));
    CHECK((g.rows_view[0].residual = ds4_gpu_tensor_view(g.residual, 0, 32 * 4)));
    CHECK((g.batch.residual = ds4_gpu_tensor_view(g.residual, 0, 32 * 4)));
    for (int i = 0; i < 32; i++) input[i] = 1;
    CHECK(ds4_gpu_tensor_write(g.norm, 0, input, sizeof(input)));
    CHECK(ds4_gpu_tensor_fill_f32(g.residual, 0, 32));
    CHECK(ds4_gpu_tensor_fill_f32(g.compressed[0], 0, 2048));
    CHECK(ds4_gpu_tensor_fill_f32(g.index_cache[0], 0, 512));
    CHECK(ds4_gpu_tensor_write(g.selected_comp, 0, ids, sizeof(ids)));
    float expected = 0;
    for (int entry = 0; entry < 5; entry++) {
        const int tier = placement[entry];
        CHECK(ds41_graph_select_tier(&g, entry));
        CHECK(g.norm->device_id == tier && g.compressed[0]->device_id == tier &&
            g.rows_view[0].residual->device_id == tier);
        CHECK(ds41_matmul(g.block, &model, &weight, g.norm, false));
        CHECK(ds4_gpu_add_tensor(g.residual, g.residual, g.block, 32));
        CHECK(ds41_bf16(g.residual, 32));
        CHECK(ds4_gpu_tensor_fill_f32(g.selected_kv, (float)(tier + 1), 512));
        CHECK(ds4_gpu_tensor_copy(g.compressed[0], (uint64_t)tier * 512 * 4,
            g.selected_kv, 0, 512 * 4));
        CHECK(ds4_gpu_tensor_copy(g.index_cache[0], (uint64_t)tier * 128 * 4,
            g.selected_kv, 0, 128 * 4));
        CHECK(ds4_gpu_dsv41_gather_kv(g.selected_kv, g.compressed[0],
            g.selected_comp, 4, 4));
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(g.residual, 0, actual, sizeof(input)));
        expected += 16 * (tier + 1);
        for (int i = 0; i < 32; i++) {
            if (actual[i] != expected)
                fprintf(stderr, "entry=%d tier=%d row=%d actual=%g expected=%g\n",
                        entry, tier, i, actual[i], expected);
            CHECK(actual[i] == expected);
        }
        CHECK(ds4_gpu_tensor_read(g.selected_kv, 0, actual, sizeof(actual)));
        for (int row = 0; row < 4; row++)
            for (int i = 0; i < 512; i++)
                CHECK(actual[row * 512 + i] == (row <= entry || entry == 4 ? row + 1 : 0));
        CHECK(ds4_gpu_tensor_read(g.index_cache[0], 0, actual, 512 * 4));
        for (int row = 0; row < 4; row++)
            for (int i = 0; i < 128; i++)
                CHECK(actual[row * 128 + i] == (row <= entry || entry == 4 ? row + 1 : 0));
    }
    /* Checkpoint-style host reads/restores survive a different cache consumer. */
    CHECK(ds4_gpu_tensor_read(g.compressed[0], 0, checkpoint, sizeof(checkpoint)));
    CHECK(ds41_graph_select_tier(&g, 2));
    CHECK(ds4_gpu_tensor_write(g.compressed[0], 0, checkpoint, sizeof(checkpoint)));
    CHECK(ds41_graph_select_tier(&g, 3));
    CHECK(ds4_gpu_tensor_read(g.compressed[0], 0, actual, sizeof(actual)));
    CHECK(memcmp(actual, checkpoint, sizeof(actual)) == 0);
    puts("V4.1 four-GPU placement, weight ownership, shared KV handoff and restore: PASS");
    rc = 0;
done:
    ds41_graph_free(&g);
    ds4_gpu_cleanup();
    return rc;
}
