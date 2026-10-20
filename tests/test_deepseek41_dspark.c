/* Native V4.1 DSpark graph oracle. No target experts are made resident. */
#include "../ds4.c"

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; } } while (0)

static bool dump_tensor(const char *dir, const char *name, ds4_gpu_tensor *x, size_t floats) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s.f32", dir, name) >= (int)sizeof(path)) return false;
    float *data = malloc(floats * sizeof(float));
    if (!data) return false;
    bool ok = ds4_gpu_tensor_read(x, 0, data, floats * sizeof(float));
    for (size_t i = 0; ok && i < floats; i++) ok = isfinite(data[i]);
    FILE *fp = ok ? fopen(path, "wb") : NULL;
    ok = fp && fwrite(data, sizeof(float), floats, fp) == floats;
    if (fp && fclose(fp)) ok = false;
    free(data);
    return ok;
}

static bool cancel_now(void *unused) {
    (void)unused;
    return true;
}

static int check_candidate_mask(void) {
    int rc = 1;
    ds41_gpu_graph g = {0};
    ds4_gpu_tensor *reference = NULL;
    const uint32_t widths[] = {1, 7, 8, 9, 127, 1025, 8191, 8192, 12307, 16383, 16384, 16385, 17017};
    const uint32_t capacity = 17017, max_blocks = (capacity + 7u) / 8u;
    float *scores = malloc(capacity * sizeof(float));
    float *expected = malloc(max_blocks * sizeof(float));
    float *actual = malloc(max_blocks * sizeof(float));
    g_ds4_shape = DS4_SHAPE_FLASH41;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++)
        g_ds4_compress_ratios[il] = ds4_expected_layer_compress_ratio(il);
    REQUIRE(scores && expected && actual && ds4_gpu_init());
    g.index_scores = ds4_gpu_tensor_alloc(capacity * sizeof(float));
    g.block_scores = ds4_gpu_tensor_alloc(max_blocks * sizeof(float));
    g.block_selected = ds4_gpu_tensor_alloc(2048u * sizeof(uint32_t));
    g.block_mask = ds4_gpu_tensor_alloc(max_blocks * sizeof(float));
    reference = ds4_gpu_tensor_alloc(max_blocks * sizeof(float));
    REQUIRE(g.index_scores && g.block_scores && g.block_selected && g.block_mask && reference);
    for (uint32_t i = 0; i < capacity; i++) scores[i] = (float)((i * 7919u) % 104729u) - 50000;
    REQUIRE(ds4_gpu_tensor_write(g.index_scores, 0, scores, capacity * sizeof(float)));
    for (unsigned wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t n = widths[wi], blocks = (n + 7u) / 8u;
        const uint32_t top = blocks < 2048u ? blocks : 2048u;
        g.pos = n - 1u;
        REQUIRE(ds4_gpu_begin_commands());
        REQUIRE(ds4_gpu_dsv41_candidate_blocks(g.block_scores, g.index_scores, n, 1, g.pos, 1));
        REQUIRE(ds4_gpu_indexer_topk_tensor(g.block_selected, g.block_scores, blocks, 1, top));
        REQUIRE(ds4_gpu_dsv4_topk_mask_tensor(reference, g.block_selected, blocks, 1, top));
        REQUIRE(ds4_gpu_tensor_fill_f32(g.block_mask, -123, max_blocks));
        REQUIRE(ds41_attention_candidates(&g, 20));
        REQUIRE(ds4_gpu_end_commands());
        REQUIRE(ds4_gpu_tensor_read(reference, 0, expected, blocks * sizeof(float)));
        REQUIRE(ds4_gpu_tensor_read(g.block_mask, 0, actual, max_blocks * sizeof(float)));
        REQUIRE(!memcmp(expected, actual, blocks * sizeof(float)));
        for (uint32_t i = blocks; i < max_blocks; i++) REQUIRE(actual[i] == -123);
        fprintf(stderr, "V4.1 candidate mask width=%u: exact\n", n);
    }
    rc = 0;
done:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(g.block_mask);
    ds4_gpu_tensor_free(g.block_selected);
    ds4_gpu_tensor_free(g.block_scores);
    ds4_gpu_tensor_free(g.index_scores);
    ds4_gpu_cleanup();
    free(actual); free(expected); free(scores);
    return rc;
}

static int check_runtime(const char *target, const char *draft, const ds4_tp_options *tp_opt) {
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_tp *tp = NULL;
    ds4_session *a = NULL, *b = NULL;
    ds4_tokens prompt = {0};
    FILE *snapshot = NULL;
    char err[256] = {0};
    ds4_engine_options opt = {.model_path = target, .mtp_path = draft,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .context_size = 2048, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(40) << 30, .dspark = true};
    if (tp_opt) { opt.tp = *tp_opt; opt.ssd_streaming = false; }
    REQUIRE(!ds4_engine_open(&engine, &opt));
    if (tp_opt) {
        ds4_tp_identity id = {.gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = ds4_engine_model_id(engine), .n_layer = DS4_N_LAYER,
            .n_embd = DS4_N_EMBD, .n_vocab = DS4_N_VOCAB,
            .quant_bits = ds4_engine_routed_quant_bits(engine), .ctx_size = 2048};
        ds4_engine_tp_gate_schedule(engine, &id.gate_slot_start, &id.gate_slot_step,
            &id.gates_per_token, id.gate_slot_mask);
        REQUIRE(ds4_tp_create(&tp, &opt.tp, &id, err, sizeof(err)));
        REQUIRE(ds4_engine_tp_bind(engine, tp, err, sizeof(err)));
    }
    REQUIRE(!ds4_session_create(&a, engine, 2048));
    REQUIRE(!ds4_session_create(&b, engine, 2048));
    for (unsigned i = 0; i < 127; i++) ds4_tokens_push(&prompt, 300 + i % 23);
    REQUIRE(!ds4_session_sync(a, &prompt, err, sizeof(err)));
    snapshot = tmpfile();
    REQUIRE(snapshot && !ds4_session_save_payload(a, snapshot, err, sizeof(err)));
    const uint64_t bytes = ds4_session_payload_bytes(a);
    for (unsigned keep = 0; keep <= 6; keep++) {
        if (tp) {
            ds4_session_invalidate(a); ds4_session_invalidate(b);
            REQUIRE(!ds4_session_sync(a, &prompt, err, sizeof(err)));
            REQUIRE(!ds4_session_sync(b, &prompt, err, sizeof(err)));
        } else {
            rewind(snapshot);
            REQUIRE(!ds4_session_load_payload(a, snapshot, bytes, err, sizeof(err)));
            rewind(snapshot);
            REQUIRE(!ds4_session_load_payload(b, snapshot, bytes, err, sizeof(err)));
        }
        int ta[6], tb[6];
        for (unsigned i = 0; i < 6; i++) {
            ta[i] = 700 + i;
            tb[i] = i < keep ? ta[i] : 900 + (int)i;
        }
        ds41_dspark *da = a->ds41_graph.dspark, *db = b->ds41_graph.dspark;
        REQUIRE(ds41_session_verify(a, ta, 6));
        REQUIRE(ds41_session_commit(a, ta, 6, 6));
        REQUIRE(ds41_session_verify(b, tb, 6));
        REQUIRE(ds41_session_commit(b, tb, 6, keep));
        double worst = 0;
        for (uint64_t i = 0; i < (uint64_t)keep * DS4_N_VOCAB; i++)
            worst = fmax(worst, fabs(da->host_verified[i] - db->host_verified[i]));
        fprintf(stderr, "verify prefix=%u causal max=%.8g\n", keep, worst);
        REQUIRE(worst < 0.0001);
        if (keep < 6) REQUIRE(ds4_session_rewind_speculative(a, 127 + keep));
        REQUIRE(a->checkpoint.len == b->checkpoint.len && a->checkpoint_valid);
        REQUIRE(!memcmp(a->logits, b->logits, DS4_N_VOCAB * sizeof(float)));
        for (unsigned i = 0; i < 8; i++) {
            REQUIRE(!ds4_session_eval(a, 1200 + i, err, sizeof(err)));
            REQUIRE(!ds4_session_eval(b, 1200 + i, err, sizeof(err)));
            worst = 0;
            for (uint32_t v = 0; v < DS4_N_VOCAB; v++)
                worst = fmax(worst, fabs(a->logits[v] - b->logits[v]));
            REQUIRE(worst < 0.0001);
        }
        REQUIRE(!ds4_session_rewind_speculative(a, a->checkpoint.len - 1));
        fprintf(stderr, "verify prefix=%u rollback and 8 continuation steps PASS\n", keep);
    }
    puts("V4.1 DSpark causal verification, all prefix commits/rewinds and snapshot load: PASS");
    for (unsigned exact = 0; exact < 2; exact++) {
        ds4_session_invalidate(a);
        REQUIRE(!ds4_session_sync(a, &prompt, err, sizeof(err)));
        engine->dspark_exact_sampling = exact != 0;
        engine->dspark_confidence_threshold = 0;
        uint64_t rng = 12345;
        for (unsigned cycle = 0; cycle < 8; cycle++) {
            const int seed = sample_argmax(a->logits, DS4_N_VOCAB);
            const int before = a->checkpoint.len;
            int accepted[6];
            const int count = ds4_session_eval_speculative(a, seed, 6, -1,
                0.7f, 0, 0.95f, 0, &rng, accepted, 6, err, sizeof(err));
            REQUIRE(count >= 1 && count <= 6 && accepted[0] == seed);
            REQUIRE(a->checkpoint.len == before + count &&
                    a->ds41_graph.pos == (uint32_t)a->checkpoint.len);
            for (int i = 0; i < count; i++)
                REQUIRE(a->checkpoint.v[before + i] == accepted[i]);
            if (count > 1) {
                REQUIRE(ds4_session_rewind_speculative(a, before + count - 1));
                REQUIRE(!ds4_session_eval(a, accepted[count - 1], err, sizeof(err)));
                REQUIRE(a->checkpoint.len == before + count);
            }
        }
        if (!tp) {
            rewind(snapshot);
            REQUIRE(!ds4_session_save_payload(a, snapshot, err, sizeof(err)));
            const uint64_t saved_bytes = ds4_session_payload_bytes(a);
            rewind(snapshot);
            REQUIRE(!ds4_session_load_payload(b, snapshot, saved_bytes, err, sizeof(err)));
            REQUIRE(!b->ds41_graph.dspark->kv_valid);
            for (unsigned i = 0; i < 3; i++) {
                REQUIRE(!ds4_session_eval(a, 1500 + i, err, sizeof(err)));
                REQUIRE(!ds4_session_eval(b, 1500 + i, err, sizeof(err)));
                REQUIRE(!memcmp(a->logits, b->logits, DS4_N_VOCAB * sizeof(float)));
            }
        }
        fprintf(stderr, "V4.1 DSpark exact=%u sampling and frontier%s: PASS\n", exact,
            tp ? " (physical TP)" : ", post-verify snapshot");
    }
    {
        int accepted[6];
        uint64_t rng = 71;
        const int pos = a->checkpoint.len;
        ds4_session_set_cancel(a, cancel_now, NULL);
        REQUIRE(ds4_session_eval_speculative(a, sample_argmax(a->logits, DS4_N_VOCAB),
            6, -1, 0.7f, 0, 0.95f, 0, &rng, accepted, 6, err, sizeof(err)) < 0);
        REQUIRE(a->checkpoint_valid && a->checkpoint.len == pos);
        ds4_session_set_cancel(a, NULL, NULL);
        err[0] = '\0';
        REQUIRE(!ds4_session_eval(a, 1600, err, sizeof(err)));
        while (prompt.len < 2043) ds4_tokens_push(&prompt, 300 + prompt.len % 23);
        ds4_session_invalidate(a);
        REQUIRE(!ds4_session_sync(a, &prompt, err, sizeof(err)));
        while (a->checkpoint.len < 2048) {
            const int before = a->checkpoint.len;
            const int n = ds4_session_eval_speculative(a, sample_argmax(a->logits, DS4_N_VOCAB),
                6, -1, 0.7f, 0, 0.95f, 0, &rng, accepted, 6, err, sizeof(err));
            REQUIRE(n >= 1 && n <= 2048 - before && a->checkpoint.len == before + n);
        }
        REQUIRE(ds4_session_eval_speculative(a, 1600, 6, -1, 0.7f, 0, 0.95f, 0,
            &rng, accepted, 6, err, sizeof(err)) == 0);
        fprintf(stderr, "V4.1 DSpark cancellation and context-limit clamping: PASS\n");
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "%s\n", err);
    if (snapshot) fclose(snapshot);
    ds4_tokens_free(&prompt);
    ds4_session_free(b); ds4_session_free(a);
    if (tp) (void)ds4_tp_send_stop(tp);
    ds4_engine_close(engine); ds4_tp_free(tp);
    return rc;
}

#ifdef __APPLE__
static char *read_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    rewind(fp);
    char *text = size > 0 ? malloc((size_t)size + 1u) : NULL;
    if (text && fread(text, 1, (size_t)size, fp) != (size_t)size) { free(text); text = NULL; }
    if (text) text[size] = 0;
    fclose(fp);
    return text;
}

/* A verified suffix must be what one-token decode computes, and any kept
 * prefix must leave the session that decode would have left: every logit of
 * every row, the serialized state, and the tokens that follow. */
static int check_verify_parity(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *scalar = NULL, *verified = NULL;
    ds4_session_snapshot base = {0}, want = {0}, got = {0};
    ds4_tokens prompt = {0}, prefix = {0};
    ds41_spec spec = {0};
    float *rows = NULL;
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    const uint32_t targets[] = {37, 38, 39};
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    REQUIRE(text);
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(prompt.len > 1100);
    REQUIRE(ds4_session_create(&scalar, engine, 4096) == 0);
    REQUIRE(ds4_session_create(&verified, engine, 4096) == 0);
    REQUIRE(ds41_spec_alloc(&spec, targets, 3));
    rows = malloc((size_t)DS4_TP_BATCH_MAX_ROWS * DS4_N_VOCAB * sizeof(float));
    REQUIRE(rows);
    /* Below and across the 128-row window, odd and even frontiers, and a
     * frontier whose rows straddle a window wrap. */
    const struct { int prefix; uint32_t count; } cases[] = {
        {40, 2}, {40, 3}, {40, 6}, {41, 5}, {126, 6}, {250, 8}, {251, 2}, {1021, 6}, {1022, 3},
    };
    /* DS4_TEST_VERIFY_CASE=PREFIX,ROWS runs that one frontier instead. */
    int only_prefix = 0, only_count = 0;
    const char *only = getenv("DS4_TEST_VERIFY_CASE");
    if (only && sscanf(only, "%d,%d", &only_prefix, &only_count) != 2) only = NULL;
    for (size_t c = 0; c < (only ? 1u : sizeof(cases) / sizeof(cases[0])); c++) {
        const int start = only ? only_prefix : cases[c].prefix;
        const uint32_t count = only ? (uint32_t)only_count : cases[c].count;
        REQUIRE(start > 0 && count >= 2 && count <= DS4_TP_BATCH_MAX_ROWS);
        const int *suffix = prompt.v + start;
        prefix.len = 0;
        for (int i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
        REQUIRE(ds4_session_sync(scalar, &prefix, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_save_snapshot(scalar, &base, err, sizeof(err)) == 0);
        /* One-token reference rows. */
        for (uint32_t i = 0; i < count; i++) {
            REQUIRE(ds4_session_eval(scalar, suffix[i], err, sizeof(err)) == 0);
            memcpy(rows + (size_t)i * DS4_N_VOCAB, scalar->logits, DS4_N_VOCAB * sizeof(float));
        }
        for (uint32_t keep = 1; keep <= count; keep++) {
            REQUIRE(ds4_session_load_snapshot(verified, &base, err, sizeof(err)) == 0);
            ds41_gpu_graph *g = &verified->ds41_graph;
            REQUIRE(g->pos == (uint32_t)start);
            REQUIRE(ds41_graph_verify(g, &engine->model, &engine->weights, suffix, count, &spec));
            REQUIRE(g->pos == (uint32_t)start);
            const float *logits = ds4_gpu_tensor_contents(spec.logits);
            REQUIRE(logits);
            bool exact = true;
            for (uint32_t i = 0; i < count; i++) {
                if (memcmp(logits + (size_t)i * DS4_N_VOCAB, rows + (size_t)i * DS4_N_VOCAB,
                           DS4_N_VOCAB * sizeof(float)) != 0) {
                    double worst = 0;
                    for (uint32_t v = 0; v < DS4_N_VOCAB; v++)
                        worst = fmax(worst, fabs((double)logits[(size_t)i * DS4_N_VOCAB + v] -
                                                 rows[(size_t)i * DS4_N_VOCAB + v]));
                    fprintf(stderr, "prefix %d rows %u: row %u differs from one-token decode (max %.9g)\n",
                            start, count, i, worst);
                    exact = false;
                }
            }
            if (!exact) goto done;
            REQUIRE(ds4_gpu_begin_commands() && ds41_spec_commit(g, &spec, keep) &&
                    ds4_gpu_end_commands());
            REQUIRE(g->pos == (uint32_t)start + keep);
            for (uint32_t i = 0; i < keep; i++) token_vec_push(&verified->checkpoint, suffix[i]);
            memcpy(verified->logits, logits + (size_t)(keep - 1u) * DS4_N_VOCAB,
                   DS4_N_VOCAB * sizeof(float));
            /* The reference for this prefix, and three tokens past it. */
            REQUIRE(ds4_session_load_snapshot(scalar, &base, err, sizeof(err)) == 0);
            for (uint32_t i = 0; i < keep; i++)
                REQUIRE(ds4_session_eval(scalar, suffix[i], err, sizeof(err)) == 0);
            REQUIRE(ds4_session_save_snapshot(scalar, &want, err, sizeof(err)) == 0);
            REQUIRE(ds4_session_save_snapshot(verified, &got, err, sizeof(err)) == 0);
            REQUIRE(want.len == got.len);
            if (memcmp(want.ptr, got.ptr, want.len) != 0) {
                size_t at = 0;
                while (((const uint8_t *)want.ptr)[at] == ((const uint8_t *)got.ptr)[at]) at++;
                fprintf(stderr, "prefix %d rows %u keep %u: state differs at byte %zu of %zu\n",
                        start, count, keep, at, (size_t)want.len);
                goto done;
            }
            for (uint32_t i = 0; i < 3; i++) {
                const int next = suffix[keep + i];
                REQUIRE(ds4_session_eval(scalar, next, err, sizeof(err)) == 0);
                REQUIRE(ds4_session_eval(verified, next, err, sizeof(err)) == 0);
                REQUIRE(memcmp(scalar->logits, verified->logits, DS4_N_VOCAB * sizeof(float)) == 0);
            }
        }
        fprintf(stderr, "V4.1 verify parity prefix=%d rows=%u: every row, kept prefix and continuation exact\n",
                start, count);
        REQUIRE(ds4_session_load_snapshot(scalar, &base, err, sizeof(err)) == 0);
    }
    puts("V4.1 speculative verification: PASS");
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    free(rows);
    ds41_spec_free(&spec);
    ds4_session_snapshot_free(&base); ds4_session_snapshot_free(&want); ds4_session_snapshot_free(&got);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(verified); ds4_session_free(scalar);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* The production continued prefill of two to eight rows is one-token decode:
 * every routed recipe keeps scalar reductions through eight rows. */
static int check_short_prefill(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *scalar = NULL, *batched = NULL;
    ds4_tokens prompt = {0}, prefix = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    REQUIRE(text && ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(ds4_session_create(&scalar, engine, 4096) == 0);
    REQUIRE(ds4_session_create(&batched, engine, 4096) == 0);
    for (int start = 40; start <= 41; start++) {
        for (int extra = 2; extra <= 8; extra += extra < 5 ? 3 : 1) {
            prefix.len = 0;
            for (int i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
            ds4_session_invalidate(scalar); ds4_session_invalidate(batched);
            REQUIRE(ds4_session_sync(scalar, &prefix, err, sizeof(err)) == 0);
            REQUIRE(ds4_session_sync(batched, &prefix, err, sizeof(err)) == 0);
            REQUIRE(memcmp(scalar->logits, batched->logits, DS4_N_VOCAB * sizeof(float)) == 0);
            for (int i = 0; i < extra; i++) {
                REQUIRE(ds4_session_eval(scalar, prompt.v[start + i], err, sizeof(err)) == 0);
                ds4_tokens_push(&prefix, prompt.v[start + i]);
            }
            REQUIRE(ds4_session_sync(batched, &prefix, err, sizeof(err)) == 0);
            double worst = 0;
            for (uint32_t v = 0; v < DS4_N_VOCAB; v++)
                worst = fmax(worst, fabs((double)scalar->logits[v] - batched->logits[v]));
            fprintf(stderr, "prefix %d + %d continued rows: max logit difference from decode %.9g\n",
                    start, extra, worst);
            REQUIRE(worst == 0);
        }
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(batched); ds4_session_free(scalar);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* Measurement, not a check: what a verified row costs next to a decoded token. */
static int time_verify(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_session_snapshot base = {0};
    ds4_tokens prompt = {0}, prefix = {0};
    ds41_spec spec = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    const uint32_t targets[] = {37, 38, 39};
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    REQUIRE(text && ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(ds4_session_create(&session, engine, 4096) == 0);
    REQUIRE(ds41_spec_alloc(&spec, targets, 3));
    const int start = 1024;
    for (int i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
    REQUIRE(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_save_snapshot(session, &base, err, sizeof(err)) == 0);
    for (int warm = 0; warm < 8; warm++)
        REQUIRE(ds4_session_eval(session, prompt.v[start + warm], err, sizeof(err)) == 0);
    double t = now_sec();
    for (int i = 0; i < 32; i++)
        REQUIRE(ds4_session_eval(session, prompt.v[start + 8 + i], err, sizeof(err)) == 0);
    const double step_ms = (now_sec() - t) * 1000.0 / 32.0;
    fprintf(stderr, "one-token decode: %.2f ms\n", step_ms);
    for (uint32_t rows = 2; rows <= 8; rows += 2) {
        const char *ab_env = getenv("DS4_TEST_VERIFY_AB_ENV");
        if (ab_env && !strncmp(ab_env,"DS4_METAL_DISABLE_V41_",sizeof("DS4_METAL_DISABLE_V41_")-1)) {
            const size_t bytes = (size_t)rows * DS4_N_VOCAB * sizeof(float);
            float *reference = malloc(bytes), *actual = malloc(bytes);
            REQUIRE(reference && actual);
            double totals[2] = {0,0};
            for (unsigned rep=0;rep<20;rep++) {
                const unsigned slot=rep%4, cycle=rep/4;
                const unsigned rollback=(slot==1 || slot==2) ^ (cycle%2);
                if (rollback) setenv(ab_env,"1",1); else unsetenv(ab_env);
                REQUIRE(ds4_session_load_snapshot(session,&base,err,sizeof(err))==0);
                const double begin=now_sec();
                REQUIRE(ds41_graph_verify(&session->ds41_graph,&engine->model,&engine->weights,
                                          prompt.v+start,rows,&spec));
                const double ms=(now_sec()-begin)*1000.0;
                if(rep>=4) totals[rollback]+=ms;
                REQUIRE(ds4_gpu_tensor_read(spec.logits,0,actual,bytes));
                if(!rep) memcpy(reference,actual,bytes);
                else REQUIRE(!memcmp(reference,actual,bytes));
            }
            unsetenv(ab_env);
            fprintf(stderr,"verify_ab env=%s rows=%u control_ms=%.6f rollback_ms=%.6f saved_percent=%.6f exact_floats=%llu\n",
                ab_env,rows,totals[0]/8.0,totals[1]/8.0,
                100.0*(totals[1]-totals[0])/totals[1],
                (unsigned long long)rows*DS4_N_VOCAB*20u);
            free(reference);free(actual);
            continue;
        }
        double total = 0;
        for (int rep = 0; rep < 5; rep++) {
            REQUIRE(ds4_session_load_snapshot(session, &base, err, sizeof(err)) == 0);
            t = now_sec();
            REQUIRE(ds41_graph_verify(&session->ds41_graph, &engine->model, &engine->weights,
                                      prompt.v + start, rows, &spec));
            if (rep) total += (now_sec() - t) * 1000.0;
        }
        fprintf(stderr, "verify %u rows: %.2f ms = %.2f decode steps, %.2f ms per row\n",
                rows, total / 4.0, total / 4.0 / step_ms, total / 4.0 / rows);
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    ds41_spec_free(&spec);
    ds4_session_snapshot_free(&base);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* Measurement: where one decoded token's GPU time goes, from 40 encodes of
 * one layer's own kernels on its real weights. */
static int time_stages(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0}, prefix = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    REQUIRE(text && ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(ds4_session_create(&session, engine, 4096) == 0);
    for (int i = 0; i < 1024; i++) ds4_tokens_push(&prefix, prompt.v[i]);
    REQUIRE(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    for (int i = 0; i < 8; i++)
        REQUIRE(ds4_session_eval(session, prompt.v[1024 + i], err, sizeof(err)) == 0);
    ds41_gpu_graph *g = &session->ds41_graph;
    const ds4_model *m = &engine->model;
    const ds4_weights *w = &engine->weights;
    (void)ds41_publish_measured_config(g, w);
    const char *names[] = {"attention projections (q_a, q_b, kv)", "attention core, index and window",
        "attention output (o_a, o_b)", "shared expert", "router and routed experts",
        "hyper-connection mixes and norms", "output head (once per token)"};
    for (int stage = 0; stage < 7; stage++) {
        double best = 1e30;
        for (int rep = 0; rep < 4; rep++) {
            const uint32_t pos = g->pos;
            REQUIRE(ds4_gpu_begin_commands());
            bool ok = true;
            const int loops = stage == 6 ? 8 : 40;
            for (int i = 0; ok && i < loops; i++) {
                const uint32_t il = stage == 6 ? 0 : (uint32_t)i;
                const ds4_layer_weights *l = &w->layer[il];
                switch (stage) {
                case 0: ok = ds41_attention_project(g, m, l); break;
                case 1: ok = ds41_attention(g, m, l, il, true); break;
                case 2: ok = ds41_attention_output(g, m, l, true); break;
                case 3: ok = ds41_shared_mid(g, m, l) &&
                            ds41_matmul(g->shared, m, l->ffn_down_shexp, g->shared_mid, true); break;
                case 4: {
                    uint64_t gate_row = 0, down_row = 0;
                    ok = tensor_nbytes(l->ffn_gate_exps->type, DS4_N_EMBD, &gate_row) &&
                        tensor_nbytes(l->ffn_down_exps->type, DS4_N_FF_EXP, &down_row) &&
                        ds41_matmul(g->route_logits, m, l->ffn_gate_inp, g->norm, false) &&
                        ds4_gpu_router_select_tensor(g->selected, g->route_weights, g->route_probs,
                            m->map, m->size, l->ffn_exp_probs_b->abs_offset, 0, 0, 0,
                            DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_EXPERT_WEIGHT_SCALE, 0, 0, true, false,
                            g->route_logits) &&
                        ds4_gpu_routed_moe_one_tensor(g->routed, g->gate, g->up, g->mid, g->experts,
                            m->map, m->size, l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset,
                            l->ffn_down_exps->abs_offset, l->ffn_gate_exps->type, l->ffn_down_exps->type,
                            gate_row * DS4_N_FF_EXP, gate_row, down_row * DS4_N_EMBD, down_row,
                            DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD, g->selected, g->route_weights,
                            DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->norm, NULL, il, true);
                    break;
                }
                case 5: ok = ds41_hc_mix(g, m, l, false) &&
                            ds41_hc_collapse_norm(g, m, l->attn_norm, g->residual, g->pre, false) &&
                            ds41_hc_mix(g, m, l, true) &&
                            ds41_hc_collapse_norm(g, m, l->ffn_norm, g->after_attn, g->attn_split, true); break;
                default: ok = ds41_graph_logits_encode(g, m, w); break;
                }
            }
            const double t = now_sec();
            REQUIRE(ok && ds4_gpu_end_commands());
            const double ms = (now_sec() - t) * 1000.0 / (stage == 6 ? 8.0 : 1.0);
            if (ms < best) best = ms;
            g->pos = pos;
        }
        fprintf(stderr, "%-44s %7.2f ms per token\n", names[stage], best);
    }
    /* How the Q8_0 row kernels scale: 40 layers' q_b projection per launch. */
    for (uint32_t rows = 1; rows <= 8; rows += rows < 2 ? 1 : 2) {
        double best = 1e30;
        for (int rep = 0; rep < 4; rep++) {
            REQUIRE(ds4_gpu_begin_commands());
            bool ok = true;
            for (uint32_t il = 0; ok && il < 40; il++)
                ok = ds41_matmul_batch(g->batch.q, m, w->layer[il].attn_q_b, g->batch.qr, rows, false);
            const double t = now_sec();
            REQUIRE(ok && ds4_gpu_end_commands());
            if ((now_sec() - t) * 1000.0 < best) best = (now_sec() - t) * 1000.0;
        }
        fprintf(stderr, "q_b projection, %u rows: %7.2f ms for 40 layers (%.2f ms per row)\n",
                rows, best, best / rows);
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

#endif

int main(int argc, char **argv) {
#ifdef __APPLE__
    if (argc == 4 && !strcmp(argv[2], "--stage-timing"))
        return time_stages(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--verify-timing"))
        return time_verify(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--short-prefill"))
        return check_short_prefill(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--verify-parity"))
        return check_verify_parity(argv[1], argv[3]);
#endif
    if (argc == 2 && !strcmp(argv[1], "--candidate-mask")) return check_candidate_mask();
    if (argc == 4 && !strcmp(argv[3], "--runtime")) return check_runtime(argv[1], argv[2], NULL);
    if (argc == 8 && !strcmp(argv[3], "--runtime-tp")) {
        char *end;
        long port = strtol(argv[5], &end, 10);
        if (*end || port < 1 || port > 65535) return 2;
        long gid = strtol(argv[7], &end, 10);
        if (*end || gid < 0 || gid > 255) return 2;
        ds4_tp_options opt = {.role = DS4_TP_LEADER, .requested = true,
            .listen_host = argv[4], .listen_port = (int)port,
            .transport = DS4_TP_TRANSPORT_RDMA, .rdma_device = argv[6],
            .rdma_gid_index = (int)gid, .rdma_gid_index_set = true};
        return check_runtime(argv[1], argv[2], &opt);
    }
    if (argc != 4) {
        fprintf(stderr, "usage: %s TARGET_GGUF DSPARK_GGUF OUTPUT_DIRECTORY\n", argv[0]);
        return 2;
    }
    int rc = 1;
    ds4_model target = {.fd = -1}, draft = {.fd = -1};
    ds4_weights tw = {0};
    ds4_dspark_weights dw = {0};
    ds41_dspark *d = NULL;
    ds4_gpu_tensor *hc = NULL;
    float *source = NULL;
    model_open(&target, argv[1], true, false);
    config_validate_model(&target);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    weights_bind(&tw, &target, false, 0, UINT32_MAX, true, false);
    model_open(&draft, argv[2], true, false);
    REQUIRE(support_model_checkpoint_compatible(&draft));
    ds4_dspark_summary summary = model_dspark_summary(&draft);
    dspark_weights_bind_optional(&dw, &draft, &summary);
    REQUIRE(dw.v41 && !dw.invalid_tensors && !dw.missing_tensors && !dw.metadata_errors);
    REQUIRE(ds4_gpu_init());
    const uint64_t offsets[] = {tw.token_embd->abs_offset, tw.output->abs_offset};
    const uint64_t sizes[] = {tw.token_embd->bytes, tw.output->bytes};
    REQUIRE(ds4_gpu_set_model_fd_for_map(target.fd, target.map));
    REQUIRE(ds4_gpu_set_model_map_spans(target.map, target.size, offsets, sizes, 2, tw.token_embd->bytes));
    REQUIRE(ds4_gpu_set_model_fd_for_map(draft.fd, draft.map));
    REQUIRE(ds4_gpu_set_model_map_range(draft.map, draft.size, draft.tensor_data_pos,
        draft.size - draft.tensor_data_pos, draft.max_tensor_bytes));
    REQUIRE(ds4_gpu_set_model_fd_for_map(target.fd, target.map));
    d = ds41_dspark_alloc(&dw);
    REQUIRE(d);
    enum { ROWS = 137 };
    const size_t elements = (size_t)ROWS * DS4_N_HC * DS4_N_EMBD;
    source = malloc(elements * sizeof(float));
    hc = ds4_gpu_tensor_alloc(elements * sizeof(float));
    REQUIRE(source && hc);
    for (unsigned stage = 0; stage < 3; stage++) {
        for (size_t i = 0; i < elements; i++)
            source[i] = ((int)((i * 17u + stage * 29u) % 257u) - 128) / 64.0f;
        REQUIRE(ds4_gpu_tensor_write(hc, 0, source, elements * sizeof(float)));
        REQUIRE(ds4_gpu_begin_commands());
        REQUIRE(ds41_dspark_capture(d, 37 + stage, 0, hc, ROWS));
        REQUIRE(ds4_gpu_end_commands());
    }
    REQUIRE(ds41_dspark_forward(d, &target, &tw, &draft, ROWS, 100, NULL));
    REQUIRE(d->kv_valid && d->kv_end == ROWS);
    REQUIRE(dump_tensor(argv[3], "main_x", d->main_x, DS41_DRAFT_WINDOW * DS4_N_EMBD));
    for (unsigned stage = 0; stage < 3; stage++) {
        char name[32];
        snprintf(name, sizeof(name), "window%u", stage);
        REQUIRE(dump_tensor(argv[3], name, d->window[stage], (DS41_DRAFT_WINDOW + DS41_DRAFT_ROWS) * DS4_N_HEAD_DIM));
    }
    REQUIRE(dump_tensor(argv[3], "hidden", d->batch.x, DS41_DRAFT_ROWS * DS4_N_EMBD));
    REQUIRE(dump_tensor(argv[3], "norm", d->batch.norm, DS41_DRAFT_ROWS * DS4_N_EMBD));
    REQUIRE(dump_tensor(argv[3], "logits", d->logits, DS41_DRAFT_ROWS * DS4_N_VOCAB));
    REQUIRE(ds4_gpu_tensor_read(d->logits, 0, d->host_logits, DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float)));
    REQUIRE(ds41_dspark_forward(d, &target, &tw, &draft, ROWS, 100, NULL));
    float *second = malloc(DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float));
    REQUIRE(second);
    bool repeated = ds4_gpu_tensor_read(d->logits, 0, second, DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float)) &&
        !memcmp(second, d->host_logits, DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float));
    free(second);
    REQUIRE(repeated);
    int previous = 100;
    for (unsigned row = 0; row < DS41_DRAFT_ROWS; row++) {
        float confidence;
        REQUIRE(ds41_dspark_markov(d, &draft, row, previous, &confidence));
        char name[32];
        snprintf(name, sizeof(name), "markov%u", row);
        REQUIRE(dump_tensor(argv[3], name, d->markov_logits, DS4_N_VOCAB));
        snprintf(name, sizeof(name), "confidence%u", row);
        REQUIRE(dump_tensor(argv[3], name, d->confidence, 1));
        previous = (int)dspark_argmax_f32(d->host_logits + (uint64_t)row * DS4_N_VOCAB, DS4_N_VOCAB);
        fprintf(stderr, "draft[%u]=%d confidence=%.7g\n", row, previous, confidence);
    }
    REQUIRE(dump_tensor(argv[3], "biased_logits", d->logits, DS41_DRAFT_ROWS * DS4_N_VOCAB));
    REQUIRE(ds4_gpu_begin_commands());
    const float pre[5][4] = {{1}, {1}, {1}, {1}, {1}};
    REQUIRE(ds4_gpu_tensor_write(d->batch.pre, 0, pre, sizeof(pre)));
    for (uint32_t row = 0; row < DS41_DRAFT_ROWS; row++) {
        const uint64_t bytes = (uint64_t)DS4_N_EMBD * DS4_N_HC * sizeof(float);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(d->batch.residual, row * bytes, bytes);
        bool ok = out && ds4_gpu_embed_token_hc_tensor(out, target.map, target.size,
            tw.token_embd->abs_offset, DS4_N_VOCAB, row ? dw.noise_token_id : 100,
            DS4_N_EMBD, DS4_N_HC);
        ds4_gpu_tensor_free(out);
        REQUIRE(ok);
    }
    REQUIRE(ds4_gpu_end_commands());
    REQUIRE(dump_tensor(argv[3], "embedded", d->batch.residual, 5u * 4u * DS4_N_EMBD));
    for (unsigned stage = 0; stage < 3; stage++) {
        REQUIRE(ds4_gpu_begin_commands());
        REQUIRE(ds41_dspark_stage(d, &draft, stage, ROWS));
        REQUIRE(ds4_gpu_end_commands());
#define STAGE_DUMP(field, count) do { \
    char name[64]; snprintf(name, sizeof(name), "stage%u_" #field, stage); \
    REQUIRE(dump_tensor(argv[3], name, d->batch.field, (count))); \
} while (0)
        STAGE_DUMP(q, 5u * DS4_N_HEAD * DS4_N_HEAD_DIM);
        STAGE_DUMP(kv, 5u * DS4_N_HEAD_DIM);
        STAGE_DUMP(heads, 5u * DS4_N_HEAD * DS4_N_HEAD_DIM);
        STAGE_DUMP(after_attn, 5u * DS4_N_HC * DS4_N_EMBD);
        STAGE_DUMP(norm, 5u * DS4_N_EMBD);
        STAGE_DUMP(route_logits, 5u * dw.n_experts);
        STAGE_DUMP(routed, 5u * DS4_N_EMBD);
        STAGE_DUMP(shared, 5u * DS4_N_EMBD);
        STAGE_DUMP(ffn_split, 5u * 24u);
        STAGE_DUMP(residual, 5u * DS4_N_HC * DS4_N_EMBD);
#undef STAGE_DUMP
    }
    puts("V4.1 draft capture, ring seed and forward: finite and repeatable");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    ds4_gpu_tensor_free(hc);
    ds41_dspark_free(d);
    ds4_gpu_cleanup();
    model_close(&draft); model_close(&target);
    free(source);
    return rc;
}
