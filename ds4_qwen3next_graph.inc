/* CUDA graph: token-at-a-time prefill and decode share one path.
 * FP16 KV storage, FP32 residuals, and no host-side expert selection. */
#include "ds4_qwen3next_gpu.h"
typedef struct {
    ds4_gpu_tensor *x, *norm, *q, *k, *v, *attn, *block, *router;
    ds4_gpu_tensor *selected, *weights, *mid, *part, *partial, *logits, *greedy, *argmax_scratch;
    ds4_gpu_tensor *qg, *qkvz, *qkv, *z, *ba, *alpha, *beta, *shared_gate;
} qwen3next_workspace;
typedef struct {
    qwen3next_workspace tier[DS4_MAX_GPUS];
    ds4_qwen3next_replay *replay[DS4_MAX_GPUS];
    int n_tiers;
    ds4_gpu_tensor *kc[DS4_MAX_LAYER], *vc[DS4_MAX_LAYER];
    ds4_gpu_tensor *state[DS4_MAX_LAYER], *history[DS4_MAX_LAYER];
    uint32_t pos, cap;
    int greedy_token;
    bool logits_pending; /* Device logits are newer than the session host copy. */
} qwen3next_graph;

/* Contiguous layer groups use every selected GPU; only the residual vector
 * crosses a boundary. This combines VRAM without tensor-parallel reductions. */
static int qwen3next_layer_tier(uint32_t layer, int n_tiers) {
    return (int)(layer * (uint32_t)n_tiers / DS4_N_LAYER);
}
static void qwen3next_graph_free(qwen3next_graph *g) {
    for (int t = 0; t < g->n_tiers; t++) {
        (void)ds4_gpu_set_current_device(t);
        (void)ds4_gpu_synchronize();
        ds4_gpu_qwen3next_replay_free(g->replay[t]);
        qwen3next_workspace *a = &g->tier[t];
#define Q3_FREE(f) ds4_gpu_tensor_free(a->f)
        Q3_FREE(x); Q3_FREE(norm); Q3_FREE(q); Q3_FREE(k); Q3_FREE(v);
        Q3_FREE(attn); Q3_FREE(block); Q3_FREE(router); Q3_FREE(selected);
        Q3_FREE(qg); Q3_FREE(qkvz); Q3_FREE(qkv); Q3_FREE(z);
        Q3_FREE(ba); Q3_FREE(alpha); Q3_FREE(beta); Q3_FREE(shared_gate);
        Q3_FREE(weights); Q3_FREE(mid); Q3_FREE(part); Q3_FREE(partial); Q3_FREE(logits); Q3_FREE(greedy); Q3_FREE(argmax_scratch);
#undef Q3_FREE
    }
    for (uint32_t i = 0; i < DS4_MAX_LAYER; i++) {
        ds4_gpu_tensor_free(g->kc[i]); ds4_gpu_tensor_free(g->vc[i]);
        ds4_gpu_tensor_free(g->state[i]); ds4_gpu_tensor_free(g->history[i]);
    }
    memset(g, 0, sizeof(*g));
}
static bool qwen3next_graph_alloc(qwen3next_graph *g, uint32_t cap, int n_tiers) {
    memset(g, 0, sizeof(*g));
    g->cap = cap;
    g->n_tiers = n_tiers;
    for (int t = 0; t < n_tiers; t++) {
        qwen3next_workspace *a = &g->tier[t];
#define Q3_ALLOC(f, n) do { a->f = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)(n) * 4); if (!a->f) goto fail; } while (0)
        Q3_ALLOC(qg, 8192); Q3_ALLOC(qkvz, 12288); Q3_ALLOC(qkv, 8192); Q3_ALLOC(z, 4096);
        Q3_ALLOC(ba, 64); Q3_ALLOC(alpha, 32); Q3_ALLOC(beta, 32); Q3_ALLOC(shared_gate, 1);
        Q3_ALLOC(x, DS4_N_EMBD); Q3_ALLOC(norm, DS4_N_EMBD);
        Q3_ALLOC(q, DS4_N_HEAD * DS4_N_HEAD_DIM);
        Q3_ALLOC(k, DS4_N_HEAD_KV * DS4_N_HEAD_DIM); Q3_ALLOC(v, DS4_N_HEAD_KV * DS4_N_HEAD_DIM);
        Q3_ALLOC(attn, DS4_N_HEAD * DS4_N_HEAD_DIM); Q3_ALLOC(block, DS4_N_EMBD);
        Q3_ALLOC(router, DS4_N_EXPERT); Q3_ALLOC(selected, DS4_N_EXPERT_USED);
        Q3_ALLOC(weights, DS4_N_EXPERT_USED); Q3_ALLOC(mid, (DS4_N_EXPERT_USED + 1) * DS4_N_FF_EXP);
        Q3_ALLOC(part, (DS4_N_EXPERT_USED + 1) * DS4_N_EMBD);
        Q3_ALLOC(partial, (uint64_t)DS4_N_HEAD * ((cap + 255u) / 256u) * (DS4_N_HEAD_DIM + 2u));
        Q3_ALLOC(logits, DS4_N_VOCAB); Q3_ALLOC(greedy,1); Q3_ALLOC(argmax_scratch,512);
#undef Q3_ALLOC
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint64_t bytes = (uint64_t)cap * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 2;
        int t = qwen3next_layer_tier(il, n_tiers);
        if ((il + 1) % 4 != 0) {
            g->state[il] = ds4_gpu_tensor_alloc_ptr_on(t, 32u*128*128*4);
            g->history[il] = ds4_gpu_tensor_alloc_ptr_on(t, 3u*8192*4);
            if (!g->state[il] || !g->history[il]) goto fail;
            continue;
        }
        g->kc[il] = ds4_gpu_tensor_alloc_ptr_on(t, bytes);
        g->vc[il] = ds4_gpu_tensor_alloc_ptr_on(t, bytes);
        if (!g->kc[il] || !g->vc[il]) goto fail;
    }
    return true;
fail:
    qwen3next_graph_free(g);
    return false;
}
static bool qwen3next_mm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                       const ds4_model *m, const ds4_tensor *w) {
    return ds4_gpu_qwen4_dense_mm_tensor(out, x, m->map, m->size, w->abs_offset,
                                         w->type, 1, w->dim[0], w->dim[1]) != 0;
}
static bool qwen3next_norm(ds4_gpu_tensor *out, ds4_gpu_tensor *x,
                         const ds4_model *m, const ds4_tensor *w) {
    return ds4_gpu_qwen3next_norm(out, x, NULL, m->map, m->size,
                                w->abs_offset, DS4_RMS_EPS) != 0;
}
static bool qwen3next_mm_group(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1, ds4_gpu_tensor *out2,
                              const ds4_gpu_tensor *x, const ds4_model *m,
                              const ds4_tensor *w0, const ds4_tensor *w1, const ds4_tensor *w2) {
    const ds4_tensor *weights[] = {w0,w1,w2};
    ds4_gpu_tensor *outputs[] = {out0,out1,out2};
    ds4_qwen3next_projection projections[3];
    const unsigned count = w2 ? 3 : 2;
    for (unsigned i = 0; i < count; i++) {
        if (weights[i]->dim[0] != w0->dim[0]) return false;
        projections[i] = (ds4_qwen3next_projection){outputs[i], weights[i]->abs_offset,
            weights[i]->type, (uint32_t)weights[i]->dim[1]};
    }
    return ds4_gpu_qwen3next_project(projections,count,x,m->map,m->size,w0->dim[0]) != 0;
}
static bool qwen3next_graph_tier(qwen3next_graph *g, const ds4_model *m,
                               const ds4_weights *w, int tier) {
    qwen3next_workspace *a = &g->tier[tier];
    bool ok = true;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (qwen3next_layer_tier(il, g->n_tiers) != tier) continue;
        const ds4_layer_weights *l = &w->layer[il];
        ok = qwen3next_norm(a->norm, a->x, m, l->attn_norm);
        if ((il + 1) % 4 == 0) {
            ok = ok && qwen3next_mm_group(a->qg, a->k, a->v, a->norm, m,
                     l->attn_q, l->attn_k, l->attn_v) &&
                 ds4_gpu_qwen3next_attention(a->attn, a->q, a->qg, a->k, a->v,
                     g->kc[il], g->vc[il], a->partial, m->map, m->size,
                     l->attn_q_norm->abs_offset, l->attn_k_norm->abs_offset,
                     DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, g->pos, g->cap,
                     DS4_N_ROT, DS4_RMS_EPS, DS4_ROPE_FREQ_BASE) &&
                 qwen3next_mm(a->block, a->attn, m, l->attn_output);
        } else {
            /* A changed prefix must replay from zero: recurrent state cannot
             * be rewound by truncating a token index as an ordinary KV can. */
            if (!g->pos) ok = ok && ds4_gpu_tensor_fill_f32(g->state[il], 0, 32u*128*128) &&
                ds4_gpu_tensor_fill_f32(g->history[il], 0, 3u*8192);
            if (l->lin_gate) {
                ok = ok && qwen3next_mm_group(a->qkv, a->z, a->ba, a->norm, m,
                     l->lin_qkv, l->lin_gate, l->lin_beta);
            } else ok = ok && qwen3next_mm_group(a->qkvz, a->ba, NULL, a->norm, m,
                     l->lin_qkv, l->lin_beta, NULL);
            ok = ok && ds4_gpu_qwen3next_prepare_recurrent(a->qkv, a->z, a->alpha, a->beta, a->qkvz, a->ba,
                     !l->lin_gate, g->history[il], m->map, m->size, l->lin_conv->abs_offset,
                     l->lin_a->abs_offset, l->lin_dt_bias->abs_offset) &&
                 ds4_gpu_qwen3next_recurrent(a->attn, g->state[il], a->qkv, a->alpha, a->beta,
                     a->z, m->map, m->size, l->lin_norm->abs_offset, DS4_RMS_EPS) &&
                 qwen3next_mm(a->block, a->attn, m, l->lin_out);
        }
        ok = ok &&
             ds4_gpu_qwen3next_norm(a->norm, a->x, a->block, m->map, m->size,
                                   l->ffn_norm->abs_offset, DS4_RMS_EPS) &&
             qwen3next_mm(a->router, a->norm, m, l->ffn_gate_inp) &&
             ds4_gpu_qwen4_router_topk_tensor(a->selected, a->weights, a->router, a->norm,
                 m->map, m->size, l->ffn_gate_inp_shexp->abs_offset, l->ffn_gate_inp_shexp->type,
                 DS4_N_EMBD, a->shared_gate, 1, DS4_N_EXPERT, DS4_N_EXPERT_USED) &&
             ds4_gpu_qwen4_moe_mid_tensor(a->mid, a->norm, a->selected, m->map, m->size,
                 l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, l->ffn_gate_exps->type,
                 DS4_N_EXPERT, 1, DS4_N_EXPERT_USED, DS4_N_EMBD, DS4_N_FF_EXP,
                 l->ffn_gate_shexp->abs_offset, l->ffn_up_shexp->abs_offset, l->ffn_gate_shexp->type) &&
             ds4_gpu_qwen3next_moe_down_residual(a->x, a->part, a->mid, a->selected,
                 a->weights, a->shared_gate, m->map, m->size,
                 l->ffn_down_exps->abs_offset, l->ffn_down_exps->type,
                 l->ffn_down_shexp->abs_offset, l->ffn_down_shexp->type);
    }
    if (ok && tier == g->n_tiers - 1) ok = qwen3next_norm(a->norm, a->x, m, w->output_norm) &&
                 qwen3next_mm(a->logits, a->norm, m, w->output) &&
                 ds4_gpu_qwen3next_argmax(a->greedy,a->argmax_scratch,a->logits,DS4_N_VOCAB);
    return ok;
}
static bool qwen3next_graph_step(qwen3next_graph *g, const ds4_model *m,
                               const ds4_weights *w, int token) {
    if (token < 0 || (uint32_t)token >= DS4_N_VOCAB || g->pos >= g->cap) return false;
    qwen3next_workspace *a = &g->tier[0];
    bool ok = ds4_gpu_set_current_device(0) == 0 && ds4_gpu_begin_commands() &&
        ds4_gpu_qwen3next_embed(a->x, m->map, m->size, w->token_embd->abs_offset,
                               w->token_embd->type, DS4_N_EMBD, (uint32_t)token);
    for (int t = 0; ok && t < g->n_tiers; t++) {
        qwen3next_workspace *next = &g->tier[t];
        if (next != a) {
            ok = ds4_gpu_tensor_copy_xdev_default(next->x, a->x, (uint64_t)DS4_N_EMBD * 4) &&
                 ds4_gpu_set_current_device(t) == 0;
            if (!ok) break;
            a = next;
        }
        const int replay = ds4_gpu_qwen3next_replay_begin(&g->replay[t], g->pos);
        if (replay == -2) { ok = false; break; }
        if (replay != 1) ok = qwen3next_graph_tier(g, m, w, t);
        if (replay == 0) {
            const int result = ds4_gpu_qwen3next_replay_end(g->replay[t], ok);
            if (result < 0) ok = false;
            /* A failed capture executed no kernels. Re-encode only a
             * successful body; a failed body invalidates the session. */
            else if (!result && ok) ok = qwen3next_graph_tier(g, m, w, t);
        }
    }
    /* Drain even a partial failed step before callers reset or free buffers. */
    bool complete = true;
    for (int t = 0; t < g->n_tiers; t++) {
        if (ds4_gpu_set_current_device(t) != 0 || !ds4_gpu_end_commands()) complete = false;
    }
    if (!ok || !complete || !ds4_gpu_tensor_read(a->greedy, 0, &g->greedy_token, sizeof(g->greedy_token))) return false;
    g->logits_pending = true;
    g->pos++;
    return true;
}

/* Install only each tier's layers, keeping embeddings on the first and the
 * output head on the last. Explicit budgets include one or more KV sessions;
 * reject an undersized even split instead of silently spilling weights. */
static bool qwen3next_prepare_tiers(const ds4_model *m, const ds4_weights *w,
                                const ds4_gpu_config *cfg, uint32_t ctx, int sessions) {
    const bool multi = cfg->n_gpus > 1;
    if (multi && !ds4_gpu_register_model_map_no_copy(m->map, m->size)) return false;
    for (int t = 0; t < cfg->n_gpus; t++) {
        ds4_tensor_range ranges[DS4_MAX_LAYER * 22 + 3];
        int n = 0, layers = 0, attention_layers = 0;
        uint64_t bytes = 0;
#define Q3_RANGE(p) do { const ds4_tensor *v = (p); \
    ranges[n++] = (ds4_tensor_range){v->abs_offset, v->bytes, cfg->device_indices[t]}; \
    bytes += v->bytes; } while (0)
        if (t == 0) Q3_RANGE(w->token_embd);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (qwen3next_layer_tier(il, cfg->n_gpus) != t) continue;
            layers++;
            const ds4_layer_weights *l = &w->layer[il];
            Q3_RANGE(l->attn_norm);
            if ((il + 1) % 4 == 0) {
                attention_layers++;
                Q3_RANGE(l->attn_q); Q3_RANGE(l->attn_k); Q3_RANGE(l->attn_v);
                Q3_RANGE(l->attn_q_norm); Q3_RANGE(l->attn_k_norm); Q3_RANGE(l->attn_output);
            } else {
                Q3_RANGE(l->lin_qkv); if (l->lin_gate) Q3_RANGE(l->lin_gate);
                Q3_RANGE(l->lin_beta); Q3_RANGE(l->lin_conv); Q3_RANGE(l->lin_a);
                Q3_RANGE(l->lin_dt_bias); Q3_RANGE(l->lin_norm); Q3_RANGE(l->lin_out);
            }
            Q3_RANGE(l->ffn_gate_inp_shexp); Q3_RANGE(l->ffn_gate_shexp);
            Q3_RANGE(l->ffn_up_shexp); Q3_RANGE(l->ffn_down_shexp);
            Q3_RANGE(l->ffn_norm); Q3_RANGE(l->ffn_gate_inp); Q3_RANGE(l->ffn_gate_exps);
            Q3_RANGE(l->ffn_up_exps); Q3_RANGE(l->ffn_down_exps);
        }
        if (t == cfg->n_gpus - 1) { Q3_RANGE(w->output_norm); Q3_RANGE(w->output); }
#undef Q3_RANGE
        uint64_t kv = (uint64_t)attention_layers * ctx * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 4 +
            (uint64_t)(layers - attention_layers) * (32u*128*128 + 3u*8192) * 4;
        uint64_t reserve = cfg->safety_margin_bytes + (uint64_t)sessions * (kv + 64ull*1024*1024);
        if (reserve > cfg->vram_bytes[t] || bytes > cfg->vram_bytes[t] - reserve) {
            fprintf(stderr, "ds4: Qwen3-Next GPU%d needs %.2f GiB for its %d layers, context and reserve; budget %.2f GiB\n",
                cfg->device_indices[t], (double)(bytes + reserve)/(1024.*1024.*1024.), layers,
                (double)cfg->vram_bytes[t]/(1024.*1024.*1024.));
            return false;
        }
        fprintf(stderr, "ds4: Qwen3-Next GPU%d: %d layers, %.2f GiB weights, %.2f GiB KV/recurrent state per session\n",
            cfg->device_indices[t], layers, (double)bytes/(1024.*1024.*1024.), (double)kv/(1024.*1024.*1024.));
        if (multi && ds4_gpu_device_cache_tensors(cfg->device_indices[t], ranges, n) != 0) return false;
    }
    return ds4_gpu_set_current_device(0) == 0;
}
