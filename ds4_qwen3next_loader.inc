/* Qwen3-Coder-Next 80B-A3B: three DeltaNet layers followed by gated GQA.
 * GGUF norms already contain 1 + HF gamma; ssm_a contains -exp(A_log). */
static void config_validate_qwen3next_model(const ds4_model *m) {
    ds4_kv *split = model_find_kv(m, "split.count");
    if (split) {
        ds4_cursor c = cursor_at(m, split->value_pos);
        uint16_t count = 0;
        if (split->type != GGUF_VALUE_UINT16 || !cursor_read(&c, &count, sizeof(count)) || count > 1)
            ds4_die("Qwen3-Next: merge GGUF shards first with llama-gguf-split --merge");
    }
    g_ds4_shape = (ds4_shape){
        .name = "Qwen3-Coder-Next-80B-A3B", .family = DS4_MODEL_FAMILY_QWEN3_NEXT,
        .variant = DS4_VARIANT_QWEN3_NEXT, .n_layer = 48, .n_embd = 2048,
        .n_vocab = 151936, .n_head = 16, .n_head_kv = 2, .n_head_dim = 256,
        .n_value_dim = 256, .n_rot = 64, .n_expert = 512, .n_expert_used = 10,
        .n_expert_shared = 1, .n_ff_exp = 512, .n_hc = 1, .rms_eps = 1e-6f,
        .n_lin_k_head = 16, .n_lin_v_head = 32, .n_lin_head_dim = 128,
        .n_lin_conv = 4, .n_full_attn_interval = 4,
        .rope_freq_base = 5000000.0f, .rope_scale_factor = 1.0f,
        .rope_orig_ctx = 262144, .expert_weight_scale = 1.0f,
    };
#define QN_U32(key, value) config_expect_u32("qwen3next." key, required_u32(m, "qwen3next." key), value)
    QN_U32("block_count", DS4_N_LAYER); QN_U32("embedding_length", DS4_N_EMBD);
    QN_U32("attention.head_count", DS4_N_HEAD); QN_U32("attention.head_count_kv", DS4_N_HEAD_KV);
    QN_U32("attention.key_length", DS4_N_HEAD_DIM); QN_U32("attention.value_length", DS4_N_HEAD_DIM);
    QN_U32("rope.dimension_count", DS4_N_ROT);
    QN_U32("expert_count", DS4_N_EXPERT); QN_U32("expert_used_count", DS4_N_EXPERT_USED);
    QN_U32("expert_feed_forward_length", DS4_N_FF_EXP);
    QN_U32("expert_shared_feed_forward_length", DS4_N_FF_EXP);
    QN_U32("ssm.conv_kernel", DS4_N_LIN_CONV); QN_U32("ssm.state_size", DS4_N_LIN_HEAD_DIM);
    QN_U32("ssm.group_count", DS4_N_LIN_K_HEAD); QN_U32("ssm.time_step_rank", DS4_N_LIN_V_HEAD);
    QN_U32("ssm.inner_size", DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM);
#undef QN_U32
    uint32_t interval = 4;
    model_get_u32(m, "qwen3next.full_attention_interval", &interval);
    config_expect_u32("full_attention_interval", interval, 4);
    /* Reject alternative recurrent layouts instead of silently using a 3:1 pattern. */
    if (model_find_kv(m, "qwen3next.attention.recurrent_layers"))
        ds4_die("Qwen3-Next: explicit recurrent layer masks are not supported");
    ds4_array_ref tokens;
    if (!model_get_array(m, "tokenizer.ggml.tokens", &tokens) || tokens.len != DS4_N_VOCAB)
        ds4_die("Qwen3-Next: expected the Coder vocabulary");
    config_expect_f32("rope.freq_base", required_f32(m, "qwen3next.rope.freq_base"), DS4_ROPE_FREQ_BASE);
    config_expect_epsilon("rms_epsilon", required_f32(m, "qwen3next.attention.layer_norm_rms_epsilon"), DS4_RMS_EPS);
    float factor = 1.0f;
    model_get_f32_compat(m, "qwen3next.rope.scaling.factor", &factor);
    config_expect_f32("rope.scaling.factor", factor, 1.0f);
    ds4_str scaling = {0}, pre = {0};
    if (model_get_string(m, "qwen3next.rope.scaling.type", &scaling) && !ds4_streq(scaling, "none"))
        ds4_die("Qwen3-Next: scaled RoPE is not supported");
    if (!model_get_string(m, "tokenizer.ggml.pre", &pre) || !ds4_streq(pre, "qwen2"))
        ds4_die("Qwen3-Next: expected the qwen2 pre-tokenizer");
    uint32_t ctx = required_u32(m, "qwen3next.context_length");
    if (!ctx || ctx > 262144) ds4_die("Qwen3-Next: unsupported native context length");
    g_ds4_shape.rope_orig_ctx = ctx;
}

static void qwen3next_tensor_layout(const ds4_tensor *t, uint32_t rank,
                                   uint64_t a, uint64_t b, uint64_t c) {
    switch (t->type) {
        case 0: case 1: case 2: case 8: case 10: case 12: case 13: case 14:
        case 16: case 30: case 39: break;
        default: ds4_die("Qwen3-Next: unsupported weight quantization");
    }
    const gguf_type_info *info = tensor_type(t->type);
    if (!info || a % info->block_elems) ds4_die("Qwen3-Next: unaligned quantized row");
    tensor_expect_layout(t, t->type, rank, a, b, c);
}

static void weights_bind_qwen3next(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output = model_find_tensor(m, "output.weight");
    if (!w->output) w->output = w->token_embd;
    qwen3next_tensor_layout(w->token_embd, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    qwen3next_tensor_layout(w->output, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    tensor_expect_layout(w->output_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_weights *l = &w->layer[il];
#define QN_BIND(field, suffix) l->field = required_tensorf(m, "blk.%u." suffix, il)
#define QN_MAT(field, a, b) qwen3next_tensor_layout(l->field, 2, a, b, 0)
#define QN_VEC(field, n) tensor_expect_layout(l->field, DS4_TENSOR_F32, 1, n, 0, 0)
        QN_BIND(attn_norm, "attn_norm.weight"); QN_BIND(ffn_norm, "post_attention_norm.weight");
        QN_VEC(attn_norm, DS4_N_EMBD); QN_VEC(ffn_norm, DS4_N_EMBD);
        if ((il + 1) % 4 == 0) {
            QN_BIND(attn_q, "attn_q.weight"); QN_BIND(attn_k, "attn_k.weight"); QN_BIND(attn_v, "attn_v.weight");
            QN_BIND(attn_q_norm, "attn_q_norm.weight"); QN_BIND(attn_k_norm, "attn_k_norm.weight");
            QN_BIND(attn_output, "attn_output.weight");
            QN_VEC(attn_q_norm, 256); QN_VEC(attn_k_norm, 256);
            QN_MAT(attn_q, 2048, 8192); QN_MAT(attn_k, 2048, 512); QN_MAT(attn_v, 2048, 512);
            QN_MAT(attn_output, 4096, 2048);
        } else {
            /* Legacy GGUF packs Q/K/V/Z per key head; newer GGUF separates Z. */
            char name[64];
            snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", il);
            l->lin_qkv = model_find_tensor(m, name);
            if (l->lin_qkv) {
                QN_BIND(lin_gate, "attn_gate.weight"); QN_MAT(lin_qkv, 2048, 8192); QN_MAT(lin_gate, 2048, 4096);
            } else { QN_BIND(lin_qkv, "ssm_in.weight"); QN_MAT(lin_qkv, 2048, 12288); }
            QN_BIND(lin_beta, "ssm_ba.weight"); QN_MAT(lin_beta, 2048, 64);
            QN_BIND(lin_conv, "ssm_conv1d.weight");
            tensor_expect_layout(l->lin_conv, DS4_TENSOR_F32, 2, 4, 8192, 0);
            QN_BIND(lin_a, "ssm_a"); QN_VEC(lin_a, 32);
            QN_BIND(lin_dt_bias, "ssm_dt.bias"); QN_VEC(lin_dt_bias, 32);
            QN_BIND(lin_norm, "ssm_norm.weight"); QN_VEC(lin_norm, 128);
            QN_BIND(lin_out, "ssm_out.weight"); QN_MAT(lin_out, 4096, 2048);
        }
        QN_BIND(ffn_gate_inp, "ffn_gate_inp.weight"); QN_MAT(ffn_gate_inp, 2048, 512);
        QN_BIND(ffn_gate_exps, "ffn_gate_exps.weight"); QN_BIND(ffn_up_exps, "ffn_up_exps.weight");
        QN_BIND(ffn_down_exps, "ffn_down_exps.weight");
        qwen3next_tensor_layout(l->ffn_gate_exps, 3, 2048, 512, 512);
        qwen3next_tensor_layout(l->ffn_up_exps, 3, 2048, 512, 512);
        qwen3next_tensor_layout(l->ffn_down_exps, 3, 512, 2048, 512);
        QN_BIND(ffn_gate_shexp, "ffn_gate_shexp.weight"); QN_MAT(ffn_gate_shexp, 2048, 512);
        QN_BIND(ffn_up_shexp, "ffn_up_shexp.weight"); QN_MAT(ffn_up_shexp, 2048, 512);
        QN_BIND(ffn_down_shexp, "ffn_down_shexp.weight"); QN_MAT(ffn_down_shexp, 512, 2048);
        QN_BIND(ffn_gate_inp_shexp, "ffn_gate_inp_shexp.weight");
        qwen3next_tensor_layout(l->ffn_gate_inp_shexp, 1, 2048, 0, 0);
        if (l->ffn_gate_exps->type != l->ffn_up_exps->type ||
            l->ffn_gate_shexp->type != l->ffn_up_shexp->type)
            ds4_die("Qwen3-Next: gate and up experts must use the same quantization");
#undef QN_BIND
#undef QN_MAT
#undef QN_VEC
    }
}
