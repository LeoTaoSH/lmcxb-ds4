/* Muse Spark Open helpers.  Activations are f32; K/V cache rows are f16.
 * The attention and expert matmuls reuse the Qwen4 kernels, while these
 * kernels implement Muse-specific normalization, gating and routing. */

static inline float muse_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = exp(x);
    return e / (1.0f + e);
}

struct ds4_metal_args_muse_attn_prep {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t use_rope;
    float rope_base;
    float eps;
    uint32_t pad0;
    uint32_t pad1;
};

kernel void kernel_muse_attn_prep(
        constant ds4_metal_args_muse_attn_prep &args,
        device const float *qproj,
        device const float *kproj,
        device const float *vproj,
        device const float *gateproj,
        device const float *q_norm,
        device const float *k_norm,
        device float *q_out,
        device float *gate_out,
        device half *k_cache,
        device half *v_cache,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.x;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint D = args.head_dim;
    const uint npt = D / 32u;
    const uint pos = args.pos0 + tok;
    threadgroup float row[64];

    if (slot < args.n_head) {
        device const float *src = qproj + ((uint64_t)tok * args.n_head + slot) * D;
        float ss = 0.0f;
        for (uint i = 0; i < npt; i++) {
            const float v = src[lane * npt + i];
            row[lane * npt + i] = v;
            ss += v * v;
        }
        ss = simd_sum(ss);
        const float r = rsqrt(ss / (float)D + args.eps);
        for (uint i = 0; i < npt; i++) row[lane * npt + i] *= r * q_norm[lane * npt + i];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (args.use_rope && lane < args.n_rot / 2u) {
            const uint i = 2u * lane;
            const uint j = i + 1u;
            const float theta = (float)pos * pow(args.rope_base, -2.0f * (float)lane / (float)args.n_rot);
            const float c = cos(theta), s = sin(theta);
            const float x0 = row[i], x1 = row[j];
            row[i] = x0 * c - x1 * s;
            row[j] = x0 * s + x1 * c;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        device float *dst = q_out + ((uint64_t)tok * args.n_head + slot) * D;
        device const float *gs = gateproj + ((uint64_t)tok * args.n_head + slot) * D;
        device float *gd = gate_out + ((uint64_t)tok * args.n_head + slot) * D;
        for (uint i = 0; i < npt; i++) {
            const uint j = lane * npt + i;
            dst[j] = row[j];
            gd[j] = gs[j];
        }
        return;
    }

    const uint h = slot - args.n_head;
    if (h >= args.n_head_kv || pos >= args.cache_cap) return;
    device const float *src = kproj + ((uint64_t)tok * args.n_head_kv + h) * D;
    float ss = 0.0f;
    for (uint i = 0; i < npt; i++) {
        const float v = src[lane * npt + i];
        row[lane * npt + i] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    const float r = rsqrt(ss / (float)D + args.eps);
    for (uint i = 0; i < npt; i++) row[lane * npt + i] *= r * k_norm[lane * npt + i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (args.use_rope && lane < args.n_rot / 2u) {
        const uint i = 2u * lane;
        const uint j = i + 1u;
        const float theta = (float)pos * pow(args.rope_base, -2.0f * (float)lane / (float)args.n_rot);
        const float c = cos(theta), s = sin(theta);
        const float x0 = row[i], x1 = row[j];
        row[i] = x0 * c - x1 * s;
        row[j] = x0 * s + x1 * c;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device half *kd = k_cache + ((uint64_t)pos * args.n_head_kv + h) * D;
    device half *vd = v_cache + ((uint64_t)pos * args.n_head_kv + h) * D;
    device const float *vs = vproj + ((uint64_t)tok * args.n_head_kv + h) * D;
    for (uint i = 0; i < npt; i++) {
        const uint j = lane * npt + i;
        kd[j] = (half)row[j];
        vd[j] = (half)vs[j];
    }
}

struct ds4_metal_args_muse_rows {
    uint32_t n_tokens;
    uint32_t dim;
    float alpha;
    float beta;
};

kernel void kernel_muse_sigmoid_mul(
        constant ds4_metal_args_muse_rows &args,
        device const float *x,
        device const float *gate,
        device float *out,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t n = (uint64_t)args.n_tokens * args.dim;
    if ((uint64_t)gid < n) out[gid] = x[gid] * muse_sigmoid(gate[gid]);
}

kernel void kernel_muse_residual_blend(
        constant ds4_metal_args_muse_rows &args,
        device const float *a,
        device const float *b,
        device const float *alpha,
        device const float *beta,
        device float *out,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t n = (uint64_t)args.n_tokens * args.dim;
    if ((uint64_t)gid < n) {
        const uint col = gid % args.dim;
        out[gid] = a[gid] * alpha[col] + b[gid] * beta[col];
    }
}

kernel void kernel_muse_weight_mul(
        constant ds4_metal_args_muse_rows &args,
        device const float *x,
        device const float *weight,
        device float *out,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t n = (uint64_t)args.n_tokens * args.dim;
    if ((uint64_t)gid < n) out[gid] = x[gid] * weight[gid % args.dim];
}

struct ds4_metal_args_muse_router {
    uint32_t n_tokens;
    uint32_t n_expert;
    uint32_t n_used;
    uint32_t pad0;
};

kernel void kernel_muse_router_topk(
        constant ds4_metal_args_muse_router &args,
        device const float *logits,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        uint tok [[thread_position_in_grid]]) {
    if (tok >= args.n_tokens) return;
    device const float *lg = logits + (uint64_t)tok * args.n_expert;
    float best_v[16];
    float best_w[16];
    int best_i[16];
    for (uint k = 0; k < args.n_used; k++) {
        best_v[k] = -INFINITY;
        best_w[k] = 0.0f;
        best_i[k] = 0x7fffffff;
    }
    for (uint e = 0; e < args.n_expert; e++) {
        const float w = muse_sigmoid(lg[e]);
        const float v = w + bias[e];
        uint at = args.n_used;
        for (uint k = 0; k < args.n_used; k++) {
            if (v > best_v[k] || (v == best_v[k] && (int)e < best_i[k])) { at = k; break; }
        }
        if (at < args.n_used) {
            for (uint k = args.n_used - 1u; k > at; k--) {
                best_v[k] = best_v[k - 1u]; best_w[k] = best_w[k - 1u]; best_i[k] = best_i[k - 1u];
            }
            best_v[at] = v; best_w[at] = w; best_i[at] = (int)e;
        }
    }
    float sum = 0.0f;
    for (uint k = 0; k < args.n_used; k++) sum += best_w[k];
    const float inv = 1.0f / max(sum, 6.103515625e-5f);
    for (uint k = 0; k < args.n_used; k++) {
        selected[(uint64_t)tok * args.n_used + k] = best_i[k];
        weights[(uint64_t)tok * args.n_used + k] = best_w[k] * inv;
    }
}

struct ds4_metal_args_muse_swa {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t window;
    uint32_t stride;
};

kernel void kernel_muse_swa_indices(
        constant ds4_metal_args_muse_swa &args,
        device int32_t *selected,
        device uint32_t *counts,
        uint tok [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    if (tok >= args.n_tokens) return;
    const uint end = args.pos0 + tok + 1u;
    const uint start = end > args.window ? end - args.window : 0u;
    const uint n = end - start;
    for (uint i = tid; i < n; i += ntg) selected[(uint64_t)tok * args.stride + i] = (int32_t)(start + i);
    if (tid == 0) counts[tok] = n;
}

struct ds4_metal_args_muse_logits {
    uint32_t count;
    float scale;
    float softcap;
    uint32_t pad0;
};

kernel void kernel_muse_logits(
        constant ds4_metal_args_muse_logits &args,
        device float *x,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.count) return;
    float v = x[gid] * args.scale;
    if (args.softcap > 0.0f) v = args.softcap * tanh(v / args.softcap);
    x[gid] = v;
}
