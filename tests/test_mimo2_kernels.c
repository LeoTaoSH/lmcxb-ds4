/* GPU kernel tests for MiMo V2.6 Flash attention: fused QKV split + partial
 * rotate-half RoPE + value scale into f16 ring caches, then GQA attention with
 * K/V head dims 192/128 (release) and 64/32 (mini), sliding window, sinks,
 * split-key merge, chunked prefill through the ring, and the refusal paths.
 * Attention is checked against a double-precision reference. Streamed expert
 * rows are checked bit-for-bit against the existing top-eight decode loop.
 * Build and run: make test-mimo2-kernels
 * Focused expert schedule: DS4_TEST_MIMO2_EXPERT_ROWS_ONLY=1 ./tests/test_mimo2_kernels */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint32_t g_rng = 0x2545f491u;

static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xffffffu) / 8388608.0f) - 1.0f;
}

static void require(bool ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
}

static float f16_round(float x) {
    return (float)(__fp16)x;
}

static void check_close(const char *what, const float *got, const double *ref, uint64_t n, double tol) {
    double worst = 0.0, scale = 1e-6;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "FAIL %s: non-finite value at %llu\n", what, (unsigned long long)i);
            exit(1);
        }
        const double d = fabs((double)got[i] - ref[i]);
        if (d > worst) { worst = d; worst_i = i; }
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    if (worst > tol * scale) {
        fprintf(stderr, "FAIL %s: max|d| %.3e (rel %.3e) at %llu: got %.7f ref %.7f\n",
                what, worst, worst / scale, (unsigned long long)worst_i, got[worst_i], ref[worst_i]);
        exit(1);
    }
    printf("  %-58s ok  max|d|=%.2e (scale %.2e)\n", what, worst, scale);
}

typedef struct {
    const char *name;
    uint32_t n_head, n_kv, key_dim, value_dim, rot_dim;
    uint32_t window;      /* 0 = global */
    uint32_t cache_cap;
    double rope_base;
    float value_scale;
    bool sinks;
    bool staged;          /* stage k/v, attend ring + staged rows, commit */
} layer_cfg;

static void rope_freqs(float *freq, uint32_t rot_dim, double base) {
    for (uint32_t i = 0; i < rot_dim / 2u; i++)
        freq[i] = (float)pow(base, -2.0 * i / rot_dim);
}

/* Reference q/k/v of one token: rotate-half RoPE on the first rot dims with
 * the same f32 angle the kernel forms, value scale on v. */
static void ref_qkv_row(const layer_cfg *c, const float *freq, const float *src, uint32_t pos,
                        double *q, double *k, double *v) {
    const uint32_t H = c->n_head, Hkv = c->n_kv, K = c->key_dim, V = c->value_dim, R = c->rot_dim;
    for (uint32_t slot = 0; slot < H + Hkv; slot++) {
        const float *x = slot < H ? src + (uint64_t)slot * K : src + (uint64_t)H * K + (uint64_t)(slot - H) * K;
        double *dst = slot < H ? q + (uint64_t)slot * K : k + (uint64_t)(slot - H) * K;
        for (uint32_t d = 0; d < K; d++) dst[d] = x[d];
        for (uint32_t i = 0; i < R / 2u; i++) {
            const float theta = (float)pos * freq[i];
            const double cs = cos((double)theta), sn = sin((double)theta);
            dst[i] = (double)x[i] * cs - (double)x[i + R / 2u] * sn;
            dst[i + R / 2u] = (double)x[i + R / 2u] * cs + (double)x[i] * sn;
        }
    }
    const float *vs = src + (uint64_t)H * K + (uint64_t)Hkv * K;
    for (uint32_t i = 0; i < Hkv * V; i++) v[i] = (double)(vs[i] * c->value_scale);
}

/* Full history for the reference: per position the input row, reference
 * q, and f16-rounded k/v as the caches hold them. */
typedef struct {
    layer_cfg c;
    float freq[64];
    uint32_t row;       /* fused qkv width */
    uint32_t n_pos;
    float  *qkv;        /* [n_pos][row] */
    double *q;          /* [n_pos][H*K] */
    double *k16;        /* [n_pos][Hkv*K] */
    double *v16;        /* [n_pos][Hkv*V] */
    float  *sinks;      /* [H] */
    ds4_gpu_tensor *g_qkv, *g_q, *g_k, *g_v, *g_heads, *g_kc, *g_vc, *g_sinks;
    uint32_t max_chunk;
} seq_state;

static void seq_init(seq_state *s, const layer_cfg *c, uint32_t n_pos, uint32_t max_chunk, float q_gain) {
    memset(s, 0, sizeof(*s));
    s->c = *c;
    s->n_pos = n_pos;
    s->max_chunk = max_chunk;
    const uint32_t H = c->n_head, Hkv = c->n_kv, K = c->key_dim, V = c->value_dim;
    s->row = H * K + Hkv * K + Hkv * V;
    rope_freqs(s->freq, c->rot_dim, c->rope_base);
    s->qkv = malloc((uint64_t)n_pos * s->row * sizeof(float));
    s->q = malloc((uint64_t)n_pos * H * K * sizeof(double));
    s->k16 = malloc((uint64_t)n_pos * Hkv * K * sizeof(double));
    s->v16 = malloc((uint64_t)n_pos * Hkv * V * sizeof(double));
    s->sinks = malloc(H * sizeof(float));
    require(s->qkv && s->q && s->k16 && s->v16 && s->sinks, "host allocation");
    for (uint64_t i = 0; i < (uint64_t)n_pos * s->row; i++) {
        const uint64_t col = i % s->row;
        s->qkv[i] = frand() * (col < (uint64_t)H * K ? q_gain : 1.0f);
    }
    for (uint32_t h = 0; h < H; h++) s->sinks[h] = 2.0f * frand() + (h % 3u == 0 ? 3.0f : 0.0f);
    double *kr = malloc((uint64_t)Hkv * K * sizeof(double));
    double *vr = malloc((uint64_t)Hkv * V * sizeof(double));
    for (uint32_t p = 0; p < n_pos; p++) {
        ref_qkv_row(c, s->freq, s->qkv + (uint64_t)p * s->row, p, s->q + (uint64_t)p * H * K, kr, vr);
        for (uint32_t i = 0; i < Hkv * K; i++) s->k16[(uint64_t)p * Hkv * K + i] = f16_round((float)kr[i]);
        for (uint32_t i = 0; i < Hkv * V; i++) s->v16[(uint64_t)p * Hkv * V + i] = f16_round((float)vr[i]);
    }
    free(kr);
    free(vr);
    const uint64_t T = max_chunk;
    s->g_qkv = ds4_gpu_tensor_alloc(T * s->row * sizeof(float));
    s->g_q = ds4_gpu_tensor_alloc(T * H * K * sizeof(float));
    s->g_k = ds4_gpu_tensor_alloc(T * Hkv * K * sizeof(float));
    s->g_v = ds4_gpu_tensor_alloc(T * Hkv * V * sizeof(float));
    s->g_heads = ds4_gpu_tensor_alloc((T + 1u) * H * V * sizeof(float));
    s->g_kc = ds4_gpu_tensor_alloc((uint64_t)c->cache_cap * Hkv * K * 2u);
    s->g_vc = ds4_gpu_tensor_alloc((uint64_t)c->cache_cap * Hkv * V * 2u);
    s->g_sinks = ds4_gpu_tensor_alloc(H * sizeof(float));
    require(s->g_qkv && s->g_q && s->g_k && s->g_v && s->g_heads && s->g_kc && s->g_vc && s->g_sinks,
            "GPU allocation");
    require(ds4_gpu_tensor_write(s->g_sinks, 0, s->sinks, H * sizeof(float)), "sinks upload");
}

static void seq_free(seq_state *s) {
    free(s->qkv); free(s->q); free(s->k16); free(s->v16); free(s->sinks);
    ds4_gpu_tensor *t[] = { s->g_qkv, s->g_q, s->g_k, s->g_v, s->g_heads, s->g_kc, s->g_vc, s->g_sinks };
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) ds4_gpu_tensor_free(t[i]);
}

/* Reference heads for query positions pos0..pos0+T-1 from the full history. */
static void ref_attention(const seq_state *s, uint32_t pos0, uint32_t T, float scale, bool sinks, double *out) {
    const uint32_t H = s->c.n_head, Hkv = s->c.n_kv, K = s->c.key_dim, V = s->c.value_dim;
    const uint32_t group = H / Hkv;
    double *sc = malloc((uint64_t)(pos0 + T) * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        const uint32_t p = pos0 + t;
        const uint32_t start = s->c.window && p + 1u > s->c.window ? p + 1u - s->c.window : 0u;
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t kvh = h / group;
            const double *qh = s->q + ((uint64_t)p * H + h) * K;
            double mx = sinks ? s->sinks[h] : -1e300;
            for (uint32_t j = start; j <= p; j++) {
                const double *kh = s->k16 + ((uint64_t)j * Hkv + kvh) * K;
                double d = 0.0;
                for (uint32_t i = 0; i < K; i++) d += qh[i] * kh[i];
                sc[j] = d * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            double den = sinks ? exp(s->sinks[h] - mx) : 0.0;
            for (uint32_t j = start; j <= p; j++) { sc[j] = exp(sc[j] - mx); den += sc[j]; }
            double *o = out + ((uint64_t)t * H + h) * V;
            for (uint32_t i = 0; i < V; i++) o[i] = 0.0;
            for (uint32_t j = start; j <= p; j++) {
                const double *vh = s->v16 + ((uint64_t)j * Hkv + kvh) * V;
                for (uint32_t i = 0; i < V; i++) o[i] += sc[j] / den * vh[i];
            }
        }
    }
    free(sc);
}

/* One chunk through the wrappers the host uses for this layer type:
 * direct (qkv stores the cache rows, attention reads only caches) or staged
 * (qkv stages k/v, attention reads ring + staged rows, commit after). */
static void seq_dispatch(seq_state *s, uint32_t pos0, uint32_t T, bool attend) {
    const layer_cfg *c = &s->c;
    const uint32_t H = c->n_head, Hkv = c->n_kv, K = c->key_dim, V = c->value_dim, cap = c->cache_cap;
    const float scale = (float)(1.0 / sqrt((double)K));
    require(ds4_gpu_mimo2_qkv_rope_cache_tensor(s->g_q, s->g_k, s->g_v,
                                                c->staged ? NULL : s->g_kc, c->staged ? NULL : s->g_vc,
                                                s->g_qkv, T, pos0, cap, H, Hkv, K, V, c->rot_dim,
                                                c->value_scale, s->freq), "qkv rope cache dispatch");
    if (attend) {
        require(ds4_gpu_mimo2_attention_tensor(s->g_heads, s->g_q, s->g_kc, s->g_vc,
                                               c->staged ? s->g_k : NULL, c->staged ? s->g_v : NULL,
                                               c->sinks ? s->g_sinks : NULL,
                                               T, pos0, cap, H, Hkv, K, V, c->window, scale),
                "attention dispatch");
    }
    if (c->staged) {
        require(ds4_gpu_mimo2_kv_commit_tensor(s->g_kc, s->g_vc, s->g_k, s->g_v, T, pos0, cap, Hkv, K, V),
                "kv commit dispatch");
    }
}

/* One chunk through both wrappers inside a command batch; checks q/k/v, the
 * stored cache rows, every untouched cache row and the heads. */
static void seq_chunk(seq_state *s, uint32_t pos0, uint32_t T, bool check_qkv, const char *label) {
    const layer_cfg *c = &s->c;
    const uint32_t H = c->n_head, Hkv = c->n_kv, K = c->key_dim, V = c->value_dim, cap = c->cache_cap;
    const float scale = (float)(1.0 / sqrt((double)K));
    require(T <= s->max_chunk && pos0 + T <= s->n_pos, "chunk bounds");
    require(ds4_gpu_tensor_write(s->g_qkv, 0, s->qkv + (uint64_t)pos0 * s->row,
                                 (uint64_t)T * s->row * sizeof(float)), "qkv upload");
    const float sentinel = 12345.0f;
    float *sent = malloc((uint64_t)H * V * sizeof(float));
    for (uint32_t i = 0; i < H * V; i++) sent[i] = sentinel;
    require(ds4_gpu_tensor_write(s->g_heads, (uint64_t)T * H * V * sizeof(float), sent,
                                 (uint64_t)H * V * sizeof(float)), "heads guard upload");

    uint16_t *kc_before = malloc((uint64_t)cap * Hkv * K * 2u);
    uint16_t *vc_before = malloc((uint64_t)cap * Hkv * V * 2u);
    require(ds4_gpu_tensor_read(s->g_kc, 0, kc_before, (uint64_t)cap * Hkv * K * 2u), "key cache read");
    require(ds4_gpu_tensor_read(s->g_vc, 0, vc_before, (uint64_t)cap * Hkv * V * 2u), "value cache read");

    require(ds4_gpu_begin_commands(), "begin commands");
    seq_dispatch(s, pos0, T, true);
    require(ds4_gpu_end_commands(), "end commands");

    char what[160];
    float *gk = malloc((uint64_t)T * Hkv * K * sizeof(float));
    float *gv = malloc((uint64_t)T * Hkv * V * sizeof(float));
    require(ds4_gpu_tensor_read(s->g_k, 0, gk, (uint64_t)T * Hkv * K * sizeof(float)), "k read");
    require(ds4_gpu_tensor_read(s->g_v, 0, gv, (uint64_t)T * Hkv * V * sizeof(float)), "v read");
    if (check_qkv) {
        float *gq = malloc((uint64_t)T * H * K * sizeof(float));
        double *rk = malloc((uint64_t)T * Hkv * K * sizeof(double));
        double *rv = malloc((uint64_t)T * Hkv * V * sizeof(double));
        double *rq = malloc((uint64_t)T * H * K * sizeof(double));
        require(ds4_gpu_tensor_read(s->g_q, 0, gq, (uint64_t)T * H * K * sizeof(float)), "q read");
        for (uint32_t t = 0; t < T; t++)
            ref_qkv_row(c, s->freq, s->qkv + (uint64_t)(pos0 + t) * s->row, pos0 + t,
                        rq + (uint64_t)t * H * K, rk + (uint64_t)t * Hkv * K, rv + (uint64_t)t * Hkv * V);
        snprintf(what, sizeof(what), "%s %s q rope @%u+%u", c->name, label, pos0, T);
        check_close(what, gq, rq, (uint64_t)T * H * K, 2e-6);
        snprintf(what, sizeof(what), "%s %s k rope @%u+%u", c->name, label, pos0, T);
        check_close(what, gk, rk, (uint64_t)T * Hkv * K, 2e-6);
        snprintf(what, sizeof(what), "%s %s v scale @%u+%u", c->name, label, pos0, T);
        check_close(what, gv, rv, (uint64_t)T * Hkv * V, 1e-7);
        free(gq); free(rk); free(rv); free(rq);
    }

    /* Cache rows: row (pos0+t) % cap holds f16(k/v of token t) bit for bit,
     * every other row is unchanged. */
    uint16_t *kc = malloc((uint64_t)cap * Hkv * K * 2u);
    uint16_t *vc = malloc((uint64_t)cap * Hkv * V * 2u);
    require(ds4_gpu_tensor_read(s->g_kc, 0, kc, (uint64_t)cap * Hkv * K * 2u), "key cache read");
    require(ds4_gpu_tensor_read(s->g_vc, 0, vc, (uint64_t)cap * Hkv * V * 2u), "value cache read");
    uint32_t bad = 0;
    for (uint32_t r = 0; r < cap; r++) {
        int32_t t_own = -1;
        for (uint32_t t = 0; t < T; t++) if ((pos0 + t) % cap == r) t_own = (int32_t)t;
        for (uint32_t i = 0; i < Hkv * K; i++) {
            const uint16_t got = kc[(uint64_t)r * Hkv * K + i];
            if (t_own < 0) { if (got != kc_before[(uint64_t)r * Hkv * K + i]) bad++; continue; }
            __fp16 h = (__fp16)gk[(uint64_t)t_own * Hkv * K + i];
            uint16_t want; memcpy(&want, &h, 2);
            if (got != want) bad++;
        }
        for (uint32_t i = 0; i < Hkv * V; i++) {
            const uint16_t got = vc[(uint64_t)r * Hkv * V + i];
            if (t_own < 0) { if (got != vc_before[(uint64_t)r * Hkv * V + i]) bad++; continue; }
            __fp16 h = (__fp16)gv[(uint64_t)t_own * Hkv * V + i];
            uint16_t want; memcpy(&want, &h, 2);
            if (got != want) bad++;
        }
    }
    free(gk); free(gv);
    if (bad) {
        fprintf(stderr, "FAIL %s %s: %u cache values differ from the expected ring contents\n", c->name, label, bad);
        exit(1);
    }

    float *gh = malloc(((uint64_t)T + 1u) * H * V * sizeof(float));
    double *rh = malloc((uint64_t)T * H * V * sizeof(double));
    require(ds4_gpu_tensor_read(s->g_heads, 0, gh, ((uint64_t)T + 1u) * H * V * sizeof(float)), "heads read");
    for (uint32_t i = 0; i < H * V; i++) {
        if (gh[(uint64_t)T * H * V + i] != sentinel) {
            fprintf(stderr, "FAIL %s %s: heads written past n_tokens\n", c->name, label);
            exit(1);
        }
    }
    ref_attention(s, pos0, T, scale, c->sinks, rh);
    snprintf(what, sizeof(what), "%s %s attention @%u+%u", c->name, label, pos0, T);
    check_close(what, gh, rh, (uint64_t)T * H * V, 2e-3);
    free(gh); free(rh); free(kc); free(vc); free(kc_before); free(vc_before); free(sent);
}

/* Feed positions 0..n_pos-1 in the given chunk pattern (cycled). */
static void run_sequence(const layer_cfg *c, uint32_t n_pos, const uint32_t *chunks, uint32_t n_chunks,
                         uint32_t check_from, float q_gain) {
    uint32_t max_chunk = 1;
    for (uint32_t i = 0; i < n_chunks; i++) if (chunks[i] > max_chunk) max_chunk = chunks[i];
    seq_state s;
    seq_init(&s, c, n_pos, max_chunk, q_gain);
    uint32_t pos = 0, ci = 0, calls = 0;
    while (pos < n_pos) {
        uint32_t T = chunks[ci++ % n_chunks];
        if (T > n_pos - pos) T = n_pos - pos;
        const bool check = pos + T > check_from;
        if (check) {
            seq_chunk(&s, pos, T, calls < 2u || pos + T == n_pos, T == 1 ? "decode" : "prefill");
        } else {
            /* Fill the history without the (expensive) reference. */
            require(ds4_gpu_tensor_write(s.g_qkv, 0, s.qkv + (uint64_t)pos * s.row,
                                         (uint64_t)T * s.row * sizeof(float)), "qkv upload");
            seq_dispatch(&s, pos, T, false);
        }
        calls++;
        pos += T;
    }
    seq_free(&s);
}

/* Refusals: shapes the kernels do not cover, undersized buffers, chunks
 * whose stored rows would clobber keys an earlier query still needs, and
 * staged attention over a ring shorter than the window reaches back. */
static void test_refusals(void) {
    const uint32_t H = 8, Hkv = 4, K = 64, V = 32, cap = 8, T = 2;
    float freq[16];
    rope_freqs(freq, 32, 1e4);
    ds4_gpu_tensor *qkv = ds4_gpu_tensor_alloc((uint64_t)16 * (H * K + Hkv * (K + V)) * 4u);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)16 * H * K * 4u);
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc((uint64_t)16 * Hkv * K * 4u);
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc((uint64_t)16 * Hkv * V * 4u);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc((uint64_t)16 * H * V * 4u);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * K * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * V * 2u);
    ds4_gpu_tensor *small_sinks = ds4_gpu_tensor_alloc(4u);
    require(qkv && q && k && v && heads && kc && vc && small_sinks, "refusal allocations");
    const float s = 0.125f;
    fprintf(stderr, "(expected refusal messages follow)\n");
    /* Direct: window == cap, a 2-row chunk overwrote the key the first query needs. */
    require(!ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, NULL, NULL, NULL, T, 20, cap, H, Hkv, K, V, cap, s),
            "direct: window == cap with 2 rows after wrap is refused");
    require(ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, NULL, NULL, NULL, T, 20, cap, H, Hkv, K, V, cap - 1u, s),
            "direct: window + T - 1 == cap is accepted");
    require(ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, NULL, NULL, NULL, T, 0, cap, H, Hkv, K, V, cap, s),
            "direct: chunk before the ring wraps is accepted");
    require(!ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, NULL, NULL, NULL, 1, cap, cap, H, Hkv, K, V, 0, s),
            "direct: global attention past the cache is refused");
    /* Staged: any chunk length once the ring covers window - 1 rows. */
    require(ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, k, v, NULL, 16, 100, cap, H, Hkv, K, V, cap + 1u, s),
            "staged: 16 rows over an 8-row ring, window 9, is accepted");
    require(!ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, k, v, NULL, 1, 100, cap, H, Hkv, K, V, cap + 2u, s),
            "staged: window beyond ring + 1 is refused");
    require(!ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, k, v, NULL, 1, 0, cap, H, Hkv, K, V, 0, s),
            "staged: global layer is refused");
    require(!ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, k, NULL, NULL, 1, 0, cap, H, Hkv, K, V, 4, s),
            "staged: k without v is refused");
    require(!ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, NULL, NULL, NULL, 1, 0, cap, H, Hkv, 128, 128, 0, s),
            "unsupported head dims are refused");
    require(!ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, NULL, NULL, NULL, 1, 0, cap, 96, 4, K, V, 0, s),
            "q group above 16 is refused");
    require(!ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, NULL, NULL, small_sinks, 1, 0, cap, H, Hkv, K, V, 4, s),
            "undersized sinks are refused");
    require(!ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, NULL, NULL, NULL, 1, 0, cap * 2u, H, Hkv, K, V, 0, s),
            "undersized cache for cap is refused");
    require(!ds4_gpu_mimo2_attention_tensor(heads, q, kc, vc, k, v, NULL, 17, 100, cap, H, Hkv, K, V, 4, s),
            "undersized staged k/v are refused");
    require(!ds4_gpu_mimo2_qkv_rope_cache_tensor(q, k, v, kc, NULL, qkv, 1, 0, cap, H, Hkv, K, V, 32, 0.7f, freq),
            "qkv with one cache is refused");
    require(!ds4_gpu_mimo2_qkv_rope_cache_tensor(q, k, v, kc, vc, qkv, 1, 0, cap, H, Hkv, K, V, 31, 0.7f, freq),
            "odd rotary dims are refused");
    require(!ds4_gpu_mimo2_qkv_rope_cache_tensor(q, k, v, kc, vc, qkv, 1, 0, cap, H, Hkv, K, V, 32, 0.7f, NULL),
            "missing rope table is refused");
    require(!ds4_gpu_mimo2_qkv_rope_cache_tensor(q, k, v, kc, vc, qkv, 1, 0, cap * 2u, H, Hkv, K, V,
                                                 32, 0.7f, freq),
            "undersized cache for qkv is refused");
    require(!ds4_gpu_mimo2_kv_commit_tensor(kc, vc, k, v, 17, 0, cap, Hkv, K, V),
            "commit from undersized staged k/v is refused");
    require(!ds4_gpu_mimo2_kv_commit_tensor(kc, vc, k, v, 1, 0, cap * 2u, Hkv, K, V),
            "commit into an undersized cache is refused");
    fprintf(stderr, "(end of expected refusal messages)\n");
    printf("  %-58s ok\n", "refusals (ring coverage, global wrap, shapes, buffers)");
    ds4_gpu_tensor *t[] = { qkv, q, k, v, heads, kc, vc, small_sinks };
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) ds4_gpu_tensor_free(t[i]);
}

/* A dominant sink drives every head of a sink layer toward zero output; a
 * very negative one leaves plain softmax. */
static void test_sink_extremes(void) {
    const layer_cfg base = { "mini SWA", 8, 4, 64, 32, 32, 6, 6, 1e4, 0.707f, true, true };
    for (int mode = 0; mode < 2; mode++) {
        seq_state s;
        seq_init(&s, &base, 3, 3, 1.0f);
        for (uint32_t h = 0; h < 8; h++) s.sinks[h] = mode == 0 ? 60.0f : -80.0f;
        require(ds4_gpu_tensor_write(s.g_sinks, 0, s.sinks, 8 * sizeof(float)), "sinks upload");
        seq_chunk(&s, 0, 3, false, mode == 0 ? "sink=+60" : "sink=-80");
        if (mode == 0) {
            float out[3 * 8 * 32];
            require(ds4_gpu_tensor_read(s.g_heads, 0, out, sizeof(out)), "heads read");
            double worst = 0.0;
            for (uint32_t i = 0; i < 3 * 8 * 32; i++) if (fabs(out[i]) > worst) worst = fabs(out[i]);
            require(worst < 1e-20, "a dominant sink suppresses the output");
            printf("  %-58s ok  max|out|=%.2e\n", "mini dominant sink suppresses output", worst);
        }
        seq_free(&s);
    }
}

/* Mini-shape sequences: staged SWA rings of exactly the window (and the
 * minimum window - 1), chunks longer than the ring, direct SWA, GA. */
static void mini_sequences(const char *tag, double b_global, double b_swa, float vs) {
    char n[8][64];
    snprintf(n[0], 64, "mini SWA staged cap=W%s", tag);
    snprintf(n[1], 64, "mini SWA staged cap=W-1%s", tag);
    snprintf(n[2], 64, "mini SWA direct%s", tag);
    snprintf(n[3], 64, "mini GA%s", tag);
    snprintf(n[4], 64, "mini SWA w1 staged%s", tag);
    snprintf(n[5], 64, "mini SWA w40 staged%s", tag);
    snprintf(n[6], 64, "mini GA long%s", tag);
    const uint32_t chunks[] = { 5, 1, 1, 3, 5, 2, 1, 4, 13 };
    const layer_cfg swa = { n[0], 8, 4, 64, 32, 32, 6, 6, b_swa, vs, true, true };
    run_sequence(&swa, 80, chunks, 9, 0, 2.0f);
    const layer_cfg swa_min = { n[1], 8, 4, 64, 32, 32, 6, 5, b_swa, vs, true, true };
    run_sequence(&swa_min, 80, chunks, 9, 0, 2.0f);
    const uint32_t direct_chunks[] = { 5, 1, 1, 3, 5, 2, 1, 4 };
    const layer_cfg swa_direct = { n[2], 8, 4, 64, 32, 32, 6, 6 + 5 - 1, b_swa, vs, true, false };
    run_sequence(&swa_direct, 60, direct_chunks, 8, 0, 2.0f);
    const layer_cfg ga = { n[3], 8, 2, 64, 32, 32, 0, 80, b_global, vs, false, false };
    run_sequence(&ga, 80, chunks, 9, 0, 2.0f);
    const layer_cfg w1 = { n[4], 8, 4, 64, 32, 32, 1, 1, b_swa, vs, true, true };
    run_sequence(&w1, 20, chunks, 9, 0, 2.0f);
    /* Windows spanning several 32-key tiles, chunks longer than the ring. */
    const uint32_t long_chunks[] = { 37, 1, 12, 29, 90 };
    const layer_cfg swa40 = { n[5], 8, 4, 64, 32, 32, 40, 40, b_swa, vs, true, true };
    run_sequence(&swa40, 300, long_chunks, 5, 0, 2.0f);
    const layer_cfg ga40 = { n[6], 8, 2, 64, 32, 32, 0, 300, b_global, vs, false, false };
    run_sequence(&ga40, 300, long_chunks, 5, 0, 2.0f);
}

static void *g_mx_test_map;
static size_t g_mx_test_bytes;
static int g_mx_test_fd = -1;

static void test_streamed_mxfp4_top8(void) {
    enum { E = 256, FF = 256, EXPERTS = 128, USED = 8 };
    const uint64_t row_bytes = (uint64_t)E / 32u * 17u;
    const uint64_t expert_bytes = row_bytes * FF;
    const uint64_t slab_bytes = expert_bytes * EXPERTS;
    const uint64_t model_bytes = 3u * slab_bytes;
    char path[] = "/tmp/ds4-mimo-mxfp4-XXXXXX";
    int fd = mkstemp(path);
    require(fd >= 0, "MXFP4 fixture file");
    unlink(path);
    require(ftruncate(fd, (off_t)model_bytes) == 0, "MXFP4 fixture size");
    uint8_t *map = mmap(NULL, model_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    require(map != MAP_FAILED, "MXFP4 fixture mapping");
    for (uint32_t part = 0; part < 3u; part++) {
        for (uint32_t e = 0; e < EXPERTS; e++) {
            for (uint32_t row = 0; row < FF; row++) {
                for (uint32_t block = 0; block < E / 32u; block++) {
                    uint8_t *w = map + (uint64_t)part * slab_bytes +
                                 (uint64_t)e * expert_bytes + row * row_bytes + block * 17u;
                    w[0] = 121u;
                    for (uint32_t i = 1; i < 17u; i++) {
                        const uint8_t lo = (uint8_t)((e + part * 3u + row + i) % 8u);
                        const uint8_t hi = (uint8_t)((e * 3u + part + block + i) % 8u);
                        w[i] = lo | (hi << 4);
                    }
                }
            }
        }
    }
    require(msync(map, model_bytes, MS_SYNC) == 0, "MXFP4 fixture flush");
    g_mx_test_map = map;
    g_mx_test_bytes = (size_t)model_bytes;
    g_mx_test_fd = fd;
    require(ds4_gpu_set_model_fd(fd), "MXFP4 fixture model fd");
    require(ds4_gpu_set_model_map_range(map, model_bytes, 0, model_bytes, slab_bytes),
            "MXFP4 resident model map");
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(E * sizeof(float));
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc(USED * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(USED * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(E * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(USED * FF * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(USED * FF * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(USED * FF * sizeof(float));
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc(USED * E * sizeof(float));
    require(x && ids && weights && out && gate && up && mid && part, "MXFP4 expert tensors");
    float input[E], route[USED], resident[E], streamed[E];
    const int32_t selected[USED] = {3, 7, 11, 17, 23, 43, 79, 101};
    for (uint32_t i = 0; i < E; i++) input[i] = frand() * 0.25f;
    for (uint32_t i = 0; i < USED; i++) route[i] = 1.0f / USED;
    require(ds4_gpu_tensor_write(x, 0, input, sizeof(input)) &&
            ds4_gpu_tensor_write(ids, 0, selected, sizeof(selected)) &&
            ds4_gpu_tensor_write(weights, 0, route, sizeof(route)), "MXFP4 inputs");
    require(ds4_gpu_routed_moe_one_tensor(out, gate, up, mid, part, map, model_bytes,
            0, slab_bytes, 2u * slab_bytes, 39, 39, expert_bytes, row_bytes,
            expert_bytes, row_bytes, E, FF, E, ids, weights, EXPERTS, USED,
            0.0f, x, NULL, 1u, true), "resident MXFP4 top-eight dispatch");
    require(ds4_gpu_tensor_read(out, 0, resident, sizeof(resident)), "resident MXFP4 result");
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(USED);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(3u * expert_bytes);
    require(ds4_gpu_set_model_map_range(map, model_bytes, 0, model_bytes, slab_bytes),
            "MXFP4 streamed model map");
    require(ds4_gpu_routed_moe_one_tensor(out, gate, up, mid, part, map, model_bytes,
            0, slab_bytes, 2u * slab_bytes, 39, 39, expert_bytes, row_bytes,
            expert_bytes, row_bytes, E, FF, E, ids, weights, EXPERTS, USED,
            0.0f, x, NULL, 1u, false), "streamed MXFP4 top-eight dispatch");
    require(ds4_gpu_tensor_read(out, 0, streamed, sizeof(streamed)), "streamed MXFP4 result");
    double ref[E];
    for (uint32_t i = 0; i < E; i++) ref[i] = resident[i];
    check_close("MXFP4 top-eight streamed versus resident", streamed, ref, E, 1e-4);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_tensor *buffers[] = { x, ids, weights, out, gate, up, mid, part };
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++)
        ds4_gpu_tensor_free(buffers[i]);
}

static void check_exact_floats(const float *got, const float *want, size_t n, const char *what) {
    for (size_t i = 0; i < n; i++) {
        uint32_t a, b;
        memcpy(&a, got + i, sizeof(a));
        memcpy(&b, want + i, sizeof(b));
        if ((a & 0x7f800000u) == 0x7f800000u ||
            (b & 0x7f800000u) == 0x7f800000u || a != b) {
            fprintf(stderr, "FAIL %s at %zu: %.9g != %.9g\n", what, i, got[i], want[i]);
            exit(1);
        }
    }
}

static void test_cached_mxfp4_prefill(void) {
    enum { E = 256, FF = 256, NE = 128, NU = 8, MAX_T = 64 };
    const uint64_t expert_bytes = (uint64_t)E / 32u * 17u * FF;
    const uint64_t slab = expert_bytes * NE;
    uint8_t *map = g_mx_test_map;
    ds4_gpu_set_ssd_streaming(false);
    for (uint64_t b = 0; b < g_mx_test_bytes / 17u; b++) {
        map[b * 17u] = 117u + (uint8_t)(b % 7u);
        for (uint32_t i = 1; i < 17u; i++) map[b * 17u + i] ^= (uint8_t)((b + i) % 3u ? 0x80u : 0x08u);
    }
    require(msync(map, g_mx_test_bytes, MS_SYNC) == 0, "signed expert fixture flush");
    void *cache_map = mmap(NULL, g_mx_test_bytes, PROT_READ, MAP_SHARED, g_mx_test_fd, 0);
    require(cache_map != MAP_FAILED, "unbound cache source map");
    const size_t nm = MAX_T * NU * FF, np = MAX_T * NU * E, no = MAX_T * E;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(no * sizeof(float));
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc(MAX_T * NU * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(MAX_T * NU * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(nm * sizeof(float));
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc(np * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(no * sizeof(float));
    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(sizeof(int32_t));
    require(x && ids && weights && mid && part && out && small, "cached prefill buffers");
    float *gm = malloc(nm * sizeof(float)), *gp = malloc(np * sizeof(float));
    float *go = malloc(no * sizeof(float)), *got = malloc(np * sizeof(float));
    require(gm && gp && go && got, "cached prefill reference buffers");
    float input[MAX_T * E], routes[MAX_T * NU];
    int32_t selected[MAX_T * NU];
    for (size_t i = 0; i < no; i++) input[i] = frand() * 0.15f;
    for (uint32_t t = 0; t < MAX_T; t++) {
        for (uint32_t i = 0; i < NU; i++) {
            selected[t * NU + i] = (int32_t)((t * 11u + i * 7u) % NE);
            routes[t * NU + i] = (float)(i + 1u) / 36.0f;
        }
    }
    require(ds4_gpu_tensor_write(x, 0, input, sizeof(input)) &&
            ds4_gpu_tensor_write(ids, 0, selected, sizeof(selected)) &&
            ds4_gpu_tensor_write(weights, 0, routes, sizeof(routes)), "cached prefill inputs");
    const uint32_t counts[] = {2, 8, 9, 11, 64};
    const uint32_t budgets[] = {8, 128};
    for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
        const uint32_t T = counts[c];
        const size_t m = (size_t)T * NU * FF, p = (size_t)T * NU * E, o = (size_t)T * E;
        require(ds4_gpu_begin_commands(), "reference begin");
        require(ds4_gpu_qwen4_moe_mid_tensor(mid, x, ids, map, g_mx_test_bytes,
                0, slab, 39, NE, T, NU, E, FF, 0, 0, UINT32_MAX), "mapped mid");
        require(ds4_gpu_qwen4_moe_down_tensor(part, mid, ids, map, g_mx_test_bytes,
                2u * slab, 39, NE, T, NU, FF, E, 0, UINT32_MAX), "mapped partials");
        require(ds4_gpu_qwen4_moe_reduce_tensor(out, part, weights, NULL, NULL, NULL, NULL,
                T, NU, NU, E, 0), "mapped reduction");
        require(ds4_gpu_end_commands(), "reference finish");
        require(ds4_gpu_tensor_read(mid, 0, gm, m * sizeof(float)) &&
                ds4_gpu_tensor_read(part, 0, gp, p * sizeof(float)) &&
                ds4_gpu_tensor_read(out, 0, go, o * sizeof(float)), "mapped references");
        for (size_t b = 0; b < sizeof(budgets) / sizeof(budgets[0]); b++) {
            ds4_gpu_set_ssd_streaming(true);
            ds4_gpu_set_streaming_expert_cache_budget(budgets[b]);
            ds4_gpu_set_streaming_expert_cache_expert_bytes(3u * expert_bytes);
            for (int pass = 0; pass < 2; pass++) {
                memset(got, 0xff, np * sizeof(float));
                require(ds4_gpu_tensor_write(mid, 0, got, m * sizeof(float)) &&
                        ds4_gpu_tensor_write(part, 0, got, p * sizeof(float)), "poison cached outputs");
                require(ds4_gpu_begin_commands(), "cached begin");
                require(ds4_gpu_mimo2_moe_cached_prefill_tensor(mid, part, x, ids,
                        cache_map, g_mx_test_bytes, 0, slab, 2u * slab, 1, NE, T, E, FF),
                        "cached mid and partials");
                require(ds4_gpu_qwen4_moe_reduce_tensor(out, part, weights, NULL, NULL, NULL, NULL,
                        T, NU, NU, E, 0) && ds4_gpu_end_commands(), "cached reduction");
                require(ds4_gpu_tensor_read(mid, 0, got, m * sizeof(float)), "read cached mid");
                check_exact_floats(got, gm, m, "cached mid exact");
                require(ds4_gpu_tensor_read(part, 0, got, p * sizeof(float)), "read cached partials");
                check_exact_floats(got, gp, p, "cached partials exact");
                require(ds4_gpu_tensor_read(out, 0, got, o * sizeof(float)), "read cached output");
                check_exact_floats(got, go, o, "cached output exact");
            }
        }
        printf("  cached MXFP4 prefill rows=%u: cold/warm, 8/128-slot budgets exact\n", T);
        ds4_gpu_set_ssd_streaming(false);
    }
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(8);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(3u * expert_bytes);
    require(!ds4_gpu_mimo2_moe_cached_prefill_tensor(mid, part, x, small,
            cache_map, g_mx_test_bytes, 0, slab, 2u * slab, 1, NE, 2, E, FF), "short IDs refused");
    require(!ds4_gpu_mimo2_moe_cached_prefill_tensor(mid, part, x, ids,
            cache_map, g_mx_test_bytes, 0, slab, 2u * slab, 1, NE, MAX_T + 1, E, FF), "oversized batch refused");
    selected[NU] = NE;
    require(ds4_gpu_tensor_write(ids, 0, selected, sizeof(selected)), "invalid ID upload");
    require(!ds4_gpu_mimo2_moe_cached_prefill_tensor(mid, part, x, ids,
            cache_map, g_mx_test_bytes, 0, slab, 2u * slab, 1, NE, 2, E, FF), "invalid ID refused");
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_tensor *buffers[] = {x, ids, weights, mid, part, out, small};
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++) ds4_gpu_tensor_free(buffers[i]);
    free(gm); free(gp); free(go); free(got); munmap(cache_map, g_mx_test_bytes);
}


static void test_shared_mxfp4_rows(void) {
    enum { E = 256, FF = 256, NE = 128, NU = 8, TMAX = DS4_MIMO2_MOE_MAX_ROWS };
    const uint64_t row_bytes = (uint64_t)E / 32u * 17u;
    const uint64_t expert_bytes = row_bytes * FF;
    const uint64_t slab = expert_bytes * NE;
    uint8_t *map = g_mx_test_map;
    ds4_gpu_set_ssd_streaming(false);
    /* Signed, nonuniform scales exercise cancellation in the unchanged row
     * dots and the ordered eight-expert sum. */
    for (uint64_t b = 0; b < g_mx_test_bytes / 17u; b++) {
        map[b * 17u] = 117u + (uint8_t)(b % 7u);
        for (uint32_t i = 1; i < 17u; i++)
            map[b * 17u + i] ^= (uint8_t)((b + i) % 3u ? 0x80u : 0x08u);
    }
    require(msync(map, g_mx_test_bytes, MS_SYNC) == 0, "shared rows fixture flush");
    const uint64_t sizes[8] = {
        E * sizeof(float), NU * sizeof(int32_t), NU * sizeof(float), E * sizeof(float),
        NU * FF * sizeof(float), NU * FF * sizeof(float),
        NU * FF * sizeof(float), NU * E * sizeof(float),
    };
    ds4_gpu_tensor *owners[TMAX][8], *views[TMAX][8];
    ds4_gpu_mimo2_moe_row rows[TMAX], shared[TMAX];
    ds4_gpu_tensor *source_ids = ds4_gpu_tensor_alloc(TMAX * NU * sizeof(int32_t));
    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(sizeof(int32_t));
    require(source_ids && small, "shared rows auxiliary tensors");
    float input[E], route[NU], want_out[TMAX][E], want_mid[TMAX][NU * FF];
    float got[NU * FF], poison[NU * FF];
    int32_t selected[TMAX][NU], invalid[NU];
    memset(poison, 0xff, sizeof(poison));
    for (uint32_t i = 0; i < NU; i++) invalid[i] = -1;
    for (uint32_t t = 0; t < TMAX; t++) {
        for (uint32_t i = 0; i < 8u; i++) {
            owners[t][i] = ds4_gpu_tensor_alloc(sizes[i] + 64u);
            require(owners[t][i] != NULL, "shared rows tensor allocation");
            views[t][i] = ds4_gpu_tensor_view(owners[t][i], 64u, sizes[i]);
            require(views[t][i] != NULL, "shared rows offset tensor view");
        }
        rows[t] = (ds4_gpu_mimo2_moe_row){
            .out = views[t][3], .gate = views[t][4], .up = views[t][5],
            .mid = views[t][6], .experts = views[t][7],
            .x = views[t][0], .selected = views[t][1], .weights = views[t][2], .add_in = NULL,
        };
        shared[t] = rows[t];
        shared[t].gate = rows[0].gate;
        shared[t].up = rows[0].up;
        shared[t].mid = rows[0].mid;
        shared[t].experts = rows[0].experts;
        for (uint32_t i = 0; i < E; i++) input[i] = frand() * (0.03f + t * 0.01f);
        for (uint32_t i = 0; i < NU; i++) route[i] = (float)(i + t + 1u) / (36.0f + 8.0f * t);
        require(ds4_gpu_tensor_write(views[t][0], 0, input, sizeof(input)) &&
                ds4_gpu_tensor_write(views[t][2], 0, route, sizeof(route)), "shared rows inputs");
    }
    const uint32_t counts[] = {1, 2, 4, 8}, budgets[] = {8, 64};
    for (uint32_t pattern = 0; pattern < 2u; pattern++) {
        for (uint32_t t = 0; t < TMAX; t++) {
            for (uint32_t i = 0; i < NU; i++) {
                /* Distinct row sets force eviction at budget8; permutations
                 * of one set stay fully warm even at that minimum budget. */
                selected[t][i] = pattern ? (int32_t)(3u + ((i + t) % NU) * 7u) :
                                          (int32_t)((t * 13u + i * 7u) % NE);
            }
            require(ds4_gpu_tensor_write(views[t][1], 0, selected[t], sizeof(selected[t])), "row picks upload");
        }
        require(ds4_gpu_tensor_write(source_ids, 0, selected, sizeof(selected)), "router source IDs upload");
        for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
            const uint32_t T = counts[c];
            for (size_t b = 0; b < sizeof(budgets) / sizeof(budgets[0]); b++) {
                ds4_gpu_set_ssd_streaming(true);
                ds4_gpu_set_streaming_expert_cache_budget(budgets[b]);
                ds4_gpu_set_streaming_expert_cache_expert_bytes(3u * expert_bytes);
                require(ds4_gpu_mimo2_moe_rows_supported(T, NE, E, FF), "shared rows supported shape");
                require(ds4_gpu_begin_commands(), "old row loop begin");
                for (uint32_t t = 0; t < T; t++) {
                    const ds4_gpu_mimo2_moe_row *r = &rows[t];
                    require(ds4_gpu_routed_moe_one_tensor(r->out, r->gate, r->up, r->mid, r->experts,
                            map, g_mx_test_bytes, 0, slab, 2u * slab, 39u, 39u,
                            expert_bytes, row_bytes, expert_bytes, row_bytes, E, FF, E,
                            r->selected, r->weights, NE, NU, 0.0f, r->x, NULL, 1u, false), "old row dispatch");
                }
                require(ds4_gpu_end_commands(), "old row loop finish");
                for (uint32_t t = 0; t < T; t++) {
                    require(ds4_gpu_tensor_read(rows[t].out, 0, want_out[t], sizeof(want_out[t])) &&
                            ds4_gpu_tensor_read(rows[t].mid, 0, want_mid[t], sizeof(want_mid[t])), "old row results");
                }
                ds4_gpu_set_streaming_expert_cache_budget(budgets[b]);
                ds4_gpu_set_streaming_expert_cache_expert_bytes(3u * expert_bytes);
                for (uint32_t pass = 0; pass < 3u; pass++) {
                    for (uint32_t t = 0; t < T; t++) {
                        require(ds4_gpu_tensor_write(rows[t].out, 0, poison, sizes[3]) &&
                                ds4_gpu_tensor_write(rows[t].mid, 0, poison, sizes[6]) &&
                                ds4_gpu_tensor_write(views[t][1], 0, invalid, sizeof(invalid)), "poison row outputs and IDs");
                    }
                    require(ds4_gpu_begin_commands(), "shared row schedule begin");
                    for (uint32_t t = 0; t < T; t++)
                        require(ds4_gpu_tensor_copy(views[t][1], 0, source_ids, t * sizeof(selected[t]),
                                                    sizeof(selected[t])), "queued router ID write");
                    require(ds4_gpu_mimo2_moe_rows_tensor(pass == 2u ? shared : rows, T,
                            map, g_mx_test_bytes, 0, slab, 2u * slab, 1u, NE, E, FF), "shared row schedule");
                    require(ds4_gpu_commands_active(), "shared row schedule keeps caller batch active");
                    require(ds4_gpu_end_commands(), "shared row schedule finish");
                    for (uint32_t t = 0; t < T; t++) {
                        require(ds4_gpu_tensor_read(rows[t].out, 0, got, sizes[3]), "shared row output read");
                        check_exact_floats(got, want_out[t], E, "shared row output exact");
                        if (pass != 2u || t == T - 1u) {
                            require(ds4_gpu_tensor_read(pass == 2u ? shared[t].mid : rows[t].mid,
                                                       0, got, sizes[6]), "shared row mid read");
                            check_exact_floats(got, want_mid[t], NU * FF, "shared row mid exact");
                        }
                    }
                }
                printf("  shared MXFP4 rows=%u budget=%u pattern=%s: cold/warm/shared-scratch exact\n",
                       T, budgets[b], pattern ? "permuted" : "distinct");
            }
        }
    }
    require(!ds4_gpu_commands_active(), "shared rows tests finish their batches");
    require(ds4_gpu_mimo2_moe_rows_tensor(rows, 2u, map, g_mx_test_bytes,
            0, slab, 2u * slab, 1u, NE, E, FF), "shared rows without caller batch");
    require(!ds4_gpu_commands_active(), "owned shared schedule finishes its batch");
    require(ds4_gpu_tensor_read(rows[1].out, 0, got, sizes[3]), "owned shared output read");
    check_exact_floats(got, want_out[1], E, "owned shared output exact");

    require(!ds4_gpu_mimo2_moe_rows_supported(0u, NE, E, FF) &&
            !ds4_gpu_mimo2_moe_rows_supported(TMAX + 1u, NE, E, FF) &&
            !ds4_gpu_mimo2_moe_rows_supported(2u, NE, E - 1u, FF) &&
            !ds4_gpu_mimo2_moe_rows_supported(2u, 127u, E, FF), "shared shape refusals");
    ds4_gpu_set_streaming_expert_cache_budget(7u);
    require(!ds4_gpu_mimo2_moe_rows_supported(2u, NE, E, FF), "short cache budget refused");
    ds4_gpu_set_streaming_expert_cache_budget(8u);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(3u * expert_bytes + 1u);
    require(!ds4_gpu_mimo2_moe_rows_supported(2u, NE, E, FF), "off-size cache refused");
    ds4_gpu_set_streaming_expert_cache_expert_bytes(3u * expert_bytes);
    setenv("DS4_METAL_DISABLE_MXFP4_SELECTED_EXPERT_VIEWS", "1", 1);
    require(!ds4_gpu_mimo2_moe_rows_supported(2u, NE, E, FF), "disabled slots path refused");
    unsetenv("DS4_METAL_DISABLE_MXFP4_SELECTED_EXPERT_VIEWS");
    shared[1].selected = small;
    require(!ds4_gpu_mimo2_moe_rows_tensor(shared, 2u, map, g_mx_test_bytes,
            0, slab, 2u * slab, 1u, NE, E, FF), "short row IDs refused");
    shared[1] = rows[1];
    shared[1].add_in = rows[0].x;
    require(!ds4_gpu_mimo2_moe_rows_tensor(shared, 2u, map, g_mx_test_bytes,
            0, slab, 2u * slab, 1u, NE, E, FF), "unsupported row addend refused");
    require(!ds4_gpu_mimo2_moe_rows_tensor(rows, 2u, map, slab,
            0, slab, 2u * slab, 1u, NE, E, FF), "short model mapping refused");
    for (uint32_t i = 0; i < E; i++) got[i] = 123.25f;
    require(ds4_gpu_tensor_write(rows[0].out, 0, got, sizes[3]), "refusal sentinel upload");
    selected[1][0] = NE;
    require(ds4_gpu_tensor_write(views[1][1], 0, selected[1], sizeof(selected[1])), "invalid later row ID");
    require(ds4_gpu_begin_commands(), "refusal caller batch begin");
    require(!ds4_gpu_mimo2_moe_rows_tensor(rows, 2u, map, g_mx_test_bytes,
            0, slab, 2u * slab, 1u, NE, E, FF), "invalid later row refused");
    require(ds4_gpu_commands_active() && ds4_gpu_end_commands(), "refusal caller batch restored");
    require(ds4_gpu_tensor_read(rows[0].out, 0, input, sizeof(input)), "refusal sentinel read");
    check_exact_floats(input, got, E, "later-row validation precedes all expert dispatch");
    ds4_gpu_set_ssd_streaming(false);
    require(!ds4_gpu_mimo2_moe_rows_supported(2u, NE, E, FF), "resident path is explicitly unsupported");
    for (uint32_t t = 0; t < TMAX; t++) {
        for (uint32_t i = 0; i < 8u; i++) {
            ds4_gpu_tensor_free(views[t][i]);
            ds4_gpu_tensor_free(owners[t][i]);
        }
    }
    ds4_gpu_tensor_free(source_ids);
    ds4_gpu_tensor_free(small);
}


static void test_verify_expert_budget(void) {
    enum { E = 256, FF = 256, NE = 128, NU = 8, T = DS4_MIMO2_MOE_MAX_ROWS };
    const uint64_t row_bytes = (uint64_t)E / 32u * 17u;
    const uint64_t expert_bytes = row_bytes * FF, slab = expert_bytes * NE;
    const uint64_t sizes[] = { E * sizeof(float), NU * sizeof(int32_t), NU * sizeof(float),
        E * sizeof(float), NU * FF * sizeof(float), NU * FF * sizeof(float),
        NU * FF * sizeof(float), NU * E * sizeof(float) };
    ds4_gpu_tensor *storage[T][8], *reference_weights[T];
    ds4_gpu_mimo2_moe_row rows[T];
    ds4_gpu_tensor *snapshot = ds4_gpu_tensor_alloc((uint64_t)T * E * sizeof(float));
    require(snapshot != NULL, "pilot snapshot allocation");
    float input[E], route[T][NU], masked[T][NU], want[T][E], got[E];
    int32_t selected[T][NU], read_ids[NU];
    float read_weights[NU];
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t i = 0; i < 8u; i++) {
            storage[t][i] = ds4_gpu_tensor_alloc(sizes[i] + 64u);
            require(storage[t][i] != NULL, "budget tensor allocation");
        }
        reference_weights[t] = ds4_gpu_tensor_alloc(NU * sizeof(float));
        require(reference_weights[t] != NULL, "budget reference weights");
        rows[t] = (ds4_gpu_mimo2_moe_row){
            .out = storage[t][3], .gate = storage[0][4], .up = storage[0][5],
            .mid = storage[0][6], .experts = storage[0][7],
            .x = storage[t][0], .selected = storage[t][1], .weights = storage[t][2],
        };
        for (uint32_t i = 0; i < E; i++) input[i] = frand() * 0.02f;
        require(ds4_gpu_tensor_write(storage[t][0], 0, input, sizeof(input)), "budget input upload");
    }
    const uint32_t budgets[] = {1u, 8u, 16u, 32u, 64u, NE};
    for (uint32_t pattern = 0; pattern < 3u; pattern++) {
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t i = 0; i < NU; i++) {
                /* Distinct ties test ID ordering and rows with no retained
                 * expert. Overlap makes global sums differ from row ranks. */
                selected[t][i] = pattern == 2u ? (int32_t)((t * 3u + i * 5u) % 37u) :
                                                     (int32_t)(t * NU + i);
                route[t][i] = pattern == 0u ? 0.125f : (float)((i + t) % NU + 1u) / 36.0f;
            }
            require(ds4_gpu_tensor_write(storage[t][1], 0, selected[t], sizeof(selected[t])) &&
                    ds4_gpu_tensor_write(storage[t][2], 0, route[t], sizeof(route[t])), "natural budget routes");
        }
        for (size_t b = 0; b < sizeof(budgets) / sizeof(budgets[0]); b++) {
            const uint32_t budget = budgets[b];
            double mass[NE] = {0};
            bool present[NE] = {0}, retained[NE] = {0};
            ds4_gpu_mimo2_verify_budget_stats expected = {0}, measured;
            for (uint32_t t = 0; t < T; t++) {
                for (uint32_t i = 0; i < NU; i++) {
                    const uint32_t id = (uint32_t)selected[t][i];
                    mass[id] += (double)route[t][i];
                    present[id] = true;
                }
            }
            /* Independent pairwise rank, rather than the implementation's
             * repeated maximum selection. Keep original IDs in the reference. */
            for (uint32_t id = 0; id < NE; id++) {
                if (!present[id]) continue;
                expected.natural_unique++;
                uint32_t rank = 0;
                for (uint32_t other = 0; other < NE; other++) {
                    if (present[other] && (mass[other] > mass[id] ||
                        (mass[other] == mass[id] && other < id))) rank++;
                }
                retained[id] = rank < budget;
                expected.kept_unique += retained[id];
            }
            expected.truncated_unique = expected.natural_unique - expected.kept_unique;
            for (uint32_t t = 0; t < T; t++) {
                for (uint32_t i = 0; i < NU; i++) {
                    const bool keep = retained[selected[t][i]];
                    masked[t][i] = keep ? route[t][i] : 0.0f;
                    expected.truncated_slots += !keep;
                }
                require(ds4_gpu_tensor_write(reference_weights[t], 0, masked[t], sizeof(masked[t])), "masked reference upload");
            }
            ds4_gpu_set_ssd_streaming(true);
            ds4_gpu_set_streaming_expert_cache_budget(64u);
            ds4_gpu_set_streaming_expert_cache_expert_bytes(3u * expert_bytes);
            require(ds4_gpu_begin_commands(), "masked reference begin");
            for (uint32_t t = 0; t < T; t++) {
                const ds4_gpu_mimo2_moe_row *r = &rows[t];
                require(ds4_gpu_routed_moe_one_tensor(r->out, r->gate, r->up, r->mid, r->experts,
                        g_mx_test_map, g_mx_test_bytes, 0, slab, 2u * slab, 39u, 39u,
                        expert_bytes, row_bytes, expert_bytes, row_bytes, E, FF, E,
                        r->selected, reference_weights[t], NE, NU, 0.0f, r->x, NULL, 1u, false), "independent masked old row");
            }
            require(ds4_gpu_end_commands(), "masked reference finish");
            for (uint32_t t = 0; t < T; t++)
                require(ds4_gpu_tensor_read(rows[t].out, 0, want[t], sizeof(want[t])), "masked reference output");
            ds4_gpu_set_ssd_streaming(true);
            require(ds4_gpu_begin_commands(), "budget caller batch begin");
            require(ds4_gpu_mimo2_moe_verify_budget_tensor(rows, T, budget,
                    g_mx_test_map, g_mx_test_bytes, 0, slab, 2u * slab,
                    1u, NE, E, FF, &measured), "budget shared row dispatch");
            require(!memcmp(&measured, &expected, sizeof(measured)), "budget actual exclusions match independent rank");
            require(ds4_gpu_commands_active(), "budget preserves caller batch");
            for (uint32_t t = 0; t < T; t++)
                require(ds4_gpu_tensor_copy(snapshot, t * E * sizeof(float), rows[t].out, 0,
                                            E * sizeof(float)), "queue pilot snapshot");
            /* Change the mask while the first dispatch and copies are queued.
             * A scratch rewrite without a fence would corrupt the snapshot. */
            ds4_gpu_mimo2_verify_budget_stats narrowed;
            require(ds4_gpu_mimo2_moe_verify_budget_tensor(rows, T, 1u,
                    g_mx_test_map, g_mx_test_bytes, 0, slab, 2u * slab,
                    1u, NE, E, FF, &narrowed), "reuse scratch with a different mask");
            require(narrowed.kept_unique == 1u && narrowed.truncated_slots > 0u,
                    "narrow pilot has actual truncation");
            require(ds4_gpu_end_commands(), "budget caller batch finish");
            require(ds4_gpu_stream_expert_cache_current_count() == expected.kept_unique,
                    "cold pilot loaded only the in-budget union");
            for (uint32_t t = 0; t < T; t++) {
                require(ds4_gpu_tensor_read(snapshot, t * E * sizeof(float), got, sizeof(got)), "budget snapshot read");
                check_exact_floats(got, want[t], E, "budget output equals independently masked old loop");
                require(ds4_gpu_tensor_read(rows[t].selected, 0, read_ids, sizeof(read_ids)) &&
                        ds4_gpu_tensor_read(rows[t].weights, 0, read_weights, sizeof(read_weights)), "natural routes preserved read");
                require(!memcmp(read_ids, selected[t], sizeof(read_ids)) &&
                        !memcmp(read_weights, route[t], sizeof(read_weights)), "pilot never mutates caller routes");
            }
            printf("  verifier budget=%u pattern=%u unique=%u kept=%u truncated_slots=%u: masked reference exact\n",
                   budget, pattern, measured.natural_unique, measured.kept_unique, measured.truncated_slots);
        }
    }
    ds4_gpu_mimo2_verify_budget_stats measured = {1u, 2u, 3u, 4u}, zero = {0};
#define BUDGET_CALL(R, NR, B, BYTES, N, IN, MID) \
    ds4_gpu_mimo2_moe_verify_budget_tensor(R, NR, B, g_mx_test_map, BYTES, \
        0, slab, 2u * slab, 1u, N, IN, MID, &measured)
    require(!BUDGET_CALL(rows, 0u, 8u, g_mx_test_bytes, NE, E, FF) &&
            !BUDGET_CALL(rows, 1u, 8u, g_mx_test_bytes, NE, E, FF) &&
            !BUDGET_CALL(rows, T + 1u, 8u, g_mx_test_bytes, NE, E, FF) &&
            !BUDGET_CALL(rows, 2u, 0u, g_mx_test_bytes, NE, E, FF) &&
            !BUDGET_CALL(rows, 2u, 257u, g_mx_test_bytes, NE, E, FF) &&
            !BUDGET_CALL(rows, 2u, NE + 1u, g_mx_test_bytes, NE, E, FF) &&
            !BUDGET_CALL(NULL, 2u, 8u, g_mx_test_bytes, NE, E, FF) &&
            !BUDGET_CALL(rows, 2u, 8u, slab, NE, E, FF) &&
            !BUDGET_CALL(rows, 2u, 8u, g_mx_test_bytes, NE, E - 1u, FF) &&
            !BUDGET_CALL(rows, 2u, 8u, g_mx_test_bytes, 127u, E, FF), "budget shape and range refusals");
    require(!memcmp(&measured, &zero, sizeof(measured)), "failed budget calls clear stats");
    ds4_gpu_mimo2_moe_row bad[T];
    memcpy(bad, rows, sizeof(bad));
    bad[1].add_in = rows[0].x;
    require(!BUDGET_CALL(bad, 2u, 8u, g_mx_test_bytes, NE, E, FF), "budget addend refused");
    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(sizeof(float));
    require(small != NULL, "budget short tensor");
    bad[1] = rows[1]; bad[1].weights = small;
    require(!BUDGET_CALL(bad, 2u, 8u, g_mx_test_bytes, NE, E, FF), "budget short weights refused");
    bad[1] = rows[1]; bad[1].selected = small;
    require(!BUDGET_CALL(bad, 2u, 8u, g_mx_test_bytes, NE, E, FF), "budget short IDs refused");
    const uint32_t bad_bits[] = {0xbe000000u, 0x7fc00000u, 0x7f800000u, 0xff800000u};
    for (uint32_t i = 0; i < E; i++) got[i] = 123.25f;
    require(ds4_gpu_tensor_write(rows[0].out, 0, got, sizeof(got)), "budget refusal sentinel");
    for (size_t b = 0; b < sizeof(bad_bits) / sizeof(bad_bits[0]) + 3u; b++) {
        float invalid_weights[NU];
        int32_t invalid_ids[NU];
        memcpy(invalid_weights, route[1], sizeof(invalid_weights));
        memcpy(invalid_ids, selected[1], sizeof(invalid_ids));
        if (b < 4u) memcpy(&invalid_weights[0], &bad_bits[b], sizeof(float));
        else if (b == 4u) memset(invalid_weights, 0, sizeof(invalid_weights));
        else if (b == 5u) invalid_ids[0] = NE;
        else invalid_ids[0] = invalid_ids[1];
        require(ds4_gpu_tensor_write(storage[1][2], 0, invalid_weights, sizeof(invalid_weights)) &&
                ds4_gpu_tensor_write(storage[1][1], 0, invalid_ids, sizeof(invalid_ids)), "bad natural router upload");
        require(ds4_gpu_begin_commands(), "budget refusal batch begin");
        require(!BUDGET_CALL(rows, 2u, 8u, g_mx_test_bytes, NE, E, FF), "invalid later natural router refused");
        require(ds4_gpu_commands_active() && ds4_gpu_end_commands(), "budget refusal batch restored");
        require(ds4_gpu_tensor_read(rows[0].out, 0, input, sizeof(input)), "budget refusal sentinel read");
        check_exact_floats(input, got, E, "budget validates every natural route before dispatch");
        require(!memcmp(&measured, &zero, sizeof(measured)), "invalid routes leave no success stats");
    }
    require(ds4_gpu_tensor_write(storage[1][2], 0, route[1], sizeof(route[1])) &&
            ds4_gpu_tensor_write(storage[1][1], 0, selected[1], sizeof(selected[1])), "restore natural routes");
    require(BUDGET_CALL(rows, 2u, NE, g_mx_test_bytes, NE, E, FF), "budget owns batch when caller has none");
    require(!ds4_gpu_commands_active() && measured.truncated_slots == 0u, "owned full-union pilot finishes batch");
    for (uint32_t t = 0; t < 2u; t++) {
        require(ds4_gpu_tensor_read(rows[t].out, 0, input, sizeof(input)), "owned full-union output read");
        check_exact_floats(input, want[t], E, "owned full-union equals old verifier");
    }
    ds4_gpu_set_ssd_streaming(false);
    require(!BUDGET_CALL(rows, 2u, 8u, g_mx_test_bytes, NE, E, FF), "resident budget pilot refused");
#undef BUDGET_CALL
    ds4_gpu_tensor_free(small);
    ds4_gpu_tensor_free(snapshot);
    for (uint32_t t = 0; t < T; t++) {
        ds4_gpu_tensor_free(reference_weights[t]);
        for (uint32_t i = 0; i < 8u; i++) ds4_gpu_tensor_free(storage[t][i]);
    }
}


int main(void) {
    require(ds4_gpu_init(), "GPU initialization");
    if (getenv("DS4_TEST_MIMO2_EXPERT_ROWS_ONLY")) {
        test_streamed_mxfp4_top8();
        test_shared_mxfp4_rows();
        test_verify_expert_budget();
        goto cleanup;
    }
    const double b_global = 1e7, b_swa = 1e4;
    const float vs = 0.707f;

    printf("mini shape (H8, K64/V32, rot32), default routing\n");
    mini_sequences("", b_global, b_swa, vs);
    test_sink_extremes();

    printf("mini shape, split keys (DS4_MIMO2_ATTN_SPLIT_KEYS=3): merge with sinks\n");
    setenv("DS4_MIMO2_ATTN_SPLIT_KEYS", "3", 1);
    {
        const uint32_t chunks[] = { 7, 1, 2, 1, 11 };
        const layer_cfg swa = { "mini SWA staged split", 8, 4, 64, 32, 32, 8, 8, b_swa, vs, true, true };
        run_sequence(&swa, 60, chunks, 5, 0, 2.0f);
        const layer_cfg ga = { "mini GA split", 8, 2, 64, 32, 32, 0, 60, b_global, vs, false, false };
        run_sequence(&ga, 60, chunks, 5, 0, 2.0f);
    }
    unsetenv("DS4_MIMO2_ATTN_SPLIT_KEYS");

    printf("matrix prefill kernel forced (DS4_MIMO2_ATTN_MM_GROUPS=1)\n");
    setenv("DS4_MIMO2_ATTN_MM_GROUPS", "1", 1);
    mini_sequences(" mm", b_global, b_swa, vs);
    {
        /* Release shape: group 16 (two tokens per block) and group 8 (four),
         * partial blocks, decode rows through the same kernel. */
        const uint32_t chunks[] = { 37, 3, 1, 64, 150 };
        const layer_cfg ga = { "release GA mm", 64, 4, 192, 128, 64, 0, 600, b_global, vs, false, false };
        run_sequence(&ga, 600, chunks, 5, 250, 2.0f);
        const layer_cfg swa = { "release SWA staged mm", 64, 8, 192, 128, 64, 128, 128, b_swa, vs, true, true };
        run_sequence(&swa, 600, chunks, 5, 250, 2.0f);
    }
    unsetenv("DS4_MIMO2_ATTN_MM_GROUPS");

    printf("release shape (H64, K192/V128, rot64), default routing\n");
    {
        /* SWA as the host runs it: 128-row ring, staged prefill of 2048 then
         * 700 rows (matrix kernel), decode and a 250-row tail to 3000
         * tokens; the ring wraps across every chunk boundary. */
        const uint32_t swa_chunks[] = { 2048, 1, 700, 1, 250 };
        const layer_cfg swa = { "release SWA staged", 64, 8, 192, 128, 64, 128, 128, b_swa, vs, true, true };
        run_sequence(&swa, 3000, swa_chunks, 5, 0, 2.0f);
        /* Staged decode-kernel path: 64-row chunks (64 x 8 groups). */
        const uint32_t small_chunks[] = { 64, 64, 64, 64, 64, 1, 1, 1, 17, 1 };
        run_sequence(&swa, 470, small_chunks, 10, 256, 2.0f);
        /* GA: kv 4 (group 16), global, no sinks.  1.4k-key history, then a
         * 64-row chunk (split keys with T > 1) and decode (splits). */
        const uint32_t ga_chunks[] = { 512, 512, 376, 64, 1, 1 };
        const layer_cfg ga = { "release GA", 64, 4, 192, 128, 64, 0, 2048, b_global, vs, false, false };
        run_sequence(&ga, 1530, ga_chunks, 6, 1400, 2.0f);
        /* 256-row GA chunk (256 x 4 groups) takes the matrix kernel. */
        const uint32_t ga_pf[] = { 512, 512, 256 };
        run_sequence(&ga, 1280, ga_pf, 3, 1024, 2.0f);
    }

    printf("rope at large positions (both thetas)\n");
    {
        /* Only qkv/rope/cache is checked here: a decode token at ~1M on a
         * global layer and at ~4096 on a sliding layer, ring rows mod cap. */
        const uint32_t H = 64, Hkv = 8, K = 192, V = 128;
        const uint32_t row = H * K + Hkv * (K + V);
        const uint32_t positions[] = { 4096u, 1048575u };
        const double bases[] = { b_swa, b_global };
        ds4_gpu_tensor *g_qkv = ds4_gpu_tensor_alloc((uint64_t)row * 4u);
        ds4_gpu_tensor *g_q = ds4_gpu_tensor_alloc((uint64_t)H * K * 4u);
        ds4_gpu_tensor *g_k = ds4_gpu_tensor_alloc((uint64_t)Hkv * K * 4u);
        ds4_gpu_tensor *g_v = ds4_gpu_tensor_alloc((uint64_t)Hkv * V * 4u);
        ds4_gpu_tensor *g_kc = ds4_gpu_tensor_alloc((uint64_t)191 * Hkv * K * 2u);
        ds4_gpu_tensor *g_vc = ds4_gpu_tensor_alloc((uint64_t)191 * Hkv * V * 2u);
        require(g_qkv && g_q && g_k && g_v && g_kc && g_vc, "rope allocations");
        float *src = malloc((uint64_t)row * 4u);
        float *gq = malloc((uint64_t)H * K * 4u), *gk = malloc((uint64_t)Hkv * K * 4u);
        double *rq = malloc((uint64_t)H * K * 8u), *rk = malloc((uint64_t)Hkv * K * 8u), *rv = malloc((uint64_t)Hkv * V * 8u);
        uint16_t *kc_row = malloc((uint64_t)Hkv * K * 2u);
        for (uint32_t i = 0; i < row; i++) src[i] = frand();
        require(ds4_gpu_tensor_write(g_qkv, 0, src, (uint64_t)row * 4u), "qkv upload");
        for (int b = 0; b < 2; b++) {
            const layer_cfg c = { b ? "release GA" : "release SWA", H, Hkv, K, V, 64, 0, 191, bases[b], vs, false, false };
            float freq[32];
            rope_freqs(freq, 64, bases[b]);
            const uint32_t pos = positions[b];
            require(ds4_gpu_mimo2_qkv_rope_cache_tensor(g_q, g_k, g_v, g_kc, g_vc, g_qkv, 1, pos, 191, H, Hkv, K, V,
                                                        64, vs, freq), "rope dispatch");
            require(ds4_gpu_tensor_read(g_q, 0, gq, (uint64_t)H * K * 4u), "q read");
            require(ds4_gpu_tensor_read(g_k, 0, gk, (uint64_t)Hkv * K * 4u), "k read");
            require(ds4_gpu_tensor_read(g_kc, (uint64_t)(pos % 191u) * Hkv * K * 2u, kc_row,
                                        (uint64_t)Hkv * K * 2u), "cache row read");
            ref_qkv_row(&c, freq, src, pos, rq, rk, rv);
            char what[128];
            snprintf(what, sizeof(what), "theta %.0e q rope @%u", bases[b], pos);
            check_close(what, gq, rq, (uint64_t)H * K, 5e-6);
            snprintf(what, sizeof(what), "theta %.0e k rope @%u", bases[b], pos);
            check_close(what, gk, rk, (uint64_t)Hkv * K, 5e-6);
            for (uint32_t i = 0; i < Hkv * K; i++) {
                __fp16 h = (__fp16)gk[i];
                uint16_t want; memcpy(&want, &h, 2);
                require(kc_row[i] == want, "cache row holds the roped key at pos % cap");
            }
            /* Un-rotated dims pass through unchanged. */
            for (uint32_t h = 0; h < H; h++)
                for (uint32_t d = 64; d < K; d++)
                    require(gq[(uint64_t)h * K + d] == src[(uint64_t)h * K + d], "no-rope dims are copied");
        }
        free(src); free(gq); free(gk); free(rq); free(rk); free(rv); free(kc_row);
        ds4_gpu_tensor *t[] = { g_qkv, g_q, g_k, g_v, g_kc, g_vc };
        for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) ds4_gpu_tensor_free(t[i]);
    }

    printf("refusals\n");
    test_refusals();

    test_streamed_mxfp4_top8();
    test_shared_mxfp4_rows();
    test_verify_expert_budget();
    test_cached_mxfp4_prefill();
cleanup:
    ds4_gpu_cleanup();
    if (g_mx_test_map) munmap(g_mx_test_map, g_mx_test_bytes);
    if (g_mx_test_fd >= 0) close(g_mx_test_fd);
    printf("mimo2 kernels: all checks passed\n");
    return 0;
}
