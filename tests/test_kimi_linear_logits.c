/* Compare Kimi's CUDA split-KV decode attention against the retained dense
 * reference path from an identical serialized session state.
 *
 * Usage: test_kimi_linear_logits MODEL [FRONTIER ...]
 * The default frontiers deliberately cross irregular split boundaries.
 */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c, ...) do {                                                   \
    if (!(c)) {                                                              \
        fprintf(stderr, "FAIL: ");                                          \
        fprintf(stderr, __VA_ARGS__);                                        \
        fputc('\n', stderr);                                                 \
        goto cleanup;                                                        \
    }                                                                        \
} while (0)

static int argmax_f32(const float *values, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}

static void top_k(const float *values, int n, int *out, int k) {
    for (int j = 0; j < k; j++) out[j] = -1;
    for (int i = 0; i < n; i++) {
        int at = k;
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || values[i] > values[out[j]]) {
                at = j;
                break;
            }
        }
        if (at == k) continue;
        for (int j = k - 1; j > at; j--) out[j] = out[j - 1];
        out[at] = i;
    }
}

static int same_set(const int *a, const int *b, int n) {
    for (int i = 0; i < n; i++) {
        int found = 0;
        for (int j = 0; j < n; j++) found |= a[i] == b[j];
        if (!found) return 0;
    }
    return 1;
}

static float env_float(const char *name, float fallback) {
    const char *text = getenv(name);
    if (!text || !text[0]) return fallback;
    char *end = NULL;
    errno = 0;
    float value = strtof(text, &end);
    if (errno || !end || *end || !isfinite(value) || value < 0.0f) {
        fprintf(stderr, "invalid %s=%s\n", name, text);
        exit(2);
    }
    return value;
}

typedef struct {
    double mean_abs;
    double rms;
    double tv;
    float max_abs;
    int worst;
    int lhs_argmax;
    int rhs_argmax;
    int top5_same;
    int finite;
} logit_metrics;

static logit_metrics compare_logits(const float *lhs, const float *rhs,
                                    int vocab) {
    logit_metrics result = {.worst = -1, .finite = 1};
    double squared = 0.0, absolute = 0.0;
    float lhs_max = -INFINITY, rhs_max = -INFINITY;
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(lhs[i]) || !isfinite(rhs[i])) result.finite = 0;
        const float delta = fabsf(lhs[i] - rhs[i]);
        squared += (double)delta * delta;
        absolute += delta;
        if (delta > result.max_abs) {
            result.max_abs = delta;
            result.worst = i;
        }
        lhs_max = fmaxf(lhs_max, lhs[i]);
        rhs_max = fmaxf(rhs_max, rhs[i]);
    }
    double lhs_z = 0.0, rhs_z = 0.0;
    for (int i = 0; i < vocab; i++) {
        lhs_z += exp((double)lhs[i] - lhs_max);
        rhs_z += exp((double)rhs[i] - rhs_max);
    }
    const double lhs_lz = lhs_max + log(lhs_z);
    const double rhs_lz = rhs_max + log(rhs_z);
    for (int i = 0; i < vocab; i++) {
        const double a = exp((double)lhs[i] - lhs_lz);
        const double b = exp((double)rhs[i] - rhs_lz);
        result.tv += 0.5 * fabs(a - b);
    }
    int lhs_top[5], rhs_top[5];
    top_k(lhs, vocab, lhs_top, 5);
    top_k(rhs, vocab, rhs_top, 5);
    result.lhs_argmax = argmax_f32(lhs, vocab);
    result.rhs_argmax = argmax_f32(rhs, vocab);
    result.top5_same = same_set(lhs_top, rhs_top, 5);
    result.mean_abs = absolute / vocab;
    result.rms = sqrt(squared / vocab);
    return result;
}

int main(int argc, char **argv) {
    static const int defaults[] = {1025, 1537, 4097, 8191};
    int *frontiers = NULL;
    int frontier_count = argc > 2 ? argc - 2 :
        (int)(sizeof(defaults) / sizeof(defaults[0]));
    int max_frontier = 0;
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_session *prefill_reference = NULL;
    ds4_tokens seed = {0}, tokens = {0};
    float *split_logits = NULL, *reference_logits = NULL;
    ds4_session_snapshot snapshot = {0};
    char error[256] = {0};

    if (argc < 2) {
        fprintf(stderr, "Usage: %s MODEL [FRONTIER ...]\n", argv[0]);
        return 2;
    }
    frontiers = calloc((size_t)frontier_count, sizeof(*frontiers));
    CHECK(frontiers != NULL, "frontier allocation");
    for (int i = 0; i < frontier_count; i++) {
        if (argc > 2) {
            char *end = NULL;
            long value = strtol(argv[i + 2], &end, 10);
            CHECK(end && !*end && value >= 1024 && value <= 131072,
                  "invalid frontier: %s", argv[i + 2]);
            frontiers[i] = (int)value;
        } else {
            frontiers[i] = defaults[i];
        }
        CHECK(i == 0 || frontiers[i] > frontiers[i - 1],
              "frontiers must be strictly increasing");
        max_frontier = frontiers[i];
    }

    ds4_engine_options options = {
        .model_path = argv[1],
        .backend = DS4_BACKEND_CUDA,
        .context_size = max_frontier + 8,
        .n_threads = 1,
    };
    CHECK(ds4_engine_open(&engine, &options) == 0, "engine open");
    CHECK(strstr(ds4_engine_model_name(engine), "Kimi Linear") != NULL,
          "model is not Kimi Linear: %s", ds4_engine_model_name(engine));
    CHECK(ds4_session_create(&session, engine, max_frontier + 8) == 0,
          "session creation");

    ds4_tokenize_text(engine,
        " The verifier checks every irregular boundary and preserves state.",
        &seed);
    CHECK(seed.len > 0, "test prompt tokenization");
    for (int i = 0; i < max_frontier; i++) {
        ds4_tokens_push(&tokens, seed.v[i % seed.len]);
    }

    const int vocab = ds4_engine_vocab_size(engine);
    split_logits = malloc((size_t)vocab * sizeof(*split_logits));
    reference_logits = malloc((size_t)vocab * sizeof(*reference_logits));
    CHECK(split_logits && reference_logits, "logit allocation");
    const float max_abs_limit = env_float("DS4_KIMI_LOGIT_MAX_ABS", 2.5f);
    const float tv_limit = env_float("DS4_KIMI_LOGIT_MAX_TV", 1.0e-3f);

    /* Compare the new query/KV-tiled batch prefill against the retained
     * per-query CUDA kernel on real model logits.  The synthetic attention
     * test below the model layer catches indexing errors; this catches any
     * error amplified by the complete residual stack. */
    ds4_tokens prefill_prefix = tokens;
    prefill_prefix.len = frontiers[0] - 1;
    unsetenv("DS4_KIMI_DISABLE_BATCH_PREFILL");
    unsetenv("DS4_CUDA_KIMI_NO_SPLIT_ATTN");
    unsetenv("DS4_CUDA_GLM_NO_TILED_MLA_ATTN");
    CHECK(ds4_session_sync(session, &prefill_prefix,
                           error, sizeof(error)) == 0,
          "tiled prefill to %d: %s", prefill_prefix.len, error);
    CHECK(ds4_session_copy_logits(session, split_logits, vocab) == vocab,
          "tiled prefill logits at %d", prefill_prefix.len);
    CHECK(ds4_session_create(&prefill_reference, engine,
                             max_frontier + 8) == 0,
          "reference prefill session creation");
    setenv("DS4_CUDA_GLM_NO_TILED_MLA_ATTN", "1", 1);
    CHECK(ds4_session_sync(prefill_reference, &prefill_prefix,
                           error, sizeof(error)) == 0,
          "reference prefill to %d: %s", prefill_prefix.len, error);
    CHECK(ds4_session_copy_logits(prefill_reference, reference_logits,
                                  vocab) == vocab,
          "reference prefill logits at %d", prefill_prefix.len);
    logit_metrics prefill = compare_logits(split_logits, reference_logits,
                                           vocab);
    printf("prefill=%d tiled/reference max_abs=%.9g worst=%d "
           "mean_abs=%.9g rms=%.9g tv=%.9g top1=%d/%d top5_same=%d\n",
           prefill_prefix.len, prefill.max_abs, prefill.worst,
           prefill.mean_abs, prefill.rms, prefill.tv,
           prefill.lhs_argmax, prefill.rhs_argmax, prefill.top5_same);
    fflush(stdout);
    CHECK(prefill.finite, "non-finite tiled/reference prefill logit");
    CHECK(prefill.max_abs <= max_abs_limit,
          "prefill max_abs %.9g exceeds %.9g",
          prefill.max_abs, max_abs_limit);
    CHECK(prefill.tv <= tv_limit, "prefill TV %.9g exceeds %.9g",
          prefill.tv, tv_limit);
    CHECK(prefill.lhs_argmax == prefill.rhs_argmax,
          "prefill top token differs: %d vs %d",
          prefill.lhs_argmax, prefill.rhs_argmax);
    ds4_session_free(prefill_reference);
    prefill_reference = NULL;
    unsetenv("DS4_CUDA_GLM_NO_TILED_MLA_ATTN");

    for (int f = 0; f < frontier_count; f++) {
        const int frontier = frontiers[f];
        ds4_tokens prefix = tokens;
        prefix.len = frontier - 1;
        CHECK(ds4_session_sync(session, &prefix, error, sizeof(error)) == 0,
              "prefill to %d: %s", frontier - 1, error);
        CHECK(ds4_session_save_snapshot(session, &snapshot,
                                        error, sizeof(error)) == 0,
              "snapshot at %d: %s", frontier - 1, error);

        unsetenv("DS4_CUDA_KIMI_NO_SPLIT_ATTN");
        CHECK(ds4_session_eval(session, tokens.v[frontier - 1],
                               error, sizeof(error)) == 0,
              "split-KV eval at %d: %s", frontier, error);
        CHECK(ds4_session_copy_logits(session, split_logits, vocab) == vocab,
              "split-KV logits at %d", frontier);

        CHECK(ds4_session_load_snapshot(session, &snapshot,
                                        error, sizeof(error)) == 0,
              "restore before reference at %d: %s", frontier, error);
        setenv("DS4_CUDA_KIMI_NO_SPLIT_ATTN", "1", 1);
        CHECK(ds4_session_eval(session, tokens.v[frontier - 1],
                               error, sizeof(error)) == 0,
              "reference eval at %d: %s", frontier, error);
        CHECK(ds4_session_copy_logits(session, reference_logits, vocab) == vocab,
              "reference logits at %d", frontier);

        double squared = 0.0, absolute = 0.0;
        float max_abs = 0.0f;
        int worst = -1;
        float split_max = -INFINITY, reference_max = -INFINITY;
        for (int i = 0; i < vocab; i++) {
            CHECK(isfinite(split_logits[i]) && isfinite(reference_logits[i]),
                  "non-finite logit at frontier %d index %d", frontier, i);
            const float delta = fabsf(split_logits[i] - reference_logits[i]);
            squared += (double)delta * delta;
            absolute += delta;
            if (delta > max_abs) {
                max_abs = delta;
                worst = i;
            }
            split_max = fmaxf(split_max, split_logits[i]);
            reference_max = fmaxf(reference_max, reference_logits[i]);
        }
        double split_z = 0.0, reference_z = 0.0;
        for (int i = 0; i < vocab; i++) {
            split_z += exp((double)split_logits[i] - split_max);
            reference_z += exp((double)reference_logits[i] - reference_max);
        }
        const double split_lz = split_max + log(split_z);
        const double reference_lz = reference_max + log(reference_z);
        double tv = 0.0;
        for (int i = 0; i < vocab; i++) {
            const double a = exp((double)split_logits[i] - split_lz);
            const double b = exp((double)reference_logits[i] - reference_lz);
            tv += 0.5 * fabs(a - b);
        }
        int split_top[5], reference_top[5];
        top_k(split_logits, vocab, split_top, 5);
        top_k(reference_logits, vocab, reference_top, 5);
        const int split_argmax = argmax_f32(split_logits, vocab);
        const int reference_argmax = argmax_f32(reference_logits, vocab);
        printf("ctx=%d max_abs=%.9g worst=%d mean_abs=%.9g rms=%.9g "
               "tv=%.9g top1=%d/%d top5_same=%d\n",
               frontier, max_abs, worst, absolute / vocab,
               sqrt(squared / vocab), tv, split_argmax, reference_argmax,
               same_set(split_top, reference_top, 5));
        fflush(stdout);
        CHECK(max_abs <= max_abs_limit,
              "frontier %d max_abs %.9g exceeds %.9g",
              frontier, max_abs, max_abs_limit);
        CHECK(tv <= tv_limit, "frontier %d TV %.9g exceeds %.9g",
              frontier, tv, tv_limit);
        CHECK(split_argmax == reference_argmax,
              "frontier %d top token differs: %d vs %d",
              frontier, split_argmax, reference_argmax);

        /* Continue the next prefill from the exact pre-decode state rather
         * than allowing either reduction order to influence later caches. */
        CHECK(ds4_session_load_snapshot(session, &snapshot,
                                        error, sizeof(error)) == 0,
              "restore after comparison at %d: %s", frontier, error);
        ds4_session_snapshot_free(&snapshot);
        unsetenv("DS4_CUDA_KIMI_NO_SPLIT_ATTN");
    }

    printf("Kimi split-KV full-logit comparison: PASS (%d frontiers)\n",
           frontier_count);
    rc = 0;

cleanup:
    unsetenv("DS4_CUDA_KIMI_NO_SPLIT_ATTN");
    unsetenv("DS4_CUDA_GLM_NO_TILED_MLA_ATTN");
    ds4_session_snapshot_free(&snapshot);
    free(reference_logits);
    free(split_logits);
    ds4_tokens_free(&tokens);
    ds4_tokens_free(&seed);
    ds4_session_free(prefill_reference);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(frontiers);
    return rc;
}
