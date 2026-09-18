#include "ds4_metrics.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_METRICS_REASONS_MAX 12
typedef struct {
    /* Sized for the longest current reason string,
     * "incoming-prompt-shorter-than-live-checkpoint" (44 bytes + NUL);
     * snprintf truncation would otherwise desynchronize the label from the
     * corresponding log line and could collide two long reasons. */
    char name[48];
    uint64_t count;
    /* Optional per-reason sum (e.g. tokens reused per cache source). */
    uint64_t value;
} metrics_reason;

#define DS4_METRICS_BUCKETS 9
/* Upper bounds (seconds) for the prompt/request latency histograms; the
 * implicit +Inf bucket is the total count. */
static const double metrics_prompt_bounds[DS4_METRICS_BUCKETS] =
    {0.5, 1, 2, 5, 10, 30, 60, 120, 300};
static const double metrics_request_bounds[DS4_METRICS_BUCKETS] =
    {1, 2, 5, 10, 30, 60, 120, 300, 600};

typedef struct {
    pthread_mutex_t mu;
    uint64_t prompt_tokens_total;
    uint64_t reused_tokens_total;
    uint64_t gen_thinking_total, gen_response_total;
    double prefill_tps_chunk, prefill_tps_avg;
    double decode_tps_chunk, decode_tps_avg;
    metrics_reason finish[DS4_METRICS_REASONS_MAX];
    double request_seconds_sum;
    uint64_t request_seconds_count;
    uint64_t request_seconds_bucket[DS4_METRICS_BUCKETS];
    double prompt_seconds_sum;
    uint64_t prompt_seconds_count;
    uint64_t prompt_seconds_bucket[DS4_METRICS_BUCKETS];
    uint64_t kv_stored_total, kv_trimmed_tokens_total;
    double kv_stored_bytes_total;
    metrics_reason kv_hit[DS4_METRICS_REASONS_MAX];
    metrics_reason kv_miss[DS4_METRICS_REASONS_MAX];
    int active_jobs;
} server_metrics;

static server_metrics g_server_metrics = { .mu = PTHREAD_MUTEX_INITIALIZER };

/* Returns the table entry for `name` (creating it), so callers accumulate
 * per-reason sums through the pointer instead of passing them as an extra
 * parameter: GCC 15.2 (-O2/-O3, x86-64) miscompiled the parameter form —
 * its IPA-CP bits lattice for a uint64_t argument fed by (uint64_t)int at
 * one call site and by constant 0 at the others concluded the low 32 bits
 * were always zero and folded 1900 into 0 (caught by the render test on
 * the ROCm build; -fno-ipa-cp made it pass).  Do not reintroduce a value
 * parameter here. */
static metrics_reason *metrics_reason_add(metrics_reason *arr,
                                          const char *name) {
    if (!name || !name[0]) name = "unknown";
    for (int i = 0; i < DS4_METRICS_REASONS_MAX; i++) {
        if (arr[i].name[0] == '\0') {
            snprintf(arr[i].name, sizeof(arr[i].name), "%s", name);
            arr[i].count = 1;
            arr[i].value = 0;
            return &arr[i];
        }
        if (!strcmp(arr[i].name, name)) {
            arr[i].count++;
            return &arr[i];
        }
    }
    /* Table full: account under the final bucket rather than dropping. */
    arr[DS4_METRICS_REASONS_MAX - 1].count++;
    return &arr[DS4_METRICS_REASONS_MAX - 1];
}

static void metrics_bucket_add(uint64_t *buckets, const double *bounds,
                               double v) {
    for (int i = 0; i < DS4_METRICS_BUCKETS; i++) {
        if (v <= bounds[i]) {
            buckets[i]++;
            return;
        }
    }
    /* Above the last bound: lands only in the implicit +Inf bucket (the
     * total count), which render derives from _count. */
}

void ds4_metrics_decode_progress(int tokens, bool thinking,
                                 double chunk_tps, double avg_tps) {
    if (tokens < 0) tokens = 0;
    pthread_mutex_lock(&g_server_metrics.mu);
    if (thinking) g_server_metrics.gen_thinking_total += (uint64_t)tokens;
    else g_server_metrics.gen_response_total += (uint64_t)tokens;
    g_server_metrics.decode_tps_chunk = chunk_tps;
    g_server_metrics.decode_tps_avg = avg_tps;
    pthread_mutex_unlock(&g_server_metrics.mu);
}

void ds4_metrics_prefill_progress(int interval_tokens, double chunk_tps,
                                  double avg_tps) {
    if (interval_tokens < 0) interval_tokens = 0;
    pthread_mutex_lock(&g_server_metrics.mu);
    g_server_metrics.prompt_tokens_total += (uint64_t)interval_tokens;
    g_server_metrics.prefill_tps_chunk = chunk_tps;
    g_server_metrics.prefill_tps_avg = avg_tps;
    pthread_mutex_unlock(&g_server_metrics.mu);
}

void ds4_metrics_prompt_done(int remainder_tokens, int reused_tokens,
                             const char *cache_source, double seconds) {
    int evaluated_tokens = remainder_tokens;
    if (evaluated_tokens < 0) evaluated_tokens = 0;
    if (reused_tokens < 0) reused_tokens = 0;
    pthread_mutex_lock(&g_server_metrics.mu);
    g_server_metrics.prompt_tokens_total += (uint64_t)evaluated_tokens;
    g_server_metrics.reused_tokens_total += (uint64_t)reused_tokens;
    if (reused_tokens > 0) {
        metrics_reason *hit =
            metrics_reason_add(g_server_metrics.kv_hit, cache_source);
        hit->value += (uint64_t)reused_tokens;
    }
    g_server_metrics.prompt_seconds_sum += seconds;
    g_server_metrics.prompt_seconds_count++;
    metrics_bucket_add(g_server_metrics.prompt_seconds_bucket,
                       metrics_prompt_bounds, seconds);
    pthread_mutex_unlock(&g_server_metrics.mu);
}

void ds4_metrics_request_finish(const char *reason, double seconds) {
    pthread_mutex_lock(&g_server_metrics.mu);
    metrics_reason_add(g_server_metrics.finish, reason);
    g_server_metrics.request_seconds_sum += seconds;
    g_server_metrics.request_seconds_count++;
    metrics_bucket_add(g_server_metrics.request_seconds_bucket,
                       metrics_request_bounds, seconds);
    pthread_mutex_unlock(&g_server_metrics.mu);
}

void ds4_metrics_kv_miss(const char *reason) {
    pthread_mutex_lock(&g_server_metrics.mu);
    metrics_reason_add(g_server_metrics.kv_miss, reason);
    pthread_mutex_unlock(&g_server_metrics.mu);
}

void ds4_metrics_kv_stored(int stored_tokens, int trimmed_tokens,
                           double stored_mib) {
    (void)stored_tokens;
    pthread_mutex_lock(&g_server_metrics.mu);
    g_server_metrics.kv_stored_total++;
    if (trimmed_tokens > 0)
        g_server_metrics.kv_trimmed_tokens_total += (uint64_t)trimmed_tokens;
    g_server_metrics.kv_stored_bytes_total += stored_mib * 1024.0 * 1024.0;
    pthread_mutex_unlock(&g_server_metrics.mu);
}

void ds4_metrics_job_begin(void) {
    pthread_mutex_lock(&g_server_metrics.mu);
    g_server_metrics.active_jobs++;
    pthread_mutex_unlock(&g_server_metrics.mu);
}

void ds4_metrics_job_end(void) {
    pthread_mutex_lock(&g_server_metrics.mu);
    if (g_server_metrics.active_jobs > 0) g_server_metrics.active_jobs--;
    pthread_mutex_unlock(&g_server_metrics.mu);
}

/* ---------------- Rendering ---------------- */

typedef struct {
    char *ptr;
    size_t len, cap;
    bool oom;
} mbuf;

static void mbuf_reserve(mbuf *b, size_t add) {
    if (b->oom) return;
    if (b->len + add + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < b->len + add + 1) cap *= 2;
    char *p = realloc(b->ptr, cap);
    if (!p) { b->oom = true; free(b->ptr); b->ptr = NULL; b->len = b->cap = 0; return; }
    b->ptr = p;
    b->cap = cap;
}

static void mbuf_puts(mbuf *b, const char *s) {
    size_t n = strlen(s);
    mbuf_reserve(b, n);
    if (b->oom) return;
    memcpy(b->ptr + b->len, s, n + 1);
    b->len += n;
}

static void mbuf_printf(mbuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[256];
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(tmp)) {
        mbuf_puts(b, tmp);
        return;
    }
    mbuf_reserve(b, (size_t)n);
    if (b->oom) return;
    va_start(ap, fmt);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap);
    va_end(ap);
    b->len += (size_t)n;
}

char *ds4_metrics_render(int ctx_limit_tokens,
                         int slot_count, const int *slot_ctx_tokens) {
    server_metrics snap;
    pthread_mutex_lock(&g_server_metrics.mu);
    snap = g_server_metrics; /* struct copy; snap.mu is never used */
    const int active_jobs = g_server_metrics.active_jobs;
    pthread_mutex_unlock(&g_server_metrics.mu);

    mbuf b = {0};
    mbuf_puts(&b, "# HELP ds4_prompt_tokens_total Prompt tokens evaluated by prefill\n"
                  "# TYPE ds4_prompt_tokens_total counter\n");
    mbuf_printf(&b, "ds4_prompt_tokens_total %llu\n",
                (unsigned long long)snap.prompt_tokens_total);
    mbuf_puts(&b, "# HELP ds4_generated_tokens_total Tokens generated, by phase\n"
                  "# TYPE ds4_generated_tokens_total counter\n");
    mbuf_printf(&b, "ds4_generated_tokens_total{phase=\"thinking\"} %llu\n",
                (unsigned long long)snap.gen_thinking_total);
    mbuf_printf(&b, "ds4_generated_tokens_total{phase=\"response\"} %llu\n",
                (unsigned long long)snap.gen_response_total);
    mbuf_puts(&b, "# HELP ds4_prefill_tps Prefill tokens/s (last chunk, running avg)\n"
                  "# TYPE ds4_prefill_tps gauge\n");
    mbuf_printf(&b, "ds4_prefill_tps{window=\"chunk\"} %.2f\n", snap.prefill_tps_chunk);
    mbuf_printf(&b, "ds4_prefill_tps{window=\"avg\"} %.2f\n", snap.prefill_tps_avg);
    mbuf_puts(&b, "# HELP ds4_decode_tps Decode tokens/s (last chunk, running avg)\n"
                  "# TYPE ds4_decode_tps gauge\n");
    mbuf_printf(&b, "ds4_decode_tps{window=\"chunk\"} %.2f\n", snap.decode_tps_chunk);
    mbuf_printf(&b, "ds4_decode_tps{window=\"avg\"} %.2f\n", snap.decode_tps_avg);
    mbuf_puts(&b, "# HELP ds4_requests_total Completed requests by finish reason\n"
                  "# TYPE ds4_requests_total counter\n");
    for (int i = 0; i < DS4_METRICS_REASONS_MAX; i++) {
        if (!snap.finish[i].name[0]) break;
        mbuf_printf(&b, "ds4_requests_total{finish=\"%s\"} %llu\n",
                    snap.finish[i].name, (unsigned long long)snap.finish[i].count);
    }
    mbuf_puts(&b, "# HELP ds4_reused_tokens_total Prompt tokens served from KV cache reuse\n"
                  "# TYPE ds4_reused_tokens_total counter\n");
    mbuf_printf(&b, "ds4_reused_tokens_total %llu\n",
                (unsigned long long)snap.reused_tokens_total);
    mbuf_puts(&b, "# HELP ds4_kv_cache_hit_total Prompts that reused cached KV, by source\n"
                  "# TYPE ds4_kv_cache_hit_total counter\n");
    for (int i = 0; i < DS4_METRICS_REASONS_MAX; i++) {
        if (!snap.kv_hit[i].name[0]) break;
        mbuf_printf(&b, "ds4_kv_cache_hit_total{source=\"%s\"} %llu\n",
                    snap.kv_hit[i].name, (unsigned long long)snap.kv_hit[i].count);
    }
    mbuf_puts(&b, "# HELP ds4_request_duration_seconds Wall time per request\n"
                  "# TYPE ds4_request_duration_seconds histogram\n");
    uint64_t cum = 0;
    for (int i = 0; i < DS4_METRICS_BUCKETS; i++) {
        cum += snap.request_seconds_bucket[i];
        mbuf_printf(&b, "ds4_request_duration_seconds_bucket{le=\"%g\"} %llu\n",
                    metrics_request_bounds[i], (unsigned long long)cum);
    }
    mbuf_printf(&b, "ds4_request_duration_seconds_bucket{le=\"+Inf\"} %llu\n",
                (unsigned long long)snap.request_seconds_count);
    mbuf_printf(&b, "ds4_request_duration_seconds_sum %.3f\n", snap.request_seconds_sum);
    mbuf_printf(&b, "ds4_request_duration_seconds_count %llu\n",
                (unsigned long long)snap.request_seconds_count);
    mbuf_puts(&b, "# HELP ds4_prompt_duration_seconds Wall time per prompt prefill\n"
                  "# TYPE ds4_prompt_duration_seconds histogram\n");
    cum = 0;
    for (int i = 0; i < DS4_METRICS_BUCKETS; i++) {
        cum += snap.prompt_seconds_bucket[i];
        mbuf_printf(&b, "ds4_prompt_duration_seconds_bucket{le=\"%g\"} %llu\n",
                    metrics_prompt_bounds[i], (unsigned long long)cum);
    }
    mbuf_printf(&b, "ds4_prompt_duration_seconds_bucket{le=\"+Inf\"} %llu\n",
                (unsigned long long)snap.prompt_seconds_count);
    mbuf_printf(&b, "ds4_prompt_duration_seconds_sum %.3f\n", snap.prompt_seconds_sum);
    mbuf_printf(&b, "ds4_prompt_duration_seconds_count %llu\n",
                (unsigned long long)snap.prompt_seconds_count);
    mbuf_puts(&b, "# HELP ds4_kv_cache_stored_total Successful KV cache stores\n"
                  "# TYPE ds4_kv_cache_stored_total counter\n");
    mbuf_printf(&b, "ds4_kv_cache_stored_total %llu\n",
                (unsigned long long)snap.kv_stored_total);
    mbuf_puts(&b, "# HELP ds4_kv_cache_stored_bytes_total Bytes written by KV cache stores\n"
                  "# TYPE ds4_kv_cache_stored_bytes_total counter\n");
    mbuf_printf(&b, "ds4_kv_cache_stored_bytes_total %.1f\n", snap.kv_stored_bytes_total);
    mbuf_puts(&b, "# HELP ds4_kv_cache_trimmed_tokens_total Tokens trimmed before store\n"
                  "# TYPE ds4_kv_cache_trimmed_tokens_total counter\n");
    mbuf_printf(&b, "ds4_kv_cache_trimmed_tokens_total %llu\n",
                (unsigned long long)snap.kv_trimmed_tokens_total);
    mbuf_puts(&b, "# HELP ds4_kv_cache_miss_total Live KV cache misses by reason\n"
                  "# TYPE ds4_kv_cache_miss_total counter\n");
    for (int i = 0; i < DS4_METRICS_REASONS_MAX; i++) {
        if (!snap.kv_miss[i].name[0]) break;
        mbuf_printf(&b, "ds4_kv_cache_miss_total{reason=\"%s\"} %llu\n",
                    snap.kv_miss[i].name, (unsigned long long)snap.kv_miss[i].count);
    }
    mbuf_puts(&b, "# HELP ds4_reused_tokens_by_source_total Prompt tokens served from KV cache reuse, by source\n"
                  "# TYPE ds4_reused_tokens_by_source_total counter\n");
    for (int i = 0; i < DS4_METRICS_REASONS_MAX; i++) {
        if (!snap.kv_hit[i].name[0]) break;
        mbuf_printf(&b, "ds4_reused_tokens_by_source_total{source=\"%s\"} %llu\n",
                    snap.kv_hit[i].name, (unsigned long long)snap.kv_hit[i].value);
    }
    mbuf_puts(&b, "# HELP ds4_active_requests Inference requests in flight (queued or generating); excludes /metrics and other informational endpoints\n"
                  "# TYPE ds4_active_requests gauge\n");
    mbuf_printf(&b, "ds4_active_requests %d\n", active_jobs);
    if (ctx_limit_tokens >= 0) {
        mbuf_puts(&b, "# HELP ds4_context_limit_tokens Configured context size\n"
                      "# TYPE ds4_context_limit_tokens gauge\n");
        mbuf_printf(&b, "ds4_context_limit_tokens %d\n", ctx_limit_tokens);
    }
    if (slot_ctx_tokens) {
        mbuf_puts(&b, "# HELP ds4_context_tokens Tokens resident per decode slot\n"
                      "# TYPE ds4_context_tokens gauge\n");
        for (int i = 0; i < slot_count; i++) {
            mbuf_printf(&b, "ds4_context_tokens{slot=\"%d\"} %d\n",
                        i, slot_ctx_tokens[i]);
        }
    }
    if (b.oom) { free(b.ptr); return NULL; }
    if (!b.ptr) {
        b.ptr = malloc(1);
        if (b.ptr) b.ptr[0] = '\0';
        return b.ptr;
    }
    return b.ptr;
}
