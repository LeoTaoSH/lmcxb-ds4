#ifndef DS4_METRICS_H
#define DS4_METRICS_H

#include <stdbool.h>

/* In-process Prometheus metrics for ds4-server (GET /metrics).
 *
 * All state and text rendering live in ds4_metrics.c so that ds4_server.c
 * only carries one-line hooks at existing log/lifecycle sites: future
 * upstream changes to ds4_server.c cannot conflict with the metrics
 * implementation itself, only (at worst) with a hook call.  Series names
 * match the journald-parsing exporter historically used with this server,
 * so existing dashboards keep working when the scrape target moves here.
 *
 * Counters cover the process lifetime; throughput gauges hold the last
 * observed value.  Updates happen at the same sites that emit the
 * corresponding log lines. */

#ifdef __cplusplus
extern "C" {
#endif

/* Decode: tokens produced since the previous chunk log, the phase they
 * belong to, and the rates already computed for the log line. */
void ds4_metrics_decode_progress(int tokens, bool thinking,
                                 double chunk_tps, double avg_tps);

/* Prefill: evaluated tokens are accumulated per chunk (not in one jump at
 * prompt completion) so a long prompt never renders as an impossible
 * rate() spike between scrapes. */
void ds4_metrics_prefill_progress(int interval_tokens, double chunk_tps,
                                  double avg_tps);

/* At prompt completion: adds only the remainder the chunk callbacks have
 * not seen, plus the reused (cache-hit) token count and its source. */
void ds4_metrics_prompt_done(int remainder_tokens, int reused_tokens,
                             const char *cache_source, double seconds);

/* Exactly once per completed request, with the final finish reason. */
void ds4_metrics_request_finish(const char *reason, double seconds);

void ds4_metrics_kv_miss(const char *reason);

/* Signature matches ds4_kvstore_metrics_fn, so it registers directly with
 * ds4_kvstore_set_metrics_fn at server startup. */
void ds4_metrics_kv_stored(int stored_tokens, int trimmed_tokens,
                           double stored_mib);

/* Bracket one inference job for ds4_active_requests: from just before it
 * is enqueued until its response completes (or enqueue fails).  Inference
 * endpoints only; informational endpoints (/metrics, /v1/models) never
 * pass through these hooks. */
void ds4_metrics_job_begin(void);
void ds4_metrics_job_end(void);

/* Renders the Prometheus exposition into a freshly malloc'd,
 * NUL-terminated buffer (caller frees).  Server-state gauges are supplied
 * by the caller so this module never includes server types:
 *   ctx_limit_tokens < 0     omits ds4_context_limit_tokens
 *   slot_ctx_tokens == NULL  omits ds4_context_tokens
 * Returns NULL on allocation failure. */
char *ds4_metrics_render(int ctx_limit_tokens,
                         int slot_count, const int *slot_ctx_tokens);

#ifdef __cplusplus
}
#endif

#endif
