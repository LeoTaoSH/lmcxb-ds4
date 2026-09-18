/* Model-free render test for ds4_metrics: feeds the hooks with known values
 * and asserts exact exposition lines.  Runs anywhere (no GPU, no model). */
#include "../ds4_metrics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "test_metrics: FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static void check_has(const char *text, const char *line) {
    if (!strstr(text, line)) {
        fprintf(stderr, "test_metrics: FAIL missing line: %s", line);
        failures++;
    }
}

int main(void) {
    ds4_metrics_prefill_progress(2748, 1900.0, 1900.0);
    ds4_metrics_prefill_tokens(51); /* body handled in 51 chunks of 1,000 */
    ds4_metrics_prefill_tokens(0);  /* no-op */
    ds4_metrics_prefill_tokens(-5); /* never removes tokens */
    ds4_metrics_prompt_done(91, 0, "none", 60.872);
    ds4_metrics_prompt_done(100, 1900, "memory-rewind", 0.4);
    ds4_metrics_decode_progress(50, true, 14.0, 14.0);
    ds4_metrics_decode_progress(39, false, 13.5, 13.8);
    ds4_metrics_request_finish("stop", 129.107);
    ds4_metrics_request_finish("stop", 10.0);
    ds4_metrics_request_finish("length", 5.0);
    ds4_metrics_kv_miss("token-mismatch");
    ds4_metrics_kv_stored(1288, 12, 39.79);
    ds4_metrics_job_begin();
    ds4_metrics_job_begin();
    ds4_metrics_job_end();
    ds4_metrics_job_end();
    ds4_metrics_job_end(); /* never goes negative */

    int slots[2] = {2839, 0};
    char *text = ds4_metrics_render(393216, 2, slots);
    CHECK(text != NULL);
    if (!text) return 1;

    check_has(text, "ds4_prompt_tokens_total 2990\n");
    check_has(text, "ds4_reused_tokens_total 1900\n");
    check_has(text, "ds4_kv_cache_hit_total{source=\"memory-rewind\"} 1\n");
    check_has(text, "ds4_reused_tokens_by_source_total{source=\"memory-rewind\"} 1900\n");
    check_has(text, "ds4_generated_tokens_total{phase=\"thinking\"} 50\n");
    check_has(text, "ds4_generated_tokens_total{phase=\"response\"} 39\n");
    check_has(text, "ds4_prefill_tps{window=\"chunk\"} 1900.00\n");
    check_has(text, "ds4_decode_tps{window=\"avg\"} 13.80\n");
    check_has(text, "ds4_requests_total{finish=\"stop\"} 2\n");
    check_has(text, "ds4_requests_total{finish=\"length\"} 1\n");
    check_has(text, "ds4_request_duration_seconds_bucket{le=\"+Inf\"} 3\n");
    check_has(text, "ds4_request_duration_seconds_count 3\n");
    check_has(text, "ds4_prompt_duration_seconds_count 2\n");
    check_has(text, "ds4_kv_cache_stored_total 1\n");
    check_has(text, "ds4_kv_cache_trimmed_tokens_total 12\n");
    check_has(text, "ds4_kv_cache_miss_total{reason=\"token-mismatch\"} 1\n");
    check_has(text, "ds4_active_requests 0\n");
    check_has(text, "ds4_context_limit_tokens 393216\n");
    check_has(text, "ds4_context_tokens{slot=\"0\"} 2839\n");
    /* bucket sanity: 129.107s request lands in the le=300 bucket */
    check_has(text, "ds4_request_duration_seconds_bucket{le=\"300\"} 3\n");
    check_has(text, "ds4_request_duration_seconds_bucket{le=\"600\"} 3\n");
    check_has(text, "ds4_request_duration_seconds_bucket{le=\"120\"} 2\n");
    free(text);

    /* Render with no server-state gauges: context series omitted. */
    text = ds4_metrics_render(-1, 0, NULL);
    CHECK(text != NULL);
    if (text) {
        CHECK(strstr(text, "ds4_context_limit_tokens") == NULL);
        CHECK(strstr(text, "ds4_context_tokens") == NULL);
        free(text);
    }

    if (failures) {
        fprintf(stderr, "test_metrics: %d failures\n", failures);
        return 1;
    }
    printf("test_metrics: ok\n");
    return 0;
}
