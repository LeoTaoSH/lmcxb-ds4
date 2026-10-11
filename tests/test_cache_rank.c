#include "ds4_cache_rank.h"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

static const double weights[DS4_CACHE_RANK_FEATURES] = {
    -0.28324965220411613, 0.2490918770777213, 0.5227117670560624,
    0.999782998528141, 0.22981648260381676, 0.0829072412440736,
    -0.1602625446313677, -0.9974721265199133, 0.19383627890883637
};

static double reference(const ds4_cache_rank_history *h, uint64_t token,
                        uint64_t clock, uint64_t last_used, uint32_t hotness) {
    const double age = h->seen ? (double)(token - h->last_token) : 0.0;
    const double feature[DS4_CACHE_RANK_FEATURES] = {
        log1p((double)(clock - last_used) / 256.0),
        log1p((double)hotness),
        log1p(h->frequency_4 * pow(2.0, -age / 4.0)),
        log1p(h->frequency_64 * pow(2.0, -age / 64.0)),
        log1p(age), log1p(h->last_gap), log1p(h->mean_gap),
        h->seen != 0, h->seen >= 2,
    };
    double result = 0.0;
    for (unsigned i = 0; i < DS4_CACHE_RANK_FEATURES; i++) result += weights[i] * feature[i];
    return result;
}

static void check(ds4_cache_rank_history *h, const double *table, uint64_t token,
                  uint64_t clock, uint64_t last_used, uint32_t hotness) {
    const double expected = reference(h, token, clock, last_used, hotness);
    const double actual = ds4_cache_rank_score(h, weights, table, token, clock, last_used, hotness);
    assert(fabs(actual - expected) < 2.0e-12);
}

int main(void) {
    const double values[] = {-DBL_MAX, -10.0, -1.0, -0.0, 0.0,
                             1.0e-300, 1.0, nextafter(1.0, 2.0), 10.0, DBL_MAX};
    for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
        for (unsigned j = 0; j < sizeof(values) / sizeof(values[0]); j++) {
            const uint64_t a = ds4_cache_rank_ordered_key(values[i]);
            const uint64_t b = ds4_cache_rank_ordered_key(values[j]);
            assert((a < b) == (values[i] < values[j]));
            assert((a == b) == (values[i] == values[j]));
        }
    }
    double *table = malloc(DS4_CACHE_RANK_RECENCY_SIZE * sizeof(*table));
    assert(table);
    for (unsigned i = 0; i < DS4_CACHE_RANK_RECENCY_SIZE; i++) table[i] = log1p(i / 256.0);

    ds4_cache_rank_history h = {0};
    check(&h, table, 1, 5000, 12, 2048);
    check(&h, table, 1, 5010, 12, 2048);
    check(&h, table, 1, 5010, 12, 1024);
    assert(h.seen == 0 && h.frequency_4 == 0.0 && h.frequency_64 == 0.0);

    ds4_cache_rank_touch(&h, 1);
    assert(h.seen == 1 && h.last_token == 1);
    assert(h.frequency_4 == 1.0 && h.frequency_64 == 1.0);
    check(&h, table, 1, 5011, 5011, 1);
    check(&h, table, 9, 6000, 5011, 1);
    check(&h, table, 17, 9000, 5011, 0);

    ds4_cache_rank_touch(&h, 9);
    assert(h.seen == 2 && h.last_gap == 8.0 && h.mean_gap == 8.0);
    assert(fabs(h.frequency_4 - 1.25) < 1.0e-14);
    assert(fabs(h.frequency_64 - (1.0 + pow(2.0, -0.125))) < 1.0e-14);
    check(&h, table, 9, 6001, 6001, 3);
    ds4_cache_rank_touch(&h, 9);
    assert(h.seen == 3 && h.last_gap == 0.0 && h.mean_gap == 6.0);
    check(&h, table, 9, 6002, 6002, 4);
    ds4_cache_rank_touch(&h, 21);
    assert(h.seen == 4 && h.last_gap == 12.0 && h.mean_gap == 7.5);
    check(&h, table, 21, 7000, 6999, 5);
    check(&h, table, 22, 7001, 6999, 5);

    check(&h, table, 1000, 70000, 4465, UINT32_MAX);
    check(&h, table, 1000, 70000, 4464, UINT32_MAX);
    check(&h, NULL, 1000, 70000, 4464, UINT32_MAX);
    h = (ds4_cache_rank_history){0};
    check(&h, table, 1, 1, 1, 0);
    assert(h.seen == 0 && h.last_gap == 0.0 && h.mean_gap == 0.0);
    free(table);
    puts("cache rank history, aging and score parity: PASS");
    return 0;
}
