#ifndef DS4_CACHE_RANK_H
#define DS4_CACHE_RANK_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

enum { DS4_CACHE_RANK_FEATURES = 9, DS4_CACHE_RANK_RECENCY_SIZE = 65536 };

typedef struct {
    uint64_t last_token;
    uint64_t seen;
    double frequency_4;
    double frequency_64;
    double last_gap;
    double mean_gap;
    uint64_t cached_token;
    uint32_t cached_hotness;
    bool cached;
    double cached_score;
} ds4_cache_rank_history;

static inline void ds4_cache_rank_touch(ds4_cache_rank_history *h, uint64_t token) {
    if (h->seen) {
        const double gap = (double)(token - h->last_token);
        h->frequency_4 *= exp2(-gap / 4.0);
        h->frequency_64 *= exp2(-gap / 64.0);
        h->last_gap = gap;
        h->mean_gap = h->seen == 1 ? gap : 0.75 * h->mean_gap + 0.25 * gap;
    }
    h->frequency_4 += 1.0;
    h->frequency_64 += 1.0;
    h->seen++;
    h->last_token = token;
    h->cached = false;
}

/* Coefficients remain fixed until the history is reset. */
static inline double ds4_cache_rank_score(
        ds4_cache_rank_history *h,
        const double weights[DS4_CACHE_RANK_FEATURES],
        const double *recency_table,
        uint64_t token, uint64_t clock, uint64_t last_used, uint32_t hotness) {
    if (!h->cached || h->cached_token != token || h->cached_hotness != hotness) {
        double score = weights[1] * log1p((double)hotness);
        if (h->seen) {
            const double age = (double)(token - h->last_token);
            score += weights[2] * log1p(h->frequency_4 * exp2(-age / 4.0)) +
                     weights[3] * log1p(h->frequency_64 * exp2(-age / 64.0)) +
                     weights[4] * log1p(age) +
                     weights[5] * log1p(h->last_gap) +
                     weights[6] * log1p(h->mean_gap) +
                     weights[7] + weights[8] * (h->seen >= 2);
        }
        h->cached_score = score;
        h->cached_token = token;
        h->cached_hotness = hotness;
        h->cached = true;
    }
    const uint64_t age = clock - last_used;
    const double clock_recency = recency_table && age < DS4_CACHE_RANK_RECENCY_SIZE ?
        recency_table[age] : log1p((double)age / 256.0);
    return weights[0] * clock_recency + h->cached_score;
}

/* Finite binary64 ordering; signed zeros remain one LRU tie. */
static inline uint64_t ds4_cache_rank_ordered_key(double score) {
    uint64_t bits;
    memcpy(&bits, &score, sizeof(bits));
    if ((bits << 1) == 0) bits = 0;
    return bits >> 63 ? ~bits : bits ^ (UINT64_C(1) << 63);
}

#endif
