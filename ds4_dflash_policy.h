#ifndef DS4_DFLASH_POLICY_H
#define DS4_DFLASH_POLICY_H

#include <stdint.h>

enum { DS4_DFLASH_POLICY_CAPS = 8 };

typedef struct {
    uint64_t calls;
    uint64_t samples[DS4_DFLASH_POLICY_CAPS];
    double milliseconds[DS4_DFLASH_POLICY_CAPS];
    double committed[DS4_DFLASH_POLICY_CAPS];
} ds4_dflash_policy;

static inline uint32_t ds4_dflash_policy_choose(const ds4_dflash_policy *p,
                                               uint32_t configured, uint32_t available) {
    if (configured >= DS4_DFLASH_POLICY_CAPS) configured = DS4_DFLASH_POLICY_CAPS - 1;
    if (available > configured) available = configured;
    uint32_t arms[4] = {0}, count = 1;
    if (available >= 1) arms[count++] = 1;
    if (available >= 3) arms[count++] = 3;
    if (configured > 1 && configured != 3 && configured <= available) arms[count++] = configured;
    uint32_t least = 0;
    for (uint32_t i = 1; i < count; i++)
        if (p->samples[arms[i]] < p->samples[arms[least]]) least = i;
    if (p->samples[arms[least]] < 2) return arms[least];
    if (p->calls % 32 == 0) return arms[(p->calls / 32) % count];
    uint32_t best = arms[0];
    double rate = p->committed[best] / p->milliseconds[best];
    for (uint32_t i = 1; i < count; i++) {
        const uint32_t arm = arms[i];
        const double candidate = p->committed[arm] / p->milliseconds[arm];
        if (candidate > rate * 1.05) {
            best = arm;
            rate = candidate;
        }
    }
    return best;
}

static inline void ds4_dflash_policy_observe(ds4_dflash_policy *p, uint32_t cap,
                                            uint32_t committed, double ms) {
    if (cap >= DS4_DFLASH_POLICY_CAPS || committed == 0 || !(ms > 0.0)) return;
    const double mix = p->samples[cap] ? 0.25 : 1.0;
    p->milliseconds[cap] += mix * (ms - p->milliseconds[cap]);
    p->committed[cap] += mix * ((double)committed - p->committed[cap]);
    p->samples[cap]++;
    p->calls++;
}

#endif
