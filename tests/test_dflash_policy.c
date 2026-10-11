#include "ds4_dflash_policy.h"
#include <assert.h>
#include <stdio.h>

static void seed(ds4_dflash_policy *p, uint32_t cap, uint32_t committed, double ms) {
    ds4_dflash_policy_observe(p, cap, committed, ms);
    ds4_dflash_policy_observe(p, cap, committed, ms);
}

int main(void) {
    ds4_dflash_policy p = {0};
    const uint32_t expected[] = {0, 1, 3, 7, 0, 1, 3, 7};
    const uint32_t committed[] = {1, 2, 0, 4, 0, 0, 0, 8};
    const double cost[] = {8, 12, 0, 16, 0, 0, 0, 64};
    for (unsigned i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
        uint32_t cap = ds4_dflash_policy_choose(&p, 7, 7);
        assert(cap == expected[i]);
        ds4_dflash_policy_observe(&p, cap, committed[cap], cost[cap]);
    }
    assert(ds4_dflash_policy_choose(&p, 7, 7) == 3);
    assert(ds4_dflash_policy_choose(&p, 7, 6) == 3);
    assert(ds4_dflash_policy_choose(&p, 7, 2) == 1);
    assert(ds4_dflash_policy_choose(&p, 1, 1) == 1);
    assert(ds4_dflash_policy_choose(&p, 0, 0) == 0);
    assert(ds4_dflash_policy_choose(&p, 2, 2) == 2);
    seed(&p, 2, 2, 40);
    assert(ds4_dflash_policy_choose(&p, 2, 2) == 1);

    for (int i = 0; i < 4; i++)
        ds4_dflash_policy_observe(&p, 3, 4, 400);
    assert(ds4_dflash_policy_choose(&p, 7, 7) == 1);

    p = (ds4_dflash_policy){0};
    seed(&p, 0, 1, 10);
    seed(&p, 1, 2, 19.3);
    seed(&p, 3, 4, 38.8);
    seed(&p, 7, 8, 77.5);
    assert(ds4_dflash_policy_choose(&p, 7, 7) == 0);

    p = (ds4_dflash_policy){0};
    seed(&p, 0, 1, 100);
    seed(&p, 1, 2, 100);
    seed(&p, 3, 4, 100);
    seed(&p, 7, 8, 1);
    unsigned explored = 0;
    for (unsigned i = 0; i < 256; i++) {
        const uint32_t cap = ds4_dflash_policy_choose(&p, 7, 7);
        assert(cap <= 7);
        explored |= 1u << cap;
        ds4_dflash_policy_observe(&p, cap, cap + 1, cap == 7 ? 1.0 : 100.0);
    }
    assert(explored == ((1u << 0) | (1u << 1) | (1u << 3) | (1u << 7)));

    p = (ds4_dflash_policy){0};
    ds4_dflash_policy_observe(&p, 8, 1, 1);
    ds4_dflash_policy_observe(&p, 1, 0, 1);
    ds4_dflash_policy_observe(&p, 1, 1, 0);
    assert(p.calls == 0);
    assert(ds4_dflash_policy_choose(&p, 7, 7) == 0);
    puts("DFlash cost adaptation, cap bounds and exploration: PASS");
    return 0;
}
