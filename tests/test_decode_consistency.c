#define main ds4_cli_main
#include "../ds4_cli.c"
#undef main
#include <assert.h>

int main(void) {
    float logits[] = {0.0f, 0.0f, 0.0f};
    float lo, hi;
    assert(!diagnostic_logits_range(logits, 3, &lo, &hi));
    logits[0] = logits[1] = logits[2] = 12.0f;
    assert(!diagnostic_logits_range(logits, 3, &lo, &hi));
    logits[0] = -4.0f;
    logits[1] = 2.0f;
    logits[2] = 0.0f;
    assert(diagnostic_logits_range(logits, 3, &lo, &hi));
    assert(lo == -4.0f && hi == 2.0f);
    logits[2] = NAN;
    assert(!diagnostic_logits_range(logits, 3, &lo, &hi));
    logits[2] = INFINITY;
    assert(!diagnostic_logits_range(logits, 3, &lo, &hi));
    logits[0] = NAN;
    assert(!diagnostic_logits_range(logits, 3, &lo, &hi));
    puts("Decode consistency rejects degenerate/nonfinite logits: PASS");
    return 0;
}
