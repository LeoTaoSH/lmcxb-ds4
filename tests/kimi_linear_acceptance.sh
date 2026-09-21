#!/bin/sh
set -eu

model=${DS4_KIMI_MODEL:?set DS4_KIMI_MODEL to the Kimi Linear Q8_0 GGUF}
ds4_bin=${DS4_KIMI_DS4_BIN:-./ds4}
prompt='Count from one to ten, separated by commas.'
fast_log=$(mktemp "${TMPDIR:-/tmp}/ds4-kimi-fast.XXXXXX")
exact_log=$(mktemp "${TMPDIR:-/tmp}/ds4-kimi-exact.XXXXXX")
trap 'rm -f "$fast_log" "$exact_log"' EXIT HUP INT TERM

run_check() {
    output=$1
    shift
    "$@" "$ds4_bin" -m "$model" --cuda --ctx 4096 --nothink --temp 0 \
        -n 16 --decode-consistency 16 -p "$prompt" >"$output" 2>&1
}

run_check "$fast_log" env
live_id=$(sed -n 's/.*live_top: {"id":\([0-9][0-9]*\).*/\1/p' "$fast_log" | tail -n 1)
fresh_id=$(sed -n 's/.*fresh_top: {"id":\([0-9][0-9]*\).*/\1/p' "$fast_log" | tail -n 1)
if [ -z "$live_id" ] || [ "$live_id" != "$fresh_id" ]; then
    cat "$fast_log" >&2
    echo "Kimi fast prefill changed the replay top token" >&2
    exit 1
fi

run_check "$exact_log" env DS4_KIMI_DISABLE_BATCH_PREFILL=1
if ! grep -Eq 'decode-consistency compared .*max_abs=0([ .]|$)' "$exact_log"; then
    cat "$exact_log" >&2
    echo "Kimi token-major replay is not logit-exact" >&2
    exit 1
fi

echo "Kimi Linear model replay: PASS (fast top token $live_id; exact fallback byte-identical)"
