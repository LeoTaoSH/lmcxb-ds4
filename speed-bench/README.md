## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` to measure a rollback control
or `--candidate-value TEXT` for a numeric runtime setting. Use `--help` to
compare explicit split schedules.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them.

Numeric runtime tuning controls can use `--candidate-value TEXT` (default `1`), and
`--prefill-chunk N` selects the same chunk size for both variants (default
4096); zero uses the graph-selected size, which MiMo requires. The control
unsets the named variable.
Use only controls read at dispatch time: this harness keeps one engine alive
and cannot compare settings cached during initialization.

`DS4_QWEN4_TIMING=2` reports each prefill chunk's position, token count, and
stage times, including the attention mixer separately from GDN/attention.
These measurements include synchronization overhead and are for attribution;
use uninstrumented alternating runs for throughput claims.
`DS4_QWEN4_MOE_PROFILE=1` further splits tiled MoE into routing, expert-list
construction, routed/shared gate-up, routed/shared down, and reduction,
reporting one line per layer. Its synchronization also makes it diagnostic only.

Qwen routed MoE quantization specialization is enabled by default on M3 Ultra.
`DS4_QWEN4_MOE_MM_SPECIALIZE=0` restores the generic kernels; `=1` opts in on
other devices. To compare the final default against rollback, pass
`--candidate-env DS4_QWEN4_MOE_MM_SPECIALIZE --candidate-value 0` to the harness.

### Streamed MiMo tuning

Both Metal A/B harnesses accept `--ssd-streaming`, `--ssd-streaming-cold`,
and `--ssd-streaming-cache-experts N|NGB`. They retain the resident defaults
when these flags are omitted. For MiMo, keep the checkpoint and automatic
cache fixed while measuring existing dispatch-time controls:

```sh
MODEL=gguf/MiMo-V2.6-Flash-MOPD-MXFP4-Q8Attn.gguf
./speed-bench/metal_prefill_variant_bench -m "$MODEL" \
  --ssd-streaming --prefill-chunk 0 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --prefix-tokens 4096 --warmup-tokens 2048 --ctx 4353 --repeats 2 \
  --candidate-env DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE
./speed-bench/metal_decode_schedule_bench -m "$MODEL" \
  --ssd-streaming --prompt-file speed-bench/promessi_sposi.txt \
  --prefix-tokens 2048 --ctx 4096 --warmup 16 --tokens 256 \
  --candidate-env DS4_METAL_STREAMING_EXPERT_PREAD_THREADS \
  --candidate-value 4 --include-selection
```

These compare warm runs with a shared engine-level expert cache. The decode
harness alternates variant order and session assignment to distribute miss
costs; its absolute rate is not a standalone-request rate. Confirm winners
with separate-process `ds4-bench` runs. The examples are comparisons, not
recommendations to enable each candidate. Keep the exact-logit gate enabled.
MiMo uses runtime environment controls here; the decode harness's split-layer
schedule applies to DeepSeek. Prefill `--interleave` accepts automatic chunks
from the graph memory plan, keeping MiMo's internal chunk limit intact.

`DS4_METAL_ENCODER_TIMELINE=FILE` records encoder kernels and GPU timestamps.
`DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE=1` reports per-layer read
and wait costs. Set these before process startup, and use them only for
attribution: their instrumentation changes execution timing.

### Concurrency: many requests at once

`session_concurrency_bench` measures the engine without HTTP overhead.

```sh
make session-concurrency-bench
./speed-bench/session_concurrency_bench -m ds4flash.gguf \
  --concurrency 1,2,4,8 --ctx 4096 --gen 128 --budget-gib 28 \
  --csv /tmp/concurrency.csv
```

Streams use different corpus offsets to avoid identical expert routing.
The report includes aggregate and per-stream throughput, step latency and
prefill time. Set `--budget-gib` to bound session allocations; monitor actual
memory and swap too. `--mixed` adds a concurrent continued prefill, `--spec`
enables Qwen MTP, and `--verify` fails on a greedy-token mismatch against
serial decoding. Use the official-continuation scorer to assess quality when
different reduction orders change a close token choice.

For an interleaved comparison against a diagnostic rollback:

```sh
./speed-bench/session_concurrency_bench --ctx 4096 --concurrency 8 \
  --candidate-env DS4_QWEN4_SESSION_BATCH --candidate-value 0 --repeat 3
```

The control unsets the variable; the candidate sets it in alternating ABBA
blocks. Use only controls read at dispatch time. Keep prompts, context, memory
pressure and thermal conditions comparable, and retain generated text when
measuring speculation. `concurrency_sweep.sh` runs each cell in a fresh process.

`serve_concurrency_bench.py` measures the HTTP server, including queueing,
prompt rendering, prefix reuse and streaming:

```sh
python3 speed-bench/serve_concurrency_bench.py \
  --concurrency 8 --prompt-tokens 4096 --max-tokens 128 --requests 32 \
  --ignore-eos --json /tmp/serve-8.json
```

Start the server separately with an appropriate model, context and
`--batched-session 8`. The report includes first-token and inter-token
latency, request latency and throughput. Prompts use fresh nonces;
`--shared-prefix` tests cache reuse instead.
