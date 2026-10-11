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

### Expert-cache trace replay

Capture one uninterrupted ordinary MXFP4 decode span on Metal:

```sh
MODEL=gguf/MiMo-V2.6-Flash-MOPD-MXFP4-Q8Attn.gguf
DS4_METAL_STREAMING_EXPERT_CACHE_TRACE=/tmp/mimo-cache.jsonl \
  ./ds4 -m "$MODEL" --ssd-streaming --ctx 4096 --raw-prompt \
  --prompt-file speed-bench/mimo-cache/coding.txt \
  --temp 0 --nothink --tokens 512
python3 speed-bench/expert_cache_replay.py /tmp/mimo-cache.jsonl \
  --label coding --output /tmp/mimo-cache-replay.json
make test-expert-cache-replay
```

The fixed synthetic inputs in `mimo-cache/` cover coding, integer reasoning,
and a tool-result continuation. They are workload fixtures, not a model-quality
suite or evidence of a live tool execution. Run each in a fresh process; retain
the checkpoint, allocated context and effective cache budget with the results.

The version-1 JSONL trace starts with actual warm-cache residents, their ages,
and all nonzero route-hotness counts. Each route records selected IDs, the
observed hit mask, cache clock, logical pread bytes and blocking read/load wall
times. A footer marks normal completion. This is separate from the legacy
six-ID `DS4_MOE_RECORD_SELECTED_IDS` format and never overrides routing.
The trace path is selected at first use. Keep one model and a fixed cache budget;
MTP/DFlash, further prefill and concurrent execution are outside this replay's
scope. The producer currently captures the MXFP4 selected-slot path.

Replay accepts plain JSONL or gzip and refuses to report policy comparisons
unless every recorded hit mask, read-byte count and clock matches the current
policy. Every policy starts from the same resident set and slot budget:

* `current`: decaying route frequency, then oldest cache use.
* `least_stale_layer_fifo`: a demand-only interpretation inspired by
  [SpecMD](https://arxiv.org/abs/2602.03921). Stale residents precede current-pass
  residents, ordered by ascending layer and FIFO. Initial residents start stale.
  The paper does not fully specify warm-state, tie and fallback rules; this
  variant permits stale-right fallback and does not reproduce its prefetching.
* `oracle`: farthest-next-use over this finite trace, for equal-sized experts
  and mandatory admission with the complete current request protected.

Reports include aggregates and per-layer results; use `--format csv` for CSV.
Pread bytes are logical requests and may be served from the OS page cache.
`pread_ms` measures blocking worker-read batches; `load_ms` includes those
reads plus cache preparation/install, so do not add them together. Neither
covers all GPU synchronization or driver waits. Alternative-policy hit rates
and read-byte reductions are simulations, not measured throughput gains.

### Learned replacement and predictive staging

These CPU tools require NumPy. The frozen ranker and its leave-one-workload-out
report are in `mimo-cache/learned/`; fresh held-out traces and native
measurements are in `mimo-cache/experiments-20261011/`.

```sh
python3 speed-bench/expert_cache_learned.py evaluate \
  speed-bench/mimo-cache/experiments-20261011/heldout/*.jsonl.gz \
  --model speed-bench/mimo-cache/learned/frozen-linear-v1.json \
  --output /tmp/learned-heldout.json
python3 speed-bench/expert_prefetch_replay.py evaluate \
  speed-bench/mimo-cache/experiments-20261011/heldout/*.jsonl.gz \
  --freeze speed-bench/mimo-cache/predictive/freeze.json \
  --output /tmp/staging-heldout.json
make test-expert-cache-learned test-expert-prefetch-replay test-expert-activation-prefetch
```

Both predictors are frozen before held-out evaluation. Future accesses are
training labels only. Staging reserves slots inside the original cache budget,
charges every issued read, and assumes data is ready immediately. Wasted reads
and evictions count against it. This optimistic replay is not a speed benchmark.

For activation-based training, `expert_activation_prefetch.py capture-commands`
prints the pinned training and held-out capture commands. The recorder needs
both `DS4_MIMO2_ACTIVATION_TRACE=FILE` and the ordinary cache trace. Its
DS4ACT01 sidecar binds each 4096-element FP32 normalized activation to the
current token, layer and real top-eight IDs. It writes only completed ordinary
decode inputs; it does not substitute experts or introduce GPU synchronization.
Train before capturing held-out inputs, then use the script's `evaluate`
command with the frozen model. Source and model hashes are checked on load.

The recorded activation model and six raw captures are retained locally under
`speed-bench/local-runs/mimo-serving-20261011/activation/`, outside Git.
Compact hashes, settings and replay results are tracked. The model's
`settings.staging` embeds the route-history harness defaults for accounting;
its actual scores come from the activation ridge heads described by
`settings.head`, without coactivation priors.

The optional native ranker uses `DS4_MIMO2_CACHE_RANK_MODEL=FILE` at engine
startup. Its captures need an audit against `expert_cache_learned.CausalCache`;
the stock replay's current-policy audit must reject them when masks differ.
The retained native audit checks every hit mask, byte count and clock, plus
512 selected-token and top-20-logit steps against ordinary decoding.

### MiMo verifier quality probe

`make mimo-verify-quality` builds the strict-floating-point
`speed-bench/mimo_verify_quality` tool; `make test-mimo-verify-quality`
checks its numerical metrics and reference-file validation. Supply a raw
prompt and a file of decimal teacher token IDs, one per line. A 128-row run
requires 129 IDs: row `i` evaluates teacher token `i` and scores
teacher token `i+1`. The default pinned MOPD model is streamed on Metal.

```sh
env -u DS4_MIMO2_VERIFY_EXPERT_BUDGET ./speed-bench/mimo_verify_quality \
  --prompt-file speed-bench/mimo-cache/holdout-20261011/probability.txt \
  --tokens /tmp/teacher.ids --limit 128 --block 8 --ctx 4096 \
  --write-reference /tmp/full.ref --json /tmp/full.json
DS4_MIMO2_VERIFY_EXPERT_BUDGET=32 ./speed-bench/mimo_verify_quality \
  --prompt-file speed-bench/mimo-cache/holdout-20261011/probability.txt \
  --tokens /tmp/teacher.ids --limit 128 --block 8 --ctx 4096 \
  --reference /tmp/full.ref --json /tmp/budget32.json
```

First compare an unmasked run and budget256 against the saved reference.
Both must match exactly. References bind model revision/size, prompt and teacher
IDs, context, vocabulary and row dimensions; truncated or nonfinite data fail.
Smaller budgets report teacher-NLL drift, KL and argmax agreement across the
actual multirow verifier path. These are continuation-distortion measurements,
not task accuracy, DFlash acceptance or throughput. See `experiments-20261011/quality/gates.json`
under `mimo-cache/` for the numerical screen used in this study.

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

For streamed MiMo, pass `--ssd-streaming` and an explicit `--budget-gib`
for session allocations in addition to the expert cache. The benchmark prices
MiMo's private transients and does not mistake an unallocated expert cache for
free session memory. A matched native-batching comparison is:

```sh
./speed-bench/session_concurrency_bench \
  -m gguf/MiMo-V2.6-Flash-MOPD-MXFP4-Q8Attn.gguf --ssd-streaming \
  --concurrency 2,4 --ctx 256 --gen 64 --warmup 8 --budget-gib 8 \
  --candidate-env DS4_MIMO2_SESSION_BATCH --candidate-value require --repeat 2
```

`require` rejects serialized fallback. Do not combine native MiMo session
batching with DFlash/MTP. Report per-stream latency with aggregate throughput;
the recorded study did not establish a consistent gain.

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
