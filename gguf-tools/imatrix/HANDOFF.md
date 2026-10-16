# Imatrix collection handoff

Updated 2026-10-16 for the next maintainer or upstream reviewer. The
`imatrix-packed-prefill` branch adds resumable calibration and batches V4.1
prompts through expensive token-wise operations while keeping their attention
independent. The full calibration run is complete; further work is review and
validation, not restarting that run.

## Branch and scope

- Fork: <https://github.com/liuliu/ds4>, branch `imatrix-packed-prefill`.
- Implementation commit: `fbeb7300ccead272fc32397b1de9d1d10747b2d2`.
- Upstream base: `0aaea5a238fb41a35106a551e73c8409dfb751ac`. This is the
  September base, not a claim of synchronization with current upstream main.
- Only imatrix implementation, tests, and documentation belong on this branch.
  Model weights, calibration outputs, local reports, and unrelated development
  history are excluded. The local `imatrix-cleanup` branch contains additional
  history and should not be substituted for this upstream-facing branch.

See [README.md](README.md) for the user-facing workflow. The implementation
commit message records the detailed design and validation history.

## Implementation map

In `ds4.c`, `imatrix_batch_read` parses the marker-separated corpus and packs
whole prompts in order without leaving the corpus modified. A prompt that
does not fit remains available to the next batch. Oversized prompts run alone;
the existing context limit still truncates each individual prompt.

`ds41_graph_prefill_sweep` shares projections, HC, Engram lookup/projection,
shared FFN, and routed MoE across the combined token rows.
`ds41_attention_packed` calls existing attention kernels on per-prompt views,
resets positions, and partitions compressed KV/index storage by published-key
counts. Odd boundaries cannot share a compressor pair. `ds41_hash_packed_tokens`
resets Engram history for every prompt. There are no sequence-length-aware
kernel changes. One model/process and graph scratch are reused across batches.

Imatrix collection disables decoder-suffix replay so all token activations are
collected at all layers. Normal inference retains its existing behavior.

`imatrix_checkpoint_prepare`, `imatrix_checkpoint_save`, and
`imatrix_checkpoint_load` manage raw sums, per-expert counts, corpus cursor,
and progress. Checkpoints are checksummed native-layout files, written through
a temporary file, flush/fsync, and rename. They are not portable interchange
files. Compatibility checks include model device/inode/size/mtime, corpus
content, shape, context, and prefill capacity. An OS upgrade or remount can
change the device identifier; do not silently bypass the check.

Checkpoint-enabled SIGINT/SIGTERM handling finishes the current prompt or pack,
saves, and exits 130/143 without publishing a partial DAT. Failed partial
forwards do not advance the checkpoint. SIGKILL cannot save in-flight work.
The default save interval is 16 prompts, rounded up to a completed pack.

`ds4.h` preserves the original collection API and adds
`ds4_engine_collect_imatrix_with_options` with a zero-initialized
`ds4_imatrix_options`. `ds4_cli.c` and `ds4_help.c` expose the options.
Packing defaults off; explicit `--imatrix-batch-tokens 0` works.
Packing is V4.1-only; checkpointing supports V4.1/GLM collection. The collector
requires Metal and V4.1 collection requires a single-host session.

## Completed calibration and file interpretation

The following local artifacts remain under `gguf-tools/imatrix/` in the
original workspace and are intentionally not tracked or uploaded:

- `DeepSeek-V4.1-Flash-Q4-routed-moe-ds4.dat`: final normalized imatrix.
- `DeepSeek-V4.1-Flash-Q4-routed-moe-packed-ds4.ckpt`: matching final raw state.
- `DeepSeek-V4.1-Flash-Q4-routed-moe-ds4.ckpt`: older pre-upgrade partial state;
  it is not the checkpoint matching the completed DAT.

The September 24 run completed 4,690 prompts, 2,897,897 tokens, and
695,495,280 routed observations. The final checkpoint cursor equals the corpus
size, 12,086,017 bytes. On October 16 the stored header/counters and DAT footer
were checked again; this was not a new GPU run or full checksum validation.

The DAT contains 120 routed tensor entries: gate/up/down for all 40 layers.
It does not contain attention, shared-expert, or all other model weight keys.
Each entry stores expert-major input-column vectors, not a full weight-shaped
tensor. Gate/up use FFN-normalized input squared; down uses the squared routed
SwiGLU row after route weighting. Values are per-expert means, with `ncall=1`
because normalization has already occurred.

The footer value 15,040 counts layer collection calls. Dividing by 40 gives
376 prefill batches, **not 376 tokens**. The DAT itself does not store total
token count or per-expert observation counts; use the matching checkpoint or
completion log.

Coverage is 15,335/15,360 expert-layer pairs (99.84%). There are 25 unobserved
pairs and 55 pairs with fewer than eight observations, including those 25.
Unobserved vectors are zeros. Replacing them with ones is an explicit
quantizer fallback, not measured calibration; preserve the original artifact
and track any transformed copy separately. More diverse calibration may help
rare experts, but the footer is not evidence of a smoke-only run.

## Validation history and limits

On September 25, the cleaned upstream-based tree passed a fresh Metal build
of all executables, `make test-imatrix`, the prefill dispatch regression,
`DS4_NO_GPU` syntax checks, and `git diff --check`.

Earlier GPU testing on September 24 used the pre-cleanup packed implementation
on macOS 27.0, M5 Max/128 GiB, V4.1 Flash Q4, SSD streaming, a 54 GiB expert
cache, and context 32768:

- All 112 independent/packed/permuted attention comparisons were bit-exact;
  independently reset Engram hashes matched exactly.
- Four prompts totaling 3,854 tokens took 134.753 s independently versus
  33.733 s packed, a 3.995x collection-only speedup. Startup, checkpoint writes,
  and DAT serialization are excluded. This is not a full-corpus ETA.
- Mean cosine across 120 imatrix entries was 0.9990124. The minimum, 0.9498554,
  was dominated by an expert receiving two samples independently versus one
  packed. Restricting to experts with at least eight samples in both runs gave
  minimum cosine 0.9997653. Batch-size rounding can change marginal routes;
  packed imatrices are not promised to be byte-identical or to yield identical
  quantization quality.

The original Q4 model is no longer present at
`gguf/DeepSeek-V4.1-Flash-Q4.gguf`, including at this handoff. A fresh GPU run
of the cleaned implementation remains outstanding. Temporary historical logs
may have expired; do not depend on their paths for reproducibility.

## Reproduction and next steps

From the branch root on a Metal host:

```sh
make -j8 all
make test-imatrix tests/test_deepseek41_prefill tests/test_deepseek41_packed
./tests/test_deepseek41_prefill --dispatch
./tests/test_deepseek41_packed /path/to/V4.1_MODEL.gguf
```

The GPU isolation test reserves a 54 GiB expert cache; use an adequately sized
host. Model-free tests cover checkpoint replacement/rejection/corruption,
corpus preservation, packing limits, resume cursors, and CLI bounds.

For a new four-prompt smoke run, use fresh output paths, not production files:

```sh
./ds4 -m /path/to/V4.1_MODEL.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 54GB --ctx 32768 \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out /path/to/new-smoke.dat \
  --imatrix-checkpoint /path/to/new-smoke.ckpt \
  --imatrix-max-prompts 4 --imatrix-batch-tokens 8192
```

Use `--imatrix-batch-tokens 0` and separate fresh files for the independent
reference. Compare all vector values and per-expert counts, retaining low-count
outliers. Then test graceful interruption/resume across multiple packs. Resume
with the same command plus `--imatrix-resume`; prompt limits are cumulative,
so increase/remove a completed smoke limit to collect further prompts.
`--imatrix-max-tokens` is incompatible with checkpointing.

Before upstreaming, review changes since the September base, rerun the tests
after any integration, and validate final quantization quality separately from
activation similarity. Do not regenerate the corpus before resuming an existing
checkpoint: corpus content is part of its identity. Keep the completed DAT and
both historical checkpoints unchanged.
