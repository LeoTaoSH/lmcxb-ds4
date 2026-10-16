# DS4 Imatrix Pipeline

This directory contains the calibration dataset and instructions used to build
activation importance matrices for DeepSeek V4 Flash and Pro GGUF
quantization.

For implementation ownership, validation history, artifact interpretation, and
remaining checks, see the [imatrix handoff](HANDOFF.md).

The current imatrix target is the routed MoE path.  Flash has 43 layers and
256 routed experts per layer.  Pro has 61 layers and 384 routed experts per
layer. V4.1 Flash has 40 layers and 384 routed experts per layer. These variants
expose three routed expert tensors per layer:

- `blk.N.ffn_gate_exps.weight`
- `blk.N.ffn_up_exps.weight`
- `blk.N.ffn_down_exps.weight`

For gate/up tensors, the collector records the squared FFN-normalized input
activation.  For down tensors, it records the squared routed SwiGLU row after
route weighting.  The result tells the quantizer which input columns are used
more heavily by the actual DS4 inference graph.

## 1. Build The Calibration Dataset

The tracked dataset is in `gguf-tools/imatrix/dataset/`.  Regenerate it from
the repository root with:

```sh
python3 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

The important output is:

```text
gguf-tools/imatrix/dataset/rendered_prompts.txt
```

It contains DS4-rendered chat prompts, separated by visible
`DS4_IMATRIX_PROMPT` markers.  The prompts include:

- C/Metal source-review prompts from this repository.
- Long-context snippets.
- Agent/tool-call prompts using DS4's DSML syntax.
- Language/prose rewriting, summarization, extraction, and translation prompts.
- `ds4-eval` GPQA Diamond, SuperGPQA, and AIME2025 benchmark prompts.
- Both thinking and non-thinking assistant prefixes.

The current tracked dataset has 4682 rendered prompts and roughly 2.91M tokens
by the coarse bytes/4 estimate.  Check
`gguf-tools/imatrix/dataset/manifest.json` for the exact generated-file
summary.

## 2. Collect The Imatrix

Use the DS4 runtime itself to collect routed MoE activation statistics.  The
collector uses the loaded GGUF metadata, so the same command shape works for
Flash and Pro.

Flash example:

```sh
./ds4 \
  -m ../deepseek-v4-quants/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out ../deepseek-v4-quants/imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat \
  --ctx 32768
```

Pro example with a smaller calibration budget:

```sh
./ds4 \
  -m ../deepseek-v4-quants/gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out ../deepseek-v4-quants/imatrix/DeepSeek-V4-Pro-Instruct-routed-moe-ds4-small.dat \
  --imatrix-max-prompts 16 \
  --imatrix-max-tokens 32768 \
  --ctx 32768
```

Useful smoke-test limits:

```sh
./ds4 \
  -m MODEL.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out /tmp/ds4-test.imatrix.dat \
  --imatrix-max-prompts 1 \
  --imatrix-max-tokens 4096
```

The collector is Metal-only because it hooks the layer-major Metal prefill graph.
It does not change inference math; it reads the already materialized MoE inputs
and accumulates `sum(x[column]^2)` per routed expert.

### Packed V4.1 calibration (isolated prompts)

Add `--imatrix-batch-tokens 8192` to pack consecutive whole prompts into one
layer-major sweep. The token budget is clamped to the graph's prefill capacity;
`0` (the default) keeps the independent-prompt reference path. Packing does not
split a prompt to fill unused space. Prompts larger than the budget run alone
using the existing chunked path, and corpus order is preserved.

The main Q/KV/output projections, HC, Engram lookup/projection, shared FFN and
routed MoE operate on the combined rows. Each prompt's attention uses the
existing kernels on a row view: RoPE positions restart at zero, compressed
KV/index caches have disjoint ranges, and selection/candidate rows stay with
their prompt. Engram n-gram history also restarts at every boundary. Odd prompt
lengths never share a compressor pair. No sequence-length-aware kernel or
cross-prompt attention mask is required. Graph scratch and SSD layer mappings
are reused in the same process; packing allocates views, not extra full graphs.
The attention-internal compressor/index projections retain the existing
per-prompt implementation. Decoder suffix replay stays disabled for imatrix.

For a small timing run:

```sh
./ds4 -m MODEL.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 54GB --ctx 32768 \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out /tmp/ds41-packed-probe.dat \
  --imatrix-batch-tokens 8192 --imatrix-max-prompts 6
```

The log reports collection elapsed time per pack, including attention, MoE,
SSD layer mapping and CPU imatrix accumulation, but excluding model startup,
checkpoint writes and final `.dat` serialization. Different matrix batch sizes
can change floating-point rounding and marginal expert routes; packed versus
independent imatrices are not promised to be byte-identical.

Low-count expert averages are especially sensitive to small route changes.
Compare coverage and sufficiently sampled expert vectors when validating a
new model/backend; attention isolation alone does not establish final
quantization quality.

### Checkpoint and resume (V4.1 / GLM collection)

For a long run, add `--imatrix-checkpoint FILE`. The collector writes a private
checkpoint with raw sums, per-expert counts, and the next prompt position after
every 16 completed prompts (change with `--imatrix-checkpoint-every N`). It
atomically replaces the previous checkpoint. For example:

```sh
./ds4 -m MODEL.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 54GB --ctx 32768 \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out calibration.dat \
  --imatrix-checkpoint calibration.ckpt \
  --imatrix-batch-tokens 8192
```

With packing, saves occur at the first completed pack crossing each checkpoint
interval. All prompts in a pack commit together, and the minimum-sample stopping
target is checked at pack boundaries. Existing single-prompt checkpoints can be
resumed with packing enabled; changing packing does not invalidate raw sums.

Send one `SIGINT` (Ctrl-C) or `SIGTERM` to stop gracefully: the current prompt/pack
finishes, a checkpoint is saved, and the process exits without writing a partial
final `.dat`. Run the same command with `--imatrix-resume` to continue. Resume
rejects a changed model file identity/size/mtime, corpus content, context, or
prefill geometry, as well as a truncated/corrupt checkpoint. A new run refuses
to overwrite an existing checkpoint unless `--imatrix-resume` is explicit. The
existing `.dat` output is not a checkpoint and cannot be resumed exactly.
`--imatrix-max-tokens` cannot be combined with checkpointing because it may
cut a prompt in half. The context limit still applies to each individual prompt.

The checkpoint is a private, native-layout format, not a portable interchange
file. Model identity includes the filesystem device and inode; moving the
model, remounting storage, or an OS upgrade can invalidate that identity even
when the weights are unchanged. Rejection diagnostics name the mismatched
field. Do not bypass those checks without independently verifying compatibility.

Library callers can zero-initialize `ds4_imatrix_options` and use
`ds4_engine_collect_imatrix_with_options` to enable checkpointing and packing.
The original `ds4_engine_collect_imatrix` entry point remains source-compatible
and keeps independent prompts without checkpointing.

### Regression tests

On a Metal build, model-free tests cover checkpoint round trips and rejection,
corpus preservation, packing limits, resume cursors, and CLI validation:

```sh
make test-imatrix
make tests/test_deepseek41_prefill
./tests/test_deepseek41_prefill --dispatch
```

The GPU attention isolation regression requires a V4.1 model and sufficient
memory for SSD streaming (the test reserves a 54 GiB expert cache). Fixed input
projections are compared exactly between independent and packed prompts in
both orders. The 112 comparisons cover singleton/odd/even lengths, raw-window
partitions, all four compressed KV owners, and index-selection/candidate reuse.
Engram hashes are also compared against independently reset histories:

```sh
make tests/test_deepseek41_packed
./tests/test_deepseek41_packed MODEL.gguf
```

### Output format and counters

The output format is llama.cpp's legacy binary `.dat` imatrix format.  DS4 packs
per-expert vectors into one entry per routed expert tensor:

```text
entry length = n_expert * n_columns
```

The quantizer slices the right expert's segment when quantizing each expert.
For Pro and V4.1 Flash, that means each routed tensor entry contains 384 vectors
instead of V4 Flash's 256. Values are per-expert mean squared activations;
entries use `ncall = 1` because they are already normalized. Experts with no
observations are written as zero vectors, not fabricated neutral weights.

The legacy footer's `chunks` field counts collection calls, one per layer per
prefill batch (or token on a single-token path). It is **not a token count**.
For example, 15040 calls over 40 layers mean 376 batches, which can contain
millions of tokens. The `.dat` does not store the total token count or individual
expert sample counts; use the completion log or matching raw checkpoint for
those. Coverage warnings remain meaningful regardless of how many tokens were
processed: some experts may be unobserved or have very few samples.

## 3. Generate GGUF Files With The Imatrix

The local C quantization tool in `gguf-tools/` supports:

```text
--imatrix FILE
--imatrix-strict
```

Example Q4 routed-expert regeneration:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template ../deepseek-v4-quants/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --out ../deepseek-v4-quants/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --imatrix ../deepseek-v4-quants/imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat
```

Example Q2 regeneration:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template ../deepseek-v4-quants/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --out ../deepseek-v4-quants/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --imatrix ../deepseek-v4-quants/imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat
```

Example Pro Q2 regeneration from a Pro template:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Pro \
  --template ../deepseek-v4-quants/gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct.gguf \
  --out ../deepseek-v4-quants/gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix-small.gguf \
  --imatrix ../deepseek-v4-quants/imatrix/DeepSeek-V4-Pro-Instruct-routed-moe-ds4-small.dat \
  --threads 24
```

`deepseek4-quantize` reads the routed expert count from the template GGUF.  Use
`--n-experts` only when working with an older template that lacks
`deepseek4.expert_count`.

For Q4, the imatrix does not change the runtime tensor type: routed experts
remain `Q4_K`.  It changes how quantization error is weighted while choosing
scales and codes.  For Q2, it replaces the previous synthetic weight-energy
fallback used for `IQ2_XXS` gate/up experts with real activation statistics.

## 4. Evaluate

Useful local tools:

```text
misc/quant_eval.c
gguf-tools/quality-testing/
```

`misc/quant_eval.c` compares local GGUF variants by greedy/top-logit behavior.
`gguf-tools/quality-testing/` can score local GGUFs against official DeepSeek
API continuations by target-token negative log likelihood.

The Q4 imatrix file uploaded to Hugging Face was tested on 100 official
DeepSeek V4 Flash continuations:

```text
old Q4 avg NLL:         0.177357819
Q4 imatrix avg NLL:     0.173895148
relative NLL change:   -1.95%
case wins:              54 imatrix / 46 old
first-token matches:    83 imatrix / 81 old
avg greedy LCP:         12.21 imatrix / 11.94 old
```

## Compatibility

The `.dat` file is intentionally in llama.cpp's legacy imatrix format, so the
data is not conceptually tied to DS4.  In practice, it is immediately useful
only with a quantizer that understands DS4's tensor names and packed per-expert
entries.  The current `deepseek4-quantize` tooling does that.

Other GGUF creation tools can use the same imatrix if they implement the same
name mapping and per-expert slicing convention.  Without that DS4-specific
mapping, a generic imatrix loader will see valid data but will not know how to
apply the packed routed-expert vectors correctly.
