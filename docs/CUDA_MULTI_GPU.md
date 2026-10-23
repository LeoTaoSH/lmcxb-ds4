# NVIDIA CUDA GPUs

[README](../README.md) | [Getting started](../README.md#start-here)

**Ada Lovelace is supported, including 48 GB L40S cards.** This is an important
use case: a model whose other inference implementations require newer GPUs
can still run here when DwarfStar supports its GGUF layout. The Flash kernels
do not require Blackwell's native FP4 instructions; Ada uses an appropriate
CUDA path. Support remains model-specific, not a promise to run arbitrary GGUFs.

## Build

Install the NVIDIA driver and CUDA toolkit, including `nvcc` and cuBLAS.
Use the local GPU architecture, or select Ada explicitly:

```sh
make cuda-generic
# For an Ada build:
make cuda CUDA_ARCH=sm_89
```

For a GB10 machine use the [DGX Spark](DGX_SPARK.md) target instead.

### Turing and DeepSeek V4.1

Turing cards such as the Quadro RTX 8000 use `CUDA_ARCH=sm_75`. Check that
your installed toolkit lists `sm_75` in `nvcc --list-gpu-code`; toolkit
version alone does not establish architecture support.

V4.1's CUDA activation rounding emulates BF16/FP8/FP4 in float32 and does not
require native BF16 or FP8 hardware. Its synthetic kernel suite can run
without model weights:

```sh
make test-deepseek41-cuda CUDA_ARCH=sm_75
# Quick activation-rounding check on a selected physical card:
CUDA_VISIBLE_DEVICES=2 ./tests/test_deepseek41_cuda --quantization
```

These checks validate kernels, not whole-model inference. The experimental
resident local CUDA layer-placement path also runs V4.1 Flash 1017 Q2 on
four 48 GiB Quadro RTX 8000s:

```sh
make ds4 test-deepseek41-multigpu CUDA_ARCH=sm_75
./ds4 --cuda \
  --model ~/models/dwarfstar-deepseek-v4.1-flash-1017-q2/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  --gpu-devices 0,1,2,3 --gpu-vram 44,44,44,44 --ctx 4096
```

Each GPU holds complete layers' weights. A serial shared workspace hands
activations and KV/index state between owners. When every peer-access probe
passes, this workspace uses managed memory. If any peer fails validation,
the workspace instead uses portable mapped pinned host memory: implicit
managed migration can otherwise traverse the failed link and silently corrupt
state even though explicit peer copies use host bounce buffers. Model weights
remain resident on their assigned GPUs in either mode. All selected GPUs must
support concurrent managed access. Engram tables remain disk-only; their size
is excluded from GPU placement. The planner reserves the entire shared state plus CUDA
scratch on **every** GPU and refuses CPU-spill layouts.

This correctness-first path uses token-major prefill and serial session
execution, not fused multi-session batches; live KV reuse and checkpoint
read/restore use the same authoritative shared state. It is not an optimized
parallel-throughput path. Local placement cannot be combined with SSD
streaming, network tensor parallelism, or DSpark/MTP speculation. Optional
network tensor parallelism still has exactly two single-GPU ranks. `--cuda-tensor-parallel` and the
paired tensor/expert path below remain V4 Flash-only.

`--decode-consistency N` checks incremental decode against a fresh prefill.
It rejects nonfinite or constant logits before selecting tokens, so two
all-zero outputs cannot masquerade as a successful consistency check.

For a single 48 GiB card, explicitly select only that card and enable SSD
expert streaming:

```sh
CUDA_VISIBLE_DEVICES=0 ./ds4 --cuda \
  --model /path/to/DeepSeek-V4.1-Flash-1017-Q2.gguf --ssd-streaming \
  --ssd-streaming-cache-experts 256 --ctx 4096
```

Admission uses the selected device's memory, not the sum of all visible
cards. Static weights, graph buffers, two GiB runtime reserve and prefill
headroom must fit before any expert cache is admitted; cache targets may be
reduced. Streaming performance on discrete Turing GPUs is not guaranteed.

The tested 1017 artifact is the community DwarfStar-format Q2 file from
`nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF`, revision
`796f58533a0e4b7a89cb006b22aeb44e783f0323`. Its source checkpoint is
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The 365,713,686,528-byte file's
SHA256 is
`f57e6c3e5e68c513249e1c472a2600bf59792ca78cacb7ac0531a6056376856e`.
This is not an arbitrary Q2_K GGUF. Use this exact artifact when reproducing
the results below rather than assuming another converter or download target
produces byte-identical weights.

October 19, 2026, four RTX 8000s on Linux with CUDA 13.4 and explicit
44 GiB budgets per GPU:

| Check | Result |
| --- | --- |
| Full next-token logits versus the same-weight single-GPU SSD reference | All 129,280 logits matched exactly |
| Incremental decode versus fresh prefill | Four decode steps after an 857-token prompt; maximum absolute difference and RMS both zero |
| Sustained generation with 8K allocated context | 128-token reply, about 5.6 generation t/s |
| Two simultaneous HTTP requests, two slots, 4K allocated context | Independent correct answers using ordered execution |

The longer consistency check used 8K allocated context, not an 8K-token
prompt, and took about 22 minutes including loading and both prefills.
The serial shared-state path is correctness-first; prefill can be much slower
than the short generation example suggests. These are bounded functional
checks, not a throughput, long-context quality, or production-load guarantee.

## Normal layer placement

Without tensor parallelism, DwarfStar places complete layers on the selected
GPUs. `--gpu-vram auto` uses reported free VRAM, reserving space for the graph
and context. Explicit budgets are comma-separated GiB values, one per device.

```sh
./download_model.sh ds4f-q2
./ds4 --cuda --gpu-devices 0,1,2,3 --gpu-vram auto --ctx 32768
```

This is also the multi-GPU placement mode for GLM. Startup refuses a layout
that would require unsupported CPU execution; reduce context or choose a
smaller model if necessary.

## Flash tensor parallelism

`--cuda-tensor-parallel` pairs GPUs and divides routed-expert work inside
each pair. Pairs also own successive layer ranges. This is an in-process
configuration, not the network `--role coordinator` / `--role worker` mode.

The device order matters. List all layer-home GPUs first, then their partners.
For physical pairs `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`, use:

```text
0,2,4,6,1,3,5,7
```

Check your machine's topology with `nvidia-smi topo -m`; do not assume this
ordering gives the best pairs on another server. Each pair stores one half
of the routed experts per GPU. Dense attention, routers, and shared experts
are replicated within the pair; the output head is vocabulary-sharded.

For the tested eight-L40S setup:

```sh
./download_model.sh ds4f-q4
./ds4-agent --cuda --cuda-tensor-parallel \
  --gpu-devices 0,2,4,6,1,3,5,7 --gpu-vram auto --ctx 100000
```

Q4 has native grouped routed kernels for multi-user throughput. Q2 needs less
memory, but unsupported grouped shapes use a slower, correct fallback.
Four 48 GB cards with Q2 and eight with Q4 are tested configurations. An even
device count alone does not guarantee memory fit.

## Serve multiple users

```sh
./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-devices 0,2,4,6,1,3,5,7 --gpu-vram auto \
  --ctx 100000 --batched-session 16 --host 0.0.0.0
```

The supported eight-L40S Flash configuration has reached roughly 126 aggregate
generation tokens/s with 16 decode rows. This is total throughput, not the
speed of one user. The recorded conditions and regression requirements are in
[the QA guide](../QA_BEFORE_RELEASES.md).

CUDA TP defaults to 2048-token prefill chunks. Keep that default for the
16-session, 100k-context setup; larger chunks need more scratch memory.
Reduce session count or context if all KV states do not fit.

The repository's [server launcher](../run-nvidia-tp-server.sh) is an example
from the L40S deployment, not a portable default: it uses host-specific model
and cache paths and defaults to MXFP4. The commands above need no launcher or
environment tuning. See [serving](SERVER.md) for disk caches and API access.
