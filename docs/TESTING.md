# Testing and Development

[README](../README.md)

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before proposing an inference change.
The [release QA guide](../QA_BEFORE_RELEASES.md) defines the hardware matrix,
checkpoint-matched quality tests, long-context agent tasks, and speed checks.
It also records which checks were not completed. A smoke test is not full QA.

## Focused checks

Model-free checks include:

```sh
make ds4_test ds4_agent_test test-session-state
./ds4_test --server
./ds4_agent_test
```

On Metal, small GPU tensor tests are available without loading a full GGUF:

```sh
make tests/test_session_state_gpu tests/test_glm53_kda tests/test_mxfp4_metal
./tests/test_session_state_gpu
./tests/test_glm53_kda
./tests/test_mxfp4_metal
```

For the experimental Kimi Linear CUDA path, run the combined model-free gate:

```sh
make test-kimi-linear-kernels
```

It checks KDA monolithic/chunked state parity, MLA kernels, and the Q8 MoE
path. Model-backed llama.cpp logit comparison and session/reset/rewind checks
remain required; see [the Kimi Linear acceptance plan](KIMI_LINEAR.md).

With the Q8_0 model available, also run:

```sh
make test-kimi-linear-model DS4_KIMI_MODEL=/path/to/kimi-linear-q8.gguf
```

That target first compares every output logit after tiled and retained
per-query MLA prefill, then compares split-KV and reference MLA decode at
irregular 1,025, 1,537, 4,097 and 8,191-token frontiers. For the extended
long-context gate through 32K, run:

```sh
make tests/test_kimi_linear_logits
./tests/test_kimi_linear_logits /path/to/kimi-linear-q8.gguf \
  1025 1537 4097 8191 16385 32767
```

The comparison requires identical top-1 tokens, total-variation distance at
most `1e-3`, and maximum raw-logit error at most `2.5`. It reports the top-5
set, RMS and mean error as additional diagnostics.

Kimi currently uses the correct serialized path for concurrent sessions, not
the native mixed-prefill/decode batch. Check that fallback against independent
control sessions with:

```sh
DS4_TEST_CUDA_SINGLE_GPU=1 DS4_TEST_ALLOW_FALLBACK=1 \
  make test-cuda-mixed-batch DS4_TEST_MODEL=/path/to/kimi-linear-q8.gguf
```

`make test` also includes model-backed tests. Select the right GGUF and ensure
that it fits before running it; do not accidentally load a large model on a
single device during multi-GPU QA. ROCm has `make test-rocm`.

Official-vector tests must use continuations from the same checkpoint as the
GGUF. Flash 0731 and Vision Experimental are not interchangeable fixtures.
See [test vectors](../tests/test-vectors/README.md) and
[quality scoring](../gguf-tools/quality-testing/README.md).

## Investigating output

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits /tmp/logits.json --nothink --prompt-file prompt.txt
./ds4-server --trace /tmp/ds4-trace.txt
```

Token dumps catch template differences without inference. Logits and
continuations help distinguish sampling changes from graph errors. Server
traces include cache and tool-parser decisions. Keep traces private when they
contain real conversations.

## Tools and source references

- [GGUF conversion and quantization](../gguf-tools/README.md)
- [Imatrix collection](../gguf-tools/imatrix/README.md) and
  [calibration corpus](../gguf-tools/imatrix/dataset/README.md)
- [Steering vectors](../dir-steering/README.md)
- [Benchmark scripts and charts](../speed-bench/README.md)
- [Evaluation data and licenses](../EVAL_DATA.md)
- [Session payload implementation](../ds4.c) and
  [cache header definitions](../ds4_kvstore.h)
- [Pipeline protocol](../ds4_distributed.c) and [TP protocol](../ds4_tp.c)

For changes to state handling, cover rewind/replay, save/load, images, and
multiple sessions as well as a fresh prompt. For distributed changes, exercise
both ranks and failures; a local command-parser test is not physical TP QA.
