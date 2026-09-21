# Kimi Linear 48B-A3B

DS4 experimentally supports the Kimi Linear 48B-A3B Instruct checkpoint on a
single CUDA GPU. This is a checkpoint-specific profile, not general support
for the Kimi model family.

| Item | Supported profile |
| --- | --- |
| Checkpoint | Kimi Linear 48B-A3B Instruct (27 layers) |
| Quantization | DS4-converted Q8_0 GGUF |
| Backend | One CUDA GPU |
| Validated hardware | NVIDIA GB10 (`sm_121`) |
| Validated context | 32,768 tokens |
| CLI, agent, server | Supported, including Kimi native tool calls |
| CPU, Metal, ROCm | Not supported for Kimi inference |
| Multi-GPU, SSD streaming | Not supported |
| Advertised 1M context | Not validated |

The loader validates the expected tensor layout and model geometry. This does
not imply support for Kimi K2, K3, VL, or arbitrary GGUFs using the
`kimi-linear` architecture name. The Base checkpoint can be converted but is
not part of the validated profile.

## Conversion

Use immutable Hugging Face and llama.cpp revisions. The scripts require commit
SHAs and record them in the GGUF metadata.

```sh
python gguf-tools/download_kimi_linear.py instruct \
  --revision <hugging-face-commit-sha> \
  --out models/Kimi-Linear-Instruct

python gguf-tools/kimi_linear_convert.py \
  --src models/Kimi-Linear-Instruct \
  --out models/Kimi-Linear-Instruct/Kimi-Linear-48B-A3B-Instruct-Q8_0.gguf \
  --source-repo moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --source-revision <hugging-face-commit-sha> \
  --llama-cpp /path/to/llama.cpp \
  --llama-revision <llama-cpp-commit-sha> \
  --outtype q8_0 --experts q8_0
```

Conversion writes to an `.incomplete` file first and never overwrites an
existing output.

## Build and run

On a DGX Spark / GB10:

```sh
make cuda-spark CUDA_ARCH=sm_121

./ds4 \
  -m models/Kimi-Linear-Instruct/Kimi-Linear-48B-A3B-Instruct-Q8_0.gguf \
  --cuda --ctx 4096 -n 512
```

For the coding agent or OpenAI-compatible server, use the same model and CUDA
options with `ds4-agent` or `ds4-server`. Kimi uses its native role template
and native JSON tool-call protocol automatically; it does not use DSML or a
reasoning block.

The seven MLA layers retain a full f16 KV cache. At 32,768 context tokens this
uses about 4.38 GiB for MLA K/V on the validated checkpoint, in addition to
model weights and runtime buffers.

## Validation

Run the model-free CUDA checks:

```sh
make test-kimi-linear-kernels CUDA_ARCH=sm_121
```

With the Q8_0 GGUF available, run the model-backed replay and logit checks:

```sh
make test-kimi-linear-model \
  DS4_KIMI_MODEL=/path/to/Kimi-Linear-48B-A3B-Instruct-Q8_0.gguf

./tests/test_kimi_linear_logits \
  /path/to/Kimi-Linear-48B-A3B-Instruct-Q8_0.gguf \
  16385 32767
```

The extended check requires identical top-1 tokens and enforces numerical
tolerances for all logits. To verify the serialized multi-session fallback:

```sh
DS4_TEST_CUDA_SINGLE_GPU=1 DS4_TEST_ALLOW_FALLBACK=1 \
  DS4_TEST_MODEL=/path/to/Kimi-Linear-48B-A3B-Instruct-Q8_0.gguf \
  ./tests/test_cuda_mixed_batch
```

The supported profile was validated on GB10 with CUDA kernel parity tests,
model logit checks through 32K context, mixed-session fallback, agent and
server unit tests, CUDA regression, and an end-to-end Compute Sanitizer
memcheck with zero reported errors. For sanitizer runs on drivers that reject
file-backed mmap registration, set `DS4_CUDA_SKIP_HOST_REGISTER=1`; this only
disables an optional host-registration optimization.

## Limitations

Kimi support remains experimental. It has not been validated on Apple Silicon,
ROCm, CPU inference, multi-GPU execution, SSD streaming, or the advertised
one-million-token context. Report the GGUF source revision, llama.cpp revision,
hardware, CUDA version, and exact command when reporting a problem.
