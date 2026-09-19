---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "DwarfStar"
  text: "Run excellent LLMs<br>on your own hardware"
  tagline: A focused native inference engine for DeepSeek V4 Flash, GLM 5.x, and Qwen3.8 Flash Next — optimized for Metal, CUDA, and ROCm.
  image:
    src: /logo.svg
    alt: DwarfStar
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started
    - theme: alt
      text: Models & Vision
      link: /models
    - theme: alt
      text: View on GitHub
      link: https://github.com/antirez/ds4

features:
  - icon: 🖥️
    title: Apple Silicon
    details: Run on M3/M5 Macs with Metal. Native inference, SSD streaming, two-Mac tensor parallelism via RDMA.
  - icon: ⚡
    title: DGX Spark
    details: Blackwell-optimized CUDA build. V4.1 Flash SSD-streaming, multi-session batching, Qwen MTP.
  - icon: 🎯
    title: CUDA Multi-GPU
    details: Ada Lovelace through Blackwell. Tensor parallelism, 16-session serving, 126 t/s aggregate.
  - icon: 🚀
    title: Strix Halo / ROCm
    details: Framework Desktop and AMD systems. Flash, GLM 5.3, SSD streaming on Radeon 8060S.
  - icon: 📦
    title: DeepSeek V4 Flash
    details: Q2 through Q4, MXFP4, vision experimental — purpose-built quantization for routed experts.
  - icon: 🔮
    title: Speculative Decoding
    details: DSpark, GLM MTP, and Qwen MTP for faster generation without quality loss.

---

## Acknowledgements

This project would not exist without **llama.cpp and GGML** — the kernels, quantization formats, GGUF ecosystem, and hard-won engineering knowledge developed there. A big thank you to Georgi Gerganov and all the contributors.