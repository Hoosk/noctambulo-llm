# 📊 Analysis and Comparison of LLM / SLM Architectures (2026)

This document contains the comparative technical analysis of the main open-source language models of the **2026** generation (Gemma 4, Qwen 3.5/3.8, DeepSeek V4 Pro, Kimi K2.7, GLM-5.2 and LLaMA 3.2/4).

---

## 📊 Comparative Table of LLM/SLM Architectures (2026)

| Model | Creator / Date | Architecture | Parameters (Total / Active) | `tie_word_embeddings` | Attention Mechanism | Normalization / Activation | Context Window | License |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **Gemma 4** | Google DeepMind *(April 2026)* | Hybrid (Dense & MoE) | E2B – 31B (MoE: 26B/3.8B) | **`true`** *(E2B/E4B)* **`false`** *(31B)* | **GQA** (Grouped Query) | RMSNorm + GeGLU | **256k** | Apache 2.0 |
| **Qwen 3.5 / 3.8** | Alibaba Cloud *(July 2026)* | Dense & MoE | 0.5B – 397B (Max: 2.4T MoE) | **`true`** *(0.5B-3B)* **`false`** *(7B+)* | **GQA + QKV Bias** | RMSNorm + SwiGLU | **1M** | Apache 2.0 / Token Plan |
| **DeepSeek V4 Pro** | DeepSeek *(April 2026)* | Sparse MoE | 1.6T / 49B active | **`false`** | **CSA** (Compressed Sparse Attention) | RMSNorm + SwiGLU | **1M** | MIT |
| **Kimi K2.7 Code** | Moonshot AI *(June 2026)* | Sparse MoE | 1T / 32B active | **`false`** | **MLA** (Multi-head Latent Attention) | RMSNorm + SwiGLU | **256k** | Modified MIT |
| **GLM-5.2** | Zhipu AI *(June 2026)* | Sparse MoE + IndexShare | 744B / ~36B active | **`false`** | **Sparse MHA + IndexShare** | RMSNorm + SwiGLU | **1M** | MIT |
| **LLaMA 3.2 / 4** | Meta AI *(2025/2026)* | Dense & MoE | 1B – 405B | **`false`** | **GQA** (Grouped Query) | RMSNorm + SwiGLU | **128k - 1M** | LLaMA 3 Community |
| **Noctámbulo Engine** | Rust + WGPU *(2026)* | SLM Causal Decoder | Configurable (5M – 3B) | **`true`** *(Tied N(0,0.02))* | **GQA + QKV Bias + Min-P + Softcap** | RMSNorm + SwiGLU | **256 - 128k** | MIT / Open Source |

---

## 🔑 Architecture Trend Analysis (2026)

### 1. `tie_word_embeddings` (Tied vs Untied)
- **Small Models / SLMs (< 3B)**: **They use `tie_word_embeddings: true`** (Gemma 4 E2B, Qwen 2.5/3 0.5B). Sharing the weights between input and output saves up to 30% of VRAM on mobile and desktop client GPUs.
- **Giant Models / MoE (> 7B)**: **They use `tie_word_embeddings: false`** (DeepSeek V4, GLM-5.2, Kimi K2.7). At large scale, the relative VRAM savings from sharing embeddings are negligible compared to the expressiveness of having separate matrices.

### 2. Evolved Attention (Beyond standard MHA)
- **GQA (Grouped Query Attention)**: Universal standard in Gemma 4, Qwen 3 and LLaMA 3.2.
- **MLA (Multi-head Latent Attention)**: Kimi K2.7 and DeepSeek compress the keys ($K$) and values ($V$) into a latent space to reduce KV-Cache consumption at 1M context.
- **QKV Bias**: Qwen 3 keeps the biases in $Q, K, V$ to avoid gradient instability in fp16/bf16 floating point on reduced-size models.

### 3. Universal Normalization and Activations
- **RMSNorm** and **SwiGLU/GeGLU** have become the absolute standard without exception in **all** 2026 flagship models.
