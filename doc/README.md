# 📖 Kuro-Neko — Documentation

Welcome to the official documentation of **Kuro-Neko** (`kuro-neko-llm`), a training and inference engine for Large Language Models (LLM) with a **Qwen (Qwen2.5 / Qwen3)** architecture developed entirely in **Rust** and GPU-accelerated through **Burn 0.21** and **WGPU**.

This set of documents provides a detailed guide on the system architecture, the use of the command-line interface (CLI), text sampling strategies, tokenizer training and model export to the standard **GGUF v3** format.

---

## 📚 Documentation Index

| Document | Description |
| :--- | :--- |
| 🏋️ **[Training (train.md)](train.md)** | Complete manual of the `train` subcommand. Detailed explanation of the 15 configuration parameters, AdamW optimizer, cosine learning rate schedule with warmup, gradient clipping and practical examples. |
| 🔤 **[Generation and Inference (generate.md)](generate.md)** | Usage guide for the `generate` subcommand. Explanation of sampling strategies (Temperature, Top-K, Top-P, Min-P, Repetition Penalty) and how the KV Cache works (Prefill & Decode). |
| 🧩 **[Custom Tokenizer (train-tokenizer.md)](train-tokenizer.md)** | Training guide for the byte-level BPE tokenizer through the `train-tokenizer` subcommand, NFKC Unicode normalization and handling of special tokens. |
| 📦 **[GGUF v3 Export (export.md)](export.md)** | Manual for converting the model to the standard `GGUF v3` format through the `export` subcommand, 32-byte tensor alignment and integration with Ollama, LM Studio and llama.cpp. |
| 📊 **[Model Statistics (stats.md)](stats.md)** | Manual of the `stats` subcommand to calculate the total number of parameters, FP32 memory footprint, FLOPs per token estimate and GQA memory savings. |
| 🧠 **[Technical Architecture (architecture.md)](architecture.md)** | In-depth technical specification of the Qwen Transformer architecture: RoPE ($\theta=10.000$), SwiGLU aligned to 256, GQA, RMSNorm ($\epsilon=10^{-6}$), Tied Embeddings and Attention Softcapping. |

---

## ⚡ Quick Workflow (Quickstart)

The complete model development lifecycle in Kuro-Neko consists of the following 5 steps:

```bash
# 1. Train a custom 8,000-token BPE tokenizer
cargo run --release -- train-tokenizer --dataset roneneldan/TinyStories --vocab-size 8000

# 2. Train the causal Transformer model with GPU acceleration (WGPU)
cargo run --release -- train --dataset roneneldan/TinyStories --max-tokens 10000000 --batch-size 128 --epochs 3

# 3. Generate autoregressive text from a seed
cargo run --release -- generate "Once upon a time" --temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05

# 4. Inspect the model architecture and memory usage
cargo run --release -- stats

# 5. Export the trained model to GGUF v3 format for external inference
cargo run --release -- export --model-name kuro-neko-v1
```

---

## 🔗 Cross References

- To check the repository's main manual, see the [root README.md](../README.md).
- If you want to understand the source code structure, check the modules section in [architecture.md](architecture.md).
