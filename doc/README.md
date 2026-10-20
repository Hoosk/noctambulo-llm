# Noctámbulo — Documentation

Welcome to the official documentation of **Noctámbulo** (`noctambulo-llm`), a training and
inference engine for Large Language Models (LLM) with a **Qwen-style** architecture written
entirely in **Rust** and GPU-accelerated through **Burn 0.21** and **WGPU**.

This documentation covers the architecture, the full CLI workflow, sampling and KV cache
internals, tokenizer training, GGUF export/import and third-party model conversion.

---

## Guides

### Architecture

| Document | Description |
| :--- | :--- |
| **[Technical Architecture](architecture.md)** | Decoder specification: RoPE (configurable theta), SwiGLU, GQA, RMSNorm (configurable eps), tied embeddings, attention softcapping, KV cache layouts and eviction. |

### Training

| Document | Description |
| :--- | :--- |
| **[Training (`train`)](training/train.md)** | Optimizer, cosine LR + warmup, gradient accumulation, sequence warmup, early stopping, regularization, datasets, checkpoints and the H2O eviction mask. |
| **[Custom Tokenizer (`train-tokenizer`)](training/train-tokenizer.md)** | Byte-level BPE training, NFKC normalization, special tokens and the generated `custom_tokenizer.json`. |

### Inference

| Document | Description |
| :--- | :--- |
| **[Generation (`generate`)](inference/generate.md)** | Sampling strategies, prefill/decode, bounded KV caches (sliding window, sinks, H2O eviction, spill/replay), quantized cache and RoPE context extension. |

### Models

| Document | Description |
| :--- | :--- |
| **[GGUF v3 Export (`export`)](models/export.md)** | Tensor layout, 32-byte alignment, metadata keys and integration with llama.cpp, LM Studio and Ollama. |
| **[Third-Party Models (`import`)](models/import.md)** | Hugging Face safetensors conversion, compatible GGUF import, fine-tuning with `train --resume <model_dir>` and supported architectures. |
| **[Model Statistics (`stats`)](models/stats.md)** | Parameters, FP32 footprint, FLOPs per token, cache bytes/token per format and eviction budgets. |

---

## Quick Workflow

The complete model development lifecycle in Noctámbulo:

```bash
# 1. Train a custom 8,000-token BPE tokenizer
cargo run --release -- train-tokenizer --dataset roneneldan/TinyStories --vocab-size 8000

# 2. Train the causal Transformer with GPU acceleration (WGPU)
cargo run --release -- train --dataset roneneldan/TinyStories \
  --max-tokens 10000000 --batch-size 128 --epochs 3

# 3. Generate autoregressive text from a seed
cargo run --release -- generate "Once upon a time" \
  --temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05

# 4. Inspect the model architecture and memory usage
cargo run --release -- stats

# 5. Export the trained model to GGUF v3 for external inference
cargo run --release -- export --model-name noctambulo-v1
```

---

## See Also

- [Root README](../README.md) — install, quickstart, web console and benchmarks.
- [Roadmap](../ROADMAP.md) — current status and backlog.
- [CLI parameters](../cli_parameters.md) — every flag of every subcommand.
