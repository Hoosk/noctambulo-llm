# 📥 Importing Third-Party Models

> Part of the [Noctámbulo documentation](../README.md) · Models.

Noctámbulo can run models downloaded from elsewhere, provided they match the engine's
decoder layout (SwiGLU, RMSNorm, GQA, RoPE, tied word embeddings). Two paths exist:

- `import-hf` — converts a Hugging Face repo or a local directory with
  `config.json` + `tokenizer.json` + `model.safetensors` (f32/f16/bf16).
- `import` — reads a GGUF file. This engine's own exports round-trip exactly; compatible
  third-party GGUFs (llama/qwen2, f32/f16 tensors, byte-level BPE vocab) are converted
  best-effort.

```bash
# From the Hub (downloads config/tokenizer/safetensors, converts to f32)
cargo run --release -- import-hf --repo Qwen/Qwen2.5-0.5B-Instruct --output-dir qwen2.5-0.5b

# From a local HF directory
cargo run --release -- import-hf --dir downloads/my-model --output-dir my-model

# A third-party GGUF (f16). Pass the exact tokenizer.json from the HF repo when the
# GGUF pre-tokenizer differs (llama.cpp prints `tokenizer.ggml.pre = qwen2` for Qwen).
cargo run --release -- import model.gguf --output-dir my-model --tokenizer tokenizer.json
```

## What is supported

| Feature | `import-hf` | `import` (GGUF) |
| :--- | :--- | :--- |
| Architectures | Llama / Qwen2 / Mistral (CausalLM) | `llama`, `qwen2` metadata |
| Weights | f32 / f16 / bf16 safetensors (sharded ok) | f32 / f16 tensors only (quantized GGUF rejected) |
| Hidden size | any `intermediate_size` | any `feed_forward_length` |
| RMSNorm eps | `rms_norm_eps` from config | `attention.layer_norm_rms_epsilon` |
| RoPE base | `rope_theta` from config (1e6 for Qwen2.5, 5e5 for Llama-3) | `rope.freq_base` |
| RoPE scaling | `linear` / `dynamic` / `yarn` | `rope.scaling.type` |
| Tokenizer | copied `tokenizer.json` (byte-level BPE) | rebuilt from the vocab, or `--tokenizer <file>` |
| Embeddings | tied only (`tie_word_embeddings`) | tied only |
| QK-norm (Qwen3), SentencePiece, untied `lm_head`, GPTQ/AWQ | rejected with a clear error | rejected |

The converted model is written as `config.json` + `model_weights.bin` +
`custom_tokenizer.json`, tagged with `import_source` (`hf` / `gguf`) and architecture
version 5, and can be used by `generate`, `eval`, `export` and `train`.

## Fine-tuning an imported model

`train --resume <model_dir>` warm-starts from a native model directory: its `config.json`
defines the architecture (no CLI architecture flags needed), its `custom_tokenizer.json`
the vocabulary, and its weights become the starting point (optimizer/schedule start fresh
when the directory has no training sidecars).

```bash
cargo run --release -- train --resume model_output/qwen2.5-0.5b \
  --file datasets/corpus.txt --epochs 1 --lr 0.00002 --max-tokens 5000000
```

The web Train wizard exposes the same feature as “Warm start from a model”.

## Notes

- Weights are converted to **f32** (the engine trains in f32 and stores records in f32);
  a 0.5B model becomes ~2 GB on disk and in VRAM.
- Some “fp16” GGUFs published by model vendors are re-quantized artifacts (their attention
  weights differ from the original safetensors). Prefer `import-hf` or an f16 GGUF known
  to be a straight conversion.
- Quantized GGUFs (Q4_K_M, Q8_0, …) are not dequantized yet; convert from safetensors or
  use an f16 GGUF.
