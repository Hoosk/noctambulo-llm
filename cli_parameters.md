# 🐈‍⬛ Kuro-Neko — Command-Line Interface (CLI) Parameters Dictionary

This file contains the dictionary of all parameters accepted by the different subcommands of the **Kuro-Neko** CLI.

---

## 🏋️ 1. `train` Subcommand Parameters
Runs the neural network training pipeline on the GPU using WGPU.

| Parameter | Type | Default | Technical Description / Function |
| :--- | :--- | :--- | :--- |
| `--dataset` | `Vec<String>` | `[]` | Dataset ID(s) on Hugging Face Hub (comma-separated, e.g. `roneneldan/TinyStories`). |
| `--file` | `Option<String>`| `None` | Path to a local plain-text or binary (`.bin`) file to train on. |
| `--dataset-file` | `String` | `"TinyStories-train.txt"` | Target file name inside the HF dataset repository. |
| `--preset` | `Option<String>` | `None` | Architecture preset: `nano` (~2M), `tiny` (~5.5M), `small` (~30M), `base` (~100M). Explicit flags take precedence. |
| `--dtype` | `String` | `"f32"` | Compute precision. Only `f32` is trainable: Burn 0.21's AdamW moment estimates underflow in f16, so `f16` is rejected with a clear error. `bf16` is not supported by the WGPU/WGSL backend. Records are always stored in f32. |
| `--batch-size` | `usize` | `128` | Batch size to process on the GPU before the weight update. |
| `--epochs` | `usize` | `3` | Number of complete passes through the dataset. With `--resume`, **additional** epochs. |
| `--max-tokens` | `usize` | `100000000` | Maximum number of tokens to load from the corpus. A value of `0` trains on the full file. The Chinchilla reference is ~20 tokens/parameter. |
| `--step` | `Option<usize>`| `None` | Sliding window step. If it is smaller than `n_ctx` it generates overlap. `None` is equivalent to `n_ctx`. |
| `--n-ctx` | `Option<usize>` | `None` → preset/`256` | Maximum length of the model's context window in tokens. |
| `--n-embd` | `Option<usize>` | `None` → preset/`256` | Embedding dimension and hidden states. Must be divisible by `n-head`. |
| `--n-head` | `Option<usize>` | `None` → preset/`4` | Number of Query attention heads ($Q$). |
| `--n-kv-head` | `Option<usize>`| `None` | Number of Key/Value heads ($K, V$). Enables GQA if it is smaller than `n-head`. `None` or `0` enables standard MHA. If there is a preset, it uses its value. |
| `--n-layer` | `Option<usize>` | `None` → preset/`4` | Number of stacked Transformer blocks. |
| `--lr` | `f64` | `0.0003` | Maximum learning rate (peak learning rate) for AdamW. |
| `--dropout` | `f64` | `0.1` | Neuron dropout probability for attention and MLP regularization. |
| `--attn-softcap` | `f32` | `0.0` | Threshold for soft-capping attention scores via $\tanh$. `0.0` disables it. |
| `--sliding-window` | `usize` | `0` | Sliding attention window in tokens (`0` = full causal attention). The KV cache at inference is bounded to `window + sink_tokens` rows. |
| `--sink-tokens` | `usize` | `0` | Initial tokens always kept attendable (attention sinks, StreamingLLM). |
| `--kv-share-group` | `usize` | `1` | Layers per cross-layer K/V sharing group (`1` = no sharing). One K/V projection and cache serve each group, shrinking KV memory by the group size at the cost of a different computation graph. |
| `--mla-latent-dim` | `usize` | `0` | Multi-head Latent Attention: compress K/V into a latent of this size and cache only the latent (`0` = disabled). Memory per token is `latent_dim` floats instead of `2·n_kv_head·head_dim`. GGUF export folds the weights exactly (`W_down · W_up`). |
| `--val-split` | `f32` | `0.1` | Percentage of the dataset reserved for validation and perplexity computation ($\text{PPL} = e^{\text{loss}}$). |
| `--grad-accum` | `usize` | `1` | Gradient accumulation steps before running the AdamW step. |
| `--save-dir` | `Option<String>`| `None` | Directory to save automatic model checkpoints at the end of each epoch. |
| `--resume` | `Option<String>`| `None` | Path to a checkpoint file or previous weights to resume training from. |
| `--seq-warmup-steps` | `usize` | `0` | Initial steps for **Progressive Sequence Scaling** (grows the context window from $N/4$ to $N$). `0` disables it. |
| `--early-stop-patience` | `usize` | `0` | Stops training after $N$ epochs without `val_loss` improvement. `0` disables it. |
| `--weight-decay` | `f64` | `0.1` | Decoupled weight decay (AdamW) applied to all parameters. `0` disables it. |
| `--label-smoothing` | `f32` | `0.0` | Label-smoothing factor for the cross-entropy loss. `0.0` disables it. |
| `--z-loss` | `f32` | `0.0` | Z-loss coefficient applied to `logsumexp(logits)²`. `0.0` disables it. |

---

## ⚡ 2. `prepare-dataset` Subcommand Parameters
Pre-tokenizes a plain-text corpus or HF dataset to an ultra-compact binary `.bin` format with a `KNKB` metadata header.

| Parameter | Type | Default | Technical Description / Function |
| :--- | :--- | :--- | :--- |
| `--dataset` | `Vec<String>` | `[]` | HF dataset ID(s) to download and pre-tokenize. |
| `--file` | `Vec<String>` | `[]` | Path to the local text file(s) to convert. |
| `--dataset-file`| `String` | `"TinyStories-train.txt"` | File name inside the HF repo. |
| `--output` | `String` | `"datasets/dataset.bin"` | Path of the resulting output binary file (`.bin`). |
| `--max-tokens` | `usize` | `0` | Maximum number of tokens to save (`0` = no limit). |
| `--weights` | `Option<String>` | `None` | Mix weights per source (comma-separated, same order as `--dataset`/`--file`). When set, sources are mixed probabilistically by these proportions. |
| `--max-chars` | `usize` | `0` | Maximum characters to write for weighted mixing (`0` = all available). |
| `--dedupe` | `bool` (flag) | `false` | Drop exact duplicate lines/documents (hashed) before tokenizing. Works for text and Parquet inputs (`.bin` inputs are loaded as-is). |

> Parquet files are ingested directly: the `text`/`content`/`body` column is tokenized on the fly into the `.bin`, without an intermediate `.txt`. Mixing or combining Parquet sources is rejected.

---

## 📊 3. `eval` Subcommand Parameters
Evaluates the cross-entropy loss, perplexity ($PPL$) and bits per byte (BPB) of a saved model over a dataset.

| Parameter | Type | Default | Technical Description / Function |
| :--- | :--- | :--- | :--- |
| `--model-dir` | `String` | `"model_output"` | Directory containing `config.json` and `model_weights.bin`. |
| `--dataset` | `Vec<String>` | `[]` | HF dataset ID(s) to evaluate. |
| `--file` | `Option<String>`| `None` | Path to the evaluation text or `.bin` file. |
| `--dataset-file`| `String` | `"TinyStories-train.txt"` | Target file name in HF. |
| `--n-ctx` | `Option<usize>`| `None` | Overrides the context window for evaluation. |
| `--eval-batch-size` | `usize` | `16` | Evaluation batch (lower it if there is OOM on small GPUs). |
| `--eval-batches` | `usize` | `0` | Maximum number of batches to evaluate (`0` = all). There used to be a fixed limit of 100. BPB is not available for `.bin` datasets (they do not know their size in bytes). |
| `--max-tokens` | `usize` | `0` | Tokenizes at most this many dataset tokens before evaluating (`0` = all). Useful to evaluate a slice of a huge corpus instead of tokenizing it fully. Pairs with `--eval-batches`. |
| `--rope-scaling` | `String` | `"none"` | RoPE context-extension strategy: `none`, `linear`, `ntk` or `yarn`. |
| `--rope-factor` | `f32` | `1.0` | Context-extension factor (e.g. `2.0` doubles the usable context). |

---

## 🔤 4. `train-tokenizer` Subcommand Parameters
Trains a byte-level BPE tokenizer based on a text corpus.

| Parameter | Type | Default | Technical Description / Function |
| :--- | :--- | :--- | :--- |
| `--dataset` | `Vec<String>` | `[]` | HF dataset ID(s) to download and use as the base corpus. |
| `--file` | `Vec<String>` | `[]` | Path to a local text file(s) for training the vocabulary. |
| `--dataset-file`| `String` | `"TinyStories-train.txt"` | Name of the file to extract from the HF dataset. |
| `--vocab-size` | `usize` | `8000` | Total size of the resulting tokenizer vocabulary table. |
| `--weights` | `Option<String>` | `None` | Mix weights per source (comma-separated, same order as `--file`). When set, sources are mixed probabilistically by these proportions. |
| `--max-chars` | `usize` | `0` | Maximum characters of mixed corpus to train the tokenizer on (`0` = all available). |

---

## 🔮 5. `generate` Subcommand Parameters
Performs autoregressive text inference using the model loaded from `--model-dir` (`model_output/` by default). The tokenizer is resolved from the model directory first (`custom_tokenizer.json`), so imported/self-contained models use their own vocabulary.

| Parameter | Type | Default | Technical Description / Function |
| :--- | :--- | :--- | :--- |
| `<prompt>` | `String` | *Required* | Initial seed text from which the model will continue the generation. |
| `--model-dir` | `String` | `"model_output"` | Directory containing `config.json` + `model_weights.bin` (+ optional `custom_tokenizer.json`). |
| `--temp` | `f32` | `0.9` | Sampling temperature. Controls the randomness of the prediction. |
| `--top-k` | `usize` | `60` | Top-K sampling. Limits the selection to the $K$ most probable tokens. |
| `--top-p` | `f32` | `0.9` | Top-P sampling (Nucleus Sampling). Limits the selection to tokens whose cumulative probability reaches $P$. |
| `--min-p` | `f32` | `0.1` | Min-P sampling. Excludes tokens with probability lower than $\text{min\_p} \times P_{\max}$. |
| `--rep-penalty` | `f32` | `1.6` | Repetition penalty. Discourages repeated tokens by dividing their logits. |
| `--max-tokens` | `usize` | `50` | Maximum number of new tokens to generate. |
| `--seed` | `Option<u64>` | `None` | Random seed of the Xorshift generator for reproducibility. |
| `--dtype` | `String` | `"f32"` | Compute precision: `f32` (fused WGPU) or `f16` (non-fused half backend; half-precision logits can flip tokens on weakly trained models). |
| `--kv-cache-dtype` | `String` | `"f32"` | KV cache precision: `f32`, `f16` (halves memory; automatically runs on the non-fused backend because the fused backend's f16 casts are silent no-ops), `int8` (4 values per i32 word + per-block scales) or `q4` (8 nibbles per i32 word + per-block scales). |
| `--rope-scaling` | `String` | `"none"` | RoPE context-extension strategy: `none`, `linear`, `ntk` or `yarn`. |
| `--rope-factor` | `f32` | `1.0` | Context-extension factor (e.g. `2.0` doubles the usable context). |

---

## 📦 6. `export` Subcommand Parameters
Exports weights to GGUF v3 binary format for external inference.

| Parameter | Type | Default | Technical Description / Function |
| :--- | :--- | :--- | :--- |
| `--model-name` | `String` | `"kuro-neko-qwen"` | Base name of the `.gguf` file that will be saved in `exported/`. |
| `--approx` | `bool` (flag) | `false` | Allow exporting a cross-layer KV sharing model (`kv_share_group > 1`) as a standard-attention approximation: the shared K/V weights are duplicated per layer, so the exported model is not identical to the engine's grouped computation. |

---

## 📥 7. `import` Subcommand Parameters
Imports a GGUF file produced by `export` (or a compatible llama-style GGUF with f32/f16 tensors) back into the native format: `config.json`, `model_weights.bin` and a `custom_tokenizer.json` rebuilt from the GGUF vocabulary. Useful to recover models that only survive as GGUF files.

| Parameter | Type | Default | Technical Description / Function |
| :--- | :--- | :--- | :--- |
| `<gguf>` | `String` | *Required* | Path to the `.gguf` file to import. |
| `--output-dir` | `String` | GGUF file stem | Directory (under the models root) that will receive the imported model. |
| `--force` | `bool` (flag) | `false` | Overwrite the output directory if it already contains a model. |

The inverse of the exporter is exact (transposes + GPT-NeoX RoPE row permutation are inverted); quantized GGUFs (Q4/Q8...) are rejected with a clear error.

---

## ⬇️ 8. `download` Subcommand Parameters
Downloads a dataset file from Hugging Face Hub; Parquet files are automatically text-extracted.

| Parameter | Type | Default | Technical Description / Function |
| :--- | :--- | :--- | :--- |
| `--repo` | `String` | *Required* | HF dataset repo ID (e.g. `HuggingFaceTB/cosmopedia-100k`). |
| `--file` | `String` | *Required* | File name inside the repo (e.g. `data/train-00000-of-00001.parquet`). |
| `--out-dir` | `String` | `"datasets"` | Output directory for the downloaded file. |
| `--extract-text` | `bool` (flag) | `true` | Extract the text column to `.txt` when the file is Parquet. |

---

## 📈 9. `stats` Subcommand Parameters
Prints the model parameter count, VRAM footprint, FLOPs statistics and KV cache estimates for every storage format (`f32`, `f16`, `int8`, `q4`, in bytes/token and total MB at `n_ctx`). It takes no parameters.

---

## 🌐 10. `serve` Subcommand Parameters
Serves the web console (React build in `web/dist`) and the JSON/SSE API to control training, generation, and export from the browser.

| Parameter | Type | Default | Technical Description / Function |
| :--- | :--- | :--- | :--- |
| `--bind` | `String` | `"127.0.0.1:8080"` | Address and port to bind (e.g. `0.0.0.0:8080` for remote access). |
| `--token` | `Option<String>` | `None` | API token required in the `X-API-Token` header (also accepted as a login cookie). If omitted, the server runs without authentication: only use it on a trusted local network. |
| `--datasets-dir` | `String` | `"datasets"` | Directory containing datasets to list/train on. |
| `--models-dir` | `String` | `"model_output"` | Directory containing model outputs and checkpoints. |

