# 🐈‍⬛ Kuro-Neko (kuro-neko-llm)

**Kuro-Neko** is a training and inference engine for Large Language Models (LLMs) with a **Qwen (Qwen2.5 / Qwen3)** architecture written entirely in **Rust**, GPU-accelerated through **Burn 0.21** and **WGPU**, with first-class export to **GGUF v3** (compatible with LM Studio, llama.cpp and Ollama) and a React web console to drive everything from the browser.

---

## 🚀 Main Features

- **Causal Transformer architecture (Qwen2.5 / Qwen3)**:
  - Causal multi-head attention with **RoPE** ($\theta = 10{,}000$), bias enabled on the QKV projections (`with_bias(true)`) and a `triu_mask` triangular mask.
  - MLP layers with **SwiGLU** (SiLU activations aligned to multiples of 256).
  - **RMSNorm** normalization with precision $\epsilon = 10^{-6}$.
  - **Tied word embeddings** with normal Gaussian initialization $\mathcal{N}(0, 0.02)$ to avoid logit saturation.
  - **Dropout** and **weight decay** ($0.1$) regularization.
- **GPU acceleration with WGPU / Burn 0.21**: cross-platform execution (Vulkan, Metal, DX12, WebGPU) with optional half-precision compute (`--dtype f16` on the non-fused backend).
- **Streaming parallel tokenization**: multithreaded line-by-line streaming loading with a byte-level BPE tokenizer, dynamic capacity estimation and error reporting.
- **Optimized training pipeline**:
  - **Data shuffling** per epoch (Fisher-Yates) for better convergence.
  - **Gradient clipping** by $L_2$ norm (`Norm(1.0)`) and **weight decay** (`0.1`).
  - **Cosine LR schedule** with configurable warmup.
  - **Sparse loss computation ($NLL$)** with `log_softmax` + `gather`.
  - **GPU loss accumulation** with no per-batch GPU $\to$ CPU synchronization stalls.
  - **Gradient accumulation**, **progressive sequence scaling** warmup and **early stopping**.
  - **Decoupled weight decay** (AdamW) applied to all parameters, optional **label smoothing** and **z-loss**.
- **Context extension (RoPE scaling)**: `linear`, `ntk` and `yarn` strategies with a configurable factor; RoPE angles are computed per position (no table cap), generation and evaluation can run past the trained `n_ctx`, and GGUF exports carry the `rope.scaling.*` metadata.
- **Bounded KV cache**: `--sliding-window` plus `--sink-tokens` turn the per-layer cache into a ring buffer of `window + sinks` rows (long prompts are prefilled in chunks); without a window the cache is preallocated to the context and written in place.
- **Cross-layer KV sharing**: `--kv-share-group N` makes each group of N layers share one K/V projection and cache (DeepSeek CSA-style), shrinking KV memory by N. Checkpoints carry an `arch_version`; GGUF export of grouped models requires `--approx` and is documented as an approximation.
- **Direct Parquet/HF ingestion**: `prepare-dataset` tokenizes a Parquet text column straight into `.bin` (no intermediate `.txt`) and `--dedupe` drops exact duplicate lines/documents.
- **Full-state checkpoints and resume**: weights, AdamW moments, LR schedule and shuffler RNG; safe stop with Ctrl+C saves an interrupted checkpoint.
- **Evaluation with BPB**: cross-entropy loss, perplexity ($PPL = e^{\text{loss}}$) and bits-per-byte (comparable across tokenizers), over all batches by default.
- **Native GGUF v3 exporter**: direct serialization of all weight and bias tensors aligned to 32 bytes with GGUF v3 headers and the `general.alignment` key.
- **Streaming generation over SSE**: the API streams tokens as they are produced, supports cancellation and reports live tokens/second. The CLI exposes the same sampler (Top-K, Top-P, Min-P, repetition penalty).
- **React web console**: React 19 + Vite + TypeScript + Tailwind v4 (built locally, no CDN) with Train, Datasets, Generate, Models, Export and Devices views, HttpOnly-cookie authentication and a self-contained production build served by Axum.
- **Model management from the browser**: rename, delete and pick models, download their weights/config/tokenizer, and export to GGUF. Long operations (Hugging Face downloads, weighted mixes, GGUF exports) run as cancellable background jobs with live byte/char/tensor progress.
- **Run history and telemetry**: every training run is persisted with its config, final loss/PPL and train/val curves so runs can be overlaid and compared, while `/api/telemetry` reports live process CPU/RSS, system memory and GPU memory counters where the OS exposes them.
- **Modern CLI with Clap**: derive-macro CLI with strict validation and automatic `--help`.

---

## 🛠️ Requirements

- **Rust**: 1.80+ on the **Stable** channel (`cargo` and `rustc`).
- **GPU drivers**: Vulkan / Metal / DX12 support. For NVIDIA machines, build with `cargo build --release --features cuda` to use the CUDA backend in the CLI.
- **Node.js 20+**: only needed to build the web console (`web/`).

---

## ⚙️ Command-Line Interface (CLI) Usage

### 1. Train the Model (`train`)

Runs GPU-accelerated training on a Hugging Face dataset or a local file:

```bash
# Train on TinyStories (default configuration: 100M tokens, batch 128, 3 epochs)
cargo run --release -- train --dataset roneneldan/TinyStories
```

#### Trainer Options (CLI Flags)

| Flag | Description | Default |
|:---|:---|:---|
| `--dataset <repo>` | Hugging Face repository ID(s) (e.g. `roneneldan/TinyStories`) | `[]` |
| `--file <path>` | Local plain-text or binary (`.bin`) file to train on | `None` |
| `--dataset-file <filename>` | File name inside the HF repo | `TinyStories-train.txt` |
| `--preset <name>` | Architecture preset: `nano` (~2M), `tiny` (~5.5M), `small` (~30M), `base` (~100M) | `None` |
| `--dtype <f32\|f16>` | Compute precision. `f16` speeds up on capable GPUs (non-fused backend); requires stable LR | `f32` |
| `--batch-size <n>` | Batch size | `128` |
| `--epochs <n>` | Number of training epochs | `3` |
| `--max-tokens <n>` | Maximum tokens to load (`0` = no limit). The Chinchilla reference is ~20 tokens/parameter | `100000000` |
| `--step <n>` | Sliding window step (`None` = `n-ctx`) | `None` |
| `--n-ctx <n>` | Model context window | `256` (or preset) |
| `--n-embd <n>` | Embedding dimension | `256` (or preset) |
| `--n-head <n>` | Causal attention heads (Query) | `4` (or preset) |
| `--n-kv-head <n>` | Key/Value heads for GQA (`None` = same as `n-head`) | `None` |
| `--n-layer <n>` | Number of Transformer blocks | `4` (or preset) |
| `--lr <float>` | Learning rate | `0.0003` |
| `--dropout <float>` | Dropout probability | `0.1` |
| `--attn-softcap <float>` | Attention capping (`0.0` disabled, `50.0` Gemma-style) | `0.0` |
| `--sliding-window <n>` | Sliding attention window (`0` = full causal); bounds the inference KV cache to `n + sink-tokens` | `0` |
| `--sink-tokens <n>` | Initial tokens always kept attendable (attention sinks) | `0` |
| `--kv-share-group <n>` | Layers per cross-layer K/V sharing group (`1` = off); one K/V projection and cache per group | `1` |
| `--val-split <float>` | Fraction reserved for validation and perplexity ($PPL$) | `0.1` |
| `--grad-accum <n>` | Micro-batches accumulated per AdamW step (true gradient mean, equivalent to a larger batch) | `1` |
| `--save-dir <path>` | Directory for automatic per-epoch checkpoint saving | `None` |
| `--resume <path>` | Checkpoint to resume from (restores weights, AdamW, schedule and RNG; `--epochs` = additional epochs) | `None` |
| `--seq-warmup-steps <n>` | Initial steps for **Progressive Sequence Scaling** (from $N/4$ to $N$) | `0` |
| `--early-stop-patience <n>` | Stops training after `n` epochs without improving `val_loss` (`0` = disabled) | `0` |
| `--weight-decay <float>` | Decoupled weight decay (AdamW) applied to all parameters (`0` disables it) | `0.1` |
| `--label-smoothing <float>` | Label-smoothing factor for the cross-entropy loss | `0.0` |
| `--z-loss <float>` | Z-loss coefficient on `logsumexp(logits)²` | `0.0` |

#### Custom Training Examples

```bash
# Quick test run with 1M tokens and GQA (n_kv_head=2)
cargo run --release -- train --dataset roneneldan/TinyStories --max-tokens 1000000 --n-kv-head 2

# Train on the FULL dataset with no token limit
cargo run --release -- train --dataset roneneldan/TinyStories --max-tokens 0

# Configure training with Sequence Length Warmup (Progressive Scaling), accumulation and checkpointing
cargo run --release -- train --file corpus.txt --seq-warmup-steps 1000 --grad-accum 4 --val-split 0.1 --save-dir checkpoints

# Configure a larger model with GQA and softcapping
cargo run --release -- train --dataset roneneldan/TinyStories --n-embd 512 --n-head 8 --n-kv-head 2 --n-layer 8 --attn-softcap 50.0
```

---

### 2. Pre-tokenize a Dataset to Binary (`prepare-dataset`)

Pre-tokenizes a corpus or Hugging Face dataset into an ultra-compact `.bin` file for instant `mmap` loading:

```bash
cargo run --release -- prepare-dataset --file corpus.txt --output datasets/corpus.bin
```

> [!NOTE]
> `.bin` v2 datasets embed a fingerprint of the tokenizer vocabulary. If you retrain the tokenizer, re-run `prepare-dataset`: loading a `.bin` tokenized with another vocabulary now fails with an explicit error instead of training on corrupt IDs.

Parquet files are ingested directly (the `text`/`content`/`body` column is tokenized on the fly, no intermediate `.txt`), and `--dedupe` removes exact duplicate lines/documents before tokenizing:

```bash
cargo run --release -- prepare-dataset \
  --file datasets/cosmopedia.parquet \
  --output datasets/cosmopedia.bin \
  --dedupe
```

Weighted mixing is also supported, so several sources can be blended by proportion while writing a single `.bin`:

```bash
cargo run --release -- prepare-dataset \
  --file stories.txt,code.txt \
  --weights 0.8,0.2 \
  --output datasets/mixed.bin
```

---

### 2b. Resume Training and Safe Stop

Training progress is persisted in `train_state.json` (schedule position, shuffler RNG, best `val_loss`) and `optimizer_state.bin` (AdamW moments), next to the weights:

```bash
# Resume 2 additional epochs from the epoch-3 checkpoint
cargo run --release -- train --dataset roneneldan/TinyStories --resume checkpoints/checkpoint_epoch_3.bin --epochs 2

# Stopping with Ctrl+C or from the web saves checkpoint_interrupted.bin + full state
```

With `--resume`, `--epochs` means **additional epochs** and the cosine schedule continues where it left off. If the checkpoint is old (no sidecars), only the weights are loaded, with a warning.

---

### 3. Evaluate Model, Perplexity and BPB (`eval`)

Evaluates the cross-entropy loss, perplexity ($PPL = e^{\text{loss}}$) and **bits-per-byte** (BPB, comparable across tokenizers) of a trained model. By default it evaluates **all** dataset batches (`--eval-batches 0`) and tokenizes the full dataset (`--max-tokens 0`); `--max-tokens <n>` tokenizes at most `n` tokens first (it pairs with `--eval-batches`) to evaluate a slice of a huge corpus. BPB is not available with `.bin` datasets because they do not keep the original size in bytes:

```bash
cargo run --release -- eval --model-dir model_output --file datasets/corpus.txt
```

---

### 4. Train a Custom Tokenizer (`train-tokenizer`)

Trains a byte-level BPE tokenizer from scratch on a specific corpus:

```bash
cargo run --release -- train-tokenizer --dataset roneneldan/TinyStories --vocab-size 8000
```

#### Tokenizer Options (CLI Flags)

| Flag | Description | Default |
|:---|:---|:---|
| `--dataset <repo>` | Hugging Face repository ID(s) (comma-separated, e.g. `roneneldan/TinyStories`) | `[]` |
| `--file <path>` | Local plain-text file(s) to train the vocabulary on | `[]` |
| `--dataset-file <filename>` | Target file name inside the HF dataset repository | `TinyStories-train.txt` |
| `--vocab-size <n>` | Total size of the resulting vocabulary table | `8000` |
| `--weights <w1,w2,...>` | Mix weights per source (same order as `--file`) | `None` |
| `--max-chars <n>` | Maximum characters of the mixed corpus to train on (`0` = all) | `0` |

---

### 5. Generate Text / Inference (`generate`)

Generates autoregressive text from a seed phrase using the trained model, with Top-K, Top-P, Min-P and repetition penalty sampling:

```bash
cargo run --release -- generate "Once upon a time" --temp 0.9 --top-k 60 --top-p 0.9 --min-p 0.1 --max-tokens 50
```

#### Generator Options (CLI Flags)

| Flag / Argument | Description | Default |
|:---|:---|:---|
| `<prompt>` | Seed phrase (prompt) to start generation (positional) | *Required* |
| `--temp <float>` | Sampling temperature controlling creativity/randomness | `0.9` |
| `--top-k <n>` | Upper bound of candidate tokens to consider (Top-K sampling) | `60` |
| `--top-p <float>` | Cumulative probability threshold for nucleus sampling (Top-P) | `0.9` |
| `--min-p <float>` | Min-P relative probability threshold | `0.1` |
| `--rep-penalty <float>` | Repetition penalty factor for repeated tokens | `1.6` |
| `--max-tokens <n>` | Maximum number of new tokens to generate | `50` |
| `--seed <n>` | Pseudo-random generator seed for reproducibility | `None` |
| `--dtype <kind>` | Compute precision: `f32` or `f16` (KV cache follows it) | `f32` |
| `--rope-scaling <kind>` | Context extension: `none`, `linear`, `ntk` or `yarn` | `none` |
| `--rope-factor <float>` | Extension factor (e.g. `2.0` doubles the usable context) | `1.0` |

The web API additionally exposes `POST /api/generate/stream` (SSE), which streams tokens as they are generated and supports cancellation. The web Generate form also exposes the compute precision (`f32`/`f16`).

---

### 6. Inspect Model Statistics (`stats`)

Prints the total parameter count, FP32 memory footprint, FLOPs per token, GQA compression and KV cache estimates (f32/f16, in bytes/token and total MB at `n_ctx`):

```bash
cargo run --release -- stats
```

---

### 7. Export to GGUF Format (`export`)

Exports the model weights stored in `model_output` directly to a **GGUF v3** binary:

```bash
cargo run --release -- export --model-name kuro-neko-2026
```

The resulting `.gguf` file is saved to `exported/kuro-neko-2026.gguf` and can be opened directly in **LM Studio**, **llama.cpp** or **Ollama**.

#### Export Options (CLI Flags)

| Flag | Description | Default |
|:---|:---|:---|
| `--model-name <name>` | Base name of the resulting `.gguf` file | `kuro-neko-qwen` |
| `--approx` | Export grouped-KV models as a standard-attention approximation (duplicated K/V weights) | off |

---

### 8. Download a Dataset (`download`)

Downloads a single dataset file from Hugging Face Hub; Parquet files are automatically text-extracted:

```bash
cargo run --release -- download --repo HuggingFaceTB/cosmopedia-100k --file data/train-00000-of-00001.parquet --out-dir datasets
```

#### Download Options (CLI Flags)

| Flag | Description | Default |
|:---|:---|:---|
| `--repo <repo>` | HF dataset repo ID | *Required* |
| `--file <file>` | File name inside the repo | *Required* |
| `--out-dir <path>` | Output directory for the downloaded file | `datasets` |
| `--extract-text` | Extract the text column to `.txt` when the file is Parquet | enabled |

---

### 9. Web Console (`serve`)

Serves the React console (built into `web/dist`) plus the JSON/SSE API to control training, generation and export from the browser:

```bash
# Build the console once (or after changing web/)
cd web && npm install && npm run build

# Start the server
cargo run --release -- serve --bind 127.0.0.1:8080 --token my-secret-token
```

Open `http://127.0.0.1:8080`, paste the token and the UI logs in. Authentication uses an **HttpOnly cookie** set by `POST /api/login`; the `X-API-Token` header is still accepted for scripts and `curl`. If `--token` is omitted, the server runs without authentication (trusted local networks only).

#### Server Options (CLI Flags)

| Flag | Description | Default |
|:---|:---|:---|
| `--bind <addr:port>` | Address and port to bind (`0.0.0.0:8080` for remote access) | `127.0.0.1:8080` |
| `--token <token>` | API token required in the `X-API-Token` header | `None` (no auth) |
| `--datasets-dir <path>` | Directory containing datasets to list/train on | `datasets` |
| `--models-dir <path>` | Directory containing model outputs and checkpoints | `model_output` |

For UI development, run the Vite dev server instead; it proxies `/api` to `http://127.0.0.1:8080`:

```bash
cd web && npm run dev
```

---

## 🏗️ Project Structure

```
kuro-neko/
├── Cargo.toml          # Dependencies (Burn 0.21, tokenizers, hf-hub, serde, clap)
├── src/
│   ├── main.rs         # Entry point and Clap derive CLI
│   ├── config.rs       # Unified ModelConfig struct
│   ├── model.rs        # LLaMA/Qwen architecture (Llm, Block, Attention, SwiGLU, weight tying)
│   ├── data.rs         # LlmTokenizer and multithreaded streaming loading
│   ├── train.rs        # GPU loop with shuffling, grad clipping, weight decay, AdamW and resume
│   ├── generate.rs     # Autoregressive inference with bounds check and XorshiftRng
│   ├── export.rs       # GGUF v3 binary serializer (32-byte aligned weights)
│   ├── backend.rs      # WGPU backend/dtype selection (fused vs non-fused)
│   ├── tui.rs          # Terminal visual utilities (ANSI, Unicode boxes and progress)
│   └── serve/          # Axum server: api/ (JSON + SSE), runner, state
├── web/                # React 19 + Vite + TypeScript + Tailwind v4 console
│   └── src/            # Shell, typed API client, state and views (Train, Datasets, Generate, Models, Export, Devices)
├── doc/                # Detailed documentation (architecture, train, generate, export, stats)
└── cli_parameters.md   # Full CLI parameter dictionary
```

## ✅ Tests and Quality

```bash
cargo test                    # 66 tests (data, model, train/resume, serve API, GGUF export)
cargo clippy --all-targets
cargo check --features cuda   # optional CUDA backend (runtime requires an NVIDIA GPU)

cd web
npm run lint                  # ESLint
npm run doctor                # react-doctor quality gate
npm run build                 # TypeScript check + production build in web/dist
npm run smoke                 # CDP browser suite (see web/smoke/README.md)
```
