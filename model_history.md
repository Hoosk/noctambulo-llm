# 📖 History and Development Tutorial: Noctámbulo (noctambulo-llm)

This document objectively records the technical evolution of **Noctámbulo**, from a basic toy model to a state-of-the-art (2026 standard) training and inference engine written entirely in **Rust**, GPU-accelerated with **Burn 0.22** and **WGPU**, and exportable to the native binary format **GGUF v3**.

---

## 📑 Table of Contents

1. [Chapter 1: The Seed and the Starting Point](#chapter-1-the-seed-and-the-starting-point)
2. [Chapter 2: Parallel Streaming Tokenization (Zero-Copy)](#chapter-2-parallel-streaming-tokenization-zero-copy)
3. [Chapter 3: Diagnosis and Elimination of GPU Bottlenecks](#chapter-3-diagnosis-and-elimination-of-gpu-bottlenecks)
4. [Chapter 4: Data Scaling, Chinchilla and CLI Parameterization](#chapter-4-data-scaling-chinchilla-and-cli-parameterization)
5. [Chapter 5: The Leap to the 2026 Architecture (LLaMA / Qwen Standard)](#chapter-5-the-leap-to-the-2026-architecture-llama--qwen-standard)
6. [Chapter 6: Native GGUF v3 Binary Exporter](#chapter-6-native-gguf-v3-binary-exporter)
7. [Quick Guide and Usage Tutorial](#quick-guide-and-usage-tutorial)
8. [🐣 For Dummies Appendix: Concepts Explained Without Technical Jargon](#for-dummies-appendix-concepts-explained-without-technical-jargon)

---

## Chapter 1: The Seed and the Starting Point

### The Goal
Build a Causal Autoregressive Language Model (LLM) from scratch using **Rust** to guarantee memory safety without a garbage collector (GC), and the **Burn** framework to take advantage of cross-platform GPU acceleration (Vulkan/Metal/DX12) through **WGPU**.

### The Initial Architecture (GPT-2 Baseline)
In the early stages, the project implemented the classic GPT-2 architecture (2019):
- **Token Embeddings (`wte`)** + **Positional Embeddings (`wpe`)**: Fixed matrices of learned embeddings.
- **LayerNorm Pre-LN**: Classic normalization with computation of mean, variance and bias parameters.
- **Multi-Head Causal Attention (MHA)**: Scaled dot-product attention with a triangular mask to prevent seeing into the future.
- **Feed-Forward MLP with GELU**: Projection $N \to 4N \to N$ with `GELU` activations.

---

## Chapter 2: Parallel Streaming Tokenization (Zero-Copy)

### The Problem
When loading large datasets (such as `TinyStories` of ~1.8 GB of plain text), the first implementation tried to tokenize the entire file by loading it completely into RAM in one go, reaching consumption peaks of up to **20 GB of RAM** and speeds of only ~2 MB/s.

### The Solution
1. **Multithreaded streaming with `encode_batch`**: `encode_file_streaming` was implemented in `src/data.rs` using `BufReader` with a 2 MB buffer and sending batches of 8,192 lines to all CPU cores.
2. **Zero-Copy Optimization in Rust**:
   - Replacement of String duplication with zero-cost ownership transfer using `std::mem::take(&mut line_batch)`.
   - In-place whitespace trimming (`trim()`).
   - Result: Speed increased to **11-40 MB/s** and RAM consumption stabilized at only **~2.4 GB**.

---

## Chapter 3: Diagnosis and Elimination of GPU Bottlenecks

During the development of the first training runs, the GPU would stall or move at a painfully slow speed (~1 batch/s). A deep audit of the code was carried out, identifying three serious performance flaws:

### 1. The GPU $\to$ CPU Stall per Batch
- **Failure**: On every iteration of the training loop, `.into_data().to_vec::<f32>()` was called to read the Loss scalar.
- **Impact**: It forced the GPU pipeline to stop completely on every batch in order to do a synchronous PCIe readback to the CPU.
- **Solution**: Accumulation of Loss tensors directly in GPU memory using `.detach()` (to avoid retaining autograd graphs) and synchronization to the CPU only at the scheduled log intervals.

### 2. The Memory Bomb: 262 MB One-Hot per Step
- **Failure**: To compute the loss with `cross_entropy_with_logits`, the targets were converted into a `one_hot` matrix of shape `[batch_size * n_ctx, vocab_size]`.
- **Impact**: With `batch_size = 128` and `n_ctx = 64`, a matrix of $8,192 \times 8,000 = 65.5 \text{ million floats}$ (**262 Megabytes**) of dynamic allocation in VRAM was created **on every training step**.
- **Solution**: Replacement with sparse cross-entropy computation:
  $$\text{loss} = -\text{mean}(\text{gather}(\text{log\_softmax}(\text{logits}), \text{targets}))$$
  Memory consumption dropped from **262 MB to 32 KB** per batch (**8,000 times less allocation**).

### 3. Direct Causal Mask
- **Failure**: In every layer and every step, position tensors `arange` were generated and a boolean inversion `mask.equal_elem(false)` was executed.
- **Solution**: Direct upper triangular mask `arange.unsqueeze(0).greater(arange.unsqueeze(1))` with no intermediate allocations.

---

## Chapter 4: Data Scaling, Chinchilla and CLI Parameterization

### Understanding the Training Scale (Chinchilla Laws)
For a small model of **~7M to 30M parameters**, trying to train it on 447 million tokens multiplied by 3 epochs (1.34 billion tokens) caused artificially long training times (days) with no improvement in the model's capability.

### Changes Applied
- **Non-overlapping window stride (`step = n_ctx`)**: Elimination of the artificial overlap that duplicated the sequences.
- **Optimal volume according to Chinchilla (`--max-tokens 50000000`)**: For the ~5M parameter model, Chinchilla's law recommends ≈20 tokens per parameter during training; 50M tokens (× 3 epochs) is the mathematically ideal volume to saturate a model of this size. The CLI's **current default** is more conservative (`10M`) to allow fast iterations; adjust according to the time available.
- **Full CLI Parameterization**: Implementation of flags in `src/main.rs` to control `--batch-size`, `--epochs`, `--max-tokens`, `--step`, `--n-ctx`, `--n-embd`, `--n-head`, `--n-layer` and `--lr`.

---

## Chapter 5: The Leap to the 2026 Architecture (LLaMA / Qwen Standard)

With the base stabilized, the model architecture was refactored in `src/model.rs` to align with the standard of modern 2026 LLMs (such as LLaMA 3/4, Qwen 2.5/3 and Gemma 2/3):

| Component | GPT-2 Baseline (Previous) | Noctámbulo 2026 (Current) | Technical Benefit |
|:---|:---|:---|:---|
| **Positions** | Learned Embeddings (`wpe`) | **RoPE (Rotary Position Embeddings)** | Eliminates the fixed `wpe` matrix. Enables dynamic context at inference. |
| **Normalization** | `LayerNorm` (with bias) | **RMSNorm** | Root-mean-square normalization without bias. ~15% faster on GPU. |
| **MLP Activation** | `GELU` | **SwiGLU FFN** | Triple gated network ($\text{SiLU}(X W_g) \odot X W_u$) $W_d$. Greater representational capacity. |
| **Inference** | Basic Random Sampling | **Top-K + Top-P + Repetition Penalty** | Eliminates repetitive loops and incoherent stories. |
| **Scheduler** | Fixed learning rate | **Cosine LR + Linear Warmup** | Smooth gradient calibration to reach lower losses (loss). |

---

## Chapter 6: Native GGUF v3 Binary Exporter

A pure binary serializer was created in Rust (`src/export.rs`) that writes the model directly in **GGUF v3** format:

1. **Header Structure**: Writes the magic signature `b"GGUF"` and the version `3u32`.
2. **Architecture Metadata**: Configures `"general.architecture" = "llama"`, `"general.name" = "noctambulo-2026"`, and inserts the number of layers, heads and context length.
3. **LLaMA Tensor Mapping**: Maps the Burn tensors to the standard llama.cpp nomenclature:
   - `token_embd.weight`
   - `blk.N.attn_norm.weight`
   - `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, `blk.N.attn_v.weight`, `blk.N.attn_output.weight`
   - `blk.N.ffn_norm.weight`
   - `blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight`, `blk.N.ffn_down.weight`
   - `output_norm.weight`
   - `output.weight`
4. **32-Byte Alignment**: Zero-pads the gaps between tensors to comply with the SIMD/GPU alignment requirement of **LM Studio** and **llama.cpp**.

---

## 🚀 Quick Guide and Usage Tutorial

### 1. Train the Model
```bash
# Optimized training (50M tokens, batch 128, 3 epochs)
cargo run --release train --dataset roneneldan/TinyStories

# Quick 2-minute test (10M tokens)
cargo run --release train --dataset roneneldan/TinyStories --max-tokens 10000000
```

### 2. Generate Text
```bash
cargo run --release generate "Once upon a time" --temp 0.7 --top-k 40 --top-p 0.9 --rep-penalty 1.1
```

### 3. Export to GGUF for LM Studio / Ollama
```bash
cargo run --release -- export --model-name noctambulo
```
The `.gguf` file generated in `exported/` can be dragged directly into **LM Studio** or loaded in **llama.cpp**.

---

## 🐣 For Dummies Appendix: Concepts Explained Without Technical Jargon

If all of the above sounds like Mandarin Chinese, this section explains each concept with simple real-life analogies.

### 1. What exactly is an LLM?
It is a **mobile phone keyboard autocorrect**, but on steroids. The phone's autocorrect looks at the last 2 or 3 words and guesses the next one. An LLM looks at hundreds or thousands of previous words, understands the relationships between them and predicts the word that makes the most sense next.

### 2. What are Tokens and Tokenization?
Artificial Intelligence **cannot read letters or whole words**, it only understands numbers. 
- **Tokenizing** is passing the text through some "smart scissors" that cut the words into little pieces (tokens). For example, the word `unfortunately` could be cut into `un-`, `fortunate-`, `ly`.
- Each little piece has a number assigned in a dictionary. The AI only sees lists of numbers like `[482, 1902, 53]`.

### 3. What is the Context Window (`n_ctx`)?
It is the AI's **short-term memory**. 
- If the context is 64 tokens, the AI can only look at the last ~40 words to know what the conversation is about. Everything before that is completely forgotten.
- If we raise the context to 512 or 2048, the AI can remember a whole chapter and won't change the protagonist's name in the middle of the story.

### 4. Why is RoPE better than the old positional method (`wpe`)?
- **The old method (`wpe`)**: It was like numbered seats in a movie theater. The AI had a fixed table that said "Seat 1, Seat 2... up to Seat 64". If you asked it to sit in seat 65, the AI panicked because that seat didn't exist on its map.
- **RoPE (2026 Method)**: Instead of fixed seats, the words "rotate" like clock hands depending on where they are. The AI doesn't memorize seats, but the angular distance between words. This makes it possible to expand the theater from 64 to 4,000 seats without having to redo the AI.

### 5. What is Loss and why do we want it to go down?
**Loss is the failing grade on the AI's exam**:
- At the beginning of training, the AI says random nonsense and gets a very high failing grade (**Loss = 9.0**).
- As it reads millions of sentences and adjusts its neurons, it starts getting the words right and the failing grade goes down (**Loss = 2.8**).
- **The lower the Loss, the better and more coherently the AI speaks**.

### 6. What was the 262 MB "One-Hot Memory Bomb" and why did we remove it?
Imagine that to tell someone the word number 4, instead of simply saying "number 4", you print an entire dictionary of **8,000 blank pages** and only put an `X` on page 4.
- That's what the previous code did: it printed 8,000 blank pages in the graphics card's memory on every training step (**262 Megabytes wasted per batch**).
- By switching to **sparse loss**, we went to sending the number 4 directly (**only 32 Kilobytes**). The graphics card went from drowning to flying.

### 7. What are RMSNorm and SwiGLU?
- **RMSNorm**: It's like removing the luggage and the rear seats from a race car to make it lighter. It eliminates unnecessary computations (the mean and the biases) so the GPU processes faster.
- **SwiGLU**: It's like adding a turbocharger to the engine. It is a type of "neuron" with a gating filter that helps the AI capture complex nuances of language with fewer total neurons.

### 8. What do Top-K, Top-P and Repetition Penalty do when generating text?
When the AI makes up a story, it doesn't always choose the most probable word (if it did, it would be incredibly boring and repetitive). It chooses by taking chances among several options, but we use three safety filters:
- **Top-K (e.g. 40)**: "Out of the 8,000 words in the dictionary, throw away the 7,960 rarest and choose only among the best 40".
- **Top-P / Nucleus (e.g. 0.9)**: "Keep only the words that together add up to 90% probability of making sense".
- **Repetition Penalty**: "If the AI starts repeating *'once upon a time upon a time upon a time'*, we give it a little pinch to force it to change words".

### 9. What is a GGUF file and why does it matter to us?
It is the **universal file format for AI** (equivalent to what `.mp3` is for songs or `.pdf` for documents).
- It packs all the AI's neurons into a single binary file.
- It lets easy-to-use desktop programs like **LM Studio** or **llama.cpp** open your model on Windows, Mac or Linux by dragging and dropping the file, with no need to know how to program.

---

## Chapter 7: The Definitive Architecture (Qwen2.5 / Qwen3 Standard)

### The Quality Diagnosis and Solution
1. **The Weight Tying Initialization Trap**:
   - When Weight Tying was enabled, the Burn library initialized the shared matrix with a standard normal distribution ($\sigma = 1.0$).
   - With $\sigma = 1.0$, the output matrix product generated gigantic logits ($\pm 300$), saturating the softmax and destroying the gradients.
   - **Qwen Solution**: Set the embedding initialization to the **$\mathcal{N}(0, 0.02)$** distribution, keeping the logits in the controlled range of $[-2, +2]$ and allowing smooth convergence.

2. **QKV Bias Attention**:
   - Following the official specification of **Qwen2.5 / Qwen3** (Alibaba), the $Q, K, V$ projections now include **bias** (`with_bias(true)`), while the output and MLP projections remain bias-free. This stabilizes the gradient flow in reduced-size models.

3. **Official Qwen Parameterization**:
    - **RMSNorm Epsilon**: $\epsilon = 10^{-6}$.
    - **RoPE Base Frequency**: $\theta = 10.000$ (for standard contexts / optimized for n_ctx <= 8K).
    - **Weight Decay**: $0.1$ to prevent overfitting in multi-epoch training runs.

4. **GGUF v3 Compatibility**:
   - The exporter serializes the attention bias tensors (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`) and exports the tied output matrix, guaranteeing full compatibility with `llama.cpp` and LM Studio.

---

## Chapter 8: Advanced Optimizations (GQA, Min-P, Softcapping & Stats CLI)

### 1. Grouped Attention (GQA - Grouped Query Attention)
- **The MHA Problem**: In standard Multi-Head Attention, each query head has its own key and value head ($K, V$), multiplying VRAM memory consumption for each generated token.
- **The GQA Solution**: Several query heads share the same $K, V$ head (`n_kv_head < n_head`). This reduces KV-Cache memory consumption by up to **75%**, allowing larger batch sizes and $4\times$ faster inference.

### 2. Min-P Mathematical Sampling
- Unlike Top-P (which uses a static cumulative cutoff), **Min-P** computes a dynamic threshold based on the probability of the most probable token ($T = \text{min\_p} \times P_{\max}$).
- When the model is 99% sure of an answer, Min-P aggressively trims noisy tokens. When the model is uncertain, Min-P preserves creative alternatives.

### 3. Logit Softcapping (Gemma 4 Math)
- Applies a soft bounding function to the attention scores:
  $$\text{scores} = \text{cap} \times \tanh\left(\frac{Q K^T}{\sqrt{d_k} \times \text{cap}}\right)$$
- Guarantees that no exponential in the softmax overflows the GPU's floating-point precision on long sequences.

### 4. `stats` Subcommand
- Allows instantly computing the model's complete mathematical footprint: total parameters, FP32 size in megabytes, FLOPs per token and VRAM savings ratio from GQA.
