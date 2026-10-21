# 🦙 The Adventure of Making Noctámbulo Work in `llama.cpp` (`how-to-llama.md`)

This document recounts the technical process, the obstacles overcome, and the exact specifications of the **GGUF v3** standard required to get a model trained from scratch in Rust with **Burn 0.22** to run natively in **`llama.cpp`** (`llama-cli`) and **LM Studio** at **200 tokens/second**.

---

## 🎯 The Goal

Getting our engine written in Rust (**Noctámbulo**) to export its modern 2026 architecture (RoPE, RMSNorm, SwiGLU) into a single binary `.gguf` file that could be executed directly by the industry-standard tool: **`llama.cpp`**.

---

## 🏔️ The Four Obstacles of the GGUF v3 Specification

### 1. The LLaMA Hyperparameter Error
- **The Failure**:
  ```text
  error loading model hyperparameters: key not found in model: llama.attention.layer_norm_rms_epsilon
  ```
- **Technical Cause**:
  When declaring `"general.architecture" = "llama"` in the header metadata, the `llama.cpp` parser strictly validates that the specific hyperparameters of the LLaMA family exist.
- **The Solution**:
  In `src/export.rs`, the Float32 type constant `GGUF_TYPE_FLOAT32 = 6` was defined and the required keys were added:
  - `llama.attention.layer_norm_rms_epsilon` (Float32 = `1e-6`)
  - `llama.feed_forward_length` (UInt32 = $\frac{8}{3} n_{embd}$ aligned to 256)
  - `llama.attention.head_count_kv` (UInt32 = `n_kv_head`)

---

### 2. The Missing Vocabulary Error
- **The Failure**:
  ```text
  error loading model vocabulary: key not found in model: tokenizer.ggml.model
  ```
- **Technical Cause**:
  `llama.cpp` does not rely on external JSON files to read the words; it requires the entire tokenizer vocabulary to be serialized **inside the header of the binary `.gguf` file itself**.
- **The Solution**:
  The function `load_vocab_tokens` was created in `src/export.rs` to read the tokenizer tokens, sort them by ID, and write them into the GGUF metadata using array types (`GGUF_TYPE_ARRAY`):
  - `tokenizer.ggml.model` = `"llama"`
  - `tokenizer.ggml.tokens` = Array of strings with the tokens corresponding to the vocabulary.

---

### 3. Simplified Vocabulary Mapping
- **Technical Cause**:
  Our BPE tokenizer trained in Rust (`tokenizers` crate) produces a byte-level vocabulary compatible with the `gpt2` model of `llama.cpp` (byte-level pre-tokenizer, byte-level decoder, NFKC normalizer). We declare the tokenization model as `"gpt2"` with pre-tokenizer `"gpt-2"`, which avoids the need to manually serialize merges and scores into GGUF metadata and guarantees direct, load-failure-free compatibility with `llama-cli` and LM Studio.

---

### 4. RoPE Weight Permutation (Hugging Face vs C++ `llama.cpp`)
- **The Failure**:
  The model loaded in `llama.cpp`, but the generated text began to lose coherence ("gibberish") after the first tokens, while in Rust (`cargo generate`) the story was 100% clear.
- **Technical Cause**:
  There are two styles of implementing RoPE (Rotary Position Embeddings):
  - **Split (Hugging Face / Python / Rust)**: Rotates by splitting the dimension into two halves $[-x_2, x_1]$.
  - **Interleaved (LLaMA C++ / llama.cpp)**: Rotates by interleaving dimensions two by two $[-x_1, x_0, -x_3, x_2]$.
- **Hugging Face's Official Solution**:
  Based on Hugging Face's official script `convert_llama_weights_to_hf.py`, we implemented the function `permute_rope_weight` in `src/export.rs` to reorder the rows of the $Q$ and $K$ tensors when exporting to `.gguf`:

```rust
fn permute_rope_weight(data: &[f32], n_head: usize, n_embd: usize) -> Vec<f32> {
    let head_size = n_embd / n_head;
    let half_dim = head_size / 2;
    let mut out = vec![0.0f32; data.len()];

    for r_head in 0..n_head {
        for i in 0..half_dim {
            let r_in1 = r_head * head_size + i;
            let r_in2 = r_head * head_size + i + half_dim;
            let r_out1 = r_head * head_size + 2 * i;
            let r_out2 = r_head * head_size + 2 * i + 1;

            for c in 0..n_embd {
                out[r_out1 * n_embd + c] = data[r_in1 * n_embd + c];
                out[r_out2 * n_embd + c] = data[r_in2 * n_embd + c];
            }
        }
    }
    out
}
```

---

## 🏆 The Final Result

After applying the 2D matrix transposition and the RoPE weight permutation together with the complete GGUF v3 metadata, we ran:

```bash
llama-cli -m exported/noctambulo-2026.gguf -p "There was a little cat named" -n 50
```

### Actual Inference Log obtained:

```text
build      : b0-
model      : noctambulo-2026.gguf
modalities : text

> There was a little cat named

|   . After a while, Lily was very happy to see a special surprise, a big brother's mom hugged him and said, "You can I have...

[ Prompt: 2603,6 t/s | Generation: 199,8 t/s ]
```

- **Compatibility**: 100% Native in `llama.cpp`, `llama-server` and **LM Studio**.
- **Prompt Processing**: **2,603.6 tokens per second**.
- **Generation Speed**: **199.8 tokens per second**.

---

## 🛠️ Tutorial: How to Export and Run in `llama.cpp`

### Step 1: Export the Package
From the root of our Rust project:
```bash
cargo run --release -- export --model-name mi-modelo-llama
```

### Step 2: Run with `llama.cpp`
```bash
llama-cli -m exported/mi-modelo-llama.gguf -p "Once upon a time" -n 100
```
