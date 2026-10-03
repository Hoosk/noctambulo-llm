# 📊 Kuro-Neko — Model Statistics (`stats`)

The `stats` subcommand provides a comprehensive diagnosis of the model architecture, the exact number of trainable parameters, the memory consumption in single precision FP32, the estimated computational intensity in FLOPs per token, the VRAM compression achieved through **Grouped Query Attention (GQA)**, and the estimated KV cache footprint per token and at full context (f32/f16).

---

## ⚙️ Subcommand Usage

The `stats` subcommand **requires no parameters** on the command line:

```bash
cargo run --release -- stats
```

---

## 📂 Source of Configuration Data

When executed, the subcommand performs the following procedure:
1. It looks for the model configuration file at `model_output/config.json`.
2. If the file exists, it reads the hyperparameters of the trained model (`n_embd`, `n_head`, `n_kv_head`, `n_layer`, `vocab_size`, `n_ctx`, `attn_softcap`).
3. If the file does not exist, it uses the system default values (`vocab_size=8000`, `n_embd=256`, `n_head=4`, `n_kv_head=4`, `n_layer=4`, `n_ctx=256`).

---

## 📐 Calculated Metrics and Mathematical Formulas

### 1. Parameter Breakdown per Transformer Block
For an embedding dimension $D = \text{n\_embd}$, number of layers $L = \text{n\_layer}$, vocabulary $V = \text{vocab\_size}$, heads $H = \text{n\_head}$, and KV heads $H_{\text{kv}} = \text{n\_kv\_head}$:

- **Dimension per Head**: $d_k = D / H$.
- **Key/Value Dimension**: $D_{\text{kv}} = H_{\text{kv}} \times d_k$.
- **SwiGLU Hidden Dimension** (Aligned to multiples of 256):
  $$D_{\text{hidden}} = \left\lceil \frac{\lfloor 8D / 3 \rfloor}{256} \right\rceil \times 256$$
- **Parameters per Layer/Block**:
  - Projection $Q$ (with bias): $D \times D + D$
  - Projections $K, V$ (with bias): $2 \times (D \times D_{\text{kv}} + D_{\text{kv}})$
  - Projection $O$ (without bias): $D \times D$
  - SwiGLU MLP layers (Gate, Up, Down): $2 \times (D \times D_{\text{hidden}}) + (D_{\text{hidden}} \times D)$
  - Layer RMSNorms: $2 \times D$
- **Total Parameters**:
  $$N_{\text{total}} = (V \times D) + L \times \text{Parameters\_per\_Block} + D_{\text{norm\_final}}$$

---

### 2. FP32 Memory Consumption
Each parameter in 32-bit floating point precision (FP32) requires 4 bytes of storage:
$$\text{FP32 Memory (MB)} = \frac{N_{\text{total}} \times 4}{1024 \times 1024}$$

---

### 3. Floating Point Operations per Token (FLOPs/Token)
In an autoregressive forward pass, each parameter participates in approximately 2 floating point operations (one multiplication and one addition):
$$\text{FLOPs per Token} = 2 \times N_{\text{total}}$$

---

### 4. KV Cache Memory Savings with GQA
It measures the percentage reduction in the size of the Key ($K$) and Value ($V$) tensors in VRAM thanks to the use of Grouped Query Attention instead of conventional Multi-Head Attention:
$$\text{GQA Saving (\%)} = \left(1 - \frac{H_{\text{kv}}}{H}\right) \times 100\%$$

### 5. KV Cache Estimates (f32 / f16)
The report estimates the Key/Value cache footprint per token and for the full context window, for both f32 (4 bytes per element) and f16 (2 bytes per element):
$$\text{KV Cache (bytes/token)} = 2 \times H_{\text{kv}} \times d_k \times L \times \text{bytes per element}$$
$$\text{KV Cache (MB at n\_ctx)} = \frac{\text{bytes/token} \times \text{n\_ctx}}{1024 \times 1024}$$

---

## 🖥️ Example Command Output

When running `cargo run --release -- stats` with the base configuration, the subcommand generates a formatted report on the console like the following:

```text
===============================================================================
🐈‍⬛ Kuro-Neko 2026 Model Architecture Statistics
===============================================================================
  Vocab Size       : 8000 tokens
  Context Window   : 256 tokens
  Embedding Dim    : 256
  Hidden SwiGLU Dim: 768 (aligned to 256)
  Query Heads      : 4
  KV Heads (GQA)   : 2 (Ratio 1:2)
  Layers           : 4
  Softcapping Cap  : Disabled
-------------------------------------------------------------------------------
  Total Parameters : 3481088 (3.48 Million)
  Model Size (FP32): 13.28 MB
  FLOPs per Token  : 0.01 GFLOPs
  GQA Cache Saving : 50.0% KV-cache VRAM reduction
  KV Cache (f32)   : 4096 B/token (1.00 MB at n_ctx=256)
  KV Cache (f16)   : 2048 B/token (0.50 MB at n_ctx=256)
===============================================================================
```
