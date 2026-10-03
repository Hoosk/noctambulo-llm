# 🏋️ Kuro-Neko — Training (`train`)

The `train` subcommand is the core of the **Kuro-Neko** engine. It allows training from scratch an autoregressive Causal Transformer language model (Qwen2.5 / Qwen3 architecture) fully accelerated by GPU through the **WGPU** backend of the **Burn 0.21** framework.

During training, Kuro-Neko processes datasets in plain text format (coming from the **Hugging Face Hub** or a local file), encodes the text with the BPE tokenizer in parallel streaming (*streaming*), builds sliding windows of context sequences (`n-ctx`) and runs the AdamW optimization loop on the GPU.

---

## ⚙️ Full CLI Parameter Table

The `train` subcommand accepts **23 CLI** configuration parameters:

| Flag | Type | Default Value | Detailed Description |
| :--- | :--- | :--- | :--- |
| `--dataset <repo>` | `Vec<String>` (comma-separated) | `[]` | Hugging Face repository identifier(s) (e.g. `roneneldan/TinyStories`). Accepts multiple repositories separated by commas for combined training. |
| `--file <path>` | `Option<String>` | `None` | Path to a local text file on the filesystem to use as the training corpus. |
| `--dataset-file <filename>` | `String` | `TinyStories-train.txt` | Name of the target file to download and inspect inside the Hugging Face repository. |
| `--preset <name>` | `Option<String>` | `None` | Architecture preset (`nano`, `tiny`, `small`, `base`); explicit dimension flags take precedence. |
| `--dtype <name>` | `String` | `f32` | Compute precision (`f32` or `f16`). `f16` uses the unfused `CubeBackend` (burn-fusion 0.21 crashes with half streams on Vulkan) and is usually 2–3× faster; it can destabilize if the LR is high. `bf16` is rejected with a clear error because WGSL does not support it. Checkpoints are saved in f32. |
| `--batch-size <n>` | `usize` | `128` | Number of sequences per training batch (*batch*). A larger size makes better use of the GPU and accelerates convergence, but requires more VRAM. |
| `--epochs <n>` | `usize` | `3` | Number of complete passes through the processed dataset. |
| `--max-tokens <n>` | `usize` | `100000000` (100M) | Maximum number of tokens to load from the dataset. A value of `0` disables the limit and loads the full dataset. The reference Chinchilla budget is ~20 tokens per parameter. |
| `--step <n>` | `Option<usize>` | `None` (`= n-ctx`) | Sliding window step in tokens. Controls the overlap between consecutive sequences. If omitted, it equals `n-ctx` (no overlap). |
| `--n-ctx <n>` | `usize` | `256` | Length of the model's context window in tokens. Determines the maximum sequence size the model can observe at once. |
| `--n-embd <n>` | `usize` | `256` | Dimension of the embedding space and the hidden states. Larger values increase representational capacity. |
| `--n-head <n>` | `usize` | `4` | Number of attention heads for Query ($Q$). The embedding dimension `n-embd` must be divisible by `n-head`. |
| `--n-kv-head <n>` | `Option<usize>` | `None` (`= n-head`) | Number of Key/Value heads ($K, V$) for Grouped Query Attention (GQA). `None` or `0` enables Multi-Head Attention (MHA). A smaller value reduces VRAM consumption in the KV cache. |
| `--n-layer <n>` | `usize` | `4` | Number of Transformer blocks stacked in the model. A higher number of layers allows learning deeper hierarchical patterns. |
| `--lr <float>` | `f64` | `0.0003` (3e-4) | Peak learning rate. It is adjusted dynamically via a cosine schedule with *warmup*. |
| `--dropout <float>` | `f64` | `0.1` | Dropout probability applied to attention probabilities and to residual/MLP projections as regularization. |
| `--attn-softcap <float>` | `f32` | `0.0` | Softcapping threshold for attention scores. `0.0` disables the function. A value such as `50.0` applies the Gemma 4-style formula to stabilize gradients. |
| `--val-split <float>` | `f32` | `0.1` | Fraction of the corpus reserved for validation (computation of Loss and Perplexity per epoch). `0.0` disables validation. |
| `--grad-accum <n>` | `usize` | `1` | Gradient accumulation steps per AdamW update. The accumulated gradient is the **mean** of the micro-batches (equivalent to a larger batch). |
| `--save-dir <path>` | `Option<String>` | `None` | Directory where `checkpoint_epoch_N.bin` and `checkpoint_best.bin` are saved at the end of each epoch. |
| `--resume <path>` | `Option<String>` | `None` | Checkpoint `.bin` (or its directory) from which to resume. Restores weights, AdamW moments, schedule position and shuffler state. `--epochs` then means **additional epochs**. |
| `--seq-warmup-steps <n>` | `usize` | `0` | Initial *Progressive Sequence Scaling* steps: the context grows from `n_ctx/4` to `n_ctx` during warmup. `0` disables it. |
| `--early-stop-patience <n>` | `usize` | `0` | Stops training after `n` consecutive epochs without improving `val_loss`. `0` disables it. |

---

## 📈 Algorithm and Implementation Details

### 1. Cosine Learning Rate Schedule with Warmup

Kuro-Neko implements a cosine decay scheme for the learning rate (*Cosine Annealing Learning Rate Schedule*) complemented by an initial linear warmup phase (*Warmup*):

- **Warmup Steps**: Computed automatically as:
  $$\text{warmup\_steps} = \begin{cases} 0 & \text{if } \text{total\_steps} \le 10 \\ \min(1000, \lfloor \frac{\text{total\_steps}}{10} \rfloor) & \text{otherwise} \end{cases}$$
- **Minimum LR**: The final learning rate decays to 10% of the peak value ($\text{min\_lr} = \text{lr} \times 0.1$).
- **Phase 1 (Linear Warmup)**: During the first $\text{warmup\_steps}$, the LR increases strictly linearly from 0 to `lr`.
- **Phase 2 (Cosine Decay)**:
  $$\text{lr}_t = \text{min\_lr} + 0.5 \times (\text{lr} - \text{min\_lr}) \times \left(1 + \cos\left(\pi \times \frac{t - \text{warmup\_steps}}{\text{total\_steps} - \text{warmup\_steps}}\right)\right)$$

---

### 2. Gradient Clipping

To prevent exploding gradients in long sequences or during steps with high losses, the optimizer applies a strict $L_2$-norm clipping of **1.0** (`GradientClippingConfig::Norm(1.0)`).

If the Euclidean norm of the global gradient vector $g$ exceeds the value $1.0$, the gradients are rescaled according to:
$$g \leftarrow g \times \frac{1.0}{\max(1.0, \|g\|_2)}$$

---

### 3. AdamW Optimizer and Weight Decay

The **AdamW** optimizer is used with the following properties:
- **Weight Decay**: Set to $WD = 0.1$ through AdamW's built-in decoupled weight decay, applied to **all** parameters. `--weight-decay 0` disables it.
- Decay is applied directly to the weights, separate from the gradient moment update, preventing important weights from being disproportionately degraded.

---

### 4. Per-Epoch Data Shuffling (Fisher-Yates Shuffling)

In each training epoch, the set of indices of the processed sequences is randomly reordered using the **Fisher-Yates** algorithm driven by a deterministic `XorshiftRng` pseudo-random number generator:

```rust
fn shuffle_indices(indices: &mut [usize], rng: &mut XorshiftRng) {
    let len = indices.len();
    for i in (1..len).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize % (i + 1);
        indices.swap(i, j);
    }
}
```
This guarantees that the batches (*batches*) of each epoch contain different combinations of sequences, preventing the model from memorizing the corpus order.

---

### 5. Sparse NLL Loss Computation on GPU

The loss function for autoregressive language modeling is computed directly on the GPU using log-softmax and the `gather` operation:

1. **Log-Softmax**:
   $$\text{log\_probs} = \text{log\_softmax}(\text{logits\_flat}, \text{dim}=1)$$
2. **Gather**: Selects the log-probabilities corresponding to the target labels (*targets*):
   $$\text{target\_log\_probs} = \text{log\_probs}.\text{gather}(1, \text{targets\_flat})$$
3. **NLL Loss**:
   $$\text{Loss} = - \text{mean}(\text{target\_log\_probs})$$

> [!IMPORTANT]
> **Asynchronous GPU Accumulation**: To avoid CPU $\to$ GPU synchronization bottlenecks at each batch step, the computed loss is detached from the autodifferentiation graph (`detach()`) and accumulated in a tensor resident in GPU memory (`accum_loss_tensor`). It is only read back to the CPU at the configured reporting intervals.

---

### 6. Real Gradient Accumulation (`--grad-accum`)

With `--grad-accum N`, each AdamW update groups $N$ micro-batches. The loss of each micro-batch is divided by the real window size ($N$, or less in the final partial window of each epoch) and the gradients are summed with `burn::optim::GradientsAccumulator`:

$$\nabla_\theta \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_i = \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \mathcal{L}_i$$

This guarantees that accumulating $N$ micro-batches of size $B$ is **mathematically equivalent** to a single batch of size $N \times B$ (linearity of the gradient), which is exactly what the test `test_accumulated_gradients_match_full_batch` verifies.

---

### 7. Persistent Training State and Resumption

At the end of each epoch (and when training is stopped) Kuro-Neko saves, alongside the weights:

- `train_state.json`: `global_step`, `total_steps`, `epochs_done`, `best_val_loss` and the internal shuffler state.
- `optimizer_state.bin`: the AdamW record (first- and second-order moments) serialized with `Optimizer::to_record`.

When resuming with `--resume <checkpoint>`, the weights **and** both sidecars are restored, so the cosine schedule continues from the saved step, the AdamW moments are not lost and the sequence order continues where it left off. `--epochs` in a resume indicates **additional** epochs.

If `train_state.json` does not exist (older checkpoints), the resume is backward compatible: only the weights are loaded and a warning is issued that the AdamW state starts from zero.

### 8. Safe Stop (Stop / Ctrl+C)

Stopping a training run from the web server or with `Ctrl+C` (CLI) does not discard the work: before exiting, `checkpoint_interrupted.bin` is saved together with `train_state.json` and `optimizer_state.bin`, and the `Checkpoint` event is emitted with `kind = "interrupted"`. The web server also uses `with_graceful_shutdown` and waits up to 30 seconds for the worker to finish persisting.

### 9. Token Budget and Early Stopping

- **Chinchilla Warning**: if `--max-tokens` falls below ~20 tokens per parameter, the trainer issues a warning at startup (e.g. a 5.5M-param model asks for ~110M tokens). Use `--max-tokens 0` to train on the full dataset.
- **Early stopping**: with `--early-stop-patience N`, the loop ends when `val_loss` does not improve for `N` epochs; on exit the weights and state are saved as in a normal stop.
- **Validation metrics**: validation evaluates **the entire** split (there is no longer a fixed cap of 100 batches) and reports Loss, PPL and BPB. BPB (`loss / ln2 × tokens/byte`) allows comparing models trained with different tokenizers; it is not available for `.bin` datasets (they do not preserve the byte size).

---

## 💻 Practical Training Examples

### Example 1: Standard Training on TinyStories (Default Configuration)
Trains on the TinyStories dataset with 10M tokens, batch size 128 and 3 epochs:
```bash
cargo run --release -- train --dataset roneneldan/TinyStories
```

### Example 2: Quick Test Training with GQA (Grouped Query Attention)
Light training with 1 million tokens and a 2:1 GQA ratio (`n-head=4`, `n-kv-head=2`):
```bash
cargo run --release -- train \
  --dataset roneneldan/TinyStories \
  --max-tokens 1000000 \
  --n-head 4 \
  --n-kv-head 2 \
  --batch-size 64 \
  --epochs 1
```

### Example 3: Training on a Custom Local File
Trains using a local text file with an extended context window of 512 tokens and sliding window overlap (`step=256`):
```bash
cargo run --release -- train \
  --file ./my_documents.txt \
  --n-ctx 512 \
  --step 256 \
  --n-embd 384 \
  --n-layer 6 \
  --lr 0.0005
```

### Example 4: Advanced Configuration with Softcapping and Multiple Dataset Repos
Trains by combining two Hugging Face repositories, with attention Softcapping enabled at `50.0` and no token limit (`max-tokens=0`):
```bash
cargo run --release -- train \
  --dataset roneneldan/TinyStories,wikitext \
  --max-tokens 0 \
  --n-embd 512 \
  --n-head 8 \
  --n-kv-head 2 \
  --n-layer 8 \
  --attn-softcap 50.0 \
  --dropout 0.05
```

### Optimizer and Loss Options

`train` exposes the following regularization knobs:

```bash
cargo run --release -- train \
  --file corpus.txt \
  --weight-decay 0.1 \
  --label-smoothing 0.05 \
  --z-loss 0.001
```

- `--weight-decay` (default `0.1`) applies **decoupled** weight decay (AdamW) to **all**
  parameters; `0` disables it.
- `--label-smoothing` (default `0.0`) mixes the negative log-likelihood with the uniform
  distribution over the vocabulary.
- `--z-loss` (default `0.0`) adds `coef · mean(logsumexp(logits)²)`, which keeps logits
  from drifting when training with large vocabularies.

All three are also available in the web Train form and the JSON API.

Parquet corpora no longer need a manual extraction step: `prepare-dataset` tokenizes the
text column directly to `.bin`, and `--dedupe` removes exact duplicate lines/documents
before tokenizing.
