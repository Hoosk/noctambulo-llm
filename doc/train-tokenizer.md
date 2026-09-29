# 🔤 Kuro-Neko — Tokenizer Training (`train-tokenizer`)

The `train-tokenizer` subcommand allows training a **Byte-Level Byte-Pair Encoding (BPE)** tokenizer adapted to the specific vocabulary and domain of your training dataset.

When training finishes, the subcommand saves the tokenization model to the `custom_tokenizer.json` file, which is automatically consumed by the `train`, `generate` and `export` subcommands.

---

## ⚙️ Full CLI Parameter Table

| Flag | Type | Default Value | Detailed Description |
| :--- | :--- | :--- | :--- |
| `--dataset <repo>` | `Vec<String>` (comma-separated) | `[]` | Hugging Face repository identifier(s) (e.g. `roneneldan/TinyStories`). Multiple repositories can be combined to build the corpus. |
| `--file <path>` | `Option<String>` | `None` | Path to a local text file on the system to use as the corpus source for tokenizer training. |
| `--dataset-file <filename>` | `String` | `TinyStories-train.txt` | Name of the target file to download from the Hugging Face repository. |
| `--vocab-size <n>` | `usize` | `8000` | Total size of the vocabulary to generate (total number of subwords and learned tokens). |

---

## 🧩 What Is Byte-Level BPE Tokenization?

The **Byte-Level Byte-Pair Encoding (BPE)** algorithm operates directly on raw UTF-8 bytes instead of explicit Unicode subwords:

1. **256-Byte Base Representation**: Initially, the 256 individual bytes (`0x00` to `0xFF`) form the base of the vocabulary.
2. **Iterative Merging of Frequent Pairs**: The algorithm analyzes the text corpus and repeatedly finds the most frequent pair of contiguous bytes or tokens, merging it into a new single token.
3. **Zero Unknown Tokens (`<unk>`)**: Since any Unicode character or sequence in any language (including emojis, mathematical symbols or binary code) can be decomposed into its constituent UTF-8 bytes, a Byte-Level tokenizer **never fails or produces `<unk>` tokens**.

---

## 🔠 NFKC Unicode Normalization

Before proceeding with the BPE frequency computation, Kuro-Neko applies a strict Unicode normalization pipeline using the **NFKC (Normalization Form KC - Compatibility Decomposition followed by Canonical Composition)** standard:

- **Character Unification**: Converts visually similar characters or compatibility variants (such as ligatures, superscripts, full-width variants) to a normalized canonical representation.
- **Consistency in the Model**: Ensures that variants of the same character do not redundantly consume multiple entries in the vocabulary table.

---

## 🏷️ Special Tokens

The trained tokenizer includes and reserves two special tokens fundamental to sequence management in language models:

| Special Token | Function |
| :--- | :--- |
| `<|endoftext|>` | Indicates the end-of-document or end-of-sequence boundary. Used during training to separate independent stories or documents and during inference to stop autoregressive generation. |
| `<|padding|>` | Token used to pad (*pad*) batches of odd-length sequences when processed on the GPU. |

---

## 💾 Output File: `custom_tokenizer.json`

When the process completes, Kuro-Neko serializes the complete tokenizer structure in JSON format to the `custom_tokenizer.json` file at the project root. This file contains:

- The BPE merge map (*merges*).
- The token-to-ID and ID-to-token vocabulary table.
- The pre-tokenization rules (splitting on whitespace and punctuation).
- The NFKC normalization and special token configuration.

---

## 💻 Usage Examples

### Example 1: Train Tokenizer from a Hugging Face Repository
Trains an 8,000-token tokenizer on the `roneneldan/TinyStories` dataset:
```bash
cargo run --release -- train-tokenizer --dataset roneneldan/TinyStories --vocab-size 8000
```

### Example 2: Train Tokenizer on a Local File
Trains a more compact tokenizer (vocabulary of 4,096 tokens) from a local file:
```bash
cargo run --release -- train-tokenizer --file ./my_local_corpus.txt --vocab-size 4096
```

### Example 3: Train Large-Vocabulary Tokenizer for a Combined Corpus
Trains a large tokenizer (vocabulary of 16,384 tokens) by combining multiple datasets:
```bash
cargo run --release -- train-tokenizer \
  --dataset roneneldan/TinyStories,wikitext \
  --vocab-size 16384
```
