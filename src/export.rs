use crate::config::ModelConfig;
use crate::model::Llm;
use burn::tensor::backend::Backend;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;

const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGML_TYPE_F32: u32 = 0;
const ALIGNMENT: u64 = 32;

fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<usize> {
    let bytes = s.as_bytes();
    let len = bytes.len() as u64;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(8 + bytes.len())
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    transposed
}

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

fn load_vocab_tokens(tokenizer_path: &str, vocab_size: usize) -> Vec<String> {
    let mut tokens = vec![String::new(); vocab_size];
    if Path::new(tokenizer_path).exists() {
        if let Ok(content) = std::fs::read_to_string(tokenizer_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(vocab_map) = v.get("model").and_then(|m| m.get("vocab")).and_then(|v| v.as_object()) {
                    for (tok_str, id_val) in vocab_map {
                        if let Some(id) = id_val.as_u64() {
                            if (id as usize) < vocab_size {
                                tokens[id as usize] = tok_str.clone();
                            }
                        }
                    }
                }
            }
        }
    }
    for (i, tok) in tokens.iter_mut().enumerate() {
        if tok.is_empty() {
            *tok = format!("<token_{}>", i);
        }
    }
    tokens
}

fn load_vocab_token_types(vocab_size: usize) -> Vec<i32> {
    let mut types = vec![1i32; vocab_size]; // 1 = GGML_TOKEN_TYPE_NORMAL
    if vocab_size > 0 { types[0] = 3; } // 3 = GGML_TOKEN_TYPE_CONTROL (<|endoftext|>)
    if vocab_size > 1 { types[1] = 3; } // 3 = GGML_TOKEN_TYPE_CONTROL (<|padding|>)
    types
}

fn load_vocab_merges(tokenizer_path: &str) -> Vec<String> {
    let mut merges = Vec::new();
    if Path::new(tokenizer_path).exists() {
        if let Ok(content) = std::fs::read_to_string(tokenizer_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(merges_arr) = v.get("model").and_then(|m| m.get("merges")).and_then(|m| m.as_array()) {
                    for m_val in merges_arr {
                        if let Some(pair) = m_val.as_array() {
                            if pair.len() == 2 {
                                if let (Some(a), Some(b)) = (pair[0].as_str(), pair[1].as_str()) {
                                    merges.push(format!("{} {}", a, b));
                                }
                            }
                        } else if let Some(m_str) = m_val.as_str() {
                            merges.push(m_str.to_string());
                        }
                    }
                }
            }
        }
    }
    merges
}

#[derive(Debug)]
struct NamedTensor {
    name: String,
    dims: Vec<u64>,
    data: Vec<f32>,
}

fn generate_model_card(model_name: &str, config: &ModelConfig) -> String {
    format!(
r#"---
license: mit
base_model: kuro-neko-2026
tags:
- rust
- burn
- llama
- gguf
- text-generation
pipeline_tag: text-generation
---

# 🐈‍⬛ {}

**{}** is a 2026-architecture Causal Language Model trained in pure Rust using [Burn 0.21](https://burn.dev) and WGPU acceleration.

## 📐 Model Architecture Specs

- **Architecture Family**: LLaMA / Qwen (2026 Standard)
- **Position Embeddings**: Rotary Position Embeddings (RoPE)
- **Normalization**: Root Mean Square Normalization (RMSNorm)
- **Activation**: SwiGLU Gated Feed-Forward Network
- **Vocabulary Size**: {}
- **Embedding Dimension (`n_embd`)**: {}
- **Attention Heads (`n_head`)**: {}
- **Transformer Layers (`n_layer`)**: {}
- **Context Window (`n_ctx`)**: {}

## 🚀 Quickstart & Usage

### 1. LM Studio
1. Drag and drop the `model.gguf` file into LM Studio.
2. Load the model and start chatting or generating text!

### 2. llama.cpp / Ollama
```bash
llama-cli -m model.gguf -p "Once upon a time"
```

### 3. Uploading to Hugging Face
To upload this model package directly to Hugging Face:

```bash
huggingface-cli login
git clone https://huggingface.co/your-username/{}
cp -r exported/{}/. {}/
cd {}
git add .
git commit -m "Initial model release"
git push
```
"#,
        model_name,
        model_name,
        config.vocab_size,
        config.n_embd,
        config.n_head,
        config.n_layer,
        config.n_ctx,
        model_name,
        model_name,
        model_name,
        model_name,
    )
}

#[allow(clippy::type_complexity)]
pub fn export_to_gguf<B: Backend>(
    model: &Llm<B>,
    config: &ModelConfig,
    model_name: &str,
    tokenizer_path: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let export_dir = format!("exported/{}", model_name);
    std::fs::create_dir_all(&export_dir)?;

    let gguf_path = format!("{}/model.gguf", export_dir);
    println!("Exporting 2026 LLaMA-style package to '{}'...", export_dir);

    let vocab_tokens = load_vocab_tokens(tokenizer_path, config.vocab_size);
    let token_types = load_vocab_token_types(config.vocab_size);
    let vocab_merges = load_vocab_merges(tokenizer_path);
    println!("Loaded {} tokens, {} types, and {} BPE merges for GGUF metadata.", vocab_tokens.len(), token_types.len(), vocab_merges.len());

    let mut named_tensors: Vec<NamedTensor> = Vec::new();

    // 1. Embeddings
    if let Ok(emb) = model.embed_tokens.weight.val().into_data().to_vec::<f32>() {
        named_tensors.push(NamedTensor {
            name: "token_embd.weight".to_string(),
            dims: vec![config.n_embd as u64, config.vocab_size as u64],
            data: emb,
        });
    }

    // 2. Transformer Blocks (LLaMA naming conventions)
    for (i, block) in model.blocks.iter().enumerate() {
        let prefix = format!("blk.{}", i);

        // Input RmsNorm
        if let Ok(gamma) = block.input_layernorm.weight.clone().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_norm.weight", prefix),
                dims: vec![config.n_embd as u64],
                data: gamma,
            });
        }

        // Self Attention Projections (Q, K, V, O) - Transpose from Burn [in, out] to PyTorch/GGML [out, in] + Permute RoPE
        if let Ok(q) = block.attn.q_proj.weight.val().into_data().to_vec::<f32>() {
            let q_trans = transpose_2d(&q, config.n_embd, config.n_embd);
            let q_permuted = permute_rope_weight(&q_trans, config.n_head, config.n_embd);
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_q.weight", prefix),
                dims: vec![config.n_embd as u64, config.n_embd as u64],
                data: q_permuted,
            });
        }
        if let Ok(k) = block.attn.k_proj.weight.val().into_data().to_vec::<f32>() {
            let k_trans = transpose_2d(&k, config.n_embd, config.n_embd);
            let k_permuted = permute_rope_weight(&k_trans, config.n_head, config.n_embd);
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_k.weight", prefix),
                dims: vec![config.n_embd as u64, config.n_embd as u64],
                data: k_permuted,
            });
        }
        if let Ok(v) = block.attn.v_proj.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_v.weight", prefix),
                dims: vec![config.n_embd as u64, config.n_embd as u64],
                data: transpose_2d(&v, config.n_embd, config.n_embd),
            });
        }
        if let Ok(out) = block.attn.out_proj.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_output.weight", prefix),
                dims: vec![config.n_embd as u64, config.n_embd as u64],
                data: transpose_2d(&out, config.n_embd, config.n_embd),
            });
        }

        // Post-Attention RmsNorm
        if let Ok(gamma) = block.post_attention_layernorm.weight.clone().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.ffn_norm.weight", prefix),
                dims: vec![config.n_embd as u64],
                data: gamma,
            });
        }

        // SwiGLU MLP Projections (Gate, Up, Down) - Transpose 2D linear weights
        let hidden_dim = (8 * config.n_embd) / 3;
        if let Ok(gate) = block.mlp.gate_proj.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.ffn_gate.weight", prefix),
                dims: vec![config.n_embd as u64, hidden_dim as u64],
                data: transpose_2d(&gate, config.n_embd, hidden_dim),
            });
        }
        if let Ok(up) = block.mlp.up_proj.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.ffn_up.weight", prefix),
                dims: vec![config.n_embd as u64, hidden_dim as u64],
                data: transpose_2d(&up, config.n_embd, hidden_dim),
            });
        }
        if let Ok(down) = block.mlp.down_proj.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.ffn_down.weight", prefix),
                dims: vec![hidden_dim as u64, config.n_embd as u64],
                data: transpose_2d(&down, hidden_dim, config.n_embd),
            });
        }
    }

    // 3. Final Norm & Output Head
    if let Ok(norm) = model.norm.weight.clone().into_data().to_vec::<f32>() {
        named_tensors.push(NamedTensor {
            name: "output_norm.weight".to_string(),
            dims: vec![config.n_embd as u64],
            data: norm,
        });
    }

    if let Ok(head) = model.lm_head.weight.val().into_data().to_vec::<f32>() {
        named_tensors.push(NamedTensor {
            name: "output.weight".to_string(),
            dims: vec![config.n_embd as u64, config.vocab_size as u64],
            data: transpose_2d(&head, config.n_embd, config.vocab_size),
        });
    }

    println!("  Extracted {} parameter tensors for GGUF serialization.", named_tensors.len());

    let file = File::create(&gguf_path)?;
    let mut w = BufWriter::new(file);
    let mut current_offset: u64 = 0;

    // Header
    w.write_all(GGUF_MAGIC)?;
    w.write_all(&GGUF_VERSION.to_le_bytes())?;

    let tensor_count = named_tensors.len() as u64;
    let hidden_dim = (8 * config.n_embd) / 3;

    let tokens_clone = vocab_tokens.clone();
    let types_clone = token_types.clone();
    let merges_clone = vocab_merges.clone();

    let kv_pairs: Vec<(&str, u32, Box<dyn Fn(&mut BufWriter<File>) -> std::io::Result<u64>>)> = vec![
        ("general.architecture", GGUF_TYPE_STRING, Box::new(|w| { Ok(write_gguf_string(w, "llama")? as u64) })),
        ("general.name", GGUF_TYPE_STRING, Box::new(move |w| { Ok(write_gguf_string(w, model_name)? as u64) })),
        ("llama.context_length", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&((config.n_ctx * 2) as u32).to_le_bytes())?; Ok(4) })),

        ("llama.embedding_length", GGUF_TYPE_UINT32, Box::new(|w| { w.write_all(&(config.n_embd as u32).to_le_bytes())?; Ok(4) })),
        ("llama.block_count", GGUF_TYPE_UINT32, Box::new(|w| { w.write_all(&(config.n_layer as u32).to_le_bytes())?; Ok(4) })),
        ("llama.feed_forward_length", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&(hidden_dim as u32).to_le_bytes())?; Ok(4) })),
        ("llama.attention.head_count", GGUF_TYPE_UINT32, Box::new(|w| { w.write_all(&(config.n_head as u32).to_le_bytes())?; Ok(4) })),
        ("llama.attention.head_count_kv", GGUF_TYPE_UINT32, Box::new(|w| { w.write_all(&(config.n_head as u32).to_le_bytes())?; Ok(4) })),
        ("llama.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, Box::new(|w| { w.write_all(&1e-5f32.to_le_bytes())?; Ok(4) })),
        ("tokenizer.ggml.model", GGUF_TYPE_STRING, Box::new(|w| { Ok(write_gguf_string(w, "gpt2")? as u64) })),
        ("tokenizer.ggml.pre", GGUF_TYPE_STRING, Box::new(|w| { Ok(write_gguf_string(w, "gpt-2")? as u64) })),
        ("tokenizer.ggml.add_prefix_space", GGUF_TYPE_BOOL, Box::new(|w| { w.write_all(&[1u8])?; Ok(1) })),

        ("tokenizer.ggml.add_bos_token", GGUF_TYPE_BOOL, Box::new(|w| { w.write_all(&[0u8])?; Ok(1) })),
        ("tokenizer.ggml.bos_token_id", GGUF_TYPE_UINT32, Box::new(|w| { w.write_all(&0u32.to_le_bytes())?; Ok(4) })),
        ("tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32, Box::new(|w| { w.write_all(&0u32.to_le_bytes())?; Ok(4) })),
        ("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, Box::new(move |w| {
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            w.write_all(&(tokens_clone.len() as u64).to_le_bytes())?;
            let mut written = 12u64;
            for tok in &tokens_clone {
                written += write_gguf_string(w, tok)? as u64;
            }
            Ok(written)
        })),
        ("tokenizer.ggml.scores", GGUF_TYPE_ARRAY, Box::new(move |w| {
            w.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes())?;
            w.write_all(&(vocab_tokens.len() as u64).to_le_bytes())?;
            let mut written = 12u64;
            let zero = 0.0f32.to_le_bytes();
            for _ in 0..vocab_tokens.len() {
                w.write_all(&zero)?;
                written += 4;
            }
            Ok(written)
        })),
        ("tokenizer.ggml.token_type", GGUF_TYPE_ARRAY, Box::new(move |w| {
            w.write_all(&GGUF_TYPE_INT32.to_le_bytes())?;
            w.write_all(&(types_clone.len() as u64).to_le_bytes())?;
            let mut written = 12u64;
            for &t in &types_clone {
                w.write_all(&t.to_le_bytes())?;
                written += 4;
            }
            Ok(written)
        })),
        ("tokenizer.ggml.merges", GGUF_TYPE_ARRAY, Box::new(move |w| {
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            w.write_all(&(merges_clone.len() as u64).to_le_bytes())?;
            let mut written = 12u64;
            for m in &merges_clone {
                written += write_gguf_string(w, m)? as u64;
            }
            Ok(written)
        })),
    ];

    w.write_all(&tensor_count.to_le_bytes())?;
    w.write_all(&(kv_pairs.len() as u64).to_le_bytes())?;
    current_offset += 4 + 4 + 8 + 8;

    // Metadata KV
    for (key, val_type, write_fn) in kv_pairs {
        current_offset += write_gguf_string(&mut w, key)? as u64;
        w.write_all(&val_type.to_le_bytes())?;
        current_offset += 4;
        let value_bytes_written = write_fn(&mut w)?;
        current_offset += value_bytes_written;
    }

    // Calculate tensor offsets & write tensor info headers
    let mut tensor_data_offsets: Vec<u64> = Vec::with_capacity(named_tensors.len());
    let mut data_offset_counter: u64 = 0;

    for tensor in &named_tensors {
        let pad = (ALIGNMENT - (data_offset_counter % ALIGNMENT)) % ALIGNMENT;
        data_offset_counter += pad;
        tensor_data_offsets.push(data_offset_counter);

        let byte_len = (tensor.data.len() * 4) as u64;
        data_offset_counter += byte_len;

        current_offset += write_gguf_string(&mut w, &tensor.name)? as u64;

        let n_dims = tensor.dims.len() as u32;
        w.write_all(&n_dims.to_le_bytes())?;
        current_offset += 4;

        for &dim in &tensor.dims {
            w.write_all(&dim.to_le_bytes())?;
            current_offset += 8;
        }

        w.write_all(&GGML_TYPE_F32.to_le_bytes())?;
        w.write_all(&tensor_data_offsets.last().copied().unwrap_or(0).to_le_bytes())?;
        current_offset += 4 + 8;
    }

    // Align header to 32-byte boundary before data section
    let header_pad = (ALIGNMENT - (current_offset % ALIGNMENT)) % ALIGNMENT;
    if header_pad > 0 {
        let pad_bytes = vec![0u8; header_pad as usize];
        w.write_all(&pad_bytes)?;
    }

    // Write raw tensor float data
    for (tensor, &_offset) in named_tensors.iter().zip(tensor_data_offsets.iter()) {
        let pad = (ALIGNMENT - ((w.stream_position()? - (current_offset + header_pad)) % ALIGNMENT)) % ALIGNMENT;
        if pad > 0 && pad < ALIGNMENT {
            let pad_bytes = vec![0u8; pad as usize];
            w.write_all(&pad_bytes)?;
        }

        let raw_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                tensor.data.as_ptr() as *const u8,
                tensor.data.len() * std::mem::size_of::<f32>(),
            )
        };
        w.write_all(raw_bytes)?;
    }

    w.flush()?;

    // Copy config.json
    let config_dest = format!("{}/config.json", export_dir);
    let config_json = serde_json::to_string_pretty(config)?;

    std::fs::write(&config_dest, config_json)?;

    // Copy tokenizer.json if available
    if Path::new(tokenizer_path).exists() {
        let tokenizer_dest = format!("{}/tokenizer.json", export_dir);
        std::fs::copy(tokenizer_path, tokenizer_dest)?;
    }

    // Generate Hugging Face README.md Model Card
    let model_card_path = format!("{}/README.md", export_dir);
    let model_card_content = generate_model_card(model_name, config);
    std::fs::write(&model_card_path, model_card_content)?;

    println!("✅ Package exported successfully to '{}':", export_dir);
    println!("  - {}/model.gguf (GGUF v3 Binary Weights with RoPE Permuted Q/K Matrices)", export_dir);
    println!("  - {}/config.json (Model Architecture Specs)", export_dir);
    println!("  - {}/tokenizer.json (Tokenizer Dictionary)", export_dir);
    println!("  - {}/README.md (Hugging Face Model Card)", export_dir);

    Ok(())
}
