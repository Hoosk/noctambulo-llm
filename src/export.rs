use crate::config::ModelConfig;
use crate::model::Llm;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;

const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGML_TYPE_F32: u32 = 0;
const ALIGNMENT: u64 = 32;

/// Inverse of [`permute_rope_weight`]: restores the original row order from a
/// RoPE-permuted weight (`out[2i] = in[i]`, `out[2i + 1] = in[i + half]`).
pub(crate) fn unpermute_rope_weight(
    data: &[f32],
    n_heads: usize,
    head_size: usize,
    cols: usize,
) -> Vec<f32> {
    let half_dim = head_size / 2;
    let mut out = vec![0.0f32; data.len()];

    for r_head in 0..n_heads {
        for i in 0..half_dim {
            let r_in1 = r_head * head_size + 2 * i;
            let r_in2 = r_head * head_size + 2 * i + 1;
            let r_out1 = r_head * head_size + i;
            let r_out2 = r_head * head_size + i + half_dim;

            for c in 0..cols {
                out[r_out1 * cols + c] = data[r_in1 * cols + c];
                out[r_out2 * cols + c] = data[r_in2 * cols + c];
            }
        }
    }
    out
}

/// Inverse of [`permute_rope_bias`].
pub(crate) fn unpermute_rope_bias(bias: &[f32], n_heads: usize, head_size: usize) -> Vec<f32> {
    let half_dim = head_size / 2;
    let mut out = vec![0.0f32; bias.len()];

    for r_head in 0..n_heads {
        for i in 0..half_dim {
            out[r_head * head_size + i] = bias[r_head * head_size + 2 * i];
            out[r_head * head_size + i + half_dim] = bias[r_head * head_size + 2 * i + 1];
        }
    }
    out
}

fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<usize> {
    let bytes = s.as_bytes();
    let len = bytes.len() as u64;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(8 + bytes.len())
}

pub(crate) fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    transposed
}

pub(crate) fn permute_rope_weight(data: &[f32], n_heads: usize, head_size: usize, cols: usize) -> Vec<f32> {
    let half_dim = head_size / 2;
    let mut out = vec![0.0f32; data.len()];

    for r_head in 0..n_heads {
        for i in 0..half_dim {
            let r_in1 = r_head * head_size + i;
            let r_in2 = r_head * head_size + i + half_dim;
            let r_out1 = r_head * head_size + 2 * i;
            let r_out2 = r_head * head_size + 2 * i + 1;

            for c in 0..cols {
                out[r_out1 * cols + c] = data[r_in1 * cols + c];
                out[r_out2 * cols + c] = data[r_in2 * cols + c];
            }
        }
    }
    out
}

pub(crate) fn permute_rope_bias(bias: &[f32], n_heads: usize, head_size: usize) -> Vec<f32> {
    let half_dim = head_size / 2;
    let mut out = vec![0.0f32; bias.len()];

    for r_head in 0..n_heads {
        for i in 0..half_dim {
            let r_in1 = r_head * head_size + i;
            let r_in2 = r_head * head_size + i + half_dim;
            let r_out1 = r_head * head_size + 2 * i;
            let r_out2 = r_head * head_size + 2 * i + 1;

            out[r_out1] = bias[r_in1];
            out[r_out2] = bias[r_in2];
        }
    }
    out
}

fn load_vocab_tokens(tokenizer_path: &str, vocab_size: usize) -> Vec<String> {
    let mut tokens = vec![String::new(); vocab_size];

    if Path::new(tokenizer_path).exists() {
        if let Ok(file_content) = std::fs::read_to_string(tokenizer_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&file_content) {
                if let Some(vocab_obj) = v.get("model").and_then(|m| m.get("vocab")).and_then(|v| v.as_object()) {
                    for (token, id_val) in vocab_obj {
                        if let Some(id) = id_val.as_u64() {
                            if (id as usize) < vocab_size {
                                tokens[id as usize] = token.clone();
                            }
                        }
                    }
                }
            }
        }
    }

    for (i, token) in tokens.iter_mut().enumerate() {
        if token.is_empty() {
            *token = format!("<token_{}>", i);
        }
    }
    tokens
}

fn load_vocab_token_types(vocab_size: usize) -> Vec<i32> {
    let mut types = vec![1i32; vocab_size]; // 1 = GGML_TOKEN_TYPE_NORMAL
    if vocab_size > 0 { types[0] = 3; } // 3 = GGML_TOKEN_TYPE_CONTROL
    if vocab_size > 1 { types[1] = 3; }
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

pub struct NamedTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub data: Vec<f32>,
}

/// Writer closure for one GGUF metadata key/value pair.
type GgufKvWriter<'a> = Box<dyn Fn(&mut BufWriter<File>) -> std::io::Result<u64> + 'a>;

/// CLI entry point: exports with no progress reporting and no cancellation.
pub fn export_to_gguf_cli(
    model: &Llm,
    config: &ModelConfig,
    tokenizer_path: &str,
    model_name: &str,
    output_dir: &str,
    approx: bool,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    export_to_gguf(
        model,
        config,
        tokenizer_path,
        model_name,
        output_dir,
        approx,
        &|_, _, _, _| {},
        &|| false,
    )
}

/// Serialize the model to GGUF v3. `progress(phase, done, total, unit)`
/// reports extraction/writing stages; `cancelled()` is polled between tensors
/// and aborts leaving the partial file behind for inspection/removal.
/// `approx` allows exporting models with cross-layer KV sharing by duplicating
/// the shared K/V weights under every layer (a standard-attention
/// approximation, not the exact grouped computation).
#[allow(clippy::too_many_arguments)]
pub fn export_to_gguf(
    model: &Llm,
    config: &ModelConfig,
    tokenizer_path: &str,
    model_name: &str,
    output_dir: &str,
    approx: bool,
    progress: &dyn Fn(&str, u64, u64, &str),
    cancelled: &dyn Fn() -> bool,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if config.kv_share_group > 1 && !approx {
        return Err(
            "cross-layer KV sharing (kv_share_group > 1) cannot be represented by llama.cpp; \
             pass --approx to export a standard-attention approximation (the shared K/V weights \
             are duplicated per layer, so the exported model is not identical)"
                .into(),
        );
    }
    std::fs::create_dir_all(output_dir)?;
    let gguf_path = format!("{}/{}.gguf", output_dir, model_name);

    println!("Exporting Qwen2.5/Qwen3 model architecture to GGUF format at: '{}'...", gguf_path);

    let vocab_tokens = load_vocab_tokens(tokenizer_path, config.vocab_size);
    let token_types = load_vocab_token_types(config.vocab_size);
    let vocab_merges = load_vocab_merges(tokenizer_path);

    let mut named_tensors: Vec<NamedTensor> = Vec::new();

    // 1. Token Embeddings
    let emb = model
        .embed_tokens
        .weight
        .val()
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| format!("Failed to extract embed_tokens weight: {:?}", e))?;

    named_tensors.push(NamedTensor {
        name: "token_embd.weight".to_string(),
        dims: vec![config.n_embd as u64, config.vocab_size as u64],
        data: emb.clone(),
    });

    // 2. Transformer Blocks
    let hidden_dim = config.ffn_hidden();
    progress("extracting tensors", 0, config.n_layer as u64, "layers");

    for (i, block) in model.blocks.iter().enumerate() {
        let prefix = format!("blk.{}", i);

        // Input RmsNorm
        let gamma = block.input_layernorm.weight.val().into_data().try_to_vec::<f32>()
            .map_err(|e| format!("Failed to extract input_layernorm weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.attn_norm.weight", prefix),
            dims: vec![config.n_embd as u64],
            data: gamma,
        });

        // Self Attention Projections (Q, K, V, O). K/V come from the layer's
        // sharing group (duplicated under every layer in --approx mode).
        let n_kv_head = config.get_n_kv_head();
        let head_dim = config.n_embd / config.n_head;
        let kv_dim = n_kv_head * head_dim;
        let group_idx = i / config.kv_share_group.max(1);

        let q = block.attn.q_proj.weight.val().into_data().try_to_vec::<f32>()
            .map_err(|e| format!("Failed to extract q_proj weight: {:?}", e))?;
        let q_trans = transpose_2d(&q, config.n_embd, config.n_embd);
        let q_permuted = permute_rope_weight(&q_trans, config.n_head, head_dim, config.n_embd);
        named_tensors.push(NamedTensor {
            name: format!("{}.attn_q.weight", prefix),
            dims: vec![config.n_embd as u64, config.n_embd as u64],
            data: q_permuted,
        });

        if let Some(ref q_b_param) = block.attn.q_proj.bias {
            let q_b = q_b_param.val().into_data().try_to_vec::<f32>()
                .map_err(|e| format!("Failed to extract q_proj bias: {:?}", e))?;
            let q_b_permuted = permute_rope_bias(&q_b, config.n_head, head_dim);
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_q.bias", prefix),
                dims: vec![config.n_embd as u64],
                data: q_b_permuted,
            });
        }

        // MLA models are folded back into an exact standard K/V projection:
        // W_eff = kv_down · k_up (Burn Linear layout [in, out]). No
        // approximation needed because this MLA variant has no decoupled RoPE.
        let (k, k_bias, v, v_bias) = if let Some(mla) = model.mla.get(group_idx) {
            let fold = |up: &burn::nn::Linear| -> Result<Vec<f32>, String> {
                mla.kv_down
                    .weight
                    .val()
                    .matmul(up.weight.val())
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| format!("Failed to fold MLA weights: {:?}", e))
            };
            (fold(&mla.k_up)?, None, fold(&mla.v_up)?, None)
        } else {
            let kv_group = &model.kv[group_idx];
            let k = kv_group.k_proj.weight.val().into_data().try_to_vec::<f32>()
                .map_err(|e| format!("Failed to extract k_proj weight: {:?}", e))?;
            let k_bias = match kv_group.k_proj.bias.as_ref() {
                Some(b) => Some(
                    b.val()
                        .into_data()
                        .try_to_vec::<f32>()
                        .map_err(|e| format!("Failed to extract k_proj bias: {:?}", e))?,
                ),
                None => None,
            };
            let v = kv_group.v_proj.weight.val().into_data().try_to_vec::<f32>()
                .map_err(|e| format!("Failed to extract v_proj weight: {:?}", e))?;
            let v_bias = match kv_group.v_proj.bias.as_ref() {
                Some(b) => Some(
                    b.val()
                        .into_data()
                        .try_to_vec::<f32>()
                        .map_err(|e| format!("Failed to extract v_proj bias: {:?}", e))?,
                ),
                None => None,
            };
            (k, k_bias, v, v_bias)
        };

        let k_trans = transpose_2d(&k, config.n_embd, kv_dim);
        let k_permuted = permute_rope_weight(&k_trans, n_kv_head, head_dim, config.n_embd);
        named_tensors.push(NamedTensor {
            name: format!("{}.attn_k.weight", prefix),
            dims: vec![config.n_embd as u64, kv_dim as u64],
            data: k_permuted,
        });

        if let Some(k_b) = k_bias {
            let k_b_permuted = permute_rope_bias(&k_b, n_kv_head, head_dim);
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_k.bias", prefix),
                dims: vec![kv_dim as u64],
                data: k_b_permuted,
            });
        }

        named_tensors.push(NamedTensor {
            name: format!("{}.attn_v.weight", prefix),
            dims: vec![config.n_embd as u64, kv_dim as u64],
            data: transpose_2d(&v, config.n_embd, kv_dim),
        });

        if let Some(v_b) = v_bias {
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_v.bias", prefix),
                dims: vec![kv_dim as u64],
                data: v_b,
            });
        }

        let out = block.attn.out_proj.weight.val().into_data().try_to_vec::<f32>()
            .map_err(|e| format!("Failed to extract out_proj weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.attn_output.weight", prefix),
            dims: vec![config.n_embd as u64, config.n_embd as u64],
            data: transpose_2d(&out, config.n_embd, config.n_embd),
        });

        // Post-Attention RmsNorm
        let gamma_post = block.post_attention_layernorm.weight.val().into_data().try_to_vec::<f32>()
            .map_err(|e| format!("Failed to extract post_attention_layernorm weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.ffn_norm.weight", prefix),
            dims: vec![config.n_embd as u64],
            data: gamma_post,
        });

        // SwiGLU MLP Projections (Gate, Up, Down)
        let gate = block.mlp.gate_proj.weight.val().into_data().try_to_vec::<f32>()
            .map_err(|e| format!("Failed to extract gate_proj weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.ffn_gate.weight", prefix),
            dims: vec![config.n_embd as u64, hidden_dim as u64],
            data: transpose_2d(&gate, config.n_embd, hidden_dim),
        });

        let up = block.mlp.up_proj.weight.val().into_data().try_to_vec::<f32>()
            .map_err(|e| format!("Failed to extract up_proj weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.ffn_up.weight", prefix),
            dims: vec![config.n_embd as u64, hidden_dim as u64],
            data: transpose_2d(&up, config.n_embd, hidden_dim),
        });

        let down = block.mlp.down_proj.weight.val().into_data().try_to_vec::<f32>()
            .map_err(|e| format!("Failed to extract down_proj weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.ffn_down.weight", prefix),
            dims: vec![hidden_dim as u64, config.n_embd as u64],
            data: transpose_2d(&down, hidden_dim, config.n_embd),
        });

        progress("extracting tensors", (i + 1) as u64, config.n_layer as u64, "layers");
        if cancelled() {
            return Err("cancelled by user".into());
        }
    }

    // 3. Final Norm & Tied Output Head
    let norm = model.norm.weight.val().into_data().try_to_vec::<f32>()
        .map_err(|e| format!("Failed to extract final norm weight: {:?}", e))?;
    named_tensors.push(NamedTensor {
        name: "output_norm.weight".to_string(),
        dims: vec![config.n_embd as u64],
        data: norm,
    });

    named_tensors.push(NamedTensor {
        name: "output.weight".to_string(),
        dims: vec![config.n_embd as u64, config.vocab_size as u64],
        data: emb,
    });

    println!("  Extracted {} parameter tensors for GGUF serialization.", named_tensors.len());

    let file = File::create(&gguf_path)?;
    let mut w = BufWriter::new(file);
    let mut current_offset: u64 = 0;

    // Header
    w.write_all(GGUF_MAGIC)?;
    w.write_all(&GGUF_VERSION.to_le_bytes())?;

    let tensor_count = named_tensors.len() as u64;

    let tokens_clone = vocab_tokens.clone();
    let types_clone = token_types.clone();
    let merges_clone = vocab_merges.clone();

    let mut kv_pairs: Vec<(&str, u32, GgufKvWriter<'_>)> = vec![
        ("general.architecture", GGUF_TYPE_STRING, Box::new(|w| { Ok(write_gguf_string(w, "llama")? as u64) })),        ("general.name", GGUF_TYPE_STRING, Box::new(move |w| { Ok(write_gguf_string(w, model_name)? as u64) })),
        ("llama.context_length", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&(config.n_ctx as u32).to_le_bytes())?; Ok(4) })),
        ("llama.embedding_length", GGUF_TYPE_UINT32, Box::new(|w| { w.write_all(&(config.n_embd as u32).to_le_bytes())?; Ok(4) })),
        ("llama.block_count", GGUF_TYPE_UINT32, Box::new(|w| { w.write_all(&(config.n_layer as u32).to_le_bytes())?; Ok(4) })),
        ("llama.feed_forward_length", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&(hidden_dim as u32).to_le_bytes())?; Ok(4) })),
        ("llama.attention.head_count", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&(config.n_head as u32).to_le_bytes())?; Ok(4) })),
        ("llama.attention.head_count_kv", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&(config.get_n_kv_head() as u32).to_le_bytes())?; Ok(4) })),
        ("llama.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, Box::new(move |w| { w.write_all(&config.rms_norm_eps.to_le_bytes())?; Ok(4) })),
        ("llama.rope.freq_base", GGUF_TYPE_FLOAT32, Box::new(move |w| { w.write_all(&config.rope_base.to_le_bytes())?; Ok(4) })),
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

    // RoPE context-extension metadata: only written when scaling is active, so
    // unscaled exports keep the exact same key set as before.
    if config.rope_kind() != crate::config::RopeScalingKind::None {
        let kind = config.rope_kind().as_str().to_string();
        let factor = config.rope_factor;
        let original_ctx = config.n_ctx as u32;
        kv_pairs.push((
            "llama.rope.scaling.type",
            GGUF_TYPE_STRING,
            Box::new(move |w| Ok(write_gguf_string(w, &kind)? as u64)),
        ));
        kv_pairs.push((
            "llama.rope.scaling.factor",
            GGUF_TYPE_FLOAT32,
            Box::new(move |w| {
                w.write_all(&factor.to_le_bytes())?;
                Ok(4)
            }),
        ));
        kv_pairs.push((
            "llama.rope.scaling.original_context_length",
            GGUF_TYPE_UINT32,
            Box::new(move |w| {
                w.write_all(&original_ctx.to_le_bytes())?;
                Ok(4)
            }),
        ));
    }

    // Sliding-window attention metadata (llama.cpp reads it for supported
    // architectures; harmless for runtimes that ignore it).
    if let Some(window) = config.sliding_window {
        kv_pairs.push((
            "llama.attention.sliding_window",
            GGUF_TYPE_UINT32,
            Box::new(move |w| {
                w.write_all(&(window as u32).to_le_bytes())?;
                Ok(4)
            }),
        ));
    }

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
        let align_offset = (ALIGNMENT - (data_offset_counter % ALIGNMENT)) % ALIGNMENT;
        data_offset_counter += align_offset;
        tensor_data_offsets.push(data_offset_counter);

        let element_count: u64 = tensor.dims.iter().product();
        let bytes_len = element_count * 4;
        data_offset_counter += bytes_len;

        current_offset += write_gguf_string(&mut w, &tensor.name)? as u64;
        w.write_all(&(tensor.dims.len() as u32).to_le_bytes())?;
        current_offset += 4;

        for d in &tensor.dims {
            w.write_all(&d.to_le_bytes())?;
            current_offset += 8;
        }

        w.write_all(&GGML_TYPE_F32.to_le_bytes())?;
        w.write_all(&tensor_data_offsets.last().unwrap().to_le_bytes())?;
        current_offset += 4 + 8;
    }

    // Alignment padding before tensor binary data
    let current_pos = current_offset;
    let padding_bytes = (ALIGNMENT - (current_pos % ALIGNMENT)) % ALIGNMENT;
    if padding_bytes > 0 {
        let zeroes = vec![0u8; padding_bytes as usize];
        w.write_all(&zeroes)?;
    }

    // Write binary tensor data
    let tensor_total = named_tensors.len() as u64;
    for (i, tensor) in named_tensors.iter().enumerate() {
        if cancelled() {
            drop(w);
            let _ = std::fs::remove_file(&gguf_path);
            return Err("cancelled by user".into());
        }
        let current_file_pos = w.stream_position()?;
        let pad = (ALIGNMENT - (current_file_pos % ALIGNMENT)) % ALIGNMENT;
        if pad > 0 {
            let zeroes = vec![0u8; pad as usize];
            w.write_all(&zeroes)?;
        }

        for &val in &tensor.data {
            w.write_all(&val.to_le_bytes())?;
        }
        progress("writing tensors", (i + 1) as u64, tensor_total, "tensors");
    }

    w.flush()?;

    let metadata = std::fs::metadata(&gguf_path)?;
    let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
    println!("GGUF export finished successfully. Binary size: {} MB", size_mb);

    Ok(gguf_path)
}
