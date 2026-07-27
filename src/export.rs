use crate::config::ModelConfig;
use crate::model::Llm;
use burn::tensor::backend::Backend;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;

const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_FLOAT32: u32 = 6;
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

fn permute_rope_bias(bias: &[f32], n_head: usize) -> Vec<f32> {
    let n_embd = bias.len();
    let head_size = n_embd / n_head;
    let half_dim = head_size / 2;
    let mut out = vec![0.0f32; n_embd];

    for r_head in 0..n_head {
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

pub struct NamedTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub data: Vec<f32>,
}

pub fn export_to_gguf<B: Backend>(
    model: &Llm<B>,
    config: &ModelConfig,
    tokenizer_path: &str,
    model_name: &str,
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    std::fs::create_dir_all(output_dir)?;
    let gguf_path = format!("{}/{}.gguf", output_dir, model_name);

    println!("Exporting Qwen2.5/Qwen3 model architecture to GGUF format at: '{}'...", gguf_path);

    let tokens = load_vocab_tokens(tokenizer_path, config.vocab_size);
    let eos_token_id = tokens
        .iter()
        .position(|t| t == "<|endoftext|>" || t == "</s>" || t == "<eos>" || t == "<|im_end|>")
        .unwrap_or(2) as u32;

    let bos_token_id = tokens
        .iter()
        .position(|t| t == "<|im_start|>" || t == "<s>" || t == "<bos>")
        .unwrap_or(1) as u32;

    let mut named_tensors: Vec<NamedTensor> = Vec::new();

    // 1. Token Embeddings
    let emb = model
        .embed_tokens
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("Failed to extract embed_tokens weight: {:?}", e))?;

    named_tensors.push(NamedTensor {
        name: "token_embd.weight".to_string(),
        dims: vec![config.n_embd as u64, config.vocab_size as u64],
        data: emb.clone(),
    });

    // 2. Transformer Blocks
    let raw_hidden = (8 * config.n_embd) / 3;
    let hidden_dim = raw_hidden.div_ceil(256) * 256;

    for (i, block) in model.blocks.iter().enumerate() {
        let prefix = format!("blk.{}", i);

        // Input RmsNorm
        let gamma = block.input_layernorm.weight.val().into_data().to_vec::<f32>()
            .map_err(|e| format!("Failed to extract input_layernorm weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.attn_norm.weight", prefix),
            dims: vec![config.n_embd as u64],
            data: gamma,
        });

        // Self Attention Projections (Q, K, V, O)
        let q = block.attn.q_proj.weight.val().into_data().to_vec::<f32>()
            .map_err(|e| format!("Failed to extract q_proj weight: {:?}", e))?;
        let q_trans = transpose_2d(&q, config.n_embd, config.n_embd);
        let q_permuted = permute_rope_weight(&q_trans, config.n_head, config.n_embd);
        named_tensors.push(NamedTensor {
            name: format!("{}.attn_q.weight", prefix),
            dims: vec![config.n_embd as u64, config.n_embd as u64],
            data: q_permuted,
        });

        if let Some(ref q_b_param) = block.attn.q_proj.bias {
            let q_b = q_b_param.val().into_data().to_vec::<f32>()
                .map_err(|e| format!("Failed to extract q_proj bias: {:?}", e))?;
            let q_b_permuted = permute_rope_bias(&q_b, config.n_head);
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_q.bias", prefix),
                dims: vec![config.n_embd as u64],
                data: q_b_permuted,
            });
        }

        let k = block.attn.k_proj.weight.val().into_data().to_vec::<f32>()
            .map_err(|e| format!("Failed to extract k_proj weight: {:?}", e))?;
        let k_trans = transpose_2d(&k, config.n_embd, config.n_embd);
        let k_permuted = permute_rope_weight(&k_trans, config.n_head, config.n_embd);
        named_tensors.push(NamedTensor {
            name: format!("{}.attn_k.weight", prefix),
            dims: vec![config.n_embd as u64, config.n_embd as u64],
            data: k_permuted,
        });

        if let Some(ref k_b_param) = block.attn.k_proj.bias {
            let k_b = k_b_param.val().into_data().to_vec::<f32>()
                .map_err(|e| format!("Failed to extract k_proj bias: {:?}", e))?;
            let k_b_permuted = permute_rope_bias(&k_b, config.n_head);
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_k.bias", prefix),
                dims: vec![config.n_embd as u64],
                data: k_b_permuted,
            });
        }

        let v = block.attn.v_proj.weight.val().into_data().to_vec::<f32>()
            .map_err(|e| format!("Failed to extract v_proj weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.attn_v.weight", prefix),
            dims: vec![config.n_embd as u64, config.n_embd as u64],
            data: transpose_2d(&v, config.n_embd, config.n_embd),
        });

        if let Some(ref v_b_param) = block.attn.v_proj.bias {
            let v_b = v_b_param.val().into_data().to_vec::<f32>()
                .map_err(|e| format!("Failed to extract v_proj bias: {:?}", e))?;
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_v.bias", prefix),
                dims: vec![config.n_embd as u64],
                data: v_b,
            });
        }

        let out = block.attn.out_proj.weight.val().into_data().to_vec::<f32>()
            .map_err(|e| format!("Failed to extract out_proj weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.attn_output.weight", prefix),
            dims: vec![config.n_embd as u64, config.n_embd as u64],
            data: transpose_2d(&out, config.n_embd, config.n_embd),
        });

        // Post-Attention RmsNorm
        let gamma_post = block.post_attention_layernorm.weight.val().into_data().to_vec::<f32>()
            .map_err(|e| format!("Failed to extract post_attention_layernorm weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.ffn_norm.weight", prefix),
            dims: vec![config.n_embd as u64],
            data: gamma_post,
        });

        // SwiGLU MLP Projections (Gate, Up, Down)
        let gate = block.mlp.gate_proj.weight.val().into_data().to_vec::<f32>()
            .map_err(|e| format!("Failed to extract gate_proj weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.ffn_gate.weight", prefix),
            dims: vec![config.n_embd as u64, hidden_dim as u64],
            data: transpose_2d(&gate, config.n_embd, hidden_dim),
        });

        let up = block.mlp.up_proj.weight.val().into_data().to_vec::<f32>()
            .map_err(|e| format!("Failed to extract up_proj weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.ffn_up.weight", prefix),
            dims: vec![config.n_embd as u64, hidden_dim as u64],
            data: transpose_2d(&up, config.n_embd, hidden_dim),
        });

        let down = block.mlp.down_proj.weight.val().into_data().to_vec::<f32>()
            .map_err(|e| format!("Failed to extract down_proj weight: {:?}", e))?;
        named_tensors.push(NamedTensor {
            name: format!("{}.ffn_down.weight", prefix),
            dims: vec![hidden_dim as u64, config.n_embd as u64],
            data: transpose_2d(&down, hidden_dim, config.n_embd),
        });
    }

    // 3. Final Norm & Tied Output Head
    let norm = model.norm.weight.val().into_data().to_vec::<f32>()
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

    // Header
    w.write_all(GGUF_MAGIC)?;
    w.write_all(&GGUF_VERSION.to_le_bytes())?;
    w.write_all(&(named_tensors.len() as u64).to_le_bytes())?;

    let head_size = (config.n_embd / config.n_head) as u32;

    let metadata_kv: Vec<(&str, u32, Box<dyn Fn(&mut BufWriter<File>) -> std::io::Result<usize>>)> = vec![
        ("general.architecture", GGUF_TYPE_STRING, Box::new(|w| write_gguf_string(w, "llama"))),
        ("general.name", GGUF_TYPE_STRING, Box::new(|w| write_gguf_string(w, model_name))),
        ("general.alignment", GGUF_TYPE_UINT32, Box::new(|w| { w.write_all(&32u32.to_le_bytes())?; Ok(4) })),
        ("llama.context_length", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&((config.n_ctx * 2) as u32).to_le_bytes())?; Ok(4) })),
        ("llama.embedding_length", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&(config.n_embd as u32).to_le_bytes())?; Ok(4) })),
        ("llama.block_count", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&(config.n_layer as u32).to_le_bytes())?; Ok(4) })),
        ("llama.feed_forward_length", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&(hidden_dim as u32).to_le_bytes())?; Ok(4) })),
        ("llama.attention.head_count", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&(config.n_head as u32).to_le_bytes())?; Ok(4) })),
        ("llama.attention.head_count_kv", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&(config.get_n_kv_head() as u32).to_le_bytes())?; Ok(4) })),
        ("llama.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, Box::new(|w| { w.write_all(&1e-6f32.to_le_bytes())?; Ok(4) })),
        ("llama.rope.dimension_count", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&head_size.to_le_bytes())?; Ok(4) })),
        ("llama.rope.freq_base", GGUF_TYPE_FLOAT32, Box::new(|w| { w.write_all(&10000.0f32.to_le_bytes())?; Ok(4) })),
        ("tokenizer.ggml.model", GGUF_TYPE_STRING, Box::new(|w| write_gguf_string(w, "llama"))),
        ("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, Box::new(move |w| {
            let mut bytes = 0;
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            w.write_all(&(tokens.len() as u64).to_le_bytes())?;
            bytes += 12;
            for t in &tokens {
                bytes += write_gguf_string(w, t)?;
            }
            Ok(bytes)
        })),
        ("tokenizer.ggml.bos_token_id", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&bos_token_id.to_le_bytes())?; Ok(4) })),
        ("tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32, Box::new(move |w| { w.write_all(&eos_token_id.to_le_bytes())?; Ok(4) })),
    ];

    w.write_all(&(metadata_kv.len() as u64).to_le_bytes())?;

    for (key, val_type, write_fn) in &metadata_kv {
        write_gguf_string(&mut w, key)?;
        w.write_all(&val_type.to_le_bytes())?;
        write_fn(&mut w)?;
    }

    struct LocalTensorHeader {
        _offset: u64,
        data_len: u64,
        tensor_idx: usize,
    }

    let mut tensor_headers = Vec::new();
    let mut current_offset = 0u64;

    for (idx, tensor) in named_tensors.iter().enumerate() {
        write_gguf_string(&mut w, &tensor.name)?;
        w.write_all(&(tensor.dims.len() as u32).to_le_bytes())?;
        for d in &tensor.dims {
            w.write_all(&d.to_le_bytes())?;
        }
        w.write_all(&GGML_TYPE_F32.to_le_bytes())?;
        w.write_all(&current_offset.to_le_bytes())?;

        let tensor_bytes = (tensor.data.len() * std::mem::size_of::<f32>()) as u64;
        tensor_headers.push(LocalTensorHeader {
            _offset: current_offset,
            data_len: tensor_bytes,
            tensor_idx: idx,
        });

        current_offset += tensor_bytes;
        let pad = (ALIGNMENT - (tensor_bytes % ALIGNMENT)) % ALIGNMENT;
        current_offset += pad;
    }

    let current_pos = w.stream_position()?;
    let pad = (ALIGNMENT - (current_pos % ALIGNMENT)) % ALIGNMENT;
    if pad > 0 {
        let zero_buf = vec![0u8; pad as usize];
        w.write_all(&zero_buf)?;
    }

    for h in &tensor_headers {
        let tensor = &named_tensors[h.tensor_idx];
        for &val in &tensor.data {
            w.write_all(&val.to_le_bytes())?;
        }
        let pad = (ALIGNMENT - (h.data_len % ALIGNMENT)) % ALIGNMENT;
        if pad > 0 {
            let zero_buf = vec![0u8; pad as usize];
            w.write_all(&zero_buf)?;
        }
    }

    w.flush()?;
    println!("GGUF export finished successfully. Binary size: {} MB", (w.stream_position()? as f64) / (1024.0 * 1024.0));

    Ok(gguf_path)
}
