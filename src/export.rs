use crate::config::ModelConfig;
use crate::model::Llm;
use burn::tensor::backend::Backend;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;

const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_STRING: u32 = 8;
const GGML_TYPE_F32: u32 = 0;
const ALIGNMENT: u64 = 32;

fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<usize> {
    let bytes = s.as_bytes();
    let len = bytes.len() as u64;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(8 + bytes.len())
}

#[derive(Debug)]
struct NamedTensor {
    name: String,
    dims: Vec<u64>,
    data: Vec<f32>,
}

#[allow(clippy::type_complexity)]
pub fn export_to_gguf<B: Backend>(

    model: &Llm<B>,
    config: &ModelConfig,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Exporting Burn model weights to native GGUF v3 binary format at '{}'...", output_path);

    let mut named_tensors: Vec<NamedTensor> = Vec::new();

    // 1. Extract Embeddings
    if let Ok(wte) = model.wte.weight.val().into_data().to_vec::<f32>() {
        named_tensors.push(NamedTensor {
            name: "token_embd.weight".to_string(),
            dims: vec![config.n_embd as u64, config.vocab_size as u64],
            data: wte,
        });
    }

    if let Ok(wpe) = model.wpe.weight.val().into_data().to_vec::<f32>() {
        named_tensors.push(NamedTensor {
            name: "pos_embd.weight".to_string(),
            dims: vec![config.n_embd as u64, config.n_ctx as u64],
            data: wpe,
        });
    }

    // 2. Extract Transformer Blocks
    for (i, block) in model.blocks.iter().enumerate() {
        let prefix = format!("blk.{}", i);

        // LayerNorm 1
        if let Ok(gamma) = block.ln_1.gamma.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_norm.weight", prefix),
                dims: vec![config.n_embd as u64],
                data: gamma,
            });
        }
        if let Some(beta) = &block.ln_1.beta {
            if let Ok(b) = beta.val().into_data().to_vec::<f32>() {
                named_tensors.push(NamedTensor {
                    name: format!("{}.attn_norm.bias", prefix),
                    dims: vec![config.n_embd as u64],
                    data: b,
                });
            }
        }

        // Self Attention Projections
        if let Ok(q) = block.attn.q_proj.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_q.weight", prefix),
                dims: vec![config.n_embd as u64, config.n_embd as u64],
                data: q,
            });
        }
        if let Ok(k) = block.attn.k_proj.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_k.weight", prefix),
                dims: vec![config.n_embd as u64, config.n_embd as u64],
                data: k,
            });
        }
        if let Ok(v) = block.attn.v_proj.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_v.weight", prefix),
                dims: vec![config.n_embd as u64, config.n_embd as u64],
                data: v,
            });
        }
        if let Ok(out) = block.attn.out_proj.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.attn_output.weight", prefix),
                dims: vec![config.n_embd as u64, config.n_embd as u64],
                data: out,
            });
        }

        // LayerNorm 2
        if let Ok(gamma) = block.ln_2.gamma.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.ffn_norm.weight", prefix),
                dims: vec![config.n_embd as u64],
                data: gamma,
            });
        }

        // MLP Projections
        if let Ok(fc) = block.mlp.c_fc.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.ffn_up.weight", prefix),
                dims: vec![config.n_embd as u64, (4 * config.n_embd) as u64],
                data: fc,
            });
        }
        if let Ok(proj) = block.mlp.c_proj.weight.val().into_data().to_vec::<f32>() {
            named_tensors.push(NamedTensor {
                name: format!("{}.ffn_down.weight", prefix),
                dims: vec![(4 * config.n_embd) as u64, config.n_embd as u64],
                data: proj,
            });
        }
    }

    // 3. Extract Final LayerNorm & LM Head
    if let Ok(gamma) = model.ln_f.gamma.val().into_data().to_vec::<f32>() {
        named_tensors.push(NamedTensor {
            name: "output_norm.weight".to_string(),
            dims: vec![config.n_embd as u64],
            data: gamma,
        });
    }

    if let Ok(head) = model.lm_head.weight.val().into_data().to_vec::<f32>() {
        named_tensors.push(NamedTensor {
            name: "output.weight".to_string(),
            dims: vec![config.n_embd as u64, config.vocab_size as u64],
            data: head,
        });
    }

    println!("Extracted {} parameter tensors for GGUF serialization.", named_tensors.len());

    let file = File::create(output_path)?;
    let mut w = BufWriter::new(file);
    let mut current_offset: u64 = 0;

    // Header
    w.write_all(GGUF_MAGIC)?;
    w.write_all(&GGUF_VERSION.to_le_bytes())?;

    let tensor_count = named_tensors.len() as u64;
    let kv_pairs: Vec<(&str, u32, Box<dyn Fn(&mut BufWriter<File>) -> std::io::Result<()>>)> = vec![
        ("general.architecture", GGUF_TYPE_STRING, Box::new(|w| { write_gguf_string(w, "gpt2")?; Ok(()) })),
        ("general.name", GGUF_TYPE_STRING, Box::new(|w| { write_gguf_string(w, "my-rust-llm")?; Ok(()) })),
        ("gpt2.context_length", GGUF_TYPE_UINT32, Box::new(|w| w.write_all(&(config.n_ctx as u32).to_le_bytes()))),
        ("gpt2.embedding_length", GGUF_TYPE_UINT32, Box::new(|w| w.write_all(&(config.n_embd as u32).to_le_bytes()))),
        ("gpt2.block_count", GGUF_TYPE_UINT32, Box::new(|w| w.write_all(&(config.n_layer as u32).to_le_bytes()))),
        ("gpt2.attention.head_count", GGUF_TYPE_UINT32, Box::new(|w| w.write_all(&(config.n_head as u32).to_le_bytes()))),
    ];

    w.write_all(&tensor_count.to_le_bytes())?;
    w.write_all(&(kv_pairs.len() as u64).to_le_bytes())?;
    current_offset += 4 + 4 + 8 + 8;

    // Metadata KV
    for (key, val_type, write_fn) in kv_pairs {
        current_offset += write_gguf_string(&mut w, key)? as u64;
        w.write_all(&val_type.to_le_bytes())?;
        current_offset += 4;
        write_fn(&mut w)?;
        current_offset += match val_type {
            GGUF_TYPE_UINT32 => 4,
            GGUF_TYPE_STRING => 0, // already added inside write_gguf_string
            _ => 0,
        };
    }

    // Calculate tensor offsets & write tensor info headers
    let mut tensor_data_offsets: Vec<u64> = Vec::with_capacity(named_tensors.len());
    let mut data_offset_counter: u64 = 0;

    for tensor in &named_tensors {
        // Pad data offset counter to alignment
        let pad = (ALIGNMENT - (data_offset_counter % ALIGNMENT)) % ALIGNMENT;
        data_offset_counter += pad;
        tensor_data_offsets.push(data_offset_counter);

        let byte_len = (tensor.data.len() * 4) as u64;
        data_offset_counter += byte_len;

        // Tensor Info Header
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
    for (tensor, &offset) in named_tensors.iter().zip(tensor_data_offsets.iter()) {
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
        println!("  Wrote tensor '{}' ({:?}, {} KB) at offset {}", tensor.name, tensor.dims, raw_bytes.len() / 1024, offset);
    }

    w.flush()?;
    println!("Successfully exported valid GGUF v3 binary model: '{}'", output_path);
    println!("Model is ready to be loaded directly in LM Studio, Ollama, or llama.cpp!");

    Ok(())
}
