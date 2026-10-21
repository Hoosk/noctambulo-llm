//! Import a GGUF file produced by `export.rs` back into the native format
//! (config.json + model_weights.bpk + custom_tokenizer.json).
//!
//! This recovers models that only survive as GGUF exports: the exporter maps
//! the Burn module layout onto llama.cpp's `llama` architecture (transposing
//! linear weights into GGML's `[out][in]` row-major layout and applying the
//! GPT-NeoX RoPE row permutation to Q/K), so importing inverts exactly those
//! transforms. Only f32/f16 tensors are supported (the exporter always writes
//! f32); quantized GGUFs are rejected with a clear error.

use crate::config::{ModelConfig, RopeScalingKind};
use crate::export::{transpose_2d, unpermute_rope_bias, unpermute_rope_weight};
use crate::model::Llm;
use crate::storage;
use burn::module::{Param, ParamId};
use burn::tensor::{Device, Tensor, TensorData};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    String(String),
    Array(Vec<GgufValue>),
}

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub dims: Vec<u64>,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub struct GgufFile {
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: BTreeMap<String, GgufTensor>,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| "GGUF parse overflow".to_string())?;
        if end > self.data.len() {
            return Err(format!(
                "GGUF truncated: need {n} bytes at offset {} but file has {}",
                self.pos,
                self.data.len()
            ));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i8(&mut self) -> Result<i8, String> {
        Ok(self.u8()? as i8)
    }

    fn i16(&mut self) -> Result<i16, String> {
        Ok(self.u16()? as i16)
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(self.u32()? as i32)
    }

    fn i64(&mut self) -> Result<i64, String> {
        Ok(self.u64()? as i64)
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| format!("GGUF string is not UTF-8: {e}"))
    }

    fn value(&mut self, kind: u32) -> Result<GgufValue, String> {
        Ok(match kind {
            GGUF_TYPE_UINT8 => GgufValue::U8(self.u8()?),
            GGUF_TYPE_INT8 => GgufValue::I8(self.i8()?),
            GGUF_TYPE_UINT16 => GgufValue::U16(self.u16()?),
            GGUF_TYPE_INT16 => GgufValue::I16(self.i16()?),
            GGUF_TYPE_UINT32 => GgufValue::U32(self.u32()?),
            GGUF_TYPE_INT32 => GgufValue::I32(self.i32()?),
            GGUF_TYPE_FLOAT32 => GgufValue::F32(self.f32()?),
            GGUF_TYPE_BOOL => GgufValue::Bool(self.u8()? != 0),
            GGUF_TYPE_STRING => GgufValue::String(self.string()?),
            GGUF_TYPE_UINT64 => GgufValue::U64(self.u64()?),
            GGUF_TYPE_INT64 => GgufValue::I64(self.i64()?),
            GGUF_TYPE_FLOAT64 => GgufValue::F64(self.f64()?),
            GGUF_TYPE_ARRAY => {
                let element = self.u32()?;
                let count = self.u64()? as usize;
                if element == GGUF_TYPE_ARRAY {
                    return Err("nested GGUF arrays are not supported".to_string());
                }
                let mut items = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    items.push(self.value(element)?);
                }
                GgufValue::Array(items)
            }
            other => return Err(format!("unknown GGUF metadata type {other}")),
        })
    }
}

/// Parse a GGUF v3 file into metadata and f32 tensors.
pub fn parse_gguf(path: &Path) -> Result<GgufFile, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read '{}': {e}", path.display()))?;
    let mut reader = Reader::new(&bytes);
    if reader.take(4)? != GGUF_MAGIC {
        return Err(format!("'{}' is not a GGUF file", path.display()));
    }
    let version = reader.u32()?;
    if version != 3 {
        return Err(format!(
            "unsupported GGUF version {version} (only v3 exports are supported)"
        ));
    }
    let tensor_count = reader.u64()? as usize;
    let kv_count = reader.u64()? as usize;

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = reader.string()?;
        let kind = reader.u32()?;
        metadata.insert(key, reader.value(kind)?);
    }

    struct TensorInfo {
        name: String,
        dims: Vec<u64>,
        kind: u32,
        offset: u64,
    }
    let mut infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let n_dims = reader.u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(reader.u64()?);
        }
        let kind = reader.u32()?;
        let offset = reader.u64()?;
        infos.push(TensorInfo {
            name,
            dims,
            kind,
            offset,
        });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(gguf_value_u64)
        .map(|v| v as usize)
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT)
        .max(1);
    let data_start = reader.pos.div_ceil(alignment) * alignment;

    let mut tensors = BTreeMap::new();
    for info in infos {
        let count: usize = info
            .dims
            .iter()
            .try_fold(1usize, |acc, d| acc.checked_mul(*d as usize))
            .ok_or_else(|| format!("tensor '{}' dimensions overflow", info.name))?;
        let bytes_per = match info.kind {
            GGML_TYPE_F32 => 4,
            GGML_TYPE_F16 => 2,
            other => {
                return Err(format!(
                    "tensor '{}' uses GGML type {other}; only f32/f16 GGUFs can be imported \
                     (re-export the model without quantization)",
                    info.name
                ));
            }
        };
        let start = data_start
            .checked_add(info.offset as usize)
            .ok_or_else(|| format!("tensor '{}' offset overflow", info.name))?;
        let end = start
            .checked_add(count * bytes_per)
            .ok_or_else(|| format!("tensor '{}' size overflow", info.name))?;
        if end > bytes.len() {
            return Err(format!(
                "tensor '{}' runs past the end of the file (offset {} + {} bytes)",
                info.name,
                info.offset,
                count * bytes_per
            ));
        }
        let data = match info.kind {
            GGML_TYPE_F32 => bytes[start..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            _ => bytes[start..end]
                .chunks_exact(2)
                .map(|c| burn::tensor::f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
                .collect(),
        };
        tensors.insert(info.name, GgufTensor { dims: info.dims, data });
    }

    Ok(GgufFile { metadata, tensors })
}

fn gguf_value_u64(value: &GgufValue) -> Option<u64> {
    match value {
        GgufValue::U8(v) => Some(*v as u64),
        GgufValue::I8(v) => Some(*v as u64),
        GgufValue::U16(v) => Some(*v as u64),
        GgufValue::I16(v) => Some(*v as u64),
        GgufValue::U32(v) => Some(*v as u64),
        GgufValue::I32(v) => Some(*v as u64),
        GgufValue::U64(v) => Some(*v),
        GgufValue::I64(v) => Some(*v as u64),
        _ => None,
    }
}

fn meta_u64(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Option<u64> {
    metadata.get(key).and_then(gguf_value_u64)
}

fn meta_f32(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Option<f32> {
    match metadata.get(key) {
        Some(GgufValue::F32(v)) => Some(*v),
        Some(GgufValue::F64(v)) => Some(*v as f32),
        _ => None,
    }
}

fn meta_string(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Option<String> {
    match metadata.get(key) {
        Some(GgufValue::String(v)) => Some(v.clone()),
        _ => None,
    }
}

fn meta_strings(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Option<Vec<String>> {
    match metadata.get(key) {
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|item| match item {
                GgufValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn meta_i32s(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Option<Vec<i32>> {
    match metadata.get(key) {
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|item| match item {
                GgufValue::I32(v) => Some(*v),
                GgufValue::U32(v) => Some(*v as i32),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn required(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Result<usize, String> {
    meta_u64(metadata, key)
        .map(|v| v as usize)
        .ok_or_else(|| format!("GGUF metadata is missing '{key}'"))
}

fn tensor<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a GgufTensor, String> {
    gguf.tensors
        .get(name)
        .ok_or_else(|| format!("GGUF is missing tensor '{name}'"))
}

fn expect_len(t: &GgufTensor, expected: usize, name: &str) -> Result<(), String> {
    if t.data.len() != expected {
        return Err(format!(
            "tensor '{name}' has {} values, expected {expected}",
            t.data.len()
        ));
    }
    Ok(())
}

/// GGUF dims use GGML's `[in, out]` order; a mismatch means the file follows a
/// different convention than this engine's exporter.
fn expect_dims(t: &GgufTensor, expected: &[usize], name: &str) -> Result<(), String> {
    let dims: Vec<usize> = t.dims.iter().map(|d| *d as usize).collect();
    if dims != expected {
        return Err(format!(
            "tensor '{name}' has GGUF dims {dims:?}, expected {expected:?}              (file was not written by this engine's exporter)"
        ));
    }
    Ok(())
}

pub(crate) fn param1(data: Vec<f32>, device: &Device) -> Param<Tensor<1>> {
    let len = data.len();
    let tensor = Tensor::<1>::from_data(TensorData::new(data, [len]), device);
    Param::initialized(ParamId::new(), tensor)
}

pub(crate) fn param2(data: Vec<f32>, dims: [usize; 2], device: &Device) -> Param<Tensor<2>> {
    let tensor = Tensor::<2>::from_data(TensorData::new(data, dims), device);
    Param::initialized(ParamId::new(), tensor)
}

/// Rebuild the custom BPE tokenizer JSON from GGUF `tokens`/`merges`.
fn tokenizer_json(tokens: &[String], merges: &[String], token_types: Option<&[i32]>) -> Value {
    let mut vocab = serde_json::Map::new();
    for (id, token) in tokens.iter().enumerate() {
        vocab.insert(token.clone(), json!(id));
    }
    let merge_values: Vec<Value> = merges
        .iter()
        .map(|merge| match merge.split_once(' ') {
            Some((a, b)) => json!([a, b]),
            None => json!(merge),
        })
        .collect();

    let special: Vec<usize> = match token_types {
        Some(types) => types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == 3)
            .map(|(i, _)| i)
            .collect(),
        None => (0..tokens.len().min(2))
            .filter(|&i| tokens[i].starts_with("<|") && tokens[i].ends_with("|>"))
            .collect(),
    };
    let added_tokens: Vec<Value> = special
        .iter()
        .map(|&id| {
            json!({
                "id": id,
                "content": tokens[id],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            })
        })
        .collect();

    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": { "type": "NFKC" },
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merge_values
        }
    })
}

/// Import a GGUF export into `output_dir` (config.json + model_weights.bpk +
/// custom_tokenizer.json). Refuses to overwrite an existing model unless
/// `force` is set. Returns the imported config.
pub fn import_gguf(
    gguf_path: &Path,
    output_dir: &Path,
    force: bool,
    device: &Device,
    tokenizer_override: Option<&Path>,
) -> Result<ModelConfig, String> {
    if !force
        && (crate::storage::model_weights_path(output_dir).exists()
            || crate::storage::legacy_weights_path(output_dir).is_some())
    {
        return Err(format!(
            "'{}' already contains a model (pass --force to overwrite)",
            output_dir.display()
        ));
    }

    let gguf = parse_gguf(gguf_path)?;
    let arch = meta_string(&gguf.metadata, "general.architecture").unwrap_or_default();
    // Third-party llama.cpp GGUFs use the architecture name as the metadata
    // prefix. `llama` and `qwen2` share the layout this engine implements;
    // anything else (qwen3 adds QK-norms, gemma uses GeGLU, ...) is rejected.
    if arch != "llama" && arch != "qwen2" {
        return Err(format!(
            "unsupported GGUF architecture '{arch}' (supported: llama, qwen2; models with \
             QK-norms, GeGLU or non-BPE tokenizers must be converted from safetensors instead)"
        ));
    }
    let key = |name: &str| format!("{arch}.{name}");
    if meta_string(&gguf.metadata, "tokenizer.ggml.model").as_deref() != Some("gpt2") {
        return Err(
            "unsupported GGUF tokenizer (only byte-level BPE / 'gpt2' vocabularies can be \
             imported; SentencePiece models need a safetensors conversion)"
                .into(),
        );
    }
    let n_ctx = required(&gguf.metadata, &key("context_length"))?;
    let n_embd = required(&gguf.metadata, &key("embedding_length"))?;
    let n_layer = required(&gguf.metadata, &key("block_count"))?;
    let n_head = required(&gguf.metadata, &key("attention.head_count"))?;
    let n_kv_head = meta_u64(&gguf.metadata, &key("attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(n_head);
    let hidden = required(&gguf.metadata, &key("feed_forward_length"))?;
    let rms_norm_eps = meta_f32(&gguf.metadata, &key("attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    let rope_base = meta_f32(&gguf.metadata, &key("rope.freq_base")).unwrap_or(10000.0);
    let tokens = meta_strings(&gguf.metadata, "tokenizer.ggml.tokens")
        .ok_or_else(|| "GGUF metadata is missing 'tokenizer.ggml.tokens'".to_string())?;
    let merges = meta_strings(&gguf.metadata, "tokenizer.ggml.merges").unwrap_or_default();
    let token_types = meta_i32s(&gguf.metadata, "tokenizer.ggml.token_type");
    let vocab_size = tokens.len();

    if n_embd == 0 || n_head == 0 || n_layer == 0 || n_ctx == 0 || vocab_size == 0 {
        return Err("GGUF metadata contains zero-sized dimensions".to_string());
    }
    if hidden == 0 {
        return Err("GGUF feed_forward_length is zero".to_string());
    }
    if !n_embd.is_multiple_of(n_head) {
        return Err(format!(
            "n_embd ({n_embd}) is not divisible by n_head ({n_head})"
        ));
    }
    if !n_head.is_multiple_of(n_kv_head) {
        return Err(format!(
            "n_head ({n_head}) is not divisible by n_kv_head ({n_kv_head})"
        ));
    }
    let head_dim = n_embd / n_head;
    if let Some(key_length) = meta_u64(&gguf.metadata, &key("attention.key_length")) {
        if key_length as usize != head_dim {
            return Err(format!(
                "GGUF attention.key_length {key_length} differs from n_embd/n_head ({head_dim}); \
                 explicit head dimensions are not supported"
            ));
        }
    }
    let kv_dim = n_kv_head * head_dim;

    let mut config = ModelConfig::new(vocab_size, n_embd, n_head, Some(n_kv_head), n_layer, n_ctx, 0.1, 0.0);
    config.ffn_hidden = Some(hidden);
    config.rms_norm_eps = rms_norm_eps;
    config.rope_base = rope_base;
    config.import_source = Some("gguf".into());
    if let Some(kind) = meta_string(&gguf.metadata, &key("rope.scaling.type")) {
        let rope_kind = RopeScalingKind::parse(&kind)?;
        let factor = meta_f32(&gguf.metadata, &key("rope.scaling.factor")).unwrap_or(1.0);
        config.set_rope_scaling(rope_kind, factor);
    }
    if let Some(window) = meta_u64(&gguf.metadata, &key("attention.sliding_window")) {
        config.sliding_window = Some(window as usize);
    }

    println!(
        "Importing GGUF '{}' ({} layers, n_embd {}, vocab {}) -> '{}'",
        gguf_path.display(),
        n_layer,
        n_embd,
        vocab_size,
        output_dir.display()
    );

    let mut model: Llm = config.init_model(device);

    // Tied input embedding. `output.weight` duplicates it in the export, so it
    // is only used to sanity-check the file.
    let emb = tensor(&gguf, "token_embd.weight")?;
    expect_len(emb, vocab_size * n_embd, "token_embd.weight")?;
    expect_dims(emb, &[n_embd, vocab_size], "token_embd.weight")?;
    model.embed_tokens.weight = param2(emb.data.clone(), [vocab_size, n_embd], device);
    if let Some(out) = gguf.tensors.get("output.weight") {
        expect_len(out, vocab_size * n_embd, "output.weight")?;
        expect_dims(out, &[n_embd, vocab_size], "output.weight")?;
        let diff = emb
            .data
            .iter()
            .zip(out.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if diff > 1e-5 {
            println!(
                "  note: output.weight differs from token_embd.weight (max |diff| {diff:.3e}); \
                 the engine uses the tied embedding"
            );
        }
    }

    let norm = tensor(&gguf, "output_norm.weight")?;
    expect_len(norm, n_embd, "output_norm.weight")?;
    expect_dims(norm, &[n_embd], "output_norm.weight")?;
    model.norm.weight = param1(norm.data.clone(), device);

    // llama.cpp stores llama-arch Q/K with the GPT-NeoX row permutation
    // (NORM rope); qwen2 uses NEOX rope and stores the original layout.
    let permuted = arch == "llama";
    for i in 0..n_layer {
        let prefix = format!("blk.{i}");

        let attn_norm = tensor(&gguf, &format!("{prefix}.attn_norm.weight"))?;
        expect_len(attn_norm, n_embd, &format!("{prefix}.attn_norm.weight"))?;
        expect_dims(attn_norm, &[n_embd], &format!("{prefix}.attn_norm.weight"))?;
        let ffn_norm = tensor(&gguf, &format!("{prefix}.ffn_norm.weight"))?;
        expect_len(ffn_norm, n_embd, &format!("{prefix}.ffn_norm.weight"))?;
        expect_dims(ffn_norm, &[n_embd], &format!("{prefix}.ffn_norm.weight"))?;

        let q = tensor(&gguf, &format!("{prefix}.attn_q.weight"))?;
        expect_len(q, n_embd * n_embd, &format!("{prefix}.attn_q.weight"))?;
        expect_dims(q, &[n_embd, n_embd], &format!("{prefix}.attn_q.weight"))?;
        let q_data = if permuted {
            unpermute_rope_weight(&q.data, n_head, head_dim, n_embd)
        } else {
            q.data.clone()
        };
        let q = transpose_2d(&q_data, n_embd, n_embd);

        let k = tensor(&gguf, &format!("{prefix}.attn_k.weight"))?;
        expect_len(k, n_embd * kv_dim, &format!("{prefix}.attn_k.weight"))?;
        expect_dims(k, &[n_embd, kv_dim], &format!("{prefix}.attn_k.weight"))?;
        let k_data = if permuted {
            unpermute_rope_weight(&k.data, n_kv_head, head_dim, n_embd)
        } else {
            k.data.clone()
        };
        let k = transpose_2d(&k_data, kv_dim, n_embd);

        let v = tensor(&gguf, &format!("{prefix}.attn_v.weight"))?;
        expect_len(v, n_embd * kv_dim, &format!("{prefix}.attn_v.weight"))?;
        expect_dims(v, &[n_embd, kv_dim], &format!("{prefix}.attn_v.weight"))?;
        let v = transpose_2d(&v.data, kv_dim, n_embd);

        let o = tensor(&gguf, &format!("{prefix}.attn_output.weight"))?;
        expect_len(o, n_embd * n_embd, &format!("{prefix}.attn_output.weight"))?;
        expect_dims(o, &[n_embd, n_embd], &format!("{prefix}.attn_output.weight"))?;
        let o = transpose_2d(&o.data, n_embd, n_embd);

        let gate = tensor(&gguf, &format!("{prefix}.ffn_gate.weight"))?;
        expect_len(gate, n_embd * hidden, &format!("{prefix}.ffn_gate.weight"))?;
        expect_dims(gate, &[n_embd, hidden], &format!("{prefix}.ffn_gate.weight"))?;
        let gate = transpose_2d(&gate.data, hidden, n_embd);

        let up = tensor(&gguf, &format!("{prefix}.ffn_up.weight"))?;
        expect_len(up, n_embd * hidden, &format!("{prefix}.ffn_up.weight"))?;
        expect_dims(up, &[n_embd, hidden], &format!("{prefix}.ffn_up.weight"))?;
        let up = transpose_2d(&up.data, hidden, n_embd);

        let down = tensor(&gguf, &format!("{prefix}.ffn_down.weight"))?;
        expect_len(down, hidden * n_embd, &format!("{prefix}.ffn_down.weight"))?;
        expect_dims(down, &[hidden, n_embd], &format!("{prefix}.ffn_down.weight"))?;
        let down = transpose_2d(&down.data, n_embd, hidden);

        let q_bias = gguf.tensors.get(&format!("{prefix}.attn_q.bias")).map(|b| {
            if permuted {
                unpermute_rope_bias(&b.data, n_head, head_dim)
            } else {
                b.data.clone()
            }
        });
        let k_bias = gguf.tensors.get(&format!("{prefix}.attn_k.bias")).map(|b| {
            if permuted {
                unpermute_rope_bias(&b.data, n_kv_head, head_dim)
            } else {
                b.data.clone()
            }
        });
        let v_bias = gguf
            .tensors
            .get(&format!("{prefix}.attn_v.bias"))
            .map(|b| b.data.clone());

        let block = &mut model.blocks[i];
        block.input_layernorm.weight = param1(attn_norm.data.clone(), device);
        block.post_attention_layernorm.weight = param1(ffn_norm.data.clone(), device);
        block.attn.q_proj.weight = param2(q, [n_embd, n_embd], device);
        block.attn.q_proj.bias = q_bias.map(|b| param1(b, device));
        block.attn.out_proj.weight = param2(o, [n_embd, n_embd], device);
        block.mlp.gate_proj.weight = param2(gate, [n_embd, hidden], device);
        block.mlp.up_proj.weight = param2(up, [n_embd, hidden], device);
        block.mlp.down_proj.weight = param2(down, [hidden, n_embd], device);

        let kv = &mut model.kv[i];
        kv.k_proj.weight = param2(k, [n_embd, kv_dim], device);
        kv.k_proj.bias = k_bias.map(|b| param1(b, device));
        kv.v_proj.weight = param2(v, [n_embd, kv_dim], device);
        kv.v_proj.bias = v_bias.map(|b| param1(b, device));
    }

    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("failed to create '{}': {e}", output_dir.display()))?;

    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("failed to serialize config: {e}"))?;
    std::fs::write(output_dir.join("config.json"), config_json)
        .map_err(|e| format!("failed to write config.json: {e}"))?;

    storage::save_module(&model, &storage::model_weights_path(output_dir))?;

    let tokenizer_path = output_dir.join("custom_tokenizer.json");
    if let Some(source) = tokenizer_override {
        std::fs::copy(source, &tokenizer_path)
            .map_err(|e| format!("failed to copy the tokenizer override: {e}"))?;
    } else {
        let pre = meta_string(&gguf.metadata, "tokenizer.ggml.pre").unwrap_or_default();
        if !pre.is_empty() && pre != "gpt-2" && pre != "gpt2" {
            println!(
                "  warning: the GGUF declares the '{pre}' pre-tokenizer; the rebuilt byte-level \
                 BPE may tokenize differently. Pass --tokenizer <tokenizer.json> for exact \
                 tokenization."
            );
        }
        let tokenizer = tokenizer_json(&tokens, &merges, token_types.as_deref());
        std::fs::write(
            &tokenizer_path,
            serde_json::to_string(&tokenizer).map_err(|e| format!("tokenizer JSON: {e}"))?,
        )
        .map_err(|e| format!("failed to write custom_tokenizer.json: {e}"))?;
    }
    // Validate the rebuilt tokenizer before declaring success.
    crate::data::LlmTokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("rebuilt tokenizer is invalid: {e}"))?;

    println!(
        "Imported '{}' -> '{}' (config.json, model_weights.bpk, custom_tokenizer.json)",
        gguf_path.display(),
        output_dir.display()
    );
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{permute_rope_bias, permute_rope_weight};

    #[test]
    fn test_transpose_roundtrip() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = transpose_2d(&data, 3, 4);
        assert_eq!(t, vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]);
        assert_eq!(transpose_2d(&t, 4, 3), data);
    }

    #[test]
    fn test_rope_permute_inverse() {
        let n_heads = 2;
        let head_dim = 4;
        let cols = 3;
        let data: Vec<f32> = (0..(n_heads * head_dim * cols))
            .map(|i| i as f32)
            .collect();
        let permuted = permute_rope_weight(&data, n_heads, head_dim, cols);
        assert_eq!(unpermute_rope_weight(&permuted, n_heads, head_dim, cols), data);

        let bias: Vec<f32> = (0..(n_heads * head_dim)).map(|i| i as f32).collect();
        let permuted = permute_rope_bias(&bias, n_heads, head_dim);
        assert_eq!(unpermute_rope_bias(&permuted, n_heads, head_dim), bias);
    }

    #[test]
    fn test_parse_minimal_gguf() {
        // Build a v3 file with one metadata string and one f32 tensor.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensors
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata
        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        let value = b"llama";
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);
        let name = b"w";
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // offset
        while bytes.len() % GGUF_DEFAULT_ALIGNMENT != 0 {
            bytes.push(0);
        }
        for v in 0..6 {
            bytes.extend_from_slice(&(v as f32).to_le_bytes());
        }

        let dir = std::env::temp_dir().join(format!("noctambulo_gguf_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.gguf");
        std::fs::write(&path, &bytes).unwrap();

        let gguf = parse_gguf(&path).unwrap();
        assert_eq!(
            meta_string(&gguf.metadata, "general.architecture").as_deref(),
            Some("llama")
        );
        let t = &gguf.tensors["w"];
        assert_eq!(t.dims, vec![3, 2]);
        assert_eq!(t.data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod roundtrip_tests {
    use super::*;
    use crate::export::permute_rope_weight;
        
    fn write_tokenizer(path: &Path, vocab: usize) {
        let mut map = serde_json::Map::new();
        for i in 0..vocab {
            map.insert(format!("t{i}"), json!(i));
        }
        let tok = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": { "type": "NFKC" },
            "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
            "post_processor": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
            "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
            "model": { "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                       "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
                       "ignore_merges": false, "vocab": map, "merges": [] }
        });
        std::fs::write(path, serde_json::to_string(&tok).unwrap()).unwrap();
    }

    #[test]
    fn test_export_import_roundtrip_preserves_weights() {
        let dir = std::env::temp_dir().join(format!("noctambulo_gguf_rt_{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        let tok_path = dir.join("tok.json");
        write_tokenizer(&tok_path, 32);

        let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        let config = ModelConfig::new(32, 16, 2, Some(1), 1, 8, 0.0, 0.0);
        let model: Llm = config.init_model(&device);
        let gguf = dir.join("rt.gguf");
        crate::export::export_to_gguf(
            &model,
            &config,
            tok_path.to_str().unwrap(),
            "rt",
            dir.to_str().unwrap(),
            false,
            &|_, _, _, _| {},
            &|| false,
        )
        .unwrap();
        assert!(gguf.exists());

        // Golden layout: the exported GGUF must follow the Hugging Face flat
        // convention ([out][in] row-major with the llama RoPE permutation), not
        // merely invert our own importer. This pins the non-square tensors,
        // where a swapped transpose still round-trips but breaks third-party
        // compatibility.
        let parsed = parse_gguf(&gguf).unwrap();
        let n_kv_head = config.get_n_kv_head();
        let head_dim = config.n_embd / config.n_head;
        let kv_dim = n_kv_head * head_dim;
        let k_burn = model.kv[0].k_proj.weight.val().into_data().try_to_vec::<f32>().unwrap();
        let expected = transpose_2d(&k_burn, config.n_embd, kv_dim);
        let expected = permute_rope_weight(&expected, n_kv_head, head_dim, config.n_embd);
        let k_tensor = parsed.tensors.get("blk.0.attn_k.weight").expect("k tensor");
        assert_eq!(k_tensor.data.len(), expected.len());
        for (a, b) in k_tensor.data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "exported k layout mismatch: {a} vs {b}");
        }

        let out = dir.join("imported");
        import_gguf(&gguf, &out, true, &device, None).unwrap();

        let imported: Llm = storage::load_module(
            config.init_model(&device),
            &storage::model_weights_path(&out),
        )
        .unwrap();

        let before = model.embed_tokens.weight.val().into_data().try_to_vec::<f32>().unwrap();
        let after = imported.embed_tokens.weight.val().into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-6, "embed mismatch {a} vs {b}");
        }

        for i in 0..config.n_layer {
            for (name, a, b) in [
                ("q", model.blocks[i].attn.q_proj.weight.val(), imported.blocks[i].attn.q_proj.weight.val()),
                ("k", model.kv[i].k_proj.weight.val(), imported.kv[i].k_proj.weight.val()),
                ("v", model.kv[i].v_proj.weight.val(), imported.kv[i].v_proj.weight.val()),
                ("o", model.blocks[i].attn.out_proj.weight.val(), imported.blocks[i].attn.out_proj.weight.val()),
                ("gate", model.blocks[i].mlp.gate_proj.weight.val(), imported.blocks[i].mlp.gate_proj.weight.val()),
                ("down", model.blocks[i].mlp.down_proj.weight.val(), imported.blocks[i].mlp.down_proj.weight.val()),
            ] {
                let av = a.into_data().try_to_vec::<f32>().unwrap();
                let bv = b.into_data().try_to_vec::<f32>().unwrap();
                for (x, y) in av.iter().zip(bv.iter()) {
                    assert!((x - y).abs() < 1e-6, "{name} mismatch {x} vs {y}");
                }
            }
            let qb = model.blocks[i].attn.q_proj.bias.as_ref().unwrap().val().into_data().try_to_vec::<f32>().unwrap();
            let qb2 = imported.blocks[i].attn.q_proj.bias.as_ref().unwrap().val().into_data().try_to_vec::<f32>().unwrap();
            for (x, y) in qb.iter().zip(qb2.iter()) {
                assert!((x - y).abs() < 1e-6, "q bias mismatch {x} vs {y}");
            }
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
