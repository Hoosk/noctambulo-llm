//! Convert a Hugging Face model (`config.json` + `tokenizer.json` +
//! `model.safetensors`) into the native format (`config.json` +
//! `model_weights.bin` + `custom_tokenizer.json`).
//!
//! Supported families: Llama / Qwen2 / Mistral style decoder-only models with
//! SwiGLU, RMSNorm, GQA, tied word embeddings and a byte-level BPE tokenizer.
//! QK-norm architectures (Qwen3), untied embeddings, SentencePiece tokenizers
//! and quantized (GPTQ/AWQ) safetensors are rejected with a clear error.
//!
//! Weights are read exactly (F32/F16/BF16) and stored in f32; no GGUF
//! quantization or llama.cpp RoPE permutation is involved, because Hugging
//! Face already stores the rotate-half layout this engine computes.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use serde::Deserialize;

use crate::config::{ModelConfig, RopeScalingKind};
use crate::export::transpose_2d;
use crate::import::{param1, param2};
use crate::model::Llm;

/// Where the Hugging Face model comes from: a Hub repo id or a local directory
/// containing `config.json`, `tokenizer.json` and the safetensors shards.
#[derive(Clone, Debug)]
pub enum HfSource {
    Repo(String),
    Dir(PathBuf),
}

impl HfSource {
    pub fn label(&self) -> String {
        match self {
            HfSource::Repo(repo) => repo.clone(),
            HfSource::Dir(dir) => dir.to_string_lossy().to_string(),
        }
    }
}

/// Files resolved for a conversion.
struct HfFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    shards: Vec<PathBuf>,
}

/// One safetensors tensor entry from the header.
#[derive(Deserialize, Debug, Clone)]
struct TensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct Shard {
    path: PathBuf,
    entries: BTreeMap<String, TensorEntry>,
    data_start: u64,
}

/// All shards of a model, with a tensor-name to shard map.
struct ShardSet {
    shards: Vec<Shard>,
    locations: HashMap<String, usize>,
}

impl ShardSet {
    fn open(paths: &[PathBuf]) -> Result<Self, String> {
        let mut shards = Vec::with_capacity(paths.len());
        let mut locations = HashMap::new();
        for path in paths {
            let mut file = File::open(path)
                .map_err(|e| format!("cannot open '{}': {e}", path.display()))?;
            let mut len_bytes = [0u8; 8];
            file.read_exact(&mut len_bytes)
                .map_err(|e| format!("'{}' is not a safetensors file: {e}", path.display()))?;
            let header_len = u64::from_le_bytes(len_bytes);
            if header_len == 0 || header_len > 512 * 1024 * 1024 {
                return Err(format!(
                    "'{}' has an implausible safetensors header ({header_len} bytes)",
                    path.display()
                ));
            }
            let mut header = vec![0u8; header_len as usize];
            file.read_exact(&mut header)
                .map_err(|e| format!("truncated safetensors header in '{}': {e}", path.display()))?;
            let parsed: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)
                .map_err(|e| format!("invalid safetensors header in '{}': {e}", path.display()))?;
            let mut entries = BTreeMap::new();
            for (name, value) in parsed {
                if name == "__metadata__" {
                    continue;
                }
                let entry: TensorEntry = serde_json::from_value(value).map_err(|e| {
                    format!("invalid tensor entry '{name}' in '{}': {e}", path.display())
                })?;
                locations.insert(name.clone(), shards.len());
                entries.insert(name, entry);
            }
            shards.push(Shard {
                path: path.clone(),
                entries,
                data_start: 8 + header_len,
            });
        }
        Ok(Self { shards, locations })
    }

    fn contains(&self, name: &str) -> bool {
        self.locations.contains_key(name)
    }

    /// Reads a tensor as f32 (F32/F16/BF16). Returns `(shape, data)`.
    fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let shard_idx = *self
            .locations
            .get(name)
            .ok_or_else(|| format!("safetensors file is missing tensor '{name}'"))?;
        let shard = &self.shards[shard_idx];
        let entry = &shard.entries[name];
        let n: usize = entry.shape.iter().product();
        let expected = match entry.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => {
                return Err(format!(
                    "tensor '{name}' uses dtype {other}; only F32/F16/BF16 safetensors can be \
                     converted (quantized GPTQ/AWQ models are not supported)"
                ))
            }
        };
        let start = shard.data_start + entry.data_offsets[0];
        let end = shard.data_start + entry.data_offsets[1];
        let byte_len = (n * expected) as u64;
        if end < start || end - start != byte_len {
            return Err(format!(
                "tensor '{name}' has inconsistent data offsets ({} bytes for {n} values of {})",
                end.saturating_sub(start),
                entry.dtype
            ));
        }
        let mut file = File::open(&shard.path)
            .map_err(|e| format!("cannot reopen '{}': {e}", shard.path.display()))?;
        file.seek(SeekFrom::Start(start))
            .map_err(|e| format!("seek failed: {e}"))?;
        let mut bytes = vec![0u8; byte_len as usize];
        file.read_exact(&mut bytes)
            .map_err(|e| format!("truncated tensor '{name}': {e}"))?;

        let data: Vec<f32> = match entry.dtype.as_str() {
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "F16" => bytes
                .chunks_exact(2)
                .map(|c| {
                    burn::tensor::f16::from_bits(u16::from_le_bytes(c.try_into().unwrap()))
                        .to_f32()
                })
                .collect(),
            "BF16" => bytes
                .chunks_exact(2)
                .map(|c| {
                    f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16)
                })
                .collect(),
            _ => unreachable!("dtype checked above"),
        };
        Ok((entry.shape.clone(), data))
    }
}

/// Subset of a Hugging Face `config.json` needed to rebuild the architecture.
#[derive(Deserialize, Debug)]
struct HfConfig {
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    max_position_embeddings: usize,
    #[serde(default)]
    rms_norm_eps: Option<f32>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    head_dim: Option<usize>,
}

/// Maps a HF `rope_scaling` block onto the engine strategies.
fn rope_scaling_from_hf(value: &serde_json::Value) -> Result<(RopeScalingKind, f32), String> {
    let kind = value
        .get("type")
        .or_else(|| value.get("rope_type"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| "rope_scaling is missing its 'type'".to_string())?;
    let factor = value
        .get("factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    match kind {
        "linear" => Ok((RopeScalingKind::Linear, factor)),
        "dynamic" => Ok((RopeScalingKind::Ntk, factor)),
        "yarn" => Ok((RopeScalingKind::Yarn, factor)),
        other => Err(format!(
            "unsupported rope_scaling type '{other}' (supported: linear, dynamic, yarn)"
        )),
    }
}

/// Resolves `config.json`, `tokenizer.json` and the safetensors shards from the
/// source, downloading from the Hub when needed.
fn resolve_files(
    source: &HfSource,
    progress: &mut dyn FnMut(&str, u64, u64, &str),
) -> Result<HfFiles, String> {
    match source {
        HfSource::Dir(dir) => {
            let config = dir.join("config.json");
            let tokenizer = dir.join("tokenizer.json");
            if !config.is_file() {
                return Err(format!("'{}' has no config.json", dir.display()));
            }
            if !tokenizer.is_file() {
                return Err(format!(
                    "'{}' has no tokenizer.json (SentencePiece-only repos are not supported)",
                    dir.display()
                ));
            }
            let shards = shards_in_dir(dir)?;
            Ok(HfFiles { config, tokenizer, shards })
        }
        HfSource::Repo(repo) => {
            let client = hf_hub::HFClientSync::new()
                .map_err(|e| format!("Hugging Face client error: {e}"))?;
            let (owner, name) = hf_hub::split_id(repo);
            let download = |file: &str| -> Result<PathBuf, String> {
                client
                    .model(owner, name)
                    .download_file()
                    .filename(file)
                    .send()
                    .map_err(|e| format!("failed to download '{file}' from '{repo}': {e}"))
            };
            let config = download("config.json")?;
            let tokenizer = download("tokenizer.json")?;

            progress("downloading shards", 0, 1, "file");
            let single = download("model.safetensors");
            let shards = match single {
                Ok(path) => vec![path],
                Err(_) => {
                    let index_path = download("model.safetensors.index.json")?;
                    let index_text = std::fs::read_to_string(&index_path)
                        .map_err(|e| format!("cannot read shard index: {e}"))?;
                    let index: serde_json::Value = serde_json::from_str(&index_text)
                        .map_err(|e| format!("invalid shard index: {e}"))?;
                    let weight_map = index
                        .get("weight_map")
                        .and_then(|v| v.as_object())
                        .ok_or_else(|| "shard index has no weight_map".to_string())?;
                    let mut names: Vec<String> = weight_map
                        .values()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect();
                    names.sort();
                    names.dedup();
                    let total = names.len() as u64;
                    let mut shards = Vec::with_capacity(names.len());
                    for (i, shard_name) in names.iter().enumerate() {
                        progress("downloading shards", i as u64, total, "files");
                        shards.push(download(shard_name)?);
                    }
                    shards
                }
            };
            Ok(HfFiles { config, tokenizer, shards })
        }
    }
}

/// Finds the safetensors shards in a local directory (single file or index).
fn shards_in_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index = dir.join("model.safetensors.index.json");
    if !index.is_file() {
        return Err(format!(
            "'{}' has neither model.safetensors nor model.safetensors.index.json",
            dir.display()
        ));
    }
    let text = std::fs::read_to_string(&index)
        .map_err(|e| format!("cannot read shard index: {e}"))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("invalid shard index: {e}"))?;
    let weight_map = parsed
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "shard index has no weight_map".to_string())?;
    let mut names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    let shards: Vec<PathBuf> = names.iter().map(|n| dir.join(n)).collect();
    for shard in &shards {
        if !shard.is_file() {
            return Err(format!("shard '{}' is missing", shard.display()));
        }
    }
    Ok(shards)
}

/// Converts a Hugging Face model into the native format. Refuses to overwrite
/// an existing model unless `force` is set.
pub fn import_hf<B: Backend>(
    source: &HfSource,
    output_dir: &Path,
    force: bool,
    device: &B::Device,
    progress: &mut dyn FnMut(&str, u64, u64, &str),
) -> Result<ModelConfig, String> {
    let weights_path = output_dir.join("model_weights.bin");
    if weights_path.exists() && !force {
        return Err(format!(
            "'{}' already contains a model (pass --force to overwrite)",
            output_dir.display()
        ));
    }

    progress("resolving files", 0, 1, "step");
    let files = resolve_files(source, progress)?;
    let hf: HfConfig = serde_json::from_str(
        &std::fs::read_to_string(&files.config)
            .map_err(|e| format!("cannot read config.json: {e}"))?,
    )
    .map_err(|e| format!("invalid Hugging Face config.json: {e}"))?;

    // Architecture gate.
    let arch = hf.architectures.first().map(String::as_str).unwrap_or("");
    if !matches!(
        arch,
        "LlamaForCausalLM" | "Qwen2ForCausalLM" | "MistralForCausalLM"
    ) {
        return Err(format!(
            "unsupported architecture '{arch}' (supported: LlamaForCausalLM, Qwen2ForCausalLM, \
             MistralForCausalLM)"
        ));
    }
    if let Some(act) = &hf.hidden_act {
        if act != "silu" {
            return Err(format!("unsupported hidden activation '{act}' (only silu/SwiGLU)"));
        }
    }
    if hf.tie_word_embeddings == Some(false) {
        return Err(
            "untied word embeddings are not supported (tie_word_embeddings must be true); the \
             engine always shares the embedding with the output projection"
                .into(),
        );
    }
    let head_dim = hf
        .head_dim
        .unwrap_or(hf.hidden_size / hf.num_attention_heads.max(1));
    if !hf.hidden_size.is_multiple_of(hf.num_attention_heads)
        || head_dim != hf.hidden_size / hf.num_attention_heads
    {
        return Err(format!(
            "explicit head_dim {head_dim} differs from hidden_size/num_attention_heads \
             ({}); explicit head dimensions are not supported",
            hf.hidden_size / hf.num_attention_heads.max(1)
        ));
    }

    progress("reading tensors", 0, 1, "step");
    let shards = ShardSet::open(&files.shards)?;
    if shards.contains("model.layers.0.self_attn.q_norm.weight") {
        return Err(
            "this model uses QK-norms (Qwen3-style), which the engine does not implement".into(),
        );
    }
    let (embed_shape, embed) = shards.read_f32("model.embed_tokens.weight")?;
    if embed_shape.len() != 2 || embed_shape[1] != hf.hidden_size {
        return Err(format!(
            "model.embed_tokens.weight has shape {embed_shape:?}, expected [vocab, {}]",
            hf.hidden_size
        ));
    }
    let vocab_size = embed_shape[0];

    let tokenizer = crate::data::LlmTokenizer::from_file(&files.tokenizer)
        .map_err(|e| format!("tokenizer.json is not loadable: {e}"))?;
    if tokenizer.vocab_size() > vocab_size {
        return Err(format!(
            "tokenizer vocabulary ({}) exceeds the embedding rows ({vocab_size})",
            tokenizer.vocab_size()
        ));
    }
    if tokenizer.vocab_size() < vocab_size {
        println!(
            "  note: the tokenizer defines {} tokens while the embedding has {vocab_size} rows; \
             the trailing rows are reserved and never produced",
            tokenizer.vocab_size()
        );
    }

    let n_kv_head = hf.num_key_value_heads.unwrap_or(hf.num_attention_heads);
    let mut config = ModelConfig::new(
        vocab_size,
        hf.hidden_size,
        hf.num_attention_heads,
        Some(n_kv_head),
        hf.num_hidden_layers,
        hf.max_position_embeddings,
        0.0,
        0.0,
    );
    config.ffn_hidden = Some(hf.intermediate_size);
    config.rms_norm_eps = hf.rms_norm_eps.unwrap_or(1e-6);
    config.rope_base = hf.rope_theta.unwrap_or(10000.0);
    config.import_source = Some("hf".into());
    if let Some(scaling) = &hf.rope_scaling {
        if !scaling.is_null() {
            let (kind, factor) = rope_scaling_from_hf(scaling)?;
            config.set_rope_scaling(kind, factor);
        }
    }

    println!(
        "Converting '{}' ({} layers, n_embd {}, hidden {}, vocab {}) -> '{}'",
        source.label(),
        hf.num_hidden_layers,
        hf.hidden_size,
        hf.intermediate_size,
        vocab_size,
        output_dir.display()
    );

    let mut model: Llm<B> = config.init_model(device);
    model.embed_tokens.weight = param2(embed, [vocab_size, hf.hidden_size], device);

    let norm = shards.read_f32("model.norm.weight")?;
    model.norm.weight = param1(norm.1, device);

    let hidden = hf.intermediate_size;
    let n_embd = hf.hidden_size;
    let n_head = hf.num_attention_heads;
    let kv_dim = n_kv_head * head_dim;
    let q_dim = n_head * head_dim;

    progress("converting tensors", 0, hf.num_hidden_layers as u64, "layers");
    for i in 0..hf.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        let attn = format!("{prefix}.self_attn");
        let read = |name: &str| -> Result<Vec<f32>, String> {
            shards
                .read_f32(name)
                .map(|(_, data)| data)
                .map_err(|e| format!("layer {i}: {e}"))
        };
        let optional = |name: &str| -> Result<Option<Vec<f32>>, String> {
            if shards.contains(name) {
                Ok(Some(read(name)?))
            } else {
                Ok(None)
            }
        };

        let q = read(&format!("{attn}.q_proj.weight"))?;
        if q.len() != q_dim * n_embd {
            return Err(format!("layer {i}: q_proj has {} values", q.len()));
        }
        let q = transpose_2d(&q, q_dim, n_embd);
        let k = read(&format!("{attn}.k_proj.weight"))?;
        if k.len() != kv_dim * n_embd {
            return Err(format!("layer {i}: k_proj has {} values", k.len()));
        }
        let k = transpose_2d(&k, kv_dim, n_embd);
        let v = read(&format!("{attn}.v_proj.weight"))?;
        if v.len() != kv_dim * n_embd {
            return Err(format!("layer {i}: v_proj has {} values", v.len()));
        }
        let v = transpose_2d(&v, kv_dim, n_embd);
        let o = read(&format!("{attn}.o_proj.weight"))?;
        if o.len() != n_embd * q_dim {
            return Err(format!("layer {i}: o_proj has {} values", o.len()));
        }
        let o = transpose_2d(&o, n_embd, q_dim);

        let gate = read(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let up = read(&format!("{prefix}.mlp.up_proj.weight"))?;
        let down = read(&format!("{prefix}.mlp.down_proj.weight"))?;
        if gate.len() != hidden * n_embd || up.len() != hidden * n_embd {
            return Err(format!("layer {i}: gate/up projection size mismatch"));
        }
        if down.len() != n_embd * hidden {
            return Err(format!("layer {i}: down_proj has {} values", down.len()));
        }
        let gate = transpose_2d(&gate, hidden, n_embd);
        let up = transpose_2d(&up, hidden, n_embd);
        let down = transpose_2d(&down, n_embd, hidden);

        let input_norm = read(&format!("{prefix}.input_layernorm.weight"))?;
        let post_norm = read(&format!("{prefix}.post_attention_layernorm.weight"))?;
        if input_norm.len() != n_embd || post_norm.len() != n_embd {
            return Err(format!("layer {i}: layernorm size mismatch"));
        }

        let q_bias = optional(&format!("{attn}.q_proj.bias"))?;
        let k_bias = optional(&format!("{attn}.k_proj.bias"))?;
        let v_bias = optional(&format!("{attn}.v_proj.bias"))?;

        let block = &mut model.blocks[i];
        block.input_layernorm.weight = param1(input_norm, device);
        block.post_attention_layernorm.weight = param1(post_norm, device);
        block.attn.q_proj.weight = param2(q, [n_embd, q_dim], device);
        block.attn.q_proj.bias = q_bias.map(|b| param1(b, device));
        block.attn.out_proj.weight = param2(o, [q_dim, n_embd], device);
        block.mlp.gate_proj.weight = param2(gate, [n_embd, hidden], device);
        block.mlp.up_proj.weight = param2(up, [n_embd, hidden], device);
        block.mlp.down_proj.weight = param2(down, [hidden, n_embd], device);

        let kv = &mut model.kv[i];
        kv.k_proj.weight = param2(k, [n_embd, kv_dim], device);
        kv.k_proj.bias = k_bias.map(|b| param1(b, device));
        kv.v_proj.weight = param2(v, [n_embd, kv_dim], device);
        kv.v_proj.bias = v_bias.map(|b| param1(b, device));

        progress("converting tensors", (i + 1) as u64, hf.num_hidden_layers as u64, "layers");
    }

    if shards.contains("lm_head.weight") && hf.tie_word_embeddings != Some(true) {
        println!(
            "  note: the model declares an untied lm_head but no tie_word_embeddings flag; \
             the engine uses the tied embedding"
        );
    }

    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("failed to create '{}': {e}", output_dir.display()))?;
    std::fs::write(
        output_dir.join("config.json"),
        serde_json::to_string_pretty(&config).map_err(|e| format!("config serialization: {e}"))?,
    )
    .map_err(|e| format!("failed to write config.json: {e}"))?;

    progress("writing weights", 0, 1, "step");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(model.into_record(), output_dir.join("model_weights"))
        .map_err(|e| format!("failed to write model_weights.bin: {e:?}"))?;

    let tokenizer_out = output_dir.join("custom_tokenizer.json");
    std::fs::copy(&files.tokenizer, &tokenizer_out)
        .map_err(|e| format!("failed to copy the tokenizer: {e}"))?;
    crate::data::LlmTokenizer::from_file(&tokenizer_out)
        .map_err(|e| format!("converted tokenizer is invalid: {e}"))?;

    println!(
        "Converted '{}' -> '{}' (config.json, model_weights.bin, custom_tokenizer.json)",
        source.label(),
        output_dir.display()
    );
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32, i32>;

    /// Writes a minimal F32 safetensors file (test helper).
    fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (name, shape, data) in tensors {
            let bytes = (data.len() * 4) as u64;
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        for (_, _, data) in tensors {
            for v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(path, out).unwrap();
    }

    fn tiny_tokenizer(dir: &Path) -> PathBuf {
        let corpus = dir.join("corpus.txt");
        std::fs::write(
            &corpus,
            "Once upon a time there was a small cat. The cat liked to play in the garden. \
             One day the cat met a dog and they became friends.\n",
        )
        .unwrap();
        let tokenizer_path = dir.join("tokenizer.json");
        crate::data::train_custom_tokenizer(
            corpus.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            300,
        )
        .expect("tiny tokenizer training failed");
        tokenizer_path
    }

    #[test]
    fn test_safetensors_read_f32_and_f16() {
        let dir = std::env::temp_dir().join(format!("kuro_st_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");
        write_safetensors(
            &path,
            &[("a".into(), vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
        );
        let set = ShardSet::open(&[path]).unwrap();
        let (shape, data) = set.read_f32("a").unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(set.read_f32("missing").is_err());

        // F16 header written by hand: one tensor with two values.
        let mut header = serde_json::Map::new();
        header.insert(
            "b".into(),
            serde_json::json!({"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}),
        );
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        for v in [1.5f32, -2.25] {
            out.extend_from_slice(&burn::tensor::f16::from_f32(v).to_bits().to_le_bytes());
        }
        let f16_path = dir.join("tiny16.safetensors");
        std::fs::write(&f16_path, out).unwrap();
        let set = ShardSet::open(&[f16_path]).unwrap();
        let (_, data) = set.read_f32("b").unwrap();
        assert!((data[0] - 1.5).abs() < 1e-3 && (data[1] + 2.25).abs() < 1e-3);
    }

    #[test]
    fn test_import_hf_roundtrip_matches_native_forward() {
        let device = NdArrayDevice::Cpu;
        let dir = std::env::temp_dir().join(format!("kuro_hf_{}", std::process::id()));
        let src = dir.join("src");
        let out = dir.join("out");
        std::fs::create_dir_all(&src).unwrap();
        let tokenizer_path = tiny_tokenizer(&src);
        let tokenizer = crate::data::LlmTokenizer::from_file(&tokenizer_path).unwrap();
        let vocab = tokenizer.vocab_size();

        let mut config = ModelConfig::new(vocab, 32, 4, Some(2), 2, 16, 0.0, 0.0);
        config.ffn_hidden = Some(48);
        config.rms_norm_eps = 1e-5;
        config.rope_base = 500000.0;
        let model: Llm<TestBackend> = config.init_model(&device);

        let flat = |t: burn::tensor::Tensor<TestBackend, 2>| {
            t.into_data().to_vec::<f32>().unwrap()
        };
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        tensors.push((
            "model.embed_tokens.weight".into(),
            vec![vocab, 32],
            flat(model.embed_tokens.weight.val()),
        ));
        tensors.push(("model.norm.weight".into(), vec![32], {
            model.norm.weight.val().into_data().to_vec::<f32>().unwrap()
        }));
        for (i, block) in model.blocks.iter().enumerate() {
            let prefix = format!("model.layers.{i}");
            let q = flat(block.attn.q_proj.weight.val());
            tensors.push((
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![32, 32],
                transpose_2d(&q, 32, 32),
            ));
            tensors.push((
                format!("{prefix}.self_attn.q_proj.bias"),
                vec![32],
                block.attn.q_proj.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap(),
            ));
            let kv = &model.kv[i];
            let k = flat(kv.k_proj.weight.val());
            tensors.push((
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![16, 32],
                transpose_2d(&k, 32, 16),
            ));
            tensors.push((
                format!("{prefix}.self_attn.k_proj.bias"),
                vec![16],
                kv.k_proj.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap(),
            ));
            let v = flat(kv.v_proj.weight.val());
            tensors.push((
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![16, 32],
                transpose_2d(&v, 32, 16),
            ));
            tensors.push((
                format!("{prefix}.self_attn.v_proj.bias"),
                vec![16],
                kv.v_proj.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap(),
            ));
            let o = flat(block.attn.out_proj.weight.val());
            tensors.push((
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![32, 32],
                transpose_2d(&o, 32, 32),
            ));
            let gate = flat(block.mlp.gate_proj.weight.val());
            tensors.push((
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![48, 32],
                transpose_2d(&gate, 32, 48),
            ));
            let up = flat(block.mlp.up_proj.weight.val());
            tensors.push((
                format!("{prefix}.mlp.up_proj.weight"),
                vec![48, 32],
                transpose_2d(&up, 32, 48),
            ));
            let down = flat(block.mlp.down_proj.weight.val());
            tensors.push((
                format!("{prefix}.mlp.down_proj.weight"),
                vec![32, 48],
                transpose_2d(&down, 48, 32),
            ));
            tensors.push((
                format!("{prefix}.input_layernorm.weight"),
                vec![32],
                block.input_layernorm.weight.val().into_data().to_vec::<f32>().unwrap(),
            ));
            tensors.push((
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![32],
                block
                    .post_attention_layernorm
                    .weight
                    .val()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            ));
        }
        write_safetensors(&src.join("model.safetensors"), &tensors);

        let hf_config = serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 16,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "tie_word_embeddings": true,
            "hidden_act": "silu",
            "attention_bias": true,
            "vocab_size": vocab,
        });
        std::fs::write(
            src.join("config.json"),
            serde_json::to_string_pretty(&hf_config).unwrap(),
        )
        .unwrap();

        let mut noop = |_: &str, _: u64, _: u64, _: &str| {};
        let imported_config =
            import_hf::<TestBackend>(&HfSource::Dir(src), &out, true, &device, &mut noop).unwrap();
        assert_eq!(imported_config.ffn_hidden, Some(48));
        assert_eq!(imported_config.rope_base, 500000.0);
        assert_eq!(imported_config.rms_norm_eps, 1e-5);
        assert_eq!(imported_config.import_source.as_deref(), Some("hf"));

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(out.join("model_weights"), &device)
            .unwrap();
        let imported: Llm<TestBackend> =
            imported_config.init_model(&device).load_record(record);

        let input = burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![1, 2, 3, 4, 5, 6], [1, 6]),
            &device,
        );
        let expected = model
            .forward(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let got = imported.forward(input).into_data().to_vec::<f32>().unwrap();
        let diff = expected
            .iter()
            .zip(got.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-5, "imported model diverged: max_diff={diff}");

        // The tokenizer must be usable from the model directory.
        assert!(out.join("custom_tokenizer.json").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_import_hf_rejects_untied_and_qknorm() {
        let device = NdArrayDevice::Cpu;
        let dir = std::env::temp_dir().join(format!("kuro_hf_bad_{}", std::process::id()));
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        let tokenizer_path = tiny_tokenizer(&src);
        let tokenizer = crate::data::LlmTokenizer::from_file(&tokenizer_path).unwrap();
        let vocab = tokenizer.vocab_size();
        write_safetensors(
            &src.join("model.safetensors"),
            &[(
                "model.embed_tokens.weight".into(),
                vec![vocab, 32],
                vec![0.0; vocab * 32],
            )],
        );
        let base = serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "max_position_embeddings": 16,
            "vocab_size": vocab,
        });
        let mut noop = |_: &str, _: u64, _: u64, _: &str| {};

        let mut untied = base.clone();
        untied["tie_word_embeddings"] = serde_json::json!(false);
        std::fs::write(src.join("config.json"), untied.to_string()).unwrap();
        let err = import_hf::<TestBackend>(&HfSource::Dir(src.clone()), &dir.join("o1"), true, &device, &mut noop)
            .unwrap_err();
        assert!(err.contains("untied"), "{err}");

        let mut qwen3 = base.clone();
        qwen3["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
        std::fs::write(src.join("config.json"), qwen3.to_string()).unwrap();
        let err = import_hf::<TestBackend>(&HfSource::Dir(src.clone()), &dir.join("o2"), true, &device, &mut noop)
            .unwrap_err();
        assert!(err.contains("unsupported architecture"), "{err}");

        std::fs::write(src.join("config.json"), base.to_string()).unwrap();
        let mut tensors = vec![(
            "model.embed_tokens.weight".into(),
            vec![vocab, 32],
            vec![0.0; vocab * 32],
        )];
        tensors.push((
            "model.layers.0.self_attn.q_norm.weight".into(),
            vec![8],
            vec![0.0; 8],
        ));
        write_safetensors(&src.join("model.safetensors"), &tensors);
        let err = import_hf::<TestBackend>(&HfSource::Dir(src), &dir.join("o3"), true, &device, &mut noop)
            .unwrap_err();
        assert!(err.contains("QK-norm"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
