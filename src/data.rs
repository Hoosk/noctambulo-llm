use hf_hub::{split_id, HFClientSync};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use crate::tui;
use std::time::Instant;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::NFKC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelPostProcessor;
use tokenizers::{Tokenizer, TokenizerBuilder};

pub struct LlmTokenizer {
    tokenizer: Tokenizer,
}

impl LlmTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer from file: {}", e))?;
        Ok(Self { tokenizer })
    }

    pub fn from_hf(repo_id: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let safe_name = repo_id.replace("/", "_");
        let local_path = PathBuf::from(format!("tokenizer_{}.json", safe_name));

        if local_path.exists() {
            println!("Loading cached tokenizer from: {:?}", local_path);
            return Self::from_file(&local_path);
        }

        println!("Downloading tokenizer config for '{}' from Hugging Face Hub...", repo_id);
        let client = HFClientSync::new().map_err(|e| format!("HF Api error: {}", e))?;
        let (owner, name) = split_id(repo_id);
        let tokenizer_path = client
            .model(owner, name)
            .download_file()
            .filename("tokenizer.json")
            .send()
            .map_err(|e| format!("HF get error: {}", e))?;

        if let Err(e) = std::fs::copy(&tokenizer_path, &local_path) {
            println!("Warning: Could not cache tokenizer locally: {}", e);
        }

        Self::from_file(tokenizer_path)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("Tokenization encoding error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Parallel multi-threaded batch tokenization using encode_batch() across all CPU cores.
    /// Uses zero-cost ownership transfers to avoid redundant Heap cloning.
    ///
    /// Returns `(tokens, bytes_consumed)`; the byte count feeds the
    /// bits-per-byte metric (0 for pre-tokenized `.bin` inputs, where the
    /// source text size is unknown).
    pub fn encode_file_streaming<P: AsRef<Path>>(
        &self,
        file_path: P,
        max_tokens: Option<usize>,
    ) -> Result<(Vec<u32>, u64), Box<dyn std::error::Error + Send + Sync>> {
        let path = file_path.as_ref();

        if is_binary_file(path)? {
            println!("Detected pre-tokenized binary dataset: {:?}", path);
            let expected_hash = Some(tokenizer_vocab_hash(self));
            let (tokens, vocab_sz) = load_tokens_from_binary(path, max_tokens, expected_hash)?;
            println!("Loaded {} pre-tokenized tokens (Vocab size in header: {}).", tokens.len(), vocab_sz);
            return Ok((tokens, 0));
        }

        let eos_id = self.end_of_text_id();

        let file = File::open(path)?;
        let total_bytes = file.metadata()?.len();

        let reader = BufReader::with_capacity(2 * 1024 * 1024, file);
        let estimated_tokens = match max_tokens {
            Some(limit) if limit > 0 => limit + 1024,
            _ => (total_bytes as usize / 3).min(500_000_000),
        };
        let mut all_tokens = Vec::with_capacity(estimated_tokens);
        let mut bytes_read = 0u64;

        let start_time = Instant::now();
        let mut last_update = Instant::now();

        tui::print_tokenization_start(total_bytes);

        let batch_line_capacity = 8192;
        let mut line_batch: Vec<String> = Vec::with_capacity(batch_line_capacity);
        let mut batch_bytes = 0u64;

        for line in reader.lines() {
            let line = line?;
            let line_len = line.len() as u64 + 1;
            batch_bytes += line_len;

            let trimmed = line.trim();
            if !trimmed.is_empty() {
                let text_entry = if trimmed.starts_with('{') && trimmed.ends_with('}') {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
                        if let Some(s) = v.get("story").and_then(|s| s.as_str()) {
                            s.to_string()
                        } else if let Some(s) = v.get("text").and_then(|s| s.as_str()) {
                            s.to_string()
                        } else if let Some(s) = v.get("story_text").and_then(|s| s.as_str()) {
                            s.to_string()
                        } else {
                            trimmed.to_string()
                        }
                    } else {
                        trimmed.to_string()
                    }
                } else {
                    trimmed.to_string()
                };

                line_batch.push(text_entry);
            }

            if line_batch.len() >= batch_line_capacity {
                bytes_read += batch_bytes;
                batch_bytes = 0;

                // Take ownership of current batch without cloning string buffers
                let current_batch = std::mem::take(&mut line_batch);
                line_batch = Vec::with_capacity(batch_line_capacity);

                match self.tokenizer.encode_batch(current_batch, true) {
                    Ok(encodings) => {
                        for enc in encodings {
                            all_tokens.extend(enc.get_ids());
                            if let Some(eos) = eos_id {
                                all_tokens.push(eos);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("\nWarning: Tokenizer batch failed to encode: {}", e);
                    }
                }

                if let Some(limit) = max_tokens {
                    if limit > 0 && all_tokens.len() >= limit {
                        break;
                    }
                }

                if last_update.elapsed().as_millis() >= 100 || bytes_read >= total_bytes {
                    last_update = Instant::now();
                    let elapsed_secs = start_time.elapsed().as_secs_f64();
                    let pct = if total_bytes > 0 { (bytes_read as f64 / total_bytes as f64) * 100.0 } else { 100.0 };
                    let bytes_per_sec = if elapsed_secs > 0.0 { bytes_read as f64 / elapsed_secs } else { 1.0 };
                    let remaining_bytes = total_bytes.saturating_sub(bytes_read) as f64;
                    let eta_secs = (remaining_bytes / bytes_per_sec) as u64;

                    let eta_min = eta_secs / 60;
                    let eta_s = eta_secs % 60;

                    let mb_read = bytes_read as f64 / 1_048_576.0;
                    let mb_total = total_bytes as f64 / 1_048_576.0;
                    let mb_per_sec = bytes_per_sec / 1_048_576.0;

                    tui::print_tokenization_progress(
                        pct,
                        mb_read,
                        mb_total,
                        mb_per_sec,
                        all_tokens.len(),
                        eta_min,
                        eta_s,
                    );
                }
            }
        }

        // Process remaining batch
        if !line_batch.is_empty() {
            let current_batch = std::mem::take(&mut line_batch);

            match self.tokenizer.encode_batch(current_batch, true) {
                Ok(encodings) => {
                    for enc in encodings {
                        all_tokens.extend(enc.get_ids());
                        if let Some(eos) = eos_id {
                            all_tokens.push(eos);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("\nWarning: Tokenizer batch failed to encode: {}", e);
                }
            }
        }

        let elapsed_secs = start_time.elapsed().as_secs_f64();
        let mb_total = total_bytes as f64 / 1_048_576.0;
        let mb_per_sec = if elapsed_secs > 0.0 { mb_total / elapsed_secs } else { 0.0 };

        tui::print_tokenization_complete(
            mb_total,
            mb_per_sec,
            all_tokens.len(),
            elapsed_secs,
        );
        Ok((all_tokens, bytes_read))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let text = self
            .tokenizer
            .decode(ids, true)
            .map_err(|e| format!("Tokenization decoding error: {}", e))?;
        Ok(text)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn end_of_text_id(&self) -> Option<u32> {
        for token in &["<|endoftext|>", "<|im_end|>", "</s>", "<eos>"] {
            if let Some(id) = self.tokenizer.token_to_id(token) {
                return Some(id);
            }
        }
        None
    }
}

pub fn download_dataset_from_hf(repo_id: &str, filename: &str) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let mut candidates = vec![filename];
    if filename != "TinyStories-train.txt" {
        candidates.push("TinyStories-train.txt");
    }
    if filename != "stories_00.json" {
        candidates.push("stories_00.json");
    }
    if filename != "tinystories.jsonl" {
        candidates.push("tinystories.jsonl");
    }
    if filename != "spanish_stories.txt" {
        candidates.push("spanish_stories.txt");
    }
    if filename != "train.txt" {
        candidates.push("train.txt");
    }
    if filename != "data.txt" {
        candidates.push("data.txt");
    }

    // 1. Check local cache first for any candidate
    for cand in &candidates {
        let dataset_name = format!("dataset_{}_{}", repo_id.replace("/", "_"), cand.replace("/", "_"));
        let root_cache = PathBuf::from(&dataset_name);
        if root_cache.exists() {
            println!("Using existing cached dataset file: {:?}", root_cache);
            return Ok(root_cache);
        }

        let datasets_dir = PathBuf::from("datasets");
        let local_cache = datasets_dir.join(&dataset_name);
        if local_cache.exists() {
            println!("Using cached dataset file: {:?}", local_cache);
            return Ok(local_cache);
        }
    }

    // 2. Download from HF Hub trying candidates in order
    let client = HFClientSync::new().map_err(|e| format!("HF Api error: {}", e))?;
    let (owner, name) = split_id(repo_id);
    let repo = client.dataset(owner, name);
    let datasets_dir = PathBuf::from("datasets");
    std::fs::create_dir_all(&datasets_dir)?;

    let mut last_err = None;
    for cand in candidates {
        let dataset_name = format!("dataset_{}_{}", repo_id.replace("/", "_"), cand.replace("/", "_"));
        let local_cache = datasets_dir.join(&dataset_name);

        println!("Downloading dataset '{}' ({}) from Hugging Face Hub...", repo_id, cand);
        match repo.download_file().filename(cand).send() {
            Ok(downloaded_file) => {
                std::fs::copy(&downloaded_file, &local_cache)?;
                println!("Saved dataset locally to {:?}", local_cache);
                return Ok(local_cache);
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }

    Err(format!("Failed to download dataset from HF repo '{}': {:?}", repo_id, last_err.unwrap()).into())
}

/// Extrae documentos de texto de un archivo Parquet (columna `text` con fallbacks
/// a `content`/`body`) y los escribe a un `.txt` destino, un documento por línea.
/// Devuelve el número de documentos escritos.
pub fn extract_parquet_text_column(
    parquet_path: &Path,
    output_txt_path: &Path,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    use arrow::array::{Array, LargeStringArray, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    println!("Extracting text column from Parquet {:?}...", parquet_path);

    let file = File::open(parquet_path)
        .map_err(|e| format!("Failed to open parquet file {:?}: {}", parquet_path, e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Failed to parse parquet {:?}: {}", parquet_path, e))?;

    let schema = builder.schema().clone();
    let mut text_col: Option<String> = None;
    for candidate in ["text", "content", "body", "raw"] {
        if schema.fields().iter().any(|f| f.name() == candidate) {
            text_col = Some(candidate.to_string());
            break;
        }
    }
    let text_col = match text_col {
        Some(col) => col,
        None => {
            let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
            return Err(format!(
                "No text column found in parquet {:?}. Columns available: {:?}",
                parquet_path, names
            )
            .into());
        }
    };

    let mut reader = builder.build()
        .map_err(|e| format!("Failed to build parquet reader: {}", e))?;

    let out_file = File::create(output_txt_path)?;
    let mut writer = std::io::BufWriter::with_capacity(2 * 1024 * 1024, out_file);
    let mut docs = 0u64;

    for batch_res in &mut reader {
        let batch = batch_res.map_err(|e| format!("Parquet read error: {}", e))?;
        let col = batch.column_by_name(&text_col);
        let Some(col) = col else { continue; };
        let data = col.as_any().downcast_ref::<StringArray>()
            .map(|a| a.iter().collect::<Vec<Option<&str>>>())
            .or_else(|| {
                col.as_any().downcast_ref::<LargeStringArray>()
                    .map(|a| a.iter().collect::<Vec<Option<&str>>>())
            });
        let Some(data) = data else { continue; };
        for val in data {
            if let Some(text) = val {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    writer.write_all(trimmed.as_bytes())?;
                    writer.write_all(b"\n")?;
                    docs += 1;
                }
            }
        }
    }

    writer.flush()?;
    println!("Parquet extraction done: {} documents written to {:?}.", docs, output_txt_path);
    Ok(docs)
}

pub fn combine_and_mix_datasets(
    sources: &[PathBuf],
    output_path: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    if sources.len() == 1 {
        return Ok(sources[0].clone());
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("Combining and mixing {} datasets into {:?}...", sources.len(), output_path);

    let mut readers: Vec<BufReader<File>> = Vec::new();
    for src in sources {
        let file = File::open(src).map_err(|e| format!("Failed to open dataset file {:?}: {}", src, e))?;
        readers.push(BufReader::with_capacity(1024 * 1024, file));
    }

    let out_file = File::create(output_path)?;
    let mut writer = std::io::BufWriter::with_capacity(2 * 1024 * 1024, out_file);

    let mut total_lines = 0u64;
    let mut active_count = readers.len();
    let mut active_flags = vec![true; readers.len()];
    let mut line_buf = String::new();

    while active_count > 0 {
        for i in 0..readers.len() {
            if !active_flags[i] {
                continue;
            }
            line_buf.clear();
            match readers[i].read_line(&mut line_buf) {
                Ok(0) => {
                    active_flags[i] = false;
                    active_count -= 1;
                }
                Ok(_) => {
                    let trimmed = line_buf.trim();
                    if !trimmed.is_empty() {
                        writer.write_all(trimmed.as_bytes())?;
                        writer.write_all(b"\n")?;
                        total_lines += 1;
                    }
                }
                Err(e) => {
                    println!("Warning: error reading from dataset {:?}: {}", sources[i], e);
                    active_flags[i] = false;
                    active_count -= 1;
                }
            }
        }
    }

    writer.flush()?;
    println!("Combined dataset ready: {} lines written to {:?}.", total_lines, output_path);
    Ok(output_path.to_path_buf())
}

/// Mezcla ponderada de datasets (uno por línea) respetando las proporciones
/// `weights` y un presupuesto máximo de caracteres `max_chars`.
/// Selección probabilística por peso en cada paso (sin sesgo de orden).
pub fn combine_and_mix_weighted(
    sources: &[(PathBuf, f32)],
    output_path: &Path,
    max_chars: usize,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let total_weight: f32 = sources.iter().map(|(_, w)| w.max(0.0)).sum();
    if total_weight <= 0.0 {
        return Err("Total weight must be > 0".into());
    }

    println!(
        "Combining {} datasets with weights {:?} (max {} chars) into {:?}...",
        sources.len(),
        sources.iter().map(|(_, w)| w).collect::<Vec<_>>(),
        max_chars,
        output_path
    );

    let mut readers: Vec<BufReader<File>> = Vec::new();
    for (src, _) in sources {
        let file = File::open(src).map_err(|e| format!("Failed to open dataset file {:?}: {}", src, e))?;
        readers.push(BufReader::with_capacity(1024 * 1024, file));
    }

    let out_file = File::create(output_path)?;
    let mut writer = std::io::BufWriter::with_capacity(2 * 1024 * 1024, out_file);

    let mut written_chars = 0usize;
    let mut total_lines = 0u64;
    let mut line_buf = String::new();
    let mut last_pct = -1i32;

    // Fuentes activas: (índice, peso). Se re-normaliza al agotar una fuente.
    let mut active: Vec<(usize, f32)> =
        sources.iter().enumerate().map(|(i, (_, w))| (i, w.max(0.0))).collect();

    // PRNG simple (xorshift64*)
    let mut state: u64 = 0x9E3779B97F4A7C15 ^ std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xDEADBEEF);

    while !active.is_empty() && written_chars < max_chars {
        // Sortear una fuente activa proporcionalmente a su peso
        let total: f32 = active.iter().map(|(_, w)| w).sum();
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545F4914F6CDD1D) >> 33;
        let r = (r as f32 / (1u64 << 32) as f32) * total;
        let mut acc = 0.0f32;
        let mut pick = 0usize;
        for (pos, (_, w)) in active.iter().enumerate() {
            acc += *w;
            if r < acc {
                pick = pos;
                break;
            }
        }
        let idx = active[pick].0;

        line_buf.clear();
        match readers[idx].read_line(&mut line_buf) {
            Ok(0) => {
                active.remove(pick);
            }
            Ok(_) => {
                let trimmed = line_buf.trim();
                if !trimmed.is_empty() {
                    let bytes = trimmed.as_bytes();
                    if written_chars + bytes.len() > max_chars {
                        active.remove(pick);
                        continue;
                    }
                    writer.write_all(bytes)?;
                    writer.write_all(b"\n")?;
                    written_chars += bytes.len() + 1;
                    total_lines += 1;

                    let pct = if max_chars == usize::MAX {
                        0
                    } else {
                        (written_chars * 100 / max_chars) as i32
                    };
                    if pct != last_pct && (pct % 2 == 0 || pct == 100) {
                        last_pct = pct;
                        print!(
                            "\r  Progress: {:>3}%  ({} lines, {:.1} MB written, {} active sources)",
                            pct,
                            total_lines,
                            written_chars as f64 / 1_048_576.0,
                            active.len()
                        );
                        let _ = std::io::stdout().flush();
                    }
                }
            }
            Err(e) => {
                println!("Warning: error reading from dataset {:?}: {}", sources[idx].0, e);
                active.remove(pick);
            }
        }
    }

    writer.flush()?;
    println!();
    println!(
        "Weighted mix ready: {} lines / {} chars written to {:?}.",
        total_lines, written_chars, output_path
    );
    Ok(output_path.to_path_buf())
}


pub fn train_custom_tokenizer(
    corpus_path: &str,
    output_json_path: &str,
    vocab_size: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Training custom BPE tokenizer on '{}' (vocab size: {})...", corpus_path, vocab_size);

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(2)
        .special_tokens(vec![
            tokenizers::AddedToken::from("<|endoftext|>", true),
            tokenizers::AddedToken::from("<|padding|>", true),
        ])
        .build();

    let mut tokenizer_impl = TokenizerBuilder::<
        BPE,
        NFKC,
        ByteLevelPreTokenizer,
        ByteLevelPostProcessor,
        ByteLevelDecoder,
    >::default()
    .with_model(BPE::default())
    .with_normalizer(Some(NFKC))
    .with_pre_tokenizer(Some(ByteLevelPreTokenizer::default()))
    .with_post_processor(Some(ByteLevelPostProcessor::default()))
    .with_decoder(Some(ByteLevelDecoder::default()))
    .build()
    .map_err(|e| format!("TokenizerBuilder failed: {}", e))?;

    tokenizer_impl.train_from_files(&mut trainer, vec![corpus_path.to_string()])
        .map_err(|e| format!("Tokenizer training failed: {}", e))?;

    let tokenizer: Tokenizer = tokenizer_impl.into();
    tokenizer.save(output_json_path, true)
        .map_err(|e| format!("Saving tokenizer failed: {}", e))?;

    println!("Successfully trained custom tokenizer -> saved to '{}'", output_json_path);
    Ok(())
}

const BINARY_MAGIC: &[u8; 4] = b"KNKB";
/// v1: magic, version, vocab_size, token_count, tokens.
/// v2: adds a tokenizer vocabulary fingerprint after vocab_size.
const BINARY_VERSION: u32 = 2;

fn fnv1a64(bytes: &[u8], mut hash: u64) -> u64 {
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Deterministic fingerprint of a tokenizer vocabulary (id -> token pairs in
/// id order). Stored in `.bin` datasets so a tokenizer swap cannot silently
/// corrupt training data.
pub fn tokenizer_vocab_hash(tokenizer: &LlmTokenizer) -> u64 {
    let vocab = tokenizer.tokenizer.get_vocab(true);
    let mut entries: Vec<(u32, &str)> = vocab.iter().map(|(token, &id)| (id, token.as_str())).collect();
    entries.sort_unstable_by_key(|(id, _)| *id);

    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash = fnv1a64(&(entries.len() as u64).to_le_bytes(), hash);
    for (id, token) in entries {
        hash = fnv1a64(&id.to_le_bytes(), hash);
        hash = fnv1a64(token.as_bytes(), hash);
    }
    hash
}

pub fn is_binary_file<P: AsRef<Path>>(path: P) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let path = path.as_ref();
    if let Some(ext) = path.extension() {
        if ext == "bin" {
            return Ok(true);
        }
    }
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_ok() {
        if &magic == BINARY_MAGIC {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn save_tokens_to_binary<P: AsRef<Path>>(
    path: P,
    tokens: &[u32],
    vocab_size: u32,
    vocab_hash: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = std::io::BufWriter::new(File::create(path)?);
    file.write_all(BINARY_MAGIC)?;
    file.write_all(&BINARY_VERSION.to_le_bytes())?;
    file.write_all(&vocab_size.to_le_bytes())?;
    file.write_all(&vocab_hash.to_le_bytes())?;
    file.write_all(&(tokens.len() as u64).to_le_bytes())?;

    for chunk in tokens.chunks(1024) {
        let mut bytes = Vec::with_capacity(chunk.len() * 4);
        for &t in chunk {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        file.write_all(&bytes)?;
    }
    file.flush()?;
    Ok(())
}

pub fn load_tokens_from_binary<P: AsRef<Path>>(
    path: P,
    max_tokens: Option<usize>,
    expected_vocab_hash: Option<u64>,
) -> Result<(Vec<u32>, u32), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != BINARY_MAGIC {
        return Err("Invalid binary dataset header: magic bytes mismatch".into());
    }

    let mut ver_buf = [0u8; 4];
    file.read_exact(&mut ver_buf)?;
    let version = u32::from_le_bytes(ver_buf);
    if version != 1 && version != BINARY_VERSION {
        return Err(format!("Unsupported binary dataset version: {}", version).into());
    }

    let mut vocab_buf = [0u8; 4];
    file.read_exact(&mut vocab_buf)?;
    let vocab_size = u32::from_le_bytes(vocab_buf);

    let stored_vocab_hash = if version >= 2 {
        let mut hash_buf = [0u8; 8];
        file.read_exact(&mut hash_buf)?;
        Some(u64::from_le_bytes(hash_buf))
    } else {
        None
    };

    match (expected_vocab_hash, stored_vocab_hash) {
        (Some(expected), Some(stored)) if expected != stored => {
            return Err(format!(
                "Tokenizer/dataset mismatch: the .bin dataset was tokenized with a different \
                 tokenizer (dataset fingerprint {stored:#018x}, current tokenizer {expected:#018x}). \
                 Re-run `prepare-dataset` with the current tokenizer."
            )
            .into());
        }
        (Some(_), None) => {
            eprintln!("Warning: legacy v1 binary dataset has no tokenizer fingerprint; skipping validation.");
        }
        _ => {}
    }

    let mut count_buf = [0u8; 8];
    file.read_exact(&mut count_buf)?;
    let total_tokens = u64::from_le_bytes(count_buf) as usize;

    let target_tokens = match max_tokens {
        Some(limit) if limit > 0 => total_tokens.min(limit),
        _ => total_tokens,
    };

    let mut tokens = Vec::with_capacity(target_tokens);
    let mut buf = [0u8; 4 * 1024];

    while tokens.len() < target_tokens {
        let needed_tokens = target_tokens - tokens.len();
        let needed_bytes = (needed_tokens * 4).min(buf.len());
        let read_bytes = file.read(&mut buf[..needed_bytes])?;
        if read_bytes == 0 {
            break;
        }
        for chunk in buf[..read_bytes].chunks_exact(4) {
            tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }

    Ok((tokens, vocab_size))
}

pub fn prepare_binary_dataset<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_path: P1,
    output_path: P2,
    tokenizer: &LlmTokenizer,
    max_tokens: Option<usize>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Preparing pre-tokenized binary dataset from {:?} to {:?}...", input_path.as_ref(), output_path.as_ref());
    let (tokens, _bytes) = tokenizer.encode_file_streaming(input_path, max_tokens)?;
    let vocab_size = tokenizer.vocab_size() as u32;
    let vocab_hash = tokenizer_vocab_hash(tokenizer);
    save_tokens_to_binary(output_path.as_ref(), &tokens, vocab_size, vocab_hash)?;
    println!("Saved binary dataset with {} tokens (vocab_size: {}) to {:?}", tokens.len(), vocab_size, output_path.as_ref());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_dataset_roundtrip() {
        let path = std::env::temp_dir().join("test_kuro_bin_roundtrip.bin");

        let original_tokens = vec![1, 100, 500, 1000, 7999, 42];
        let original_vocab_size = 8000;
        let original_hash = 0xDEAD_BEEF_CAFE_F00Du64;

        save_tokens_to_binary(&path, &original_tokens, original_vocab_size, original_hash).unwrap();

        assert!(is_binary_file(&path).unwrap());

        let (loaded_tokens, loaded_vocab) =
            load_tokens_from_binary(&path, None, Some(original_hash)).unwrap();
        assert_eq!(loaded_tokens, original_tokens);
        assert_eq!(loaded_vocab, original_vocab_size);

        // Without an expected fingerprint the file still loads.
        let (loaded_tokens, _) = load_tokens_from_binary(&path, None, None).unwrap();
        assert_eq!(loaded_tokens, original_tokens);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_binary_dataset_max_tokens_limit() {
        let path = std::env::temp_dir().join("test_kuro_bin_limit.bin");

        let original_tokens: Vec<u32> = (0..100).collect();
        save_tokens_to_binary(&path, &original_tokens, 1000, 0x1234).unwrap();

        let (loaded_tokens, _) = load_tokens_from_binary(&path, Some(30), Some(0x1234)).unwrap();
        assert_eq!(loaded_tokens.len(), 30);
        assert_eq!(loaded_tokens, (0..30).collect::<Vec<u32>>());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_binary_vocab_hash_mismatch_rejected() {
        let path = std::env::temp_dir().join("test_kuro_bin_mismatch.bin");

        save_tokens_to_binary(&path, &[1, 2, 3], 100, 0xAAAA).unwrap();

        let err = load_tokens_from_binary(&path, None, Some(0xBBBB)).unwrap_err();
        assert!(
            err.to_string().contains("mismatch"),
            "expected tokenizer mismatch error, got: {err}"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_binary_v1_without_hash_still_loads() {
        let path = std::env::temp_dir().join("test_kuro_bin_v1.bin");

        let tokens: Vec<u32> = vec![5, 6, 7];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(BINARY_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&100u32.to_le_bytes());
        bytes.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for token in &tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();

        // A v1 file cannot be validated: it must load with a warning, not fail.
        let (loaded, vocab) = load_tokens_from_binary(&path, None, Some(0x1234)).unwrap();
        assert_eq!(loaded, tokens);
        assert_eq!(vocab, 100);

        let _ = std::fs::remove_file(&path);
    }
}

