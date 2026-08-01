use hf_hub::api::sync::Api;
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
        let api = Api::new().map_err(|e| format!("HF Api error: {}", e))?;
        let repo = api.model(repo_id.to_string());
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| format!("HF get error: {}", e))?;

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
    pub fn encode_file_streaming<P: AsRef<Path>>(
        &self,
        file_path: P,
        max_tokens: Option<usize>,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        let path = file_path.as_ref();

        if is_binary_file(path)? {
            println!("Detected pre-tokenized binary dataset: {:?}", path);
            let (tokens, vocab_sz) = load_tokens_from_binary(path, max_tokens)?;
            println!("Loaded {} pre-tokenized tokens (Vocab size in header: {}).", tokens.len(), vocab_sz);
            return Ok(tokens);
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
        Ok(all_tokens)
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
        let dataset_name = format!("dataset_{}_{}", repo_id.replace("/", "_"), cand);
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
    let api = Api::new().map_err(|e| format!("HF Api error: {}", e))?;
    let repo = api.dataset(repo_id.to_string());
    let datasets_dir = PathBuf::from("datasets");
    std::fs::create_dir_all(&datasets_dir)?;

    let mut last_err = None;
    for cand in &candidates {
        let dataset_name = format!("dataset_{}_{}", repo_id.replace("/", "_"), cand);
        let local_cache = datasets_dir.join(&dataset_name);

        println!("Downloading dataset '{}' ({}) from Hugging Face Hub...", repo_id, cand);
        match repo.get(cand) {
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
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = std::io::BufWriter::new(File::create(path)?);
    file.write_all(BINARY_MAGIC)?;
    file.write_all(&1u32.to_le_bytes())?;
    file.write_all(&vocab_size.to_le_bytes())?;
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
    if version != 1 {
        return Err(format!("Unsupported binary dataset version: {}", version).into());
    }

    let mut vocab_buf = [0u8; 4];
    file.read_exact(&mut vocab_buf)?;
    let vocab_size = u32::from_le_bytes(vocab_buf);

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
    let tokens = tokenizer.encode_file_streaming(input_path, max_tokens)?;
    let vocab_size = tokenizer.vocab_size() as u32;
    save_tokens_to_binary(output_path.as_ref(), &tokens, vocab_size)?;
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

        save_tokens_to_binary(&path, &original_tokens, original_vocab_size).unwrap();

        assert!(is_binary_file(&path).unwrap());

        let (loaded_tokens, loaded_vocab) = load_tokens_from_binary(&path, None).unwrap();
        assert_eq!(loaded_tokens, original_tokens);
        assert_eq!(loaded_vocab, original_vocab_size);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_binary_dataset_max_tokens_limit() {
        let path = std::env::temp_dir().join("test_kuro_bin_limit.bin");

        let original_tokens: Vec<u32> = (0..100).collect();
        save_tokens_to_binary(&path, &original_tokens, 1000).unwrap();

        let (loaded_tokens, _) = load_tokens_from_binary(&path, Some(30)).unwrap();
        assert_eq!(loaded_tokens.len(), 30);
        assert_eq!(loaded_tokens, (0..30).collect::<Vec<u32>>());

        let _ = std::fs::remove_file(&path);
    }
}

