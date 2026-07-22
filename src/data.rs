use hf_hub::api::sync::Api;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
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
    ) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        let path = file_path.as_ref();
        let file = File::open(path)?;
        let total_bytes = file.metadata()?.len();

        let reader = BufReader::with_capacity(2 * 1024 * 1024, file);
        let mut all_tokens = Vec::with_capacity(300_000_000);
        let mut bytes_read = 0u64;

        let start_time = Instant::now();
        let mut last_update = Instant::now();

        println!(
            "Streaming parallel multi-threaded tokenization ({:.2} MB)...",
            total_bytes as f64 / 1_048_576.0
        );

        let batch_line_capacity = 8192;
        let mut line_batch: Vec<String> = Vec::with_capacity(batch_line_capacity);
        let mut batch_bytes = 0u64;

        for line in reader.lines() {
            let line = line?;
            let line_len = line.len() as u64 + 1;
            batch_bytes += line_len;

            let trimmed = line.trim();
            if !trimmed.is_empty() {
                line_batch.push(trimmed.to_string());
            }

            if line_batch.len() >= batch_line_capacity {
                bytes_read += batch_bytes;
                batch_bytes = 0;

                // Take ownership of current batch without cloning string buffers
                let current_batch = std::mem::take(&mut line_batch);
                line_batch = Vec::with_capacity(batch_line_capacity);

                if let Ok(encodings) = self.tokenizer.encode_batch(current_batch, true) {
                    for enc in encodings {
                        all_tokens.extend(enc.get_ids());
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

                    print!(
                        "\r[Tokenizing] {:5.1}% ({:.1}/{:.1} MB) | {:.1} MB/s | Tokens: {} | ETA: {:02}m {:02}s   ",
                        pct, mb_read, mb_total, mb_per_sec, all_tokens.len(), eta_min, eta_s
                    );
                    let _ = std::io::stdout().flush();
                }
            }
        }

        // Process remaining batch
        if !line_batch.is_empty() {
            let current_batch = std::mem::take(&mut line_batch);

            if let Ok(encodings) = self.tokenizer.encode_batch(current_batch, true) {
                for enc in encodings {
                    all_tokens.extend(enc.get_ids());
                }
            }
        }

        let elapsed_secs = start_time.elapsed().as_secs_f64();
        let mb_total = total_bytes as f64 / 1_048_576.0;
        let mb_per_sec = if elapsed_secs > 0.0 { mb_total / elapsed_secs } else { 0.0 };

        print!(
            "\r[Tokenizing] 100.0% ({:.1}/{:.1} MB) | {:.1} MB/s | Tokens: {} | ETA: 00m 00s   \n",
            mb_total, mb_total, mb_per_sec, all_tokens.len()
        );
        let _ = std::io::stdout().flush();

        println!("Parallel tokenization complete in {:.2}s: {} total tokens loaded.", elapsed_secs, all_tokens.len());
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
    let local_cache = PathBuf::from(format!("dataset_{}_{}", repo_id.replace("/", "_"), filename));
    if local_cache.exists() {
        println!("Using cached dataset file: {:?}", local_cache);
        return Ok(local_cache);
    }

    println!("Downloading dataset '{}' ({}) from Hugging Face Hub...", repo_id, filename);
    let api = Api::new().map_err(|e| format!("HF Api error: {}", e))?;
    let repo = api.dataset(repo_id.to_string());
    let downloaded_file = repo.get(filename).map_err(|e| format!("HF dataset get error: {}", e))?;

    std::fs::copy(&downloaded_file, &local_cache)?;
    println!("Saved dataset locally to {:?}", local_cache);
    Ok(local_cache)
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
