use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use hf_hub::{HFClientSync, split_id};
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::NFKC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelPostProcessor;
use tokenizers::{Tokenizer, TokenizerBuilder};

use crate::tui;

use super::binary::{is_binary_file, load_tokens_from_binary, tokenizer_vocab_hash};
use super::document::DocumentStream;
use super::mix::MultiSourceDocumentStream;

pub struct LlmTokenizer {
    pub(crate) tokenizer: Tokenizer,
}

/// UTF-8-safe incremental decoder used by streaming generation: token ids are
/// fed one by one and it only emits chunks that are valid on their own.
pub struct LlmDecodeStream<'a> {
    inner: tokenizers::DecodeStream<
        'a,
        tokenizers::ModelWrapper,
        tokenizers::NormalizerWrapper,
        tokenizers::PreTokenizerWrapper,
        tokenizers::PostProcessorWrapper,
        tokenizers::DecoderWrapper,
    >,
}

impl LlmDecodeStream<'_> {
    pub fn step(
        &mut self,
        id: u32,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        self.inner
            .step(id)
            .map_err(|e| format!("Decode stream error: {e}").into())
    }
}

/// Path of the tokenizer that belongs to `model_dir`, falling back to the
/// working directory and `None` (caller may use the Hugging Face fallback).
pub fn resolve_tokenizer_path(model_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = model_dir {
        let candidate = dir.join("custom_tokenizer.json");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let cwd = PathBuf::from("custom_tokenizer.json");
    cwd.is_file().then_some(cwd)
}

/// Load the tokenizer that belongs to `model_dir` (see
/// [`resolve_tokenizer_path`]), falling back to the Qwen2.5 HF tokenizer.
pub fn resolve_tokenizer(
    model_dir: Option<&Path>,
) -> Result<LlmTokenizer, Box<dyn std::error::Error + Send + Sync>> {
    match resolve_tokenizer_path(model_dir) {
        Some(path) => LlmTokenizer::from_file(path),
        None => LlmTokenizer::from_hf("Qwen/Qwen2.5-0.5B"),
    }
}

/// Encode one batch of documents into `tokens`, appending the EOS id when the
/// tokenizer defines one.
pub(crate) fn encode_doc_batch(
    tokenizer: &LlmTokenizer,
    batch: &mut Vec<String>,
    tokens: &mut Vec<u32>,
    eos_id: Option<u32>,
) {
    if batch.is_empty() {
        return;
    }
    let current = std::mem::take(batch);
    match tokenizer.tokenizer.encode_batch(current, true) {
        Ok(encodings) => {
            for enc in encodings {
                tokens.extend(enc.get_ids());
                if let Some(eos) = eos_id {
                    tokens.push(eos);
                }
            }
        }
        Err(e) => eprintln!("\nWarning: Tokenizer batch failed to encode: {e}"),
    }
}

fn hash_document(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

impl LlmTokenizer {
    pub(crate) fn inner(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Incremental decoder for token-by-token streaming.
    pub fn decode_stream(&self, skip_special_tokens: bool) -> LlmDecodeStream<'_> {
        LlmDecodeStream {
            inner: self.tokenizer.decode_stream(skip_special_tokens),
        }
    }

    pub fn from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer from file: {}", e))?;
        Ok(Self { tokenizer })
    }

    pub fn from_hf(repo_id: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let safe_name = repo_id.replace('/', "_");
        let local_path = PathBuf::from(format!("tokenizer_{}.json", safe_name));

        if local_path.exists() {
            println!("Loading cached tokenizer from: {:?}", local_path);
            return Self::from_file(&local_path);
        }

        println!(
            "Downloading tokenizer config for '{}' from Hugging Face Hub...",
            repo_id
        );
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

    /// Parallel multi-threaded batch tokenization across all CPU cores.
    /// Reads complete documents from `.parquet`, `.jsonl`, `.json`,
    /// `<|endoftext|>`-delimited `.txt`, line `.txt`, or pre-tokenized `.bin`.
    pub fn encode_file_streaming<P: AsRef<Path>>(
        &self,
        file_path: P,
        max_tokens: Option<usize>,
    ) -> Result<(Vec<u32>, u64), Box<dyn std::error::Error + Send + Sync>> {
        let (tokens, bytes, _duplicates) =
            self.encode_file_streaming_opts(file_path, max_tokens, false, None, None)?;
        Ok((tokens, bytes))
    }

    /// Streaming document encoding with a `(bytes_read, total_bytes)` progress
    /// callback and cooperative cancellation.
    pub fn encode_file_streaming_with_progress<P: AsRef<Path>>(
        &self,
        file_path: P,
        max_tokens: Option<usize>,
        progress: Option<&dyn Fn(u64, u64)>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Vec<u32>, u64), Box<dyn std::error::Error + Send + Sync>> {
        let (tokens, bytes, _duplicates) =
            self.encode_file_streaming_opts(file_path, max_tokens, false, progress, cancel)?;
        Ok((tokens, bytes))
    }

    /// Streaming document encoding with exact-duplicate removal (documents are
    /// hashed and repeated ones skipped).
    pub fn encode_file_streaming_deduped<P: AsRef<Path>>(
        &self,
        file_path: P,
        max_tokens: Option<usize>,
    ) -> Result<(Vec<u32>, u64, u64), Box<dyn std::error::Error + Send + Sync>> {
        self.encode_file_streaming_opts(file_path, max_tokens, true, None, None)
    }

    pub(crate) fn encode_file_streaming_opts<P: AsRef<Path>>(
        &self,
        file_path: P,
        max_tokens: Option<usize>,
        dedupe: bool,
        progress: Option<&dyn Fn(u64, u64)>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Vec<u32>, u64, u64), Box<dyn std::error::Error + Send + Sync>> {
        let path = file_path.as_ref();

        if is_binary_file(path)? {
            println!("Detected pre-tokenized binary dataset: {:?}", path);
            let expected_hash = Some(tokenizer_vocab_hash(self));
            let (tokens, vocab_sz) = load_tokens_from_binary(path, max_tokens, expected_hash)?;
            println!(
                "Loaded {} pre-tokenized tokens (Vocab size in header: {}).",
                tokens.len(),
                vocab_sz
            );
            return Ok((tokens, 0, 0));
        }

        let mut stream = DocumentStream::open(path)?;
        let total_bytes = stream.total_bytes();
        self.encode_document_iter(
            || stream.next_document(),
            total_bytes,
            max_tokens,
            dedupe,
            progress,
            cancel,
        )
    }

    /// Stream and tokenize multiple weighted dataset sources directly in memory
    /// without creating an intermediate `combined_corpus.txt` on disk.
    #[allow(dead_code)]
    pub fn encode_sources_streaming_with_progress(
        &self,
        sources: &[(PathBuf, f32)],
        max_tokens: Option<usize>,
        max_chars: Option<usize>,
        progress: Option<&dyn Fn(u64, u64)>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Vec<u32>, u64), Box<dyn std::error::Error + Send + Sync>> {
        let (tokens, bytes, _duplicates) = self.encode_sources_streaming_opts(
            sources, max_tokens, max_chars, false, progress, cancel,
        )?;
        Ok((tokens, bytes))
    }

    pub(crate) fn encode_sources_streaming_opts(
        &self,
        sources: &[(PathBuf, f32)],
        max_tokens: Option<usize>,
        max_chars: Option<usize>,
        dedupe: bool,
        progress: Option<&dyn Fn(u64, u64)>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Vec<u32>, u64, u64), Box<dyn std::error::Error + Send + Sync>> {
        if sources.len() == 1 && max_chars.is_none() {
            return self.encode_file_streaming_opts(
                &sources[0].0,
                max_tokens,
                dedupe,
                progress,
                cancel,
            );
        }

        let mut mixer = MultiSourceDocumentStream::open(sources, max_chars)?;
        let total_bytes = mixer.total_bytes();
        self.encode_document_iter(
            || mixer.next_document(),
            total_bytes,
            max_tokens,
            dedupe,
            progress,
            cancel,
        )
    }

    fn encode_document_iter<F>(
        &self,
        mut next_doc: F,
        total_bytes: u64,
        max_tokens: Option<usize>,
        dedupe: bool,
        progress: Option<&dyn Fn(u64, u64)>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(Vec<u32>, u64, u64), Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnMut() -> Result<Option<super::document::Document>, Box<dyn std::error::Error + Send + Sync>>,
    {
        let eos_id = self.end_of_text_id();
        let estimated_tokens = match max_tokens {
            Some(limit) if limit > 0 => limit + 1024,
            _ => (total_bytes as usize / 3).min(500_000_000),
        };
        let mut all_tokens = Vec::with_capacity(estimated_tokens);
        let mut bytes_read = 0u64;
        let mut duplicates = 0u64;
        let mut seen: HashSet<u64> = HashSet::new();

        let start_time = Instant::now();
        let mut last_update = Instant::now();

        tui::print_tokenization_start(total_bytes);
        if let Some(callback) = progress {
            callback(0, total_bytes);
        }

        let batch_doc_capacity = 4096;
        let mut doc_batch: Vec<String> = Vec::with_capacity(batch_doc_capacity);
        let mut batch_bytes = 0u64;

        while let Some(doc) = next_doc()? {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err("cancelled by user".into());
            }

            batch_bytes += doc.bytes_consumed.max(doc.text.len() as u64);

            if dedupe && !seen.insert(hash_document(&doc.text)) {
                duplicates += 1;
            } else {
                doc_batch.push(doc.text);
            }

            if doc_batch.len() >= batch_doc_capacity || batch_bytes >= 4 * 1024 * 1024 {
                bytes_read += batch_bytes;
                batch_bytes = 0;

                encode_doc_batch(self, &mut doc_batch, &mut all_tokens, eos_id);
                doc_batch.reserve(batch_doc_capacity);

                if let Some(limit) = max_tokens
                    && limit > 0
                    && all_tokens.len() >= limit
                {
                    break;
                }

                if last_update.elapsed().as_millis() >= 100 || bytes_read >= total_bytes {
                    last_update = Instant::now();
                    let elapsed_secs = start_time.elapsed().as_secs_f64();
                    let pct = if total_bytes > 0 {
                        ((bytes_read as f64 / total_bytes as f64) * 100.0).min(100.0)
                    } else {
                        100.0
                    };
                    let bytes_per_sec = if elapsed_secs > 0.0 {
                        bytes_read as f64 / elapsed_secs
                    } else {
                        1.0
                    };
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
                    if let Some(callback) = progress {
                        callback(bytes_read.min(total_bytes), total_bytes);
                    }
                }
            }
        }

        if !doc_batch.is_empty() {
            bytes_read += batch_bytes;
            encode_doc_batch(self, &mut doc_batch, &mut all_tokens, eos_id);
        }

        let elapsed_secs = start_time.elapsed().as_secs_f64();
        let mb_total = total_bytes as f64 / 1_048_576.0;
        let mb_per_sec = if elapsed_secs > 0.0 {
            mb_total / elapsed_secs
        } else {
            0.0
        };

        if let Some(callback) = progress {
            callback(total_bytes, total_bytes);
        }
        tui::print_tokenization_complete(mb_total, mb_per_sec, all_tokens.len(), elapsed_secs);
        Ok((all_tokens, bytes_read, duplicates))
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

/// Tokenize a Parquet file's text column straight into token ids, skipping any
/// intermediate `.txt` materialization. Returns `(tokens, documents, duplicates)`.
pub fn encode_parquet_streaming(
    parquet_path: &Path,
    tokenizer: &LlmTokenizer,
    max_tokens: Option<usize>,
    dedupe: bool,
) -> Result<(Vec<u32>, u64, u64), Box<dyn std::error::Error + Send + Sync>> {
    println!("Tokenizing text column from Parquet {:?}...", parquet_path);

    let mut stream = DocumentStream::open(parquet_path)?;
    let eos_id = tokenizer.end_of_text_id();
    let limit = max_tokens.filter(|l| *l > 0);
    let mut tokens: Vec<u32> = Vec::new();
    let mut docs = 0u64;
    let mut duplicates = 0u64;
    let mut seen: HashSet<u64> = HashSet::new();
    let mut batch: Vec<String> = Vec::with_capacity(1024);
    let start_time = Instant::now();
    let mut last_report = Instant::now();
    let mut limit_hit = false;

    while let Some(doc) = stream.next_document()? {
        if dedupe && !seen.insert(hash_document(&doc.text)) {
            duplicates += 1;
            continue;
        }
        batch.push(doc.text);
        docs += 1;
        if batch.len() >= 1024 {
            encode_doc_batch(tokenizer, &mut batch, &mut tokens, eos_id);
            if let Some(l) = limit
                && tokens.len() >= l
            {
                limit_hit = true;
                break;
            }
        }

        if last_report.elapsed().as_millis() >= 500 {
            last_report = Instant::now();
            print!(
                "\r  {docs} documents, {} duplicate(s), {} tokens...",
                duplicates,
                tokens.len()
            );
            let _ = std::io::stdout().flush();
        }
    }

    if !limit_hit {
        encode_doc_batch(tokenizer, &mut batch, &mut tokens, eos_id);
    }
    if let Some(l) = limit {
        tokens.truncate(l);
    }

    println!(
        "\nParquet tokenization done: {docs} documents, {duplicates} duplicate(s), {} tokens in {:.1}s.",
        tokens.len(),
        start_time.elapsed().as_secs_f64()
    );
    Ok((tokens, docs, duplicates))
}

pub fn train_custom_tokenizer(
    corpus_path: &str,
    output_json_path: &str,
    vocab_size: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!(
        "Training custom BPE tokenizer on '{}' (vocab size: {})...",
        corpus_path, vocab_size
    );

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

    tokenizer_impl
        .train_from_files(&mut trainer, vec![corpus_path.to_string()])
        .map_err(|e| format!("Tokenizer training failed: {}", e))?;

    let tokenizer: Tokenizer = tokenizer_impl.into();
    tokenizer
        .save(output_json_path, true)
        .map_err(|e| format!("Saving tokenizer failed: {}", e))?;

    println!(
        "Successfully trained custom tokenizer -> saved to '{}'",
        output_json_path
    );
    Ok(())
}
