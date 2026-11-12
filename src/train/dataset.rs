//! Tokenized training dataset with sliding-window sampling.

use std::path::Path;

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::data::LlmTokenizer;
use crate::tui;

#[derive(Clone)]
pub struct TextDataset {
    pub tokens: Vec<u32>,
    pub n_ctx: usize,
    pub step: usize,
    /// Source text size (bytes) for the kept tokens; 0 when unknown (`.bin`
    /// inputs). Powers the bits-per-byte metric.
    pub total_bytes: u64,
}

impl TextDataset {
    pub fn split(self, val_ratio: f32) -> (Self, Self) {
        let val_ratio = val_ratio.clamp(0.0, 0.9);
        if val_ratio <= 0.0 || self.tokens.len() < (self.n_ctx + 1) * 2 {
            return (
                self.clone(),
                Self { tokens: vec![], n_ctx: self.n_ctx, step: self.step, total_bytes: 0 },
            );
        }

        let split_idx = ((self.tokens.len() as f32) * (1.0 - val_ratio)) as usize;
        let train_tokens = self.tokens[..split_idx].to_vec();
        let val_tokens = self.tokens[split_idx..].to_vec();

        // Split the byte count proportionally to the token split.
        let train_bytes = if self.tokens.is_empty() {
            0
        } else {
            (self.total_bytes as f64 * (split_idx as f64 / self.tokens.len() as f64)) as u64
        };

        (
            Self { tokens: train_tokens, n_ctx: self.n_ctx, step: self.step, total_bytes: train_bytes },
            Self {
                tokens: val_tokens,
                n_ctx: self.n_ctx,
                step: self.step,
                total_bytes: self.total_bytes.saturating_sub(train_bytes),
            },
        )
    }

    pub fn from_file<P: AsRef<Path>>(
        path: P,
        tokenizer: &LlmTokenizer,
        n_ctx: usize,
        custom_step: Option<usize>,
        max_tokens: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::from_file_with_progress(path, tokenizer, n_ctx, custom_step, max_tokens, None, None)
    }

    pub fn from_file_with_progress<P: AsRef<Path>>(
        path: P,
        tokenizer: &LlmTokenizer,
        n_ctx: usize,
        custom_step: Option<usize>,
        max_tokens: Option<usize>,
        progress: Option<&dyn Fn(u64, u64)>,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let (mut tokens, mut total_bytes) =
            tokenizer.encode_file_streaming_with_progress(path, max_tokens, progress, cancel)?;
        if let Some(limit) = max_tokens
            && limit > 0 && tokens.len() > limit {
                tui::print_warning(&format!("Limiting training dataset to {} tokens.", tui::format_number(limit)));
                // Keep the byte count in proportion to the kept tokens.
                total_bytes = (total_bytes as f64 * (limit as f64 / tokens.len() as f64)) as u64;
                tokens.truncate(limit);
            }
        let step = custom_step.unwrap_or(n_ctx);
        let step = std::cmp::max(1, step);
        Ok(Self { tokens, n_ctx, step, total_bytes })
    }

    /// Build a dataset from several weighted sources, mixing complete documents
    /// in memory (no intermediate `combined_corpus.txt` on disk). Each source
    /// may be `.parquet`, `.jsonl`, `.json`, or `.txt`.
    pub fn from_sources_with_progress(
        sources: &[(std::path::PathBuf, f32)],
        tokenizer: &LlmTokenizer,
        n_ctx: usize,
        custom_step: Option<usize>,
        max_tokens: Option<usize>,
        progress: Option<&dyn Fn(u64, u64)>,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let (mut tokens, mut total_bytes) = tokenizer
            .encode_sources_streaming_with_progress(sources, max_tokens, None, progress, cancel)?;
        if let Some(limit) = max_tokens
            && limit > 0
            && tokens.len() > limit
        {
            tui::print_warning(&format!(
                "Limiting training dataset to {} tokens.",
                tui::format_number(limit)
            ));
            total_bytes = (total_bytes as f64 * (limit as f64 / tokens.len() as f64)) as u64;
            tokens.truncate(limit);
        }
        let step = custom_step.unwrap_or(n_ctx);
        let step = std::cmp::max(1, step);
        Ok(Self { tokens, n_ctx, step, total_bytes })
    }

    #[allow(dead_code)]
    pub fn from_raw_text(
        text: &str,
        tokenizer: &LlmTokenizer,
        n_ctx: usize,
        custom_step: Option<usize>,
        max_tokens: Option<usize>,
    ) -> Self {
        let mut tokens = tokenizer.encode(text).unwrap_or_default();
        let mut total_bytes = text.len() as u64;
        if let Some(limit) = max_tokens
            && limit > 0 && tokens.len() > limit {
                total_bytes = (total_bytes as f64 * (limit as f64 / tokens.len() as f64)) as u64;
                tokens.truncate(limit);
            }
        let step = custom_step.unwrap_or(n_ctx);
        let step = std::cmp::max(1, step);
        Self { tokens, n_ctx, step, total_bytes }
    }

    pub fn len(&self) -> usize {
        let chunk_size = self.n_ctx + 1;
        if self.tokens.len() < chunk_size {
            0
        } else {
            (self.tokens.len() - chunk_size) / self.step + 1
        }
    }

    pub fn get_batch(
        &self,
        indices: &[usize],
        n_ctx: usize,
        device: &Device,
    ) -> (Tensor<2, Int>, Tensor<2, Int>) {
        let batch_size = indices.len();
        let chunk_size = n_ctx + 1;

        let mut inputs_flat: Vec<i32> = Vec::with_capacity(batch_size * n_ctx);
        let mut targets_flat: Vec<i32> = Vec::with_capacity(batch_size * n_ctx);

        for &idx in indices {
            let start = idx * self.step;
            debug_assert!(
                start + chunk_size <= self.tokens.len(),
                "get_batch index out of bounds: start={}, chunk_size={}, tokens_len={}",
                start,
                chunk_size,
                self.tokens.len()
            );

            let sample = &self.tokens[start..start + chunk_size];
            inputs_flat.extend(sample[0..n_ctx].iter().map(|&v| v as i32));
            targets_flat.extend(sample[1..n_ctx + 1].iter().map(|&v| v as i32));
        }

        let inputs_data = TensorData::new(inputs_flat, [batch_size, n_ctx]);
        let targets_data = TensorData::new(targets_flat, [batch_size, n_ctx]);

        let inputs = Tensor::<2, Int>::from_data(inputs_data, device);
        let targets = Tensor::<2, Int>::from_data(targets_data, device);

        (inputs, targets)
    }
}
