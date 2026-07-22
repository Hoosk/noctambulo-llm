use crate::config::ModelConfig;
use crate::data::LlmTokenizer;
use crate::generate::generate;
use crate::model::LlmConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};



use std::io::Write;
use std::path::Path;
use std::time::Instant;

pub struct TextDataset {
    tokens: Vec<u32>,
    n_ctx: usize,
    step: usize,
}

impl TextDataset {
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        tokenizer: &LlmTokenizer,
        n_ctx: usize,
        custom_step: Option<usize>,
        max_tokens: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut tokens = tokenizer.encode_file_streaming(path)?;
        if let Some(limit) = max_tokens {
            if limit > 0 && tokens.len() > limit {
                println!("Limiting training dataset to {} tokens (out of {} total loaded).", limit, tokens.len());
                tokens.truncate(limit);
            }
        }
        let step = custom_step.unwrap_or(n_ctx);
        let step = std::cmp::max(1, step);
        Ok(Self { tokens, n_ctx, step })
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
        if let Some(limit) = max_tokens {
            if limit > 0 && tokens.len() > limit {
                tokens.truncate(limit);
            }
        }
        let step = custom_step.unwrap_or(n_ctx);
        let step = std::cmp::max(1, step);
        Self { tokens, n_ctx, step }
    }


    pub fn len(&self) -> usize {
        let chunk_size = self.n_ctx + 1;
        if self.tokens.len() < chunk_size {
            0
        } else {
            (self.tokens.len() - chunk_size) / self.step + 1
        }
    }

    pub fn get_batch<B: AutodiffBackend>(
        &self,
        indices: &[usize],
        n_ctx: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
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

        let inputs = Tensor::<B, 2, Int>::from_data(inputs_data, device);
        let targets = Tensor::<B, 2, Int>::from_data(targets_data, device);

        (inputs, targets)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn train<B: AutodiffBackend>(
    dataset_path: Option<&Path>,
    raw_text: Option<&str>,
    model_config: ModelConfig,
    tokenizer: &LlmTokenizer,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    custom_step: Option<usize>,
    max_tokens: Option<usize>,
    device: &B::Device,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dataset = if let Some(path) = dataset_path {
        TextDataset::from_file(path, tokenizer, model_config.n_ctx, custom_step, max_tokens)?
    } else if let Some(text) = raw_text {
        TextDataset::from_raw_text(text, tokenizer, model_config.n_ctx, custom_step, max_tokens)
    } else {
        return Err("No dataset path or raw text provided for training.".into());
    };

    if dataset.len() == 0 {
        return Err("Dataset contains 0 valid sequence samples for the current context length.".into());
    }

    println!("Dataset prepared: {} total training sequences.", dataset.len());

    let llm_config = LlmConfig {
        vocab_size: model_config.vocab_size,
        n_embd: model_config.n_embd,
        n_head: model_config.n_head,
        n_layer: model_config.n_layer,
        n_ctx: model_config.n_ctx,
    };

    println!("Initializing model architecture on WGPU device...");
    let mut model = llm_config.init::<B>(device);
    let mut optim = AdamWConfig::new().init();

    let num_batches = dataset.len() / batch_size;
    let total_steps = epochs * num_batches;
    let warmup_steps = std::cmp::min(1000, total_steps / 10);
    let min_lr = lr * 0.1;

    println!("Starting 2026 LLM training loop: {} epochs, {} batches/epoch (Total steps: {})...", epochs, num_batches, total_steps);
    println!("===============================================================================");

    let mut global_step = 0usize;

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        let mut accum_loss_tensor = Tensor::<B, 1>::zeros([1], device);
        let mut accum_count = 0usize;
        let mut total_epoch_loss = 0.0f32;
        let mut total_batches = 0usize;
        let mut last_live_update = Instant::now();

        let log_interval = std::cmp::max(1, num_batches / 10);

        for batch_idx in 0..num_batches {
            global_step += 1;

            // Cosine Learning Rate Schedule with Warmup
            let current_lr = if global_step <= warmup_steps {
                lr * (global_step as f64 / warmup_steps as f64)
            } else {
                let progress = (global_step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
                let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                min_lr + (lr - min_lr) * cosine_decay
            };

            let start = batch_idx * batch_size;
            let end = start + batch_size;
            let indices: Vec<usize> = (start..end).collect();

            let (inputs, targets) = dataset.get_batch::<B>(&indices, model_config.n_ctx, device);
            let logits = model.forward(inputs);
            let [b, t, v] = logits.dims();

            let logits_flat = logits.reshape([b * t, v]);
            let log_probs = burn::tensor::activation::log_softmax(logits_flat, 1);
            let targets_flat = targets.reshape([b * t, 1]);
            let target_log_probs = log_probs.gather(1, targets_flat);
            let loss = target_log_probs.mean().neg();

            // Accumulate loss ON GPU detached from autograd graph
            accum_loss_tensor = accum_loss_tensor + loss.clone().detach();
            accum_count += 1;

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(current_lr, model, grads_params);

            // Live progress line
            if last_live_update.elapsed().as_millis() >= 500 {
                last_live_update = Instant::now();
                let elapsed = epoch_start.elapsed().as_secs_f64();
                let done = (batch_idx + 1) as f64;
                let total = num_batches as f64;
                let pct = done / total * 100.0;
                let batches_per_sec = if elapsed > 0.0 { done / elapsed } else { 0.0 };
                let remaining = total - done;
                let eta_secs = if batches_per_sec > 0.0 { (remaining / batches_per_sec) as u64 } else { 0 };
                let eta_min = eta_secs / 60;
                let eta_s = eta_secs % 60;
                let elapsed_min = elapsed as u64 / 60;
                let elapsed_s = elapsed as u64 % 60;

                print!(
                    "\r  [Epoch {}/{}] {:5.1}% | Batch {}/{} | {:.0} batch/s | LR: {:.2e} | Elapsed: {:02}m {:02}s | ETA: {:02}m {:02}s   ",
                    epoch, epochs, pct, batch_idx + 1, num_batches, batches_per_sec, current_lr, elapsed_min, elapsed_s, eta_min, eta_s
                );
                let _ = std::io::stdout().flush();
            }

            // Full GPU-synced log line at intervals
            let is_log_step = (batch_idx + 1) % log_interval == 0 || batch_idx == num_batches - 1;
            if is_log_step {
                let interval_loss = accum_loss_tensor
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default()
                    .first()
                    .copied()
                    .unwrap_or(0.0);

                total_epoch_loss += interval_loss;
                total_batches += accum_count;

                let interval_avg = interval_loss / (accum_count as f32);
                let current_epoch_avg = total_epoch_loss / (total_batches as f32);
                let progress = ((batch_idx + 1) as f32 / num_batches as f32) * 100.0;

                println!(
                    "\r  [Epoch {:2}/{:2}] Batch {:6}/{:6} ({:5.1}%) | Step Loss: {:.4} | Avg Loss: {:.4} | LR: {:.2e}        ",
                    epoch, epochs, batch_idx + 1, num_batches, progress, interval_avg, current_epoch_avg, current_lr
                );

                // Reset GPU loss accumulator for next interval
                accum_loss_tensor = Tensor::<B, 1>::zeros([1], device);
                accum_count = 0;
            }
        }

        let elapsed = epoch_start.elapsed().as_secs_f32();
        let final_epoch_avg = if total_batches > 0 { total_epoch_loss / total_batches as f32 } else { 0.0 };
        println!(
            "-------------------------------------------------------------------------------"
        );
        println!("🚀 Epoch {}/{} Complete | Avg Loss: {:.4} | Time: {:.2}s", epoch, epochs, final_epoch_avg, elapsed);

        if let Ok(sample) = generate::<B::InnerBackend>(
            &model.valid(),
            tokenizer,
            "Once upon a time",
            25,
            0.7,
            40,
            0.9,
            1.1,
            42 + epoch as u64,
            device,
        ) {
            println!("  ✨ Story Generation Preview: \"{}\"", sample.replace("\n", " "));
        }
        println!(
            "==============================================================================="
        );
    }


    println!("Training finished successfully!");

    std::fs::create_dir_all(output_dir)?;
    let config_path = Path::new(output_dir).join("config.json");
    let config_json = serde_json::to_string_pretty(&model_config)?;
    std::fs::write(&config_path, config_json)?;

    let weights_path = Path::new(output_dir).join("model_weights");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(model.into_record(), weights_path.clone())
        .map_err(|e| format!("Failed to save model record: {:?}", e))?;

    println!("Saved config to {:?}", config_path);
    println!("Saved model weights to {:?}", weights_path.with_extension("bin"));

    Ok(())
}
