use crate::config::ModelConfig;
use crate::data::LlmTokenizer;
use crate::generate::{generate, XorshiftRng};
use crate::tui;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};
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
        let mut tokens = tokenizer.encode_file_streaming(path, max_tokens)?;
        if let Some(limit) = max_tokens {
            if limit > 0 && tokens.len() > limit {
                tui::print_warning(&format!("Limiting training dataset to {} tokens.", tui::format_number(limit)));
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

fn shuffle_indices(indices: &mut [usize], rng: &mut XorshiftRng) {
    let len = indices.len();
    for i in (1..len).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize % (i + 1);
        indices.swap(i, j);
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

    tui::print_info("Initializing model architecture on WGPU device...");
    let mut model = model_config.init_model::<B>(device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.1f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let effective_batch_size = std::cmp::min(batch_size, dataset.len());
    let num_batches = dataset.len().div_ceil(effective_batch_size);
    let total_steps = epochs * num_batches;
    let warmup_steps = if total_steps <= 10 { 0 } else { std::cmp::min(1000, total_steps / 10) };
    let min_lr = lr * 0.1;

    tui::print_training_header(
        &model_config,
        dataset.len(),
        dataset.tokens.len(),
        epochs,
        num_batches,
        effective_batch_size,
        lr,
    );

    let mut global_step = 0usize;
    let mut shuffle_rng = XorshiftRng::new(42);
    let mut last_loss = 0.0f32;
    let mut last_avg_loss = 0.0f32;
    let training_start = Instant::now();

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        let mut accum_loss_tensor = Tensor::<B, 1>::zeros([1], device);
        let mut accum_count = 0usize;
        let mut total_epoch_loss = 0.0f32;
        let mut total_batches = 0usize;
        let mut last_live_update = Instant::now();

        let log_interval = std::cmp::max(1, num_batches / 10);

        // Shuffle sequence sample indices per epoch
        let mut sample_indices: Vec<usize> = (0..dataset.len()).collect();
        shuffle_indices(&mut sample_indices, &mut shuffle_rng);

        for batch_idx in 0..num_batches {
            global_step += 1;

            // Cosine Learning Rate Schedule with Warmup
            let current_lr = if warmup_steps > 0 && global_step <= warmup_steps {
                lr * (global_step as f64 / warmup_steps as f64)
            } else if total_steps > warmup_steps {
                let progress = (global_step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
                let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                min_lr + (lr - min_lr) * cosine_decay
            } else {
                lr
            };

            let start = batch_idx * effective_batch_size;
            let end = std::cmp::min(start + effective_batch_size, dataset.len());
            let indices: Vec<usize> = sample_indices[start..end].to_vec();

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

                last_loss = interval_loss / (accum_count as f32);
                last_avg_loss = total_epoch_loss / (total_batches as f32);

                // Reset GPU loss accumulator for next interval
                accum_loss_tensor = Tensor::<B, 1>::zeros([1], device);
                accum_count = 0;
            }

            // Live progress line
            if last_live_update.elapsed().as_millis() >= 500 || is_log_step {
                last_live_update = Instant::now();
                let total_elapsed = training_start.elapsed().as_secs_f64();
                let done_steps = (epoch - 1) * num_batches + batch_idx + 1;
                let pct = (done_steps as f64 / total_steps as f64) * 100.0;
                let batches_per_sec = if total_elapsed > 0.0 { done_steps as f64 / total_elapsed } else { 0.0 };
                let remaining_steps = total_steps.saturating_sub(done_steps);
                let eta_secs = if batches_per_sec > 0.0 { (remaining_steps as f64 / batches_per_sec) as u64 } else { 0 };
                let eta_min = eta_secs / 60;
                let eta_s = eta_secs % 60;
                let elapsed_min = total_elapsed as u64 / 60;
                let elapsed_s = total_elapsed as u64 % 60;

                tui::print_training_progress(
                    epoch,
                    epochs,
                    pct,
                    batch_idx + 1,
                    num_batches,
                    batches_per_sec,
                    current_lr,
                    last_loss,
                    last_avg_loss,
                    elapsed_min,
                    elapsed_s,
                    eta_min,
                    eta_s,
                );
            }
        }
        println!();

        let elapsed = epoch_start.elapsed().as_secs_f32();
        let final_epoch_avg = if total_batches > 0 { total_epoch_loss / total_batches as f32 } else { 0.0 };

        let preview = generate::<B::InnerBackend>(
            &model.valid(),
            tokenizer,
            "Once upon a time",
            25,
            0.7,
            40,
            0.9,
            0.05,
            1.1,
            42 + epoch as u64,
            device,
        ).ok();

        tui::print_epoch_complete(
            epoch,
            epochs,
            final_epoch_avg,
            elapsed,
            preview.as_deref(),
        );
    }

    std::fs::create_dir_all(output_dir)?;
    let config_path = Path::new(output_dir).join("config.json");
    let config_json = serde_json::to_string_pretty(&model_config)?;
    std::fs::write(&config_path, config_json)?;

    let weights_path = Path::new(output_dir).join("model_weights");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(model.into_record(), weights_path.clone())
        .map_err(|e| format!("Failed to save model record: {:?}", e))?;

    tui::print_training_complete(&config_path, &weights_path.with_extension("bin"));

    Ok(())
}
