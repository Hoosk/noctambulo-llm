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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::time::Instant;

#[derive(Clone)]
pub struct TextDataset {
    pub tokens: Vec<u32>,
    pub n_ctx: usize,
    pub step: usize,
}

impl TextDataset {
    pub fn split(self, val_ratio: f32) -> (Self, Self) {
        let val_ratio = val_ratio.clamp(0.0, 0.9);
        if val_ratio <= 0.0 || self.tokens.len() < (self.n_ctx + 1) * 2 {
            return (self.clone(), Self { tokens: vec![], n_ctx: self.n_ctx, step: self.step });
        }

        let split_idx = ((self.tokens.len() as f32) * (1.0 - val_ratio)) as usize;
        let train_tokens = self.tokens[..split_idx].to_vec();
        let val_tokens = self.tokens[split_idx..].to_vec();

        (
            Self { tokens: train_tokens, n_ctx: self.n_ctx, step: self.step },
            Self { tokens: val_tokens, n_ctx: self.n_ctx, step: self.step },
        )
    }

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

    pub fn get_batch<B: burn::tensor::backend::Backend>(
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

/// Event emitted by the training loop, consumed either by the CLI (prints) or
/// by the web server (SSE / live metrics).
#[derive(Debug, Clone, serde::Serialize)]
#[allow(clippy::too_many_arguments, dead_code)]
pub enum TrainingEvent {
    Started {
        epochs: usize,
        total_steps: usize,
        num_batches: usize,
        batch_size: usize,
        lr: f64,
        n_ctx: usize,
        start_time_unix: u64,
    },
    Progress {
        step: usize,
        epoch: usize,
        epochs: usize,
        batch: usize,
        num_batches: usize,
        pct: f64,
        loss: f32,
        avg_loss: f32,
        lr: f64,
        batches_per_sec: f64,
        elapsed_secs: f64,
        eta_secs: f64,
        current_n_ctx: usize,
    },
    Val {
        loss: f32,
        ppl: f32,
        epoch: usize,
    },
    Checkpoint {
        path: String,
        kind: &'static str,
        epoch: usize,
        batch: usize,
    },
    Preview {
        text: String,
        epoch: usize,
    },
    EpochComplete {
        epoch: usize,
        epochs: usize,
        avg_loss: f32,
        elapsed_secs: f32,
    },
    Done {
        final_loss: f32,
        total_time_secs: f64,
        weights_path: String,
        config_path: String,
    },
    Stopped {
        reason: String,
        epoch: usize,
        batch: usize,
    },
}

fn shuffle_indices(indices: &mut [usize], rng: &mut XorshiftRng) {
    let len = indices.len();
    for i in (1..len).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize % (i + 1);
        indices.swap(i, j);
    }
}

pub fn compute_scheduled_n_ctx(
    global_step: usize,
    warmup_steps: usize,
    target_n_ctx: usize,
) -> usize {
    if warmup_steps == 0 || global_step >= warmup_steps {
        return target_n_ctx;
    }

    let min_ctx = std::cmp::max(32, target_n_ctx / 4);
    if min_ctx >= target_n_ctx {
        return target_n_ctx;
    }

    let progress = (global_step as f32 / warmup_steps as f32).clamp(0.0, 1.0);
    let scheduled = (min_ctx as f32 + progress * (target_n_ctx - min_ctx) as f32) as usize;

    let aligned = (scheduled / 16) * 16;
    aligned.clamp(min_ctx, target_n_ctx)
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
    val_split: f32,
    grad_accum_steps: usize,
    seq_warmup_steps: usize,
    save_dir: Option<&Path>,
    resume_path: Option<&Path>,
    device: &B::Device,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_thread = cancel.clone();
    let handle = std::thread::spawn(move || {
        for event in rx {
            match &event {
                TrainingEvent::Progress { loss, avg_loss, lr, pct, batch, num_batches, epoch, epochs, batches_per_sec, elapsed_secs, eta_secs, .. } => {
                    let elapsed_min = *elapsed_secs as u64 / 60;
                    let elapsed_s = *elapsed_secs as u64 % 60;
                    let eta_min = *eta_secs as u64 / 60;
                    let eta_s = *eta_secs as u64 % 60;
                    tui::print_training_progress(
                        *epoch, *epochs, *pct, *batch, *num_batches, *batches_per_sec, *lr,
                        *loss, *avg_loss, elapsed_min, elapsed_s, eta_min, eta_s,
                    );
                }
                TrainingEvent::Val { loss, ppl, .. } => {
                    println!("\n   └── Validation Loss: {:.4} | Validation Perplexity (PPL): {:.4}", loss, ppl);
                }
                TrainingEvent::Checkpoint { path, kind, .. } => {
                    if *kind == "best" {
                        println!("   ★ New best val_loss → {:?}", path);
                    } else {
                        println!("Saved checkpoint: {:?}", path);
                    }
                }
                TrainingEvent::Preview { text, .. } => {
                    println!("  ✨ \"{}\"", text);
                }
                TrainingEvent::EpochComplete { epoch, epochs, avg_loss, elapsed_secs } => {
                    tui::print_epoch_complete(*epoch, *epochs, *avg_loss, *elapsed_secs, None);
                }
                TrainingEvent::Done { weights_path, config_path, .. } => {
                    tui::print_training_complete(Path::new(config_path), Path::new(weights_path));
                }
                TrainingEvent::Started { .. } => {}
                TrainingEvent::Stopped { reason, .. } => {
                    println!("\n  ⏹ Training stopped: {}", reason);
                }
            }
            let _ = event;
        }
    });

    let result = train_with_events::<B>(
        dataset_path, raw_text, model_config, tokenizer, epochs, batch_size, lr,
        custom_step, max_tokens, val_split, grad_accum_steps, seq_warmup_steps,
        save_dir, resume_path, device, output_dir, &tx, &cancel_thread, None,
    );

    drop(tx);
    let _ = handle.join();
    result
}

#[allow(clippy::too_many_arguments)]
pub fn train_with_events<B: AutodiffBackend>(
    dataset_path: Option<&Path>,
    raw_text: Option<&str>,
    model_config: ModelConfig,
    tokenizer: &LlmTokenizer,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    custom_step: Option<usize>,
    max_tokens: Option<usize>,
    val_split: f32,
    grad_accum_steps: usize,
    seq_warmup_steps: usize,
    save_dir: Option<&Path>,
    resume_path: Option<&Path>,
    device: &B::Device,
    output_dir: &str,
    tx: &Sender<TrainingEvent>,
    cancel: &Arc<AtomicBool>,
    pause: Option<&Arc<AtomicBool>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let full_dataset = if let Some(path) = dataset_path {
        TextDataset::from_file(path, tokenizer, model_config.n_ctx, custom_step, max_tokens)?
    } else if let Some(text) = raw_text {
        TextDataset::from_raw_text(text, tokenizer, model_config.n_ctx, custom_step, max_tokens)
    } else {
        return Err("No dataset path or raw text provided for training.".into());
    };

    if full_dataset.len() == 0 {
        return Err("Dataset contains 0 valid sequence samples for the current context length.".into());
    }

    let (train_dataset, val_dataset) = if val_split > 0.0 {
        let (tr, va) = full_dataset.split(val_split);
        println!("Dataset split: {} training samples, {} validation samples ({:.1}% val).", tr.len(), va.len(), val_split * 100.0);
        (tr, va)
    } else {
        (full_dataset, TextDataset { tokens: vec![], n_ctx: model_config.n_ctx, step: custom_step.unwrap_or(model_config.n_ctx) })
    };

    tui::print_info("Initializing model architecture on WGPU device...");
    let mut model = model_config.init_model::<B>(device);

    if let Some(res_path) = resume_path {
        let rec_file = if res_path.is_file() {
            res_path.with_extension("")
        } else {
            res_path.to_path_buf()
        };
        println!("Resuming model weights from {:?}...", rec_file);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(rec_file.clone(), device)
            .map_err(|e| format!("Failed to load checkpoint from {:?}: {:?}", res_path, e))?;
        model = model.load_record(record);
        println!("Successfully loaded checkpoint weights!");
    }

    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.1f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let effective_batch_size = std::cmp::min(batch_size, train_dataset.len());
    let num_batches = train_dataset.len().div_ceil(effective_batch_size);
    let total_steps = epochs * num_batches;
    let warmup_steps = if total_steps <= 10 { 0 } else { std::cmp::min(1000, total_steps / 10) };
    let min_lr = lr * 0.1;
    let grad_accum = grad_accum_steps.max(1);

    if seq_warmup_steps > 0 {
        println!("Sequence Length Warmup active: scaling context length over the first {} steps.", seq_warmup_steps);
    }

    tui::print_training_header(
        &model_config,
        train_dataset.len(),
        train_dataset.tokens.len(),
        epochs,
        num_batches,
        effective_batch_size,
        lr,
    );

    let _ = tx.send(TrainingEvent::Started {
        epochs,
        total_steps,
        num_batches,
        batch_size: effective_batch_size,
        lr,
        n_ctx: model_config.n_ctx,
        start_time_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    });

    let mut global_step = 0usize;
    let mut shuffle_rng = XorshiftRng::new(42);
    let mut last_loss = 0.0f32;
    let mut last_avg_loss = 0.0f32;
    let mut best_val_loss: Option<f32> = None;
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
        let mut sample_indices: Vec<usize> = (0..train_dataset.len()).collect();
        shuffle_indices(&mut sample_indices, &mut shuffle_rng);

        for batch_idx in 0..num_batches {
            if cancel.load(Ordering::Relaxed) {
                let _ = tx.send(TrainingEvent::Stopped {
                    reason: "User requested stop".to_string(),
                    epoch,
                    batch: batch_idx,
                });
                return Ok(());
            }

            // Pause: block the loop until resumed (or cancelled).
            if let Some(pause_flag) = pause {
                while pause_flag.load(Ordering::Relaxed) {
                    if cancel.load(Ordering::Relaxed) {
                        let _ = tx.send(TrainingEvent::Stopped {
                            reason: "User requested stop while paused".to_string(),
                            epoch,
                            batch: batch_idx,
                        });
                        return Ok(());
                    }
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            }

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

            let current_n_ctx = compute_scheduled_n_ctx(global_step, seq_warmup_steps, model_config.n_ctx);

            let start = batch_idx * effective_batch_size;
            let end = std::cmp::min(start + effective_batch_size, train_dataset.len());
            let indices: Vec<usize> = sample_indices[start..end].to_vec();

            let (inputs, targets) = train_dataset.get_batch::<B>(&indices, current_n_ctx, device);
            let logits = model.forward(inputs);
            let [b, t, v] = logits.dims();

            let logits_flat = logits.reshape([b * t, v]);
            let log_probs = burn::tensor::activation::log_softmax(logits_flat, 1);
            let targets_flat = targets.reshape([b * t, 1]);
            let target_log_probs = log_probs.gather(1, targets_flat);
            let mut loss = target_log_probs.mean().neg();

            if grad_accum > 1 {
                loss = loss / (grad_accum as f32);
            }

            // Accumulate loss ON GPU detached from autograd graph
            accum_loss_tensor = accum_loss_tensor + loss.clone().detach();
            accum_count += 1;

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);

            if (batch_idx + 1) % grad_accum == 0 || batch_idx == num_batches - 1 {
                model = optim.step(current_lr, model, grads_params);
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

                let _ = tx.send(TrainingEvent::Progress {
                    step: global_step,
                    epoch,
                    epochs,
                    batch: batch_idx + 1,
                    num_batches,
                    pct,
                    loss: last_loss,
                    avg_loss: last_avg_loss,
                    lr: current_lr,
                    batches_per_sec,
                    elapsed_secs: total_elapsed,
                    eta_secs: eta_secs as f64,
                    current_n_ctx,
                });
            }
        }
        println!();

        let elapsed = epoch_start.elapsed().as_secs_f32();
        let final_epoch_avg = if total_batches > 0 { total_epoch_loss / total_batches as f32 } else { 0.0 };

        if val_dataset.len() > 0 {
            let (val_loss, val_ppl) = evaluate_dataset(&model.valid(), &val_dataset, effective_batch_size, device);
            println!("   └── Validation Loss: {:.4} | Validation Perplexity (PPL): {:.4}", val_loss, val_ppl);
            let _ = tx.send(TrainingEvent::Val { loss: val_loss, ppl: val_ppl, epoch });

            let is_best = match best_val_loss {
                None => true,
                Some(prev) => val_loss < prev,
            };
            if is_best {
                best_val_loss = Some(val_loss);
                if let Some(s_dir) = save_dir {
                    let best_path = s_dir.join("checkpoint_best");
                    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
                    if let Err(e) = recorder.record(model.clone().into_record(), best_path.clone()) {
                        println!("Warning: Failed to save best checkpoint at {:?}: {:?}", best_path, e);
                    } else {
                        println!("   ★ New best val_loss={:.4} → {:?}", val_loss, best_path.with_extension("bin"));
                        let _ = tx.send(TrainingEvent::Checkpoint {
                            path: best_path.with_extension("bin").to_string_lossy().to_string(),
                            kind: "best",
                            epoch,
                            batch: 0,
                        });
                    }
                }
            }
        }

        if let Some(s_dir) = save_dir {
            std::fs::create_dir_all(s_dir)?;
            let ckpt_path = s_dir.join(format!("checkpoint_epoch_{}", epoch));
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            if let Err(e) = recorder.record(model.clone().into_record(), ckpt_path.clone()) {
                println!("Warning: Failed to save checkpoint at {:?}: {:?}", ckpt_path, e);
            } else {
                println!("Saved checkpoint: {:?}", ckpt_path.with_extension("bin"));
                let _ = tx.send(TrainingEvent::Checkpoint {
                    path: ckpt_path.with_extension("bin").to_string_lossy().to_string(),
                    kind: "epoch",
                    epoch,
                    batch: 0,
                });
            }
        }

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

        if let Some(text) = &preview {
            let _ = tx.send(TrainingEvent::Preview { text: text.clone(), epoch });
        }

        tui::print_epoch_complete(
            epoch,
            epochs,
            final_epoch_avg,
            elapsed,
            preview.as_deref(),
        );
        let _ = tx.send(TrainingEvent::EpochComplete {
            epoch,
            epochs,
            avg_loss: final_epoch_avg,
            elapsed_secs: elapsed,
        });
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

    let _ = tx.send(TrainingEvent::Done {
        final_loss: last_avg_loss,
        total_time_secs: training_start.elapsed().as_secs_f64(),
        weights_path: weights_path.with_extension("bin").to_string_lossy().to_string(),
        config_path: config_path.to_string_lossy().to_string(),
    });

    Ok(())
}

pub fn evaluate_dataset<B: burn::tensor::backend::Backend>(
    model: &crate::model::Llm<B>,
    dataset: &TextDataset,
    batch_size: usize,
    device: &B::Device,
) -> (f32, f32) {
    if dataset.len() == 0 {
        return (0.0, 1.0);
    }
    let total_batches = dataset.len().div_ceil(batch_size);
    let eval_batches = total_batches.min(100);
    let mut total_loss = 0.0f32;

    for batch_idx in 0..eval_batches {
        let start = batch_idx * batch_size;
        let end = std::cmp::min(start + batch_size, dataset.len());
        let indices: Vec<usize> = (start..end).collect();

        let (inputs, targets) = dataset.get_batch::<B>(&indices, model.n_ctx, device);
        let logits = model.forward(inputs);
        let [b, t, v] = logits.dims();

        let logits_flat = logits.reshape([b * t, v]);
        let log_probs = burn::tensor::activation::log_softmax(logits_flat, 1);
        let targets_flat = targets.reshape([b * t, 1]);
        let target_log_probs = log_probs.gather(1, targets_flat);
        let loss = target_log_probs.mean().neg();

        let loss_val = loss
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.0);

        total_loss += loss_val;
    }

    let avg_loss = total_loss / (eval_batches as f32);
    let perplexity = avg_loss.exp();
    (avg_loss, perplexity)
}

pub fn run_eval<B: burn::tensor::backend::Backend>(
    model_dir: &Path,
    dataset_path: &Path,
    tokenizer: &LlmTokenizer,
    n_ctx: Option<usize>,
    eval_batch_size: usize,
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(format!("Model config not found at {:?}", config_path).into());
    }
    let config_str = std::fs::read_to_string(&config_path)?;
    let mut config: ModelConfig = serde_json::from_str(&config_str)?;

    if let Some(ctx) = n_ctx {
        config.n_ctx = ctx;
    }

    println!("Loading model config from {:?}...", config_path);
    let model: crate::model::Llm<B> = config.init_model(device);

    let weights_path = model_dir.join("model_weights.bin");
    let model = if weights_path.exists() {
        println!("Loading model weights from {:?}...", weights_path);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(model_dir.join("model_weights"), device)
            .map_err(|e| format!("Failed to load model record: {:?}", e))?;
        model.load_record(record)
    } else {
        println!("Warning: model_weights.bin not found in {:?}. Evaluating initialized model.", model_dir);
        model
    };

    let dataset = TextDataset::from_file(dataset_path, tokenizer, config.n_ctx, None, None)?;
    println!("Evaluating model on dataset {:?} ({} samples, {} tokens) [batch={}]...", dataset_path, dataset.len(), dataset.tokens.len(), eval_batch_size);

    let start_time = Instant::now();
    let (loss, ppl) = evaluate_dataset::<B>(&model, &dataset, eval_batch_size, device);
    let elapsed = start_time.elapsed().as_secs_f64();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    EVALUATION RESULTS                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Cross-Entropy Loss : {:<39.4} ║", loss);
    println!("║ Perplexity (PPL)   : {:<39.4} ║", ppl);
    println!("║ Evaluation Time    : {:<35.2}s ║", elapsed);
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_val_split() {
        let tokens: Vec<u32> = (0..1000).collect();
        let dataset = TextDataset { tokens, n_ctx: 32, step: 32 };

        let (train_ds, val_ds) = dataset.split(0.2);
        assert_eq!(train_ds.tokens.len(), 800);
        assert_eq!(val_ds.tokens.len(), 200);
    }

    #[test]
    fn test_perplexity_formula() {
        let loss = 2.302585f32; // ln(10)
        let ppl = loss.exp();
        assert!((ppl - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_progress_event_json_keys() {
        let ev = TrainingEvent::Progress {
            step: 1,
            epoch: 1,
            epochs: 3,
            batch: 10,
            num_batches: 100,
            pct: 10.0,
            loss: 6.5,
            avg_loss: 7.0,
            lr: 0.00005,
            batches_per_sec: 12.3,
            elapsed_secs: 30.0,
            eta_secs: 270.0,
            current_n_ctx: 256,
        };
        let json = serde_json::to_string(&ev).unwrap();
        println!("PROGRESS_JSON={json}");
        assert!(json.contains("\"loss\""));
        assert!(json.contains("\"avg_loss\""));
        assert!(json.contains("\"num_batches\""));
        assert!(json.contains("\"batches_per_sec\""));
        assert!(json.contains("\"elapsed_secs\""));
        assert!(json.contains("\"eta_secs\""));
        assert!(json.contains("\"current_n_ctx\""));
    }

    #[test]
    fn test_sequence_length_warmup() {
        let target_ctx = 256;
        let warmup_steps = 1000;

        let ctx_start = compute_scheduled_n_ctx(0, warmup_steps, target_ctx);
        let ctx_mid = compute_scheduled_n_ctx(500, warmup_steps, target_ctx);
        let ctx_end = compute_scheduled_n_ctx(1000, warmup_steps, target_ctx);
        let ctx_after = compute_scheduled_n_ctx(1500, warmup_steps, target_ctx);

        assert_eq!(ctx_start, 64);
        assert!(ctx_mid > ctx_start && ctx_mid < ctx_end);
        assert_eq!(ctx_end, target_ctx);
        assert_eq!(ctx_after, target_ctx);
    }
}

