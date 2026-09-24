use crate::config::ModelConfig;
use crate::data::LlmTokenizer;
use crate::generate::{generate, XorshiftRng};
use crate::model::Llm;
use crate::tui;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::time::Instant;

/// JSON sidecar file holding scheduler position, RNG and best validation loss.
pub const TRAIN_STATE_FILE: &str = "train_state.json";
/// Base name (the recorder appends `.bin`) of the serialized optimizer state.
pub const OPTIMIZER_STATE_FILE: &str = "optimizer_state";

/// Persistent training state so `--resume` continues the LR schedule, the
/// shuffler RNG and the AdamW moments instead of restarting them.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TrainState {
    pub global_step: usize,
    pub total_steps: usize,
    pub epochs_done: usize,
    pub best_val_loss: Option<f32>,
    /// Internal Xorshift state of the epoch shuffler (0 = default seed).
    pub rng_state: u64,
}

/// Directory holding the sidecar artifacts (`train_state.json`,
/// `optimizer_state.bin`) for the path passed to `--resume`.
fn resume_artifacts_dir(resume_path: &Path) -> PathBuf {
    if resume_path.is_dir() {
        return resume_path.to_path_buf();
    }
    resume_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn load_train_state(resume_path: Option<&Path>) -> Option<TrainState> {
    let path = resume_path?;
    let bytes = std::fs::read(resume_artifacts_dir(path).join(TRAIN_STATE_FILE)).ok()?;
    serde_json::from_slice::<TrainState>(&bytes).ok()
}

/// Persist the scheduler/RNG state plus the optimizer moments next to each
/// other so `--resume` can restore the full training state, not just weights.
fn persist_train_state<B, M, O>(
    optim: &O,
    state: &TrainState,
    dir: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    std::fs::create_dir_all(dir)?;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(optim.to_record(), dir.join(OPTIMIZER_STATE_FILE))
        .map_err(|e| format!("Failed to save optimizer state: {e:?}"))?;
    std::fs::write(dir.join(TRAIN_STATE_FILE), serde_json::to_string_pretty(state)?)?;
    Ok(())
}

/// Reload the AdamW moments from `optimizer_state.bin` when present. Returns
/// the optimizer and whether a record was actually applied.
fn load_optimizer_record<B, M, O>(optim: O, dir: &Path, device: &B::Device) -> (O, bool)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let path = dir.join(OPTIMIZER_STATE_FILE);
    if !path.with_extension("bin").is_file() {
        return (optim, false);
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    match recorder.load::<O::Record>(path, device) {
        Ok(record) => (optim.load_record(record), true),
        Err(_) => (optim, false),
    }
}

/// Save the model weights + full training state when training is interrupted
/// (user stop or Ctrl+C) so no work is lost.
fn save_interrupted_checkpoint<B: AutodiffBackend>(
    model: &Llm<B>,
    optim: &impl Optimizer<Llm<B>, B>,
    train_state: &TrainState,
    output_dir: &str,
    tx: &Sender<TrainingEvent>,
) {
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        emit_log(tx, format!("Warning: failed to save interrupted checkpoint: {e}"));
        return;
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let path = Path::new(output_dir).join("checkpoint_interrupted");
    match recorder.record(model.clone().into_record(), path.clone()) {
        Ok(()) => {
            let saved = path.with_extension("bin");
            emit_log(tx, format!("Training interrupted: checkpoint saved to {saved:?}"));
            let _ = tx.send(TrainingEvent::Checkpoint {
                path: saved.to_string_lossy().to_string(),
                kind: "interrupted".into(),
                epoch: train_state.epochs_done,
                batch: 0,
            });
            if let Err(e) = persist_train_state(optim, train_state, Path::new(output_dir)) {
                emit_log(tx, format!("Warning: failed to save training state: {e}"));
            }
        }
        Err(e) => emit_log(tx, format!("Warning: failed to save interrupted checkpoint: {e:?}")),
    }
}

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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
        kind: String,
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
    LogEvent {
        line: String,
        ts: u64,
    },
}

/// Print a diagnostic line to the CLI and forward it as a `LogEvent` so the
/// web UI can show it live (SSE + `/api/train/logs`).
fn emit_log(tx: &Sender<TrainingEvent>, line: impl std::fmt::Display) {
    println!("{line}");
    let _ = tx.send(TrainingEvent::LogEvent {
        line: line.to_string(),
        ts: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    });
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
    cancel: Option<Arc<AtomicBool>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
    let cancel = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
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
                    if kind == "best" {
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
                TrainingEvent::LogEvent { .. } => {}
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
        emit_log(tx, format!("Dataset split: {} training samples, {} validation samples ({:.1}% val).", tr.len(), va.len(), val_split * 100.0));
        (tr, va)
    } else {
        (full_dataset, TextDataset { tokens: vec![], n_ctx: model_config.n_ctx, step: custom_step.unwrap_or(model_config.n_ctx) })
    };

    tui::print_info("Initializing model architecture on WGPU device...");
    let mut model = model_config.init_model::<B>(device);

    let resume_state = load_train_state(resume_path);

    if let Some(res_path) = resume_path {
        let rec_file = if res_path.is_file() {
            res_path.with_extension("")
        } else {
            res_path.to_path_buf()
        };
        emit_log(tx, format!("Resuming model weights from {:?}...", rec_file));
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(rec_file.clone(), device)
            .map_err(|e| format!("Failed to load checkpoint from {:?}: {:?}", res_path, e))?;
        model = model.load_record(record);
        emit_log(tx, "Successfully loaded checkpoint weights!");
    }

    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.1f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // Restore AdamW moments + scheduler position when a sidecar state exists.
    if resume_state.is_some() {
        let dir = resume_artifacts_dir(resume_path.expect("resume_state implies a resume path"));
        let (restored, loaded) = load_optimizer_record::<B, Llm<B>, _>(optim, &dir, device);
        optim = restored;
        if loaded {
            emit_log(tx, "Successfully loaded optimizer state (AdamW moments)!");
        } else {
            emit_log(tx, "No optimizer state found next to the checkpoint; AdamW moments start fresh.");
        }
    }

    let mut train_state = resume_state.unwrap_or_default();
    if train_state.global_step > 0 || train_state.epochs_done > 0 {
        emit_log(tx, format!(
            "Resuming training state: step {}/{}, epochs done: {}.",
            train_state.global_step, train_state.total_steps, train_state.epochs_done
        ));
    }

    let effective_batch_size = std::cmp::min(batch_size, train_dataset.len());
    let num_batches = train_dataset.len().div_ceil(effective_batch_size);
    // `--epochs` on resume means *additional* epochs: the schedule continues
    // from the persisted step instead of restarting.
    let total_steps = train_state.global_step + epochs * num_batches;
    train_state.total_steps = total_steps;
    let warmup_steps = if total_steps <= 10 { 0 } else { std::cmp::min(1000, total_steps / 10) };
    let min_lr = lr * 0.1;
    let grad_accum = grad_accum_steps.max(1);

    if seq_warmup_steps > 0 {
        emit_log(tx, format!("Sequence Length Warmup active: scaling context length over the first {} steps.", seq_warmup_steps));
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

    let mut global_step = train_state.global_step;
    let run_step_baseline = global_step;
    let mut shuffle_rng = XorshiftRng::new(train_state.rng_state);
    let mut last_loss = 0.0f32;
    let mut last_avg_loss = 0.0f32;
    let mut best_val_loss: Option<f32> = train_state.best_val_loss;
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

        let mut grad_accumulator = GradientsAccumulator::<Llm<B>>::new();
        let mut window_start = 0usize;

        for batch_idx in 0..num_batches {
            if cancel.load(Ordering::Relaxed) {
                train_state.global_step = global_step;
                train_state.best_val_loss = best_val_loss;
                train_state.rng_state = shuffle_rng.state();
                save_interrupted_checkpoint(&model, &optim, &train_state, output_dir, tx);
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
                        train_state.global_step = global_step;
                        train_state.best_val_loss = best_val_loss;
                        train_state.rng_state = shuffle_rng.state();
                        save_interrupted_checkpoint(&model, &optim, &train_state, output_dir, tx);
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
            let loss = target_log_probs.mean().neg();

            // Metrics use the unscaled loss; gradients are scaled by the
            // micro-batch window size so the accumulated update equals the mean
            // gradient over the window (true gradient accumulation).
            accum_loss_tensor = accum_loss_tensor + loss.clone().detach();
            accum_count += 1;

            let window_len = std::cmp::min(grad_accum, num_batches - window_start);
            let scaled_loss = loss / (window_len as f32);
            let grads = GradientsParams::from_grads(scaled_loss.backward(), &model);
            grad_accumulator.accumulate(&model, grads);

            if batch_idx + 1 - window_start == window_len {
                let grads = grad_accumulator.grads();
                model = optim.step(current_lr, model, grads);
                window_start = batch_idx + 1;
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

                // Non-finite loss (divergence / high LR) must not poison the
                // running average or the persisted loss history.
                if interval_loss.is_finite() {
                    total_epoch_loss += interval_loss;
                    total_batches += accum_count;

                    last_loss = interval_loss / (accum_count as f32);
                    last_avg_loss = total_epoch_loss / (total_batches as f32);
                } else {
                    emit_log(tx, "Warning: non-finite loss detected, keeping previous value.");
                }

                // Reset GPU loss accumulator for next interval
                accum_loss_tensor = Tensor::<B, 1>::zeros([1], device);
                accum_count = 0;
            }

            // Live progress line
            if last_live_update.elapsed().as_millis() >= 500 || is_log_step {
                last_live_update = Instant::now();
                let total_elapsed = training_start.elapsed().as_secs_f64();
                let done_steps = global_step;
                let run_steps = global_step.saturating_sub(run_step_baseline).max(1);
                let pct = (done_steps as f64 / total_steps as f64) * 100.0;
                let batches_per_sec = if total_elapsed > 0.0 { run_steps as f64 / total_elapsed } else { 0.0 };
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

        // Keep the persisted training state in sync at the epoch boundary so a
        // later `--resume` continues the schedule instead of restarting it.
        train_state.global_step = global_step;
        train_state.epochs_done += 1;
        train_state.best_val_loss = best_val_loss;
        train_state.rng_state = shuffle_rng.state();

        if val_dataset.len() > 0 {
            let (val_loss, val_ppl) = evaluate_dataset(&model.valid(), &val_dataset, effective_batch_size, device);
            if !val_loss.is_finite() {
                emit_log(tx, "Warning: validation loss is non-finite; skipping validation this epoch.");
            } else {
                emit_log(tx, format!("   └── Validation Loss: {:.4} | Validation Perplexity (PPL): {:.4}", val_loss, val_ppl));
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
                            emit_log(tx, format!("Warning: Failed to save best checkpoint at {:?}: {:?}", best_path, e));
                        } else {
                            emit_log(tx, format!("   ★ New best val_loss={:.4} → {:?}", val_loss, best_path.with_extension("bin")));
                            let _ = tx.send(TrainingEvent::Checkpoint {
                                path: best_path.with_extension("bin").to_string_lossy().to_string(),
                                kind: "best".into(),
                                epoch,
                                batch: 0,
                            });
                        }
                    }
                }
            }
        }

        if let Some(s_dir) = save_dir {
            std::fs::create_dir_all(s_dir)?;
            let ckpt_path = s_dir.join(format!("checkpoint_epoch_{}", epoch));
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            if let Err(e) = recorder.record(model.clone().into_record(), ckpt_path.clone()) {
                emit_log(tx, format!("Warning: Failed to save checkpoint at {:?}: {:?}", ckpt_path, e));
            } else {
                emit_log(tx, format!("Saved checkpoint: {:?}", ckpt_path.with_extension("bin")));
                let _ = tx.send(TrainingEvent::Checkpoint {
                    path: ckpt_path.with_extension("bin").to_string_lossy().to_string(),
                    kind: "epoch".into(),
                    epoch,
                    batch: 0,
                });
            }
        }

        let state_dir = save_dir.unwrap_or_else(|| Path::new(output_dir));
        if let Err(e) = persist_train_state(&optim, &train_state, state_dir) {
            emit_log(tx, format!("Warning: failed to save training state: {e}"));
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

    train_state.global_step = global_step;
    train_state.best_val_loss = best_val_loss;
    train_state.rng_state = shuffle_rng.state();
    if let Err(e) = persist_train_state(&optim, &train_state, Path::new(output_dir)) {
        emit_log(tx, format!("Warning: failed to save training state: {e}"));
    }

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
    use burn::backend::Autodiff;
    use burn::tensor::backend::Backend;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use crate::model::Llm;
    use std::path::PathBuf;

    /// CPU-only, deterministic autodiff backend for end-to-end training tests.
    type TestBackend = Autodiff<NdArray<f32, i32>>;

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
        let loss = std::f32::consts::LN_10;
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

    #[test]
    fn test_log_event_serialization() {
        let ev = TrainingEvent::LogEvent {
            line: "Dataset split: 100 training samples".into(),
            ts: 1700000000,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"LogEvent\""));
        assert!(json.contains("\"line\""));
        assert!(json.contains("\"ts\""));
        assert!(json.contains("Dataset split"));
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kuro_neko_test_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_tokenizer_and_corpus(dir: &Path) -> (LlmTokenizer, String) {
        let corpus = "once upon a time there was a small black cat. the cat liked to sleep in the warm sun and chase little birds in the garden. one day the cat met a white dog and they became the best of friends. ".repeat(30);
        let corpus_path = dir.join("corpus.txt");
        std::fs::write(&corpus_path, &corpus).unwrap();
        let tokenizer_path = dir.join("tokenizer.json");
        crate::data::train_custom_tokenizer(
            corpus_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            512,
        )
        .unwrap();
        let tokenizer = LlmTokenizer::from_file(&tokenizer_path).unwrap();
        (tokenizer, corpus)
    }

    fn save_initial_weights<B: Backend>(
        dir: &Path,
        config: &ModelConfig,
        device: &B::Device,
    ) -> PathBuf {
        let model: Llm<B> = config.init_model(device);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let path = dir.join("init");
        recorder.record(model.into_record(), path.clone()).unwrap();
        path.with_extension("bin")
    }

    #[allow(clippy::too_many_arguments)]
    fn run_train_once<B: AutodiffBackend>(
        config: &ModelConfig,
        tokenizer: &LlmTokenizer,
        corpus: &str,
        epochs: usize,
        batch_size: usize,
        grad_accum: usize,
        max_tokens: usize,
        resume: &Path,
        save_dir: Option<&Path>,
        output_dir: &Path,
        device: &B::Device,
    ) -> Vec<(usize, f64)> {
        let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        train_with_events::<B>(
            None,
            Some(corpus),
            config.clone(),
            tokenizer,
            epochs,
            batch_size,
            0.1,
            None,
            Some(max_tokens),
            0.0,
            grad_accum,
            0,
            save_dir,
            Some(resume),
            device,
            output_dir.to_str().unwrap(),
            &tx,
            &cancel,
            None,
        )
        .expect("tiny training run failed");
        drop(tx);
        rx.into_iter()
            .filter_map(|event| match event {
                TrainingEvent::Progress { step, lr, .. } => Some((step, lr)),
                _ => None,
            })
            .collect()
    }

    fn load_flat_weights<B: Backend>(dir: &Path, config: &ModelConfig, device: &B::Device) -> Vec<f32> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(dir.join("model_weights"), device)
            .expect("failed to load trained weights");
        let model: Llm<B> = config.init_model(device).load_record(record);
        let mut out = Vec::new();
        out.extend(model.embed_tokens.weight.val().into_data().to_vec::<f32>().unwrap());
        out.extend(model.blocks[0].attn.q_proj.weight.val().into_data().to_vec::<f32>().unwrap());
        out.extend(model.blocks[0].mlp.down_proj.weight.val().into_data().to_vec::<f32>().unwrap());
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn flat_autodiff_weights(model: &Llm<TestBackend>) -> Vec<f32> {
        let model = model.valid();
        let mut out = Vec::new();
        out.extend(model.embed_tokens.weight.val().into_data().to_vec::<f32>().unwrap());
        out.extend(model.blocks[0].attn.q_proj.weight.val().into_data().to_vec::<f32>().unwrap());
        out.extend(model.blocks[0].mlp.down_proj.weight.val().into_data().to_vec::<f32>().unwrap());
        out
    }

    fn fixed_grads(model: &Llm<TestBackend>, input: &Tensor<TestBackend, 2, Int>) -> GradientsParams {
        let logits = model.forward(input.clone());
        let loss = logits.mean();
        GradientsParams::from_grads(loss.backward(), model)
    }

    /// Finding #1: gradient accumulation must be mathematically equivalent to a
    /// single larger batch (linearity of the gradient). Today the training loop
    /// steps with only the last micro-batch gradients, so this test fails.
    #[test]
    fn test_grad_accum_equivalence() {
        let device = NdArrayDevice::Cpu;
        let dir = test_dir("grad_accum");
        let (tokenizer, corpus) = tiny_tokenizer_and_corpus(&dir);
        let config = ModelConfig::new(tokenizer.vocab_size(), 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let init = save_initial_weights::<NdArray<f32, i32>>(&dir, &config, &device);

        // 20 tokens at n_ctx=8 -> exactly 2 sequences.
        let out_ref = dir.join("out_ref");
        run_train_once::<TestBackend>(
            &config, &tokenizer, &corpus, 1, 2, 1, 20, &init, None, &out_ref, &device,
        );

        let out_accum = dir.join("out_accum");
        run_train_once::<TestBackend>(
            &config, &tokenizer, &corpus, 1, 1, 2, 20, &init, None, &out_accum, &device,
        );

        let w_ref = load_flat_weights::<NdArray<f32, i32>>(&out_ref, &config, &device);
        let w_accum = load_flat_weights::<NdArray<f32, i32>>(&out_accum, &config, &device);
        let diff = max_abs_diff(&w_ref, &w_accum);
        eprintln!("[GRAD_ACCUM] max_abs_diff(batch=2 vs batch=1+accum=2) = {diff:.6e}");
        // End-to-end check: AdamW normalizes each element, so float reduction
        // differences (batch-2 matmul vs two batch-1 matmuls) get amplified up
        // to ~1e-4. Dropping a micro-batch changes the update by ~2*min_lr
        // (~2e-2), i.e. two orders of magnitude above this threshold.
        assert!(
            diff < 1e-3,
            "gradient accumulation with 2 micro-batches must equal one batch of 2: \
             max_abs_diff={diff:.6e}"
        );
    }

    /// Optimizer-free proof of finding #1: accumulated micro-batch gradients
    /// must be identical to the full-batch gradient (linearity + 1/N scaling).
    #[test]
    fn test_accumulated_gradients_match_full_batch() {
        let device = NdArrayDevice::Cpu;
        let config = ModelConfig::new(64, 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let model = config.init_model::<TestBackend>(&device);

        let seq_a = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1i32, 2, 3, 4, 5, 6, 7, 8], [1, 8]),
            &device,
        );
        let seq_b = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![8i32, 7, 6, 5, 4, 3, 2, 1], [1, 8]),
            &device,
        );

        // Reference: single full-batch forward.
        let full_batch = Tensor::cat(vec![seq_a.clone(), seq_b.clone()], 0);
        let full_loss = model.forward(full_batch).mean();
        let full_grads = GradientsParams::from_grads(full_loss.backward(), &model);

        // Accumulated: two micro-batches, each scaled by 1/N.
        let mut accumulator = GradientsAccumulator::<Llm<TestBackend>>::new();
        for seq in [seq_a, seq_b] {
            let loss = model.forward(seq).mean() / 2.0;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            accumulator.accumulate(&model, grads);
        }
        let accum_grads = accumulator.grads();

        let param_id = model.embed_tokens.weight.id;
        let full = full_grads
            .get::<NdArray<f32, i32>, 2>(param_id)
            .expect("full-batch embedding gradient")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let accumulated = accum_grads
            .get::<NdArray<f32, i32>, 2>(param_id)
            .expect("accumulated embedding gradient")
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        let diff = max_abs_diff(&full, &accumulated);
        let scale = full.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!("[GRAD_GRADS] max_abs_diff={diff:.3e} | gradient scale {scale:.3e}");
        assert!(
            diff <= 1e-5 * scale.max(1.0),
            "accumulated gradients must match the full-batch gradient: \
             max_abs_diff={diff:.3e} (scale {scale:.3e})"
        );
    }

    /// Finding #2 (scheduler): resuming from an epoch checkpoint must continue
    /// the LR schedule, not restart it. Compares the first LR after resume with
    /// the LR at the equivalent step of an uninterrupted 2-epoch run.
    #[test]
    fn test_resume_restores_lr_schedule() {
        let device = NdArrayDevice::Cpu;
        let dir = test_dir("resume_lr");
        let (tokenizer, corpus) = tiny_tokenizer_and_corpus(&dir);
        let config = ModelConfig::new(tokenizer.vocab_size(), 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let init = save_initial_weights::<NdArray<f32, i32>>(&dir, &config, &device);

        // 72 tokens -> 8 sequences -> 8 batches per epoch (batch_size=1).
        let continuous = run_train_once::<TestBackend>(
            &config, &tokenizer, &corpus, 2, 1, 1, 72, &init, None, &dir.join("continuous"), &device,
        );
        let lr_continuous_step_9 = continuous
            .iter()
            .find(|(step, _)| *step == 9)
            .map(|(_, lr)| *lr)
            .expect("continuous run must emit progress for step 9");

        let ck_dir = dir.join("ck");
        run_train_once::<TestBackend>(
            &config, &tokenizer, &corpus, 1, 1, 1, 72, &init, Some(&ck_dir), &dir.join("split_a"), &device,
        );
        let ckpt = ck_dir.join("checkpoint_epoch_1.bin");
        assert!(ckpt.is_file(), "epoch checkpoint was not written to {ckpt:?}");

        let resumed = run_train_once::<TestBackend>(
            &config, &tokenizer, &corpus, 1, 1, 1, 72, &ckpt, None, &dir.join("split_b"), &device,
        );
        let (first_step, lr_resumed) = *resumed.first().expect("resumed run must emit progress");
        eprintln!(
            "[RESUME] first step after resume = {first_step}, lr = {lr_resumed:.8} | \
             continuous step 9 lr = {lr_continuous_step_9:.8}"
        );
        assert!(
            (lr_resumed - lr_continuous_step_9).abs() < 1e-9,
            "the LR schedule must continue after resume: resumed lr={lr_resumed:.8} \
             vs continuous step-9 lr={lr_continuous_step_9:.8}"
        );
    }

    /// Finding #2 (optimizer): dropping the AdamW moments on resume changes the
    /// next update, while `Optimizer::to_record`/`load_record` reproduces it.
    #[test]
    fn test_adamw_state_survives_record_roundtrip() {
        let device = NdArrayDevice::Cpu;
        let config = ModelConfig::new(64, 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1i32; 8], [1, 8]),
            &device,
        );

        // Step 1: shared baseline weights + optimizer state.
        let mut model = config.init_model::<TestBackend>(&device);
        let mut optim = AdamWConfig::new()
            .with_weight_decay(0.1f32)
            .init::<TestBackend, Llm<TestBackend>>();
        let grads = fixed_grads(&model, &input);
        model = optim.step(1e-2, model, grads);
        // Snapshot the optimizer state right after the first step (what a
        // proper checkpoint would persist alongside the weights).
        let record = optim.to_record();

        // Reference: same optimizer object continues (moments preserved).
        let mut model_ref = model.clone();
        let grads_ref = fixed_grads(&model_ref, &input);
        model_ref = optim.step(1e-2, model_ref, grads_ref);

        // Naive resume: only weights survive, optimizer moments are reset.
        let mut model_naive = model.clone();
        let mut optim_naive = AdamWConfig::new()
            .with_weight_decay(0.1f32)
            .init::<TestBackend, Llm<TestBackend>>();
        let grads_naive = fixed_grads(&model_naive, &input);
        model_naive = optim_naive.step(1e-2, model_naive, grads_naive);

        // Proper resume: weights + optimizer record captured after step 1.
        let mut model_proper = model.clone();
        let mut optim_proper = AdamWConfig::new()
            .with_weight_decay(0.1f32)
            .init::<TestBackend, Llm<TestBackend>>()
            .load_record(record);
        let grads_proper = fixed_grads(&model_proper, &input);
        model_proper = optim_proper.step(1e-2, model_proper, grads_proper);

        let w_ref = flat_autodiff_weights(&model_ref);
        let w_naive = flat_autodiff_weights(&model_naive);
        let w_proper = flat_autodiff_weights(&model_proper);

        let diff_naive = max_abs_diff(&w_ref, &w_naive);
        let diff_proper = max_abs_diff(&w_ref, &w_proper);
        eprintln!(
            "[ADAMW] naive resume diff = {diff_naive:.6e} | optimizer-record resume diff = {diff_proper:.6e}"
        );

        assert!(
            diff_naive > 1e-6,
            "a checkpoint without the optimizer record must change the next update \
             (diff={diff_naive:.6e}); if this fails the test setup is wrong"
        );
        assert!(
            diff_proper < 1e-6,
            "restoring Optimizer::to_record/load_record must reproduce the next update \
             exactly (diff={diff_proper:.6e})"
        );
    }

    /// Finding #3: stopping training (user stop / Ctrl+C) must persist an
    /// interrupted checkpoint plus the full training state.
    #[test]
    fn test_cancel_saves_interrupted_checkpoint() {
        let device = NdArrayDevice::Cpu;
        let dir = test_dir("cancel_save");
        let (tokenizer, corpus) = tiny_tokenizer_and_corpus(&dir);
        let config = ModelConfig::new(tokenizer.vocab_size(), 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let init = save_initial_weights::<NdArray<f32, i32>>(&dir, &config, &device);
        let out = dir.join("out");

        let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_thread = cancel.clone();
        let worker = std::thread::spawn(move || {
            let device = NdArrayDevice::Cpu;
            train_with_events::<TestBackend>(
                None, Some(&corpus), config, &tokenizer, 4, 1, 0.1, None, Some(72), 0.0, 1, 0,
                None, Some(&init), &device, out.to_str().unwrap(), &tx, &cancel, None,
            )
            .expect("training run failed");
        });

        // Wait until the first micro-batch completes, then request a stop.
        loop {
            if let TrainingEvent::Progress { .. } = rx.recv().expect("training event") {
                break;
            }
        }
        cancel_thread.store(true, Ordering::SeqCst);
        worker.join().expect("worker thread panicked");

        assert!(
            dir.join("out/checkpoint_interrupted.bin").is_file(),
            "interrupted checkpoint missing"
        );
        assert!(dir.join("out/train_state.json").is_file(), "train_state.json missing");
        assert!(
            dir.join("out/optimizer_state.bin").is_file(),
            "optimizer_state.bin missing"
        );
    }
}

