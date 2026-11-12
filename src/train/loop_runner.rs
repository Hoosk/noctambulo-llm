//! The training loop: gradient accumulation, schedules, validation, and checkpoints.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Instant;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams};
use burn::tensor::{Device, Tensor};

use crate::config::ModelConfig;
use crate::data::LlmTokenizer;
use crate::generate::{XorshiftRng, generate};
use crate::model::Llm;
use crate::storage;
use crate::tui;

use super::dataset::TextDataset;
use super::eval::evaluate_dataset;
use super::loss::{
    LossOptions, compute_loss, compute_scheduled_n_ctx, should_stop_early, shuffle_indices,
};
use super::state::{
    TrainingEvent, emit_log, load_optimizer_record, load_train_state, persist_train_state,
    resume_artifacts_dir, save_interrupted_checkpoint,
};

#[allow(clippy::too_many_arguments)]
pub fn train(
    dataset_path: Option<&Path>,
    raw_text: Option<&str>,
    dataset_sources: Option<&[(PathBuf, f32)]>,
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
    device: &Device,
    output_dir: &str,
    early_stop_patience: usize,
    loss_opts: LossOptions,
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

    let result = train_with_events(
        dataset_path, raw_text, dataset_sources, model_config, tokenizer, epochs, batch_size, lr,
        custom_step, max_tokens, val_split, grad_accum_steps, seq_warmup_steps,
        save_dir, resume_path, device, output_dir, early_stop_patience,
        loss_opts, &tx, &cancel_thread, None,
    );

    drop(tx);
    let _ = handle.join();
    result
}

#[allow(clippy::too_many_arguments)]
pub fn train_with_events(
    dataset_path: Option<&Path>,
    raw_text: Option<&str>,
    dataset_sources: Option<&[(PathBuf, f32)]>,
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
    device: &Device,
    output_dir: &str,
    early_stop_patience: usize,
    loss_opts: LossOptions,
    tx: &Sender<TrainingEvent>,
    cancel: &Arc<AtomicBool>,
    pause: Option<&Arc<AtomicBool>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let last_pct = std::cell::Cell::new(-10i64);
    let progress = |read: u64, total: u64| {
        let pct = read.saturating_mul(100).checked_div(total).unwrap_or(100) as i64;
        if pct >= last_pct.get() + 5 {
            last_pct.set(pct);
            emit_log(
                tx,
                format!(
                    "Tokenizing dataset: {pct}% ({:.0}/{:.0} MB)",
                    read as f64 / 1_048_576.0,
                    total as f64 / 1_048_576.0
                ),
            );
        }
    };

    let full_dataset = if let Some(sources) = dataset_sources {
        emit_log(
            tx,
            format!("Loading {} dataset sources (streamed mix)...", sources.len()),
        );
        let dataset = TextDataset::from_sources_with_progress(
            sources,
            tokenizer,
            model_config.n_ctx,
            custom_step,
            max_tokens,
            Some(&progress),
            Some(cancel.as_ref()),
        )?;
        emit_log(tx, format!("Dataset ready: {} tokens.", dataset.tokens.len()));
        dataset
    } else if let Some(path) = dataset_path {
        emit_log(tx, format!("Loading dataset '{}'...", path.display()));
        let dataset = TextDataset::from_file_with_progress(
            path,
            tokenizer,
            model_config.n_ctx,
            custom_step,
            max_tokens,
            Some(&progress),
            Some(cancel.as_ref()),
        )?;
        emit_log(tx, format!("Dataset ready: {} tokens.", dataset.tokens.len()));
        dataset
    } else if let Some(text) = raw_text {
        TextDataset::from_raw_text(text, tokenizer, model_config.n_ctx, custom_step, max_tokens)
    } else {
        return Err("No dataset path, sources or raw text provided for training.".into());
    };

    if full_dataset.len() == 0 {
        return Err("Dataset contains 0 valid sequence samples for the current context length.".into());
    }

    let (train_dataset, val_dataset) = if val_split > 0.0 {
        let (tr, va) = full_dataset.split(val_split);
        emit_log(tx, format!("Dataset split: {} training samples, {} validation samples ({:.1}% val).", tr.len(), va.len(), val_split * 100.0));
        (tr, va)
    } else {
        (full_dataset, TextDataset { tokens: vec![], n_ctx: model_config.n_ctx, step: custom_step.unwrap_or(model_config.n_ctx), total_bytes: 0 })
    };

    tui::print_info("Initializing model architecture on the compute device...");
    let mut model = model_config.init_model(device).train();
    if model.embed_tokens.weight.val().dtype() == burn::tensor::DType::F16 {
        return Err(
            "f16 training is not supported: AdamW moment estimates underflow in \
             f16 (the first update explodes and the model stops learning). \
             Train with --dtype f32 (default)."
                .into(),
        );
    }

    let resume_state = load_train_state(resume_path);

    if let Some(res_path) = resume_path {
        let rec_file = if res_path.is_dir() {
            storage::model_weights_path(res_path)
        } else {
            res_path.to_path_buf()
        };
        emit_log(tx, format!("Resuming model weights from {:?}...", rec_file));
        model = storage::load_module(model, &rec_file)
            .map_err(|e| format!("Failed to load checkpoint from {res_path:?}: {e}"))?
            .train();
        emit_log(tx, "Successfully loaded checkpoint weights!");
    }

    // AdamW's built-in decoupled weight decay. A previous ModuleMapper-based
    // implementation (rank >= 2 only) froze every mapped parameter: the mapped
    // tensor stopped receiving gradients, so training stayed at ln(vocab).
    let mut optim = AdamWConfig::new()
        .with_weight_decay(loss_opts.weight_decay as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // Restore AdamW moments + scheduler position when a sidecar state exists.
    if resume_state.is_some() {
        let dir = resume_artifacts_dir(resume_path.expect("resume_state implies a resume path"));
        let (restored, loaded) = load_optimizer_record(optim, &dir)?;
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

    // Chinchilla-style budget check: ~20 tokens per parameter saturates a model
    // of this size; warning early avoids a silently undertrained run.
    if let Some(limit) = max_tokens
        && limit > 0 {
            let params = model_config.param_count();
            let recommended = params.saturating_mul(20);
            if limit < recommended {
                emit_log(tx, format!(
                    "Note: {} tokens is below the ~{} token budget suggested for {} ({:.1}M params). \
                     Consider --max-tokens 0 (full dataset) or a larger value.",
                    tui::format_number(limit),
                    tui::format_number(recommended),
                    "this model",
                    params as f64 / 1_000_000.0
                ));
            }
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
    let mut epochs_without_improvement = 0usize;
    let training_start = Instant::now();

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        // Initialized from the first loss so the accumulator keeps the loss
        // dtype; a zero-initialized f32 tensor added to an f16 loss would
        // corrupt the result.
        let mut accum_loss_tensor: Option<Tensor<1>> = None;
        let mut accum_count = 0usize;
        let mut total_epoch_loss = 0.0f32;
        let mut total_batches = 0usize;
        let mut last_live_update = Instant::now();

        let log_interval = std::cmp::max(1, num_batches / 10);

        // Shuffle sequence sample indices per epoch
        let mut sample_indices: Vec<usize> = (0..train_dataset.len()).collect();
        shuffle_indices(&mut sample_indices, &mut shuffle_rng);

        let mut grad_accumulator = GradientsAccumulator::<Llm>::new();
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

            let (inputs, targets) = train_dataset.get_batch(&indices, current_n_ctx, device);
            let logits = model.forward(inputs);
            let [b, t, v] = logits.dims();

            let logits_flat = logits.reshape([b * t, v]);
            let targets_flat = targets.reshape([b * t, 1]);
            let loss = compute_loss(logits_flat, targets_flat, loss_opts);

            // Metrics use the unscaled loss; gradients are scaled by the
            // micro-batch window size so the accumulated update equals the mean
            // gradient over the window (true gradient accumulation).
            let detached = loss.clone().detach();
            accum_loss_tensor = Some(match accum_loss_tensor.take() {
                Some(acc) => acc + detached,
                None => detached,
            });
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
                    .as_ref()
                    .map(|acc| acc.clone().into_data().convert::<f32>().try_to_vec::<f32>())
                    .and_then(|r| r.ok())
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
                accum_loss_tensor = None;
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
            // Evaluate the whole validation split (max_batches = 0): the old
            // hard cap of 100 batches silently made PPL incomparable across
            // datasets of different sizes.
            let (val_loss, val_ppl, val_bpb) =
                evaluate_dataset(&model.valid(), &val_dataset, effective_batch_size, 0, device);
            if !val_loss.is_finite() {
                emit_log(tx, "Warning: validation loss is non-finite; skipping validation this epoch.");
            } else {
                match val_bpb {
                    Some(bpb) => emit_log(tx, format!(
                        "   └── Validation Loss: {:.4} | Perplexity (PPL): {:.4} | Bits/Byte (BPB): {:.4}",
                        val_loss, val_ppl, bpb
                    )),
                    None => emit_log(tx, format!("   └── Validation Loss: {:.4} | Validation Perplexity (PPL): {:.4}", val_loss, val_ppl)),
                }
                let _ = tx.send(TrainingEvent::Val { loss: val_loss, ppl: val_ppl, bpb: val_bpb, epoch });

                let is_best = match best_val_loss {
                    None => true,
                    Some(prev) => val_loss < prev,
                };
                if is_best {
                    epochs_without_improvement = 0;
                    best_val_loss = Some(val_loss);
                    if let Some(s_dir) = save_dir {
                        let best_path = s_dir.join("checkpoint_best.bpk");
                        if let Err(e) = storage::save_module(&model, &best_path) {
                            emit_log(tx, format!("Warning: Failed to save best checkpoint at {:?}: {:?}", best_path, e));
                        } else {
                            emit_log(tx, format!("   ★ New best val_loss={:.4} → {:?}", val_loss, best_path));
                            let _ = tx.send(TrainingEvent::Checkpoint {
                                path: best_path.to_string_lossy().to_string(),
                                kind: "best".into(),
                                epoch,
                                batch: 0,
                            });
                        }
                    }
                } else {
                    epochs_without_improvement += 1;
                }
            }
        }

        if let Some(s_dir) = save_dir {
            std::fs::create_dir_all(s_dir)?;
            let ckpt_path = s_dir.join(format!("checkpoint_epoch_{epoch}.bpk"));
            if let Err(e) = storage::save_module(&model, &ckpt_path) {
                emit_log(tx, format!("Warning: Failed to save checkpoint at {:?}: {:?}", ckpt_path, e));
            } else {
                emit_log(tx, format!("Saved checkpoint: {:?}", ckpt_path));
                let _ = tx.send(TrainingEvent::Checkpoint {
                    path: ckpt_path.to_string_lossy().to_string(),
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

        let preview = generate(
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
            crate::model::KvCacheDtype::F32,
            model_config
                .eviction_geometry()
                .ok()
                .flatten()
                .map(crate::generate::EvictionOptions::from),
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

        if should_stop_early(early_stop_patience, epochs_without_improvement) {
            emit_log(tx, format!(
                "Early stopping: validation loss did not improve for {} epoch(s).",
                early_stop_patience
            ));
            break;
        }
    }

    std::fs::create_dir_all(output_dir)?;
    let config_path = Path::new(output_dir).join("config.json");
    let config_json = serde_json::to_string_pretty(&model_config)?;
    std::fs::write(&config_path, config_json)?;

    let weights_path = storage::model_weights_path(Path::new(output_dir));
    storage::save_module(&model, &weights_path)
        .map_err(|e| format!("Failed to save model record: {e}"))?;

    train_state.global_step = global_step;
    train_state.best_val_loss = best_val_loss;
    train_state.rng_state = shuffle_rng.state();
    if let Err(e) = persist_train_state(&optim, &train_state, Path::new(output_dir)) {
        emit_log(tx, format!("Warning: failed to save training state: {e}"));
    }

    let _ = tx.send(TrainingEvent::Done {
        final_loss: last_avg_loss,
        total_time_secs: training_start.elapsed().as_secs_f64(),
        weights_path: weights_path.to_string_lossy().to_string(),
        config_path: config_path.to_string_lossy().to_string(),
    });

    Ok(())
}
