//! Training events, persistent state, and checkpoint/optimizer sidecars.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;

use burn::optim::{ModuleOptimizer, OptimizerRecord};

use crate::model::Llm;
use crate::storage;

/// JSON sidecar file holding scheduler position, RNG and best validation loss.
pub const TRAIN_STATE_FILE: &str = "train_state.json";
/// File name of the serialized optimizer state (burnpack).
pub const OPTIMIZER_STATE_FILE: &str = "optimizer_state.bpk";

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
pub(crate) fn resume_artifacts_dir(resume_path: &Path) -> PathBuf {
    if resume_path.is_dir() {
        return resume_path.to_path_buf();
    }
    resume_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub(crate) fn load_train_state(resume_path: Option<&Path>) -> Option<TrainState> {
    let path = resume_path?;
    let bytes = std::fs::read(resume_artifacts_dir(path).join(TRAIN_STATE_FILE)).ok()?;
    serde_json::from_slice::<TrainState>(&bytes).ok()
}

/// Persist the scheduler/RNG state plus the optimizer moments next to each
/// other so `--resume` can restore the full training state, not just weights.
pub(crate) fn persist_train_state(
    optim: &ModuleOptimizer,
    state: &TrainState,
    dir: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    std::fs::create_dir_all(dir)?;
    optim
        .save(dir.join(OPTIMIZER_STATE_FILE))
        .map_err(|e| format!("Failed to save optimizer state: {e:?}"))?;
    std::fs::write(dir.join(TRAIN_STATE_FILE), serde_json::to_string_pretty(state)?)?;
    Ok(())
}

/// Reload the AdamW moments from `optimizer_state.bpk` when present. Missing
/// state is fine (weights-only resume); a corrupt or incompatible record is an
/// error so resume never silently restarts Adam from zero.
pub(crate) fn load_optimizer_record(
    optim: ModuleOptimizer,
    dir: &Path,
) -> Result<(ModuleOptimizer, bool), String> {
    let path = dir.join(OPTIMIZER_STATE_FILE);
    if !path.is_file() {
        return Ok((optim, false));
    }
    let record = OptimizerRecord::load(&path)
        .map_err(|e| format!("failed to load optimizer state '{}': {e:?}", path.display()))?;
    Ok((optim.load_record(record), true))
}

/// Save the model weights + full training state when training is interrupted
/// (user stop or Ctrl+C) so no work is lost.
pub(crate) fn save_interrupted_checkpoint(
    model: &Llm,
    optim: &ModuleOptimizer,
    train_state: &TrainState,
    output_dir: &str,
    tx: &Sender<TrainingEvent>,
) {
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        emit_log(tx, format!("Warning: failed to save interrupted checkpoint: {e}"));
        return;
    }
    let saved = Path::new(output_dir).join("checkpoint_interrupted.bpk");
    match storage::save_module(model, &saved) {
        Ok(()) => {
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
        Err(e) => emit_log(tx, format!("Warning: failed to save interrupted checkpoint: {e}")),
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
        /// Bits per byte (None when the source byte count is unknown, e.g. `.bin` datasets).
        bpb: Option<f32>,
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
pub(crate) fn emit_log(tx: &Sender<TrainingEvent>, line: impl std::fmt::Display) {
    println!("{line}");
    let _ = tx.send(TrainingEvent::LogEvent {
        line: line.to_string(),
        ts: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    });
}
