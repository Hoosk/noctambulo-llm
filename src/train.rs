//! Training pipeline: datasets, loss/schedules, state, evaluation, and the loop.
//!
//! The implementation is split into focused submodules:
//! - [`dataset`]: tokenized dataset and sliding-window batches.
//! - [`loss`]: cross-entropy/label smoothing/z-loss and small schedules.
//! - [`state`]: events, persistent training state, optimizer sidecars.
//! - [`eval`]: evaluation metrics (loss / PPL / BPB).
//! - [`loop_runner`]: the training loop.

mod dataset;
mod eval;
mod loss;
mod loop_runner;
mod state;

#[allow(unused_imports)]
pub use dataset::TextDataset;
#[allow(unused_imports)]
pub use eval::{evaluate_dataset, run_eval};
#[allow(unused_imports)]
pub use loss::{LossOptions, compute_scheduled_n_ctx};
pub use loop_runner::{train, train_with_events};
#[allow(unused_imports)]
pub use state::{OPTIMIZER_STATE_FILE, TRAIN_STATE_FILE, TrainState, TrainingEvent};

#[cfg(test)]
mod tests;

// Names inherited by the in-crate test module (`use super::*`), mirroring the
// previous single-file layout.
#[allow(unused_imports)]
use crate::config::ModelConfig;
#[allow(unused_imports)]
use crate::data::LlmTokenizer;
#[allow(unused_imports)]
use crate::generate::XorshiftRng;
#[allow(unused_imports)]
use crate::model::Llm;
#[allow(unused_imports)]
use burn::module::Module;
#[allow(unused_imports)]
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, ModuleOptimizer};
#[allow(unused_imports)]
use burn::tensor::{Device, Int, Tensor, TensorData};
#[allow(unused_imports)]
use std::path::{Path, PathBuf};
#[allow(unused_imports)]
use std::sync::Arc;
#[allow(unused_imports)]
use std::sync::atomic::{AtomicBool, Ordering};

#[allow(unused_imports)]
pub(crate) use loss::{compute_loss, should_stop_early};
#[allow(unused_imports)]
pub(crate) use state::{load_optimizer_record, persist_train_state};
