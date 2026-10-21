//! Native model storage (Burn 0.22 burnpack files).
//!
//! The engine stores models as `model_weights.bpk` and training sidecars as
//! `optimizer_state.bpk` / `checkpoint_*.bpk`. The old Burn 0.21 bincode
//! records (`model_weights.bin`) are not readable with the new store; helper
//! functions return an explicit error pointing at re-import/retrain.

use std::path::{Path, PathBuf};

use burn::store::{BurnpackStore, ModuleSnapshot};

/// File name of the native model weights (burnpack).
pub const MODEL_WEIGHTS_FILE: &str = "model_weights.bpk";

pub fn model_weights_path(dir: &Path) -> PathBuf {
    dir.join(MODEL_WEIGHTS_FILE)
}

/// Legacy Burn 0.21 bincode record, if present.
pub fn legacy_weights_path(dir: &Path) -> Option<PathBuf> {
    let path = dir.join("model_weights.bin");
    path.is_file().then_some(path)
}

/// Saves a module as burnpack, overwriting an existing file.
pub fn save_module<M: ModuleSnapshot>(module: &M, path: &Path) -> Result<(), String> {
    let mut store = BurnpackStore::from_file(path).overwrite(true);
    module
        .save_into(&mut store)
        .map_err(|e| format!("failed to save '{}': {e}", path.display()))
}

/// Loads burnpack weights into `module`, returning the updated module.
pub fn load_module<M: ModuleSnapshot>(mut module: M, path: &Path) -> Result<M, String> {
    if !path.is_file() {
        if let Some(legacy) = legacy_weights_path(path.parent().unwrap_or(Path::new("."))) {
            return Err(format!(
                "'{}' is a Burn 0.21 bincode checkpoint; this build uses burnpack. \
                 Re-import the model (import-hf/import) or retrain it (legacy file: '{}')",
                path.display(),
                legacy.display()
            ));
        }
        return Err(format!("model weights not found at '{}'", path.display()));
    }
    let mut store = BurnpackStore::from_file(path);
    let result = module
        .load_from(&mut store)
        .map_err(|e| format!("failed to load '{}': {e}", path.display()))?;
    if std::env::var_os("NOCTAMBULO_DEBUG_LOAD").is_some() {
        eprintln!(
            "[load] {} applied={} missing={} unused={} errors={} skipped={}",
            path.display(),
            result.applied.len(),
            result.missing.len(),
            result.unused.len(),
            result.errors.len(),
            result.skipped.len()
        );
        if let Some(first) = result.unused.first() {
            eprintln!("[load] first unused: {first}");
        }
        if let Some(first) = result.applied.first() {
            eprintln!("[load] first applied: {first}");
        }
    }
    if !result.errors.is_empty() {
        return Err(format!(
            "failed to apply weights from '{}': {:?}",
            path.display(),
            result.errors
        ));
    }
    if !result.missing.is_empty() {
        return Err(format!(
            "weights at '{}' are missing {} tensor(s), e.g. '{}'; the checkpoint does not match \
             this architecture",
            path.display(),
            result.missing.len(),
            result.missing[0].0
        ));
    }
    Ok(module)
}
