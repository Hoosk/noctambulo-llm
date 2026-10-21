//! Persisted history of training runs, with train/val curves so the UI can
//! compare runs. Records are appended as JSON lines (`.noctambulo/runs.jsonl`); on
//! load the last record for each id wins, so a run can be written twice
//! (started, then finished) without losing history.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::serve::runner::TrainParams;
use crate::serve::state::LossPoint;

/// Runs kept before pruning the oldest finished ones.
pub const MAX_RUNS: usize = 100;
pub const RUNS_FILE: &str = ".noctambulo/runs.jsonl";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub id: String,
    pub job_id: String,
    pub dataset: String,
    pub output_dir: String,
    pub dtype: String,
    pub device: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub grad_accum: usize,
    /// `running` | `done` | `stopped` | `failed`
    pub status: String,
    pub note: Option<String>,
    pub started_unix: u64,
    pub finished_unix: Option<u64>,
    pub final_loss: Option<f32>,
    pub final_ppl: Option<f32>,
    pub loss_points: Vec<LossPoint>,
    pub val_points: Vec<LossPoint>,
}

/// List-view projection without the curves.
#[derive(Debug, Clone, Serialize)]
pub struct RunSummary {
    pub id: String,
    pub job_id: String,
    pub dataset: String,
    pub output_dir: String,
    pub dtype: String,
    pub device: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub grad_accum: usize,
    pub status: String,
    pub note: Option<String>,
    pub started_unix: u64,
    pub finished_unix: Option<u64>,
    pub final_loss: Option<f32>,
    pub final_ppl: Option<f32>,
}

impl From<&RunRecord> for RunSummary {
    fn from(r: &RunRecord) -> Self {
        Self {
            id: r.id.clone(),
            job_id: r.job_id.clone(),
            dataset: r.dataset.clone(),
            output_dir: r.output_dir.clone(),
            dtype: r.dtype.clone(),
            device: r.device.clone(),
            epochs: r.epochs,
            batch_size: r.batch_size,
            lr: r.lr,
            n_ctx: r.n_ctx,
            n_embd: r.n_embd,
            n_layer: r.n_layer,
            grad_accum: r.grad_accum,
            status: r.status.clone(),
            note: r.note.clone(),
            started_unix: r.started_unix,
            finished_unix: r.finished_unix,
            final_loss: r.final_loss,
            final_ppl: r.final_ppl,
        }
    }
}

pub struct RunsStore {
    path: PathBuf,
    runs: Mutex<Vec<RunRecord>>,
}

impl RunsStore {
    /// Load history from disk, reducing the JSONL by id (last record wins).
    pub fn load(path: PathBuf) -> Self {
        let mut by_id: HashMap<String, RunRecord> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                let Ok(record) = serde_json::from_str::<RunRecord>(line) else {
                    continue;
                };
                if !by_id.contains_key(&record.id) {
                    order.push(record.id.clone());
                }
                by_id.insert(record.id.clone(), record);
            }
        }
        let runs = order.into_iter().filter_map(|id| by_id.remove(&id)).collect();
        Self {
            path,
            runs: Mutex::new(runs),
        }
    }

    /// Register a new run and return its id.
    pub fn begin(&self, params: &TrainParams, job_id: &str) -> String {
        let id = format!("run-{job_id}");
        let dataset = params
            .file
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "(none)".into());
        let record = RunRecord {
            id: id.clone(),
            job_id: job_id.to_string(),
            dataset,
            output_dir: params.output_dir.clone(),
            dtype: params.dtype.as_str().to_string(),
            device: params.device.clone(),
            epochs: params.epochs,
            batch_size: params.batch_size,
            lr: params.lr,
            n_ctx: params.n_ctx,
            n_embd: params.n_embd,
            n_layer: params.n_layer,
            grad_accum: params.grad_accum,
            status: "running".into(),
            note: None,
            started_unix: now_unix(),
            finished_unix: None,
            final_loss: None,
            final_ppl: None,
            loss_points: Vec::new(),
            val_points: Vec::new(),
        };
        {
            let mut runs = self.runs.lock().unwrap();
            runs.push(record.clone());
            prune(&mut runs);
        }
        self.append(&record);
        id
    }

    /// Record a validation point for a running job.
    pub fn val_point(&self, id: &str, step: usize, loss: f32, ppl: f32) {
        if !loss.is_finite() || loss <= 0.0 {
            return;
        }
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.iter_mut().rev().find(|r| r.id == id) {
            run.val_points.push(LossPoint { x: step, y: loss });
            run.final_ppl = Some(ppl);
        }
    }

    /// Mark a run as finished, storing the final metrics and curves.
    pub fn finish(
        &self,
        id: &str,
        status: &str,
        note: Option<String>,
        final_loss: Option<f32>,
        loss_points: Vec<LossPoint>,
    ) {
        let record = {
            let mut runs = self.runs.lock().unwrap();
            let Some(run) = runs.iter_mut().rev().find(|r| r.id == id) else {
                return;
            };
            run.status = status.into();
            run.note = note;
            run.final_loss = final_loss;
            run.finished_unix = Some(now_unix());
            run.loss_points = loss_points;
            run.clone()
        };
        self.append(&record);
    }

    /// Newest first.
    pub fn list(&self) -> Vec<RunSummary> {
        self.runs
            .lock()
            .unwrap()
            .iter()
            .rev()
            .map(RunSummary::from)
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<RunRecord> {
        self.runs
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find(|r| r.id == id)
            .cloned()
    }

    fn append(&self, record: &RunRecord) {
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            && let Ok(json) = serde_json::to_string(record) {
                let _ = writeln!(f, "{json}");
            }
    }
}

fn prune(runs: &mut Vec<RunRecord>) {
    if runs.len() <= MAX_RUNS {
        return;
    }
    let excess = runs.len() - MAX_RUNS;
    let mut removed = 0;
    let mut i = 0;
    while i < runs.len() && removed < excess {
        if runs[i].status != "running" {
            runs.remove(i);
            removed += 1;
        } else {
            i += 1;
        }
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DType;

    fn params() -> TrainParams {
        TrainParams {
            file: Some(PathBuf::from("datasets/corpus.txt")),
            dtype: DType::F16,
            epochs: 3,
            batch_size: 64,
            lr: 3e-4,
            max_tokens: None,
            val_split: 0.1,
            grad_accum: 2,
            seq_warmup_steps: 0,
            n_ctx: 256,
            n_embd: 256,
            n_head: 4,
            n_kv_head: Some(2),
            n_layer: 4,
            dropout: 0.1,
            attn_softcap: 0.0,
            sliding_window: 0,
            sink_tokens: 0,
            kv_share_group: 1,
            mla_latent_dim: 0,
            kv_eviction: "none".into(),
            kv_budget: 0,
            kv_recent: 64,
            save_dir: None,
            resume: None,
            output_dir: "model_output".into(),
            device: "DiscreteGpu(0)".into(),
            early_stop_patience: 0,
            weight_decay: 0.1,
            label_smoothing: 0.0,
            z_loss: 0.0,
        }
    }

    #[test]
    fn test_run_lifecycle_and_persistence() {
        let path = std::env::temp_dir().join(format!("noctambulo_runs_test_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let store = RunsStore::load(path.clone());
        let id = store.begin(&params(), "job-123");
        assert_eq!(id, "run-job-123");

        store.val_point(&id, 10, 5.0, 148.4);
        store.finish(
            &id,
            "done",
            None,
            Some(4.2),
            vec![LossPoint { x: 1, y: 6.0 }, LossPoint { x: 10, y: 4.2 }],
        );

        let summary = store.list();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].status, "done");
        assert_eq!(summary[0].dataset, "corpus.txt");
        assert_eq!(summary[0].final_loss, Some(4.2));
        assert_eq!(summary[0].final_ppl, Some(148.4));

        let record = store.get(&id).unwrap();
        assert_eq!(record.loss_points.len(), 2);
        assert_eq!(record.val_points.len(), 1);
        assert_eq!(record.dtype, "f16");

        // A fresh store must rebuild the same history from the JSONL file.
        let reloaded = RunsStore::load(path.clone());
        let summary = reloaded.list();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].id, id);
        assert_eq!(reloaded.get(&id).unwrap().loss_points.len(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_finish_unknown_run_is_ignored() {
        let path = std::env::temp_dir().join(format!("noctambulo_runs_missing_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let store = RunsStore::load(path.clone());
        store.finish("nope", "done", None, None, Vec::new());
        assert!(store.list().is_empty());
        let _ = std::fs::remove_file(&path);
    }
}
