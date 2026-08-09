use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::RwLock;

use crate::train::TrainingEvent;

/// One captured diagnostic line from the training worker.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LogLine {
    pub line: String,
    pub ts: u64,
}

/// A single loss observation for the chart, keyed by global step.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LossPoint {
    pub x: usize,
    pub y: f32,
}

/// Snapshot of the currently running (or last) training job.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct JobStatus {
    pub running: bool,
    pub stopping: bool,
    pub job_id: String,
    pub last_event: Option<TrainingEvent>,
    pub message: Option<String>,
}

/// Everything persisted to `.kuro/state.json` between server restarts.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct PersistedState {
    job: JobStatus,
    loss_points: Vec<LossPoint>,
}

/// Shared server state, wrapped in `Arc` and handed to every handler.
pub struct ServerState {
    pub token: RwLock<Option<String>>,
    pub datasets_dir: PathBuf,
    pub models_dir: PathBuf,
    pub job: RwLock<JobStatus>,
    pub events: tokio::sync::broadcast::Sender<TrainingEvent>,
    /// Ring buffer of the most recent diagnostic lines emitted by the worker.
    logs: RwLock<LogBuffer>,
    /// Loss history for chart backfill across reloads / restarts.
    loss_points: RwLock<Vec<LossPoint>>,
}

#[derive(Default)]
struct LogBuffer {
    lines: VecDeque<LogLine>,
    total: usize,
}

const MAX_LOG_LINES: usize = 5000;
const STATE_FILE: &str = ".kuro/state.json";

impl ServerState {
    pub fn new(token: Option<String>, datasets_dir: PathBuf, models_dir: PathBuf) -> Self {
        let (tx, _rx) = tokio::sync::broadcast::channel::<TrainingEvent>(1024);
        let state = Self {
            token: RwLock::new(token),
            datasets_dir,
            models_dir,
            job: RwLock::new(JobStatus::default()),
            events: tx,
            logs: RwLock::new(LogBuffer::default()),
            loss_points: RwLock::new(Vec::new()),
        };
        state.load_state();
        state
    }

    /// Append a diagnostic line to the in-memory ring buffer.
    pub fn push_log(&self, line: String, ts: u64) {
        let mut buf = self.logs.write().unwrap();
        if buf.lines.len() >= MAX_LOG_LINES {
            buf.lines.pop_front();
        }
        buf.lines.push_back(LogLine { line, ts });
        buf.total += 1;
    }

    /// Return all lines with an index `>= since`, plus the current total so the
    /// caller can poll incrementally. Lines evicted from the buffer are lost,
    /// so `since` older than the kept window returns whatever is left.
    pub fn log_since(&self, since: usize) -> (Vec<LogLine>, usize) {
        let buf = self.logs.read().unwrap();
        let kept_start = buf.total.saturating_sub(buf.lines.len());
        let start = since.saturating_sub(kept_start);
        let lines: Vec<LogLine> = buf.lines.iter().skip(start).cloned().collect();
        (lines, buf.total)
    }

    /// Record a loss point (deduped by step) and persist.
    pub fn push_loss_point(&self, x: usize, y: f32) {
        {
            let mut pts = self.loss_points.write().unwrap();
            match pts.last_mut() {
                Some(last) if last.x == x => last.y = y,
                _ => {
                    let overflow = pts.len().saturating_sub(1999);
                    if overflow > 0 {
                        pts.drain(0..overflow);
                    }
                    pts.push(LossPoint { x, y });
                }
            }
        }
        self.persist();
    }

    /// Snapshot of the loss history for chart backfill.
    pub fn loss_points_snapshot(&self) -> Vec<LossPoint> {
        self.loss_points.read().unwrap().clone()
    }

    /// Persist job status + loss history to `.kuro/state.json`.
    pub fn persist(&self) {
        let state = PersistedState {
            job: self.job.read().unwrap().clone(),
            loss_points: self.loss_points_snapshot(),
        };
        if let Ok(bytes) = serde_json::to_vec_pretty(&state) {
            if std::fs::create_dir_all(".kuro").is_ok() {
                let _ = std::fs::write(STATE_FILE, bytes);
            }
        }
    }

    /// Load `.kuro/state.json` if present. A restarted server is never running;
    /// the job it "remembers" is a stale snapshot, not a live worker.
    fn load_state(&self) {
        let Ok(bytes) = std::fs::read(STATE_FILE) else {
            return;
        };
        let Ok(state) = serde_json::from_slice::<PersistedState>(&bytes) else {
            return;
        };
        let mut job = state.job;
        job.running = false;
        job.stopping = false;
        *self.job.write().unwrap() = job;
        *self.loss_points.write().unwrap() = state.loss_points;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persisted_state_roundtrip() {
        let state = PersistedState {
            job: JobStatus {
                running: true,
                stopping: false,
                job_id: "job-123".into(),
                last_event: Some(TrainingEvent::Progress {
                    step: 5,
                    epoch: 1,
                    epochs: 1,
                    batch: 5,
                    num_batches: 100,
                    pct: 5.0,
                    loss: 6.5,
                    avg_loss: 6.7,
                    lr: 5e-5,
                    batches_per_sec: 10.0,
                    elapsed_secs: 5.0,
                    eta_secs: 95.0,
                    current_n_ctx: 256,
                }),
                message: Some("ok".into()),
            },
            loss_points: vec![LossPoint { x: 1, y: 8.0 }, LossPoint { x: 2, y: 7.5 }],
        };

        let bytes = serde_json::to_vec(&state).unwrap();
        let back: PersistedState = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(back.job.job_id, "job-123");
        assert_eq!(back.loss_points.len(), 2);
        assert_eq!(back.loss_points[1].y, 7.5);
        match back.job.last_event {
            Some(TrainingEvent::Progress { step, .. }) => assert_eq!(step, 5),
            _ => panic!("expected Progress event"),
        }
    }
}
