use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::RwLock;

use crate::train::TrainingEvent;

use super::runs::{RUNS_FILE, RunsStore};

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

/// Everything persisted to `.noctambulo/state.json` between server restarts.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct PersistedState {
    job: JobStatus,
    loss_points: Vec<LossPoint>,
    #[serde(default)]
    val_points: Vec<LossPoint>,
}

/// Minimum interval between `persist()` writes during a running job. Progress
/// events arrive at least twice per second; rewriting the state file on each of
/// them is pure disk churn.
pub const PERSIST_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// Rare or terminal events must be written immediately (they carry state the
/// UI/backfill needs even if the server restarts right after).
pub fn forces_persist(event: &TrainingEvent) -> bool {
    matches!(
        event,
        TrainingEvent::Done { .. }
            | TrainingEvent::Stopped { .. }
            | TrainingEvent::Checkpoint { .. }
            | TrainingEvent::Val { .. }
            | TrainingEvent::EpochComplete { .. }
    )
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
    /// Validation-loss history, persisted alongside the training loss curve.
    val_points: RwLock<Vec<LossPoint>>,
    /// Persisted history of training runs (curves included).
    pub runs: RunsStore,
}

#[derive(Default)]
struct LogBuffer {
    lines: VecDeque<LogLine>,
    total: usize,
}

const MAX_LOG_LINES: usize = 5000;
const STATE_FILE: &str = ".noctambulo/state.json";
const STATE_TMP: &str = ".noctambulo/state.json.tmp";

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
            val_points: RwLock::new(Vec::new()),
            runs: RunsStore::load(PathBuf::from(RUNS_FILE)),
        };
        state.load_state();
        state
    }

    /// Atomically claim the single job slot for a new job and reset per-job
    /// state (loss curve, log buffer, last event). The synchronous claim
    /// closes the double-launch race window.
    pub fn begin_job(&self, job_id: &str) -> Result<(), String> {
        {
            let mut job = self.job.write().unwrap();
            if job.running {
                return Err("A training job is already running.".into());
            }
            job.running = true;
            job.stopping = false;
            job.job_id = job_id.to_string();
            job.last_event = None;
            job.message = None;
        }
        self.loss_points.write().unwrap().clear();
        self.val_points.write().unwrap().clear();
        // Clear kept log lines but keep `total` monotonic so clients holding an
        // old cursor never re-receive the previous job's lines.
        self.logs.write().unwrap().lines.clear();
        Ok(())
    }

    /// Release the job slot after a failed launch (nothing was spawned).
    pub fn abort_job(&self, message: String) {
        let mut job = self.job.write().unwrap();
        job.running = false;
        job.message = Some(message);
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

    /// Monotonic count of log lines ever captured (survives buffer eviction and
    /// job restarts), used by clients to backfill incrementally.
    pub fn logs_total(&self) -> usize {
        self.logs.read().unwrap().total
    }

    /// Record a loss point (deduped by step). Provisional zeros and NaN/inf
    /// values from the trainer are not real metrics, so they are dropped.
    pub fn push_loss_point(&self, x: usize, y: f32) {
        if !y.is_finite() || y <= 0.0 {
            return;
        }
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

    /// Snapshot of the loss history for chart backfill.
    pub fn loss_points_snapshot(&self) -> Vec<LossPoint> {
        self.loss_points.read().unwrap().clone()
    }

    /// Record a validation-loss point (deduped by step, same filtering as the
    /// training loss).
    pub fn push_val_point(&self, x: usize, y: f32) {
        if !y.is_finite() || y <= 0.0 {
            return;
        }
        let mut pts = self.val_points.write().unwrap();
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

    /// Snapshot of the validation-loss history for chart backfill.
    pub fn val_points_snapshot(&self) -> Vec<LossPoint> {
        self.val_points.read().unwrap().clone()
    }

    /// Persist job status + loss history to `.noctambulo/state.json`.
    /// Written atomically (temp + rename) so a crash mid-write can never
    /// leave a truncated file that would wipe the state on next load.
    pub fn persist(&self) {
        let state = PersistedState {
            job: self.job.read().unwrap().clone(),
            loss_points: self.loss_points_snapshot(),
            val_points: self.val_points_snapshot(),
        };
        if let Ok(bytes) = serde_json::to_vec(&state) {
            if std::fs::create_dir_all(".noctambulo").is_ok() && std::fs::write(STATE_TMP, bytes).is_ok() {
                let _ = std::fs::rename(STATE_TMP, STATE_FILE);
            }
        }
    }

    /// Load `.noctambulo/state.json` if present. A restarted server is never running;
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
        *self.val_points.write().unwrap() = state.val_points;
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
            val_points: vec![LossPoint { x: 2, y: 7.8 }],
        };

        let bytes = serde_json::to_vec(&state).unwrap();
        let back: PersistedState = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(back.job.job_id, "job-123");
        assert_eq!(back.loss_points.len(), 2);
        assert_eq!(back.loss_points[1].y, 7.5);
        assert_eq!(back.val_points.len(), 1);
        match back.job.last_event {
            Some(TrainingEvent::Progress { step, .. }) => assert_eq!(step, 5),
            _ => panic!("expected Progress event"),
        }
    }

    fn fresh_state() -> ServerState {
        let st = ServerState::new(None, PathBuf::from("d"), PathBuf::from("m"));
        // A real .noctambulo/state.json may exist in the working directory; tests
        // must not depend on (or mutate) it.
        st.loss_points.write().unwrap().clear();
        st.logs.write().unwrap().lines.clear();
        st.logs.write().unwrap().total = 0;
        st
    }

    #[test]
    fn test_log_since_incremental() {
        let st = fresh_state();
        for i in 0..10 {
            st.push_log(format!("l{i}"), i as u64);
        }
        let (lines, total) = st.log_since(0);
        assert_eq!((total, lines.len()), (10, 10));
        assert_eq!(lines[0].line, "l0");

        let (tail, _) = st.log_since(8);
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].line, "l8");

        let (none, _) = st.log_since(10);
        assert!(none.is_empty());
    }

    #[test]
    fn test_forces_persist_only_on_rare_events() {
        assert!(forces_persist(&TrainingEvent::Done {
            final_loss: 1.0,
            total_time_secs: 1.0,
            weights_path: "w".into(),
            config_path: "c".into(),
        }));
        assert!(forces_persist(&TrainingEvent::Stopped {
            reason: "stop".into(),
            epoch: 1,
            batch: 0,
        }));
        assert!(forces_persist(&TrainingEvent::Val { loss: 1.0, ppl: 2.0, bpb: Some(1.2), epoch: 1 }));
        assert!(!forces_persist(&TrainingEvent::LogEvent { line: "x".into(), ts: 0 }));
        assert!(!forces_persist(&TrainingEvent::Progress {
            step: 1,
            epoch: 1,
            epochs: 1,
            batch: 1,
            num_batches: 1,
            pct: 1.0,
            loss: 1.0,
            avg_loss: 1.0,
            lr: 1e-4,
            batches_per_sec: 1.0,
            elapsed_secs: 1.0,
            eta_secs: 1.0,
            current_n_ctx: 8,
        }));
    }

    #[test]
    fn test_push_loss_point_filters_and_dedups() {
        let st = fresh_state();
        st.push_loss_point(1, 0.0); // provisional zero: dropped
        st.push_loss_point(1, f32::NAN); // non-finite: dropped
        st.push_loss_point(1, f32::INFINITY);
        assert!(st.loss_points_snapshot().is_empty());

        st.push_loss_point(1, 5.0);
        st.push_loss_point(1, 4.5); // same step: overwrite, not append
        let pts = st.loss_points_snapshot();
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].y, 4.5);

        st.push_loss_point(2, 4.0);
        assert_eq!(st.loss_points_snapshot().len(), 2);
    }

    #[test]
    fn test_begin_job_claims_and_resets() {
        let st = fresh_state();
        st.push_loss_point(1, 1.0);
        st.push_log("old-line".into(), 0);

        st.begin_job("job-x").unwrap();
        assert!(st.loss_points_snapshot().is_empty());
        let (lines, total) = st.log_since(0);
        assert!(lines.is_empty()); // lines cleared...
        assert_eq!(total, 1); // ...but the cursor stays monotonic

        assert!(st.begin_job("job-y").is_err()); // slot is occupied

        st.abort_job("boom".into());
        assert!(!st.job.read().unwrap().running);
        st.begin_job("job-z").unwrap(); // released again
        assert!(st.job.read().unwrap().running);
        assert_eq!(st.job.read().unwrap().job_id, "job-z");
        st.abort_job("cleanup".into());
    }
}
