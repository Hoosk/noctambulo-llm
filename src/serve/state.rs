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

/// Snapshot of the currently running (or last) training job.
#[derive(Debug, Clone, Default)]
pub struct JobStatus {
    pub running: bool,
    pub stopping: bool,
    pub job_id: String,
    pub last_event: Option<TrainingEvent>,
    pub message: Option<String>,
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
}

#[derive(Default)]
struct LogBuffer {
    lines: VecDeque<LogLine>,
    total: usize,
}

const MAX_LOG_LINES: usize = 5000;

impl ServerState {
    pub fn new(token: Option<String>, datasets_dir: PathBuf, models_dir: PathBuf) -> Self {
        let (tx, _rx) = tokio::sync::broadcast::channel::<TrainingEvent>(1024);
        Self {
            token: RwLock::new(token),
            datasets_dir,
            models_dir,
            job: RwLock::new(JobStatus::default()),
            events: tx,
            logs: RwLock::new(LogBuffer::default()),
        }
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
}
