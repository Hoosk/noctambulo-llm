use std::path::PathBuf;
use std::sync::RwLock;

use crate::train::TrainingEvent;

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
}

impl ServerState {
    pub fn new(token: Option<String>, datasets_dir: PathBuf, models_dir: PathBuf) -> Self {
        let (tx, _rx) = tokio::sync::broadcast::channel::<TrainingEvent>(1024);
        Self {
            token: RwLock::new(token),
            datasets_dir,
            models_dir,
            job: RwLock::new(JobStatus::default()),
            events: tx,
        }
    }
}
