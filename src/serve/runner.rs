use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use crate::data::LlmTokenizer;
use crate::train::{TrainingEvent, train_with_events};
use crate::serve::state::ServerState;

/// Parameters of a training job, mirroring the CLI `train` subcommand.
#[derive(Debug, Clone)]
pub struct TrainParams {
    pub file: Option<PathBuf>,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub max_tokens: Option<usize>,
    pub val_split: f32,
    pub grad_accum: usize,
    pub seq_warmup_steps: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: Option<usize>,
    pub n_layer: usize,
    pub dropout: f64,
    pub attn_softcap: f32,
    pub save_dir: Option<PathBuf>,
    pub resume: Option<PathBuf>,
    pub output_dir: String,
}

/// Shared handle for the currently running job, used to request a graceful stop
/// or pause/resume.
pub struct JobHandle {
    pub cancel: Arc<AtomicBool>,
    pub paused: Arc<AtomicBool>,
    #[allow(dead_code)]
    pub job_id: String,
}

/// Registry of running jobs (only one at a time for now).
pub struct JobRegistry {
    inner: Mutex<Option<JobHandle>>,
}

impl JobRegistry {
    pub fn new() -> Self {
        Self { inner: Mutex::new(None) }
    }

    pub fn start(&self, handle: JobHandle) {
        *self.inner.lock().unwrap() = Some(handle);
    }

    pub fn request_stop(&self) {
        if let Some(h) = self.inner.lock().unwrap().as_ref() {
            h.cancel.store(true, Ordering::Relaxed);
        }
    }

    pub fn set_paused(&self, paused: bool) {
        if let Some(h) = self.inner.lock().unwrap().as_ref() {
            h.paused.store(paused, Ordering::Relaxed);
        }
    }

    pub fn is_paused(&self) -> bool {
        self.inner
            .lock()
            .unwrap()
            .as_ref()
            .map(|h| h.paused.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    pub fn clear(&self) {
        *self.inner.lock().unwrap() = None;
    }
}

impl Default for JobRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn start_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Launch a training job. Runs in a dedicated OS thread because Burn/WGPU tensors
/// are not `Send`. Events are forwarded to the SSE broadcast channel.
pub fn launch_job(state: &Arc<ServerState>, registry: &Arc<JobRegistry>, params: TrainParams) -> Result<String, String> {
    if state.job.read().unwrap().running {
        return Err("A training job is already running.".into());
    }

    let job_id = format!("job-{}", start_unix());
    let cancel = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(false));
    registry.start(JobHandle {
        cancel: cancel.clone(),
        paused: paused.clone(),
        job_id: job_id.clone(),
    });

    let (tx, rx) = mpsc::channel::<TrainingEvent>();

    let state_events = state.events.clone();
    let state_job = state.clone();

    // Forwarder: drains the worker channel and publishes to SSE + job status.
    let forwarder_state = state.clone();
    let forwarder_events = state.events.clone();
    let job_id_fwd = job_id.clone();
    let forwarder = std::thread::spawn(move || {
        let log_path = {
            std::fs::create_dir_all("logs").ok();
            PathBuf::from("logs").join(format!("train_{}.log", job_id_fwd))
        };
        while let Ok(event) = rx.recv() {
            {
                let mut job = forwarder_state.job.write().unwrap();
                job.running = true;
                job.job_id = job_id_fwd.clone();
                job.last_event = Some(event.clone());
                if matches!(
                    event,
                    TrainingEvent::Done { .. } | TrainingEvent::Stopped { .. }
                ) {
                    job.running = false;
                }
            }
            match &event {
                TrainingEvent::LogEvent { line, ts } => {
                    forwarder_state.push_log(line.clone(), *ts);
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&log_path)
                    {
                        use std::io::Write;
                        let _ = writeln!(f, "{line}");
                    }
                }
                TrainingEvent::Progress { step, loss, .. } => {
                    forwarder_state.push_loss_point(*step, *loss);
                }
                _ => {}
            }
            forwarder_state.persist();
            let _ = forwarder_events.send(event);
        }
    });

    let worker_state = state.clone();
    let worker_tokenizer = load_tokenizer()?;
    let worker_cancel = cancel.clone();
    let worker_pause = paused.clone();

    let worker = std::thread::spawn(move || {
        let config = crate::config::ModelConfig::new(
            worker_tokenizer.vocab_size(),
            params.n_embd,
            params.n_head,
            params.n_kv_head,
            params.n_layer,
            params.n_ctx,
            params.dropout,
            params.attn_softcap,
        );
        let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;

        let result = train_with_events::<burn::backend::Autodiff<burn::backend::Wgpu>>(
            params.file.as_deref(),
            None,
            config,
            &worker_tokenizer,
            params.epochs,
            params.batch_size,
            params.lr,
            None, // custom_step
            params.max_tokens,
            params.val_split,
            params.grad_accum,
            params.seq_warmup_steps,
            params.save_dir.as_deref(),
            params.resume.as_deref(),
            &device,
            &params.output_dir,
            &tx,
            &worker_cancel,
            Some(&worker_pause),
        );

        if let Err(e) = result {
            let _ = tx.send(TrainingEvent::Stopped {
                reason: format!("Training error: {e}"),
                epoch: 0,
                batch: 0,
            });
        }
        drop(tx);
    });

    // Keep handles alive / reap threads lazily.
    let reap_state = worker_state;
    let reap_registry = registry.clone();
    std::thread::spawn(move || {
        let _ = worker.join();
        let _ = forwarder.join();
        {
            let mut job = reap_state.job.write().unwrap();
            if !matches!(job.last_event, Some(TrainingEvent::Done { .. }) | Some(TrainingEvent::Stopped { .. })) {
                job.message = Some("Worker exited without Done/Stopped event.".into());
            }
        }
        reap_registry.clear();
    });

    let _ = state_events;
    let _ = state_job;
    Ok(job_id)
}

fn load_tokenizer() -> Result<LlmTokenizer, String> {
    let path = "custom_tokenizer.json";
    if std::path::Path::new(path).exists() {
        LlmTokenizer::from_file(path).map_err(|e| format!("Failed to load tokenizer: {e}"))
    } else {
        Err(format!("Tokenizer not found at {path}. Train it first."))
    }
}

/// Sleep helper (kept for potential polling loops).
#[allow(dead_code)]
fn _tick() {
    std::thread::sleep(Duration::from_millis(200));
}
