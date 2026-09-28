use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use crate::data::LlmTokenizer;
use crate::train::{TrainingEvent, train_with_events};
use crate::serve::state::{PERSIST_INTERVAL, ServerState, forces_persist};

/// Parameters of a training job, mirroring the CLI `train` subcommand.
#[derive(Debug, Clone)]
pub struct TrainParams {
    pub file: Option<PathBuf>,
    pub dtype: crate::config::DType,
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
    pub device: String,
    pub early_stop_patience: usize,
}

/// Parse the machine-readable device selector (`DiscreteGpu(0)`, `Cpu`,
/// `DefaultDevice`, ...) into a `WgpuDevice`. Unknown selectors fall back to
/// the default so a stale frontend can never crash the worker.
fn parse_device(selector: &str) -> burn::backend::wgpu::WgpuDevice {
    use burn::backend::wgpu::WgpuDevice;
    let s = selector.trim();
    if s.eq_ignore_ascii_case("cpu") {
        return WgpuDevice::Cpu;
    }
    for (prefix, mk) in [
        ("DiscreteGpu(", WgpuDevice::DiscreteGpu as fn(usize) -> WgpuDevice),
        ("IntegratedGpu(", WgpuDevice::IntegratedGpu as fn(usize) -> WgpuDevice),
        ("VirtualGpu(", WgpuDevice::VirtualGpu as fn(usize) -> WgpuDevice),
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            if let Some(idx) = rest.trim_end_matches(')').parse::<usize>().ok() {
                return mk(idx);
            }
        }
    }
    WgpuDevice::DefaultDevice
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
    let job_id = format!("job-{}", start_unix());

    // Claim the single job slot synchronously (closes the double-launch race)
    // and reset per-job state (loss curve, log buffer).
    state.begin_job(&job_id)?;

    // Load the tokenizer BEFORE spawning any thread so a failure leaves no
    // orphaned registry handle or forwarder behind.
    let worker_tokenizer = match load_tokenizer() {
        Ok(t) => t,
        Err(e) => {
            state.abort_job(e.clone());
            return Err(e);
        }
    };

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
        let mut last_persist: Option<std::time::Instant> = None;
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
                    job.stopping = false;
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
            // Throttle disk writes: persistence is needed for restart/backfill,
            // not on every progress tick.
            let due = last_persist
                .map(|t| t.elapsed() >= PERSIST_INTERVAL)
                .unwrap_or(true);
            if forces_persist(&event) || due {
                forwarder_state.persist();
                last_persist = Some(std::time::Instant::now());
            }
            let _ = forwarder_events.send(event);
        }
    });

    let worker_state = state.clone();
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
        let device = parse_device(&params.device);
        let _ = tx.send(TrainingEvent::LogEvent {
            line: format!("Compute precision: {}", params.dtype.as_str()),
            ts: start_unix(),
        });

        // Dispatch the generic trainer over the requested compute precision.
        macro_rules! run_training {
            ($backend:ty) => {
                train_with_events::<$backend>(
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
                    params.early_stop_patience,
                    &tx,
                    &worker_cancel,
                    Some(&worker_pause),
                )
            };
        }

        let result = match params.dtype {
            crate::config::DType::F32 => run_training!(burn::backend::Autodiff<burn::backend::Wgpu>),
            crate::config::DType::F16 => run_training!(
                burn::backend::Autodiff<crate::backend::WgpuHalf<burn::tensor::f16>>
            ),
        };

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::WgpuDevice;

    #[test]
    fn test_parse_device_selectors() {
        assert!(matches!(parse_device("DefaultDevice"), WgpuDevice::DefaultDevice));
        assert!(matches!(parse_device("Cpu"), WgpuDevice::Cpu));
        assert!(matches!(parse_device("DiscreteGpu(0)"), WgpuDevice::DiscreteGpu(0)));
        assert!(matches!(parse_device("DiscreteGpu(2)"), WgpuDevice::DiscreteGpu(2)));
        assert!(matches!(parse_device("IntegratedGpu(1)"), WgpuDevice::IntegratedGpu(1)));
        assert!(matches!(parse_device("VirtualGpu(3)"), WgpuDevice::VirtualGpu(3)));
    }

    #[test]
    fn test_parse_device_falls_back_on_garbage() {
        // Unknown / malformed selectors must never crash the worker.
        assert!(matches!(parse_device(""), WgpuDevice::DefaultDevice));
        assert!(matches!(parse_device("Intel Arc"), WgpuDevice::DefaultDevice));
        assert!(matches!(parse_device("DiscreteGpu(x)"), WgpuDevice::DefaultDevice));
        assert!(matches!(parse_device("../../etc"), WgpuDevice::DefaultDevice));
    }
}
