use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}},
};
use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::config::DType;
use crate::serve::runner::{TrainParams, launch_job};
use crate::train::TrainingEvent;

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;
use super::paths::{is_safe_child_name, is_safe_rel_path};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TrainRequest {
    file: String,
    #[serde(default = "default_epochs")]
    epochs: usize,
    #[serde(default = "default_batch")]
    batch_size: usize,
    #[serde(default = "default_lr")]
    lr: f64,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default = "default_val_split")]
    val_split: f32,
    #[serde(default = "default_grad_accum")]
    grad_accum: usize,
    #[serde(default)]
    seq_warmup_steps: usize,
    #[serde(default)]
    early_stop_patience: usize,
    #[serde(default = "default_weight_decay")]
    weight_decay: f64,
    #[serde(default)]
    label_smoothing: f32,
    #[serde(default)]
    z_loss: f32,
    #[serde(default = "default_nctx")]
    n_ctx: usize,
    #[serde(default = "default_nembd")]
    n_embd: usize,
    #[serde(default = "default_nhead")]
    n_head: usize,
    #[serde(default)]
    n_kv_head: Option<usize>,
    #[serde(default = "default_nlayer")]
    n_layer: usize,
    #[serde(default = "default_dropout")]
    dropout: f64,
    #[serde(default)]
    attn_softcap: f32,
    /// Sliding attention window in tokens (0 = full causal attention).
    #[serde(default)]
    sliding_window: usize,
    /// Initial tokens always kept attendable (attention sinks).
    #[serde(default)]
    sink_tokens: usize,
    /// Layers per cross-layer K/V sharing group (0/1 = no sharing).
    #[serde(default)]
    kv_share_group: usize,
    /// MLA latent dimension (0 = disabled).
    #[serde(default)]
    mla_latent_dim: usize,
    #[serde(default)]
    save_dir: Option<String>,
    #[serde(default)]
    resume: Option<String>,
    #[serde(default = "default_output_dir")]
    output_dir: String,
    #[serde(default = "default_device")]
    device: String,
    #[serde(default = "default_dtype")]
    dtype: String,
}

fn default_device() -> String {
    "DefaultDevice".into()
}

fn default_dtype() -> String {
    "f32".into()
}

fn default_epochs() -> usize {
    1
}

fn default_batch() -> usize {
    32
}

fn default_lr() -> f64 {
    5e-5
}

fn default_val_split() -> f32 {
    0.02
}

fn default_grad_accum() -> usize {
    1
}

fn default_nctx() -> usize {
    256
}

fn default_nembd() -> usize {
    256
}

fn default_nhead() -> usize {
    4
}

fn default_nlayer() -> usize {
    4
}

fn default_dropout() -> f64 {
    0.1
}

fn default_weight_decay() -> f64 {
    0.1
}

fn default_output_dir() -> String {
    "model_output".into()
}

fn path_buf_from(s: String) -> std::path::PathBuf {
    std::path::PathBuf::from(s)
}

/// Resolve the dataset path for a run: bare names live in the datasets dir,
/// explicit paths must canonicalize inside it.
fn resolve_train_dataset(state: &AppState, file: &str) -> Result<std::path::PathBuf, ApiError> {
    let file_path = if is_safe_child_name(file) {
        state.server.datasets_dir.join(file)
    } else {
        let canonical = std::fs::canonicalize(file)
            .map_err(|_| ApiError::bad_request(format!("dataset not found: {file}")))?;
        let canonical_root = std::fs::canonicalize(&state.server.datasets_dir)
            .unwrap_or_else(|_| state.server.datasets_dir.clone());
        if !canonical.starts_with(&canonical_root) {
            return Err(ApiError::forbidden("dataset outside the datasets directory"));
        }
        canonical
    };
    if !file_path.is_file() {
        return Err(ApiError::bad_request(format!(
            "dataset not found: {}",
            file_path.display()
        )));
    }
    Ok(file_path)
}

#[allow(clippy::unused_async)]
pub(super) async fn train(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;

    let file_path = resolve_train_dataset(&state, &req.file)?;

    // Only relative, traversal-free paths are accepted for checkpoints and
    // output directories, resolved against the server working directory.
    if let Some(save_dir) = &req.save_dir {
        if !is_safe_rel_path(save_dir) {
            return Err(ApiError::bad_request("invalid save_dir"));
        }
    }
    if let Some(resume) = &req.resume {
        if !is_safe_rel_path(resume) {
            return Err(ApiError::bad_request("invalid resume path"));
        }
    }
    if !is_safe_rel_path(&req.output_dir) {
        return Err(ApiError::bad_request("invalid output_dir"));
    }

    let dtype = DType::parse(&req.dtype).map_err(ApiError::bad_request)?;

    let params = TrainParams {
        file: Some(file_path),
        dtype,
        epochs: req.epochs,
        batch_size: req.batch_size,
        lr: req.lr,
        max_tokens: req.max_tokens,
        val_split: req.val_split,
        grad_accum: req.grad_accum,
        seq_warmup_steps: req.seq_warmup_steps,
        n_ctx: req.n_ctx,
        n_embd: req.n_embd,
        n_head: req.n_head,
        n_kv_head: req.n_kv_head,
        n_layer: req.n_layer,
        dropout: req.dropout,
        attn_softcap: req.attn_softcap,
        sliding_window: req.sliding_window,
        sink_tokens: req.sink_tokens,
        kv_share_group: req.kv_share_group,
        mla_latent_dim: req.mla_latent_dim,
        save_dir: req.save_dir.map(path_buf_from),
        resume: req.resume.map(path_buf_from),
        output_dir: req.output_dir,
        device: req.device,
        early_stop_patience: req.early_stop_patience,
        weight_decay: req.weight_decay,
        label_smoothing: req.label_smoothing,
        z_loss: req.z_loss,
    };

    match launch_job(&state.server, &state.registry, params) {
        Ok(job_id) => Ok(Json(serde_json::json!({ "jobId": job_id, "status": "started" }))),
        Err(e) => Err(ApiError::conflict(e)),
    }
}

#[derive(Serialize)]
pub(super) struct StatusResponse {
    running: bool,
    stopping: bool,
    paused: bool,
    job_id: String,
    message: Option<String>,
    last_event: Option<TrainingEvent>,
    loss_points: Vec<crate::serve::state::LossPoint>,
    val_points: Vec<crate::serve::state::LossPoint>,
    /// Monotonic log cursor so clients can backfill missed lines without polling.
    logs_total: usize,
}

#[allow(clippy::unused_async)]
pub(super) async fn train_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<StatusResponse>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let job = state.server.job.read().unwrap().clone();
    Ok(Json(StatusResponse {
        running: job.running,
        stopping: job.stopping,
        paused: state.registry.is_paused(),
        job_id: job.job_id,
        message: job.message,
        last_event: job.last_event,
        loss_points: state.server.loss_points_snapshot(),
        val_points: state.server.val_points_snapshot(),
        logs_total: state.server.logs_total(),
    }))
}

#[derive(Serialize)]
pub(super) struct LogsResponse {
    lines: Vec<crate::serve::state::LogLine>,
    total: usize,
}

#[allow(clippy::unused_async)]
pub(super) async fn train_logs(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Query(query): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<LogsResponse>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let since = query
        .get("since")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let (lines, total) = state.server.log_since(since);
    Ok(Json(LogsResponse { lines, total }))
}

#[allow(clippy::unused_async)]
pub(super) async fn train_stop(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    {
        let mut job = state.server.job.write().unwrap();
        job.stopping = true;
    }
    state.registry.request_stop();
    Ok(Json(serde_json::json!({ "status": "stopping" })))
}

#[allow(clippy::unused_async)]
pub(super) async fn train_pause(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    state.registry.set_paused(true);
    Ok(Json(serde_json::json!({ "status": "paused" })))
}

#[allow(clippy::unused_async)]
pub(super) async fn train_resume(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    state.registry.set_paused(false);
    Ok(Json(serde_json::json!({ "status": "resumed" })))
}

fn event_stream(state: Arc<crate::serve::state::ServerState>) -> impl Stream<Item = Result<Event, axum::Error>> + Send + 'static {
    let mut rx = state.events.subscribe();
    async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    let json = serde_json::to_string(&ev).unwrap_or_default();
                    yield Ok(Event::default().event("training").data(json));
                }
                // A slow consumer missed events: skip them and keep streaming
                // instead of tearing the connection down.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    }
}

pub(super) async fn train_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    check_token(&state.server, &headers, None)?;
    Ok(Sse::new(event_stream(state.server.clone()))
        .keep_alive(KeepAlive::default())
        .into_response())
}
