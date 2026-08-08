use std::path::Path;
use std::sync::Arc;

use axum::{
    Router,
    extract::{Path as AxumPath, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response, sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
    Json,
};
use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::serve::runner::{JobRegistry, TrainParams, launch_job};
use crate::serve::state::ServerState;
use crate::train::TrainingEvent;

pub fn router(state: Arc<ServerState>) -> Router {
    let registry = Arc::new(JobRegistry::new());
    Router::new()
        .route("/api/devices", get(devices))
        .route("/api/datasets", get(datasets))
        .route("/api/models", get(models))
        .route("/api/train", post(train))
        .route("/api/train/status", get(train_status))
        .route("/api/train/stop", post(train_stop))
        .route("/api/train/stream", get(train_stream))
        .route("/api/download/{path}", get(download))
        .with_state(AppState {
            server: state,
            registry,
        })
}

#[derive(Clone)]
struct AppState {
    server: Arc<ServerState>,
    registry: Arc<JobRegistry>,
}

fn check_token(server: &ServerState, headers: &HeaderMap) -> Result<(), Response> {
    let expected = server.token.read().unwrap().clone();
    if let Some(expected) = expected {
        let provided = headers
            .get("x-api-token")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if provided != expected {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "invalid or missing token" })),
            )
                .into_response());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct DeviceInfo {
    name: String,
}

#[allow(clippy::unused_async)]
async fn devices(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<DeviceInfo>>, Response> {
    check_token(&state.server, &headers)?;
    Ok(Json(vec![
        DeviceInfo { name: "DefaultDevice".into() },
        DeviceInfo { name: "DiscreteGpu(0)".into() },
        DeviceInfo { name: "DiscreteGpu(1)".into() },
        DeviceInfo { name: "IntegratedGpu(0)".into() },
        DeviceInfo { name: "Cpu".into() },
    ]))
}

// ---------------------------------------------------------------------------
// Datasets / models listing
// ---------------------------------------------------------------------------

fn list_entries(dir: &Path, exts: &[&str], relative: bool) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_file()
                && exts.iter().any(|x| {
                    p.extension()
                        .map(|e| e.to_string_lossy().eq_ignore_ascii_case(x))
                        .unwrap_or(false)
                })
            {
                if relative {
                    out.push(e.file_name().to_string_lossy().to_string());
                } else {
                    out.push(p.to_string_lossy().to_string());
                }
            }
        }
    }
    out.sort();
    out
}

#[allow(clippy::unused_async)]
async fn datasets(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<Vec<String>>, Response> {
    check_token(&state.server, &headers)?;
    Ok(Json(list_entries(&state.server.datasets_dir, &["txt", "bin"], true)))
}

#[allow(clippy::unused_async)]
async fn models(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<Vec<String>>, Response> {
    check_token(&state.server, &headers)?;
    let mut out = list_entries(&state.server.models_dir, &["json"], false);
    if let Ok(rd) = std::fs::read_dir(&state.server.models_dir) {
        for e in rd.flatten() {
            if e.path().is_dir() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("model_output") || name.starts_with("checkpoint") {
                    out.push(format!("{}/config.json", e.path().to_string_lossy()));
                }
            }
        }
    }
    out.sort();
    out.dedup();
    Ok(Json(out))
}

// ---------------------------------------------------------------------------
// Train
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrainRequest {
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
    #[serde(default)]
    save_dir: Option<String>,
    #[serde(default)]
    resume: Option<String>,
    #[serde(default = "default_output_dir")]
    output_dir: String,
}

fn default_epochs() -> usize { 1 }
fn default_batch() -> usize { 32 }
fn default_lr() -> f64 { 5e-5 }
fn default_val_split() -> f32 { 0.02 }
fn default_grad_accum() -> usize { 1 }
fn default_nctx() -> usize { 256 }
fn default_nembd() -> usize { 256 }
fn default_nhead() -> usize { 4 }
fn default_nlayer() -> usize { 4 }
fn default_dropout() -> f64 { 0.1 }
fn default_output_dir() -> String { "model_output".into() }

#[allow(clippy::unused_async)]
async fn train(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainRequest>,
) -> Result<Json<serde_json::Value>, Response> {
    check_token(&state.server, &headers)?;

    let file_path = {
        let p = Path::new(&req.file);
        if p.is_absolute() || p.components().count() > 1 {
            p.to_path_buf()
        } else {
            state.server.datasets_dir.join(p)
        }
    };
    if !file_path.exists() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("dataset not found: {}", file_path.display()) })),
        )
            .into_response());
    }

    let params = TrainParams {
        file: Some(file_path),
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
        save_dir: req.save_dir.map(path_buf_from),
        resume: req.resume.map(path_buf_from),
        output_dir: req.output_dir,
    };

    match launch_job(&state.server, &state.registry, params) {
        Ok(job_id) => Ok(Json(serde_json::json!({ "jobId": job_id, "status": "started" }))),
        Err(e) => Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": e })),
        )
            .into_response()),
    }
}

fn path_buf_from(s: String) -> std::path::PathBuf {
    std::path::PathBuf::from(s)
}

// ---------------------------------------------------------------------------
// Status / stop / SSE
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct StatusResponse {
    running: bool,
    stopping: bool,
    job_id: String,
    message: Option<String>,
    last_event: Option<TrainingEvent>,
}

#[allow(clippy::unused_async)]
async fn train_status(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<StatusResponse>, Response> {
    check_token(&state.server, &headers)?;
    let job = state.server.job.read().unwrap().clone();
    Ok(Json(StatusResponse {
        running: job.running,
        stopping: job.stopping,
        job_id: job.job_id,
        message: job.message,
        last_event: job.last_event,
    }))
}

#[allow(clippy::unused_async)]
async fn train_stop(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<serde_json::Value>, Response> {
    check_token(&state.server, &headers)?;
    {
        let mut job = state.server.job.write().unwrap();
        job.stopping = true;
    }
    state.registry.request_stop();
    Ok(Json(serde_json::json!({ "status": "stopping" })))
}

fn event_stream(state: Arc<ServerState>) -> impl Stream<Item = Result<Event, axum::Error>> + Send + 'static {
    let mut rx = state.events.subscribe();
    async_stream::stream! {
        while let Ok(ev) = rx.recv().await {
            let json = serde_json::to_string(&ev).unwrap_or_default();
            yield Ok(Event::default().event("training").data(json));
        }
    }
}

async fn train_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, Response> {
    check_token(&state.server, &headers)?;
    Ok(Sse::new(event_stream(state.server.clone()))
        .keep_alive(KeepAlive::default())
        .into_response())
}

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------

#[allow(clippy::unused_async)]
async fn download(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(path): AxumPath<String>,
) -> Result<Response, Response> {
    check_token(&state.server, &headers)?;
    let p = Path::new(&path);
    if !p.exists() {
        return Err((StatusCode::NOT_FOUND, "not found").into_response());
    }
    let Ok(bytes) = tokio::fs::read(p).await else {
        return Err((StatusCode::INTERNAL_SERVER_ERROR, "read error").into_response());
    };
    let fname = p.file_name().unwrap_or_default().to_string_lossy().to_string();
    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_DISPOSITION, format!("attachment; filename=\"{fname}\""))],
        bytes,
    )
        .into_response())
}
