use std::io::BufRead;
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
        .route("/api/datasets/download", post(dataset_download))
        .route("/api/datasets/rename", post(dataset_rename))
        .route("/api/datasets/combine", post(dataset_combine))
        .route("/api/datasets/info/{name}", get(dataset_info))
        .route("/api/datasets/delete/{name}", axum::routing::delete(dataset_delete))
        .route("/api/models", get(models))
        .route("/api/train", post(train))
        .route("/api/train/status", get(train_status))
        .route("/api/train/stop", post(train_stop))
        .route("/api/train/stream", get(train_stream))
        .route("/api/download/{path}", get(download))
        .route("/api/generate", post(generate_text))
        .route("/api/export", post(export_model))
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

fn check_token(server: &ServerState, headers: &HeaderMap, query: Option<&str>) -> Result<(), Response> {
    let expected = server.token.read().unwrap().clone();
    if let Some(expected) = expected {
        let provided = headers
            .get("x-api-token")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| query_token(query).unwrap_or_default());
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

fn query_token(query: Option<&str>) -> Option<String> {
    let query = query?;
    for pair in query.split('&') {
        let mut parts = pair.splitn(2, '=');
        if parts.next() == Some("token") {
            return parts.next().map(|v| v.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct DeviceInfo {
    name: String,
    backend: String,
    device_type: String,
    vendor: u32,
    device: u32,
    driver: String,
    vram_bytes: Option<u64>,
    selectable: bool,
}

fn enum_adapter_specs() -> Vec<DeviceInfo> {
    use wgpu::Backends;

    let instance = wgpu::Instance::default();

    let adapters = futures::executor::block_on(instance.enumerate_adapters(Backends::all()));
    if adapters.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for a in adapters {
        let info = a.get_info();
        let backend_name = match info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DX12",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "Other",
        };
        let type_name = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => "Discrete GPU",
            wgpu::DeviceType::IntegratedGpu => "Integrated GPU",
            wgpu::DeviceType::VirtualGpu => "Virtual GPU",
            wgpu::DeviceType::Cpu => "CPU",
            _ => "Other",
        };

        let mut vram_bytes = None;
        if let Ok((device, _queue)) = futures::executor::block_on(
            a.request_device(&wgpu::DeviceDescriptor::default()),
        ) {
            let limits = device.limits();
            let max_buf = limits.max_buffer_size;
            if max_buf > 0 {
                vram_bytes = Some(max_buf);
            }
        }

        out.push(DeviceInfo {
            name: info.name.clone(),
            backend: backend_name.into(),
            device_type: type_name.into(),
            vendor: info.vendor,
            device: info.device,
            driver: info.driver.clone(),
            vram_bytes,
            selectable: true,
        });
    }
    out
}

#[allow(clippy::unused_async)]
async fn devices(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<DeviceInfo>>, Response> {
    check_token(&state.server, &headers, None)?;

    // WGPU adapter enumeration can be slow/blocking; run it off the async core.
    // Guard against panics from missing GPU drivers so the server stays up.
    let (tx, rx) = std::sync::mpsc::channel::<Vec<DeviceInfo>>();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(enum_adapter_specs));
        let _ = tx.send(result.unwrap_or_default());
    });
    let specs = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap_or_default();

    if specs.is_empty() {
        // Fallback: expose the selectors we know the runner supports.
        return Ok(Json(vec![
            DeviceInfo {
                name: "DefaultDevice".into(),
                backend: "wgpu".into(),
                device_type: "Auto".into(),
                vendor: 0,
                device: 0,
                driver: String::new(),
                vram_bytes: None,
                selectable: true,
            },
            DeviceInfo {
                name: "DiscreteGpu(0)".into(),
                backend: "wgpu".into(),
                device_type: "Discrete GPU".into(),
                vendor: 0,
                device: 0,
                driver: String::new(),
                vram_bytes: None,
                selectable: true,
            },
            DeviceInfo {
                name: "Cpu".into(),
                backend: "wgpu".into(),
                device_type: "CPU".into(),
                vendor: 0,
                device: 0,
                driver: String::new(),
                vram_bytes: None,
                selectable: true,
            },
        ]));
    }

    Ok(Json(specs))
}

// ---------------------------------------------------------------------------
// Datasets / models listing
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct DatasetInfo {
    name: String,
    size_bytes: u64,
    modified_unix: u64,
    ext: String,
}

fn list_dataset_infos(dir: &Path) -> Vec<DatasetInfo> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_file() {
                continue;
            }
            let ext = p.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();
            if ext != "txt" && ext != "bin" {
                continue;
            }
            let meta = match p.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            out.push(DatasetInfo {
                name: e.file_name().to_string_lossy().to_string(),
                size_bytes: meta.len(),
                modified_unix: modified,
                ext,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Resolve a user-supplied dataset name to a safe path inside the datasets dir.
/// Only a bare file name (no path separators / traversal) is accepted.
fn resolve_dataset(dir: &Path, name: &str) -> Result<std::path::PathBuf, Response> {
    let file_name = Path::new(name).file_name().and_then(|n| n.to_str());
    let file_name = match file_name {
        Some(n) if !n.is_empty() && n == name => n,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "invalid dataset path" })),
            )
                .into_response())
        }
    };

    let cand = dir.join(file_name);
    if !cand.is_file() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("dataset not found: {name}") })),
        )
            .into_response());
    }
    Ok(cand)
}

#[allow(clippy::unused_async)]
async fn datasets(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<Vec<DatasetInfo>>, Response> {
    check_token(&state.server, &headers, None)?;
    Ok(Json(list_dataset_infos(&state.server.datasets_dir)))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DatasetDownloadRequest {
    repo_id: String,
    #[serde(default = "default_download_filename")]
    filename: String,
}

fn default_download_filename() -> String { "TinyStories-train.txt".into() }

#[allow(clippy::unused_async)]
async fn dataset_download(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DatasetDownloadRequest>,
) -> Result<Json<serde_json::Value>, Response> {
    check_token(&state.server, &headers, None)?;
    let repo_id = req.repo_id.trim().to_string();
    if repo_id.is_empty() {
        return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "repoId is required" }))).into_response());
    }

    let (tx, rx) = std::sync::mpsc::channel::<Result<std::path::PathBuf, String>>();
    std::thread::spawn(move || {
        let _ = tx.send(
            crate::data::download_dataset_from_hf(&repo_id, &req.filename)
                .map_err(|e| e.to_string()),
        );
    });

    match rx.recv_timeout(std::time::Duration::from_secs(300)) {
        Ok(Ok(path)) => Ok(Json(serde_json::json!({
            "status": "ok",
            "path": path.to_string_lossy(),
            "name": path.file_name().unwrap_or_default().to_string_lossy(),
        }))),
        Ok(Err(e)) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))).into_response()),
        Err(_) => Err((StatusCode::GATEWAY_TIMEOUT, Json(serde_json::json!({ "error": "download timed out" }))).into_response()),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DatasetRenameRequest {
    from: String,
    to: String,
}

#[allow(clippy::unused_async)]
async fn dataset_rename(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DatasetRenameRequest>,
) -> Result<Json<serde_json::Value>, Response> {
    check_token(&state.server, &headers, None)?;
    let from = resolve_dataset(&state.server.datasets_dir, &req.from)?;

    let to_name = Path::new(&req.to).file_name().and_then(|n| n.to_str());
    let to_name = match to_name {
        Some(n) if !n.is_empty() && n == req.to => n,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "invalid target name" })),
            )
                .into_response())
        }
    };
    let to = state.server.datasets_dir.join(to_name);
    if to.exists() {
        return Err((StatusCode::CONFLICT, Json(serde_json::json!({ "error": "target already exists" }))).into_response());
    }
    std::fs::rename(&from, &to).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response()
    })?;
    Ok(Json(serde_json::json!({ "status": "ok", "from": req.from, "to": req.to })))
}

#[allow(clippy::unused_async)]
async fn dataset_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<serde_json::Value>, Response> {
    check_token(&state.server, &headers, None)?;
    let path = resolve_dataset(&state.server.datasets_dir, &name)?;
    std::fs::remove_file(&path).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response()
    })?;
    Ok(Json(serde_json::json!({ "status": "ok", "deleted": name })))
}

#[allow(clippy::unused_async)]
async fn dataset_info(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<serde_json::Value>, Response> {
    check_token(&state.server, &headers, None)?;
    let path = resolve_dataset(&state.server.datasets_dir, &name)?;

    let size_bytes = path.metadata().map(|m| m.len()).unwrap_or(0);
    let ext = path.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();

    let mut info = serde_json::json!({
        "name": name,
        "size_bytes": size_bytes,
        "ext": ext,
    });

    // Cheap text stats: line count (for .txt) without loading fully.
    if ext == "txt" {
        if let Ok(f) = std::fs::File::open(&path) {
            let reader = std::io::BufReader::new(f);
            let lines = reader.lines().count();
            info["lines"] = serde_json::json!(lines);
        }
    }

    Ok(Json(info))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DatasetCombineRequest {
    sources: Vec<CombineSource>,
    #[serde(default)]
    max_chars: Option<usize>,
    #[serde(default = "default_combine_output")]
    output: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CombineSource {
    path: String,
    weight: f32,
}

fn default_combine_output() -> String { "weighted_mix_web.txt".into() }

#[allow(clippy::unused_async)]
async fn dataset_combine(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DatasetCombineRequest>,
) -> Result<Json<serde_json::Value>, Response> {
    check_token(&state.server, &headers, None)?;
    if req.sources.is_empty() {
        return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "sources is required" }))).into_response());
    }

    let mut sources = Vec::new();
    for s in &req.sources {
        let p = resolve_dataset(&state.server.datasets_dir, &s.path)?;
        sources.push((p, s.weight));
    }

    let output_path = state.server.datasets_dir.join(&req.output);
    let max_chars = req.max_chars.unwrap_or(usize::MAX);
    let output_clone = output_path.clone();
    let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
    std::thread::spawn(move || {
        let result = crate::data::combine_and_mix_weighted(&sources, &output_clone, max_chars)
            .map(|_| ())
            .map_err(|e| e.to_string());
        let _ = tx.send(result);
    });

    match rx.recv_timeout(std::time::Duration::from_secs(600)) {
        Ok(Ok(())) => Ok(Json(serde_json::json!({
            "status": "ok",
            "output": output_path.to_string_lossy(),
        }))),
        Ok(Err(e)) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))).into_response()),
        Err(_) => Err((StatusCode::GATEWAY_TIMEOUT, Json(serde_json::json!({ "error": "combine timed out" }))).into_response()),
    }
}

#[derive(Serialize)]
struct ModelInfo {
    dir: String,
    name: String,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    n_ctx: usize,
    vocab: usize,
    params_m: f64,
    size_mb: f64,
    default: bool,
}

fn model_meta(dir: &Path) -> Option<ModelInfo> {
    let config_path = dir.join("config.json");
    let weights_path = dir.join("model_weights.bin");
    if !config_path.is_file() || !weights_path.is_file() {
        return None;
    }
    let cfg: crate::config::ModelConfig = serde_json::from_str(&std::fs::read_to_string(config_path).ok()?).ok()?;
    let size_mb = weights_path.metadata().ok()?.len() as f64 / (1024.0 * 1024.0);

    let n_embd = cfg.n_embd;
    let n_head = cfg.n_head;
    let n_kv_head = cfg.get_n_kv_head();
    let head_dim = n_embd / n_head;
    let kv_dim = n_kv_head * head_dim;
    let hidden = ((8 * n_embd) / 3).div_ceil(256) * 256;
    let embed = cfg.vocab_size * n_embd;
    let q = n_embd * n_embd + n_embd;
    let k = n_embd * kv_dim + kv_dim;
    let v = n_embd * kv_dim + kv_dim;
    let o = n_embd * n_embd;
    let block = n_embd + q + k + v + o + n_embd + n_embd * hidden + n_embd * hidden + hidden * n_embd;
    let params = embed + cfg.n_layer * block + n_embd;

    Some(ModelInfo {
        dir: dir.to_string_lossy().to_string(),
        name: dir.file_name().unwrap_or_default().to_string_lossy().to_string(),
        n_embd,
        n_head,
        n_layer: cfg.n_layer,
        n_ctx: cfg.n_ctx,
        vocab: cfg.vocab_size,
        params_m: params as f64 / 1_000_000.0,
        size_mb,
        default: dir.file_name().unwrap_or_default() == "model_output",
    })
}

#[allow(clippy::unused_async)]
async fn models(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<Vec<ModelInfo>>, Response> {
    check_token(&state.server, &headers, None)?;
    let mut out: Vec<ModelInfo> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&state.server.models_dir) {
        for e in rd.flatten() {
            let dir = e.path();
            if !dir.is_dir() {
                continue;
            }
            if let Some(info) = model_meta(&dir) {
                out.push(info);
            }
        }
    }
    out.sort_by(|a, b| b.default.cmp(&a.default).then(a.name.cmp(&b.name)));
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
    check_token(&state.server, &headers, None)?;

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
    check_token(&state.server, &headers, None)?;
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
    check_token(&state.server, &headers, None)?;
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
    check_token(&state.server, &headers, None)?;
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
    axum::extract::Query(query): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Response, Response> {
    check_token(&state.server, &headers, query.get("token").map(|s| s.as_str()))?;
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

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateRequest {
    model_dir: String,
    prompt: String,
    #[serde(default = "default_max_new_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temp")]
    temperature: f32,
    #[serde(default = "default_topk")]
    top_k: usize,
    #[serde(default = "default_topp")]
    top_p: f32,
    #[serde(default = "default_minp")]
    min_p: f32,
    #[serde(default = "default_rep")]
    repetition_penalty: f32,
}

fn default_max_new_tokens() -> usize { 60 }
fn default_temp() -> f32 { 0.8 }
fn default_topk() -> usize { 40 }
fn default_topp() -> f32 { 0.9 }
fn default_minp() -> f32 { 0.05 }
fn default_rep() -> f32 { 1.2 }

#[allow(clippy::unused_async)]
async fn generate_text(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<serde_json::Value>, Response> {
    check_token(&state.server, &headers, None)?;

    let model_dir = if req.model_dir.is_empty() {
        state.server.models_dir.join("model_output")
    } else {
        Path::new(&req.model_dir).to_path_buf()
    };

    let state_models = state.server.models_dir.clone();
    let model_dir = if model_dir.is_absolute() {
        model_dir
    } else if model_dir.exists() {
        model_dir
    } else {
        state_models.join(model_dir)
    };

    let prompt = req.prompt;
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_k = req.top_k;
    let top_p = req.top_p;
    let min_p = req.min_p;
    let rep = req.repetition_penalty;

    let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();
    std::thread::spawn(move || {
        let result = run_generate(model_dir, &prompt, max_tokens, temperature, top_k, top_p, min_p, rep);
        let _ = tx.send(result);
    });

    match rx.recv() {
        Ok(Ok(text)) => Ok(Json(serde_json::json!({ "text": text }))),
        Ok(Err(e)) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))).into_response()),
        Err(_) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": "generate thread died" }))).into_response()),
    }
}

fn run_generate(
    model_dir: std::path::PathBuf,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rep: f32,
) -> Result<String, String> {
    use burn::backend::Wgpu;
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model_weights");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("config.json not found in {}: {e}", model_dir.display()))?;
    let config: crate::config::ModelConfig =
        serde_json::from_str(&config_str).map_err(|e| format!("bad config: {e}"))?;

    let tokenizer_path = "custom_tokenizer.json";
    let tokenizer = crate::data::LlmTokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("tokenizer not found: {e}"))?;

    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    let model: crate::model::Llm<Wgpu> = config.init_model(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(weights_path.clone(), &device)
        .map_err(|e| format!("failed to load weights: {e:?}"))?;
    let model = model.load_record(record);

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);

    crate::generate::generate(
        &model, &tokenizer, prompt, max_tokens, temperature, top_k, top_p, min_p, rep, seed, &device,
    )
    .map_err(|e| format!("generate error: {e}"))
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportRequest {
    model_dir: String,
    #[serde(default = "default_model_name")]
    model_name: String,
}

fn default_model_name() -> String { "kuro-neko".into() }

#[allow(clippy::unused_async)]
async fn export_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ExportRequest>,
) -> Result<Json<serde_json::Value>, Response> {
    check_token(&state.server, &headers, None)?;

    let model_dir = if req.model_dir.is_empty() {
        state.server.models_dir.join("model_output")
    } else {
        Path::new(&req.model_dir).to_path_buf()
    };
    let model_dir = if model_dir.is_absolute() || model_dir.exists() {
        model_dir
    } else {
        state.server.models_dir.join(model_dir)
    };

    let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();
    let name = req.model_name;
    std::thread::spawn(move || {
        let _ = tx.send(run_export(model_dir, &name));
    });

    match rx.recv() {
        Ok(Ok(path)) => Ok(Json(serde_json::json!({ "gguf": path }))),
        Ok(Err(e)) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))).into_response()),
        Err(_) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": "export thread died" }))).into_response()),
    }
}

fn run_export(model_dir: std::path::PathBuf, model_name: &str) -> Result<String, String> {
    use burn::backend::Wgpu;
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model_weights");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("config.json not found in {}: {e}", model_dir.display()))?;
    let config: crate::config::ModelConfig =
        serde_json::from_str(&config_str).map_err(|e| format!("bad config: {e}"))?;

    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    let model: crate::model::Llm<Wgpu> = config.init_model(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(weights_path.clone(), &device)
        .map_err(|e| format!("failed to load weights: {e:?}"))?;
    let model = model.load_record(record);

    crate::export::export_to_gguf(&model, &config, "custom_tokenizer.json", model_name, "exported")
        .map_err(|e| format!("export error: {e}"))
}
