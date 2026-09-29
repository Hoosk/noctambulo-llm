use std::path::Path;

use axum::{Json, extract::State, http::HeaderMap};
use serde::Serialize;

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;
use super::paths::is_safe_rel_path;

#[derive(Serialize)]
pub(super) struct ModelInfo {
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
    let cfg: crate::config::ModelConfig =
        serde_json::from_str(&std::fs::read_to_string(config_path).ok()?).ok()?;
    let size_mb = weights_path.metadata().ok()?.len() as f64 / (1024.0 * 1024.0);

    let params = cfg.param_count();

    Some(ModelInfo {
        dir: dir.to_string_lossy().to_string(),
        name: dir.file_name().unwrap_or_default().to_string_lossy().to_string(),
        n_embd: cfg.n_embd,
        n_head: cfg.n_head,
        n_layer: cfg.n_layer,
        n_ctx: cfg.n_ctx,
        vocab: cfg.vocab_size,
        params_m: params as f64 / 1_000_000.0,
        size_mb,
        default: dir.file_name().unwrap_or_default() == "model_output",
    })
}

#[allow(clippy::unused_async)]
pub(super) async fn models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<ModelInfo>>, ApiError> {
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

#[derive(Serialize)]
pub(super) struct CheckpointInfo {
    name: String,
    path: String,
    size_bytes: u64,
    modified_unix: u64,
}

/// List `.bin` checkpoint files inside a user-provided directory (default
/// `checkpoints_web`). Path is resolved against the server working directory.
#[allow(clippy::unused_async)]
pub(super) async fn checkpoints(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Query(query): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Vec<CheckpointInfo>>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let dir = query.get("dir").map(String::as_str).unwrap_or("checkpoints_web");
    if !is_safe_rel_path(dir) {
        return Err(ApiError::bad_request("invalid dir"));
    }
    let dir = Path::new(dir);
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_file() {
                continue;
            }
            let ext = p.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();
            if ext != "bin" {
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
            out.push(CheckpointInfo {
                name: e.file_name().to_string_lossy().to_string(),
                path: p.to_string_lossy().to_string(),
                size_bytes: meta.len(),
                modified_unix: modified,
            });
        }
    }
    out.sort_by_key(|c| std::cmp::Reverse(c.modified_unix));
    Ok(Json(out))
}
