use std::path::Path;

use axum::{
    Json,
    extract::{Path as AxumPath, Query, State},
    http::{HeaderMap, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;
use super::paths::{is_safe_child_name, is_safe_rel_path, resolve_model_child_dir};

#[derive(Serialize)]
pub(super) struct ModelInfo {
    dir: String,
    name: String,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    n_ctx: usize,
    vocab: usize,
    ffn_hidden: usize,
    rope_base: f32,
    rms_norm_eps: f32,
    import_source: Option<String>,
    params_m: f64,
    size_mb: f64,
    default: bool,
    arch_version: u32,
    compatible: bool,
    /// True for Burn 0.21 `.bin` checkpoints, which this build cannot load.
    legacy_weights: bool,
    steps: Option<u64>,
    best_val_loss: Option<f64>,
}

fn model_meta(dir: &Path) -> Option<ModelInfo> {
    let config_path = dir.join("config.json");
    let weights_path = crate::storage::model_weights_path(dir);
    let legacy_weights = !weights_path.is_file()
        && crate::storage::legacy_weights_path(dir).is_some();
    if !config_path.is_file() || (!weights_path.is_file() && !legacy_weights) {
        return None;
    }
    let weights_path = if weights_path.is_file() {
        weights_path
    } else {
        crate::storage::legacy_weights_path(dir).unwrap()
    };
    let cfg: crate::config::ModelConfig =
        serde_json::from_str(&std::fs::read_to_string(config_path).ok()?).ok()?;
    let size_mb = weights_path.metadata().ok()?.len() as f64 / (1024.0 * 1024.0);

    let params = cfg.param_count();
    let (steps, best_val_loss) = std::fs::read_to_string(dir.join("train_state.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .map(|v| {
            (
                v.get("global_step").and_then(|x| x.as_u64()),
                v.get("best_val_loss").and_then(|x| x.as_f64()),
            )
        })
        .unwrap_or((None, None));

    Some(ModelInfo {
        dir: dir.to_string_lossy().to_string(),
        name: dir.file_name().unwrap_or_default().to_string_lossy().to_string(),
        n_embd: cfg.n_embd,
        n_head: cfg.n_head,
        n_layer: cfg.n_layer,
        n_ctx: cfg.n_ctx,
        vocab: cfg.vocab_size,
        ffn_hidden: cfg.ffn_hidden(),
        rope_base: cfg.rope_base,
        rms_norm_eps: cfg.rms_norm_eps,
        import_source: cfg.import_source.clone(),
        params_m: params as f64 / 1_000_000.0,
        size_mb,
        default: dir.file_name().unwrap_or_default() == "model_output",
        arch_version: cfg.arch_version,
        compatible: cfg.is_compatible() && !legacy_weights,
        legacy_weights,
        steps,
        best_val_loss,
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
            if ext != "bpk" {
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

/// Files downloadable from a model directory, plus the shared tokenizer when
/// it exists next to the server.
#[derive(Serialize)]
pub(super) struct ModelFileInfo {
    name: String,
    size_bytes: u64,
    modified_unix: u64,
    /// True for files shared across models (e.g. `custom_tokenizer.json`).
    shared: bool,
}

#[derive(Serialize)]
pub(super) struct ModelFilesResponse {
    dir: String,
    name: String,
    files: Vec<ModelFileInfo>,
}

fn file_info(path: &Path, shared: bool) -> Option<ModelFileInfo> {
    let meta = path.metadata().ok()?;
    if !meta.is_file() {
        return None;
    }
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Some(ModelFileInfo {
        name: path.file_name()?.to_string_lossy().to_string(),
        size_bytes: meta.len(),
        modified_unix: modified,
        shared,
    })
}

/// A model must at least carry weights and config, otherwise it is just a
/// random directory and must not expose files (or be deleted by accident).
fn ensure_model_dir(dir: &Path) -> Result<(), ApiError> {
    if !dir.join("config.json").is_file()
        || (!crate::storage::model_weights_path(dir).is_file()
            && crate::storage::legacy_weights_path(dir).is_none())
    {
        return Err(ApiError::not_found(
            "not a model directory (config.json + model_weights.bpk required)",
        ));
    }
    Ok(())
}

/// Model mutations are refused while training runs: the worker may be writing
/// into one of these directories.
fn ensure_idle(state: &AppState) -> Result<(), ApiError> {
    if state.server.job.read().unwrap().running {
        return Err(ApiError::conflict("cannot modify models while training is running"));
    }
    Ok(())
}

#[derive(Deserialize)]
pub(super) struct DirQuery {
    #[serde(default)]
    dir: String,
}

#[allow(clippy::unused_async)]
pub(super) async fn model_files(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<DirQuery>,
) -> Result<Json<ModelFilesResponse>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let dir = resolve_model_child_dir(&state.server.models_dir, &q.dir)?;
    ensure_model_dir(&dir)?;

    let mut files = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            if let Some(info) = file_info(&e.path(), false) {
                files.push(info);
            }
        }
    }
    let shared = Path::new("custom_tokenizer.json");
    let has_local_tokenizer = files.iter().any(|f| f.name == "custom_tokenizer.json");
    if !has_local_tokenizer {
        if let Some(info) = file_info(shared, true) {
            files.push(info);
        }
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(Json(ModelFilesResponse {
        dir: dir.to_string_lossy().to_string(),
        name: dir.file_name().unwrap_or_default().to_string_lossy().to_string(),
        files,
    }))
}

#[derive(Deserialize)]
pub(super) struct FileQuery {
    #[serde(default)]
    dir: String,
    file: String,
}

/// Download a single file from a model directory (or the shared tokenizer).
/// Exists separately from `/api/download` so that a models root pointing at the
/// working directory can never expose arbitrary files.
pub(super) async fn model_file(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<FileQuery>,
) -> Result<Response, ApiError> {
    check_token(&state.server, &headers, None)?;
    if !is_safe_child_name(&q.file) {
        return Err(ApiError::bad_request("invalid file name"));
    }
    let dir = resolve_model_child_dir(&state.server.models_dir, &q.dir)?;
    ensure_model_dir(&dir)?;

    let candidate = dir.join(&q.file);
    let path = if candidate.is_file() {
        candidate
    } else if q.file == "custom_tokenizer.json" && Path::new("custom_tokenizer.json").is_file() {
        std::path::PathBuf::from("custom_tokenizer.json")
    } else {
        return Err(ApiError::not_found(format!("file not found: {}", q.file)));
    };

    let bytes = tokio::fs::read(&path).await.map_err(|e| ApiError::internal(e.to_string()))?;
    let fname = path.file_name().unwrap_or_default().to_string_lossy().to_string();
    Ok((
        [
            (header::CONTENT_TYPE, "application/octet-stream".to_string()),
            (header::CONTENT_DISPOSITION, format!("attachment; filename=\"{fname}\"")),
        ],
        bytes,
    )
        .into_response())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelImportRequest {
    /// Path to the GGUF file relative to the server working directory.
    gguf_path: String,
    /// Output model directory under the models root (default: GGUF stem).
    #[serde(default)]
    output_dir: Option<String>,
    #[serde(default)]
    force: bool,
}

#[allow(clippy::unused_async)]
pub(super) async fn model_import(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ModelImportRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    ensure_idle(&state)?;

    if !is_safe_rel_path(&req.gguf_path) {
        return Err(ApiError::bad_request(
            "invalid GGUF path (must be relative with no traversal)",
        ));
    }
    let gguf = std::path::PathBuf::from(&req.gguf_path);
    if !gguf.is_file() {
        return Err(ApiError::not_found(format!(
            "GGUF file not found: {}",
            req.gguf_path
        )));
    }

    let name = req
        .output_dir
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            gguf.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "imported_model".to_string())
        });
    if !is_safe_child_name(&name) || name.starts_with('.') {
        return Err(ApiError::bad_request("invalid output directory name"));
    }
    let out = state.server.models_dir.join(&name);
    if out.exists() && !req.force {
        return Err(ApiError::conflict(format!(
            "'{name}' already exists (enable overwrite to replace it)"
        )));
    }

    let force = req.force;
    let out_clone = out.clone();
    let result = tokio::task::spawn_blocking(move || {
        crate::import::import_gguf(
            &gguf,
            &out_clone,
            force,
            &crate::backend::cpu_device(),
            None,
        )
    })
    .await
    .map_err(|e| ApiError::internal(format!("import task failed: {e}")))?;
    result.map_err(ApiError::bad_request)?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "dir": out.to_string_lossy(),
        "name": name,
    })))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelRenameRequest {
    dir: String,
    new_name: String,
}

#[allow(clippy::unused_async)]
pub(super) async fn model_rename(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ModelRenameRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    ensure_idle(&state)?;

    let new_name = req.new_name.trim().to_string();
    if new_name.is_empty() || !is_safe_child_name(&new_name) || new_name.starts_with('.') {
        return Err(ApiError::bad_request("invalid model name"));
    }

    let from = resolve_model_child_dir(&state.server.models_dir, &req.dir)?;
    ensure_model_dir(&from)?;
    if from.file_name().and_then(|n| n.to_str()) == Some(new_name.as_str()) {
        return Ok(Json(serde_json::json!({ "status": "ok", "dir": from })));
    }

    let to = from
        .parent()
        .ok_or_else(|| ApiError::internal("model dir has no parent"))?
        .join(&new_name);
    if to.exists() {
        return Err(ApiError::conflict("a model with that name already exists"));
    }
    std::fs::rename(&from, &to).map_err(|e| ApiError::internal(e.to_string()))?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "dir": to.to_string_lossy(),
        "name": new_name,
    })))
}

#[allow(clippy::unused_async)]
pub(super) async fn model_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    ensure_idle(&state)?;

    let dir = resolve_model_child_dir(&state.server.models_dir, &name)?;
    ensure_model_dir(&dir)?;
    std::fs::remove_dir_all(&dir).map_err(|e| ApiError::internal(e.to_string()))?;

    Ok(Json(serde_json::json!({ "status": "ok", "deleted": name })))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelImportHfRequest {
    /// Hugging Face repo id (exactly one of `repo` / `dir`).
    #[serde(default)]
    repo: Option<String>,
    /// Local directory with config.json + tokenizer.json + safetensors.
    #[serde(default)]
    dir: Option<String>,
    /// Output model directory under the models root (default: source name).
    #[serde(default)]
    output_dir: Option<String>,
    #[serde(default)]
    force: bool,
}

/// Convert a Hugging Face model into the native format as a background
/// operation (download + tensor conversion can take minutes).
#[allow(clippy::unused_async)]
pub(super) async fn model_import_hf(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ModelImportHfRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    ensure_idle(&state)?;

    let source = match (req.repo, req.dir) {
        (Some(repo), None) => {
            let repo = repo.trim().to_string();
            if repo.is_empty() || !repo.contains('/') {
                return Err(ApiError::bad_request("invalid Hugging Face repo id"));
            }
            crate::hf_import::HfSource::Repo(repo)
        }
        (None, Some(dir)) => {
            if !is_safe_rel_path(&dir) {
                return Err(ApiError::bad_request("invalid dir"));
            }
            crate::hf_import::HfSource::Dir(std::path::PathBuf::from(dir))
        }
        _ => return Err(ApiError::bad_request("pass exactly one of repo or dir")),
    };
    let default_name = match &source {
        crate::hf_import::HfSource::Repo(repo) => {
            repo.rsplit('/').next().unwrap_or(repo).to_string()
        }
        crate::hf_import::HfSource::Dir(dir) => dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "imported_model".to_string()),
    };
    let name = req
        .output_dir
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or(default_name);
    if !is_safe_child_name(&name) || name.starts_with('.') {
        return Err(ApiError::bad_request("invalid output directory name"));
    }
    let out = state.server.models_dir.join(&name);
    if out.exists() && !req.force {
        return Err(ApiError::conflict(format!(
            "'{name}' already exists (enable overwrite to replace it)"
        )));
    }

    let force = req.force;
    let op = state
        .ops
        .start("import-hf", &format!("{} → {name}", source.label()));
    let op_id = op.id().to_string();
    let name_for_op = name.clone();
    tokio::task::spawn_blocking(move || {
        if op.is_cancelled() {
            op.cancelled();
            return;
        }
        let result = {
            let mut progress =
                |phase: &str, done: u64, total: u64, unit: &str| op.progress(phase, done, total, unit);
            crate::hf_import::import_hf(
                &source,
                &out,
                force,
                &crate::backend::cpu_device(),
                &mut progress,
            )
        };
        match result {
            Ok(cfg) => op.finish(
                &format!("{name_for_op} converted"),
                Some(serde_json::json!({
                    "dir": out.to_string_lossy(),
                    "name": name_for_op,
                    "layers": cfg.n_layer,
                    "n_embd": cfg.n_embd,
                })),
            ),
            Err(e) => {
                if op.is_cancelled() {
                    op.cancelled();
                } else {
                    op.fail(&e);
                }
            }
        }
    });

    Ok(Json(serde_json::json!({
        "status": "started",
        "opId": op_id,
        "name": name,
    })))
}
