use std::io::BufRead;
use std::path::Path;

use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::HeaderMap,
};
use serde::{Deserialize, Serialize};

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;
use super::paths::{resolve_dataset, resolve_dataset_output};

#[derive(Serialize)]
pub(super) struct DatasetInfo {
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

#[allow(clippy::unused_async)]
pub(super) async fn datasets(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<DatasetInfo>>, ApiError> {
    check_token(&state.server, &headers, None)?;
    Ok(Json(list_dataset_infos(&state.server.datasets_dir)))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct DatasetDownloadRequest {
    repo_id: String,
    #[serde(default = "default_download_filename")]
    filename: String,
}

fn default_download_filename() -> String {
    "TinyStories-train.txt".into()
}

pub(super) async fn dataset_download(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DatasetDownloadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let repo_id = req.repo_id.trim().to_string();
    if repo_id.is_empty() {
        return Err(ApiError::bad_request("repoId is required"));
    }
    let filename = req.filename.clone();

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(300),
        tokio::task::spawn_blocking(move || {
            crate::data::download_dataset_from_hf(&repo_id, &filename).map_err(|e| e.to_string())
        }),
    )
    .await;

    match result {
        Ok(Ok(Ok(path))) => Ok(Json(serde_json::json!({
            "status": "ok",
            "path": path.to_string_lossy(),
            "name": path.file_name().unwrap_or_default().to_string_lossy(),
        }))),
        Ok(Ok(Err(e))) => Err(ApiError::internal(e)),
        Ok(Err(e)) => Err(ApiError::internal(format!("download task failed: {e}"))),
        Err(_) => Err(ApiError::gateway_timeout("download timed out")),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct DatasetSearchRequest {
    q: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
}

fn default_search_limit() -> usize {
    15
}

#[derive(Serialize)]
pub(super) struct DatasetHit {
    id: String,
    downloads: u64,
    likes: u64,
    last_modified: String,
    tags: Vec<String>,
}

pub(super) async fn dataset_search(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Query(params): axum::extract::Query<DatasetSearchRequest>,
) -> Result<Json<Vec<DatasetHit>>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let q = params.q.trim().to_string();
    if q.is_empty() {
        return Ok(Json(Vec::new()));
    }

    let limit = params.limit.clamp(1, 50);
    let url = format!(
        "https://huggingface.co/api/datasets?search={}&limit={}",
        urlencode(&q),
        limit
    );

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(25),
        tokio::task::spawn_blocking(move || -> Result<Vec<DatasetHit>, String> {
            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .build()
                .map_err(|e| e.to_string())?;
            let resp = client
                .get(&url)
                .header("User-Agent", "kuro-neko-llm/0.1")
                .send()
                .map_err(|e| e.to_string())?;
            if !resp.status().is_success() {
                return Err(format!("HF API status {}", resp.status()));
            }
            let raw: Vec<serde_json::Value> = resp.json().map_err(|e| e.to_string())?;
            let mut hits = Vec::new();
            for item in raw.into_iter().take(limit) {
                hits.push(DatasetHit {
                    id: item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    downloads: item.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                    likes: item.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                    last_modified: item
                        .get("lastModified")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    tags: item
                        .get("tags")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|t| t.as_str().map(|s| s.to_string()))
                                .filter(|t| t.starts_with("language:"))
                                .collect()
                        })
                        .unwrap_or_default(),
                });
            }
            Ok(hits)
        }),
    )
    .await;

    match result {
        Ok(Ok(Ok(hits))) => Ok(Json(hits)),
        Ok(Ok(Err(e))) => Err(ApiError::bad_gateway(e)),
        Ok(Err(e)) => Err(ApiError::bad_gateway(format!("search task failed: {e}"))),
        Err(_) => Err(ApiError::gateway_timeout("search timed out")),
    }
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct DatasetRenameRequest {
    from: String,
    to: String,
}

#[allow(clippy::unused_async)]
pub(super) async fn dataset_rename(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DatasetRenameRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let from = resolve_dataset(&state.server.datasets_dir, &req.from)?;

    let to_name = Path::new(&req.to).file_name().and_then(|n| n.to_str());
    let to_name = match to_name {
        Some(n) if !n.is_empty() && n == req.to => n,
        _ => return Err(ApiError::bad_request("invalid target name")),
    };
    let to = state.server.datasets_dir.join(to_name);
    if to.exists() {
        return Err(ApiError::conflict("target already exists"));
    }
    std::fs::rename(&from, &to).map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(serde_json::json!({ "status": "ok", "from": req.from, "to": req.to })))
}

#[allow(clippy::unused_async)]
pub(super) async fn dataset_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let path = resolve_dataset(&state.server.datasets_dir, &name)?;
    std::fs::remove_file(&path).map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(serde_json::json!({ "status": "ok", "deleted": name })))
}

pub(super) async fn dataset_info(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let path = resolve_dataset(&state.server.datasets_dir, &name)?;

    let size_bytes = path.metadata().map(|m| m.len()).unwrap_or(0);
    let ext = path.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();

    let mut info = serde_json::json!({
        "name": name,
        "size_bytes": size_bytes,
        "ext": ext,
    });

    // Cheap text stats: line count (for .txt) on a blocking thread so huge
    // corpora (several GB) cannot park an async worker.
    if ext == "txt" {
        let lines = tokio::task::spawn_blocking(move || -> std::io::Result<usize> {
            let reader = std::io::BufReader::new(std::fs::File::open(&path)?);
            Ok(reader.lines().count())
        })
        .await;
        if let Ok(Ok(lines)) = lines {
            info["lines"] = serde_json::json!(lines);
        }
    }

    Ok(Json(info))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct DatasetCombineRequest {
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

fn default_combine_output() -> String {
    "weighted_mix_web.txt".into()
}

pub(super) async fn dataset_combine(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DatasetCombineRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    if req.sources.is_empty() {
        return Err(ApiError::bad_request("sources is required"));
    }

    let mut sources = Vec::new();
    for s in &req.sources {
        let p = resolve_dataset(&state.server.datasets_dir, &s.path)?;
        sources.push((p, s.weight));
    }

    let output_path = resolve_dataset_output(&state.server.datasets_dir, &req.output)?;
    let max_chars = req.max_chars.unwrap_or(usize::MAX);
    let output_clone = output_path.clone();

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(600),
        tokio::task::spawn_blocking(move || {
            crate::data::combine_and_mix_weighted(&sources, &output_clone, max_chars)
                .map(|_| ())
                .map_err(|e| e.to_string())
        }),
    )
    .await;

    match result {
        Ok(Ok(Ok(()))) => Ok(Json(serde_json::json!({
            "status": "ok",
            "output": output_path.to_string_lossy(),
        }))),
        Ok(Ok(Err(e))) => Err(ApiError::internal(e)),
        Ok(Err(e)) => Err(ApiError::internal(format!("combine task failed: {e}"))),
        Err(_) => Err(ApiError::gateway_timeout("combine timed out")),
    }
}
