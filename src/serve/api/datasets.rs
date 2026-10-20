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
    #[serde(default)]
    filenames: Vec<String>,
}

/// Resolve the files to download: the explicit ones, or the best training
/// file of the repo when the request omits them.
async fn resolve_download_files(repo_id: &str, filenames: &[String]) -> Result<Vec<String>, ApiError> {
    let requested: Vec<String> = filenames
        .iter()
        .map(|f| f.trim())
        .filter(|f| !f.is_empty())
        .map(str::to_string)
        .collect();
    if !requested.is_empty() {
        return Ok(requested);
    }
    let repo = repo_id.to_string();
    let files = tokio::task::spawn_blocking(move || crate::data::hf_repo_files(&repo))
        .await
        .map_err(|e| ApiError::internal(format!("repo listing task failed: {e}")))?
        .map_err(ApiError::bad_gateway)?;
    crate::data::pick_default_dataset_file(&files)
        .map(|file| vec![file])
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "No usable data file found in '{repo_id}' ({} files). Pick one explicitly.",
                files.len()
            ))
        })
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
    let files = resolve_download_files(&repo_id, &req.filenames).await?;
    let label = if files.len() == 1 {
        format!("{repo_id} · {}", files[0])
    } else {
        format!("{repo_id} · {} files", files.len())
    };

    // Long downloads run as background operations so the UI can follow byte
    // progress (and cancel) instead of holding a request open for minutes.
    let op = state.ops.start("download", &label);
    let op_id = op.id().to_string();
    let files_for_op = files.clone();
    tokio::task::spawn_blocking(move || {
        let total_files = files_for_op.len();
        let mut paths: Vec<std::path::PathBuf> = Vec::new();
        for (i, file) in files_for_op.iter().enumerate() {
            let prefix = if total_files > 1 {
                format!("file {}/{total_files} · ", i + 1)
            } else {
                String::new()
            };
            let progress = |phase: &str, done: u64, total: u64| {
                op.progress(&format!("{prefix}{phase}"), done, total, "bytes");
            };
            let cancelled = || op.is_cancelled();
            match crate::data::download_dataset_streaming(&repo_id, file, &progress, &cancelled) {
                Ok(path) => paths.push(path),
                Err(e) => {
                    if op.is_cancelled() {
                        op.cancelled();
                    } else {
                        op.fail(&format!("{file}: {e}"));
                    }
                    return;
                }
            }
        }
        let names: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string())
            .collect();
        let message = if names.len() == 1 {
            format!("Downloaded {}", names[0])
        } else {
            format!("Downloaded {} files", names.len())
        };
        op.finish(
            &message,
            Some(serde_json::json!({
                "paths": paths.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
                "names": names,
            })),
        );
    });

    Ok(Json(serde_json::json!({ "opId": op_id, "status": "started", "files": files })))
}

#[derive(Deserialize)]
pub(super) struct RepoFilesQuery {
    repo: String,
}

#[derive(Serialize)]
pub(super) struct RepoFilesResponse {
    files: Vec<String>,
    default: Option<String>,
}

/// File listing of a HF dataset repo, so the UI can offer the real files
/// instead of guessing names.
pub(super) async fn dataset_repo_files(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Query(params): axum::extract::Query<RepoFilesQuery>,
) -> Result<Json<RepoFilesResponse>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let repo = params.repo.trim().to_string();
    if repo.is_empty() {
        return Err(ApiError::bad_request("repo is required"));
    }
    let files = tokio::task::spawn_blocking(move || crate::data::hf_repo_files(&repo))
        .await
        .map_err(|e| ApiError::internal(format!("repo listing task failed: {e}")))?
        .map_err(ApiError::bad_gateway)?;
    let default = crate::data::pick_default_dataset_file(&files);
    Ok(Json(RepoFilesResponse { files, default }))
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
                .header("User-Agent", "noctambulo-llm/0.1")
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

    // Long mix runs as a background operation with char-level progress.
    let op = state.ops.start("combine", &format!("{} sources → {}", sources.len(), req.output));
    let op_id = op.id().to_string();
    tokio::task::spawn_blocking(move || {
        let progress = |done: u64, total: u64| op.progress("mixing", done, total, "chars");
        let cancelled = || op.is_cancelled();
        match crate::data::combine_and_mix_weighted_with(&sources, &output_clone, max_chars, &progress, &cancelled) {
            Ok(path) => op.finish(
                &format!("Mix created: {}", path.display()),
                Some(serde_json::json!({ "output": path.to_string_lossy() })),
            ),
            Err(e) => {
                if op.is_cancelled() {
                    op.cancelled();
                } else {
                    op.fail(&e.to_string());
                }
            }
        }
    });

    Ok(Json(serde_json::json!({ "opId": op_id, "status": "started" })))
}
