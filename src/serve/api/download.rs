use std::path::Path;

use axum::{
    extract::{Path as AxumPath, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;
use super::paths::is_safe_rel_path;

#[allow(clippy::unused_async)]
pub(super) async fn download(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(path): AxumPath<String>,
    axum::extract::Query(query): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Response, ApiError> {
    check_token(&state.server, &headers, query.get("token").map(|s| s.as_str()))?;

    // Only files under known output roots are served: the datasets dir, the
    // models dir and the exported/ directory (GGUF). Absolute paths and
    // traversal (`..`) are rejected outright.
    if !is_safe_rel_path(&path) {
        return Err(ApiError::bad_request("invalid path"));
    }
    let p = Path::new(&path);
    let cwd = std::env::current_dir().unwrap_or_default();
    let allowed_roots = {
        let mut roots = vec![state.server.datasets_dir.clone(), state.server.models_dir.clone()];
        roots.push(cwd.join("exported"));
        roots
    };
    let canonical = std::fs::canonicalize(p).map_err(|_| ApiError::not_found("not found"))?;
    let inside = allowed_roots.iter().any(|root| {
        // Never treat the working directory itself as an allowed root, only
        // its explicit children (exported/, datasets/, models_dir).
        if root == &cwd {
            return false;
        }
        std::fs::canonicalize(root)
            .map(|r| canonical.starts_with(&r))
            .unwrap_or(false)
    });
    if !inside {
        return Err(ApiError::forbidden("forbidden"));
    }
    let Ok(bytes) = tokio::fs::read(p).await else {
        return Err(ApiError::internal("read error"));
    };
    let fname = p.file_name().unwrap_or_default().to_string_lossy().to_string();
    Ok((
        StatusCode::OK,
        [(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{fname}\""))],
        bytes,
    )
        .into_response())
}
