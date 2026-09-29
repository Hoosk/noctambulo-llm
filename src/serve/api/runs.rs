use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::HeaderMap,
};

use crate::serve::runs::{RunRecord, RunSummary};

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;

#[allow(clippy::unused_async)]
pub(super) async fn list_runs(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<RunSummary>>, ApiError> {
    check_token(&state.server, &headers, None)?;
    Ok(Json(state.server.runs.list()))
}

#[allow(clippy::unused_async)]
pub(super) async fn get_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<RunRecord>, ApiError> {
    check_token(&state.server, &headers, None)?;
    state
        .server
        .runs
        .get(&id)
        .map(Json)
        .ok_or_else(|| ApiError::not_found(format!("run not found: {id}")))
}
