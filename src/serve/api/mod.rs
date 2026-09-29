//! HTTP API: router, shared app state, error type and per-area handlers.

mod auth;
mod datasets;
mod devices;
mod download;
mod error;
mod export;
mod generate;
mod models;
mod ops;
mod paths;
mod runs;
mod telemetry;
mod train;

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::http::StatusCode;
use axum::routing::{any, delete, get, post};

use crate::serve::ops::OpsRegistry;
use crate::serve::runner::JobRegistry;
use crate::serve::state::ServerState;

use auth::{login, logout, ping};
use datasets::{
    dataset_combine, dataset_delete, dataset_download, dataset_info, dataset_rename, dataset_search,
    datasets,
};
use devices::devices;
use download::download;
use export::export_model;
use generate::{generate_stream, generate_text};
use models::{checkpoints, model_delete, model_file, model_files, model_rename, models};
use train::{train, train_logs, train_pause, train_resume, train_status, train_stop, train_stream};

/// Build the API router. Also returns the job registry so the server can
/// request a graceful training stop on shutdown (Ctrl+C).
pub fn router(state: Arc<ServerState>) -> (Router, Arc<JobRegistry>) {
    let registry = Arc::new(JobRegistry::new());
    let ops_registry = Arc::new(OpsRegistry::new());
    let router = Router::new()
        .route("/api/ping", get(ping))
        .route("/api/login", post(login))
        .route("/api/logout", post(logout))
        .route("/api/devices", get(devices))
        .route("/api/telemetry", get(telemetry::telemetry))
        .route("/api/datasets", get(datasets))
        .route("/api/datasets/download", post(dataset_download))
        .route("/api/datasets/search", get(dataset_search))
        .route("/api/datasets/rename", post(dataset_rename))
        .route("/api/datasets/combine", post(dataset_combine))
        .route("/api/datasets/info/:name", get(dataset_info))
        .route("/api/datasets/delete/:name", delete(dataset_delete))
        .route("/api/models", get(models))
        .route("/api/models/files", get(model_files))
        .route("/api/models/file", get(model_file))
        .route("/api/models/rename", post(model_rename))
        .route("/api/models/:name", delete(model_delete))
        .route("/api/checkpoints", get(checkpoints))
        .route("/api/train", post(train))
        .route("/api/train/status", get(train_status))
        .route("/api/train/logs", get(train_logs))
        .route("/api/train/stop", post(train_stop))
        .route("/api/train/pause", post(train_pause))
        .route("/api/train/resume", post(train_resume))
        .route("/api/train/stream", get(train_stream))
        .route("/api/download/*rest", get(download))
        .route("/api/generate", post(generate_text))
        .route("/api/generate/stream", post(generate_stream))
        .route("/api/export", post(export_model))
        .route("/api/ops", get(ops::ops))
        .route("/api/ops/stream", get(ops::ops_stream))
        .route("/api/ops/:id/cancel", post(ops::op_cancel))
        .route("/api/runs", get(runs::list_runs))
        .route("/api/runs/:id", get(runs::get_run))
        // Unknown API paths must 404 as JSON, not fall through to the SPA.
        .route("/api/*rest", any(not_found))
        .with_state(AppState {
            server: state,
            registry: registry.clone(),
            ops: ops_registry,
        });
    (router, registry)
}

async fn not_found() -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "not found" })))
}

/// Shared handler state. Visible to the child modules and the test suite.
#[derive(Clone)]
struct AppState {
    server: Arc<ServerState>,
    registry: Arc<JobRegistry>,
    ops: Arc<OpsRegistry>,
}

#[cfg(test)]
mod tests;
