use std::sync::Arc;

use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::HeaderMap,
    response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}},
};
use futures_core::Stream;

use crate::serve::ops::{OpStatus, OpsRegistry};

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;

pub(super) async fn ops(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<OpStatus>>, ApiError> {
    check_token(&state.server, &headers, None)?;
    Ok(Json(state.ops.snapshot()))
}

#[allow(clippy::unused_async)]
pub(super) async fn op_cancel(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    if state.ops.cancel(&id) {
        Ok(Json(serde_json::json!({ "status": "cancelling", "id": id })))
    } else {
        Err(ApiError::not_found("no running operation with that id"))
    }
}

fn event_stream(ops_registry: Arc<OpsRegistry>) -> impl Stream<Item = Result<Event, axum::Error>> + Send + 'static {
    let mut rx = ops_registry.subscribe();
    async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    let json = serde_json::to_string(&ev).unwrap_or_default();
                    yield Ok(Event::default().event("op").data(json));
                }
                // A slow consumer missed events: skip them and keep streaming.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    }
}

pub(super) async fn ops_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    check_token(&state.server, &headers, None)?;
    Ok(Sse::new(event_stream(state.ops.clone()))
        .keep_alive(KeepAlive::default())
        .into_response())
}
