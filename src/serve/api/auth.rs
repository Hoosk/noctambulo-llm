use axum::{
    Json,
    extract::State,
    http::{HeaderMap, header},
    response::{IntoResponse, Response},
};
use serde::Deserialize;

use super::AppState;
use super::error::ApiError;

/// Name of the HttpOnly session cookie set by `/api/login`. Keeping the token
/// out of JavaScript storage protects it from XSS exfiltration; the CLI and
/// scripts keep using the `X-API-Token` header.
const COOKIE_NAME: &str = "noctambulo_token";

const COOKIE_MAX_AGE_SECS: u64 = 60 * 60 * 24 * 30;

pub(super) fn check_token(
    server: &crate::serve::state::ServerState,
    headers: &HeaderMap,
    query: Option<&str>,
) -> Result<(), ApiError> {
    let expected = server.token.read().unwrap().clone();
    if let Some(expected) = expected {
        let provided = headers
            .get("x-api-token")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .or_else(|| cookie_token(headers))
            .or_else(|| query_token(query))
            .unwrap_or_default();
        if provided != expected {
            return Err(ApiError::unauthorized());
        }
    }
    Ok(())
}

/// Extract the session token from the `Cookie` header, if present.
pub(super) fn cookie_token(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(header::COOKIE)?.to_str().ok()?;
    for part in raw.split(';') {
        let Some((name, value)) = part.trim().split_once('=') else {
            continue;
        };
        if name == COOKIE_NAME {
            return cookie_decode(value);
        }
    }
    None
}

/// Percent-encode a cookie value so arbitrary tokens survive the round trip.
pub(super) fn cookie_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

pub(super) fn cookie_decode(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = std::str::from_utf8(bytes.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

pub(super) fn query_token(query: Option<&str>) -> Option<String> {
    let query = query?;
    for pair in query.split('&') {
        let mut parts = pair.splitn(2, '=');
        if parts.next() == Some("token") {
            return parts.next().map(|v| v.to_string());
        }
    }
    None
}

#[derive(Deserialize)]
pub(super) struct LoginRequest {
    token: String,
}

/// Exchange a valid API token for an HttpOnly session cookie. The token never
/// lands in `localStorage`, so a script injection cannot read it back.
#[allow(clippy::unused_async)]
pub(super) async fn login(State(state): State<AppState>, Json(req): Json<LoginRequest>) -> Response {
    let expected = state.server.token.read().unwrap().clone();
    match expected {
        Some(expected) if req.token == expected => {
            let cookie = format!(
                "{COOKIE_NAME}={}; HttpOnly; SameSite=Strict; Path=/; Max-Age={}",
                cookie_encode(&req.token),
                COOKIE_MAX_AGE_SECS
            );
            ([(header::SET_COOKIE, cookie)], Json(serde_json::json!({ "ok": true }))).into_response()
        }
        Some(_) => ApiError::unauthorized().into_response(),
        // Open server (no token configured): nothing to authenticate.
        None => Json(serde_json::json!({ "ok": true })).into_response(),
    }
}

#[allow(clippy::unused_async)]
pub(super) async fn logout() -> Response {
    let cookie = format!("{COOKIE_NAME}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0");
    ([(header::SET_COOKIE, cookie)], Json(serde_json::json!({ "ok": true }))).into_response()
}

/// Lightweight authenticated health check used by the frontend token gate
/// (much faster than enumerating WGPU adapters).
#[allow(clippy::unused_async)]
pub(super) async fn ping(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    Ok(Json(serde_json::json!({ "ok": true })))
}
