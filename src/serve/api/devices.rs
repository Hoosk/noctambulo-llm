use axum::{
    Json,
    extract::{Path, State},
    http::HeaderMap,
};
use serde::{Deserialize, Serialize};

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;

#[derive(Serialize)]
pub(super) struct DeviceInfo {
    name: String,
    backend: String,
    device_type: String,
    vendor: u32,
    device: u32,
    driver: String,
    vram_bytes: Option<u64>,
    selectable: bool,
    /// Machine-readable selector consumed by the runner and CLI.
    selector: String,
    supports_f32: bool,
    supports_f16: bool,
    supports_bf16: bool,
    remote: bool,
    default: bool,
    status: String,
}

impl DeviceInfo {
    fn from_choice(choice: crate::backend::DeviceChoice, default: bool) -> Self {
        Self {
            name: choice.name,
            backend: choice.backend,
            device_type: choice.device_type,
            vendor: choice.vendor,
            device: choice.device,
            driver: choice.driver,
            vram_bytes: choice.vram_bytes,
            selectable: choice.selectable,
            selector: choice.selector,
            supports_f32: choice.supports_f32,
            supports_f16: choice.supports_f16,
            supports_bf16: choice.supports_bf16,
            remote: choice.remote,
            default,
            status: "online".into(),
        }
    }
}

pub(super) async fn devices(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<DeviceInfo>>, ApiError> {
    check_token(&state.server, &headers, None)?;

    let default_selector = state.server.default_device_selector();
    let mut infos: Vec<DeviceInfo> = crate::backend::local_devices()
        .into_iter()
        .map(|choice| {
            let is_default = default_selector
                .as_deref()
                .map(|s| s == choice.selector)
                .unwrap_or(choice.selector == "auto");
            DeviceInfo::from_choice(choice, is_default)
        })
        .collect();

    // Remote probes can block on the network: keep them off the async workers
    // and cap the wait so one dead host cannot hang the endpoint.
    let probe = tokio::time::timeout(
        std::time::Duration::from_secs(8),
        tokio::task::spawn_blocking(crate::backend::probe_remote_nodes),
    )
    .await;
    if let Ok(Ok(nodes)) = probe {
        for node in nodes {
            if let Some(error) = node.error {
                infos.push(DeviceInfo {
                    name: node.node.name,
                    backend: "Remote".into(),
                    device_type: "Remote".into(),
                    vendor: 0,
                    device: 0,
                    driver: String::new(),
                    vram_bytes: None,
                    selectable: false,
                    selector: format!("remote://{}#0", node.node.id),
                    supports_f32: false,
                    supports_f16: false,
                    supports_bf16: false,
                    remote: true,
                    default: false,
                    status: format!("offline: {error}"),
                });
            } else {
                for choice in node.devices {
                    let is_default = default_selector
                        .as_deref()
                        .map(|s| s == choice.selector)
                        .unwrap_or(false);
                    infos.push(DeviceInfo::from_choice(choice, is_default));
                }
            }
        }
    }

    Ok(Json(infos))
}

#[derive(Serialize)]
pub(super) struct RemoteNodeInfo {
    id: String,
    name: String,
    url: String,
    has_token: bool,
    created_unix: u64,
}

#[allow(clippy::unused_async)]
pub(super) async fn remote_nodes(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<RemoteNodeInfo>>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let nodes = state
        .server
        .remote_nodes_snapshot()
        .into_iter()
        .map(|node| RemoteNodeInfo {
            id: node.id,
            name: node.name,
            url: node.url,
            has_token: node.token.is_some(),
            created_unix: node.created_unix,
        })
        .collect();
    Ok(Json(nodes))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct AddRemoteNodeRequest {
    name: Option<String>,
    url: String,
    token: Option<String>,
}

pub(super) async fn add_remote_node(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddRemoteNodeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;

    let url = req.url.trim().to_string();
    if !(url.starts_with("ws://")
        || url.starts_with("wss://")
        || url.starts_with("iroh://"))
    {
        return Err(ApiError::bad_request(
            "remote url must start with ws://, wss:// or iroh://",
        ));
    }
    let token = req.token.filter(|t| !t.is_empty());
    let name = req
        .name
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| remote_name_from_url(&url));

    let probe_url = url.clone();
    let probe_token = token.clone();
    let probe = tokio::time::timeout(
        std::time::Duration::from_secs(12),
        tokio::task::spawn_blocking(move || {
            crate::backend::probe_remote_url(&probe_url, probe_token.as_deref())
        }),
    )
    .await;
    let device_count = match probe {
        Ok(Ok(Ok(count))) => count,
        Ok(Ok(Err(err))) => return Err(ApiError::bad_request(err)),
        Ok(Err(err)) => return Err(ApiError::internal(format!("remote probe failed: {err}"))),
        Err(_) => return Err(ApiError::gateway_timeout("remote host did not answer in 12s")),
    };

    let created_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let node = crate::backend::RemoteNode {
        id: format!("{}-{nanos}", slug(&name)),
        name: name.clone(),
        url: url.clone(),
        token,
        created_unix,
    };
    state.server.put_remote_node(node.clone());

    Ok(Json(serde_json::json!({
        "id": node.id,
        "name": name,
        "url": url,
        "devices": device_count,
        "status": "online",
    })))
}

pub(super) async fn remove_remote_node(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    if state.server.remove_remote_node(&id) {
        Ok(Json(serde_json::json!({ "removed": id })))
    } else {
        Err(ApiError::not_found("remote node not found"))
    }
}

#[derive(Deserialize)]
pub(super) struct DefaultDeviceRequest {
    selector: String,
}

#[allow(clippy::unused_async)]
pub(super) async fn set_default_device(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DefaultDeviceRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;
    let selector = req.selector.trim();
    let selector = (!selector.is_empty()).then(|| selector.to_string());
    state.server.set_default_device_selector(selector.clone());
    Ok(Json(serde_json::json!({
        "default": selector.unwrap_or_else(|| "auto".into()),
    })))
}

fn remote_name_from_url(url: &str) -> String {
    url.split("://")
        .nth(1)
        .unwrap_or(url)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(url)
        .to_string()
}

fn slug(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() { "node".into() } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugifies_node_names() {
        assert_eq!(slug("My GPU (LAN)"), "my-gpu-lan");
        assert_eq!(slug("  "), "node");
    }

    #[test]
    fn extracts_remote_host_name() {
        assert_eq!(remote_name_from_url("ws://192.168.1.5:3000"), "192.168.1.5:3000");
        assert_eq!(remote_name_from_url("iroh://abc123"), "abc123");
    }
}
