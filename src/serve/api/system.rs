use axum::{Json, extract::State, http::HeaderMap};
use serde::Serialize;
use sysinfo::{Disks, System};

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;

#[derive(Serialize)]
pub(super) struct DiskInfo {
    name: String,
    mount_point: String,
    total_bytes: u64,
    available_bytes: u64,
}

#[derive(Serialize)]
pub(super) struct SystemInfo {
    hostname: String,
    os_name: String,
    os_version: String,
    kernel_version: String,
    cpu_brand: String,
    cpu_physical_cores: Option<usize>,
    cpu_logical_cores: usize,
    cpu_frequency_mhz: u64,
    total_memory_bytes: u64,
    used_memory_bytes: u64,
    total_swap_bytes: u64,
    used_swap_bytes: u64,
    uptime_secs: u64,
    load_average: [f64; 3],
    disks: Vec<DiskInfo>,
}

fn collect() -> SystemInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpus = sys.cpus();
    let cpu_brand = cpus
        .first()
        .map(|c| c.brand().trim().to_string())
        .unwrap_or_default();
    let cpu_frequency_mhz = cpus.iter().map(|c| c.frequency()).max().unwrap_or(0);
    let load = System::load_average();

    let disks = Disks::new_with_refreshed_list()
        .list()
        .iter()
        .map(|d| DiskInfo {
            name: d.name().to_string_lossy().to_string(),
            mount_point: d.mount_point().to_string_lossy().to_string(),
            total_bytes: d.total_space(),
            available_bytes: d.available_space(),
        })
        .collect();

    SystemInfo {
        hostname: System::host_name().unwrap_or_else(|| "unknown".into()),
        os_name: System::name().unwrap_or_else(|| "unknown".into()),
        os_version: System::os_version().unwrap_or_else(|| "unknown".into()),
        kernel_version: System::kernel_version().unwrap_or_else(|| "unknown".into()),
        cpu_brand,
        cpu_physical_cores: System::physical_core_count(),
        cpu_logical_cores: cpus.len(),
        cpu_frequency_mhz,
        total_memory_bytes: sys.total_memory(),
        used_memory_bytes: sys.used_memory(),
        total_swap_bytes: sys.total_swap(),
        used_swap_bytes: sys.used_swap(),
        uptime_secs: System::uptime(),
        load_average: [load.one, load.five, load.fifteen],
        disks,
    }
}

pub(super) async fn system(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<SystemInfo>, ApiError> {
    check_token(&state.server, &headers, None)?;

    // Reading /proc and enumerating disks is blocking; keep it off the async
    // worker pool.
    let info = tokio::task::spawn_blocking(collect)
        .await
        .map_err(|e| ApiError::internal(format!("system info task failed: {e}")))?;
    Ok(Json(info))
}
