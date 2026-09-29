use axum::{Json, extract::State, http::HeaderMap};
use serde::Serialize;

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
    /// Machine-readable selector consumed by the runner to build a `WgpuDevice`.
    selector: String,
}

fn enum_adapter_specs() -> Vec<DeviceInfo> {
    use wgpu::Backends;

    let instance = wgpu::Instance::default();

    let adapters = futures::executor::block_on(instance.enumerate_adapters(Backends::all()));
    if adapters.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut discrete_idx = 0usize;
    let mut integrated_idx = 0usize;
    let mut virtual_idx = 0usize;
    for a in adapters {
        let info = a.get_info();
        let backend_name = match info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DX12",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "Other",
        };
        let (type_name, selector) = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => {
                let sel = format!("DiscreteGpu({discrete_idx})");
                discrete_idx += 1;
                ("Discrete GPU".to_string(), sel)
            }
            wgpu::DeviceType::IntegratedGpu => {
                let sel = format!("IntegratedGpu({integrated_idx})");
                integrated_idx += 1;
                ("Integrated GPU".to_string(), sel)
            }
            wgpu::DeviceType::VirtualGpu => {
                let sel = format!("VirtualGpu({virtual_idx})");
                virtual_idx += 1;
                ("Virtual GPU".to_string(), sel)
            }
            wgpu::DeviceType::Cpu => ("CPU".to_string(), "Cpu".to_string()),
            _ => ("Other".to_string(), "DefaultDevice".to_string()),
        };

        let mut vram_bytes = None;
        if let Ok((device, _queue)) =
            futures::executor::block_on(a.request_device(&wgpu::DeviceDescriptor::default()))
        {
            let limits = device.limits();
            let max_buf = limits.max_buffer_size;
            if max_buf > 0 {
                vram_bytes = Some(max_buf);
            }
        }

        out.push(DeviceInfo {
            name: info.name.clone(),
            backend: backend_name.into(),
            device_type: type_name,
            vendor: info.vendor,
            device: info.device,
            driver: info.driver.clone(),
            vram_bytes,
            selectable: true,
            selector,
        });
    }
    out
}

fn fallback_devices() -> Vec<DeviceInfo> {
    vec![
        DeviceInfo {
            name: "DefaultDevice".into(),
            backend: "wgpu".into(),
            device_type: "Auto".into(),
            vendor: 0,
            device: 0,
            driver: String::new(),
            vram_bytes: None,
            selectable: true,
            selector: "DefaultDevice".into(),
        },
        DeviceInfo {
            name: "DiscreteGpu(0)".into(),
            backend: "wgpu".into(),
            device_type: "Discrete GPU".into(),
            vendor: 0,
            device: 0,
            driver: String::new(),
            vram_bytes: None,
            selectable: true,
            selector: "DiscreteGpu(0)".into(),
        },
        DeviceInfo {
            name: "Cpu".into(),
            backend: "wgpu".into(),
            device_type: "CPU".into(),
            vendor: 0,
            device: 0,
            driver: String::new(),
            vram_bytes: None,
            selectable: true,
            selector: "Cpu".into(),
        },
    ]
}

pub(super) async fn devices(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<DeviceInfo>>, ApiError> {
    check_token(&state.server, &headers, None)?;

    // WGPU adapter enumeration is blocking; run it on a blocking thread pool so
    // it never parks an async worker. Timeout keeps a hung driver from locking
    // the endpoint (the detached task is simply abandoned).
    let specs = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::task::spawn_blocking(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(enum_adapter_specs))
                .unwrap_or_default()
        }),
    )
    .await
    .ok()
    .and_then(Result::ok)
    .unwrap_or_default();

    if specs.is_empty() {
        // Fallback: expose the selectors we know the runner supports.
        return Ok(Json(fallback_devices()));
    }

    Ok(Json(specs))
}
