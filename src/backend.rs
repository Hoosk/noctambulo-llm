//! Backend and device selection.
//!
//! Burn 0.22 dispatches tensor operations at runtime, so the engine is no
//! longer generic over a backend: there is a single [`Device`] type and the
//! enabled Cargo features (`wgpu`, `flex`, `cuda`) select the available
//! backends.

use burn::tensor::{Device, DeviceKind};

/// Default compute device for the CLI: CUDA with the `cuda` feature, else WGPU.
#[cfg(feature = "cuda")]
pub fn default_device() -> Device {
    Device::cuda(0)
}

/// Default compute device for the CLI: CUDA with the `cuda` feature, else WGPU.
#[cfg(not(feature = "cuda"))]
pub fn default_device() -> Device {
    // Debug escape hatch: force the pure-Rust CPU backend to isolate GPU
    // kernel issues (`NOCTAMBULO_CPU=1`).
    if std::env::var_os("NOCTAMBULO_CPU").is_some() {
        return Device::flex();
    }
    Device::wgpu(DeviceKind::DefaultDevice)
}

/// Default device with autodiff enabled (training).
pub fn training_device() -> Device {
    default_device().autodiff()
}

/// CPU-only device: deterministic tests and GPU-free model conversion.
pub fn cpu_device() -> Device {
    Device::flex()
}

/// Human-readable backend name for logs.
pub fn backend_label() -> &'static str {
    if cfg!(feature = "cuda") {
        "CUDA"
    } else {
        "WGPU"
    }
}

/// Map the web UI device selectors (`DiscreteGpu(0)`, `Cpu`, ...) to a WGPU
/// [`Device`]. Unknown selectors fall back to the default device so a stale
/// frontend can never crash the worker.
pub fn wgpu_device_from_selector(selector: &str) -> Device {
    let s = selector.trim();
    let kind = if s.eq_ignore_ascii_case("cpu") {
        DeviceKind::Cpu
    } else if let Some(rest) = s.strip_prefix("DiscreteGpu(") {
        rest.trim_end_matches(')')
            .parse()
            .map(DeviceKind::DiscreteGpu)
            .unwrap_or_default()
    } else if let Some(rest) = s.strip_prefix("IntegratedGpu(") {
        rest.trim_end_matches(')')
            .parse()
            .map(DeviceKind::IntegratedGpu)
            .unwrap_or_default()
    } else if let Some(rest) = s.strip_prefix("VirtualGpu(") {
        rest.trim_end_matches(')')
            .parse()
            .map(DeviceKind::VirtualGpu)
            .unwrap_or_default()
    } else {
        DeviceKind::DefaultDevice
    };
    Device::wgpu(kind)
}
