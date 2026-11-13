//! Backend and device selection.
//!
//! Burn 0.22 dispatches tensor operations at runtime, so the engine is not
//! generic over a backend: there is a single [`Device`] type and the enabled
//! Cargo features select the available backends. This module owns the mapping
//! from user-facing selectors (`wgpu:integrated(0)`, `vulkan`, `flex`,
//! `ws://host:port#0`, ...) to that device, plus startup defaults for the CLI.

use std::sync::OnceLock;

use burn::tensor::{Device, DeviceKind, FloatDType};

static DEFAULT_DEVICE: OnceLock<Device> = OnceLock::new();
static DEFAULT_LABEL: OnceLock<String> = OnceLock::new();
static REMOTE_TOKEN: OnceLock<String> = OnceLock::new();

/// A resolved backend selector.
#[derive(Debug, Clone)]
pub struct SelectedDevice {
    pub device: Device,
    /// Short human-readable backend name (`Vulkan`, `WGPU`, `Flex (CPU)`, ...).
    pub backend: String,
    /// Canonical selector, safe to show in logs and the UI.
    pub selector: String,
}

/// One selectable device, with the metadata the CLI and Web UI display.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DeviceChoice {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub selector: String,
    pub vendor: u32,
    pub device: u32,
    pub driver: String,
    pub vram_bytes: Option<u64>,
    pub supports_f32: bool,
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub remote: bool,
    pub selectable: bool,
}

/// Install the process-wide default device before any command runs.
pub fn set_default_device(device: Device, label: impl Into<String>) -> Result<(), Device> {
    let label = label.into();
    DEFAULT_LABEL.set(label).ok();
    DEFAULT_DEVICE.set(device).map_err(|device| device)
}

/// Token used by remote selectors when `NOCTAMBULO_REMOTE_TOKEN` is not set.
pub fn set_remote_token(token: Option<String>) {
    if let Some(token) = token.filter(|t| !t.is_empty()) {
        REMOTE_TOKEN.set(token).ok();
    }
}

fn remote_token() -> Option<String> {
    REMOTE_TOKEN
        .get()
        .cloned()
        .or_else(|| std::env::var("NOCTAMBULO_REMOTE_TOKEN").ok())
        .filter(|t| !t.is_empty())
}

/// Default compute device for the CLI: an explicit selector, then `cuda` when
/// compiled, then WGPU, then Flex. `NOCTAMBULO_CPU=1` still forces Flex.
pub fn default_device() -> Device {
    if let Some(device) = DEFAULT_DEVICE.get() {
        return device.clone();
    }
    if std::env::var_os("NOCTAMBULO_CPU").is_some() {
        return Device::flex();
    }
    if let Ok(selector) = std::env::var("NOCTAMBULO_DEVICE") {
        match device_from_selector(&selector) {
            Ok(device) => return device,
            Err(err) => eprintln!("warning: {err}; using the automatic backend"),
        }
    }
    auto_device()
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
pub fn backend_label() -> String {
    if let Some(label) = DEFAULT_LABEL.get() {
        return label.clone();
    }
    if std::env::var_os("NOCTAMBULO_CPU").is_some() {
        return "Flex (CPU)".into();
    }
    if let Ok(selector) = std::env::var("NOCTAMBULO_DEVICE") {
        return selector;
    }
    if cfg!(feature = "cuda") {
        "CUDA".into()
    } else if cfg!(feature = "wgpu") {
        "WGPU".into()
    } else if cfg!(feature = "flex") {
        "Flex (CPU)".into()
    } else {
        "unknown".into()
    }
}

fn auto_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        return Device::cuda(0);
    }
    #[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
    {
        return Device::wgpu(DeviceKind::DefaultDevice);
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "wgpu"), feature = "flex"))]
    {
        return Device::flex();
    }
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "wgpu"),
        not(feature = "flex")
    ))]
    {
        panic!("no Burn execution backend compiled in");
    }
}

/// Parse a selector into a device. Remote selectors connect eagerly.
pub fn device_from_selector(selector: &str) -> Result<Device, String> {
    resolve_device_with_token(selector, remote_token().as_deref()).map(|selected| selected.device)
}

/// Like [`device_from_selector`], but never fails: unknown selectors fall back
/// to the automatic backend (a stale Web UI can never crash a worker).
pub fn device_from_selector_or_default(selector: &str) -> Device {
    match device_from_selector(selector) {
        Ok(device) => device,
        Err(err) => {
            eprintln!("warning: {err}; using the automatic backend");
            auto_device()
        }
    }
}

/// Backwards-compatible helper used by the web runner.
pub fn wgpu_device_from_selector(selector: &str) -> Device {
    device_from_selector_or_default(selector)
}

/// Parse a selector, using `token` for remote hosts instead of the environment.
pub fn resolve_device_with_token(
    selector: &str,
    token: Option<&str>,
) -> Result<SelectedDevice, String> {
    let raw = selector.trim();
    let lower = raw.to_ascii_lowercase();

    if raw.is_empty()
        || matches!(lower.as_str(), "auto" | "default" | "defaultdevice")
    {
        let device = auto_device();
        return Ok(SelectedDevice {
            device,
            backend: auto_label(),
            selector: "auto".into(),
        });
    }

    if lower.starts_with("ws://") || lower.starts_with("wss://") || lower.starts_with("iroh://") {
        return resolve_remote(raw, token);
    }

    if let Some(kind) = legacy_wgpu_kind(raw) {
        let canonical = canonical_wgpu(&kind, "wgpu");
        return Ok(selected(wgpu_device(kind)?, "WGPU", canonical));
    }

    if let Some((prefix, rest)) = split_prefix(&lower) {
        if let Some(kind) = wgpu_kind(rest) {
            let canonical = canonical_wgpu(&kind, prefix);
            let backend = match prefix {
                "wgpu" => wgpu_device(kind)?,
                "vulkan" => vulkan_device(kind)?,
                "webgpu" => webgpu_device(kind)?,
                "metal" => metal_device(kind)?,
                other => return Err(format!("unknown backend prefix '{other}'")),
            };
            let name = match prefix {
                "vulkan" => "Vulkan",
                "webgpu" => "WebGPU",
                "metal" => "Metal",
                _ => "WGPU",
            };
            return Ok(selected(backend, name, canonical));
        }
    }

    let default_kind = DeviceKind::DefaultDevice;
    match lower.as_str() {
        "wgpu" => {
            return Ok(selected(
                wgpu_device(default_kind)?,
                "WGPU",
                "wgpu".into(),
            ));
        }
        "vulkan" => {
            return Ok(selected(
                vulkan_device(default_kind)?,
                "Vulkan",
                "vulkan".into(),
            ));
        }
        "webgpu" => {
            return Ok(selected(
                webgpu_device(default_kind)?,
                "WebGPU",
                "webgpu".into(),
            ));
        }
        "metal" => {
            return Ok(selected(
                metal_device(default_kind)?,
                "Metal",
                "metal".into(),
            ));
        }
        _ => {}
    }

    if let Some(kind) = wgpu_kind(&lower) {
        let canonical = canonical_wgpu(&kind, "wgpu");
        return Ok(selected(wgpu_device(kind)?, "WGPU", canonical));
    }

    if matches!(lower.as_str(), "flex" | "cpu-flex" | "flex-cpu") {
        return Ok(selected(Device::flex(), "Flex (CPU)", "flex".into()));
    }

    if matches!(lower.as_str(), "cpu" | "cubecl-cpu" | "cubecl") {
        return Ok(selected(cpu_backend_device()?, "CubeCL CPU", "cpu".into()));
    }

    if lower == "cuda" || lower.starts_with("cuda:") {
        let index = parse_index_suffix(&lower, "cuda:")?;
        return Ok(selected(
            cuda_device(index)?,
            "CUDA",
            format!("cuda:{index}"),
        ));
    }

    if lower == "rocm" || lower.starts_with("rocm:") {
        let index = parse_index_suffix(&lower, "rocm:")?;
        return Ok(selected(
            rocm_device(index)?,
            "ROCm",
            format!("rocm:{index}"),
        ));
    }

    Err(format!("unknown device selector '{raw}'"))
}

fn selected(device: Device, backend: impl Into<String>, selector: String) -> SelectedDevice {
    SelectedDevice {
        device,
        backend: backend.into(),
        selector,
    }
}

fn auto_label() -> String {
    #[cfg(feature = "cuda")]
    {
        return "CUDA".into();
    }
    #[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
    {
        return "WGPU".into();
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "wgpu"), feature = "flex"))]
    {
        return "Flex (CPU)".into();
    }
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "wgpu"),
        not(feature = "flex")
    ))]
    {
        "unknown".into()
    }
}

fn split_prefix(selector: &str) -> Option<(&str, &str)> {
    let (prefix, rest) = selector.split_once(':')?;
    Some((prefix, rest))
}

fn parse_index_suffix(selector: &str, prefix: &str) -> Result<usize, String> {
    let rest = selector.strip_prefix(prefix).unwrap_or("");
    if rest.is_empty() {
        return Ok(0);
    }
    rest.parse()
        .map_err(|_| format!("invalid device index in '{selector}'"))
}

fn legacy_wgpu_kind(selector: &str) -> Option<DeviceKind> {
    let s = selector.trim();
    if s.eq_ignore_ascii_case("cpu") {
        return Some(DeviceKind::Cpu);
    }
    if let Some(rest) = strip_call(s, "DiscreteGpu") {
        return rest.parse().ok().map(DeviceKind::DiscreteGpu);
    }
    if let Some(rest) = strip_call(s, "IntegratedGpu") {
        return rest.parse().ok().map(DeviceKind::IntegratedGpu);
    }
    if let Some(rest) = strip_call(s, "VirtualGpu") {
        return rest.parse().ok().map(DeviceKind::VirtualGpu);
    }
    None
}

fn strip_call<'a>(selector: &'a str, name: &str) -> Option<&'a str> {
    selector
        .strip_prefix(name)?
        .strip_prefix('(')?
        .strip_suffix(')')
        .map(str::trim)
}

fn wgpu_kind(selector: &str) -> Option<DeviceKind> {
    let s = selector.trim();
    if s.is_empty() || matches!(s, "default" | "defaultdevice") {
        return Some(DeviceKind::DefaultDevice);
    }
    if s == "cpu" {
        return Some(DeviceKind::Cpu);
    }
    if let Some(rest) = strip_call(s, "discrete") {
        return rest.parse().ok().map(DeviceKind::DiscreteGpu);
    }
    if let Some(rest) = strip_call(s, "integrated") {
        return rest.parse().ok().map(DeviceKind::IntegratedGpu);
    }
    if let Some(rest) = strip_call(s, "virtual") {
        return rest.parse().ok().map(DeviceKind::VirtualGpu);
    }
    if let Some(rest) = strip_call(s, "discretegpu") {
        return rest.parse().ok().map(DeviceKind::DiscreteGpu);
    }
    if let Some(rest) = strip_call(s, "integratedgpu") {
        return rest.parse().ok().map(DeviceKind::IntegratedGpu);
    }
    if let Some(rest) = strip_call(s, "virtualgpu") {
        return rest.parse().ok().map(DeviceKind::VirtualGpu);
    }
    None
}

fn kind_suffix(kind: &DeviceKind) -> String {
    match kind {
        DeviceKind::DiscreteGpu(i) => format!("discrete({i})"),
        DeviceKind::IntegratedGpu(i) => format!("integrated({i})"),
        DeviceKind::VirtualGpu(i) => format!("virtual({i})"),
        DeviceKind::Cpu => "cpu".into(),
        DeviceKind::DefaultDevice => "default".into(),
        DeviceKind::Existing(_) => "existing".into(),
    }
}

fn canonical_wgpu(kind: &DeviceKind, prefix: &str) -> String {
    let suffix = kind_suffix(kind);
    if suffix == "default" {
        prefix.to_string()
    } else {
        format!("{prefix}:{suffix}")
    }
}

fn wgpu_device(kind: DeviceKind) -> Result<Device, String> {
    #[cfg(feature = "wgpu")]
    {
        Ok(Device::wgpu(kind))
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = kind;
        Err("WGPU support is not compiled in (enable the `wgpu` feature)".into())
    }
}

fn vulkan_device(kind: DeviceKind) -> Result<Device, String> {
    #[cfg(feature = "vulkan")]
    {
        Ok(Device::vulkan(kind))
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = kind;
        Err("Vulkan support is not compiled in (enable the `vulkan` feature)".into())
    }
}

fn webgpu_device(kind: DeviceKind) -> Result<Device, String> {
    #[cfg(feature = "webgpu")]
    {
        Ok(Device::webgpu(kind))
    }
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = kind;
        Err("WebGPU support is not compiled in (enable the `webgpu` feature)".into())
    }
}

fn metal_device(kind: DeviceKind) -> Result<Device, String> {
    #[cfg(feature = "metal")]
    {
        Ok(Device::metal(kind))
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = kind;
        Err("Metal support is not compiled in (enable the `metal` feature on macOS)".into())
    }
}

fn cpu_backend_device() -> Result<Device, String> {
    #[cfg(feature = "cpu")]
    {
        Ok(Device::cpu())
    }
    #[cfg(not(feature = "cpu"))]
    {
        Err("CubeCL CPU is not compiled in (enable the `cpu` feature; it needs LLVM)".into())
    }
}

fn cuda_device(index: usize) -> Result<Device, String> {
    #[cfg(feature = "cuda")]
    {
        Ok(Device::cuda(index))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = index;
        Err("CUDA is not compiled in (enable the `cuda` feature)".into())
    }
}

fn rocm_device(index: usize) -> Result<Device, String> {
    #[cfg(feature = "rocm")]
    {
        Ok(Device::rocm(index))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = index;
        Err("ROCm is not compiled in (enable the `rocm` feature)".into())
    }
}

fn resolve_remote(selector: &str, token: Option<&str>) -> Result<SelectedDevice, String> {
    let (base, index) = split_remote_index(selector);
    #[cfg(feature = "remote")]
    {
        use burn::remote::{EndpointId, RemoteHost};

        let host = if base.starts_with("ws://") || base.starts_with("wss://") {
            #[cfg(feature = "remote-websocket")]
            {
                RemoteHost::websocket(base)
            }
            #[cfg(not(feature = "remote-websocket"))]
            {
                return Err(
                    "WebSocket remote support is not compiled in (enable `remote-websocket`)"
                        .into(),
                );
            }
        } else {
            let id = base
                .strip_prefix("iroh://")
                .ok_or_else(|| format!("invalid remote selector '{selector}'"))?;
            let id = id.split(['/', '?']).next().unwrap_or(id);
            let id: EndpointId = id
                .parse()
                .map_err(|_| format!("invalid Iroh endpoint id '{id}'"))?;
            RemoteHost::iroh(id)
        };
        let host = match token {
            Some(token) if !token.is_empty() => host.with_credential(token),
            _ => host,
        };
        let device = Device::remote_options(&host)
            .device_index(index)
            .init()
            .map_err(|err| format!("cannot connect to remote compute host '{base}': {err}"))?;
        Ok(SelectedDevice {
            device,
            backend: "Remote".into(),
            selector: format!("{base}#{index}"),
        })
    }
    #[cfg(not(feature = "remote"))]
    {
        let _ = (base, index, token);
        Err("remote support is not compiled in (enable the `remote` feature)".into())
    }
}

fn split_remote_index(selector: &str) -> (&str, usize) {
    if let Some((base, fragment)) = selector.rsplit_once('#')
        && let Ok(index) = fragment.parse::<usize>()
    {
        return (base, index);
    }
    (selector, 0)
}

/// Every locally selectable backend/device, with canonical selectors.
pub fn local_devices() -> Vec<DeviceChoice> {
    let mut out = wgpu_choices();

    #[cfg(feature = "vulkan")]
    if !out.iter().any(|d| d.backend == "Vulkan") {
        out.push(synthetic(
            "Vulkan (pinned)",
            "Vulkan",
            "GPU",
            "vulkan",
        ));
    }

    #[cfg(feature = "cuda")]
    out.push(synthetic("CUDA (device 0)", "CUDA", "GPU", "cuda:0"));

    #[cfg(feature = "rocm")]
    out.push(synthetic("ROCm (device 0)", "ROCm", "GPU", "rocm:0"));

    #[cfg(feature = "cpu")]
    out.push(synthetic("CPU (CubeCL)", "CubeCL CPU", "CPU", "cpu"));

    if cfg!(feature = "flex") {
        let device = Device::flex();
        out.push(DeviceChoice {
            name: "CPU (Flex, SIMD + Rayon)".into(),
            backend: "Flex (CPU)".into(),
            device_type: "CPU".into(),
            selector: "flex".into(),
            vendor: 0,
            device: 0,
            driver: String::new(),
            vram_bytes: None,
            supports_f32: supports(&device, FloatDType::F32),
            supports_f16: supports(&device, FloatDType::F16),
            supports_bf16: supports(&device, FloatDType::BF16),
            remote: false,
            selectable: true,
        });
    }

    out
}

fn supports(device: &Device, dtype: FloatDType) -> bool {
    device.supports_dtype(dtype)
}

fn synthetic(name: &str, backend: &str, device_type: &str, selector: &str) -> DeviceChoice {
    DeviceChoice {
        name: name.into(),
        backend: backend.into(),
        device_type: device_type.into(),
        selector: selector.into(),
        vendor: 0,
        device: 0,
        driver: String::new(),
        vram_bytes: None,
        supports_f32: true,
        supports_f16: true,
        supports_bf16: false,
        remote: false,
        selectable: true,
    }
}

fn wgpu_choices() -> Vec<DeviceChoice> {
    use std::collections::HashMap;

    let mut out = Vec::new();
    let instance = wgpu::Instance::default();
    let adapters = futures::executor::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let mut counts: HashMap<(u8, u8), usize> = HashMap::new();

    for adapter in adapters {
        let info = adapter.get_info();
        let backend_name = match info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DX12",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "Other",
        };
        let prefix = match info.backend {
            wgpu::Backend::Vulkan => "vulkan",
            wgpu::Backend::Metal => "metal",
            wgpu::Backend::BrowserWebGpu => "webgpu",
            _ => "wgpu",
        };
        let (kind, type_name, type_code) = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => ("discrete", "Discrete GPU", 0u8),
            wgpu::DeviceType::IntegratedGpu => ("integrated", "Integrated GPU", 1u8),
            wgpu::DeviceType::VirtualGpu => ("virtual", "Virtual GPU", 2u8),
            wgpu::DeviceType::Cpu => ("cpu", "CPU", 3u8),
            _ => ("", "Other", 4u8),
        };
        let backend_code = match info.backend {
            wgpu::Backend::Vulkan => 0u8,
            wgpu::Backend::Metal => 1,
            wgpu::Backend::Dx12 => 2,
            wgpu::Backend::Gl => 3,
            wgpu::Backend::BrowserWebGpu => 4,
            _ => 5,
        };
        let index = if kind.is_empty() {
            0
        } else {
            let next = counts.entry((backend_code, type_code)).or_insert(0);
            let index = *next;
            *next += 1;
            index
        };
        let selector = if kind.is_empty() {
            prefix.to_string()
        } else if kind == "cpu" {
            format!("{prefix}:cpu")
        } else {
            format!("{prefix}:{kind}({index})")
        };

        let mut vram_bytes = None;
        if let Ok((device, _queue)) = futures::executor::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default()),
        ) {
            let max_buf = device.limits().max_buffer_size;
            if max_buf > 0 {
                vram_bytes = Some(max_buf);
            }
        }

        out.push(DeviceChoice {
            name: info.name.clone(),
            backend: backend_name.into(),
            device_type: type_name.into(),
            selector,
            vendor: info.vendor,
            device: info.device,
            driver: info.driver.clone(),
            vram_bytes,
            supports_f32: true,
            supports_f16: true,
            supports_bf16: false,
            remote: false,
            selectable: true,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_portable_selectors() {
        assert!(resolve_device_with_token("flex", None).is_ok());
        assert!(resolve_device_with_token("wgpu", None).is_ok());
        assert!(resolve_device_with_token("vulkan:integrated(0)", None).is_ok());
        assert!(resolve_device_with_token("wgpu:cpu", None).is_ok());
        assert!(resolve_device_with_token("DiscreteGpu(2)", None).is_ok());
        assert!(resolve_device_with_token("IntegratedGpu(1)", None).is_ok());
        assert!(resolve_device_with_token("VirtualGpu(3)", None).is_ok());
        assert!(resolve_device_with_token("auto", None).is_ok());
    }

    #[test]
    fn rejects_unknown_selectors() {
        assert!(resolve_device_with_token("tpu:0", None).is_err());
        assert!(resolve_device_with_token("cuda:x", None).is_err());
    }

    #[test]
    fn empty_selector_is_automatic() {
        let resolved = resolve_device_with_token("", None).unwrap();
        assert_eq!(resolved.selector, "auto");
    }

    #[test]
    fn canonical_selectors_round_trip() {
        for selector in ["wgpu", "wgpu:cpu", "wgpu:discrete(0)", "vulkan", "flex"] {
            let resolved = resolve_device_with_token(selector, None).unwrap();
            assert_eq!(resolved.selector, selector);
        }
    }

    #[test]
    fn local_devices_include_flex_and_selectors() {
        let devices = local_devices();
        assert!(devices.iter().any(|d| d.selector == "flex"));
        assert!(devices.iter().all(|d| !d.selector.is_empty()));
    }
}
