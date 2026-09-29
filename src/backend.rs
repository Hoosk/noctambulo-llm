//! Backend selection.
//!
//! Default builds run on WGPU with a non-fused half-precision variant (see
//! below). Building with `--features cuda` swaps the CLI backend to CUDA while
//! the web server keeps using WGPU for adapter enumeration.

use burn::backend::wgpu::{CubeBackend, WgpuRuntime};

/// Non-fused WGPU backend generic over the float element type.
///
/// burn-fusion 0.21 panics on bf16 streams ("Ordering is bigger than
/// operations") on at least the Vulkan/Intel stack, and f16 shares the same
/// code path. Using the plain `CubeBackend` for half dtypes keeps `--dtype
/// f16/bf16` usable at a small performance cost; f32 keeps the fused `Wgpu`.
pub type WgpuHalf<F> = CubeBackend<WgpuRuntime, F, i32, u32>;

/// Default compute backend for the CLI: WGPU, or CUDA with the `cuda` feature.
#[cfg(feature = "cuda")]
pub type MainBackend = burn::backend::Cuda;
#[cfg(not(feature = "cuda"))]
pub type MainBackend = burn::backend::Wgpu;

/// Half-precision variant used by `--dtype f16`.
#[cfg(feature = "cuda")]
pub type MainHalf = burn::backend::Cuda<burn::tensor::f16>;
#[cfg(not(feature = "cuda"))]
pub type MainHalf = WgpuHalf<burn::tensor::f16>;

/// Device type matching [`MainBackend`].
#[cfg(feature = "cuda")]
pub type MainDevice = burn::backend::cuda::CudaDevice;
#[cfg(not(feature = "cuda"))]
pub type MainDevice = burn::backend::wgpu::WgpuDevice;

/// Default device for the active backend.
pub fn default_device() -> MainDevice {
    MainDevice::default()
}

/// Human-readable backend name for logs.
pub fn backend_label() -> &'static str {
    if cfg!(feature = "cuda") {
        "CUDA"
    } else {
        "WGPU"
    }
}
