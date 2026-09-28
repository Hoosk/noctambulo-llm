//! Half-precision WGPU backend without the fusion layer.
//!
//! burn-fusion 0.21 panics on bf16 streams ("Ordering is bigger than
//! operations") on at least the Vulkan/Intel stack, and f16 shares the same
//! code path. Using the plain `CubeBackend` for half dtypes keeps `--dtype
//! f16/bf16` usable at a small performance cost; f32 keeps the fused `Wgpu`.

use burn::backend::wgpu::{CubeBackend, WgpuRuntime};

/// Non-fused WGPU backend generic over the float element type.
pub type WgpuHalf<F> = CubeBackend<WgpuRuntime, F, i32, u32>;
