//! Decoder-only transformer: layers, KV cache, RoPE, and quantization.
//!
//! The public API is re-exported here; implementation lives in focused
//! submodules:
//! - [`quant`]: KV cache storage dtypes and block quantization.
//! - [`cache`]: ring KV cache, heavy-hitter eviction, host-RAM spill/replay.
//! - [`rope`]: rotary embeddings and context-extension scaling.
//! - [`layers`]: attention, K/V & MLA projections, MLP, and blocks.
//! - [`llm`]: the `Llm` model and configuration-driven initialization.

mod cache;
mod layers;
mod llm;
mod quant;
mod rope;

#[allow(unused_imports)]
pub use cache::{KvCache, SpillStore};
#[allow(unused_imports)]
pub use layers::{
    Block, BlockConfig, CausalSelfAttention, CausalSelfAttentionConfig, KvProj, KvProjConfig,
    MlaProj, MlaProjConfig, RmsNorm, SwiGluMlp, SwiGluMlpConfig,
};
pub use llm::Llm;
#[allow(unused_imports)]
pub use quant::{KvCacheDtype, QUANT_BLOCK};
#[allow(unused_imports)]
pub use rope::{
    ROPE_BASE_THETA, RopeSpec, extended_ctx_len, rope_cos_sin, rope_cos_sin_at, rope_spec,
};

#[cfg(test)]
mod parity_tests;
#[cfg(test)]
mod tests;

// Names inherited by the in-crate test modules (`use super::*`), mirroring the
// previous single-file layout.
#[allow(unused_imports)]
use crate::config::{ModelConfig, RopeScalingKind};
#[allow(unused_imports)]
use burn::tensor::{Bool, DType, Device, FloatDType, Int, Tensor, TensorData};
#[allow(unused_imports)]
pub(crate) use cache::{
    alloc_evict_slots, cache_col_positions, cache_positions_i32, cache_write, cache_write_latent,
    sample_keep_mask,
};
#[allow(unused_imports)]
pub(crate) use quant::{
    assert_f16_storage, cache_storage_dtype, dequantize_rows, quant_block, quantize_rows,
};
