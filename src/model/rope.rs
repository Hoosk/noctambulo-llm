//! Rotary position embeddings and context-extension scaling (linear/ntk/yarn).

use burn::tensor::{Device, Tensor, TensorData};

use crate::config::RopeScalingKind;

/// Standard RoPE base, optimal for contexts up to ~8K without scaling.
pub const ROPE_BASE_THETA: f32 = 10000.0;
const YARN_BETA_FAST: f32 = 32.0;
const YARN_BETA_SLOW: f32 = 1.0;

/// Effective number of positions the model can address. With `factor > 1`
/// RoPE scaling stretches the usable window; the RoPE angles are computed per
/// position, so there is no table cap. The KV cache is bounded separately by
/// the sliding window when one is configured.
pub fn extended_ctx_len(n_ctx: usize, kind: RopeScalingKind, factor: f32) -> usize {
    if kind == RopeScalingKind::None || factor <= 1.0 || !factor.is_finite() {
        return n_ctx;
    }
    ((n_ctx as f32 * factor).ceil() as usize).max(n_ctx)
}

/// Resolved RoPE frequencies for one head size.
#[derive(Debug, Clone)]
pub struct RopeSpec {
    /// Inverse frequencies for the first half of the head dimension.
    pub freqs: Vec<f32>,
    /// Multiplier applied to cos/sin (YaRN attention scaling).
    pub mscale: f32,
}

/// Build the inverse-frequency table for a context-extension strategy.
///
/// - `None`: the classic `1 / base^(2i/d)` table.
/// - `Linear`: positions are divided by `factor`, equivalent to scaling every
///   inverse frequency by `1/factor`.
/// - `Ntk`: dynamic-NTK base adjustment `base' = base * factor^(d/(d-2))`.
/// - `Yarn`: piecewise interpolation between the linear-scaled and original
///   frequencies (ramp between the 1- and 32-rotation dimensions) plus the
///   standard `mscale` attention factor.
///
/// `base` is the model's RoPE theta (10000 by default; imported models carry
/// their own value, e.g. 1e6 for Qwen2.5 or 5e5 for Llama-3).
pub fn rope_spec(
    n_ctx: usize,
    head_dim: usize,
    kind: RopeScalingKind,
    factor: f32,
    base: f32,
) -> RopeSpec {
    let half = head_dim / 2;
    let factor = if factor.is_finite() && factor > 1.0 { factor } else { 1.0 };
    let base = if base.is_finite() && base > 0.0 { base } else { ROPE_BASE_THETA };

    let inv_freqs = |base: f32| -> Vec<f32> {
        (0..half)
            .map(|i| {
                let exponent = (2 * i) as f32 / head_dim as f32;
                1.0 / base.powf(exponent)
            })
            .collect()
    };

    match kind {
        RopeScalingKind::None => RopeSpec {
            freqs: inv_freqs(base),
            mscale: 1.0,
        },
        RopeScalingKind::Linear => RopeSpec {
            freqs: inv_freqs(base).into_iter().map(|f| f / factor).collect(),
            mscale: 1.0,
        },
        RopeScalingKind::Ntk => {
            let exponent = head_dim as f32 / (head_dim as f32 - 2.0).max(1.0);
            let base = base * factor.powf(exponent);
            RopeSpec {
                freqs: inv_freqs(base),
                mscale: 1.0,
            }
        }
        RopeScalingKind::Yarn => {
            let original = inv_freqs(base);
            if factor <= 1.0 {
                return RopeSpec {
                    freqs: original,
                    mscale: 1.0,
                };
            }
            let interpolated: Vec<f32> = original.iter().map(|f| f / factor).collect();

            // Dimension (in rotation cycles) at which a frequency completes
            // `rotations` turns inside the original context window. `beta_fast`
            // bounds the high-frequency end (kept as-is) and `beta_slow` the
            // low-frequency end (interpolated), matching the YaRN paper.
            let correction = |rotations: f32| -> f32 {
                let dim = head_dim as f32;
                (dim * ((n_ctx as f32) / (rotations * 2.0 * std::f32::consts::PI)).ln())
                    / (2.0 * base.ln())
            };
            let low = correction(YARN_BETA_FAST).floor().clamp(0.0, (half - 1) as f32);
            let high = correction(YARN_BETA_SLOW).ceil().clamp(0.0, (half - 1) as f32);

            let freqs = (0..half)
                .map(|i| {
                    let ramp = if high > low {
                        ((i as f32 - low) / (high - low)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    // High frequencies (low i) keep the original table;
                    // low frequencies are interpolated by `1/factor`.
                    original[i] * (1.0 - ramp) + interpolated[i] * ramp
                })
                .collect();

            RopeSpec {
                freqs,
                mscale: 0.1 * factor.ln() + 1.0,
            }
        }
    }
}

/// Build the cos/sin rotation tensors for the absolute positions
/// `start_pos..start_pos + seq_len`, one row per position. Angles are computed
/// on the fly (no precomputed table), so any position is addressable.
pub fn rope_cos_sin(
    freqs: &[f32],
    mscale: f32,
    start_pos: usize,
    seq_len: usize,
    device: &Device,
) -> (Tensor<4>, Tensor<4>) {
    let half_dim = freqs.len();
    let mut angles_vec = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..(start_pos + seq_len) {
        let p = pos as f32;
        for &f in freqs {
            angles_vec.push(p * f);
        }
    }

    let angles_data = TensorData::new(angles_vec, [1, 1, seq_len, half_dim]);
    let angles = Tensor::<4>::from_data(angles_data, device);
    let cos = angles.clone().cos() * mscale;
    let sin = angles.sin() * mscale;

    let cos = Tensor::cat(vec![cos.clone(), cos], 3);
    let sin = Tensor::cat(vec![sin.clone(), sin], 3);
    (cos, sin)
}

/// Like [`rope_cos_sin`] but for an arbitrary list of absolute positions
/// (ring-cache slots are not contiguous). Sentinel positions (`i32::MAX`,
/// empty slots) are treated as 0; they are masked out by the attention.
pub fn rope_cos_sin_at(
    freqs: &[f32],
    mscale: f32,
    positions: &[i32],
    device: &Device,
) -> (Tensor<4>, Tensor<4>) {
    let half_dim = freqs.len();
    let n = positions.len();
    let mut angles_vec = Vec::with_capacity(n * half_dim);
    for &pos in positions {
        let p = if pos == i32::MAX { 0.0 } else { pos as f32 };
        for &f in freqs {
            angles_vec.push(p * f);
        }
    }
    let angles = Tensor::<4>::from_data(TensorData::new(angles_vec, [1, 1, n, half_dim]), device);
    let cos = angles.clone().cos() * mscale;
    let sin = angles.sin() * mscale;
    let cos = Tensor::cat(vec![cos.clone(), cos], 3);
    let sin = Tensor::cat(vec![sin.clone(), sin], 3);
    (cos, sin)
}
