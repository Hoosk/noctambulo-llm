use crate::config::{ModelConfig, RopeScalingKind};
use burn::module::{Module, ModuleMapper, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, DType, ElementConversion, FloatDType, Int, Tensor, TensorData};
use std::sync::atomic::{AtomicU64, Ordering};

/// True when the backend computes in f16. Burn 0.21 creates f32 tensors from
/// `from_data`/`zeros` even on an f16 backend, so callers must normalize them.
pub(crate) fn backend_is_f16<B: Backend>() -> bool {
    std::any::TypeId::of::<B::FloatElem>() == std::any::TypeId::of::<burn::tensor::f16>()
}

/// Module mapper that casts every float parameter to f16.
struct CastFloatDtype;
impl<B: Backend> ModuleMapper<B> for CastFloatDtype {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        let (_, tensor, _) = param.consume();
        Param::from_tensor(tensor.cast(FloatDType::F16))
    }
}

/// KV cache storage format. `Int8` and `Q4` keep the cache quantized with
/// symmetric per-row (per token, all heads) block quantization: one f32 scale
/// per `QUANT_BLOCK` elements and packed integer data (4 int8 or 8 int4 per
/// i32 word). Burn 0.21's native `QFloat` cannot be written with
/// `slice_assign`, so the ring buffer stores the packed words directly and
/// dequantizes on read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheDtype {
    F32,
    F16,
    Int8,
    Q4,
}

/// Elements per quantization block (scale granularity).
pub const QUANT_BLOCK: usize = 32;

impl KvCacheDtype {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => Ok(Self::F32),
            "f16" | "fp16" | "half" => Ok(Self::F16),
            "int8" | "i8" | "q8" => Ok(Self::Int8),
            "q4" | "int4" | "i4" => Ok(Self::Q4),
            other => Err(format!(
                "Unknown KV cache dtype '{other}'. Available: f32, f16, int8, q4"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Int8 => "int8",
            Self::Q4 => "q4",
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Int8 | Self::Q4)
    }

    /// Values packed into each i32 word.
    fn pack(&self) -> usize {
        match self {
            Self::Int8 => 4,
            Self::Q4 => 8,
            _ => 1,
        }
    }

    /// Bits per packed value.
    fn bits(&self) -> usize {
        match self {
            Self::Int8 => 8,
            Self::Q4 => 4,
            _ => 32,
        }
    }

    /// Largest magnitude representable by the quantizer.
    fn levels(&self) -> f32 {
        match self {
            Self::Int8 => 127.0,
            Self::Q4 => 7.0,
            _ => 1.0,
        }
    }

    /// Bytes per cached element, including the amortized f32 scales.
    pub fn bytes_per_elem(&self, row_len: usize) -> f32 {
        match self {
            Self::F32 => 4.0,
            Self::F16 => 2.0,
            Self::Int8 => 1.0 + 4.0 / quant_block(row_len) as f32,
            Self::Q4 => 0.5 + 4.0 / quant_block(row_len) as f32,
        }
    }
}

/// Scale block for a row of `row_len` elements: 32 when it divides the row,
/// otherwise a single scale for the whole row (still exact for short rows).
fn quant_block(row_len: usize) -> usize {
    if row_len.is_multiple_of(QUANT_BLOCK) {
        QUANT_BLOCK
    } else {
        row_len
    }
}

/// Quantize rows `[rows, n]` into packed i32 words `[rows, n / pack]` plus one
/// f32 scale per block `[rows, n / block]`.
fn quantize_rows<B: Backend>(
    x: Tensor<B, 2>,
    dtype: KvCacheDtype,
) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
    assert!(dtype.is_quantized(), "quantize_rows needs a quantized dtype");
    let [rows, n] = x.dims();
    let pack = dtype.pack();
    let block = quant_block(n);
    assert!(
        n.is_multiple_of(pack),
        "row length {n} must be divisible by {pack} for {} packing",
        dtype.as_str()
    );
    let levels = dtype.levels();

    let xb = x.reshape([rows, n / block, block]);
    let scale = (xb.clone().abs().max_dim(2) / levels).clamp(1e-8, 1e30);
    let q = (xb / scale.clone())
        .round()
        .clamp(-levels, levels)
        .int()
        .reshape([rows, n / pack, pack]);

    // Pack the signed quantized values into bytes with `(q << shift) & mask`
    // instead of shifting the unsigned offset value: the shifted intermediate
    // always fits in i32 (unlike `offset << 24`), which burn-ndarray's checked
    // i64 -> i32 conversion requires. The resulting bit pattern is identical to
    // two's-complement `(q + offset)` bytes.
    let mut packed: Option<Tensor<B, 2, Int>> = None;
    for i in 0..pack {
        let part = q
            .clone()
            .slice([0..rows, 0..(n / pack), i..(i + 1)])
            .reshape([rows, n / pack]);
        let shifted = part.bitwise_left_shift_scalar(((dtype.bits() * i) as i32).elem());
        let mask = ((1u64 << dtype.bits()) - 1) as u32;
        let masked = shifted.bitwise_and_scalar(((mask << (dtype.bits() * i)) as i32).elem());
        packed = Some(match packed {
            None => masked,
            Some(acc) => acc.bitwise_or(masked),
        });
    }
    (packed.expect("packed words"), scale.reshape([rows, n / block]))
}

/// Inverse of [`quantize_rows`]: unpack `[rows, n / pack]` i32 words plus
/// per-block scales back to f32 `[rows, n]`.
fn dequantize_rows<B: Backend>(
    packed: Tensor<B, 2, Int>,
    scales: Tensor<B, 2>,
    dtype: KvCacheDtype,
) -> Tensor<B, 2> {
    assert!(
        dtype.is_quantized(),
        "dequantize_rows needs a quantized dtype"
    );
    let [rows, words] = packed.dims();
    let pack = dtype.pack();
    let n = words * pack;
    let block = quant_block(n);
    let bits = dtype.bits();
    let mask = (1i32 << bits) - 1;
    // Two's-complement bytes/nibbles: flip the sign bit and subtract it back
    // (no overflow, unlike shifting the top byte into the sign position).
    let sign_bit = 1i32 << (bits - 1);

    let mut parts: Vec<Tensor<B, 2, Int>> = Vec::with_capacity(pack);
    for i in 0..pack {
        let part = packed
            .clone()
            .bitwise_right_shift_scalar(((bits * i) as i32).elem())
            .bitwise_and_scalar(mask.elem());
        parts.push(part);
    }
    let u = Tensor::<B, 2, Int>::stack::<3>(parts, 2).reshape([rows, n]);
    let q = (u.bitwise_xor_scalar(sign_bit.elem()) - sign_bit).float();
    let scale = scales
        .reshape([rows, n / block, 1])
        .expand([rows, n / block, block])
        .reshape([rows, n]);
    q * scale
}

/// Panics when the backend reports f16 tensors but silently stores f32 values.
/// The fused WGPU backend does exactly that (its f16 casts are no-ops), so an
/// f16 cache created there would waste the memory saving without any error;
/// fail loudly instead of pretending.
fn assert_f16_storage<B: Backend>(device: &B::Device) {
    let probe: Tensor<B, 1> = Tensor::from_floats([1.0001, -2.0002], device);
    if probe.dtype() != DType::F32 {
        return; // native f16 backend: storage is real
    }
    let roundtrip = probe.clone().cast(FloatDType::F16).cast(FloatDType::F32);
    let before = probe.into_data().to_vec::<f32>().unwrap();
    let after = roundtrip.into_data().to_vec::<f32>().unwrap();
    assert!(
        before != after,
        "this backend reports f16 tensors but stores f32 values (f16 casts are \
         no-ops); use the non-fused backend for an f16 KV cache"
    );
}

/// Preallocated per-layer KV cache.
///
/// The buffers are sized once (capacity = trained context window, possibly
/// extended by RoPE scaling) and written with `slice_assign`, which mutates in
/// place when the handle has a single owner. This removes the per-token
/// `Tensor::cat` reallocation the engine used to do, which copied the whole
/// cache on every decoded token. Quantized dtypes store packed i32 words plus
/// per-block f32 scales.
#[derive(Clone, Debug)]
pub struct KvCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
    /// Packed quantized K rows (or the MLA latent) `[capacity, words]`.
    pub kq: Option<Tensor<B, 2, Int>>,
    pub ks: Option<Tensor<B, 2>>,
    pub vq: Option<Tensor<B, 2, Int>>,
    pub vs: Option<Tensor<B, 2>>,
    /// Number of valid rows (`<= capacity`).
    pub len: usize,
    /// Number of ring slots (window + sinks, or the full context window).
    pub capacity: usize,
    /// Absolute position stored in each ring slot (`usize::MAX` = empty).
    pub positions: Vec<usize>,
    /// Storage format for the cached rows.
    pub dtype: KvCacheDtype,
    /// Accumulated attention mass per slot, always f32. `Some` enables
    /// heavy-hitter eviction; the accumulator is independent of the storage
    /// dtype (f16/int8/q4 caches still score in f32).
    pub scores: Option<Tensor<B, 1>>,
    /// Most recent tokens protected from eviction (0 when disabled).
    pub recent: usize,
}

impl<B: Backend> KvCache<B> {
    /// Allocate zeroed buffers up front, sized to `capacity` positions.
    pub fn preallocated(
        batch: usize,
        n_kv_head: usize,
        head_dim: usize,
        capacity: usize,
        dtype: KvCacheDtype,
        device: &B::Device,
    ) -> Self {
        if dtype == KvCacheDtype::F16 {
            assert_f16_storage::<B>(device);
        }
        if dtype.is_quantized() {
            let row_len = n_kv_head * head_dim;
            let pack = dtype.pack();
            let block = quant_block(row_len);
            assert!(
                row_len.is_multiple_of(pack),
                "KV row length {row_len} must be divisible by {pack} for {} packing",
                dtype.as_str()
            );
            assert!(
                row_len.is_multiple_of(block),
                "KV row length {row_len} must be divisible by {block} for block scales"
            );
            let data: Tensor<B, 2, Int> = Tensor::zeros([capacity, row_len / pack], device);
            let scales: Tensor<B, 2> = Tensor::ones([capacity, row_len / block], device);
            let scales = if backend_is_f16::<B>() {
                scales.cast(FloatDType::F16)
            } else {
                scales
            };
            return Self {
                k: None,
                v: None,
                kq: Some(data.clone()),
                ks: Some(scales.clone()),
                vq: Some(data),
                vs: Some(scales),
                len: 0,
                capacity,
                positions: vec![usize::MAX; capacity],
                dtype,
                scores: None,
                recent: 0,
            };
        }
        let shape = [batch, n_kv_head, capacity, head_dim];
        let k: Tensor<B, 4> = Tensor::zeros(shape, device);
        let v: Tensor<B, 4> = Tensor::zeros(shape, device);
        let to_storage = |t: Tensor<B, 4>| {
            if dtype == KvCacheDtype::F16 || backend_is_f16::<B>() {
                t.cast(FloatDType::F16)
            } else {
                t
            }
        };
        Self {
            k: Some(to_storage(k)),
            v: Some(to_storage(v)),
            kq: None,
            ks: None,
            vq: None,
            vs: None,
            len: 0,
            capacity,
            positions: vec![usize::MAX; capacity],
            dtype,
            scores: None,
            recent: 0,
        }
    }

    /// Ring slot for an absolute position. The first `sink_tokens` slots are
    /// reserved for the attention sinks and never overwritten; the rest ring.
    pub fn slot_for(&self, pos: usize, sink_tokens: usize) -> usize {
        if pos < sink_tokens {
            pos
        } else {
            let ring = self.capacity - sink_tokens;
            sink_tokens + (pos - sink_tokens) % ring
        }
    }

    /// MLA variant: only the latent buffer (`k`) is allocated; `v` stays
    /// unused because K and V are expanded from the same latent.
    pub fn preallocated_single(
        batch: usize,
        heads: usize,
        dim: usize,
        capacity: usize,
        dtype: KvCacheDtype,
        device: &B::Device,
    ) -> Self {
        if dtype == KvCacheDtype::F16 {
            assert_f16_storage::<B>(device);
        }
        if dtype.is_quantized() {
            let row_len = heads * dim;
            let pack = dtype.pack();
            let block = quant_block(row_len);
            assert!(
                row_len.is_multiple_of(pack),
                "MLA latent row {row_len} must be divisible by {pack} for {} packing",
                dtype.as_str()
            );
            assert!(
                row_len.is_multiple_of(block),
                "MLA latent row {row_len} must be divisible by {block} for block scales"
            );
            let data: Tensor<B, 2, Int> = Tensor::zeros([capacity, row_len / pack], device);
            let scales: Tensor<B, 2> = Tensor::ones([capacity, row_len / block], device);
            let scales = if backend_is_f16::<B>() {
                scales.cast(FloatDType::F16)
            } else {
                scales
            };
            return Self {
                k: None,
                v: None,
                kq: Some(data),
                ks: Some(scales),
                vq: None,
                vs: None,
                len: 0,
                capacity,
                positions: vec![usize::MAX; capacity],
                dtype,
                scores: None,
                recent: 0,
            };
        }
        let k: Tensor<B, 4> = Tensor::zeros([batch, heads, capacity, dim], device);
        let k = if dtype == KvCacheDtype::F16 || backend_is_f16::<B>() {
            k.cast(FloatDType::F16)
        } else {
            k
        };
        Self {
            k: Some(k),
            v: None,
            kq: None,
            ks: None,
            vq: None,
            vs: None,
            len: 0,
            capacity,
            positions: vec![usize::MAX; capacity],
            dtype,
            scores: None,
            recent: 0,
        }
    }

    /// Turns on heavy-hitter eviction: allocates the f32 attention-mass
    /// accumulator and protects the last `recent` tokens from eviction.
    pub fn enable_eviction(&mut self, recent: usize) {
        let device = match (&self.k, &self.v, &self.kq) {
            (Some(k), _, _) => k.device(),
            (_, Some(v), _) => v.device(),
            (_, _, Some(q)) => q.device(),
            _ => panic!("KV cache has no storage to derive the device from"),
        };
        self.scores = Some(Tensor::zeros([self.capacity], &device));
        self.recent = recent;
    }

    pub fn eviction_enabled(&self) -> bool {
        self.scores.is_some()
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: burn::module::Param<Tensor<B, 1>>,
    pub eps: f32,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        let weight = Tensor::ones([dim], device);
        Self {
            weight: burn::module::Param::from_tensor(weight),
            eps: 1e-6, // Qwen2.5 / Qwen3 standard RMSNorm epsilon
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(D - 1);
        let rms = (variance + self.eps).sqrt();
        let norm = x / rms;
        let mut shape = [1; D];
        shape[D - 1] = self.weight.dims()[0];
        let weight_broadcast = self.weight.val().reshape(shape);
        norm * weight_broadcast
    }
}

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
pub fn rope_spec(n_ctx: usize, head_dim: usize, kind: RopeScalingKind, factor: f32) -> RopeSpec {
    let half = head_dim / 2;
    let factor = if factor.is_finite() && factor > 1.0 { factor } else { 1.0 };
    let base = ROPE_BASE_THETA;

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
pub fn rope_cos_sin<B: Backend>(
    freqs: &[f32],
    mscale: f32,
    start_pos: usize,
    seq_len: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half_dim = freqs.len();
    let mut angles_vec = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..(start_pos + seq_len) {
        let p = pos as f32;
        for &f in freqs {
            angles_vec.push(p * f);
        }
    }

    let angles_data = TensorData::new(angles_vec, [1, 1, seq_len, half_dim]);
    let angles = Tensor::<B, 4>::from_data(angles_data, device);
    let angles = if backend_is_f16::<B>() {
        angles.cast(FloatDType::F16)
    } else {
        angles
    };
    let cos = angles.clone().cos() * mscale;
    let sin = angles.sin() * mscale;

    let cos = Tensor::cat(vec![cos.clone(), cos], 3);
    let sin = Tensor::cat(vec![sin.clone(), sin], 3);
    (cos, sin)
}

/// Like [`rope_cos_sin`] but for an arbitrary list of absolute positions
/// (ring-cache slots are not contiguous). Sentinel positions (`i32::MAX`,
/// empty slots) are treated as 0; they are masked out by the attention.
pub fn rope_cos_sin_at<B: Backend>(
    freqs: &[f32],
    mscale: f32,
    positions: &[i32],
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half_dim = freqs.len();
    let n = positions.len();
    let mut angles_vec = Vec::with_capacity(n * half_dim);
    for &pos in positions {
        let p = if pos == i32::MAX { 0.0 } else { pos as f32 };
        for &f in freqs {
            angles_vec.push(p * f);
        }
    }
    let angles = Tensor::<B, 4>::from_data(TensorData::new(angles_vec, [1, 1, n, half_dim]), device);
    let angles = if backend_is_f16::<B>() {
        angles.cast(FloatDType::F16)
    } else {
        angles
    };
    let cos = angles.clone().cos() * mscale;
    let sin = angles.sin() * mscale;
    let cos = Tensor::cat(vec![cos.clone(), cos], 3);
    let sin = Tensor::cat(vec![sin.clone(), sin], 3);
    (cos, sin)
}

fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        x
    } else {
        let [b, n_kv_head, s, d] = x.dims();
        x.unsqueeze_dim::<5>(2)
            .expand([b, n_kv_head, n_rep, s, d])
            .reshape([b, n_kv_head * n_rep, s, d])
    }
}

fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..s, half..d]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub attn_dropout: Dropout,
    pub resid_dropout: Dropout,
    /// Inverse RoPE frequencies for the first half of each head. Angles are
    /// computed per position at forward time (no precomputed table).
    pub rope_freqs: Vec<f32>,
    /// YaRN attention scaling applied to cos/sin.
    pub rope_mscale: f32,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    /// Trained context window (eviction keep-mask density).
    pub n_ctx: usize,
    pub attn_softcap: f32,
    /// Sliding attention window in tokens (`None` = full causal attention).
    pub sliding_window: Option<usize>,
    /// Number of initial tokens kept attendable forever (attention sinks).
    pub sink_tokens: usize,
    /// Heavy-hitter eviction budget in rows (`None` = dense/ring mask).
    pub evict_budget: Option<usize>,
    /// Recent tokens protected from eviction (training mask).
    pub evict_recent: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    /// Allowed-attention mask over absolute row and column positions: causal,
    /// plus the bounded-cache geometries when configured (sliding window +
    /// sinks, or the eviction keep-mask used during training). Columns with
    /// the `i32::MAX` sentinel (empty ring slots) are always masked out.
    fn allowed_mask(
        &self,
        row_pos: Tensor<B, 1, Int>,
        col_pos: Tensor<B, 1, Int>,
        keep_old: Option<&Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2, Bool> {
        let [rows] = row_pos.dims();
        let [cols] = col_pos.dims();
        let r = row_pos.reshape([rows, 1]);
        let c = col_pos.reshape([1, cols]);
        let causal = c.clone().lower_equal(r.clone());
        if let (Some(budget), Some(keep)) = (self.evict_budget, keep_old) {
            // Eviction-aligned training mask: dense until the cache budget is
            // reached, then sinks + recent + the sampled heavy hitters.
            let dense = r.clone().lower_elem(budget as i32);
            let recent = c.clone().greater(r.sub_scalar(self.evict_recent as i32));
            let sinks = c.lower_elem(self.sink_tokens as i32);
            let sparse = sinks.bool_or(recent).bool_or(keep.clone());
            return causal.bool_and(dense.bool_or(sparse));
        }
        match self.sliding_window {
            Some(window) => {
                // Last `window` positions: distance < window  <=>  col > row - window.
                let recent = c.clone().greater(r.sub_scalar(window as i32));
                let mut allowed = causal.bool_and(recent);
                if self.sink_tokens > 0 {
                    allowed = allowed.bool_or(c.lower_elem(self.sink_tokens as i32));
                }
                allowed
            }
            None => causal,
        }
    }

    /// Query projection + RoPE for the absolute positions
    /// `start_pos..start_pos + seq_len`.
    pub fn query(&self, x: Tensor<B, 3>, start_pos: usize) -> Tensor<B, 4> {
        let [batch_size, seq_len, n_embd] = x.dims();
        let head_size = n_embd / self.n_head;
        let q = self
            .q_proj
            .forward(x)
            .reshape([batch_size, seq_len, self.n_head, head_size])
            .swap_dims(1, 2);
        let device = q.device();
        let (cos, sin) =
            rope_cos_sin::<B>(&self.rope_freqs, self.rope_mscale, start_pos, seq_len, &device);
        let q_rot = rotate_half(q.clone());
        (q * cos) + (q_rot * sin)
    }

    /// Attention output for a precomputed query and the K/V rows whose
    /// absolute positions are `col_pos`. When `collect_mass` is set, also
    /// returns the attention mass received by every K/V column (summed over
    /// the batch, heads and query rows), which drives heavy-hitter eviction.
    #[allow(clippy::too_many_arguments)]
    pub fn attend(
        &self,
        q_embed: Tensor<B, 4>,
        k_full: Tensor<B, 4>,
        v_full: Tensor<B, 4>,
        row_pos: Tensor<B, 1, Int>,
        col_pos: Tensor<B, 1, Int>,
        keep_old: Option<Tensor<B, 2, Bool>>,
        collect_mass: bool,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 1>>) {
        let [batch_size, _, seq_len, _] = q_embed.dims();
        let head_size = self.n_embd / self.n_head;
        let n_rep = self.n_head / self.n_kv_head;

        let k_repeat = repeat_kv(k_full, n_rep);
        let v_repeat = repeat_kv(v_full, n_rep);

        let k_t = k_repeat.swap_dims(2, 3);
        let scale = 1.0 / (head_size as f64).sqrt();
        let mut scores = q_embed.matmul(k_t) * scale;

        if self.attn_softcap > 0.0 {
            let cap = self.attn_softcap as f64;
            scores = (scores / cap).tanh() * cap;
        }

        let disallowed = self
            .allowed_mask(row_pos, col_pos, keep_old.as_ref())
            .bool_not();
        scores = scores.mask_fill(disallowed.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0), -1e4);

        let attn_probs = burn::tensor::activation::softmax(scores, 3);
        let mass = if collect_mass {
            let [_, _, _, cols] = attn_probs.dims();
            Some(
                attn_probs
                    .clone()
                    .sum_dim(0)
                    .sum_dim(1)
                    .sum_dim(2)
                    .reshape([cols]),
            )
        } else {
            None
        };
        let attn_probs = self.attn_dropout.forward(attn_probs);
        let out = attn_probs.matmul(v_repeat);

        let out = out.swap_dims(1, 2).reshape([batch_size, seq_len, self.n_embd]);
        (self.resid_dropout.forward(self.out_proj.forward(out)), mass)
    }
}

/// K/V projections shared by a group of layers (cross-layer KV reuse).
#[derive(Module, Debug)]
pub struct KvProj<B: Backend> {
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub rope_freqs: Vec<f32>,
    pub rope_mscale: f32,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
}

impl<B: Backend> KvProj<B> {
    /// Project and RoPE-rotate K (V stays unrotated) for the absolute
    /// positions `start_pos..start_pos + seq_len`.
    pub fn forward(&self, x: Tensor<B, 3>, start_pos: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch_size, seq_len, n_embd] = x.dims();
        let head_size = n_embd / self.n_head;
        let k = self
            .k_proj
            .forward(x.clone())
            .reshape([batch_size, seq_len, self.n_kv_head, head_size])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([batch_size, seq_len, self.n_kv_head, head_size])
            .swap_dims(1, 2);
        let device = k.device();
        let (cos, sin) =
            rope_cos_sin::<B>(&self.rope_freqs, self.rope_mscale, start_pos, seq_len, &device);
        let k_rot = rotate_half(k.clone());
        ((k * cos) + (k_rot * sin), v)
    }
}

/// Multi-head Latent Attention projections (DeepSeek MLA-style): K/V are
/// compressed into a small latent that is the only thing cached; K/V are
/// expanded from the latent at attention time. With no decoupled RoPE the
/// folded weights (`k_up · kv_down`) are exactly equivalent to a standard
/// K/V projection, so GGUF export needs no approximation.
#[derive(Module, Debug)]
pub struct MlaProj<B: Backend> {
    pub kv_down: Linear<B>,
    pub k_up: Linear<B>,
    pub v_up: Linear<B>,
    pub rope_freqs: Vec<f32>,
    pub rope_mscale: f32,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub d_latent: usize,
}

impl<B: Backend> MlaProj<B> {
    /// Compress `x` into the latent that is written to the cache:
    /// `[batch, 1, seq, d_latent]`.
    pub fn compress(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, seq, _] = x.dims();
        self.kv_down
            .forward(x)
            .reshape([batch, seq, 1, self.d_latent])
            .swap_dims(1, 2)
    }

    /// Expand cached latents `[batch, 1, n, d_latent]` into RoPE-rotated K and
    /// V `[batch, n_kv_head, n, head_dim]` at the given absolute positions.
    pub fn expand(&self, latents: Tensor<B, 4>, positions: &[i32]) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, _, n, _] = latents.dims();
        let head_dim = self.n_embd / self.n_head;
        let device = latents.device();

        let k = self
            .k_up
            .forward(latents.clone())
            .reshape([batch, 1, n, self.n_kv_head, head_dim])
            .swap_dims(1, 3)
            .reshape([batch, self.n_kv_head, n, head_dim]);
        let v = self
            .v_up
            .forward(latents)
            .reshape([batch, 1, n, self.n_kv_head, head_dim])
            .swap_dims(1, 3)
            .reshape([batch, self.n_kv_head, n, head_dim]);

        let (cos, sin) = rope_cos_sin_at::<B>(&self.rope_freqs, self.rope_mscale, positions, &device);
        let k_rot = rotate_half(k.clone());
        ((k * cos) + (k_rot * sin), v)
    }
}

/// Contiguous ring segments `(slot, offset, len)` covering the `new_seq_len`
/// rows written at `start_pos`.
fn cache_segments<B: Backend>(
    cache: &KvCache<B>,
    start_pos: usize,
    new_seq_len: usize,
    sink_tokens: usize,
) -> Vec<(usize, usize, usize)> {
    let mut segments = Vec::new();
    let mut offset = 0usize;
    while offset < new_seq_len {
        let pos = start_pos + offset;
        let slot = cache.slot_for(pos, sink_tokens);
        let mut len = 1usize;
        while offset + len < new_seq_len {
            let next_pos = start_pos + offset + len;
            if cache.slot_for(next_pos, sink_tokens) == slot + len {
                len += 1;
            } else {
                break;
            }
        }
        segments.push((slot, offset, len));
        offset += len;
    }
    segments
}

/// Training-time eviction simulation step counter: every forward pass samples
/// a different heavy-hitter keep set, so the model cannot overfit one cache
/// layout.
static KEEP_STEP: AtomicU64 = AtomicU64::new(0);

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Column keep-mask that simulates the steady-state bounded cache: sinks and
/// recent columns are handled positionally by the mask, while old columns
/// survive with probability `hitters / (n_ctx - sinks - recent)`.
fn sample_keep_mask<B: Backend>(
    seq_len: usize,
    n_ctx: usize,
    budget: usize,
    recent: usize,
    sinks: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let old_region = n_ctx.saturating_sub(sinks + recent).max(1);
    let hitters = budget.saturating_sub(sinks + recent);
    let p = (hitters as f64 / old_region as f64).min(1.0);
    let step = KEEP_STEP.fetch_add(1, Ordering::Relaxed);
    let mut bits = vec![false; seq_len];
    for (c, bit) in bits.iter_mut().enumerate() {
        let h = splitmix64(step ^ (c as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let u = (h >> 11) as f64 / (1u64 << 53) as f64;
        *bit = u < p;
    }
    Tensor::<B, 1, Bool>::from_data(TensorData::new(bits, [seq_len]), device)
        .reshape([1, seq_len])
}

/// Slot assigned to every new row when heavy-hitter eviction is enabled.
/// While the cache has free rows they are appended in order; once full the
/// victim is the live row with the lowest accumulated attention score that is
/// neither a sink, nor inside the recent window, nor written by this same
/// call. `positions` and `len` are updated here; victim scores reset to zero.
fn alloc_evict_slots<B: Backend>(
    cache: &mut KvCache<B>,
    start_pos: usize,
    new_seq_len: usize,
    sink_tokens: usize,
) -> Vec<usize> {
    assert!(
        cache.capacity > sink_tokens + cache.recent,
        "KV budget ({}) must exceed sink tokens ({}) + recent tokens ({})",
        cache.capacity,
        sink_tokens,
        cache.recent
    );
    let scores_host: Option<Vec<f32>> = if cache.len + new_seq_len > cache.capacity {
        Some(
            cache
                .scores
                .as_ref()
                .expect("eviction allocator needs the score accumulator")
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("f32 score readback"),
        )
    } else {
        None
    };

    let mut slots = Vec::with_capacity(new_seq_len);
    let mut victims: Vec<usize> = Vec::new();
    for offset in 0..new_seq_len {
        let pos = start_pos + offset;
        if cache.len < cache.capacity {
            let slot = cache.len;
            cache.len += 1;
            cache.positions[slot] = pos;
            slots.push(slot);
            continue;
        }
        let scores = scores_host.as_ref().expect("scores read back when full");
        // Rows at or after `cutoff` are the recent window and stay put.
        let cutoff = pos.saturating_sub(cache.recent);
        let victim = (0..cache.capacity)
            .filter(|&s| {
                let p = cache.positions[s];
                p != usize::MAX
                    && p >= sink_tokens
                    && p < cutoff
                    && !victims.contains(&s)
                    && !slots.contains(&s)
            })
            .min_by(|&a, &b| {
                scores[a]
                    .partial_cmp(&scores[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| cache.positions[a].cmp(&cache.positions[b]))
            })
            .expect("budget must exceed sinks + recent so a victim exists");
        cache.positions[victim] = pos;
        victims.push(victim);
        slots.push(victim);
    }

    if !victims.is_empty() {
        // Fresh rows start from a clean score; multiply by a 0/1 mask (Burn's
        // scatter only implements `Add` and its Int dtype plumbing differs per
        // backend, so the mask is the portable option).
        let mut scores = cache.scores.take().expect("scores enabled");
        let device = scores.device();
        let mask: Vec<f32> = (0..cache.capacity)
            .map(|s| if victims.contains(&s) { 0.0 } else { 1.0 })
            .collect();
        let mask = Tensor::<B, 1>::from_data(
            TensorData::new(mask, [cache.capacity]),
            &device,
        );
        scores = scores * mask;
        cache.scores = Some(scores);
    }
    slots
}

/// Writes K/V rows for the absolute positions `start_pos..` into the ring
/// cache with contiguous `slice_assign` segments (in place when the buffer has
/// a single owner). Shared by every group leader.
fn cache_write<B: Backend>(
    cache: &mut KvCache<B>,
    k_embed: Tensor<B, 4>,
    v: Tensor<B, 4>,
    start_pos: usize,
    sink_tokens: usize,
) {
    let [batch_size, n_kv_head, new_seq_len, head_size] = k_embed.dims();
    assert!(
        cache.capacity > sink_tokens,
        "KV cache capacity ({}) must exceed the sink tokens ({})",
        cache.capacity,
        sink_tokens
    );
    if cache.eviction_enabled() {
        cache_write_evict(cache, k_embed, v, start_pos, sink_tokens);
        return;
    }
    for pos in start_pos..(start_pos + new_seq_len) {
        let slot = cache.slot_for(pos, sink_tokens);
        cache.positions[slot] = pos;
    }
    let segments = cache_segments(cache, start_pos, new_seq_len, sink_tokens);

    if cache.dtype.is_quantized() {
        let row_len = n_kv_head * head_size;
        let pack = cache.dtype.pack();
        let block = quant_block(row_len);
        let words = row_len / pack;
        let blocks = row_len / block;
        let to_rows =
            |x: Tensor<B, 4>| x.swap_dims(1, 2).reshape([batch_size * new_seq_len, row_len]);
        let (kp, ksc) = quantize_rows(to_rows(k_embed), cache.dtype);
        let (vp, vsc) = quantize_rows(to_rows(v), cache.dtype);
        let mut kq = cache.kq.take().expect("kv cache kq buffer");
        let mut ks = cache.ks.take().expect("kv cache ks buffer");
        let mut vq = cache.vq.take().expect("kv cache vq buffer");
        let mut vs = cache.vs.take().expect("kv cache vs buffer");
        for (slot, offset, len) in segments {
            kq = kq.slice_assign(
                [slot..slot + len, 0..words],
                kp.clone().slice([offset..offset + len, 0..words]),
            );
            ks = ks.slice_assign(
                [slot..slot + len, 0..blocks],
                ksc.clone().slice([offset..offset + len, 0..blocks]),
            );
            vq = vq.slice_assign(
                [slot..slot + len, 0..words],
                vp.clone().slice([offset..offset + len, 0..words]),
            );
            vs = vs.slice_assign(
                [slot..slot + len, 0..blocks],
                vsc.clone().slice([offset..offset + len, 0..blocks]),
            );
        }
        cache.len = (cache.len + new_seq_len).min(cache.capacity);
        cache.kq = Some(kq);
        cache.ks = Some(ks);
        cache.vq = Some(vq);
        cache.vs = Some(vs);
        return;
    }

    let k_embed = if cache.dtype == KvCacheDtype::F16 {
        k_embed.cast(FloatDType::F16)
    } else {
        k_embed
    };
    let v = if cache.dtype == KvCacheDtype::F16 {
        v.cast(FloatDType::F16)
    } else {
        v
    };
    let mut k_buf = cache.k.take().expect("kv cache k buffer");
    let mut v_buf = cache.v.take().expect("kv cache v buffer");
    for (slot, offset, len) in segments {
        let write_slice = [0..batch_size, 0..n_kv_head, slot..(slot + len), 0..head_size];
        let k_part = k_embed
            .clone()
            .slice([0..batch_size, 0..n_kv_head, offset..(offset + len), 0..head_size]);
        let v_part = v
            .clone()
            .slice([0..batch_size, 0..n_kv_head, offset..(offset + len), 0..head_size]);
        k_buf = k_buf.slice_assign(write_slice.clone(), k_part);
        v_buf = v_buf.slice_assign(write_slice, v_part);
    }
    cache.len = (cache.len + new_seq_len).min(cache.capacity);
    cache.k = Some(k_buf);
    cache.v = Some(v_buf);
}

/// Heavy-hitter variant of [`cache_write`]: rows go to the slots chosen by
/// [`alloc_evict_slots`] (append while there is room, otherwise the victim
/// slot), written one row at a time because victims are not contiguous.
fn cache_write_evict<B: Backend>(
    cache: &mut KvCache<B>,
    k_embed: Tensor<B, 4>,
    v: Tensor<B, 4>,
    start_pos: usize,
    sink_tokens: usize,
) {
    let [batch_size, n_kv_head, new_seq_len, head_size] = k_embed.dims();
    let slots = alloc_evict_slots(cache, start_pos, new_seq_len, sink_tokens);

    if cache.dtype.is_quantized() {
        let row_len = n_kv_head * head_size;
        let pack = cache.dtype.pack();
        let block = quant_block(row_len);
        let words = row_len / pack;
        let blocks = row_len / block;
        let to_rows =
            |x: Tensor<B, 4>| x.swap_dims(1, 2).reshape([batch_size * new_seq_len, row_len]);
        let (kp, ksc) = quantize_rows(to_rows(k_embed), cache.dtype);
        let (vp, vsc) = quantize_rows(to_rows(v), cache.dtype);
        let mut kq = cache.kq.take().expect("kv cache kq buffer");
        let mut ks = cache.ks.take().expect("kv cache ks buffer");
        let mut vq = cache.vq.take().expect("kv cache vq buffer");
        let mut vs = cache.vs.take().expect("kv cache vs buffer");
        for (row, &slot) in slots.iter().enumerate() {
            kq = kq.slice_assign(
                [slot..slot + 1, 0..words],
                kp.clone().slice([row..row + 1, 0..words]),
            );
            ks = ks.slice_assign(
                [slot..slot + 1, 0..blocks],
                ksc.clone().slice([row..row + 1, 0..blocks]),
            );
            vq = vq.slice_assign(
                [slot..slot + 1, 0..words],
                vp.clone().slice([row..row + 1, 0..words]),
            );
            vs = vs.slice_assign(
                [slot..slot + 1, 0..blocks],
                vsc.clone().slice([row..row + 1, 0..blocks]),
            );
        }
        cache.kq = Some(kq);
        cache.ks = Some(ks);
        cache.vq = Some(vq);
        cache.vs = Some(vs);
        return;
    }

    let k_embed = if cache.dtype == KvCacheDtype::F16 {
        k_embed.cast(FloatDType::F16)
    } else {
        k_embed
    };
    let v = if cache.dtype == KvCacheDtype::F16 {
        v.cast(FloatDType::F16)
    } else {
        v
    };
    let mut k_buf = cache.k.take().expect("kv cache k buffer");
    let mut v_buf = cache.v.take().expect("kv cache v buffer");
    for (row, &slot) in slots.iter().enumerate() {
        let write_slice = [0..batch_size, 0..n_kv_head, slot..slot + 1, 0..head_size];
        let k_part = k_embed
            .clone()
            .slice([0..batch_size, 0..n_kv_head, row..row + 1, 0..head_size]);
        let v_part = v
            .clone()
            .slice([0..batch_size, 0..n_kv_head, row..row + 1, 0..head_size]);
        k_buf = k_buf.slice_assign(write_slice.clone(), k_part);
        v_buf = v_buf.slice_assign(write_slice, v_part);
    }
    cache.k = Some(k_buf);
    cache.v = Some(v_buf);
}

/// Absolute position of every ring slot as `i32` (`i32::MAX` for empty slots,
/// masked out by the attention).
fn cache_positions_i32<B: Backend>(cache: &KvCache<B>) -> Vec<i32> {
    cache
        .positions
        .iter()
        .map(|&p| if p == usize::MAX { i32::MAX } else { p as i32 })
        .collect()
}

/// MLA variant of [`cache_write`]: only the latent (`k`) buffer is written.
fn cache_write_latent<B: Backend>(
    cache: &mut KvCache<B>,
    latent: Tensor<B, 4>,
    start_pos: usize,
    sink_tokens: usize,
) {
    let [batch_size, heads, new_seq_len, dim] = latent.dims();
    assert!(
        cache.capacity > sink_tokens,
        "KV cache capacity ({}) must exceed the sink tokens ({})",
        cache.capacity,
        sink_tokens
    );
    if cache.eviction_enabled() {
        cache_write_latent_evict(cache, latent, start_pos, sink_tokens);
        return;
    }
    for pos in start_pos..(start_pos + new_seq_len) {
        let slot = cache.slot_for(pos, sink_tokens);
        cache.positions[slot] = pos;
    }
    let segments = cache_segments(cache, start_pos, new_seq_len, sink_tokens);

    if cache.dtype.is_quantized() {
        let row_len = heads * dim;
        let pack = cache.dtype.pack();
        let block = quant_block(row_len);
        let words = row_len / pack;
        let blocks = row_len / block;
        let rows = latent
            .swap_dims(1, 2)
            .reshape([batch_size * new_seq_len, row_len]);
        let (packed, scales) = quantize_rows(rows, cache.dtype);
        let mut kq = cache.kq.take().expect("mla cache kq buffer");
        let mut ks = cache.ks.take().expect("mla cache ks buffer");
        for (slot, offset, len) in segments {
            kq = kq.slice_assign(
                [slot..slot + len, 0..words],
                packed.clone().slice([offset..offset + len, 0..words]),
            );
            ks = ks.slice_assign(
                [slot..slot + len, 0..blocks],
                scales.clone().slice([offset..offset + len, 0..blocks]),
            );
        }
        cache.len = (cache.len + new_seq_len).min(cache.capacity);
        cache.kq = Some(kq);
        cache.ks = Some(ks);
        return;
    }

    let latent = if cache.dtype == KvCacheDtype::F16 {
        latent.cast(FloatDType::F16)
    } else {
        latent
    };
    let mut k_buf = cache.k.take().expect("kv cache latent buffer");
    for (slot, offset, len) in segments {
        let write_slice = [0..batch_size, 0..heads, slot..(slot + len), 0..dim];
        let part = latent
            .clone()
            .slice([0..batch_size, 0..heads, offset..(offset + len), 0..dim]);
        k_buf = k_buf.slice_assign(write_slice, part);
    }
    cache.len = (cache.len + new_seq_len).min(cache.capacity);
    cache.k = Some(k_buf);
}

/// Heavy-hitter variant of [`cache_write_latent`], one row at a time into the
/// slots chosen by [`alloc_evict_slots`].
fn cache_write_latent_evict<B: Backend>(
    cache: &mut KvCache<B>,
    latent: Tensor<B, 4>,
    start_pos: usize,
    sink_tokens: usize,
) {
    let [batch_size, heads, new_seq_len, dim] = latent.dims();
    let slots = alloc_evict_slots(cache, start_pos, new_seq_len, sink_tokens);

    if cache.dtype.is_quantized() {
        let row_len = heads * dim;
        let pack = cache.dtype.pack();
        let block = quant_block(row_len);
        let words = row_len / pack;
        let blocks = row_len / block;
        let rows = latent
            .swap_dims(1, 2)
            .reshape([batch_size * new_seq_len, row_len]);
        let (packed, scales) = quantize_rows(rows, cache.dtype);
        let mut kq = cache.kq.take().expect("mla cache kq buffer");
        let mut ks = cache.ks.take().expect("mla cache ks buffer");
        for (row, &slot) in slots.iter().enumerate() {
            kq = kq.slice_assign(
                [slot..slot + 1, 0..words],
                packed.clone().slice([row..row + 1, 0..words]),
            );
            ks = ks.slice_assign(
                [slot..slot + 1, 0..blocks],
                scales.clone().slice([row..row + 1, 0..blocks]),
            );
        }
        cache.kq = Some(kq);
        cache.ks = Some(ks);
        return;
    }

    let latent = if cache.dtype == KvCacheDtype::F16 {
        latent.cast(FloatDType::F16)
    } else {
        latent
    };
    let mut k_buf = cache.k.take().expect("kv cache latent buffer");
    for (row, &slot) in slots.iter().enumerate() {
        let write_slice = [0..batch_size, 0..heads, slot..slot + 1, 0..dim];
        let part = latent
            .clone()
            .slice([0..batch_size, 0..heads, row..row + 1, 0..dim]);
        k_buf = k_buf.slice_assign(write_slice, part);
    }
    cache.k = Some(k_buf);
}

/// Absolute position of every ring slot as an `Int` tensor (`i32::MAX` for
/// empty slots, masked out by the attention).
fn cache_col_positions<B: Backend>(cache: &KvCache<B>, device: &B::Device) -> Tensor<B, 1, Int> {
    let col_pos = cache_positions_i32(cache);
    Tensor::<B, 1, Int>::from_data(TensorData::new(col_pos, [cache.capacity]), device)
}

#[derive(Clone, Debug)]
pub struct CausalSelfAttentionConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_ctx: usize,
    pub dropout: f64,
    pub attn_softcap: f32,
    pub rope_kind: RopeScalingKind,
    pub rope_factor: f32,
    pub sliding_window: Option<usize>,
    pub sink_tokens: usize,
    /// Heavy-hitter eviction budget in rows (`None` = dense/ring mask).
    pub evict_budget: Option<usize>,
    /// Recent tokens protected from eviction (training mask).
    pub evict_recent: usize,
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        let head_dim = self.n_embd / self.n_head;
        let rope = rope_spec(self.n_ctx, head_dim, self.rope_kind, self.rope_factor);

        CausalSelfAttention {
            q_proj: LinearConfig::new(self.n_embd, self.n_embd).with_bias(true).init(device),
            out_proj: LinearConfig::new(self.n_embd, self.n_embd).with_bias(false).init(device),
            attn_dropout: DropoutConfig::new(self.dropout).init(),
            resid_dropout: DropoutConfig::new(self.dropout).init(),
            rope_freqs: rope.freqs,
            rope_mscale: rope.mscale,
            n_head: self.n_head,
            n_kv_head: self.n_kv_head,
            n_embd: self.n_embd,
            n_ctx: self.n_ctx,
            attn_softcap: self.attn_softcap,
            sliding_window: self.sliding_window,
            sink_tokens: self.sink_tokens,
            evict_budget: self.evict_budget,
            evict_recent: self.evict_recent,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KvProjConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_ctx: usize,
    pub rope_kind: RopeScalingKind,
    pub rope_factor: f32,
}

#[derive(Clone, Debug)]
pub struct MlaProjConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_ctx: usize,
    pub rope_kind: RopeScalingKind,
    pub rope_factor: f32,
    pub d_latent: usize,
}

impl MlaProjConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MlaProj<B> {
        let head_dim = self.n_embd / self.n_head;
        let kv_dim = self.n_kv_head * head_dim;
        let rope = rope_spec(self.n_ctx, head_dim, self.rope_kind, self.rope_factor);
        MlaProj {
            kv_down: LinearConfig::new(self.n_embd, self.d_latent).with_bias(false).init(device),
            k_up: LinearConfig::new(self.d_latent, kv_dim).with_bias(false).init(device),
            v_up: LinearConfig::new(self.d_latent, kv_dim).with_bias(false).init(device),
            rope_freqs: rope.freqs,
            rope_mscale: rope.mscale,
            n_head: self.n_head,
            n_kv_head: self.n_kv_head,
            n_embd: self.n_embd,
            d_latent: self.d_latent,
        }
    }
}

impl KvProjConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KvProj<B> {
        let head_dim = self.n_embd / self.n_head;
        let kv_dim = self.n_kv_head * head_dim;
        let rope = rope_spec(self.n_ctx, head_dim, self.rope_kind, self.rope_factor);
        KvProj {
            k_proj: LinearConfig::new(self.n_embd, kv_dim).with_bias(true).init(device),
            v_proj: LinearConfig::new(self.n_embd, kv_dim).with_bias(true).init(device),
            rope_freqs: rope.freqs,
            rope_mscale: rope.mscale,
            n_head: self.n_head,
            n_kv_head: self.n_kv_head,
            n_embd: self.n_embd,
        }
    }
}

#[derive(Module, Debug)]
pub struct SwiGluMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
    pub mlp_dropout: Dropout,
}

impl<B: Backend> SwiGluMlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = burn::tensor::activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        let out = self.down_proj.forward(gate * up);
        self.mlp_dropout.forward(out)
    }
}

#[derive(Clone, Debug)]
pub struct SwiGluMlpConfig {
    pub n_embd: usize,
    pub dropout: f64,
}

impl SwiGluMlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGluMlp<B> {
        let raw_hidden_dim = (8 * self.n_embd) / 3;
        let hidden_dim = raw_hidden_dim.div_ceil(256) * 256;

        SwiGluMlp {
            gate_proj: LinearConfig::new(self.n_embd, hidden_dim).with_bias(false).init(device),
            up_proj: LinearConfig::new(self.n_embd, hidden_dim).with_bias(false).init(device),
            down_proj: LinearConfig::new(hidden_dim, self.n_embd).with_bias(false).init(device),
            mlp_dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub attn: CausalSelfAttention<B>,
    pub mlp: SwiGluMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Block<B> {
    /// Runs the block with K/V rows already projected by the group leader and
    /// their absolute column positions. `keep_old` is the training-time
    /// eviction keep-mask and `collect_mass` requests the per-column attention
    /// mass used by heavy-hitter eviction during cached decoding.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        k_full: Tensor<B, 4>,
        v_full: Tensor<B, 4>,
        start_pos: usize,
        row_pos: Tensor<B, 1, Int>,
        col_pos: Tensor<B, 1, Int>,
        keep_old: Option<Tensor<B, 2, Bool>>,
        collect_mass: bool,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 1>>) {
        let ln = self.input_layernorm.forward(x.clone());
        let q = self.attn.query(ln, start_pos);
        let (attn_out, mass) =
            self.attn
                .attend(q, k_full, v_full, row_pos, col_pos, keep_old, collect_mass);
        let x = x + attn_out;
        (
            x.clone() + self.mlp.forward(self.post_attention_layernorm.forward(x)),
            mass,
        )
    }
}

#[derive(Clone, Debug)]
pub struct BlockConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_ctx: usize,
    pub dropout: f64,
    pub attn_softcap: f32,
    pub rope_kind: RopeScalingKind,
    pub rope_factor: f32,
    pub sliding_window: Option<usize>,
    pub sink_tokens: usize,
    pub evict_budget: Option<usize>,
    pub evict_recent: usize,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            attn: CausalSelfAttentionConfig {
                n_embd: self.n_embd,
                n_head: self.n_head,
                n_kv_head: self.n_kv_head,
                n_ctx: self.n_ctx,
                dropout: self.dropout,
                attn_softcap: self.attn_softcap,
                rope_kind: self.rope_kind,
                rope_factor: self.rope_factor,
                sliding_window: self.sliding_window,
                sink_tokens: self.sink_tokens,
                evict_budget: self.evict_budget,
                evict_recent: self.evict_recent,
            }
            .init(device),
            mlp: SwiGluMlpConfig {
                n_embd: self.n_embd,
                dropout: self.dropout,
            }
            .init(device),
            input_layernorm: RmsNorm::new(self.n_embd, device),
            post_attention_layernorm: RmsNorm::new(self.n_embd, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Llm<B: Backend> {
    pub embed_tokens: Embedding<B>,
    /// One K/V projection per sharing group (`ceil(n_layer / kv_share_group)`).
    /// Empty when MLA is enabled.
    pub kv: Vec<KvProj<B>>,
    /// One MLA projection per sharing group; empty unless `mla_latent_dim` is
    /// set. Only the compressed latent is cached.
    pub mla: Vec<MlaProj<B>>,
    pub blocks: Vec<Block<B>>,
    pub norm: RmsNorm<B>,
    pub n_ctx: usize,
    /// Position limit: `n_ctx` extended by the active RoPE scaling factor.
    pub max_seq: usize,
    /// Layers per K/V group (`1` = no cross-layer reuse).
    pub kv_share_group: usize,
    /// Heavy-hitter eviction budget in rows (`None` = dense/ring cache). The
    /// training forward uses it to build the eviction keep-mask.
    pub evict_budget: Option<usize>,
    /// Recent tokens protected from eviction (training mask).
    pub evict_recent: usize,
}

impl<B: Backend> Llm<B> {
    /// Total number of positions the model can attend to (the config `n_ctx`
    /// extended by the active RoPE scaling factor).
    pub fn max_seq_len(&self) -> usize {
        self.max_seq
    }

    /// Whether the model compresses K/V into an MLA latent.
    pub fn use_mla(&self) -> bool {
        !self.mla.is_empty()
    }

    /// Number of attention sinks configured on the blocks.
    pub fn sink_tokens(&self) -> usize {
        self.blocks.first().map(|b| b.attn.sink_tokens).unwrap_or(0)
    }

    /// Number of rows each layer cache needs. With a sliding window the cache
    /// is bounded to `window + sinks`; otherwise it spans the whole context.
    pub fn kv_capacity(&self) -> usize {
        match self.blocks.first() {
            Some(b) => match b.attn.sliding_window {
                Some(window) => window + b.attn.sink_tokens,
                None => self.max_seq,
            },
            None => self.max_seq,
        }
    }

    /// Index of the K/V group that serves `layer`.
    fn group_of(&self, layer: usize) -> usize {
        layer / self.kv_share_group.max(1)
    }

    /// Normalizes every float parameter to the backend compute dtype. Burn
    /// 0.21's `from_data`/`zeros` create f32 tensors even on an f16 backend,
    /// and the f16 kernels then read those buffers as f16 (garbage). Call this
    /// after `init_model` (already applied) and after every `load_record`.
    pub fn align_compute_dtype(self) -> Self {
        if !backend_is_f16::<B>() {
            return self;
        }
        self.map(&mut CastFloatDtype)
    }

    /// Preallocated caches (one per K/V group) sized to `capacity` positions.
    /// MLA groups cache only the compressed latent. Quantized dtypes store
    /// packed integer words plus per-block scales and dequantize on read.
    pub fn new_kv_caches(
        &self,
        capacity: usize,
        dtype: KvCacheDtype,
        device: &B::Device,
    ) -> Vec<KvCache<B>> {
        if self.use_mla() {
            self.mla
                .iter()
                .map(|m| {
                    KvCache::preallocated_single(1, 1, m.d_latent, capacity, dtype, device)
                })
                .collect()
        } else {
            self.kv
                .iter()
                .map(|kv| {
                    let head_dim = kv.n_embd / kv.n_head;
                    KvCache::preallocated(1, kv.n_kv_head, head_dim, capacity, dtype, device)
                })
                .collect()
        }
    }

    /// Forward pass over `input`. A single-token input is padded to two rows
    /// before the matmuls because the autotuned M=1 GEMV kernel returns wrong
    /// results on WGPU for output widths >= 32; only the first output row is
    /// returned.
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, seq_len] = input.dims();
        if seq_len == 1 {
            let input2 = Tensor::cat(vec![input.clone(), input], 1);
            let out = self.forward_impl(input2);
            let [_, _, vocab] = out.dims();
            return out.slice([0..1, 0..1, 0..vocab]);
        }
        self.forward_impl(input)
    }

    fn forward_impl(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let tok_emb = self.embed_tokens.forward(input);
        let [_, seq_len, _] = tok_emb.dims();
        let device = tok_emb.device();
        let row_pos = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device);
        let col_pos = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device);
        let group = self.kv_share_group.max(1);
        let positions: Vec<i32> = (0..seq_len as i32).collect();
        // Eviction-aligned training: sample this batch's heavy-hitter columns.
        let keep_old = self.evict_budget.map(|budget| {
            sample_keep_mask::<B>(
                seq_len,
                self.n_ctx,
                budget,
                self.evict_recent,
                self.sink_tokens(),
                &device,
            )
        });

        let mut x = tok_emb;
        let mut group_kv: Option<(Tensor<B, 4>, Tensor<B, 4>)> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            if i % group == 0 {
                let ln = block.input_layernorm.forward(x.clone());
                group_kv = Some(if self.use_mla() {
                    let latent = self.mla[self.group_of(i)].compress(ln);
                    self.mla[self.group_of(i)].expand(latent, &positions)
                } else {
                    self.kv[self.group_of(i)].forward(ln, 0)
                });
            }
            let (k, v) = group_kv.as_ref().expect("group leader computed K/V");
            let (x_new, _) = block.forward(
                x,
                k.clone(),
                v.clone(),
                0,
                row_pos.clone(),
                col_pos.clone(),
                keep_old.clone(),
                false,
            );
            x = x_new;
        }

        let x = self.norm.forward(x);
        // Qwen Weight Tying: tie_word_embeddings = true
        let weight = self.embed_tokens.weight.val().transpose().unsqueeze_dim::<3>(0);
        x.matmul(weight)
    }

    /// Forward pass with a KV cache. A single-token input is padded to two
    /// rows before the matmuls (see [`Llm::forward`]); only the first row is
    /// written to the cache and returned.
    pub fn forward_with_cache(
        &self,
        input: Tensor<B, 2, Int>,
        start_pos: usize,
        caches: &mut [KvCache<B>],
    ) -> Tensor<B, 3> {
        let [_, seq_len] = input.dims();
        if seq_len == 1 {
            let input2 = Tensor::cat(vec![input.clone(), input], 1);
            let out = self.forward_with_cache_impl(input2, start_pos, caches, 1);
            let [_, _, vocab] = out.dims();
            return out.slice([0..1, 0..1, 0..vocab]);
        }
        self.forward_with_cache_impl(input, start_pos, caches, seq_len)
    }

    fn forward_with_cache_impl(
        &self,
        input: Tensor<B, 2, Int>,
        start_pos: usize,
        caches: &mut [KvCache<B>],
        cache_rows: usize,
    ) -> Tensor<B, 3> {
        let tok_emb = self.embed_tokens.forward(input);
        let [_, seq_len, _] = tok_emb.dims();
        let device = tok_emb.device();
        let row_pos = Tensor::<B, 1, Int>::arange(
            start_pos as i64..(start_pos + seq_len) as i64,
            &device,
        );
        let group = self.kv_share_group.max(1);
        // f16 cache rows only need a cast back when compute is f32; on an f16
        // compute backend the cache already matches the compute dtype (casting
        // to f32 there would corrupt the tensor).
        let f16_compute = tok_emb.dtype() == DType::F16;

        let mut x = tok_emb;
        for (i, block) in self.blocks.iter().enumerate() {
            let g = self.group_of(i);
            if i % group == 0 {
                let ln = block.input_layernorm.forward(x.clone());
                if self.use_mla() {
                    let latent = self.mla[g].compress(ln);
                    let latent = latent.slice([
                        0..1,
                        0..1,
                        0..cache_rows,
                        0..self.mla[g].d_latent,
                    ]);
                    cache_write_latent(&mut caches[g], latent, start_pos, block.attn.sink_tokens);
                } else {
                    let kv = &self.kv[g];
                    let head_dim = kv.n_embd / kv.n_head;
                    let (k, v) = kv.forward(ln, start_pos);
                    let k = k.slice([0..1, 0..kv.n_kv_head, 0..cache_rows, 0..head_dim]);
                    let v = v.slice([0..1, 0..kv.n_kv_head, 0..cache_rows, 0..head_dim]);
                    cache_write(&mut caches[g], k, v, start_pos, block.attn.sink_tokens);
                }
            }
            let cache = &caches[g];
            let (k_full, v_full) = if self.use_mla() {
                let latent = if cache.dtype.is_quantized() {
                    dequantize_rows(
                        cache.kq.clone().expect("mla cache kq buffer"),
                        cache.ks.clone().expect("mla cache ks buffer"),
                        cache.dtype,
                    )
                    .reshape([1, 1, cache.capacity, self.mla[g].d_latent])
                } else {
                    let latent = cache.k.clone().expect("mla cache latent buffer");
                    if cache.dtype == KvCacheDtype::F16 && !f16_compute {
                        latent.cast(FloatDType::F32)
                    } else {
                        latent
                    }
                };
                let positions = cache_positions_i32(cache);
                self.mla[g].expand(latent, &positions)
            } else if cache.dtype.is_quantized() {
                let kv = &self.kv[g];
                let head_dim = kv.n_embd / kv.n_head;
                // Packed rows are slot-major with (head, dim) flattened, so
                // reshape through [capacity, head, dim] before moving the head
                // axis in front of the sequence axis.
                let k = dequantize_rows(
                    cache.kq.clone().expect("kv cache kq buffer"),
                    cache.ks.clone().expect("kv cache ks buffer"),
                    cache.dtype,
                )
                .reshape([cache.capacity, kv.n_kv_head, head_dim])
                .swap_dims(0, 1)
                .unsqueeze_dim::<4>(0);
                let v = dequantize_rows(
                    cache.vq.clone().expect("kv cache vq buffer"),
                    cache.vs.clone().expect("kv cache vs buffer"),
                    cache.dtype,
                )
                .reshape([cache.capacity, kv.n_kv_head, head_dim])
                .swap_dims(0, 1)
                .unsqueeze_dim::<4>(0);
                (k, v)
            } else {
                let k = cache.k.clone().expect("kv cache k buffer");
                let v = cache.v.clone().expect("kv cache v buffer");
                if cache.dtype == KvCacheDtype::F16 && !f16_compute {
                    (k.cast(FloatDType::F32), v.cast(FloatDType::F32))
                } else {
                    (k, v)
                }
            };
            let col_pos = cache_col_positions(cache, &device);
            let want_mass = cache.eviction_enabled();
            let (x_new, mass) = block.forward(
                x,
                k_full,
                v_full,
                start_pos,
                row_pos.clone(),
                col_pos,
                None,
                want_mass,
            );
            x = x_new;
            if let Some(mass) = mass {
                let scores = caches[g].scores.take().expect("eviction scores enabled");
                caches[g].scores = Some(scores + mass.cast(FloatDType::F32));
            }
        }

        let x = self.norm.forward(x);
        // Qwen Weight Tying: tie_word_embeddings = true
        let weight = self.embed_tokens.weight.val().transpose().unsqueeze_dim::<3>(0);
        x.matmul(weight)
    }
}

impl ModelConfig {
    pub fn init_model<B: Backend>(&self, device: &B::Device) -> Llm<B> {
        let kv_head = self.get_n_kv_head();
        let rope_kind = self.rope_kind();
        let rope_factor = self.rope_factor;
        let max_seq = extended_ctx_len(self.n_ctx, rope_kind, rope_factor);
        // Invalid eviction configs are rejected by the CLI/API entry points;
        // here they only downgrade to the dense mask.
        let eviction = self.eviction_geometry().ok().flatten();
        let blocks = (0..self.n_layer)
            .map(|_| {
                BlockConfig {
                    n_embd: self.n_embd,
                    n_head: self.n_head,
                    n_kv_head: kv_head,
                    n_ctx: self.n_ctx,
                    dropout: self.dropout,
                    attn_softcap: self.attn_softcap,
                    rope_kind,
                    rope_factor,
                    sliding_window: self.sliding_window,
                    sink_tokens: self.sink_tokens,
                    evict_budget: eviction.map(|e| e.budget),
                    evict_recent: eviction.map(|e| e.recent).unwrap_or(0),
                }
                .init(device)
            })
            .collect();
        let kv: Vec<KvProj<B>> = if self.use_mla() {
            Vec::new()
        } else {
            (0..self.n_kv_groups())
                .map(|_| {
                    KvProjConfig {
                        n_embd: self.n_embd,
                        n_head: self.n_head,
                        n_kv_head: kv_head,
                        n_ctx: self.n_ctx,
                        rope_kind,
                        rope_factor,
                    }
                    .init(device)
                })
                .collect()
        };
        let mla: Vec<MlaProj<B>> = if self.use_mla() {
            (0..self.n_kv_groups())
                .map(|_| {
                    MlaProjConfig {
                        n_embd: self.n_embd,
                        n_head: self.n_head,
                        n_kv_head: kv_head,
                        n_ctx: self.n_ctx,
                        rope_kind,
                        rope_factor,
                        d_latent: self.mla_dim(),
                    }
                    .init(device)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Qwen Normal(0, 0.02) weight initialization for tied word embeddings
        let std = 0.02f64;
        let embed_weight = Tensor::<B, 2>::random(
            [self.vocab_size, self.n_embd],
            burn::tensor::Distribution::Normal(0.0, std),
            device,
        );
        let embed_tokens = Embedding {
            weight: burn::module::Param::from_tensor(embed_weight),
        };

        Llm {
            embed_tokens,
            kv,
            mla,
            blocks,
            norm: RmsNorm::new(self.n_embd, device),
            n_ctx: self.n_ctx,
            max_seq,
            kv_share_group: self.kv_share_group.max(1),
            evict_budget: eviction.map(|e| e.budget),
            evict_recent: eviction.map(|e| e.recent).unwrap_or(0),
        }
        .align_compute_dtype()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::Wgpu;

    #[test]
    fn test_rope_spec_linear_scales_frequencies() {
        let base = rope_spec(256, 64, RopeScalingKind::None, 1.0);
        let linear = rope_spec(256, 64, RopeScalingKind::Linear, 4.0);
        assert_eq!(base.freqs.len(), linear.freqs.len());
        for (b, l) in base.freqs.iter().zip(linear.freqs.iter()) {
            assert!((b / 4.0 - l).abs() < 1e-7, "linear scaling mismatch: {b} vs {l}");
        }
        assert_eq!(linear.mscale, 1.0);
        // A factor <= 1 is a no-op.
        let noop = rope_spec(256, 64, RopeScalingKind::Linear, 1.0);
        assert_eq!(noop.freqs[1], base.freqs[1]);
    }

    #[test]
    fn test_rope_spec_ntk_lowers_frequencies() {
        let base = rope_spec(256, 64, RopeScalingKind::None, 1.0);
        let ntk = rope_spec(256, 64, RopeScalingKind::Ntk, 4.0);
        // The first frequency is 1.0 regardless of base; later ones shrink.
        assert!((base.freqs[0] - 1.0).abs() < 1e-6);
        assert!((ntk.freqs[0] - 1.0).abs() < 1e-6);
        assert!(ntk.freqs[1] < base.freqs[1]);
        assert!(ntk.freqs[ntk.freqs.len() - 2] < base.freqs[base.freqs.len() - 2]);
    }

    #[test]
    fn test_rope_spec_yarn_ramp_and_mscale() {
        let factor = 4.0;
        let half = 32; // head_dim = 64
        let original = rope_spec(256, 64, RopeScalingKind::None, 1.0);
        let yarn = rope_spec(256, 64, RopeScalingKind::Yarn, factor);

        // High frequencies (low dimensions) keep the original table; low
        // frequencies (high dimensions) are interpolated by `1/factor`.
        assert!((yarn.freqs[0] - original.freqs[0]).abs() < 1e-6);
        assert!((yarn.freqs[half - 1] - original.freqs[half - 1] / factor).abs() < 1e-6);
        assert!((yarn.mscale - (0.1 * factor.ln() + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_extended_ctx_len() {
        assert_eq!(extended_ctx_len(256, RopeScalingKind::None, 4.0), 256);
        assert_eq!(extended_ctx_len(256, RopeScalingKind::Linear, 1.0), 256);
        assert_eq!(extended_ctx_len(256, RopeScalingKind::Linear, 2.0), 512);
        assert_eq!(extended_ctx_len(256, RopeScalingKind::Yarn, 2.5), 640);
        // No table cap: angles are computed per position.
        assert_eq!(extended_ctx_len(4096, RopeScalingKind::Ntk, 100.0), 409_600);
    }

    #[test]
    fn test_rope_spec_wiring() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;

        let base = CausalSelfAttentionConfig {
            n_embd: 32,
            n_head: 4,
            n_kv_head: 4,
            n_ctx: 8,
            dropout: 0.0,
            attn_softcap: 0.0,
            rope_kind: RopeScalingKind::None,
            rope_factor: 1.0,
            sliding_window: None,
            sink_tokens: 0,
            evict_budget: None,
            evict_recent: 0,
        };
        let attn = base.init::<NdArray<f32, i32>>(&device);
        assert_eq!(attn.rope_freqs.len(), 4); // head_dim / 2
        assert_eq!(attn.rope_mscale, 1.0);

        let yarn = CausalSelfAttentionConfig {
            rope_kind: RopeScalingKind::Yarn,
            rope_factor: 2.0,
            ..base
        };
        let attn = yarn.init::<NdArray<f32, i32>>(&device);
        assert!((attn.rope_mscale - (0.1 * 2.0f32.ln() + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cached_forward_matches_full_forward() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let config = ModelConfig::new(50, 32, 4, Some(2), 2, 8, 0.0, 0.0);
        let model: Llm<NdArray<f32, i32>> = config.init_model(&device);

        let tokens = vec![1i32, 2, 3, 4, 5, 6];
        let input = Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
            TensorData::new(tokens.clone(), [1, 6]),
            &device,
        );

        // Prefill path: the cache must reproduce the uncached logits exactly.
        let full = model.forward(input.clone()).into_data().to_vec::<f32>().unwrap();
        let mut caches: Vec<KvCache<NdArray<f32, i32>>> =
            model.new_kv_caches(model.max_seq_len(), KvCacheDtype::F32, &device);
        let cached = model
            .forward_with_cache(input, 0, &mut caches)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(full.len(), cached.len());
        let max_diff = full
            .iter()
            .zip(cached.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "prefill cache diverged from the full forward pass: max_diff={max_diff}"
        );

        // Decode path: feeding one new token at start_pos=6 must match the
        // corresponding row of a single 7-token uncached forward.
        let full7 = model
            .forward(Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
                TensorData::new(vec![1i32, 2, 3, 4, 5, 6, 7], [1, 7]),
                &device,
            ))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let step = Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
            TensorData::new(vec![7i32], [1, 1]),
            &device,
        );
        let step_logits = model
            .forward_with_cache(step, 6, &mut caches)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let vocab = 50;
        let expected = &full7[6 * vocab..7 * vocab];
        let step_diff = expected
            .iter()
            .zip(step_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            step_diff < 1e-4,
            "decode cache diverged from the full forward pass: max_diff={step_diff}"
        );
        assert_eq!(caches[0].len, 7);
    }

    #[test]
    fn test_forward_with_cache_past_n_ctx_with_scaling() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.set_rope_scaling(RopeScalingKind::Linear, 2.0);
        let model: Llm<NdArray<f32, i32>> = config.init_model(&device);
        assert_eq!(model.max_seq_len(), 16);
        let mut caches: Vec<KvCache<NdArray<f32, i32>>> =
            model.new_kv_caches(model.max_seq_len(), KvCacheDtype::F32, &device);

        let prefill = Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
            TensorData::new(vec![1i32, 2, 3, 4, 5, 6], [1, 6]),
            &device,
        );
        let out = model.forward_with_cache(prefill, 0, &mut caches);
        assert_eq!(out.dims(), [1, 6, 50]);

        // start_pos 6 + 6 tokens = 12, past the trained n_ctx (8) but within
        // the extended cache (16).
        let next = Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
            TensorData::new(vec![7i32, 8, 9, 10, 11, 12], [1, 6]),
            &device,
        );
        let out = model.forward_with_cache(next, 6, &mut caches);
        assert_eq!(out.dims(), [1, 6, 50]);
        assert_eq!(caches[0].len, 12);
        assert_eq!(caches[0].capacity, 16);
        assert!(12 <= model.max_seq_len());
    }

    #[test]
    fn test_model_initialization_and_forward() {
        let device = WgpuDevice::default();
        let config = ModelConfig::new(100, 64, 4, Some(2), 2, 32, 0.0, 0.0);
        let model: Llm<Wgpu> = config.init_model(&device);

        let input_data = TensorData::new(vec![1i32, 2, 3, 4], [1, 4]);
        let input = Tensor::<Wgpu, 2, Int>::from_data(input_data, &device);

        let logits = model.forward(input);
        assert_eq!(logits.dims(), [1, 4, 100]);
    }

    #[test]
    fn test_causal_mask_isolates_position_zero() {
        let device = WgpuDevice::default();
        let config = ModelConfig::new(100, 64, 4, Some(2), 2, 32, 0.0, 0.0);
        let model: Llm<Wgpu> = config.init_model(&device);

        let input_a = TensorData::new(vec![7i32, 11, 13, 17], [1, 4]);
        let input_b = TensorData::new(vec![7i32, 99, 13, 17], [1, 4]);
        let ta = Tensor::<Wgpu, 2, Int>::from_data(input_a, &device);
        let tb = Tensor::<Wgpu, 2, Int>::from_data(input_b, &device);

        let la = model.forward(ta);
        let lb = model.forward(tb);

        let da = la.into_data().to_vec::<f32>().unwrap_or_default();
        let db = lb.into_data().to_vec::<f32>().unwrap_or_default();

        let v = 100;
        let pos0_a = &da[0..v];
        let pos0_b = &db[0..v];

        let max_diff = pos0_a
            .iter()
            .zip(pos0_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let changed_pos1 = pos0_a
            .iter()
            .zip(pos0_b.iter())
            .filter(|(a, b)| ((*a) - (*b)).abs() > 1e-5)
            .count();

        eprintln!("[CAUSALITY TEST] max_diff pos0 = {max_diff:.6}, changed_pos1 = {changed_pos1}/{v}");
        assert!(max_diff < 1e-4, "Position 0 logits changed when token at position 1 changed (max_diff={max_diff}). Causal mask broken!");
    }

    /// Runs the same token sequence through the cached decode loop and through
    /// a single full forward, comparing every row. Exercises the ring buffer,
    /// the per-position RoPE angles and the window/sink mask together.
    fn assert_cached_matches_full_with_window(
        window: Option<usize>,
        sink_tokens: usize,
        kv_share_group: usize,
        mla_dim: Option<usize>,
    ) -> (usize, usize) {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 4, 8, 0.0, 0.0);
        config.sliding_window = window;
        config.sink_tokens = sink_tokens;
        config.kv_share_group = kv_share_group;
        config.mla_latent_dim = mla_dim;
        let model: Llm<NdArray<f32, i32>> = config.init_model(&device);
        assert_eq!(model.new_kv_caches(1, KvCacheDtype::F32, &device).len(), config.n_kv_groups());

        let tokens: Vec<i32> = vec![3, 5, 7, 11, 13, 17];
        let full = model
            .forward(Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
                TensorData::new(tokens.clone(), [1, tokens.len()]),
                &device,
            ))
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        let capacity = model.kv_capacity();
        let expected_capacity = window.map(|w| w + sink_tokens).unwrap_or(model.max_seq_len());
        assert_eq!(capacity, expected_capacity);

        let mut caches = model.new_kv_caches(capacity, KvCacheDtype::F32, &device);
        for (pos, &tok) in tokens.iter().enumerate() {
            let step = Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
                TensorData::new(vec![tok], [1, 1]),
                &device,
            );
            let logits = model
                .forward_with_cache(step, pos, &mut caches)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let expected = &full[pos * 50..(pos + 1) * 50];
            let diff = expected
                .iter()
                .zip(logits.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                diff < 1e-4,
                "cached decode diverged at pos {pos} (window={window:?}, sinks={sink_tokens}): max_diff={diff}"
            );
        }
        (capacity, model.max_seq_len())
    }

    #[test]
    fn test_cached_decode_matches_full_without_window() {
        let (capacity, max_seq) = assert_cached_matches_full_with_window(None, 0, 1, None);
        assert_eq!(capacity, max_seq);
    }

    #[test]
    fn test_cached_decode_matches_full_with_ring_window() {
        let (capacity, _) = assert_cached_matches_full_with_window(Some(2), 0, 1, None);
        // Bounded cache: only `window` slots, all overwritten by the end.
        assert_eq!(capacity, 2);
    }

    #[test]
    fn test_cached_decode_matches_full_with_sinks() {
        let (capacity, _) = assert_cached_matches_full_with_window(Some(2), 1, 1, None);
        assert_eq!(capacity, 3);
    }

    #[test]
    fn test_cached_decode_matches_full_with_kv_sharing() {
        // 4 layers, 2 per group -> 2 caches; training and cached paths must
        // agree on the shared-K/V computation.
        let (capacity, _) = assert_cached_matches_full_with_window(None, 0, 2, None);
        assert_eq!(capacity, 8);
    }

    #[test]
    fn test_cached_decode_matches_full_with_mla() {
        let (capacity, _) = assert_cached_matches_full_with_window(None, 0, 1, Some(16));
        assert_eq!(capacity, 8);
    }

    #[test]
    fn test_evict_slots_protect_sinks_and_recent() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut cache =
            KvCache::<NdArray<f32, i32>>::preallocated(1, 1, 4, 8, KvCacheDtype::F32, &device);
        cache.enable_eviction(2);
        // Fill the cache: rows 0..8 are appended in order.
        let slots = alloc_evict_slots(&mut cache, 0, 8, 2);
        assert_eq!(slots, (0..8).collect::<Vec<_>>());
        assert_eq!(cache.len, 8);
        // pos 8: cutoff 6 -> victim is the lowest-position row among 2..6.
        let slots = alloc_evict_slots(&mut cache, 8, 1, 2);
        assert_eq!(slots, vec![2]);
        assert_eq!(cache.positions[2], 8);
        // pos 9: cutoff 7 -> victim among 3..7 (position 8 is recent).
        let slots = alloc_evict_slots(&mut cache, 9, 1, 2);
        assert_eq!(slots, vec![3]);
        // Sinks and recent rows survive; the live set is exactly the budget.
        assert_eq!(cache.positions[0], 0);
        assert_eq!(cache.positions[1], 1);
        assert!(cache.positions.contains(&8));
        assert!(cache.positions.contains(&9));
        assert!(cache.positions.iter().all(|p| *p != usize::MAX));
    }

    #[test]
    fn test_evict_picks_lowest_score_and_resets_it() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut cache =
            KvCache::<NdArray<f32, i32>>::preallocated(1, 1, 4, 6, KvCacheDtype::F32, &device);
        cache.enable_eviction(1);
        let _ = alloc_evict_slots(&mut cache, 0, 6, 1);
        // Position 3 is the least important; position 2 is hot.
        cache.scores = Some(Tensor::from_data(
            TensorData::new(vec![9.0, 9.0, 50.0, 0.5, 9.0, 9.0], [6]),
            &device,
        ));
        // pos 6: cutoff 5 -> candidates 1..5, lowest score is position 3.
        let slots = alloc_evict_slots(&mut cache, 6, 1, 1);
        assert_eq!(slots, vec![3]);
        let scores = cache
            .scores
            .clone()
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(scores[3], 0.0, "victim score must reset");
        assert_eq!(scores[2], 50.0, "hot rows keep their score");
    }

    #[test]
    fn test_sample_keep_mask_density_and_stochasticity() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let n_ctx = 4096;
        let sinks = 4;
        let recent = 64;
        let budget = 516; // hitters = 448, p = 448 / 4028 ~ 0.111
        let mask = sample_keep_mask::<NdArray<f32, i32>>(
            n_ctx, n_ctx, budget, recent, sinks, &device,
        );
        assert_eq!(mask.dims(), [1, n_ctx]);
        let kept = mask
            .clone()
            .into_data()
            .to_vec::<bool>()
            .unwrap()
            .iter()
            .filter(|b| **b)
            .count();
        let expected = 448.0f64;
        // Binomial sd ~ 20; allow 4 sd of slack.
        assert!(
            (kept as f64 - expected).abs() < 80.0,
            "keep-mask density off: kept {kept}, expected ~{expected}"
        );
        let other = sample_keep_mask::<NdArray<f32, i32>>(
            n_ctx, n_ctx, budget, recent, sinks, &device,
        );
        assert_ne!(
            mask.into_data().to_vec::<bool>().unwrap(),
            other.into_data().to_vec::<bool>().unwrap(),
            "each forward must resample the keep set"
        );
    }

    #[test]
    fn test_evict_cached_decode_stays_bounded_and_finite() {
        use burn_ndarray::{NdArray, NdArrayDevice};

        fn run(dtype: KvCacheDtype) {
            let device = NdArrayDevice::Cpu;
            let mut config = ModelConfig::new(50, 32, 4, Some(2), 2, 32, 0.0, 0.0);
            config.sink_tokens = 2;
            config.kv_eviction = Some("h2o".into());
            config.kv_budget = Some(8);
            config.kv_recent = 2;
            let geo = config.eviction_geometry().unwrap().unwrap();
            let model: Llm<NdArray<f32, i32>> = config.init_model(&device);
            let capacity = geo.budget.min(model.max_seq_len());
            let mut caches = model.new_kv_caches(capacity, dtype, &device);
            for c in caches.iter_mut() {
                c.enable_eviction(geo.recent);
            }
            for pos in 0..20 {
                let tok = (pos % config.vocab_size) as i32;
                let input = Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
                    TensorData::new(vec![tok], [1, 1]),
                    &device,
                );
                let logits = model.forward_with_cache(input, pos, &mut caches);
                let vals = logits.into_data().to_vec::<f32>().unwrap();
                assert!(
                    vals.iter().all(|v| v.is_finite()),
                    "{} eviction produced non-finite logits at pos {pos}",
                    dtype.as_str()
                );
            }
            for c in &caches {
                assert_eq!(c.len, capacity, "cache must stay at the budget");
                let mut alive: Vec<usize> = c
                    .positions
                    .iter()
                    .copied()
                    .filter(|p| *p != usize::MAX)
                    .collect();
                let before = alive.len();
                alive.sort_unstable();
                alive.dedup();
                assert_eq!(alive.len(), before, "slot positions must be unique");
            }
        }

        // burn-ndarray has no f16 storage, so the f16 cache path is covered by
        // the WGPU tests; f32/int8/q4 exercise every eviction writer branch.
        for dtype in [KvCacheDtype::F32, KvCacheDtype::Int8, KvCacheDtype::Q4] {
            run(dtype);
        }
    }

    #[test]
    fn test_mla_cache_is_latent_only() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 2, 8, 0.0, 0.0);
        config.mla_latent_dim = Some(16);
        let model: Llm<NdArray<f32, i32>> = config.init_model(&device);
        let caches = model.new_kv_caches(4, KvCacheDtype::F32, &device);
        assert_eq!(caches.len(), 2);
        assert_eq!(caches[0].k.as_ref().unwrap().dims(), [1, 1, 4, 16]);
        assert!(caches[0].v.is_none(), "MLA caches only the latent");
    }

    #[test]
    fn test_mla_weight_folding_matches_expansion() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.mla_latent_dim = Some(16);
        let model: Llm<NdArray<f32, i32>> = config.init_model(&device);
        let mla = &model.mla[0];

        let x = Tensor::<NdArray<f32, i32>, 2>::from_data(
            TensorData::new(
                (0..32).map(|i| (i as f32) * 0.01 - 0.16).collect::<Vec<f32>>(),
                [1, 32],
            ),
            &device,
        );
        let direct = mla
            .k_up
            .forward(mla.kv_down.forward(x.clone()))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // Export folds W_eff = kv_down.weight @ k_up.weight (Burn layout) and
        // applies it as x @ W_eff.
        let folded_w = mla.kv_down.weight.val().matmul(mla.k_up.weight.val());
        let folded = x
            .matmul(folded_w)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let diff = direct
            .iter()
            .zip(folded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-5, "folded MLA weights differ from expansion: {diff}");
    }

    #[test]
    fn test_quantized_rows_roundtrip() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let data: Vec<f32> = (0..8 * 64)
            .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.013)
            .collect();
        let x = Tensor::<NdArray<f32, i32>, 2>::from_data(TensorData::new(data, [8, 64]), &device);

        for dtype in [KvCacheDtype::Int8, KvCacheDtype::Q4] {
            let (packed, scales) = quantize_rows(x.clone(), dtype);
            assert_eq!(packed.dims(), [8, 64 / dtype.pack()]);
            let back = dequantize_rows(packed, scales.clone(), dtype);
            let xv = x.clone().into_data().to_vec::<f32>().unwrap();
            let bv = back.into_data().to_vec::<f32>().unwrap();
            let sv = scales.into_data().to_vec::<f32>().unwrap();
            let max_scale = sv.iter().cloned().fold(0.0f32, f32::max);
            let max_err = xv
                .iter()
                .zip(bv.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err <= max_scale / 2.0 + 1e-6,
                "{} roundtrip error {max_err} exceeds half the max scale {max_scale}",
                dtype.as_str()
            );
            assert!(max_err > 0.0, "quantization should not be lossless");
        }
    }

    #[test]
    fn test_zero_packed_word_dequantizes_to_zero() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        for dtype in [KvCacheDtype::Int8, KvCacheDtype::Q4] {
            let row_len = 64;
            let words = row_len / dtype.pack();
            let packed =
                Tensor::<NdArray<f32, i32>, 2, Int>::zeros([2, words], &device);
            let scales =
                Tensor::<NdArray<f32, i32>, 2>::ones([2, row_len / quant_block(row_len)], &device);
            let back = dequantize_rows(packed, scales, dtype);
            let values = back.into_data().to_vec::<f32>().unwrap();
            assert!(
                values.iter().all(|v| *v == 0.0),
                "empty {} slots must dequantize to exactly zero",
                dtype.as_str()
            );
        }
    }

    #[test]
    fn test_quantized_cache_close_to_f32_cache() {
        // Same decode path, only the cache storage changes. Quantization error
        // is bounded per block; the tolerances cover the accumulated effect
        // over five decode steps.
        use crate::backend::WgpuHalf;

        fn check<B: Backend>(
            device: &B::Device,
            config: ModelConfig,
            dtype: KvCacheDtype,
            tol: f32,
        ) {
            let model: Llm<B> = config.init_model(device);
            let tokens: Vec<i32> = vec![3, 5, 7, 11, 13];

            let run = |dt: KvCacheDtype| -> Vec<Vec<f32>> {
                let mut caches = model.new_kv_caches(8, dt, device);
                tokens
                    .iter()
                    .enumerate()
                    .map(|(pos, &tok)| {
                        let step = Tensor::<B, 2, Int>::from_data(
                            TensorData::new(vec![tok], [1, 1]),
                            device,
                        );
                        model
                            .forward_with_cache(step, pos, &mut caches)
                            .into_data()
                            .to_vec::<f32>()
                            .unwrap()
                    })
                    .collect()
            };
            let reference = run(KvCacheDtype::F32);
            let quantized = run(dtype);
            for pos in 0..tokens.len() {
                let diff = reference[pos]
                    .iter()
                    .zip(quantized[pos].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    diff < tol,
                    "{} cache diverged from f32 at pos {pos}: max_diff={diff}",
                    dtype.as_str()
                );
            }
        }

        let mut mla = ModelConfig::new(50, 64, 4, Some(2), 2, 16, 0.0, 0.0);
        mla.mla_latent_dim = Some(32);
        let device = WgpuDevice::default();
        for (dtype, tol) in [(KvCacheDtype::Int8, 5e-2), (KvCacheDtype::Q4, 2e-1)] {
            check::<Wgpu>(
                &device,
                ModelConfig::new(50, 64, 4, Some(2), 2, 16, 0.0, 0.0),
                dtype,
                tol,
            );
            check::<WgpuHalf<f32>>(
                &device,
                ModelConfig::new(50, 64, 4, Some(2), 2, 16, 0.0, 0.0),
                dtype,
                tol,
            );
            check::<WgpuHalf<f32>>(&device, mla.clone(), dtype, tol);
        }
    }

    #[test]
    fn test_f16_cache_close_to_f32_cache() {
        // Same decode path, only the cache dtype changes: isolates f16 rounding
        // from any backend-level differences between batched and single-token
        // forwards.
        use crate::backend::WgpuHalf;

        fn check<B: Backend>(device: &B::Device) {
            let config = ModelConfig::new(50, 32, 4, Some(2), 2, 8, 0.0, 0.0);
            let model: Llm<B> = config.init_model(device);
            let tokens: Vec<i32> = vec![3, 5, 7, 11, 13];

            let run = |f16: bool| -> Vec<Vec<f32>> {
                let mut caches = model.new_kv_caches(8, if f16 { KvCacheDtype::F16 } else { KvCacheDtype::F32 }, device);
                tokens
                    .iter()
                    .enumerate()
                    .map(|(pos, &tok)| {
                        let step = Tensor::<B, 2, Int>::from_data(
                            TensorData::new(vec![tok], [1, 1]),
                            device,
                        );
                        model
                            .forward_with_cache(step, pos, &mut caches)
                            .into_data()
                            .to_vec::<f32>()
                            .unwrap()
                    })
                    .collect()
            };
            let f32_logits = run(false);
            let f16_logits = run(true);
            for pos in 0..tokens.len() {
                let diff = f32_logits[pos]
                    .iter()
                    .zip(f16_logits[pos].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    diff < 5e-2,
                    "f16 cache diverged from f32 cache at pos {pos}: max_diff={diff}"
                );
            }
        }

        let device = WgpuDevice::default();
        check::<WgpuHalf<f32>>(&device);
    }

    #[test]
    fn test_f16_cache_really_stores_f16() {
        use crate::backend::WgpuHalf;
        let device = WgpuDevice::default();
        let config = ModelConfig::new(50, 32, 4, Some(2), 2, 8, 0.0, 0.0);
        let model: Llm<WgpuHalf<f32>> = config.init_model(&device);
        let caches = model.new_kv_caches(8, KvCacheDtype::F16, &device);
        assert_eq!(caches[0].k.as_ref().unwrap().dtype(), DType::F16);
        assert_eq!(caches[0].v.as_ref().unwrap().dtype(), DType::F16);
    }

    #[test]
    #[should_panic(expected = "f16 casts are")]
    fn test_fused_backend_rejects_f16_cache() {
        let device = WgpuDevice::default();
        let _ = KvCache::<Wgpu>::preallocated(1, 2, 8, 4, KvCacheDtype::F16, &device);
    }

    #[test]
    fn test_kv_sharing_reduces_cache_count() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 4, 8, 0.0, 0.0);
        config.kv_share_group = 4;
        let model: Llm<NdArray<f32, i32>> = config.init_model(&device);
        assert_eq!(model.new_kv_caches(8, KvCacheDtype::F32, &device).len(), 1);
        // The grouped model is a different function than the ungrouped one.
        let input = Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
            TensorData::new(vec![1i32, 2, 3, 4], [1, 4]),
            &device,
        );
        let grouped = model.forward(input.clone()).into_data().to_vec::<f32>().unwrap();
        config.kv_share_group = 1;
        let plain: Llm<NdArray<f32, i32>> = config.init_model(&device);
        let plain_out = plain.forward(input).into_data().to_vec::<f32>().unwrap();
        let diff = grouped
            .iter()
            .zip(plain_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff > 1e-4, "grouped KV must change the computation (diff={diff})");
    }

    #[test]
    fn test_ring_cache_keeps_only_window_positions() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.sliding_window = Some(2);
        let model: Llm<NdArray<f32, i32>> = config.init_model(&device);
        let mut caches = model.new_kv_caches(model.kv_capacity(), KvCacheDtype::F32, &device);

        for pos in 0..6usize {
            let step = Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
                TensorData::new(vec![pos as i32 + 1], [1, 1]),
                &device,
            );
            model.forward_with_cache(step, pos, &mut caches);
        }

        let cache = &caches[0];
        assert_eq!(cache.len, 2);
        let mut stored: Vec<usize> = cache
            .positions
            .iter()
            .copied()
            .filter(|&p| p != usize::MAX)
            .collect();
        stored.sort_unstable();
        assert_eq!(stored, vec![4, 5]);
    }

    #[test]
    fn test_sliding_window_limits_attention_span() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.sliding_window = Some(2);
        let model: Llm<NdArray<f32, i32>> = config.init_model(&device);

        let run = |first: i32| -> Vec<f32> {
            model
                .forward(Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
                    TensorData::new(vec![first, 11, 13, 17], [1, 4]),
                    &device,
                ))
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        let a = run(7);
        let b = run(49);

        let row_diff = |row: usize| -> f32 {
            a[row * 50..(row + 1) * 50]
                .iter()
                .zip(b[row * 50..(row + 1) * 50].iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        // Row 1 sees position 0 (distance 1), row 3 does not (distance 3 > 2).
        assert!(row_diff(1) > 1e-4, "window should include the previous token");
        assert!(row_diff(3) < 1e-4, "window must drop positions older than 2");
    }

    #[test]
    fn test_sink_tokens_keep_first_position_visible() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.sliding_window = Some(2);
        config.sink_tokens = 1;
        let model: Llm<NdArray<f32, i32>> = config.init_model(&device);

        let run = |first: i32| -> Vec<f32> {
            model
                .forward(Tensor::<NdArray<f32, i32>, 2, Int>::from_data(
                    TensorData::new(vec![first, 11, 13, 17], [1, 4]),
                    &device,
                ))
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        let a = run(7);
        let b = run(49);
        let diff = a[3 * 50..4 * 50]
            .iter()
            .zip(b[3 * 50..4 * 50].iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff > 1e-4,
            "sink token at position 0 must stay visible at row 3 (diff={diff})"
        );
    }
    /// Regression test for the burn-wgpu `autotune` GEMV bug: with autotune
    /// enabled, the M=1 matmul path (token-by-token generation) picked a GEMV
    /// kernel that returned wrong values for output widths >= 32, so cached
    /// decoding diverged from the batched forward pass. The non-autotuned
    /// fallback kernels are exact; this test fails if autotune is re-enabled
    /// or the fallback regresses.
    #[test]
    fn test_wgpu_cached_decode_matches_full_forward() {
        use crate::backend::WgpuHalf;

        fn check<B: Backend>(device: &B::Device) {
            let config = ModelConfig::new(50, 32, 4, Some(2), 2, 8, 0.0, 0.0);
            let model: Llm<B> = config.init_model(device);
            let input = Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![1i32, 2, 3, 4, 5, 6], [1, 6]),
                device,
            );
            let full = model
                .forward(input.clone())
                .into_data()
                .to_vec::<f32>()
                .unwrap();

            let mut caches = model.new_kv_caches(model.max_seq_len(), KvCacheDtype::F32, device);
            let cached = model
                .forward_with_cache(input, 0, &mut caches)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let prefill_diff = full
                .iter()
                .zip(cached.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                prefill_diff < 1e-4,
                "prefill cache diverged from the full forward: max_diff={prefill_diff}"
            );

            // One decode step must match the last row of a 7-token forward.
            let full7 = model
                .forward(Tensor::<B, 2, Int>::from_data(
                    TensorData::new(vec![1i32, 2, 3, 4, 5, 6, 7], [1, 7]),
                    device,
                ))
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let step =
                Tensor::<B, 2, Int>::from_data(TensorData::new(vec![7i32], [1, 1]), device);
            let step_logits = model
                .forward_with_cache(step, 6, &mut caches)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let vocab = 50;
            let step_diff = full7[6 * vocab..7 * vocab]
                .iter()
                .zip(step_logits.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                step_diff < 1e-4,
                "single-token decode diverged from the full forward: max_diff={step_diff}"
            );
        }

        let device = WgpuDevice::default();
        check::<Wgpu>(&device);
        check::<WgpuHalf<f32>>(&device);
    }
}
