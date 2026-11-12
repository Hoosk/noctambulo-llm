//! KV cache quantization formats (int8 / q4) and block packing.

use burn::tensor::{DType, Device, FloatDType, Int, Tensor};

/// KV cache storage format. `Int8` and `Q4` keep the cache quantized with
/// symmetric per-row (per token, all heads) block quantization: one f32 scale
/// per `QUANT_BLOCK` elements and packed integer data (4 int8 or 8 int4 per
/// i32 word). Burn's native `QFloat` cannot be written with `slice_assign`,
/// so the ring buffer stores the packed words directly and dequantizes on read.
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
    pub(crate) fn pack(&self) -> usize {
        match self {
            Self::Int8 => 4,
            Self::Q4 => 8,
            _ => 1,
        }
    }

    /// Bits per packed value.
    pub(crate) fn bits(&self) -> usize {
        match self {
            Self::Int8 => 8,
            Self::Q4 => 4,
            _ => 32,
        }
    }

    /// Largest magnitude representable by the quantizer.
    pub(crate) fn levels(&self) -> f32 {
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
pub(crate) fn quant_block(row_len: usize) -> usize {
    if row_len.is_multiple_of(QUANT_BLOCK) {
        QUANT_BLOCK
    } else {
        row_len
    }
}

/// Float storage dtype used by a cache format.
pub(crate) fn cache_storage_dtype(dtype: KvCacheDtype) -> DType {
    match dtype {
        KvCacheDtype::F16 => DType::F16,
        _ => DType::F32,
    }
}

/// Quantize rows `[rows, n]` into packed i32 words `[rows, n / pack]` plus one
/// f32 scale per block `[rows, n / block]`.
pub(crate) fn quantize_rows(
    x: Tensor<2>,
    dtype: KvCacheDtype,
) -> (Tensor<2, Int>, Tensor<2>) {
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

    // Quantization math is always f32: scales stay f32 regardless of the
    // compute dtype, and the dequantized rows are cast to the compute dtype on
    // read.
    let x = x.cast(FloatDType::F32);
    let xb = x.reshape([rows, n / block, block]);
    let scale = (xb.clone().abs().max_dim(2) / levels).clamp(1e-8, 1e30);
    let q = (xb / scale.clone())
        .round()
        .clamp(-levels, levels)
        .int()
        .reshape([rows, n / pack, pack]);

    // Pack the signed quantized values into bytes with `(q << shift) & mask`
    // instead of shifting the unsigned offset value: the shifted intermediate
    // always fits in i32 (unlike `offset << 24`), which the CPU backend's checked
    // i64 -> i32 conversion requires. The resulting bit pattern is identical to
    // two's-complement `(q + offset)` bytes.
    let mut packed: Option<Tensor<2, Int>> = None;
    for i in 0..pack {
        let part = q
            .clone()
            .slice([0..rows, 0..(n / pack), i..(i + 1)])
            .reshape([rows, n / pack]);
        let shifted = part.bitwise_left_shift_scalar((dtype.bits() * i) as i32);
        let mask = ((1u64 << dtype.bits()) - 1) as u32;
        let masked = shifted.bitwise_and_scalar((mask << (dtype.bits() * i)) as i32);
        packed = Some(match packed {
            None => masked,
            Some(acc) => acc.bitwise_or(masked),
        });
    }
    (packed.expect("packed words"), scale.reshape([rows, n / block]))
}

/// Inverse of [`quantize_rows`]: unpack `[rows, n / pack]` i32 words plus
/// per-block scales back to f32 `[rows, n]`.
pub(crate) fn dequantize_rows(
    packed: Tensor<2, Int>,
    scales: Tensor<2>,
    dtype: KvCacheDtype,
) -> Tensor<2> {
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

    let mut parts: Vec<Tensor<2, Int>> = Vec::with_capacity(pack);
    for i in 0..pack {
        let part = packed
            .clone()
            .bitwise_right_shift_scalar((bits * i) as i32)
            .bitwise_and_scalar(mask);
        parts.push(part);
    }
    let u = Tensor::<2, Int>::stack::<3>(parts, 2).reshape([rows, n]);
    let q = (u.bitwise_xor_scalar(sign_bit) - sign_bit)
        .float()
        .cast(FloatDType::F32);
    let scale = scales
        .reshape([rows, n / block, 1])
        .expand([rows, n / block, block])
        .reshape([rows, n]);
    q * scale
}

/// Fails early when the device cannot store f16 tensors.
pub(crate) fn assert_f16_storage(device: &Device) {
    assert!(
        device.supports_dtype(DType::F16),
        "this device does not support f16 storage; use an f32 KV cache"
    );
}
