use crate::config::{ModelConfig, RopeScalingKind};
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Embedding, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

#[derive(Clone, Debug)]
pub struct KvCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn empty() -> Self {
        Self::new()
    }
}

impl<B: Backend> Default for KvCache<B> {
    fn default() -> Self {
        Self::new()
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

/// Upper bound for the RoPE cache when context scaling is enabled; keeps the
/// per-layer cache tensors bounded for very large factors.
pub const MAX_EXTENDED_CTX: usize = 16384;

/// Effective number of positions covered by the RoPE cache. Scaling with
/// `factor > 1` extends it up to [`MAX_EXTENDED_CTX`].
pub fn extended_ctx_len(n_ctx: usize, kind: RopeScalingKind, factor: f32) -> usize {
    if kind == RopeScalingKind::None || factor <= 1.0 || !factor.is_finite() {
        return n_ctx;
    }
    ((n_ctx as f32 * factor).ceil() as usize).clamp(n_ctx, MAX_EXTENDED_CTX)
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

pub fn precompute_rope_cache<B: Backend>(
    n_ctx: usize,
    head_dim: usize,
    spec: &RopeSpec,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half_dim = head_dim / 2;
    debug_assert_eq!(spec.freqs.len(), half_dim);

    let mut angles_vec = Vec::with_capacity(n_ctx * half_dim);
    for pos in 0..n_ctx {
        for &f in &spec.freqs {
            angles_vec.push(pos as f32 * f);
        }
    }

    let angles_data = TensorData::new(angles_vec, [1, 1, n_ctx, half_dim]);
    let angles = Tensor::<B, 4>::from_data(angles_data, device);
    let cos = angles.clone().cos() * spec.mscale;
    let sin = angles.sin() * spec.mscale;

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

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub attn_dropout: Dropout,
    pub resid_dropout: Dropout,
    pub cos: Tensor<B, 4>,
    pub sin: Tensor<B, 4>,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub attn_softcap: f32,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, n_embd] = x.dims();
        let head_size = n_embd / self.n_head;
        let n_rep = self.n_head / self.n_kv_head;

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q
            .reshape([batch_size, seq_len, self.n_head, head_size])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_kv_head, head_size])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_kv_head, head_size])
            .swap_dims(1, 2);

        let cos = self.cos.clone().slice([0..1, 0..1, 0..seq_len, 0..head_size]);
        let sin = self.sin.clone().slice([0..1, 0..1, 0..seq_len, 0..head_size]);

        let q_rot = rotate_half(q.clone());
        let q_embed = (q * cos.clone()) + (q_rot * sin.clone());

        let k_rot = rotate_half(k.clone());
        let k_embed = (k * cos) + (k_rot * sin);

        let k_repeat = repeat_kv(k_embed, n_rep);
        let v_repeat = repeat_kv(v, n_rep);

        let k_t = k_repeat.swap_dims(2, 3);
        let scale = 1.0 / (head_size as f64).sqrt();
        let mut scores = q_embed.matmul(k_t) * scale;

        if self.attn_softcap > 0.0 {
            let cap = self.attn_softcap as f64;
            scores = (scores / cap).tanh() * cap;
        }

        let device = scores.device();
        let causal_mask = Tensor::<B, 2, burn::tensor::Bool>::tril_mask([seq_len, seq_len], 0, &device)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0);

        scores = scores.mask_fill(causal_mask, -1e4);

        let attn_probs = burn::tensor::activation::softmax(scores, 3);
        let attn_probs = self.attn_dropout.forward(attn_probs);
        let out = attn_probs.matmul(v_repeat);

        let out = out.swap_dims(1, 2).reshape([batch_size, seq_len, n_embd]);
        let out = self.out_proj.forward(out);
        self.resid_dropout.forward(out)
    }

    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        start_pos: usize,
        cache: &mut KvCache<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, new_seq_len, n_embd] = x.dims();
        let head_size = n_embd / self.n_head;
        let n_rep = self.n_head / self.n_kv_head;

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q
            .reshape([batch_size, new_seq_len, self.n_head, head_size])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, new_seq_len, self.n_kv_head, head_size])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, new_seq_len, self.n_kv_head, head_size])
            .swap_dims(1, 2);

        // Within the cache, slice the precomputed (possibly context-extended)
        // table. Generation paths enforce the limit with a clear error.
        let cached_len = self.cos.dims()[2];
        assert!(
            start_pos + new_seq_len <= cached_len,
            "context window exhausted: asked for positions {start_pos}..{} but the RoPE cache holds {cached_len} (increase --rope-factor)",
            start_pos + new_seq_len
        );
        let cos = self
            .cos
            .clone()
            .slice([0..1, 0..1, start_pos..(start_pos + new_seq_len), 0..head_size]);
        let sin = self
            .sin
            .clone()
            .slice([0..1, 0..1, start_pos..(start_pos + new_seq_len), 0..head_size]);
        let q_rot = rotate_half(q.clone());
        let q_embed = (q * cos.clone()) + (q_rot * sin.clone());

        let k_rot = rotate_half(k.clone());
        let k_embed = (k * cos) + (k_rot * sin);

        let (k_full, v_full) = match (&cache.k, &cache.v) {
            (Some(prev_k), Some(prev_v)) => {
                let new_k = Tensor::cat(vec![prev_k.clone(), k_embed], 2);
                let new_v = Tensor::cat(vec![prev_v.clone(), v], 2);
                (new_k, new_v)
            }
            _ => (k_embed, v),
        };

        cache.k = Some(k_full.clone());
        cache.v = Some(v_full.clone());

        let k_repeat = repeat_kv(k_full, n_rep);
        let v_repeat = repeat_kv(v_full, n_rep);

        let full_len = k_repeat.dims()[2];
        let k_t = k_repeat.swap_dims(2, 3);
        let scale = 1.0 / (head_size as f64).sqrt();
        let mut scores = q_embed.matmul(k_t) * scale;

        if self.attn_softcap > 0.0 {
            let cap = self.attn_softcap as f64;
            scores = (scores / cap).tanh() * cap;
        }

        if new_seq_len > 1 {
            let device = scores.device();
            let row_idx = Tensor::<B, 1, Int>::arange(start_pos as i64..(start_pos + new_seq_len) as i64, &device);
            let col_idx = Tensor::<B, 1, Int>::arange(0..full_len as i64, &device);
            let upper_mask = col_idx
                .unsqueeze_dim::<2>(0)
                .greater(row_idx.unsqueeze_dim::<2>(1))
                .unsqueeze_dim::<3>(0)
                .unsqueeze_dim::<4>(0);
            scores = scores.mask_fill(upper_mask, -1e4);
        }

        let attn_probs = burn::tensor::activation::softmax(scores, 3);
        let attn_probs = self.attn_dropout.forward(attn_probs);
        let out = attn_probs.matmul(v_repeat);

        let out = out.swap_dims(1, 2).reshape([batch_size, new_seq_len, n_embd]);
        let out = self.out_proj.forward(out);
        self.resid_dropout.forward(out)
    }
}

fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..s, half..d]);
    Tensor::cat(vec![x2.neg(), x1], 3)
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
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        let head_dim = self.n_embd / self.n_head;
        let kv_dim = self.n_kv_head * head_dim;
        let rope = rope_spec(self.n_ctx, head_dim, self.rope_kind, self.rope_factor);
        let cache_len = extended_ctx_len(self.n_ctx, self.rope_kind, self.rope_factor);
        let (cos, sin) = precompute_rope_cache(cache_len, head_dim, &rope, device);

        CausalSelfAttention {
            q_proj: LinearConfig::new(self.n_embd, self.n_embd).with_bias(true).init(device),
            k_proj: LinearConfig::new(self.n_embd, kv_dim).with_bias(true).init(device),
            v_proj: LinearConfig::new(self.n_embd, kv_dim).with_bias(true).init(device),
            out_proj: LinearConfig::new(self.n_embd, self.n_embd).with_bias(false).init(device),
            attn_dropout: DropoutConfig::new(self.dropout).init(),
            resid_dropout: DropoutConfig::new(self.dropout).init(),
            cos,
            sin,
            n_head: self.n_head,
            n_kv_head: self.n_kv_head,
            n_embd: self.n_embd,
            attn_softcap: self.attn_softcap,
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
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.input_layernorm.forward(x));
        x.clone() + self.mlp.forward(self.post_attention_layernorm.forward(x))
    }

    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        start_pos: usize,
        cache: &mut KvCache<B>,
    ) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward_with_cache(self.input_layernorm.forward(x), start_pos, cache);
        x.clone() + self.mlp.forward(self.post_attention_layernorm.forward(x))
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
    pub blocks: Vec<Block<B>>,
    pub norm: RmsNorm<B>,
    pub n_ctx: usize,
}

impl<B: Backend> Llm<B> {
    /// Total number of positions the model can attend to (the config `n_ctx`
    /// extended by the active RoPE scaling factor).
    pub fn max_seq_len(&self) -> usize {
        self.blocks
            .first()
            .map(|b| b.attn.cos.dims()[2])
            .unwrap_or(self.n_ctx)
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let tok_emb = self.embed_tokens.forward(input);

        let mut x = tok_emb;
        for block in &self.blocks {
            x = block.forward(x);
        }

        let x = self.norm.forward(x);
        // Qwen Weight Tying: tie_word_embeddings = true
        let weight = self.embed_tokens.weight.val().transpose().unsqueeze_dim::<3>(0);
        x.matmul(weight)
    }

    pub fn forward_with_cache(
        &self,
        input: Tensor<B, 2, Int>,
        start_pos: usize,
        caches: &mut [KvCache<B>],
    ) -> Tensor<B, 3> {
        let tok_emb = self.embed_tokens.forward(input);

        let mut x = tok_emb;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_cache(x, start_pos, &mut caches[i]);
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
                }
                .init(device)
            })
            .collect();

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
            blocks,
            norm: RmsNorm::new(self.n_embd, device),
            n_ctx: self.n_ctx,
        }
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
        // Huge factors are clamped to the cache cap.
        assert_eq!(extended_ctx_len(4096, RopeScalingKind::Ntk, 100.0), MAX_EXTENDED_CTX);
    }

    #[test]
    fn test_attention_cache_extends_with_scaling() {
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
        };
        let attn = base.init::<NdArray<f32, i32>>(&device);
        assert_eq!(attn.cos.dims()[2], 8);

        let scaled = CausalSelfAttentionConfig {
            rope_kind: RopeScalingKind::Linear,
            rope_factor: 2.0,
            ..base
        };
        let attn = scaled.init::<NdArray<f32, i32>>(&device);
        assert_eq!(attn.cos.dims()[2], 16);
    }

    #[test]
    fn test_forward_with_cache_past_n_ctx_with_scaling() {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::Cpu;
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.set_rope_scaling(RopeScalingKind::Linear, 2.0);
        let model: Llm<NdArray<f32, i32>> = config.init_model(&device);
        assert_eq!(model.max_seq_len(), 16);
        let mut caches: Vec<KvCache<NdArray<f32, i32>>> = vec![KvCache::new()];

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
        assert_eq!(caches[0].k.as_ref().unwrap().dims()[2], 12);
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
}

