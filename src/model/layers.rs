//! Transformer building blocks: RMSNorm, attention, K/V & MLA projections, MLP.

use burn::module::{Module, Param, ParamId};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::{Bool, DType, Device, Int, Tensor};

use crate::config::RopeScalingKind;

use super::rope::{rope_cos_sin, rope_cos_sin_at, rope_spec};

#[derive(Module, Debug)]
pub struct RmsNorm {
    pub weight: burn::module::Param<Tensor<1>>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f32, device: &Device) -> Self {
        let mut weight = Tensor::ones([dim], (device, DType::F32));
        if device.is_autodiff() {
            weight = weight.require_grad();
        }
        Self {
            weight: Param::initialized(ParamId::new(), weight),
            eps,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(D - 1);
        let rms = (variance + self.eps).sqrt();
        let norm = x / rms;
        let mut shape = [1; D];
        shape[D - 1] = self.weight.dims()[0];
        let weight_broadcast = self.weight.val().reshape(shape);
        norm * weight_broadcast
    }
}

fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        x
    } else {
        let [b, n_kv_head, s, d] = x.dims();
        x.unsqueeze_dim::<5>(2)
            .expand([b, n_kv_head, n_rep, s, d])
            .reshape([b, n_kv_head * n_rep, s, d])
    }
}

fn rotate_half(x: Tensor<4>) -> Tensor<4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..s, half..d]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

#[derive(Module, Debug)]
pub struct CausalSelfAttention {
    pub q_proj: Linear,
    pub out_proj: Linear,
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

impl CausalSelfAttention {
    /// Allowed-attention mask over absolute row and column positions: causal,
    /// plus the bounded-cache geometries when configured (sliding window +
    /// sinks, or the eviction keep-mask used during training). Columns with
    /// the `i32::MAX` sentinel (empty ring slots) are always masked out.
    fn allowed_mask(
        &self,
        row_pos: Tensor<1, Int>,
        col_pos: Tensor<1, Int>,
        keep_old: Option<&Tensor<2, Bool>>,
    ) -> Tensor<2, Bool> {
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
    pub fn query(&self, x: Tensor<3>, start_pos: usize) -> Tensor<4> {
        let [batch_size, seq_len, n_embd] = x.dims();
        let head_size = n_embd / self.n_head;
        let q = self
            .q_proj
            .forward(x)
            .reshape([batch_size, seq_len, self.n_head, head_size])
            .swap_dims(1, 2);
        let device = q.device();
        let (cos, sin) =
            rope_cos_sin(&self.rope_freqs, self.rope_mscale, start_pos, seq_len, &device);
        let cos = cos.cast(q.dtype());
        let sin = sin.cast(q.dtype());
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
        q_embed: Tensor<4>,
        k_full: Tensor<4>,
        v_full: Tensor<4>,
        row_pos: Tensor<1, Int>,
        col_pos: Tensor<1, Int>,
        keep_old: Option<Tensor<2, Bool>>,
        collect_mass: bool,
    ) -> (Tensor<3>, Option<Tensor<1>>) {
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
pub struct KvProj {
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub rope_freqs: Vec<f32>,
    pub rope_mscale: f32,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
}

impl KvProj {
    /// Project and RoPE-rotate K (V stays unrotated) for the absolute
    /// positions `start_pos..start_pos + seq_len`.
    pub fn forward(&self, x: Tensor<3>, start_pos: usize) -> (Tensor<4>, Tensor<4>) {
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
            rope_cos_sin(&self.rope_freqs, self.rope_mscale, start_pos, seq_len, &device);
        let cos = cos.cast(k.dtype());
        let sin = sin.cast(k.dtype());
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
pub struct MlaProj {
    pub kv_down: Linear,
    pub k_up: Linear,
    pub v_up: Linear,
    pub rope_freqs: Vec<f32>,
    pub rope_mscale: f32,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub d_latent: usize,
}

impl MlaProj {
    /// Compress `x` into the latent that is written to the cache:
    /// `[batch, 1, seq, d_latent]`.
    pub fn compress(&self, x: Tensor<3>) -> Tensor<4> {
        let [batch, seq, _] = x.dims();
        self.kv_down
            .forward(x)
            .reshape([batch, seq, 1, self.d_latent])
            .swap_dims(1, 2)
    }

    /// Expand cached latents `[batch, 1, n, d_latent]` into RoPE-rotated K and
    /// V `[batch, n_kv_head, n, head_dim]` at the given absolute positions.
    pub fn expand(&self, latents: Tensor<4>, positions: &[i32]) -> (Tensor<4>, Tensor<4>) {
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

        let (cos, sin) = rope_cos_sin_at(&self.rope_freqs, self.rope_mscale, positions, &device);
        let cos = cos.cast(k.dtype());
        let sin = sin.cast(k.dtype());
        let k_rot = rotate_half(k.clone());
        ((k * cos) + (k_rot * sin), v)
    }
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
    /// RoPE theta (10000 by default; imported models may differ).
    pub rope_base: f32,
    pub sliding_window: Option<usize>,
    pub sink_tokens: usize,
    /// Heavy-hitter eviction budget in rows (`None` = dense/ring mask).
    pub evict_budget: Option<usize>,
    /// Recent tokens protected from eviction (training mask).
    pub evict_recent: usize,
}

impl CausalSelfAttentionConfig {
    pub fn init(&self, device: &Device) -> CausalSelfAttention {
        let head_dim = self.n_embd / self.n_head;
        let rope = rope_spec(
            self.n_ctx,
            head_dim,
            self.rope_kind,
            self.rope_factor,
            self.rope_base,
        );

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
    pub rope_base: f32,
}

#[derive(Clone, Debug)]
pub struct MlaProjConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_ctx: usize,
    pub rope_kind: RopeScalingKind,
    pub rope_factor: f32,
    pub rope_base: f32,
    pub d_latent: usize,
}

impl MlaProjConfig {
    pub fn init(&self, device: &Device) -> MlaProj {
        let head_dim = self.n_embd / self.n_head;
        let kv_dim = self.n_kv_head * head_dim;
        let rope = rope_spec(
            self.n_ctx,
            head_dim,
            self.rope_kind,
            self.rope_factor,
            self.rope_base,
        );
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
    pub fn init(&self, device: &Device) -> KvProj {
        let head_dim = self.n_embd / self.n_head;
        let kv_dim = self.n_kv_head * head_dim;
        let rope = rope_spec(
            self.n_ctx,
            head_dim,
            self.rope_kind,
            self.rope_factor,
            self.rope_base,
        );
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
pub struct SwiGluMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    pub mlp_dropout: Dropout,
}

impl SwiGluMlp {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = burn::tensor::activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        let out = self.down_proj.forward(gate * up);
        self.mlp_dropout.forward(out)
    }
}

#[derive(Clone, Debug)]
pub struct SwiGluMlpConfig {
    pub n_embd: usize,
    /// FFN hidden size (0 = engine default aligned to 256).
    pub hidden: usize,
    pub dropout: f64,
}

impl SwiGluMlpConfig {
    pub fn init(&self, device: &Device) -> SwiGluMlp {
        let hidden_dim = if self.hidden > 0 {
            self.hidden
        } else {
            ((8 * self.n_embd) / 3).div_ceil(256) * 256
        };

        SwiGluMlp {
            gate_proj: LinearConfig::new(self.n_embd, hidden_dim).with_bias(false).init(device),
            up_proj: LinearConfig::new(self.n_embd, hidden_dim).with_bias(false).init(device),
            down_proj: LinearConfig::new(hidden_dim, self.n_embd).with_bias(false).init(device),
            mlp_dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Block {
    pub attn: CausalSelfAttention,
    pub mlp: SwiGluMlp,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl Block {
    /// Runs the block with K/V rows already projected by the group leader and
    /// their absolute column positions. `keep_old` is the training-time
    /// eviction keep-mask and `collect_mass` requests the per-column attention
    /// mass used by heavy-hitter eviction during cached decoding.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: Tensor<3>,
        k_full: Tensor<4>,
        v_full: Tensor<4>,
        start_pos: usize,
        row_pos: Tensor<1, Int>,
        col_pos: Tensor<1, Int>,
        keep_old: Option<Tensor<2, Bool>>,
        collect_mass: bool,
    ) -> (Tensor<3>, Option<Tensor<1>>) {
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
    pub rope_base: f32,
    pub sliding_window: Option<usize>,
    pub sink_tokens: usize,
    pub evict_budget: Option<usize>,
    pub evict_recent: usize,
    /// FFN hidden size (0 = engine default aligned to 256).
    pub hidden: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl BlockConfig {
    pub fn init(&self, device: &Device) -> Block {
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
                rope_base: self.rope_base,
                sliding_window: self.sliding_window,
                sink_tokens: self.sink_tokens,
                evict_budget: self.evict_budget,
                evict_recent: self.evict_recent,
            }
            .init(device),
            mlp: SwiGluMlpConfig {
                n_embd: self.n_embd,
                hidden: self.hidden,
                dropout: self.dropout,
            }
            .init(device),
            input_layernorm: RmsNorm::new(self.n_embd, self.rms_norm_eps, device),
            post_attention_layernorm: RmsNorm::new(self.n_embd, self.rms_norm_eps, device),
        }
    }
}
