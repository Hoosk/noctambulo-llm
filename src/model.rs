use crate::config::ModelConfig;
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

pub fn precompute_rope_cache<B: Backend>(n_ctx: usize, head_dim: usize, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half_dim = head_dim / 2;
    let rope_theta = 10000.0f32; // Standard RoPE base (optimal for n_ctx <= 8K)
    let mut freqs = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let exponent = (2 * i) as f32 / head_dim as f32;
        let theta = 1.0 / rope_theta.powf(exponent);
        freqs.push(theta);
    }

    let mut angles_vec = Vec::with_capacity(n_ctx * half_dim);
    for pos in 0..n_ctx {
        for &f in &freqs {
            angles_vec.push(pos as f32 * f);
        }
    }

    let angles_data = TensorData::new(angles_vec, [1, 1, n_ctx, half_dim]);
    let angles = Tensor::<B, 4>::from_data(angles_data, device);
    let cos = angles.clone().cos();
    let sin = angles.sin();

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
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        let head_dim = self.n_embd / self.n_head;
        let kv_dim = self.n_kv_head * head_dim;
        let (cos, sin) = precompute_rope_cache(self.n_ctx, head_dim, device);

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
        let blocks = (0..self.n_layer)
            .map(|_| {
                BlockConfig {
                    n_embd: self.n_embd,
                    n_head: self.n_head,
                    n_kv_head: kv_head,
                    n_ctx: self.n_ctx,
                    dropout: self.dropout,
                    attn_softcap: self.attn_softcap,
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

