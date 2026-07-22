use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

#[derive(Clone, Debug)]
pub struct KvCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Tensor<B, 1>,
    pub eps: f32,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        let weight = Tensor::<B, 1>::ones([dim], device);
        Self { weight, eps: 1e-5 }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(D - 1);
        let rms = (variance + self.eps).sqrt();
        let norm = x / rms;

        let mut shape = [1; D];
        shape[D - 1] = self.weight.dims()[0];
        let weight_broadcast = self.weight.clone().reshape(shape);
        norm * weight_broadcast
    }
}

pub fn precompute_rope_cache<B: Backend>(n_ctx: usize, head_dim: usize, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half_dim = head_dim / 2;
    let mut freqs = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let exponent = (2 * i) as f32 / head_dim as f32;
        let theta = 1.0 / 10000.0f32.powf(exponent);
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

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub cos: Tensor<B, 4>,
    pub sin: Tensor<B, 4>,
    pub n_head: usize,
    pub n_embd: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, n_embd] = x.dims();
        let head_size = n_embd / self.n_head;
        let half_dim = head_size / 2;

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q
            .reshape([batch_size, seq_len, self.n_head, head_size])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_head, head_size])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_head, head_size])
            .swap_dims(1, 2);

        // Slice precomputed GPU RoPE tensors (0 Host-to-Device PCIe transfers!)
        let cos_slice = self.cos.clone().slice([0..1, 0..1, 0..seq_len, 0..head_size]);
        let sin_slice = self.sin.clone().slice([0..1, 0..1, 0..seq_len, 0..head_size]);

        let q_x1 = q.clone().slice([0..batch_size, 0..self.n_head, 0..seq_len, 0..half_dim]);
        let q_x2 = q.clone().slice([0..batch_size, 0..self.n_head, 0..seq_len, half_dim..head_size]);
        let rotated_q = Tensor::cat(vec![q_x2.neg(), q_x1], 3);
        let q = q * cos_slice.clone() + rotated_q * sin_slice.clone();

        let k_x1 = k.clone().slice([0..batch_size, 0..self.n_head, 0..seq_len, 0..half_dim]);
        let k_x2 = k.clone().slice([0..batch_size, 0..self.n_head, 0..seq_len, half_dim..head_size]);
        let rotated_k = Tensor::cat(vec![k_x2.neg(), k_x1], 3);
        let k = k * cos_slice + rotated_k * sin_slice;

        let k_t = k.swap_dims(2, 3);
        let scale = 1.0 / (head_size as f64).sqrt();
        let mut scores = q.matmul(k_t) * scale;

        let device = scores.device();
        let arange = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device);
        let upper_mask = arange
            .clone()
            .unsqueeze_dim::<2>(0)
            .greater(arange.unsqueeze_dim::<2>(1))
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0);

        scores = scores.mask_fill(upper_mask, -1e9);

        let attn_probs = burn::tensor::activation::softmax(scores, 3);
        let out = attn_probs.matmul(v);

        let out = out.swap_dims(1, 2).reshape([batch_size, seq_len, n_embd]);
        self.out_proj.forward(out)
    }

    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        start_pos: usize,
        cache: &mut KvCache<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, new_seq_len, n_embd] = x.dims();
        let head_size = n_embd / self.n_head;
        let half_dim = head_size / 2;

        let q = self.q_proj.forward(x.clone());
        let k_new = self.k_proj.forward(x.clone());
        let v_new = self.v_proj.forward(x);

        let q = q
            .reshape([batch_size, new_seq_len, self.n_head, head_size])
            .swap_dims(1, 2);
        let k_new = k_new
            .reshape([batch_size, new_seq_len, self.n_head, head_size])
            .swap_dims(1, 2);
        let v_new = v_new
            .reshape([batch_size, new_seq_len, self.n_head, head_size])
            .swap_dims(1, 2);

        let cos_slice = self.cos.clone().slice([0..1, 0..1, start_pos..(start_pos + new_seq_len), 0..head_size]);
        let sin_slice = self.sin.clone().slice([0..1, 0..1, start_pos..(start_pos + new_seq_len), 0..head_size]);

        let q_x1 = q.clone().slice([0..batch_size, 0..self.n_head, 0..new_seq_len, 0..half_dim]);
        let q_x2 = q.clone().slice([0..batch_size, 0..self.n_head, 0..new_seq_len, half_dim..head_size]);
        let rotated_q = Tensor::cat(vec![q_x2.neg(), q_x1], 3);
        let q = q * cos_slice.clone() + rotated_q * sin_slice.clone();

        let k_x1 = k_new.clone().slice([0..batch_size, 0..self.n_head, 0..new_seq_len, 0..half_dim]);
        let k_x2 = k_new.clone().slice([0..batch_size, 0..self.n_head, 0..new_seq_len, half_dim..head_size]);
        let rotated_k = Tensor::cat(vec![k_x2.neg(), k_x1], 3);
        let k_new = k_new * cos_slice + rotated_k * sin_slice;

        let k_full = match &cache.k {
            Some(cached_k) => Tensor::cat(vec![cached_k.clone(), k_new], 2),
            None => k_new,
        };
        let v_full = match &cache.v {
            Some(cached_v) => Tensor::cat(vec![cached_v.clone(), v_new], 2),
            None => v_new,
        };

        cache.k = Some(k_full.clone());
        cache.v = Some(v_full.clone());

        let full_len = k_full.dims()[2];
        let k_t = k_full.swap_dims(2, 3);
        let scale = 1.0 / (head_size as f64).sqrt();
        let mut scores = q.matmul(k_t) * scale;

        if new_seq_len > 1 {
            let device = scores.device();
            let row_idx = Tensor::<B, 1, Int>::arange(start_pos as i64..(start_pos + new_seq_len) as i64, &device);
            let col_idx = Tensor::<B, 1, Int>::arange(0..full_len as i64, &device);
            let upper_mask = col_idx
                .unsqueeze_dim::<2>(0)
                .greater(row_idx.unsqueeze_dim::<2>(1))
                .unsqueeze_dim::<3>(0)
                .unsqueeze_dim::<4>(0);
            scores = scores.mask_fill(upper_mask, -1e9);
        }

        let attn_probs = burn::tensor::activation::softmax(scores, 3);
        let out = attn_probs.matmul(v_full);

        let out = out.swap_dims(1, 2).reshape([batch_size, new_seq_len, n_embd]);
        self.out_proj.forward(out)
    }
}

#[derive(Clone, Debug)]
pub struct CausalSelfAttentionConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_ctx: usize,
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        let head_dim = self.n_embd / self.n_head;
        let (cos, sin) = precompute_rope_cache(self.n_ctx, head_dim, device);

        CausalSelfAttention {
            q_proj: LinearConfig::new(self.n_embd, self.n_embd).with_bias(false).init(device),
            k_proj: LinearConfig::new(self.n_embd, self.n_embd).with_bias(false).init(device),
            v_proj: LinearConfig::new(self.n_embd, self.n_embd).with_bias(false).init(device),
            out_proj: LinearConfig::new(self.n_embd, self.n_embd).with_bias(false).init(device),
            cos,
            sin,
            n_head: self.n_head,
            n_embd: self.n_embd,
        }
    }
}

#[derive(Module, Debug)]
pub struct SwiGluMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> SwiGluMlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = burn::tensor::activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

#[derive(Clone, Debug)]
pub struct SwiGluMlpConfig {
    pub n_embd: usize,
}

impl SwiGluMlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGluMlp<B> {
        let hidden_dim = (8 * self.n_embd) / 3;
        SwiGluMlp {
            gate_proj: LinearConfig::new(self.n_embd, hidden_dim).with_bias(false).init(device),
            up_proj: LinearConfig::new(self.n_embd, hidden_dim).with_bias(false).init(device),
            down_proj: LinearConfig::new(hidden_dim, self.n_embd).with_bias(false).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub input_layernorm: RmsNorm<B>,
    pub attn: CausalSelfAttention<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub mlp: SwiGluMlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_norm1 = self.input_layernorm.forward(x.clone());
        let x = x + self.attn.forward(x_norm1);

        let x_norm2 = self.post_attention_layernorm.forward(x.clone());
        x + self.mlp.forward(x_norm2)
    }

    pub fn forward_with_cache(&self, x: Tensor<B, 3>, start_pos: usize, cache: &mut KvCache<B>) -> Tensor<B, 3> {
        let x_norm1 = self.input_layernorm.forward(x.clone());
        let x = x + self.attn.forward_with_cache(x_norm1, start_pos, cache);

        let x_norm2 = self.post_attention_layernorm.forward(x.clone());
        x + self.mlp.forward(x_norm2)
    }
}

#[derive(Clone, Debug)]
pub struct BlockConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_ctx: usize,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            input_layernorm: RmsNorm::new(self.n_embd, device),
            attn: CausalSelfAttentionConfig {
                n_embd: self.n_embd,
                n_head: self.n_head,
                n_ctx: self.n_ctx,
            }
            .init(device),
            post_attention_layernorm: RmsNorm::new(self.n_embd, device),
            mlp: SwiGluMlpConfig { n_embd: self.n_embd }.init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Llm<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub blocks: Vec<Block<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
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
        self.lm_head.forward(x)
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
        self.lm_head.forward(x)
    }
}

#[derive(Clone, Debug)]
pub struct LlmConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_ctx: usize,
}

impl LlmConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Llm<B> {
        let blocks = (0..self.n_layer)
            .map(|_| {
                BlockConfig {
                    n_embd: self.n_embd,
                    n_head: self.n_head,
                    n_ctx: self.n_ctx,
                }
                .init(device)
            })
            .collect();

        Llm {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.n_embd).init(device),
            blocks,
            norm: RmsNorm::new(self.n_embd, device),
            lm_head: LinearConfig::new(self.n_embd, self.vocab_size).with_bias(false).init(device),
            n_ctx: self.n_ctx,
        }
    }
}
