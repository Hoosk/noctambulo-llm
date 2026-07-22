use burn::module::Module;
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub n_head: usize,
    pub n_embd: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, n_embd] = x.dims();
        let head_size = n_embd / self.n_head;

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
}

#[derive(Clone, Debug)]
pub struct CausalSelfAttentionConfig {
    pub n_embd: usize,
    pub n_head: usize,
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        CausalSelfAttention {
            q_proj: LinearConfig::new(self.n_embd, self.n_embd).init(device),
            k_proj: LinearConfig::new(self.n_embd, self.n_embd).init(device),
            v_proj: LinearConfig::new(self.n_embd, self.n_embd).init(device),
            out_proj: LinearConfig::new(self.n_embd, self.n_embd).init(device),
            n_head: self.n_head,
            n_embd: self.n_embd,
        }
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub c_fc: Linear<B>,
    pub c_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(x);
        let x = burn::tensor::activation::gelu(x);
        self.c_proj.forward(x)
    }
}

#[derive(Clone, Debug)]
pub struct MlpConfig {
    pub n_embd: usize,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            c_fc: LinearConfig::new(self.n_embd, 4 * self.n_embd).init(device),
            c_proj: LinearConfig::new(4 * self.n_embd, self.n_embd).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub ln_1: LayerNorm<B>,
    pub attn: CausalSelfAttention<B>,
    pub ln_2: LayerNorm<B>,
    pub mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_norm1 = self.ln_1.forward(x.clone());
        let x = x + self.attn.forward(x_norm1);

        let x_norm2 = self.ln_2.forward(x.clone());
        x + self.mlp.forward(x_norm2)
    }

}

#[derive(Clone, Debug)]
pub struct BlockConfig {
    pub n_embd: usize,
    pub n_head: usize,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            ln_1: LayerNormConfig::new(self.n_embd).init(device),
            attn: CausalSelfAttentionConfig {
                n_embd: self.n_embd,
                n_head: self.n_head,
            }
            .init(device),
            ln_2: LayerNormConfig::new(self.n_embd).init(device),
            mlp: MlpConfig { n_embd: self.n_embd }.init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Llm<B: Backend> {
    pub wte: Embedding<B>,
    pub wpe: Embedding<B>,
    pub blocks: Vec<Block<B>>,
    pub ln_f: LayerNorm<B>,
    pub lm_head: Linear<B>,
    pub n_ctx: usize,
}

impl<B: Backend> Llm<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch_size, seq_len] = input.dims();
        let device = input.device();

        let tok_emb = self.wte.forward(input);
        let pos_indices = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device).unsqueeze_dim::<2>(0);
        let pos_emb = self.wpe.forward(pos_indices);

        let x = tok_emb + pos_emb;

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }

        let x = self.ln_f.forward(x);
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
                }
                .init(device)
            })
            .collect();

        Llm {
            wte: EmbeddingConfig::new(self.vocab_size, self.n_embd).init(device),
            wpe: EmbeddingConfig::new(self.n_ctx, self.n_embd).init(device),
            blocks,
            ln_f: LayerNormConfig::new(self.n_embd).init(device),
            lm_head: LinearConfig::new(self.n_embd, self.vocab_size).init(device),
            n_ctx: self.n_ctx,
        }
    }
}
