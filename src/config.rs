use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    #[serde(default)]
    pub n_kv_head: usize,
    pub n_layer: usize,
    pub n_ctx: usize,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default)]
    pub attn_softcap: f32,
}

impl ModelConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        n_embd: usize,
        n_head: usize,
        n_kv_head: Option<usize>,
        n_layer: usize,
        n_ctx: usize,
        dropout: f64,
        attn_softcap: f32,
    ) -> Self {
        assert!(
            n_embd.is_multiple_of(n_head),
            "n_embd ({}) must be evenly divisible by n_head ({})",
            n_embd,
            n_head
        );

        let kv_head = n_kv_head.unwrap_or(n_head);
        let kv_head = if kv_head == 0 { n_head } else { kv_head };

        assert!(
            n_head.is_multiple_of(kv_head),
            "n_head ({}) must be divisible by n_kv_head ({}) for GQA",
            n_head,
            kv_head
        );

        assert!(vocab_size > 0, "vocab_size must be greater than 0");
        assert!(n_ctx > 0, "n_ctx must be greater than 0");
        assert!(n_layer > 0, "n_layer must be greater than 0");

        Self {
            vocab_size,
            n_embd,
            n_head,
            n_kv_head: kv_head,
            n_layer,
            n_ctx,
            dropout,
            attn_softcap,
        }
    }

    pub fn get_n_kv_head(&self) -> usize {
        if self.n_kv_head == 0 {
            self.n_head
        } else {
            self.n_kv_head
        }
    }
}
