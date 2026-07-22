use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_ctx: usize,
}

impl ModelConfig {
    pub fn new(vocab_size: usize, n_embd: usize, n_head: usize, n_layer: usize, n_ctx: usize) -> Self {
        assert!(
            n_embd.is_multiple_of(n_head),
            "n_embd ({}) must be evenly divisible by n_head ({})",
            n_embd,
            n_head
        );

        assert!(vocab_size > 0, "vocab_size must be greater than 0");
        assert!(n_ctx > 0, "n_ctx must be greater than 0");
        assert!(n_layer > 0, "n_layer must be greater than 0");

        Self {
            vocab_size,
            n_embd,
            n_head,
            n_layer,
            n_ctx,
        }
    }
}
