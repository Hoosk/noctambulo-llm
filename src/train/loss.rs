//! Loss functions, LR/context schedules, and small training helpers.

use burn::tensor::{Int, Tensor};

use crate::generate::XorshiftRng;

/// Training loss and decay knobs (Stage D6).
#[derive(Debug, Clone, Copy)]
pub struct LossOptions {
    /// Decoupled weight decay applied only to rank >= 2 parameters (weight
    /// matrices and embeddings); norms and biases are excluded.
    pub weight_decay: f64,
    /// Label-smoothing factor (0.0 disables it).
    pub label_smoothing: f32,
    /// Z-loss coefficient on `logsumexp(logits)^2` (0.0 disables it).
    pub z_loss: f32,
}

impl Default for LossOptions {
    fn default() -> Self {
        Self {
            weight_decay: 0.1,
            label_smoothing: 0.0,
            z_loss: 0.0,
        }
    }
}

/// Cross-entropy with optional label smoothing and z-loss. Kept generic so the
/// math can be unit-tested on the CPU backend.
pub(crate) fn compute_loss(
    logits_flat: Tensor<2>,
    targets_flat: Tensor<2, Int>,
    opts: LossOptions,
) -> Tensor<1> {
    let log_probs = burn::tensor::activation::log_softmax(logits_flat.clone(), 1);
    let nll = log_probs.clone().gather(1, targets_flat).mean().neg();
    let mut loss = if opts.label_smoothing > 0.0 {
        let smooth = log_probs.mean().neg();
        let eps = opts.label_smoothing as f64;
        nll * (1.0 - eps) + smooth * eps
    } else {
        nll
    };

    if opts.z_loss > 0.0 {
        let max = logits_flat.clone().max_dim(1);
        let z = (logits_flat - max.clone()).exp().sum_dim(1).log() + max;
        loss = loss + z.powf_scalar(2.0).mean() * opts.z_loss;
    }

    loss
}

pub(crate) fn shuffle_indices(indices: &mut [usize], rng: &mut XorshiftRng) {
    let len = indices.len();
    for i in (1..len).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize % (i + 1);
        indices.swap(i, j);
    }
}

/// True when early stopping should trigger (patience 0 disables it).
pub(crate) fn should_stop_early(patience: usize, epochs_without_improvement: usize) -> bool {
    patience > 0 && epochs_without_improvement >= patience
}

pub fn compute_scheduled_n_ctx(
    global_step: usize,
    warmup_steps: usize,
    target_n_ctx: usize,
) -> usize {
    if warmup_steps == 0 || global_step >= warmup_steps {
        return target_n_ctx;
    }

    let min_ctx = std::cmp::max(32, target_n_ctx / 4);
    if min_ctx >= target_n_ctx {
        return target_n_ctx;
    }

    let progress = (global_step as f32 / warmup_steps as f32).clamp(0.0, 1.0);
    let scheduled = (min_ctx as f32 + progress * (target_n_ctx - min_ctx) as f32) as usize;

    let aligned = (scheduled / 16) * 16;
    aligned.clamp(min_ctx, target_n_ctx)
}
