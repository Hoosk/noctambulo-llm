use crate::data::LlmTokenizer;
use crate::model::{KvCache, Llm};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

pub struct XorshiftRng {
    state: u64,
}

impl XorshiftRng {
    pub fn new(seed: u64) -> Self {
        let seed = if seed == 0 { 42 } else { seed };
        Self { state: seed }
    }

    pub fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        let scrambled = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        (scrambled as f64 / u64::MAX as f64) as f32
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_logits(
    logits_vec: &mut [f32],
    tokens: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    rng: &mut XorshiftRng,
) -> u32 {
    if repetition_penalty > 1.0 {
        let mut seen = std::collections::HashSet::new();
        for &token in tokens {
            if seen.insert(token) {
                let idx = token as usize;
                if idx < logits_vec.len() {
                    if logits_vec[idx] < 0.0 {
                        logits_vec[idx] *= repetition_penalty;
                    } else {
                        logits_vec[idx] /= repetition_penalty;
                    }
                }
            }
        }
    }

    let temp = if temperature <= 0.0 { 0.001 } else { temperature };
    for val in logits_vec.iter_mut() {
        *val /= temp;
    }

    let mut indexed_scores: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if top_k > 0 && top_k < indexed_scores.len() {
        indexed_scores.truncate(top_k);
    }

    // Top-P Nucleus Sampling
    if top_p > 0.0 && top_p < 1.0 {
        let max_val = indexed_scores.first().map(|s| s.1).unwrap_or(0.0);
        let mut sum_exp = 0.0f32;
        let mut exps: Vec<f32> = Vec::with_capacity(indexed_scores.len());

        for (_, score) in &indexed_scores {
            let exp = (score - max_val).exp();
            exps.push(exp);
            sum_exp += exp;
        }

        let mut cumulative = 0.0f32;
        let mut cutoff_idx = indexed_scores.len();

        for (i, exp) in exps.iter().enumerate() {
            let prob = exp / sum_exp;
            cumulative += prob;
            if cumulative >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        indexed_scores.truncate(cutoff_idx);
    }

    // Min-P Sampling Math (Relative Probability Thresholding)
    if min_p > 0.0 && min_p <= 1.0 && !indexed_scores.is_empty() {
        let max_val = indexed_scores[0].1;
        let mut sum_exp = 0.0f32;
        let exps: Vec<f32> = indexed_scores
            .iter()
            .map(|(_, score)| {
                let e = (score - max_val).exp();
                sum_exp += e;
                e
            })
            .collect();

        if sum_exp > 0.0 {
            let max_prob = exps[0] / sum_exp;
            let threshold = min_p * max_prob;
            let mut keep_count = 0;
            for (i, &exp) in exps.iter().enumerate() {
                let prob = exp / sum_exp;
                if prob >= threshold {
                    keep_count = i + 1;
                } else {
                    break;
                }
            }
            if keep_count > 0 {
                indexed_scores.truncate(keep_count);
            }
        }
    }

    let max_val = indexed_scores.first().map(|s| s.1).unwrap_or(0.0);
    let mut exps: Vec<f32> = Vec::with_capacity(indexed_scores.len());
    let mut sum_exp = 0.0f32;

    for (_, score) in &indexed_scores {
        let exp = (score - max_val).exp();
        exps.push(exp);
        sum_exp += exp;
    }

    let r = rng.next_f32();
    let mut cumulative_prob = 0.0f32;
    let mut sampled_token = indexed_scores.first().map(|s| s.0 as u32).unwrap_or(0);

    for (i, (token_idx, _)) in indexed_scores.iter().enumerate() {
        let prob = exps[i] / sum_exp;
        cumulative_prob += prob;
        if r <= cumulative_prob {
            sampled_token = *token_idx as u32;
            break;
        }
    }

    sampled_token
}

#[allow(clippy::too_many_arguments)]
pub fn generate<B: Backend>(
    model: &Llm<B>,
    tokenizer: &LlmTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    seed: u64,
    device: &B::Device,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut rng = XorshiftRng::new(seed);
    let mut tokens = tokenizer.encode(prompt)?;
    if tokens.is_empty() {
        return Ok(String::new());
    }

    if tokens.len() > model.n_ctx {
        eprintln!(
            "Warning: prompt length ({}) exceeds model context window ({}). Truncating prompt.",
            tokens.len(),
            model.n_ctx
        );
        tokens.truncate(model.n_ctx);
    }

    let initial_prompt_len = tokens.len();
    let eos_token_id = tokenizer.end_of_text_id();

    let mut caches: Vec<KvCache<B>> = (0..model.blocks.len())
        .map(|_| KvCache::empty())
        .collect();

    // 1. Prefill Phase: process prompt tokens in one forward pass
    let prompt_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let prompt_len = prompt_i32.len();
    let input_data = TensorData::new(prompt_i32, [1, prompt_len]);
    let input = Tensor::<B, 2, Int>::from_data(input_data, device);

    let logits = model.forward_with_cache(input, 0, &mut caches);
    let last_logits = logits.slice([0..1, (prompt_len - 1)..prompt_len]);
    let mut logits_vec: Vec<f32> = last_logits
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("Failed to convert tensor data: {:?}", e))?;

    let sampled_token = sample_logits(
        &mut logits_vec,
        &tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        &mut rng,
    );
    tokens.push(sampled_token);

    if let Some(eos_id) = eos_token_id {
        if sampled_token == eos_id {
            let decoded_text = tokenizer.decode(&tokens)?;
            return Ok(decoded_text);
        }
    }

    // 2. Decode Phase: process 1 new token per step using KV Cache (O(1) decoding)
    for _ in 1..max_new_tokens {
        let last_token = tokens.last().copied().unwrap_or(0) as i32;
        let start_pos = tokens.len() - 1;

        if start_pos >= model.n_ctx {
            println!("\n[Reached maximum context length ({}). Stopping generation.]", model.n_ctx);
            break;
        }

        let input_data = TensorData::new(vec![last_token], [1, 1]);
        let input = Tensor::<B, 2, Int>::from_data(input_data, device);

        let logits = model.forward_with_cache(input, start_pos, &mut caches);
        let last_logits = logits.slice([0..1, 0..1]);
        let mut logits_vec: Vec<f32> = last_logits
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("Failed to convert tensor data: {:?}", e))?;

        let sampled_token = sample_logits(
            &mut logits_vec,
            &tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            &mut rng,
        );
        tokens.push(sampled_token);

        if let Some(eos_id) = eos_token_id {
            if sampled_token == eos_id && (tokens.len() - initial_prompt_len) >= 3 {
                break;
            }
        }
    }

    let decoded_text = tokenizer.decode(&tokens)?;
    Ok(decoded_text)
}
