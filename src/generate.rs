use crate::data::LlmTokenizer;
use crate::model::{KvCache, Llm};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use std::time::Instant;

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

    /// Current internal state, so the epoch shuffler survives a resume.
    pub fn state(&self) -> u64 {
        self.state
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
    cache_f16: bool,
    device: &B::Device,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    generate_streaming(
        model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, min_p,
        repetition_penalty, seed, cache_f16, device, |_| true,
    )
}

/// Same as [`generate`], but calls `on_token` with each freshly decoded chunk
/// as soon as it is available. Returning `false` from the callback stops the
/// generation (used by the SSE endpoint when the client disconnects).
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming<B: Backend>(
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
    cache_f16: bool,
    device: &B::Device,
    mut on_token: impl FnMut(&str) -> bool,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut rng = XorshiftRng::new(seed);
    let mut tokens = tokenizer.encode(prompt)?;
    if tokens.is_empty() {
        return Ok(String::new());
    }

    // `max_seq_len` is the trained context window extended by the active RoPE
    // scaling factor (equal to `n_ctx` when scaling is off); `kv_capacity` is
    // the per-layer cache size, bounded by the sliding window when configured.
    let max_seq = model.max_seq_len();
    let capacity = model.kv_capacity();
    if tokens.len() > max_seq {
        eprintln!(
            "Warning: prompt length ({}) exceeds model context window ({}). Keeping the most recent {} tokens.",
            tokens.len(),
            max_seq,
            max_seq
        );
        let drop = tokens.len() - max_seq;
        tokens.drain(0..drop);
    }

    let initial_prompt_len = tokens.len();
    let eos_token_id = tokenizer.end_of_text_id();
    let mut decode_stream = tokenizer.decode_stream(true);

    let mut caches: Vec<KvCache<B>> = model.new_kv_caches(capacity, cache_f16, device);

    // 1. Prefill Phase: process the prompt in chunks of at most `capacity`
    // tokens so a ring cache never sees duplicate slots in a single call.
    let prompt_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let prompt_len = prompt_i32.len();
    let mut last_logits: Option<Tensor<B, 3>> = None;
    let mut pos = 0;
    while pos < prompt_len {
        let chunk = (prompt_len - pos).min(capacity);
        let chunk_data = prompt_i32[pos..(pos + chunk)].to_vec();
        let input =
            Tensor::<B, 2, Int>::from_data(TensorData::new(chunk_data, [1, chunk]), device);
        let logits = model.forward_with_cache(input, pos, &mut caches);
        if pos + chunk == prompt_len {
            last_logits = Some(logits.slice([0..1, (chunk - 1)..chunk]));
        }
        pos += chunk;
    }
    let last_logits = last_logits.expect("prompt is not empty");
    let mut logits_vec: Vec<f32> = last_logits
        .into_data()
        .convert::<f32>()
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

    if let Some(chunk) = decode_stream.step(sampled_token)? {
        if !on_token(&chunk) {
            return tokenizer.decode(&tokens);
        }
    }

    if let Some(eos_id) = eos_token_id {
        if sampled_token == eos_id {
            let decoded_text = tokenizer.decode(&tokens)?;
            return Ok(decoded_text);
        }
    }

    let gen_start = Instant::now();

    // 2. Decode Phase: process 1 new token per step using KV Cache (O(1) decoding)
    for _ in 1..max_new_tokens {
        let last_token = tokens.last().copied().unwrap_or(0) as i32;
        let start_pos = tokens.len() - 1;

        if start_pos >= max_seq {
            println!("\n[Reached maximum context length ({max_seq}). Stopping generation.]");
            break;
        }

        let input_data = TensorData::new(vec![last_token], [1, 1]);
        let input = Tensor::<B, 2, Int>::from_data(input_data, device);

        let logits = model.forward_with_cache(input, start_pos, &mut caches);
        let last_logits = logits.slice([0..1, 0..1]);
        let mut logits_vec: Vec<f32> = last_logits
            .into_data()
            .convert::<f32>()
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

        if let Some(chunk) = decode_stream.step(sampled_token)? {
            if !on_token(&chunk) {
                break;
            }
        }

        if let Some(eos_id) = eos_token_id {
            if sampled_token == eos_id && (tokens.len() - initial_prompt_len) >= 3 {
                break;
            }
        }
    }

    let elapsed = gen_start.elapsed().as_secs_f64();
    let generated_count = tokens.len().saturating_sub(initial_prompt_len);
    let speed = if elapsed > 0.0 { generated_count as f64 / elapsed } else { 0.0 };

    let decoded_text = tokenizer.decode(&tokens)?;
    println!(
        "\n[Generation Complete: {} tokens generated in {:.2}s ({:.1} tokens/sec)]",
        generated_count, elapsed, speed
    );
    Ok(decoded_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift_rng() {
        let mut rng = XorshiftRng::new(42);
        let v1 = rng.next_f32();
        let v2 = rng.next_f32();
        assert!((0.0..=1.0).contains(&v1));
        assert!((0.0..=1.0).contains(&v2));
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_sample_logits_repetition_penalty() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let tokens = vec![3u32]; // Token 3 was seen
        let mut rng = XorshiftRng::new(123);

        // Without penalty, token 3 has highest logit (4.0)
        // With penalty 2.0, token 3's logit becomes 4.0 / 2.0 = 2.0
        let _token = sample_logits(
            &mut logits,
            &tokens,
            1.0,
            0,
            1.0,
            0.0,
            2.0, // repetition penalty
            &mut rng,
        );

        assert_eq!(logits[3], 2.0); // 4.0 / 2.0
    }

    #[test]
    fn test_sample_logits_temperature_zero() {
        let mut logits = vec![1.0f32, 5.0, 2.0];
        let tokens = vec![];
        let mut rng = XorshiftRng::new(1);

        let token = sample_logits(&mut logits, &tokens, 0.001, 1, 1.0, 0.0, 1.0, &mut rng);
        assert_eq!(token, 1); // highest score index is 1
    }
}

