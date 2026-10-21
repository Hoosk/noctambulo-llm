use crate::config::EvictionGeometry;
use crate::data::LlmTokenizer;
use crate::model::{KvCache, KvCacheDtype, Llm};

/// Runtime heavy-hitter eviction options: the trained geometry plus the
/// optional host-RAM spill archive and budget-growth schedule. Growth starts
/// from `geometry.budget` and adds `grow_rows` rows every `grow_interval`
/// processed tokens until `max_budget`; every new slot is filled by replaying
/// the highest-score spilled rows.
#[derive(Clone, Copy, Debug)]
pub struct EvictionOptions {
    pub geometry: EvictionGeometry,
    /// Archive evicted rows in host RAM instead of discarding them.
    pub spill: bool,
    /// Physical cache rows (replay ceiling). `0` = the initial budget.
    pub max_budget: usize,
    /// Rows added to the resident budget per growth step (0 = no growth).
    pub grow_rows: usize,
    /// Processed tokens between growth steps (0 = no growth).
    pub grow_interval: usize,
}

impl From<EvictionGeometry> for EvictionOptions {
    fn from(geometry: EvictionGeometry) -> Self {
        Self {
            geometry,
            spill: false,
            max_budget: 0,
            grow_rows: 0,
            grow_interval: 0,
        }
    }
}

impl EvictionOptions {
    /// Physical cache capacity (replay ceiling), clamped to the context by the
    /// caller.
    pub fn capacity(&self) -> usize {
        if self.max_budget > 0 {
            self.max_budget.max(self.geometry.budget)
        } else {
            self.geometry.budget
        }
    }

    /// Rejects impossible combinations before any model is loaded.
    pub fn validate(&self) -> Result<(), String> {
        if (self.spill || self.grow_rows > 0 || self.max_budget > 0)
            && self.geometry.budget == 0
        {
            return Err("KV spill/growth requires heavy-hitter eviction (kv_eviction=h2o)".into());
        }
        if self.max_budget > 0 && self.max_budget < self.geometry.budget {
            return Err(format!(
                "kv_max_budget ({}) cannot be smaller than the initial budget ({})",
                self.max_budget, self.geometry.budget
            ));
        }
        if self.grow_rows > 0 && self.geometry.budget == self.capacity() {
            return Err(
                "kv_grow needs kv_max_budget greater than the initial budget".into(),
            );
        }
        Ok(())
    }
}

/// Advances the budget-growth schedule: adds `grow_rows` rows to every cache
/// budget each `grow_interval` processed tokens (spill replay fills the slots).
/// Returns the number of replayed rows.
fn advance_growth(
    caches: &mut [KvCache],
    grow_rows: usize,
    grow_interval: usize,
    since_growth: &mut usize,
    processed: usize,
) -> usize {
    if grow_rows == 0 || grow_interval == 0 {
        return 0;
    }
    *since_growth += processed;
    let mut replayed = 0;
    while *since_growth >= grow_interval {
        *since_growth -= grow_interval;
        let mut grew = false;
        for cache in caches.iter_mut() {
            if cache.budget < cache.capacity {
                replayed += cache.grow_and_replay(grow_rows);
                grew = true;
            } else {
                replayed += cache.replay_spilled();
            }
        }
        if !grew {
            break;
        }
    }
    replayed
}
use burn::tensor::{Int, Tensor, TensorData, Device};
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
pub fn generate(
    model: &Llm,
    tokenizer: &LlmTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    seed: u64,
    cache_dtype: KvCacheDtype,
    eviction: Option<EvictionOptions>,
    device: &Device,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    generate_streaming(
        model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, min_p,
        repetition_penalty, seed, cache_dtype, eviction, device, |_| true,
    )
}

/// Same as [`generate`], but calls `on_token` with each freshly decoded chunk
/// as soon as it is available. Returning `false` from the callback stops the
/// generation (used by the SSE endpoint when the client disconnects).
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming(
    model: &Llm,
    tokenizer: &LlmTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    seed: u64,
    cache_dtype: KvCacheDtype,
    eviction: Option<EvictionOptions>,
    device: &Device,
    mut on_token: impl FnMut(&str) -> bool,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut rng = XorshiftRng::new(seed);
    let mut tokens = tokenizer.encode(prompt)?;
    if tokens.is_empty() {
        return Ok(String::new());
    }

    // `max_seq_len` is the trained context window extended by the active RoPE
    // scaling factor (equal to `n_ctx` when scaling is off); the cache
    // capacity is the eviction budget (or its replay ceiling when the budget
    // grows), otherwise the per-layer size bounded by the sliding window.
    let max_seq = model.max_seq_len();
    let capacity = match &eviction {
        Some(opts) => opts.capacity().min(max_seq),
        None => model.kv_capacity(),
    };
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

    let mut caches: Vec<KvCache> = model.new_kv_caches(capacity, cache_dtype, device);
    let mut since_growth = 0usize;
    if let Some(opts) = &eviction {
        let geo = opts.geometry;
        let budget = geo.budget.min(capacity);
        for cache in caches.iter_mut() {
            cache.enable_eviction_with(geo.recent, budget, opts.spill);
        }
        let growth = if opts.grow_rows > 0 && opts.grow_interval > 0 {
            format!(
                ", growing to {} rows ({} every {} tokens)",
                capacity, opts.grow_rows, opts.grow_interval
            )
        } else {
            String::new()
        };
        let spill = if opts.spill { ", spill to host RAM" } else { "" };
        println!(
            "\n[KV cache: heavy-hitter eviction, budget {} rows ({} sinks, {} recent, {} hitters){growth}{spill}]",
            budget,
            geo.sinks,
            geo.recent,
            budget.saturating_sub(geo.sinks + geo.recent)
        );
    }

    // 1. Prefill Phase: process the prompt in chunks that are homogeneous for
    // the cache policy. A ring cache takes at most `capacity` rows per call;
    // with eviction, a call either appends only (chunk <= free rows) or evicts
    // only (chunk <= evictable rows = budget - sinks - recent), so a victim is
    // always available and no slot is written twice in one call.
    let prompt_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let prompt_len = prompt_i32.len();
    let mut last_logits: Option<Tensor<3>> = None;
    let mut pos = 0;
    while pos < prompt_len {
        let step = if caches[0].eviction_enabled() {
            let free = caches[0].budget.saturating_sub(caches[0].len);
            let evictable = caches[0]
                .budget
                .saturating_sub(caches[0].recent + model.sink_tokens())
                .max(1);
            if free > 0 {
                free.min(evictable)
            } else {
                evictable
            }
        } else {
            capacity
        };
        let chunk = (prompt_len - pos).min(step);
        let chunk_data = prompt_i32[pos..(pos + chunk)].to_vec();
        let input =
            Tensor::<2, Int>::from_data(TensorData::new(chunk_data, [1, chunk]), device);
        let logits = model.forward_with_cache(input, pos, &mut caches);
        if let Some(opts) = &eviction {
            advance_growth(
                &mut caches,
                opts.grow_rows,
                opts.grow_interval,
                &mut since_growth,
                chunk,
            );
        }
        if pos + chunk == prompt_len {
            last_logits = Some(logits.slice([0..1, (chunk - 1)..chunk]));
        }
        pos += chunk;
    }
    let last_logits = last_logits.expect("prompt is not empty");
    let mut logits_vec: Vec<f32> = last_logits
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
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

    if let Some(chunk) = decode_stream.step(sampled_token)?
        && !on_token(&chunk) {
            return tokenizer.decode(&tokens);
        }

    if let Some(eos_id) = eos_token_id
        && sampled_token == eos_id {
            let decoded_text = tokenizer.decode(&tokens)?;
            return Ok(decoded_text);
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
        let input = Tensor::<2, Int>::from_data(input_data, device);

        let logits = model.forward_with_cache(input, start_pos, &mut caches);
        if let Some(opts) = &eviction {
            advance_growth(
                &mut caches,
                opts.grow_rows,
                opts.grow_interval,
                &mut since_growth,
                1,
            );
        }
        let last_logits = logits.slice([0..1, 0..1]);
        let mut logits_vec: Vec<f32> = last_logits
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
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

        if let Some(chunk) = decode_stream.step(sampled_token)?
            && !on_token(&chunk) {
                break;
            }

        if let Some(eos_id) = eos_token_id
            && sampled_token == eos_id && (tokens.len() - initial_prompt_len) >= 3 {
                break;
            }
    }

    let elapsed = gen_start.elapsed().as_secs_f64();
    let generated_count = tokens.len().saturating_sub(initial_prompt_len);
    let speed = if elapsed > 0.0 { generated_count as f64 / elapsed } else { 0.0 };

    let decoded_text = tokenizer.decode(&tokens)?;
    if let Some(opts) = &eviction
        && (opts.spill || opts.grow_rows > 0) {
            let spilled: usize = caches.iter().map(|c| c.spilled_total).sum();
            let replayed: usize = caches.iter().map(|c| c.replayed_total).sum();
            let pending: usize = caches
                .iter()
                .filter_map(|c| c.spill.as_ref())
                .map(|s| s.len())
                .sum();
            let bytes: usize = caches
                .iter()
                .filter_map(|c| c.spill.as_ref())
                .map(|s| s.bytes())
                .sum();
            let budget = caches.first().map(|c| c.budget).unwrap_or(0);
            println!(
                "\n[KV spill: {} rows archived in host RAM ({:.2} MB), {} replayed, {} still spilled; resident budget {} / {} rows]",
                spilled,
                bytes as f64 / (1024.0 * 1024.0),
                replayed,
                pending,
                budget,
                capacity
            );
        }
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

    #[test]
    fn test_eviction_options_validate() {
        let geometry = EvictionGeometry { budget: 128, recent: 16, sinks: 4, hitters: 108 };
        let base = EvictionOptions::from(geometry);
        assert!(base.validate().is_ok());
        assert_eq!(base.capacity(), 128);

        // Spill alone is always valid; growth needs headroom.
        assert!(EvictionOptions { spill: true, ..base }.validate().is_ok());
        let err = EvictionOptions { grow_rows: 32, ..base }
            .validate()
            .unwrap_err();
        assert!(err.contains("kv_max_budget"), "{err}");

        let ok = EvictionOptions {
            spill: true,
            max_budget: 512,
            grow_rows: 32,
            grow_interval: 64,
            ..base
        };
        assert!(ok.validate().is_ok());
        assert_eq!(ok.capacity(), 512);

        // The ceiling cannot undercut the initial budget.
        let err = EvictionOptions { max_budget: 64, ..base }
            .validate()
            .unwrap_err();
        assert!(err.contains("smaller"), "{err}");
    }
}

