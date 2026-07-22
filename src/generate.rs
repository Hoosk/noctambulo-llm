use crate::data::LlmTokenizer;
use crate::model::Llm;
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
        (self.state as f64 / u64::MAX as f64) as f32
    }
}

pub fn generate<B: Backend>(
    model: &Llm<B>,
    tokenizer: &LlmTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
    device: &B::Device,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut rng = XorshiftRng::new(seed);
    let mut tokens = tokenizer.encode(prompt)?;
    let initial_prompt_len = tokens.len();
    let n_ctx = model.n_ctx;

    let eos_token_id = tokenizer.end_of_text_id();

    for _ in 0..max_new_tokens {
        let seq_len = tokens.len();
        let start_idx = seq_len.saturating_sub(n_ctx);

        let active_tokens = &tokens[start_idx..];

        let batch_size = 1;
        let active_len = active_tokens.len();

        let tokens_i32: Vec<i32> = active_tokens.iter().map(|&t| t as i32).collect();
        let input_data = TensorData::new(tokens_i32, [batch_size, active_len]);
        let input = Tensor::<B, 2, Int>::from_data(input_data, device);

        let logits = model.forward(input);
        let [_b, t, _vocab_size] = logits.dims();

        let last_logits = logits.slice([0..1, (t - 1)..t]);
        let logits_vec: Vec<f32> = last_logits
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("Failed to convert tensor data: {:?}", e))?;

        let next_token = if temperature <= 0.0 {
            let mut max_idx = 0;
            let mut max_val = f32::MIN;
            for (idx, &val) in logits_vec.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = idx;
                }
            }
            max_idx as u32
        } else {
            let max_val = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = logits_vec
                .iter()
                .map(|&val| ((val - max_val) / temperature).exp())
                .collect();

            let sum_exp: f32 = exp_logits.iter().sum();
            let probs: Vec<f32> = exp_logits.iter().map(|&val| val / sum_exp).collect();

            let r = rng.next_f32();
            let mut cumulative_prob = 0.0;
            let mut sampled_idx = 0;

            for (idx, &prob) in probs.iter().enumerate() {
                cumulative_prob += prob;
                if r <= cumulative_prob {
                    sampled_idx = idx;
                    break;
                }
            }
            sampled_idx as u32
        };

        // Check for End of Text token (only stop if we have generated at least 3 new tokens)
        if let Some(eos_id) = eos_token_id {
            if next_token == eos_id && (tokens.len() - initial_prompt_len) >= 3 {
                break;
            }
        }

        tokens.push(next_token);
    }

    let decoded_text = tokenizer.decode(&tokens)?;
    Ok(decoded_text)
}
