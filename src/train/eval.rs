//! Evaluation entry point and loss/perplexity/BPB metrics.

use std::path::Path;
use std::time::Instant;

use burn::module::Module;
use burn::tensor::Device;

use crate::config::ModelConfig;
use crate::data::LlmTokenizer;
use crate::storage;

use super::dataset::TextDataset;

pub fn evaluate_dataset(
    model: &crate::model::Llm,
    dataset: &TextDataset,
    batch_size: usize,
    max_batches: usize,
    device: &Device,
) -> (f32, f32, Option<f32>) {
    if dataset.len() == 0 {
        return (0.0, 1.0, None);
    }
    let total_batches = dataset.len().div_ceil(batch_size);
    let eval_batches = if max_batches == 0 {
        total_batches
    } else {
        total_batches.min(max_batches)
    };
    let mut total_loss = 0.0f32;

    for batch_idx in 0..eval_batches {
        let start = batch_idx * batch_size;
        let end = std::cmp::min(start + batch_size, dataset.len());
        let indices: Vec<usize> = (start..end).collect();

        let (inputs, targets) = dataset.get_batch(&indices, model.n_ctx, device);
        let logits = model.forward(inputs);
        let [b, t, v] = logits.dims();

        let logits_flat = logits.reshape([b * t, v]);
        let log_probs = burn::tensor::activation::log_softmax(logits_flat, 1);
        let targets_flat = targets.reshape([b * t, 1]);
        let target_log_probs = log_probs.gather(1, targets_flat);
        let loss = target_log_probs.mean().neg();

        let loss_val = loss
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.0);

        total_loss += loss_val;
    }

    let avg_loss = total_loss / (eval_batches as f32);
    let perplexity = avg_loss.exp();
    // BPB = (NLL in bits) / source bytes. The tokens/bytes ratio of the full
    // dataset is a valid estimator because both scale together with the
    // evaluated subset.
    let bpb = if dataset.total_bytes > 0 && !dataset.tokens.is_empty() {
        Some(avg_loss / std::f32::consts::LN_2 * (dataset.tokens.len() as f32 / dataset.total_bytes as f32))
    } else {
        None
    };
    (avg_loss, perplexity, bpb)
}

#[allow(clippy::too_many_arguments)]
pub fn run_eval(
    model_dir: &Path,
    dataset_path: &Path,
    tokenizer: &LlmTokenizer,
    n_ctx: Option<usize>,
    eval_batch_size: usize,
    eval_batches: usize,
    max_tokens: usize,
    device: &Device,
    rope_override: Option<(crate::config::RopeScalingKind, f32)>,
    dense: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(format!("Model config not found at {:?}", config_path).into());
    }
    let config_str = std::fs::read_to_string(&config_path)?;
    let mut config: ModelConfig = serde_json::from_str(&config_str)?;
    config.check_arch_version(&config_path.display().to_string())?;

    if let Some(ctx) = n_ctx {
        config.n_ctx = ctx;
    }
    if let Some((kind, factor)) = rope_override {
        config.set_rope_scaling(kind, factor);
        println!("RoPE scaling override: {} (factor {})", kind.as_str(), config.rope_factor);
    }
    if dense {
        config.kv_eviction = None;
        println!("Dense evaluation: the checkpoint's heavy-hitter eviction mask is disabled.");
    } else if let Some(geo) = config.eviction_geometry()? {
        println!(
            "Eviction-aware evaluation: budget {} rows ({} sinks, {} recent, {} hitters).",
            geo.budget, geo.sinks, geo.recent, geo.hitters
        );
    }

    println!("Loading model config from {:?}...", config_path);
    let model: crate::model::Llm = config.init_model(device);

    let weights_path = crate::storage::model_weights_path(model_dir);
    let model = if weights_path.exists() {
        println!("Loading model weights from {:?}...", weights_path);
        storage::load_module(model, &storage::model_weights_path(model_dir))
            .map_err(|e| format!("Failed to load model record: {e}"))?
            .valid()
    } else if let Some(legacy) = crate::storage::legacy_weights_path(model_dir) {
        return Err(format!(
            "'{}' is a Burn 0.21 bincode checkpoint; this build uses burnpack. Re-import the \
             model (import-hf/import) or retrain it (legacy file: '{}')",
            model_dir.display(),
            legacy.display()
        )
        .into());
    } else {
        println!("Warning: model_weights.bpk not found in {:?}. Evaluating initialized model.", model_dir);
        model
    };

    let token_limit = if max_tokens == 0 { None } else { Some(max_tokens) };
    let dataset = TextDataset::from_file(dataset_path, tokenizer, config.n_ctx, None, token_limit)?;
    let batch_limit = if eval_batches == 0 { "all".to_string() } else { eval_batches.to_string() };
    println!(
        "Evaluating model on dataset {:?} ({} samples, {} tokens) [batch={}, batches={}]...",
        dataset_path, dataset.len(), dataset.tokens.len(), eval_batch_size, batch_limit
    );

    let start_time = Instant::now();
    let (loss, ppl, bpb) = evaluate_dataset(&model, &dataset, eval_batch_size, eval_batches, device);
    let elapsed = start_time.elapsed().as_secs_f64();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    EVALUATION RESULTS                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Cross-Entropy Loss : {:<39.4} ║", loss);
    println!("║ Perplexity (PPL)   : {:<39.4} ║", ppl);
    match bpb {
        Some(bpb) => println!("║ Bits per Byte (BPB): {:<39.4} ║", bpb),
        None => println!("║ Bits per Byte (BPB): {:<39} ║", "n/a (.bin input has no byte count)"),
    }
    println!("║ Evaluation Time    : {:<35.2}s ║", elapsed);
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
