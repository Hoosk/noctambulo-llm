mod config;
mod data;
mod export;
mod generate;
mod model;
mod train;
mod tui;

use clap::{Args, Parser, Subcommand};
use config::ModelConfig;
use data::{combine_and_mix_datasets, combine_and_mix_weighted, download_dataset_from_hf, extract_parquet_text_column, prepare_binary_dataset, train_custom_tokenizer, LlmTokenizer};
use export::export_to_gguf;
use generate::generate;
use model::Llm;
use std::path::{Path, PathBuf};
use train::{run_eval, train};

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

const DEFAULT_CORPUS: &str = "\
Rust is a multi-paradigm, general-purpose programming language. \
Rust is designed for performance and safety, especially safe concurrency. \
Rust is syntactically similar to C++, but can guarantee memory safety by using \
a borrow checker to validate references. Rust achieves memory safety without \
garbage collection, and reference counting is optional. \
Rust is a great language to implement deep learning models because of its performance \
and safety. Burn is a new deep learning framework written in Rust. \
It provides flexible and efficient tensor operations on both CPU and GPU. \
By combining Rust and Burn, we can build custom neural networks and train them \
safely. Let us write a custom language model in Rust using Burn. \
This model will learn the statistics of the words and start producing text \
that sounds like a tutorial or technical documentation on Rust and Burn. \
";

#[derive(Parser)]
#[command(name = "kuro-neko", author, version, about = "🐈‍⬛ Kuro-Neko 2026 Engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a custom BPE tokenizer on corpus or datasets
    TrainTokenizer(TrainTokenizerArgs),
    /// Download a dataset file from Hugging Face Hub (Parquet auto-extracts text)
    Download(DownloadArgs),
    /// Pre-tokenize corpus or HF dataset to compact binary dataset format (.bin)
    PrepareDataset(PrepareDatasetArgs),
    /// Train the causal transformer model
    Train(TrainArgs),
    /// Evaluate model cross-entropy loss and perplexity (PPL) on dataset
    Eval(EvalArgs),
    /// Generate text autoregressively from a prompt
    Generate(GenerateArgs),
    /// Export model weights and metadata to GGUF format
    Export(ExportArgs),
    /// Print model parameter count, VRAM footprint, and FLOPs statistics
    Stats,
}

#[derive(Args)]
struct TrainTokenizerArgs {
    /// HF dataset repo ID(s) (comma-separated, e.g. roneneldan/TinyStories)
    #[arg(long, value_delimiter = ',')]
    dataset: Vec<String>,
    /// Local dataset file path(s) (comma-separated, order matches --weights)
    #[arg(long, value_delimiter = ',')]
    file: Vec<String>,
    /// Target file name inside HF dataset repo
    #[arg(long, default_value = "TinyStories-train.txt")]
    dataset_file: String,
    /// Vocabulary size
    #[arg(long, default_value_t = 8000)]
    vocab_size: usize,
    /// Mix weights per source (comma-separated, same order as --file).
    /// When set, sources are mixed probabilistically by these proportions.
    #[arg(long, value_delimiter = ',')]
    weights: Vec<f32>,
    /// Max characters of mixed corpus to train tokenizer on (0 = all available)
    #[arg(long, default_value_t = 0)]
    max_chars: usize,
}

#[derive(Args)]
struct DownloadArgs {
    /// HF dataset repo ID (e.g. HuggingFaceTB/cosmopedia-100k)
    #[arg(long)]
    repo: String,
    /// File name inside the repo (e.g. data/train-00000-of-00001.parquet)
    #[arg(long)]
    file: String,
    /// Output directory for the downloaded file (default: datasets/)
    #[arg(long, default_value = "datasets")]
    out_dir: String,
    /// Extract text column to .txt when the file is Parquet (default: true)
    #[arg(long, default_value_t = true)]
    extract_text: bool,
}

#[derive(Args)]
struct PrepareDatasetArgs {
    /// HF dataset repo ID(s) (comma-separated, e.g. roneneldan/TinyStories)
    #[arg(long, value_delimiter = ',')]
    dataset: Vec<String>,
    /// Local dataset file path(s) (comma-separated, order matches --weights)
    #[arg(long, value_delimiter = ',')]
    file: Vec<String>,
    /// Target file name inside HF dataset repo
    #[arg(long, default_value = "TinyStories-train.txt")]
    dataset_file: String,
    /// Output binary dataset file path (.bin)
    #[arg(long, default_value = "datasets/dataset.bin")]
    output: String,
    /// Max dataset tokens to tokenize (0 = all)
    #[arg(long, default_value_t = 0)]
    max_tokens: usize,
    /// Mix weights per source (comma-separated, same order as --dataset/--file).
    /// When set, sources are mixed probabilistically by these proportions.
    #[arg(long, value_delimiter = ',')]
    weights: Vec<f32>,
    /// Max characters to write for weighted mixing (0 = all available)
    #[arg(long, default_value_t = 0)]
    max_chars: usize,
}

#[derive(Args)]
struct EvalArgs {
    /// Model output directory containing config.json and model_weights.bin
    #[arg(long, default_value = "model_output")]
    model_dir: String,
    /// HF dataset repo ID(s)
    #[arg(long, value_delimiter = ',')]
    dataset: Vec<String>,
    /// Local dataset file path (.txt or .bin)
    #[arg(long)]
    file: Option<String>,
    /// Target file name inside HF dataset repo
    #[arg(long, default_value = "TinyStories-train.txt")]
    dataset_file: String,
    /// Context window size override
    #[arg(long)]
    n_ctx: Option<usize>,
    /// Evaluation batch size (default 16, lower if OOM on small GPUs)
    #[arg(long, default_value_t = 16)]
    eval_batch_size: usize,
}

#[derive(Args)]
struct TrainArgs {
    /// HF dataset repo ID(s) (comma-separated, e.g. roneneldan/TinyStories)
    #[arg(long, value_delimiter = ',')]
    dataset: Vec<String>,
    /// Local dataset file path
    #[arg(long)]
    file: Option<String>,
    /// Target file name inside HF dataset repo
    #[arg(long, default_value = "TinyStories-train.txt")]
    dataset_file: String,
    /// Batch size
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    /// Number of epochs
    #[arg(long, default_value_t = 3)]
    epochs: usize,
    /// Max dataset tokens to load (0 = all)
    #[arg(long, default_value_t = 10_000_000)]
    max_tokens: usize,
    /// Sliding window step in tokens (default: same as n-ctx)
    #[arg(long)]
    step: Option<usize>,
    /// Context window size
    #[arg(long, default_value_t = 256)]
    n_ctx: usize,
    /// Embedding dimension
    #[arg(long, default_value_t = 256)]
    n_embd: usize,
    /// Attention query heads count
    #[arg(long, default_value_t = 4)]
    n_head: usize,
    /// Attention key/value heads count for GQA (0 or None = same as n-head)
    #[arg(long)]
    n_kv_head: Option<usize>,
    /// Transformer layers count
    #[arg(long, default_value_t = 4)]
    n_layer: usize,
    /// Learning rate
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    /// Dropout probability
    #[arg(long, default_value_t = 0.1)]
    dropout: f64,
    /// Attention score softcapping threshold (0.0 = disabled, 50.0 = Gemma 4 math)
    #[arg(long, default_value_t = 0.0)]
    attn_softcap: f32,
    /// Validation set ratio (0.1 = 10% validation split)
    #[arg(long, default_value_t = 0.1)]
    val_split: f32,
    /// Gradient accumulation steps (default = 1)
    #[arg(long, default_value_t = 1)]
    grad_accum: usize,
    /// Directory to save epoch checkpoints (default: None)
    #[arg(long)]
    save_dir: Option<String>,
    /// Checkpoint file path to resume training from
    #[arg(long)]
    resume: Option<String>,
    /// Number of steps for progressive sequence length warmup (0 = disabled)
    #[arg(long, default_value_t = 0)]
    seq_warmup_steps: usize,
}

#[derive(Args)]
struct GenerateArgs {
    /// Prompt text for completion
    prompt: String,
    /// Sampling temperature (0.9 works best for the trained SLM)
    #[arg(long, default_value_t = 0.9)]
    temp: f32,
    /// Top-K sampling
    #[arg(long, default_value_t = 60)]
    top_k: usize,
    /// Top-P (nucleus) sampling
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    /// Min-P sampling threshold (0.1 prevents token loops in trained SLMs)
    #[arg(long, default_value_t = 0.1)]
    min_p: f32,
    /// Repetition penalty (1.6 prevents token loops in trained SLMs)
    #[arg(long, default_value_t = 1.6)]
    rep_penalty: f32,
    /// Maximum new tokens to generate
    #[arg(long, default_value_t = 50)]
    max_tokens: usize,
    /// Random seed
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Args)]
struct ExportArgs {
    /// Name of the exported GGUF model binary
    #[arg(long, default_value = "kuro-neko-qwen")]
    model_name: String,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    let output_dir = "model_output";
    let custom_tokenizer_file = "custom_tokenizer.json";
    let device = WgpuDevice::default();

    match cli.command {
        Commands::TrainTokenizer(args) => {
            let mut sources: Vec<PathBuf> = Vec::new();
            for repo in &args.dataset {
                let repo = repo.trim();
                if !repo.is_empty() {
                    let downloaded = download_dataset_from_hf(repo, &args.dataset_file)?;
                    sources.push(downloaded);
                }
            }
            if !args.file.is_empty() {
                for f in &args.file {
                    let f = f.trim();
                    if !f.is_empty() {
                        sources.push(PathBuf::from(f));
                    }
                }
            }

            let corpus_path = if sources.is_empty() {
                let default_file = Path::new("corpus.txt");
                if !default_file.exists() {
                    std::fs::write(default_file, DEFAULT_CORPUS)?;
                }
                default_file.to_string_lossy().to_string()
            } else if !args.weights.is_empty() {
                if args.weights.len() != sources.len() {
                    return Err(format!(
                        "Weights count ({}) must match sources count ({})",
                        args.weights.len(),
                        sources.len()
                    )
                    .into());
                }
                let weighted: Vec<(PathBuf, f32)> =
                    sources.iter().cloned().zip(args.weights.iter().copied()).collect();
                let max_chars = if args.max_chars == 0 { usize::MAX } else { args.max_chars };
                combine_and_mix_weighted(&weighted, Path::new("datasets/weighted_mix.txt"), max_chars)?
                    .to_string_lossy()
                    .to_string()
            } else if sources.len() == 1 {
                sources[0].to_string_lossy().to_string()
            } else {
                let combined = combine_and_mix_datasets(&sources, Path::new("datasets/combined_corpus.txt"))?;
                combined.to_string_lossy().to_string()
            };

            println!("Training custom BPE tokenizer (vocab_size={}) on '{}'...", args.vocab_size, corpus_path);
            train_custom_tokenizer(&corpus_path, custom_tokenizer_file, args.vocab_size)?;
            println!("Tokenizer successfully trained and saved to '{}'.", custom_tokenizer_file);
        }
        Commands::Download(args) => {
            let downloaded = download_dataset_from_hf(&args.repo, &args.file)?;
            let out_dir = PathBuf::from(&args.out_dir);
            std::fs::create_dir_all(&out_dir)?;

            let file_name = downloaded
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| args.file.clone());
            let saved = out_dir.join(&file_name);

            if saved != downloaded {
                std::fs::copy(&downloaded, &saved)?;
            }

            if args.extract_text && file_name.ends_with(".parquet") {
                let txt_path = out_dir.join(format!("{}_text.txt", file_name.trim_end_matches(".parquet")));
                extract_parquet_text_column(&saved, &txt_path)?;
                println!("Extracted text written to {:?}", txt_path);
            } else {
                println!("Dataset file saved to {:?}", saved);
            }
        }
        Commands::PrepareDataset(args) => {
            let mut sources: Vec<PathBuf> = Vec::new();
            for repo in &args.dataset {
                let repo = repo.trim();
                if !repo.is_empty() {
                    let downloaded = download_dataset_from_hf(repo, &args.dataset_file)?;
                    sources.push(downloaded);
                }
            }
            if !args.file.is_empty() {
                for f in &args.file {
                    let f = f.trim();
                    if !f.is_empty() {
                        sources.push(PathBuf::from(f));
                    }
                }
            }

            let corpus_path = if sources.is_empty() {
                let default_file = Path::new("corpus.txt");
                if !default_file.exists() {
                    std::fs::write(default_file, DEFAULT_CORPUS)?;
                }
                default_file.to_path_buf()
            } else if !args.weights.is_empty() {
                if args.weights.len() != sources.len() {
                    return Err(format!(
                        "Weights count ({}) must match sources count ({})",
                        args.weights.len(),
                        sources.len()
                    )
                    .into());
                }
                if Path::new(&args.output) == Path::new("datasets/weighted_mix.txt") {
                    return Err(
                        "--output cannot be datasets/weighted_mix.txt: it is the intermediate \
                         text mix path and would be overwritten by the binary dataset."
                            .into(),
                    );
                }
                let weighted: Vec<(PathBuf, f32)> =
                    sources.iter().cloned().zip(args.weights.iter().copied()).collect();
                let max_chars = if args.max_chars == 0 { usize::MAX } else { args.max_chars };
                combine_and_mix_weighted(&weighted, Path::new("datasets/weighted_mix.txt"), max_chars)?
            } else if sources.len() == 1 {
                sources[0].clone()
            } else {
                combine_and_mix_datasets(&sources, Path::new("datasets/combined_corpus.txt"))?
            };

            let tokenizer = if Path::new(custom_tokenizer_file).exists() {
                LlmTokenizer::from_file(custom_tokenizer_file)?
            } else {
                println!("No custom tokenizer found. Training default tokenizer first...");
                train_custom_tokenizer("corpus.txt", custom_tokenizer_file, 8000)?;
                LlmTokenizer::from_file(custom_tokenizer_file)?
            };

            let max_tokens = if args.max_tokens == 0 { None } else { Some(args.max_tokens) };
            prepare_binary_dataset(&corpus_path, &args.output, &tokenizer, max_tokens)?;
        }
        Commands::Train(args) => {
            let mut sources: Vec<PathBuf> = Vec::new();
            let mut raw_text: Option<String> = None;

            for repo in &args.dataset {
                let repo = repo.trim();
                if !repo.is_empty() {
                    let downloaded = download_dataset_from_hf(repo, &args.dataset_file)?;
                    sources.push(downloaded);
                }
            }
            if let Some(file_path) = args.file {
                sources.push(PathBuf::from(file_path));
            }

            let dataset_path_buf: Option<PathBuf> = if sources.is_empty() {
                let mut found_dataset: Option<PathBuf> = None;
                if let Ok(entries) = std::fs::read_dir(".") {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        if name.starts_with("dataset_") && (name.ends_with(".txt") || name.ends_with(".bin")) {
                            found_dataset = Some(entry.path());
                            break;
                        }
                    }
                }
                if let Some(ds_path) = found_dataset {
                    tui::print_info(&format!("Auto-detected cached dataset file: {:?}", ds_path));
                    Some(ds_path)
                } else if Path::new("corpus.txt").exists() {
                    Some(PathBuf::from("corpus.txt"))
                } else {
                    raw_text = Some(DEFAULT_CORPUS.to_string());
                    None
                }
            } else if sources.len() == 1 {
                Some(sources[0].clone())
            } else {
                Some(combine_and_mix_datasets(&sources, Path::new("datasets/combined_corpus.txt"))?)
            };

            let tokenizer = if Path::new(custom_tokenizer_file).exists() {
                tui::print_info(&format!("Loading trained custom tokenizer from '{}'...", custom_tokenizer_file));
                LlmTokenizer::from_file(custom_tokenizer_file)?
            } else {
                tui::print_warning("No custom tokenizer found. Training custom tokenizer first...");
                train_custom_tokenizer("corpus.txt", custom_tokenizer_file, 8000)?;
                LlmTokenizer::from_file(custom_tokenizer_file)?
            };

            let vocab_size = tokenizer.vocab_size();
            tui::print_success(&format!("Tokenizer ready. Total vocabulary size: {}", tui::format_number(vocab_size)));

            let max_tokens = if args.max_tokens == 0 { None } else { Some(args.max_tokens) };

            let model_config = ModelConfig::new(
                vocab_size,
                args.n_embd,
                args.n_head,
                args.n_kv_head,
                args.n_layer,
                args.n_ctx,
                args.dropout,
                args.attn_softcap,
            );

            let save_dir_buf = args.save_dir.as_ref().map(PathBuf::from);
            let resume_path_buf = args.resume.as_ref().map(PathBuf::from);

            train::<Autodiff<Wgpu>>(
                dataset_path_buf.as_deref(),
                raw_text.as_deref(),
                model_config,
                &tokenizer,
                args.epochs,
                args.batch_size,
                args.lr,
                args.step,
                max_tokens,
                args.val_split,
                args.grad_accum,
                args.seq_warmup_steps,
                save_dir_buf.as_deref(),
                resume_path_buf.as_deref(),
                &device,
                output_dir,
            )?;
        }
        Commands::Eval(args) => {
            let mut sources: Vec<PathBuf> = Vec::new();
            for repo in &args.dataset {
                let repo = repo.trim();
                if !repo.is_empty() {
                    let downloaded = download_dataset_from_hf(repo, &args.dataset_file)?;
                    sources.push(downloaded);
                }
            }
            if let Some(file_path) = args.file {
                sources.push(PathBuf::from(file_path));
            }

            let dataset_path = if sources.is_empty() {
                if Path::new("datasets/dataset.bin").exists() {
                    PathBuf::from("datasets/dataset.bin")
                } else if Path::new("corpus.txt").exists() {
                    PathBuf::from("corpus.txt")
                } else {
                    return Err("No evaluation dataset specified (--file or --dataset).".into());
                }
            } else {
                sources[0].clone()
            };

            let tokenizer = if Path::new(custom_tokenizer_file).exists() {
                LlmTokenizer::from_file(custom_tokenizer_file)?
            } else {
                LlmTokenizer::from_hf("Qwen/Qwen2.5-0.5B")?
            };

            run_eval::<Wgpu>(
                Path::new(&args.model_dir),
                &dataset_path,
                &tokenizer,
                args.n_ctx,
                args.eval_batch_size,
                &device,
            )?;
        }
        Commands::Generate(args) => {
            let config_path = Path::new(output_dir).join("config.json");
            let weights_path = Path::new(output_dir).join("model_weights");

            if !config_path.exists() {
                println!("Error: Model files not found in '{}'. Run 'cargo run train' first.", output_dir);
                return Ok(());
            }

            let config_str = std::fs::read_to_string(config_path)?;
            let model_config: ModelConfig = serde_json::from_str(&config_str)?;

            let tokenizer = if Path::new(custom_tokenizer_file).exists() {
                LlmTokenizer::from_file(custom_tokenizer_file)?
            } else {
                LlmTokenizer::from_hf("Qwen/Qwen2.5-0.5B")?
            };

            let model: Llm<Wgpu> = model_config.init_model(&device);
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            let record = recorder
                .load(weights_path, &device)
                .map_err(|e| format!("Failed to load weights: {:?}", e))?;
            let model = model.load_record(record);

            let seed = args.seed.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(42)
            });

            let result = generate(
                &model,
                &tokenizer,
                &args.prompt,
                args.max_tokens,
                args.temp,
                args.top_k,
                args.top_p,
                args.min_p,
                args.rep_penalty,
                seed,
                &device,
            )?;

            println!("Generation Result:\n{}", result);
        }
        Commands::Export(args) => {
            let config_path = Path::new(output_dir).join("config.json");
            let weights_path = Path::new(output_dir).join("model_weights");

            if !config_path.exists() {
                println!("Error: Model files not found in '{}'. Run 'cargo run train' first.", output_dir);
                return Ok(());
            }

            let config_str = std::fs::read_to_string(config_path)?;
            let model_config: ModelConfig = serde_json::from_str(&config_str)?;

            let model: Llm<Wgpu> = model_config.init_model(&device);
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            let record = recorder
                .load(weights_path, &device)
                .map_err(|e| format!("Failed to load weights: {:?}", e))?;
            let model = model.load_record(record);

            export_to_gguf(&model, &model_config, custom_tokenizer_file, &args.model_name, "exported")?;
        }
        Commands::Stats => {
            let config_path = Path::new(output_dir).join("config.json");
            let model_config = if config_path.exists() {
                let config_str = std::fs::read_to_string(&config_path)?;
                serde_json::from_str::<ModelConfig>(&config_str)?
            } else {
                ModelConfig::new(8000, 256, 4, None, 4, 256, 0.1, 0.0)
            };

            let n_embd = model_config.n_embd;
            let n_head = model_config.n_head;
            let n_kv_head = model_config.get_n_kv_head();
            let head_dim = n_embd / n_head;
            let kv_dim = n_kv_head * head_dim;
            let n_layer = model_config.n_layer;
            let vocab_size = model_config.vocab_size;

            let raw_hidden = (8 * n_embd) / 3;
            let hidden_dim = raw_hidden.div_ceil(256) * 256;

            // Parameter Count Breakdown
            let embed_params = vocab_size * n_embd;
            let q_params = n_embd * n_embd + n_embd; // with bias
            let k_params = n_embd * kv_dim + kv_dim; // with bias
            let v_params = n_embd * kv_dim + kv_dim; // with bias
            let out_params = n_embd * n_embd;
            let attn_norm_params = n_embd;
            let ffn_norm_params = n_embd;
            let gate_params = n_embd * hidden_dim;
            let up_params = n_embd * hidden_dim;
            let down_params = hidden_dim * n_embd;

            let block_params = attn_norm_params + q_params + k_params + v_params + out_params + ffn_norm_params + gate_params + up_params + down_params;
            let final_norm_params = n_embd;
            let total_params = embed_params + (n_layer * block_params) + final_norm_params;

            let fp32_bytes = total_params * 4;
            let fp32_mb = fp32_bytes as f64 / (1024.0 * 1024.0);
            let flops_per_token = 2 * total_params;

            println!("===============================================================================");
            println!("🐈‍⬛ Kuro-Neko 2026 Model Architecture Statistics");
            println!("===============================================================================");
            println!("  Vocab Size       : {} tokens", vocab_size);
            println!("  Context Window   : {} tokens", model_config.n_ctx);
            println!("  Embedding Dim    : {}", n_embd);
            println!("  Hidden SwiGLU Dim: {} (aligned to 256)", hidden_dim);
            println!("  Query Heads      : {}", n_head);
            println!("  KV Heads (GQA)   : {} (Ratio 1:{})", n_kv_head, n_head / n_kv_head);
            println!("  Layers           : {}", n_layer);
            println!("  Softcapping Cap  : {}", if model_config.attn_softcap > 0.0 { format!("{:.1}", model_config.attn_softcap) } else { "Disabled".to_string() });
            println!("-------------------------------------------------------------------------------");
            println!("  Total Parameters : {} ({:.2} Million)", total_params, total_params as f64 / 1_000_000.0);
            println!("  Model Size (FP32): {:.2} MB", fp32_mb);
            println!("  FLOPs per Token  : {:.2} GFLOPs", flops_per_token as f64 / 1_000_000_000.0);
            println!("  GQA Cache Saving : {:.1}% KV-cache VRAM reduction", (1.0 - (n_kv_head as f64 / n_head as f64)) * 100.0);
            println!("===============================================================================");
        }
    }

    Ok(())
}
