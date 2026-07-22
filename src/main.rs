mod config;
mod data;
mod export;
mod generate;
mod model;
mod train;

use config::ModelConfig;
use data::{download_dataset_from_hf, train_custom_tokenizer, LlmTokenizer};
use export::export_to_gguf;
use generate::generate;
use model::{LlmConfig, Llm};
use std::path::{Path, PathBuf};
use train::train;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::backend::Autodiff;
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

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("==========================================================================");
        println!("                       🔥 My-Rust-LLM Engine 🔥                          ");
        println!("==========================================================================");
        println!("Commands:");
        println!("  cargo run --release train-tokenizer [--dataset <repo>] [--file <path>] [--vocab-size <n>]");
        println!("  cargo run --release train [OPTIONS]");
        println!("      Options:");
        println!("        --dataset <repo>     HF dataset repo ID (e.g. roneneldan/TinyStories)");
        println!("        --file <path>        Local dataset file path");
        println!("        --batch-size <n>     Batch size (default: 128)");
        println!("        --epochs <n>         Number of epochs (default: 3)");
        println!("        --max-tokens <n>     Max dataset tokens to load (default: 50000000, 0=all)");
        println!("        --step <n>           Sliding window step in tokens (default: same as n-ctx)");
        println!("        --n-ctx <n>          Context window size (default: 64)");
        println!("        --n-embd <n>         Embedding dimension (default: 256)");
        println!("        --n-head <n>         Attention heads count (default: 4)");
        println!("        --n-layer <n>        Transformer layers count (default: 4)");
        println!("        --lr <float>         Learning rate (default: 0.001)");
        println!("  cargo run --release generate \"<prompt>\"");
        println!("  cargo run --release export [--outfile <path.gguf>]");
        println!("==========================================================================");
        return Ok(());
    }

    let mode = &args[1];

    let default_vocab_size = 8000;
    let mut n_embd = 256;
    let mut n_head = 4;
    let mut n_layer = 4;
    let mut n_ctx = 64;
    let mut epochs = 3;
    let mut batch_size = 128;
    let mut max_tokens: Option<usize> = Some(50_000_000); // 50M tokens is ideal for a 7M model
    let mut custom_step: Option<usize> = None;
    let mut lr = 1e-3;

    let output_dir = "model_output";
    let custom_tokenizer_file = "custom_tokenizer.json";

    let device = WgpuDevice::default();

    match mode.as_str() {
        "train-tokenizer" => {
            let mut vocab_size = default_vocab_size;
            let mut corpus_path = String::new();

            for i in 2..args.len() {
                if args[i] == "--vocab-size" && i + 1 < args.len() {
                    vocab_size = args[i + 1].parse().unwrap_or(default_vocab_size);
                } else if args[i] == "--dataset" && i + 1 < args.len() {
                    let repo = &args[i + 1];
                    let downloaded = download_dataset_from_hf(repo, "TinyStories-train.txt")?;
                    corpus_path = downloaded.to_string_lossy().to_string();
                } else if args[i] == "--file" && i + 1 < args.len() {
                    corpus_path = args[i + 1].clone();
                }
            }

            if corpus_path.is_empty() {
                let default_file = Path::new("corpus.txt");
                if !default_file.exists() {
                    std::fs::write(default_file, DEFAULT_CORPUS)?;
                }
                corpus_path = default_file.to_string_lossy().to_string();
            }

            train_custom_tokenizer(&corpus_path, custom_tokenizer_file, vocab_size)?;
        }
        "train" => {
            let mut dataset_path_buf: Option<PathBuf> = None;
            let mut raw_text: Option<String> = None;

            for i in 2..args.len() {
                if args[i] == "--dataset" && i + 1 < args.len() {
                    let repo = &args[i + 1];
                    let downloaded = download_dataset_from_hf(repo, "TinyStories-train.txt")?;
                    dataset_path_buf = Some(downloaded);
                } else if args[i] == "--file" && i + 1 < args.len() {
                    dataset_path_buf = Some(PathBuf::from(&args[i + 1]));
                } else if args[i] == "--batch-size" && i + 1 < args.len() {
                    batch_size = args[i + 1].parse().unwrap_or(128);
                } else if args[i] == "--epochs" && i + 1 < args.len() {
                    epochs = args[i + 1].parse().unwrap_or(3);
                } else if args[i] == "--max-tokens" && i + 1 < args.len() {
                    let val: usize = args[i + 1].parse().unwrap_or(50_000_000);
                    max_tokens = if val == 0 { None } else { Some(val) };
                } else if args[i] == "--step" && i + 1 < args.len() {
                    custom_step = Some(args[i + 1].parse().unwrap_or(64));
                } else if args[i] == "--n-ctx" && i + 1 < args.len() {
                    n_ctx = args[i + 1].parse().unwrap_or(64);
                } else if args[i] == "--n-embd" && i + 1 < args.len() {
                    n_embd = args[i + 1].parse().unwrap_or(256);
                } else if args[i] == "--n-head" && i + 1 < args.len() {
                    n_head = args[i + 1].parse().unwrap_or(4);
                } else if args[i] == "--n-layer" && i + 1 < args.len() {
                    n_layer = args[i + 1].parse().unwrap_or(4);
                } else if args[i] == "--lr" && i + 1 < args.len() {
                    lr = args[i + 1].parse().unwrap_or(1e-3);
                }
            }

            if dataset_path_buf.is_none() {
                if Path::new("corpus.txt").exists() {
                    dataset_path_buf = Some(PathBuf::from("corpus.txt"));
                } else {
                    raw_text = Some(DEFAULT_CORPUS.to_string());
                }
            }

            let tokenizer = if Path::new(custom_tokenizer_file).exists() {
                println!("Loading trained custom tokenizer from '{}'...", custom_tokenizer_file);
                LlmTokenizer::from_file(custom_tokenizer_file)?
            } else {
                println!("No custom tokenizer found. Training custom tokenizer first...");
                train_custom_tokenizer("corpus.txt", custom_tokenizer_file, default_vocab_size)?;
                LlmTokenizer::from_file(custom_tokenizer_file)?
            };

            let vocab_size = tokenizer.vocab_size();
            println!("Tokenizer ready. Total vocabulary size: {}", vocab_size);

            let model_config = ModelConfig::new(vocab_size, n_embd, n_head, n_layer, n_ctx);
            train::<Autodiff<Wgpu>>(
                dataset_path_buf.as_deref(),
                raw_text.as_deref(),
                model_config,
                &tokenizer,
                epochs,
                batch_size,
                lr,
                custom_step,
                max_tokens,
                &device,
                output_dir,
            )?;
        }
        "generate" => {
            if args.len() < 3 {
                println!("Error: Please provide a prompt for completion.");
                println!("Example: cargo run generate \"Once upon a time\"");
                return Ok(());
            }
            let prompt = &args[2];

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

            let llm_config = LlmConfig {
                vocab_size: model_config.vocab_size,
                n_embd: model_config.n_embd,
                n_head: model_config.n_head,
                n_layer: model_config.n_layer,
                n_ctx: model_config.n_ctx,
            };

            let model: Llm<Wgpu> = llm_config.init(&device);
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            let record = recorder.load(weights_path, &device)
                .map_err(|e| format!("Failed to load weights: {:?}", e))?;
            let model = model.load_record(record);

            let result = generate(&model, &tokenizer, prompt, 50, 0.7, 42, &device)?;
            println!("Generation Result:\n{}", result);
        }
        "export" => {
            let mut gguf_path = format!("{}/model.gguf", output_dir);
            for i in 2..args.len() {
                if args[i] == "--outfile" && i + 1 < args.len() {
                    gguf_path = args[i + 1].clone();
                }
            }

            let config_path = Path::new(output_dir).join("config.json");
            let weights_path = Path::new(output_dir).join("model_weights");

            if !config_path.exists() {
                println!("Error: Model files not found in '{}'. Run 'cargo run train' first.", output_dir);
                return Ok(());
            }

            let config_str = std::fs::read_to_string(config_path)?;
            let model_config: ModelConfig = serde_json::from_str(&config_str)?;

            let llm_config = LlmConfig {
                vocab_size: model_config.vocab_size,
                n_embd: model_config.n_embd,
                n_head: model_config.n_head,
                n_layer: model_config.n_layer,
                n_ctx: model_config.n_ctx,
            };

            let model: Llm<Wgpu> = llm_config.init(&device);
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            let record = recorder.load(weights_path, &device)
                .map_err(|e| format!("Failed to load weights: {:?}", e))?;
            let model = model.load_record(record);

            export_to_gguf(&model, &model_config, &gguf_path)?;
        }
        _ => {
            println!("Unknown command: '{}'. Use 'train-tokenizer', 'train', 'generate', or 'export'.", mode);
        }
    }

    Ok(())
}
