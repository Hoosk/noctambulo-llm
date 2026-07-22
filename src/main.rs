mod config;
mod data;
mod export;
mod generate;
mod model;
mod train;

use config::ModelConfig;
use data::{combine_and_mix_datasets, download_dataset_from_hf, train_custom_tokenizer, LlmTokenizer};

use export::export_to_gguf;
use generate::generate;
use model::{LlmConfig, Llm};
use std::path::{Path, PathBuf};
use train::train;

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

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("==========================================================================");
        println!("                    🔥 Kuro-Neko 2026 Engine 🔥                          ");
        println!("==========================================================================");
        println!("Commands:");
        println!("  cargo run --release train-tokenizer [--dataset <repo>] [--file <path>] [--vocab-size <n>]");
        println!("  cargo run --release train [OPTIONS]");
        println!("      Options:");
        println!("        --dataset <repo>     HF dataset repo ID (e.g. roneneldan/TinyStories)");
        println!("        --file <path>        Local dataset file path");
        println!("        --batch-size <n>     Batch size (default: 128)");
        println!("        --epochs <n>         Number of epochs (default: 3)");
        println!("        --max-tokens <n>     Max dataset tokens to load (default: 10000000, 0=all)");

        println!("        --step <n>           Sliding window step in tokens (default: same as n-ctx)");
        println!("        --n-ctx <n>          Context window size (default: 64)");
        println!("        --n-embd <n>         Embedding dimension (default: 256)");
        println!("        --n-head <n>         Attention heads count (default: 4)");
        println!("        --n-layer <n>        Transformer layers count (default: 4)");
        println!("        --lr <float>         Learning rate (default: 0.001)");
        println!("  cargo run --release generate \"<prompt>\" [--temp <f>] [--top-k <n>] [--top-p <f>] [--rep-penalty <f>]");
        println!("  cargo run --release export [--outfile <path.gguf>]");
        println!("==========================================================================");
        return Ok(());
    }

    let mode = &args[1];

    let default_vocab_size = 8000;
    let mut n_embd = 256;
    let mut n_head = 4;
    let mut n_layer = 4;
    let mut n_ctx = 256;
    let mut epochs = 3;
    let mut batch_size = 128;
    let mut max_tokens: Option<usize> = Some(10_000_000);


    let mut custom_step: Option<usize> = None;
    let mut lr = 1e-3;

    let output_dir = "model_output";
    let custom_tokenizer_file = "custom_tokenizer.json";

    let device = WgpuDevice::default();

    match mode.as_str() {
        "train-tokenizer" => {
            let mut vocab_size = default_vocab_size;
            let mut sources: Vec<PathBuf> = Vec::new();

            let mut i = 2;
            while i < args.len() {
                if args[i] == "--vocab-size" && i + 1 < args.len() {
                    vocab_size = args[i + 1].parse().unwrap_or(default_vocab_size);
                    i += 2;
                } else if (args[i] == "--dataset" || args[i] == "--datasets") && i + 1 < args.len() {
                    for repo in args[i + 1].split(',') {
                        let repo = repo.trim();
                        if !repo.is_empty() {
                            let downloaded = download_dataset_from_hf(repo, "TinyStories-train.txt")?;
                            sources.push(downloaded);
                        }
                    }
                    i += 2;
                } else if args[i] == "--file" && i + 1 < args.len() {
                    sources.push(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    i += 1;
                }
            }

            let corpus_path = if sources.is_empty() {
                let default_file = Path::new("corpus.txt");
                if !default_file.exists() {
                    std::fs::write(default_file, DEFAULT_CORPUS)?;
                }
                default_file.to_string_lossy().to_string()
            } else if sources.len() == 1 {
                sources[0].to_string_lossy().to_string()
            } else {
                let combined = combine_and_mix_datasets(&sources, Path::new("datasets/combined_corpus.txt"))?;
                combined.to_string_lossy().to_string()
            };

            train_custom_tokenizer(&corpus_path, custom_tokenizer_file, vocab_size)?;
        }
        "train" => {
            let mut sources: Vec<PathBuf> = Vec::new();
            let mut raw_text: Option<String> = None;

            let mut i = 2;
            while i < args.len() {
                if (args[i] == "--dataset" || args[i] == "--datasets") && i + 1 < args.len() {
                    for repo in args[i + 1].split(',') {
                        let repo = repo.trim();
                        if !repo.is_empty() {
                            let downloaded = download_dataset_from_hf(repo, "TinyStories-train.txt")?;
                            sources.push(downloaded);
                        }
                    }
                    i += 2;
                } else if args[i] == "--file" && i + 1 < args.len() {
                    sources.push(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else if args[i] == "--batch-size" && i + 1 < args.len() {
                    batch_size = args[i + 1].parse().unwrap_or(128);
                    i += 2;
                } else if args[i] == "--epochs" && i + 1 < args.len() {
                    epochs = args[i + 1].parse().unwrap_or(3);
                    i += 2;
                } else if args[i] == "--max-tokens" && i + 1 < args.len() {
                    let val: usize = args[i + 1].parse().unwrap_or(50_000_000);
                    max_tokens = if val == 0 { None } else { Some(val) };
                    i += 2;
                } else if args[i] == "--step" && i + 1 < args.len() {
                    custom_step = Some(args[i + 1].parse().unwrap_or(64));
                    i += 2;
                } else if args[i] == "--n-ctx" && i + 1 < args.len() {
                    n_ctx = args[i + 1].parse().unwrap_or(256);
                    i += 2;
                } else if args[i] == "--n-embd" && i + 1 < args.len() {
                    n_embd = args[i + 1].parse().unwrap_or(256);
                    i += 2;
                } else if args[i] == "--n-head" && i + 1 < args.len() {
                    n_head = args[i + 1].parse().unwrap_or(4);
                    i += 2;
                } else if args[i] == "--n-layer" && i + 1 < args.len() {
                    n_layer = args[i + 1].parse().unwrap_or(4);
                    i += 2;
                } else if args[i] == "--lr" && i + 1 < args.len() {
                    lr = args[i + 1].parse().unwrap_or(1e-3);
                    i += 2;
                } else {
                    i += 1;
                }
            }

            let dataset_path_buf: Option<PathBuf> = if sources.is_empty() {
                if Path::new("corpus.txt").exists() {
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
                println!("Example: cargo run generate \"Once upon a time\" --temp 0.7 --top-k 40 --top-p 0.9");
                return Ok(());
            }

            let mut prompt = String::new();
            let mut temp = 0.7f32;
            let mut top_k = 40usize;
            let mut top_p = 0.9f32;
            let mut rep_penalty = 1.1f32;
            let mut max_new_tokens = 50usize;
            let mut user_seed: Option<u64> = None;

            let mut i = 2;
            while i < args.len() {
                if args[i] == "--temp" && i + 1 < args.len() {
                    temp = args[i + 1].parse().unwrap_or(0.7);
                    i += 2;
                } else if args[i] == "--top-k" && i + 1 < args.len() {
                    top_k = args[i + 1].parse().unwrap_or(40);
                    i += 2;
                } else if args[i] == "--top-p" && i + 1 < args.len() {
                    top_p = args[i + 1].parse().unwrap_or(0.9);
                    i += 2;
                } else if args[i] == "--rep-penalty" && i + 1 < args.len() {
                    rep_penalty = args[i + 1].parse().unwrap_or(1.1);
                    i += 2;
                } else if (args[i] == "--tokens" || args[i] == "--max-tokens" || args[i] == "--length") && i + 1 < args.len() {
                    max_new_tokens = args[i + 1].parse().unwrap_or(50);
                    i += 2;
                } else if args[i] == "--seed" && i + 1 < args.len() {
                    user_seed = args[i + 1].parse().ok();
                    i += 2;
                } else if prompt.is_empty() {
                    prompt = args[i].clone();
                    i += 1;
                } else {
                    i += 1;
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

            let seed = user_seed.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(42)
            });

            let result = generate(&model, &tokenizer, &prompt, max_new_tokens, temp, top_k, top_p, rep_penalty, seed, &device)?;

            println!("Generation Result:\n{}", result);

        }
        "export" => {
            let mut model_name = "kuro-neko-2026".to_string();
            for i in 2..args.len() {
                if (args[i] == "--model-name" || args[i] == "--outfile" || args[i] == "--name") && i + 1 < args.len() {
                    model_name = args[i + 1].clone();
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

            export_to_gguf(&model, &model_config, &model_name, custom_tokenizer_file)?;
        }

        _ => {
            println!("Unknown command: '{}'. Use 'train-tokenizer', 'train', 'generate', or 'export'.", mode);
        }
    }

    Ok(())
}
