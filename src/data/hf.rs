use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use hf_hub::{HFClientSync, split_id};

pub fn download_dataset_from_hf(
    repo_id: &str,
    filename: &str,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let mut candidates = vec![filename];
    if filename != "TinyStories-train.txt" {
        candidates.push("TinyStories-train.txt");
    }
    if filename != "stories_00.json" {
        candidates.push("stories_00.json");
    }
    if filename != "tinystories.jsonl" {
        candidates.push("tinystories.jsonl");
    }
    if filename != "spanish_stories.txt" {
        candidates.push("spanish_stories.txt");
    }
    if filename != "train.txt" {
        candidates.push("train.txt");
    }
    if filename != "data.txt" {
        candidates.push("data.txt");
    }

    // 1. Check local cache first for any candidate
    for cand in &candidates {
        let dataset_name = format!(
            "dataset_{}_{}",
            repo_id.replace('/', "_"),
            cand.replace('/', "_")
        );
        let root_cache = PathBuf::from(&dataset_name);
        if root_cache.exists() {
            println!("Using existing cached dataset file: {:?}", root_cache);
            return Ok(root_cache);
        }

        let datasets_dir = PathBuf::from("datasets");
        let local_cache = datasets_dir.join(&dataset_name);
        if local_cache.exists() {
            println!("Using cached dataset file: {:?}", local_cache);
            return Ok(local_cache);
        }
    }

    // 2. Download from HF Hub trying candidates in order
    let client = HFClientSync::new().map_err(|e| format!("HF Api error: {}", e))?;
    let (owner, name) = split_id(repo_id);
    let repo = client.dataset(owner, name);
    let datasets_dir = PathBuf::from("datasets");
    std::fs::create_dir_all(&datasets_dir)?;

    let mut last_err = None;
    for cand in candidates {
        let dataset_name = format!(
            "dataset_{}_{}",
            repo_id.replace('/', "_"),
            cand.replace('/', "_")
        );
        let local_cache = datasets_dir.join(&dataset_name);

        println!(
            "Downloading dataset '{}' ({}) from Hugging Face Hub...",
            repo_id, cand
        );
        match repo.download_file().filename(cand).send() {
            Ok(downloaded_file) => {
                std::fs::copy(&downloaded_file, &local_cache)?;
                println!("Saved dataset locally to {:?}", local_cache);
                return Ok(local_cache);
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }

    Err(format!(
        "Failed to download dataset from HF repo '{}': {:?}",
        repo_id,
        last_err.unwrap()
    )
    .into())
}

/// File names inside a Hugging Face dataset repo (`siblings[].rfilename`).
pub fn hf_repo_files(repo_id: &str) -> Result<Vec<String>, String> {
    let token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
    let client = reqwest::blocking::Client::builder()
        .user_agent("noctambulo-llm/0.1")
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!("https://huggingface.co/api/datasets/{repo_id}");
    let mut req = client.get(&url);
    if let Some(t) = &token {
        req = req.bearer_auth(t);
    }
    let resp = req.send().map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HF API status {}", resp.status()));
    }
    let body: serde_json::Value = resp.json().map_err(|e| e.to_string())?;
    Ok(body
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|f| {
                    f.get("rfilename")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default())
}

/// Best default training file out of a repo listing: real text data first,
/// then Parquet, then JSON/CSV. Prefers files with `train` in the name and
/// ignores hidden/README files.
pub fn pick_default_dataset_file(files: &[String]) -> Option<String> {
    let usable = |f: &&String| {
        let name = f.rsplit('/').next().unwrap_or(f);
        !name.starts_with('.')
            && !name.eq_ignore_ascii_case("readme.md")
            && !name.eq_ignore_ascii_case("config.json")
    };
    let pick = |exts: &[&str], prefer_train: bool| {
        files
            .iter()
            .filter(usable)
            .find(|f| {
                let lower = f.to_lowercase();
                exts.iter().any(|e| lower.ends_with(e))
                    && (!prefer_train || lower.contains("train"))
            })
            .cloned()
    };
    pick(&[".txt"], true)
        .or_else(|| pick(&[".parquet"], true))
        .or_else(|| pick(&[".jsonl", ".json"], true))
        .or_else(|| pick(&[".csv"], true))
        .or_else(|| pick(&[".txt"], false))
        .or_else(|| pick(&[".parquet"], false))
        .or_else(|| pick(&[".jsonl", ".json"], false))
        .or_else(|| pick(&[".csv"], false))
}

fn encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

/// Streaming HF download with byte-level progress and cooperative
/// cancellation. Keeps the same cache naming as `download_dataset_from_hf`, so
/// files already cached by the CLI are reused. Parquet files are kept natively
/// (no duplicate `_text.txt` dump on disk) because the document pipeline reads
/// `.parquet` directly.
pub fn download_dataset_streaming(
    repo_id: &str,
    filename: &str,
    progress: &dyn Fn(&str, u64, u64),
    cancelled: &dyn Fn() -> bool,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let dataset_name = format!(
        "dataset_{}_{}",
        repo_id.replace('/', "_"),
        filename.replace('/', "_")
    );

    // 1. Local cache first (root, then datasets/) to avoid re-downloading.
    for base in [PathBuf::from("."), PathBuf::from("datasets")] {
        let cached = base.join(&dataset_name);
        if cached.is_file() {
            return Ok(cached);
        }
    }

    let datasets_dir = Path::new("datasets");
    std::fs::create_dir_all(datasets_dir)?;
    let local_cache = datasets_dir.join(&dataset_name);

    let token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
    let client = reqwest::blocking::Client::builder()
        .user_agent("noctambulo-llm/0.1")
        .build()?;

    let encoded: Vec<String> = filename.split('/').map(encode_path_segment).collect();
    let url = format!(
        "https://huggingface.co/datasets/{}/resolve/main/{}",
        repo_id,
        encoded.join("/")
    );

    progress(&format!("requesting {filename}"), 0, 0);
    let mut req = client.get(&url);
    if let Some(t) = &token {
        req = req.bearer_auth(t);
    }
    let mut resp = req.send().map_err(|e| format!("{filename}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "Failed to download '{filename}' from '{repo_id}': HTTP {}",
            resp.status()
        )
        .into());
    }

    let total = resp.content_length().unwrap_or(0);
    let tmp = local_cache.with_extension("part");
    let mut file = std::io::BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
    let mut buf = vec![0u8; 1 << 16];
    let mut done = 0u64;
    loop {
        if cancelled() {
            drop(file);
            let _ = std::fs::remove_file(&tmp);
            return Err("cancelled by user".into());
        }
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        progress(&format!("downloading {filename}"), done, total);
    }
    file.flush()?;
    drop(file);
    std::fs::rename(&tmp, &local_cache)?;
    Ok(local_cache)
}
