use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use super::tokenizer::LlmTokenizer;

pub const BINARY_MAGIC: &[u8; 4] = b"KNKB";
/// v1: magic, version, vocab_size, token_count, tokens.
/// v2: adds a tokenizer vocabulary fingerprint after vocab_size.
pub const BINARY_VERSION: u32 = 2;

fn fnv1a64(bytes: &[u8], mut hash: u64) -> u64 {
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Deterministic fingerprint of a tokenizer vocabulary (`id -> token` pairs in
/// id order). Stored in `.bin` datasets so a tokenizer swap cannot silently
/// corrupt training data.
pub fn tokenizer_vocab_hash(tokenizer: &LlmTokenizer) -> u64 {
    let vocab = tokenizer.inner().get_vocab(true);
    let mut entries: Vec<(u32, &str)> = vocab
        .iter()
        .map(|(token, &id)| (id, token.as_str()))
        .collect();
    entries.sort_unstable_by_key(|(id, _)| *id);

    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash = fnv1a64(&(entries.len() as u64).to_le_bytes(), hash);
    for (id, token) in entries {
        hash = fnv1a64(&id.to_le_bytes(), hash);
        hash = fnv1a64(token.as_bytes(), hash);
    }
    hash
}

pub fn is_binary_file<P: AsRef<Path>>(
    path: P,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let path = path.as_ref();
    if let Some(ext) = path.extension()
        && ext.eq_ignore_ascii_case("bin")
    {
        return Ok(true);
    }
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_ok() && &magic == BINARY_MAGIC {
        return Ok(true);
    }
    Ok(false)
}

pub fn save_tokens_to_binary<P: AsRef<Path>>(
    path: P,
    tokens: &[u32],
    vocab_size: u32,
    vocab_hash: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = std::io::BufWriter::new(File::create(path)?);
    file.write_all(BINARY_MAGIC)?;
    file.write_all(&BINARY_VERSION.to_le_bytes())?;
    file.write_all(&vocab_size.to_le_bytes())?;
    file.write_all(&vocab_hash.to_le_bytes())?;
    file.write_all(&(tokens.len() as u64).to_le_bytes())?;

    for chunk in tokens.chunks(1024) {
        let mut bytes = Vec::with_capacity(chunk.len() * 4);
        for &t in chunk {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        file.write_all(&bytes)?;
    }
    file.flush()?;
    Ok(())
}

pub fn load_tokens_from_binary<P: AsRef<Path>>(
    path: P,
    max_tokens: Option<usize>,
    expected_vocab_hash: Option<u64>,
) -> Result<(Vec<u32>, u32), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != BINARY_MAGIC {
        return Err("Invalid binary dataset header: magic bytes mismatch".into());
    }

    let mut ver_buf = [0u8; 4];
    file.read_exact(&mut ver_buf)?;
    let version = u32::from_le_bytes(ver_buf);
    if version != 1 && version != BINARY_VERSION {
        return Err(format!("Unsupported binary dataset version: {}", version).into());
    }

    let mut vocab_buf = [0u8; 4];
    file.read_exact(&mut vocab_buf)?;
    let vocab_size = u32::from_le_bytes(vocab_buf);

    let stored_vocab_hash = if version >= 2 {
        let mut hash_buf = [0u8; 8];
        file.read_exact(&mut hash_buf)?;
        Some(u64::from_le_bytes(hash_buf))
    } else {
        None
    };

    match (expected_vocab_hash, stored_vocab_hash) {
        (Some(expected), Some(stored)) if expected != stored => {
            return Err(format!(
                "Tokenizer/dataset mismatch: the .bin dataset was tokenized with a different \
                 tokenizer (dataset fingerprint {stored:#018x}, current tokenizer {expected:#018x}). \
                 Re-run `prepare-dataset` with the current tokenizer."
            )
            .into());
        }
        (Some(_), None) => {
            eprintln!(
                "Warning: legacy v1 binary dataset has no tokenizer fingerprint; skipping validation."
            );
        }
        _ => {}
    }

    let mut count_buf = [0u8; 8];
    file.read_exact(&mut count_buf)?;
    let total_tokens = u64::from_le_bytes(count_buf) as usize;

    let target_tokens = match max_tokens {
        Some(limit) if limit > 0 => total_tokens.min(limit),
        _ => total_tokens,
    };

    let mut tokens = Vec::with_capacity(target_tokens);
    let mut buf = [0u8; 4 * 1024];

    while tokens.len() < target_tokens {
        let needed_tokens = target_tokens - tokens.len();
        let needed_bytes = (needed_tokens * 4).min(buf.len());
        let read_bytes = file.read(&mut buf[..needed_bytes])?;
        if read_bytes == 0 {
            break;
        }
        for chunk in buf[..read_bytes].chunks_exact(4) {
            tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }

    Ok((tokens, vocab_size))
}

pub fn prepare_binary_dataset<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_path: P1,
    output_path: P2,
    tokenizer: &LlmTokenizer,
    max_tokens: Option<usize>,
    dedupe: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!(
        "Preparing pre-tokenized binary dataset from {:?} to {:?}...",
        input_path.as_ref(),
        output_path.as_ref()
    );

    let is_parquet = input_path
        .as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("parquet"));

    let tokens = if is_parquet {
        let (tokens, docs, duplicates) =
            super::tokenizer::encode_parquet_streaming(input_path.as_ref(), tokenizer, max_tokens, dedupe)?;
        if dedupe {
            println!("Parquet ingest: {docs} documents, {duplicates} duplicate(s) skipped.");
        }
        tokens
    } else if dedupe {
        let (tokens, _bytes, duplicates) =
            tokenizer.encode_file_streaming_deduped(input_path.as_ref(), max_tokens)?;
        println!("Dropped {duplicates} duplicate document(s).");
        tokens
    } else {
        let (tokens, _bytes) = tokenizer.encode_file_streaming(input_path.as_ref(), max_tokens)?;
        tokens
    };

    let vocab_size = tokenizer.vocab_size() as u32;
    let vocab_hash = tokenizer_vocab_hash(tokenizer);
    save_tokens_to_binary(output_path.as_ref(), &tokens, vocab_size, vocab_hash)?;
    println!(
        "Saved binary dataset with {} tokens (vocab_size: {}) to {:?}",
        tokens.len(),
        vocab_size,
        output_path.as_ref()
    );
    Ok(())
}

/// Tokenize and mix multiple dataset sources (any combination of `.parquet`,
/// `.jsonl`, `.json`, `.txt`) directly into a `.bin` file in streaming batches
/// without materializing an intermediate `combined_corpus.txt` or
/// `weighted_mix.txt` on disk.
pub fn prepare_binary_dataset_from_sources<P: AsRef<Path>>(
    sources: &[(PathBuf, f32)],
    output_path: P,
    tokenizer: &LlmTokenizer,
    max_tokens: Option<usize>,
    max_chars: Option<usize>,
    dedupe: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if sources.len() == 1 && max_chars.is_none() {
        return prepare_binary_dataset(&sources[0].0, output_path, tokenizer, max_tokens, dedupe);
    }

    println!(
        "Preparing pre-tokenized binary dataset directly from {} sources into {:?}...",
        sources.len(),
        output_path.as_ref()
    );

    let (tokens, _bytes, duplicates) =
        tokenizer.encode_sources_streaming_opts(sources, max_tokens, max_chars, dedupe, None, None)?;
    if dedupe {
        println!("Dropped {duplicates} duplicate document(s).");
    }

    let vocab_size = tokenizer.vocab_size() as u32;
    let vocab_hash = tokenizer_vocab_hash(tokenizer);
    save_tokens_to_binary(output_path.as_ref(), &tokens, vocab_size, vocab_hash)?;
    println!(
        "Saved binary dataset with {} tokens (vocab_size: {}) to {:?}",
        tokens.len(),
        vocab_size,
        output_path.as_ref()
    );
    Ok(())
}
