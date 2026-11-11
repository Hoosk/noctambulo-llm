use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::binary::is_binary_file;
use super::document::{Document, DocumentStream};

/// Ensure none of the sources is a raw binary token file (`.bin`), which has
/// no document boundaries or UTF-8 text.
fn reject_binary_sources(
    sources: &[(PathBuf, f32)],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    for (src, _) in sources {
        let is_bin_ext = src
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("bin"));
        if is_bin_ext || is_binary_file(src).unwrap_or(false) {
            return Err(format!(
                "Refusing to mix binary dataset {:?}: weighted mixing reads documents and would produce an empty mix. Convert it to .txt first.",
                src
            )
            .into());
        }
    }
    Ok(())
}

/// Streaming document mixer across multiple dataset files (`.parquet`, `.jsonl`,
/// `.json`, or `.txt`). Yields complete [`Document`]s sampled probabilistically
/// according to source weights without writing any intermediate file to disk.
pub struct MultiSourceDocumentStream {
    streams: Vec<(PathBuf, DocumentStream)>,
    active: Vec<(usize, f32)>,
    max_chars: usize,
    emitted_chars: usize,
    total_bytes: u64,
    rng_state: u64,
}

impl MultiSourceDocumentStream {
    /// Open a weighted multi-source document stream.
    pub fn open(
        sources: &[(PathBuf, f32)],
        max_chars: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if sources.is_empty() {
            return Err("At least one dataset source is required".into());
        }
        reject_binary_sources(sources)?;

        let total_weight: f32 = sources.iter().map(|(_, w)| w.max(0.0)).sum();
        if total_weight <= 0.0 {
            return Err("Total weight must be > 0".into());
        }

        let mut streams = Vec::with_capacity(sources.len());
        let mut total_bytes = 0u64;
        for (src, _) in sources {
            let stream = DocumentStream::open(src)?;
            total_bytes = total_bytes.saturating_add(stream.total_bytes());
            streams.push((src.clone(), stream));
        }

        let active: Vec<(usize, f32)> = sources
            .iter()
            .enumerate()
            .filter_map(|(i, (_, w))| (*w > 0.0).then_some((i, *w)))
            .collect();

        let rng_state = 0x9E37_79B9_7F4A_7C15u64
            ^ std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xDEAD_BEEF);

        Ok(Self {
            streams,
            active,
            max_chars: max_chars.filter(|&m| m > 0).unwrap_or(usize::MAX),
            emitted_chars: 0,
            total_bytes,
            rng_state: if rng_state == 0 { 0xDEAD_BEEF } else { rng_state },
        })
    }

    /// Open an unweighted multi-source stream where each source is weighted
    /// proportionally to its byte size so all sources remain mixed evenly from
    /// start to end (preventing `val_split` at the tail from containing only the
    /// largest dataset).
    #[allow(dead_code)]
    pub fn open_proportional(
        sources: &[PathBuf],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weighted: Vec<(PathBuf, f32)> = sources
            .iter()
            .map(|p| {
                let bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(1).max(1);
                (p.clone(), bytes as f32)
            })
            .collect();
        Self::open(&weighted, None)
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    pub fn active_sources(&self) -> usize {
        self.active.len()
    }

    pub fn emitted_chars(&self) -> usize {
        self.emitted_chars
    }

    fn next_rand_unit(&mut self) -> f32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        let r = self.rng_state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33;
        r as f32 / (1u64 << 32) as f32
    }

    pub fn next_document(
        &mut self,
    ) -> Result<Option<Document>, Box<dyn std::error::Error + Send + Sync>> {
        while !self.active.is_empty() && self.emitted_chars < self.max_chars {
            let total: f32 = self.active.iter().map(|(_, w)| *w).sum();
            let r = self.next_rand_unit() * total;
            let mut acc = 0.0f32;
            let mut pick = 0usize;
            for (pos, (_, w)) in self.active.iter().enumerate() {
                acc += *w;
                if r < acc {
                    pick = pos;
                    break;
                }
            }
            let idx = self.active[pick].0;

            match self.streams[idx].1.next_document() {
                Ok(Some(doc)) => {
                    let doc_len = doc.text.len();
                    if self.emitted_chars + doc_len > self.max_chars && self.emitted_chars > 0 {
                        self.active.remove(pick);
                        continue;
                    }
                    self.emitted_chars += doc_len + 1;
                    return Ok(Some(doc));
                }
                Ok(None) => {
                    self.active.remove(pick);
                }
                Err(e) => {
                    println!(
                        "Warning: error reading from dataset {:?}: {}",
                        self.streams[idx].0, e
                    );
                    self.active.remove(pick);
                }
            }
        }
        Ok(None)
    }
}

/// Write a document to a combined `.txt` file. If the document spans multiple
/// lines (or `use_eot_delimiter` is set), appends `\n<|endoftext|>\n` so
/// subsequent passes preserve exact multi-paragraph document boundaries.
fn write_mixed_document(
    writer: &mut std::io::BufWriter<File>,
    text: &str,
    use_eot_delimiter: &mut bool,
) -> std::io::Result<usize> {
    if text.contains('\n') {
        *use_eot_delimiter = true;
    }
    let bytes = text.as_bytes();
    writer.write_all(bytes)?;
    if *use_eot_delimiter {
        writer.write_all(b"\n<|endoftext|>\n")?;
        Ok(bytes.len() + 14)
    } else {
        writer.write_all(b"\n")?;
        Ok(bytes.len() + 1)
    }
}

/// Combine and mix multiple dataset sources at the **document** level into
/// `output_path`, weighting each source proportionally to its size so the mix
/// stays balanced from start to finish.
pub fn combine_and_mix_datasets(
    sources: &[PathBuf],
    output_path: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    if sources.len() == 1 {
        return Ok(sources[0].clone());
    }

    let weighted: Vec<(PathBuf, f32)> = sources
        .iter()
        .map(|p| {
            let bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(1).max(1);
            (p.clone(), bytes as f32)
        })
        .collect();

    combine_and_mix_weighted(&weighted, output_path, usize::MAX)
}

/// Weighted dataset mix at the document level honoring `weights` proportions
/// and a maximum character budget `max_chars`.
pub fn combine_and_mix_weighted(
    sources: &[(PathBuf, f32)],
    output_path: &Path,
    max_chars: usize,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    combine_and_mix_weighted_with(sources, output_path, max_chars, &|_, _| {}, &|| false)
}

/// Same as [`combine_and_mix_weighted`] with progress reporting (characters
/// written) and cooperative cancellation, used by the web server.
pub fn combine_and_mix_weighted_with(
    sources: &[(PathBuf, f32)],
    output_path: &Path,
    max_chars: usize,
    progress: &dyn Fn(u64, u64),
    cancelled: &dyn Fn() -> bool,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    reject_binary_sources(sources)?;

    let total_weight: f32 = sources.iter().map(|(_, w)| w.max(0.0)).sum();
    if total_weight <= 0.0 {
        return Err("Total weight must be > 0".into());
    }

    let mut mixer = MultiSourceDocumentStream::open(
        sources,
        if max_chars == usize::MAX {
            None
        } else {
            Some(max_chars)
        },
    )?;

    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    println!(
        "Combining {} datasets by document with weights {:?} (max {} chars) into {:?}...",
        sources.len(),
        sources.iter().map(|(_, w)| w).collect::<Vec<_>>(),
        max_chars,
        output_path
    );

    let out_file = File::create(output_path)?;
    let mut writer = std::io::BufWriter::with_capacity(2 * 1024 * 1024, out_file);

    let mut total_docs = 0u64;
    let mut last_pct = -1i32;
    let mut iterations: u64 = 0;
    let mut last_report = std::time::Instant::now();
    let progress_total: u64 = if max_chars == usize::MAX {
        0
    } else {
        max_chars as u64
    };
    let mut use_eot_delimiter = false;

    while let Some(doc) = mixer.next_document()? {
        iterations += 1;
        if iterations.is_multiple_of(1024) && cancelled() {
            writer.flush().ok();
            drop(writer);
            let _ = std::fs::remove_file(output_path);
            return Err("cancelled by user".into());
        }

        write_mixed_document(&mut writer, &doc.text, &mut use_eot_delimiter)?;
        total_docs += 1;
        let written_chars = mixer.emitted_chars();

        if last_report.elapsed() >= std::time::Duration::from_millis(200) {
            last_report = std::time::Instant::now();
            progress(written_chars as u64, progress_total);
        }

        let pct = if max_chars == usize::MAX {
            0
        } else {
            (written_chars * 100 / max_chars) as i32
        };
        if pct != last_pct && (pct % 2 == 0 || pct == 100) {
            last_pct = pct;
            print!(
                "\r  Progress: {:>3}%  ({} docs, {:.1} MB written, {} active sources)",
                pct,
                total_docs,
                written_chars as f64 / 1_048_576.0,
                mixer.active_sources()
            );
            let _ = std::io::stdout().flush();
        }
    }

    writer.flush()?;
    println!();
    let written_chars = mixer.emitted_chars();
    let final_total = if progress_total > 0 {
        progress_total
    } else {
        written_chars as u64
    };
    progress(written_chars as u64, final_total);
    println!(
        "Document mix ready: {} documents / {} chars written to {:?}.",
        total_docs, written_chars, output_path
    );
    Ok(output_path.to_path_buf())
}
