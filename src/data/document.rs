use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use arrow::array::{Array, LargeStringArray, StringArray};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

/// Standard document separator used in multi-line text corpora (e.g. TinyStories
/// and extracted Parquet text dumps).
pub const EOT_MARKER: &str = "<|endoftext|>";

/// Upper bound on a single document chunk passed to `encode_batch` so giant
/// single-line dumps cannot stall progress reporting or cancellation.
pub const MAX_DOC_CHUNK_BYTES: usize = 256 * 1024;

/// A single logical document yielded by [`DocumentStream`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub text: String,
    /// Approximate or exact source file bytes consumed to produce this document.
    pub bytes_consumed: u64,
}

/// Detected on-disk format of a text/structured dataset file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    /// Apache Parquet column store (`.parquet`).
    Parquet,
    /// Newline-delimited JSON objects (`.jsonl` or `.txt` with `{...}` lines).
    Jsonl,
    /// Top-level JSON array of objects (`[{...}, {...}]`, e.g. `.json`).
    JsonArray,
    /// Plain text where multi-line documents are separated by `<|endoftext|>`.
    DelimitedText,
    /// Plain text where each non-empty line is a standalone document.
    LineText,
}

/// Inspect the file extension and first 64 KB to determine how documents are
/// structured inside `path`.
pub fn detect_format(path: &Path) -> std::io::Result<DocumentFormat> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if ext == "parquet" {
        return Ok(DocumentFormat::Parquet);
    }
    if ext == "jsonl" {
        return Ok(DocumentFormat::Jsonl);
    }

    let mut file = File::open(path)?;
    let mut peek = vec![0u8; 64 * 1024];
    let n = file.read(&mut peek)?;
    peek.truncate(n);

    let sample = String::from_utf8_lossy(&peek);
    let trimmed = sample.trim_start();

    if trimmed.starts_with('[') {
        return Ok(DocumentFormat::JsonArray);
    }
    if ext == "json" && trimmed.starts_with('{') {
        return Ok(DocumentFormat::Jsonl);
    }
    if sample.contains(EOT_MARKER) {
        return Ok(DocumentFormat::DelimitedText);
    }

    Ok(DocumentFormat::LineText)
}

/// Pick the text column out of a Parquet schema. Shared by extraction and
/// direct streaming tokenization.
pub fn parquet_text_column(
    schema: &arrow::datatypes::Schema,
    parquet_path: &Path,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    for candidate in ["text", "content", "body", "raw", "story", "story_text"] {
        if schema.fields().iter().any(|f| f.name() == candidate) {
            return Ok(candidate.to_string());
        }
    }
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    Err(format!(
        "No text column found in parquet {:?}. Columns available: {:?}",
        parquet_path, names
    )
    .into())
}

/// Extract clean natural-language text from a parsed JSON document object.
///
/// Supports:
/// - Direct text keys: `story`, `text`, `story_text`, `content`, `body`
/// - Instruction / QA records (e.g. Dolly, Alpaca): `instruction`/`prompt`/`question`
///   + optional `context`/`input` + `response`/`output`/`answer`
/// - Chat turn arrays: `messages` or `conversations`
pub fn extract_json_document(v: &serde_json::Value) -> Option<String> {
    for key in ["story", "text", "story_text", "content", "body"] {
        if let Some(s) = v.get(key).and_then(|s| s.as_str()) {
            let t = s.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }

    // Instruction / context / response triples (e.g. databricks-dolly-15k)
    let instruction = v
        .get("instruction")
        .or_else(|| v.get("prompt"))
        .or_else(|| v.get("question"))
        .and_then(|s| s.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let context = v
        .get("context")
        .or_else(|| v.get("input"))
        .and_then(|s| s.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let response = v
        .get("response")
        .or_else(|| v.get("output"))
        .or_else(|| v.get("answer"))
        .and_then(|s| s.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    if instruction.is_some() || response.is_some() {
        let mut parts = Vec::with_capacity(3);
        if let Some(inst) = instruction {
            parts.push(inst);
        }
        if let Some(ctx) = context {
            parts.push(ctx);
        }
        if let Some(resp) = response {
            parts.push(resp);
        }
        if !parts.is_empty() {
            return Some(parts.join("\n\n"));
        }
    }

    // Chat-style messages array
    for key in ["messages", "conversations"] {
        if let Some(arr) = v.get(key).and_then(|a| a.as_array()) {
            let turns: Vec<&str> = arr
                .iter()
                .filter_map(|item| {
                    item.get("content")
                        .or_else(|| item.get("value"))
                        .and_then(|s| s.as_str())
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                })
                .collect();
            if !turns.is_empty() {
                return Some(turns.join("\n"));
            }
        }
    }

    None
}

/// Parse a line that may be a JSON object `{...}`, falling back to the raw line
/// when it is plain text.
pub fn parse_line_or_json(trimmed: &str) -> String {
    if trimmed.starts_with('{')
        && trimmed.ends_with('}')
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed)
        && let Some(extracted) = extract_json_document(&v)
    {
        return extracted;
    }
    trimmed.to_string()
}

/// Split a string into owned UTF-8-safe pieces of at most `max_bytes`. Long
/// documents (concatenated dumps can be hundreds of MB) would otherwise reach
/// `encode_batch` whole, freezing progress and cooperative cancellation.
pub fn split_utf8_chunks(s: String, max_bytes: usize) -> Vec<String> {
    if s.len() <= max_bytes {
        return vec![s];
    }
    let mut out = Vec::with_capacity(s.len() / max_bytes + 1);
    let mut start = 0usize;
    while start < s.len() {
        let mut end = (start + max_bytes).min(s.len());
        while end < s.len() && !s.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            end = s.len();
        }
        out.push(s[start..end].to_string());
        start = end;
    }
    out
}

enum ReaderBackend {
    Parquet {
        reader: ParquetRecordBatchReader,
        text_col: String,
    },
    DelimitedText {
        reader: BufReader<File>,
        line_buf: String,
        doc_buf: String,
         pending_bytes: u64,
        eof: bool,
    },
    LineOrJsonl {
        reader: BufReader<File>,
        line_buf: String,
    },
    JsonArray {
        reader: BufReader<File>,
        byte_buf: Vec<u8>,
        buf_pos: usize,
        buf_len: usize,
        obj_buf: Vec<u8>,
        depth: usize,
        in_string: bool,
        escaped: bool,
        eof: bool,
    },
}

/// Streaming iterator yielding complete [`Document`]s from `.parquet`, `.jsonl`,
/// `.json` arrays, `<|endoftext|>`-delimited `.txt`, or line-based `.txt`.
pub struct DocumentStream {
    backend: ReaderBackend,
    queue: VecDeque<Document>,
    total_bytes: u64,
}

impl DocumentStream {
    pub fn open<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();
        let total_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let format = detect_format(path)
            .map_err(|e| format!("Failed to inspect dataset {:?}: {}", path, e))?;

        let backend = match format {
            DocumentFormat::Parquet => {
                let file = File::open(path)
                    .map_err(|e| format!("Failed to open parquet file {:?}: {}", path, e))?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                    .map_err(|e| format!("Failed to parse parquet {:?}: {}", path, e))?;
                let text_col = parquet_text_column(&builder.schema().clone(), path)?;
                let reader = builder
                    .build()
                    .map_err(|e| format!("Failed to build parquet reader: {}", e))?;
                ReaderBackend::Parquet { reader, text_col }
            }
            DocumentFormat::DelimitedText => {
                let file = File::open(path)
                    .map_err(|e| format!("Failed to open dataset file {:?}: {}", path, e))?;
                ReaderBackend::DelimitedText {
                    reader: BufReader::with_capacity(2 * 1024 * 1024, file),
                    line_buf: String::new(),
                    doc_buf: String::new(),
                    pending_bytes: 0,
                    eof: false,
                }
            }
            DocumentFormat::Jsonl | DocumentFormat::LineText => {
                let file = File::open(path)
                    .map_err(|e| format!("Failed to open dataset file {:?}: {}", path, e))?;
                ReaderBackend::LineOrJsonl {
                    reader: BufReader::with_capacity(2 * 1024 * 1024, file),
                    line_buf: String::new(),
                }
            }
            DocumentFormat::JsonArray => {
                let file = File::open(path)
                    .map_err(|e| format!("Failed to open JSON dataset {:?}: {}", path, e))?;
                ReaderBackend::JsonArray {
                    reader: BufReader::with_capacity(2 * 1024 * 1024, file),
                    byte_buf: vec![0u8; 64 * 1024],
                    buf_pos: 0,
                    buf_len: 0,
                    obj_buf: Vec::with_capacity(4096),
                    depth: 0,
                    in_string: false,
                    escaped: false,
                    eof: false,
                }
            }
        };

        Ok(Self {
            backend,
            queue: VecDeque::new(),
            total_bytes,
        })
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    fn enqueue_document(&mut self, text: String, bytes_consumed: u64) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        let chunks = split_utf8_chunks(trimmed.to_string(), MAX_DOC_CHUNK_BYTES);
        let count = chunks.len() as u64;
        for (i, chunk) in chunks.into_iter().enumerate() {
            let b = if i + 1 == count as usize {
                bytes_consumed.saturating_sub((bytes_consumed / count) * (count - 1))
            } else {
                bytes_consumed / count
            };
            self.queue.push_back(Document {
                text: chunk,
                bytes_consumed: b,
            });
        }
    }

    /// Read the next complete document from the underlying file.
    pub fn next_document(
        &mut self,
    ) -> Result<Option<Document>, Box<dyn std::error::Error + Send + Sync>> {
        while self.queue.is_empty() {
            match &mut self.backend {
                ReaderBackend::Parquet { reader, text_col } => {
                    let Some(batch_res) = reader.next() else {
                        return Ok(None);
                    };
                    let record = batch_res.map_err(|e| format!("Parquet read error: {e}"))?;
                    let Some(col) = record.column_by_name(text_col) else {
                        continue;
                    };
                    let values: Vec<&str> = col
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .map(|a| a.iter().flatten().collect())
                        .or_else(|| {
                            col.as_any()
                                .downcast_ref::<LargeStringArray>()
                                .map(|a| a.iter().flatten().collect())
                        })
                        .unwrap_or_default();

                    let mut docs_to_push = Vec::with_capacity(values.len());
                    for value in values {
                        let t = value.trim();
                        if !t.is_empty() {
                            docs_to_push.push((t.to_string(), t.len() as u64));
                        }
                    }
                    for (doc, bytes) in docs_to_push {
                        self.enqueue_document(doc, bytes);
                    }
                }
                ReaderBackend::DelimitedText {
                    reader,
                    line_buf,
                    doc_buf,
                    pending_bytes,
                    eof,
                } => {
                    if *eof {
                        if !doc_buf.trim().is_empty() {
                            let finished = std::mem::take(doc_buf);
                            let bytes = std::mem::replace(pending_bytes, 0);
                            self.enqueue_document(finished, bytes);
                            break;
                        }
                        return Ok(None);
                    }

                    line_buf.clear();
                    let n = reader.read_line(line_buf)?;
                    if n == 0 {
                        *eof = true;
                        if !doc_buf.trim().is_empty() {
                            let finished = std::mem::take(doc_buf);
                            let bytes = std::mem::replace(pending_bytes, 0);
                            self.enqueue_document(finished, bytes);
                            break;
                        }
                        return Ok(None);
                    }

                    *pending_bytes += n as u64;
                    let line_no_newline = line_buf.trim_end_matches(['\r', '\n']);

                    if line_no_newline.contains(EOT_MARKER) {
                        let mut parts = line_no_newline.split(EOT_MARKER);
                        let mut completed_docs: Vec<(String, u64)> = Vec::new();
                        if let Some(first) = parts.next() {
                            let first_trimmed = first.trim();
                            if !first_trimmed.is_empty() {
                                if !doc_buf.is_empty() {
                                    doc_buf.push('\n');
                                }
                                doc_buf.push_str(first_trimmed);
                            }
                        }
                        for part in parts {
                            if !doc_buf.trim().is_empty() {
                                let finished = std::mem::take(doc_buf);
                                let bytes = std::mem::replace(pending_bytes, 0);
                                completed_docs.push((finished, bytes));
                            } else {
                                *pending_bytes = 0;
                            }
                            let part_trimmed = part.trim();
                            if !part_trimmed.is_empty() {
                                doc_buf.push_str(part_trimmed);
                            }
                        }
                        for (doc, bytes) in completed_docs {
                            self.enqueue_document(doc, bytes);
                        }
                    } else {
                        let trimmed = line_no_newline.trim();
                        if !trimmed.is_empty() {
                            if !doc_buf.is_empty() {
                                doc_buf.push('\n');
                            }
                            doc_buf.push_str(trimmed);
                        }
                    }
                }
                ReaderBackend::LineOrJsonl { reader, line_buf } => {
                    line_buf.clear();
                    let n = reader.read_line(line_buf)?;
                    if n == 0 {
                        return Ok(None);
                    }
                    let trimmed = line_buf.trim();
                    if !trimmed.is_empty() {
                        let doc = parse_line_or_json(trimmed);
                        self.enqueue_document(doc, n as u64);
                    }
                }
                ReaderBackend::JsonArray {
                    reader,
                    byte_buf,
                    buf_pos,
                    buf_len,
                    obj_buf,
                    depth,
                    in_string,
                    escaped,
                    eof,
                } => {
                    if *eof {
                        return Ok(None);
                    }
                    let mut raw_bytes_scanned = 0u64;
                    let mut extracted_doc: Option<String> = None;

                    while extracted_doc.is_none() {
                        if *buf_pos >= *buf_len {
                            *buf_len = reader.read(byte_buf)?;
                            *buf_pos = 0;
                            if *buf_len == 0 {
                                *eof = true;
                                break;
                            }
                        }

                        while *buf_pos < *buf_len {
                            let b = byte_buf[*buf_pos];
                            *buf_pos += 1;
                            raw_bytes_scanned += 1;

                            if *depth == 0 {
                                if b == b'{' {
                                    *depth = 1;
                                    *in_string = false;
                                    *escaped = false;
                                    obj_buf.clear();
                                    obj_buf.push(b);
                                }
                                continue;
                            }

                            obj_buf.push(b);
                            if *in_string {
                                if *escaped {
                                    *escaped = false;
                                } else if b == b'\\' {
                                    *escaped = true;
                                } else if b == b'"' {
                                    *in_string = false;
                                }
                            } else {
                                match b {
                                    b'"' => *in_string = true,
                                    b'{' => *depth += 1,
                                    b'}' => {
                                        *depth -= 1;
                                        if *depth == 0 {
                                            if let Ok(v) =
                                                serde_json::from_slice::<serde_json::Value>(obj_buf)
                                                && let Some(doc) = extract_json_document(&v)
                                            {
                                                extracted_doc = Some(doc);
                                                obj_buf.clear();
                                                break;
                                            }
                                            obj_buf.clear();
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    if let Some(doc) = extracted_doc {
                        self.enqueue_document(doc, raw_bytes_scanned);
                    } else if *eof {
                        return Ok(None);
                    }
                }
            }
        }

        Ok(self.queue.pop_front())
    }
}

/// Extracts text documents from a Parquet file and writes them to a destination
/// `.txt` separated by `<|endoftext|>` so multi-paragraph documents preserve
/// both their internal newlines and their document boundaries.
pub fn extract_parquet_text_column(
    parquet_path: &Path,
    output_txt_path: &Path,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Write;

    println!("Extracting text column from Parquet {:?}...", parquet_path);
    let mut stream = DocumentStream::open(parquet_path)?;
    let out_file = File::create(output_txt_path)?;
    let mut writer = std::io::BufWriter::with_capacity(2 * 1024 * 1024, out_file);
    let mut docs = 0u64;

    while let Some(doc) = stream.next_document()? {
        writer.write_all(doc.text.as_bytes())?;
        writer.write_all(b"\n<|endoftext|>\n")?;
        docs += 1;
    }

    writer.flush()?;
    println!(
        "Parquet extraction done: {} documents written to {:?}.",
        docs, output_txt_path
    );
    Ok(docs)
}
