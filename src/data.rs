//! Dataset ingestion, document-level streaming, multi-source mixing,
//! tokenization, and pre-tokenized binary dataset (`KNKB`) I/O.
//!
//! Organized into focused submodules:
//! - [`document`]: Format-aware streaming document readers (`.parquet`, `.jsonl`,
//!   `.json` arrays, `<|endoftext|>`-delimited multi-line `.txt`, and line `.txt`).
//! - [`mix`]: Document-level proportional & weighted dataset mixing (in-memory
//!   streaming or on-disk).
//! - [`binary`]: Pre-tokenized binary dataset (`KNKB` v1/v2) with vocabulary
//!   hash validation.
//! - [`tokenizer`]: BPE tokenizer training, resolution, and multi-threaded
//!   batch encoding.
//! - [`hf`]: Hugging Face Hub dataset listing and streaming downloads.

pub mod binary;
pub mod document;
pub mod hf;
pub mod mix;
pub mod tokenizer;

#[allow(unused_imports)]
pub use binary::{
    BINARY_MAGIC, BINARY_VERSION, is_binary_file, load_tokens_from_binary, prepare_binary_dataset,
    prepare_binary_dataset_from_sources, save_tokens_to_binary, tokenizer_vocab_hash,
};
#[allow(unused_imports)]
pub use document::{
    Document, DocumentFormat, DocumentStream, EOT_MARKER, MAX_DOC_CHUNK_BYTES, detect_format,
    extract_json_document, extract_parquet_text_column, parse_line_or_json, split_utf8_chunks,
};
pub use hf::{
    download_dataset_from_hf, download_dataset_streaming, hf_repo_files, pick_default_dataset_file,
};
#[allow(unused_imports)]
pub use mix::{
    MultiSourceDocumentStream, combine_and_mix_datasets, combine_and_mix_weighted,
    combine_and_mix_weighted_with,
};
#[allow(unused_imports)]
pub use tokenizer::{
    LlmDecodeStream, LlmTokenizer, encode_parquet_streaming, resolve_tokenizer,
    resolve_tokenizer_path, train_custom_tokenizer,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::path::{Path, PathBuf};

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "noctambulo_data_{name}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_tokenizer(dir: &Path) -> LlmTokenizer {
        let corpus = "the quick brown fox jumps over the lazy dog. the cat sat on the mat and looked at the sun. ".repeat(40);
        let corpus_path = dir.join("tokenizer_corpus.txt");
        std::fs::write(&corpus_path, &corpus).unwrap();
        let tokenizer_path = dir.join("tokenizer.json");
        train_custom_tokenizer(
            corpus_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            256,
        )
        .unwrap();
        LlmTokenizer::from_file(&tokenizer_path).unwrap()
    }

    #[test]
    fn test_binary_dataset_roundtrip() {
        let path = std::env::temp_dir().join("test_noctambulo_bin_roundtrip.bin");

        let original_tokens = vec![1, 100, 500, 1000, 7999, 42];
        let original_vocab_size = 8000;
        let original_hash = 0xDEAD_BEEF_CAFE_F00Du64;

        save_tokens_to_binary(&path, &original_tokens, original_vocab_size, original_hash).unwrap();

        assert!(is_binary_file(&path).unwrap());

        let (loaded_tokens, loaded_vocab) =
            load_tokens_from_binary(&path, None, Some(original_hash)).unwrap();
        assert_eq!(loaded_tokens, original_tokens);
        assert_eq!(loaded_vocab, original_vocab_size);

        // Without an expected fingerprint the file still loads.
        let (loaded_tokens, _) = load_tokens_from_binary(&path, None, None).unwrap();
        assert_eq!(loaded_tokens, original_tokens);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_binary_dataset_max_tokens_limit() {
        let path = std::env::temp_dir().join("test_noctambulo_bin_limit.bin");

        let original_tokens: Vec<u32> = (0..100).collect();
        save_tokens_to_binary(&path, &original_tokens, 1000, 0x1234).unwrap();

        let (loaded_tokens, _) = load_tokens_from_binary(&path, Some(30), Some(0x1234)).unwrap();
        assert_eq!(loaded_tokens.len(), 30);
        assert_eq!(loaded_tokens, (0..30).collect::<Vec<u32>>());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_binary_vocab_hash_mismatch_rejected() {
        let path = std::env::temp_dir().join("test_noctambulo_bin_mismatch.bin");

        save_tokens_to_binary(&path, &[1, 2, 3], 100, 0xAAAA).unwrap();

        let err = load_tokens_from_binary(&path, None, Some(0xBBBB)).unwrap_err();
        assert!(
            err.to_string().contains("mismatch"),
            "expected tokenizer mismatch error, got: {err}"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_binary_v1_without_hash_still_loads() {
        let path = std::env::temp_dir().join("test_noctambulo_bin_v1.bin");

        let tokens: Vec<u32> = vec![5, 6, 7];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(BINARY_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&100u32.to_le_bytes());
        bytes.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for token in &tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();

        let (loaded, vocab) = load_tokens_from_binary(&path, None, Some(0x1234)).unwrap();
        assert_eq!(loaded, tokens);
        assert_eq!(vocab, 100);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_pick_default_dataset_file() {
        let files = vec![
            ".gitattributes".to_string(),
            "README.md".to_string(),
            "data/validation-00000-of-00001.parquet".to_string(),
            "data/train-00000-of-00002.parquet".to_string(),
            "data/train-00001-of-00002.parquet".to_string(),
        ];
        assert_eq!(
            pick_default_dataset_file(&files).as_deref(),
            Some("data/train-00000-of-00002.parquet")
        );

        let files = vec!["data/train.parquet".to_string(), "data/train.txt".to_string()];
        assert_eq!(
            pick_default_dataset_file(&files).as_deref(),
            Some("data/train.txt")
        );

        let files = vec!["README.md".to_string(), "config.json".to_string()];
        assert_eq!(pick_default_dataset_file(&files), None);

        assert_eq!(pick_default_dataset_file(&[]), None);
    }

    #[test]
    fn test_combine_and_mix_weighted_rejects_bin_sources() {
        let txt = std::env::temp_dir().join("test_noctambulo_mix_a.txt");
        let bin = std::env::temp_dir().join("test_noctambulo_mix_b.bin");
        let out = std::env::temp_dir().join("test_noctambulo_mix_out.txt");
        std::fs::write(&txt, "hello\nworld\n").unwrap();
        std::fs::write(&bin, b"not really tokens").unwrap();

        let sources = vec![(txt.clone(), 1.0), (bin.clone(), 1.0)];
        let err = combine_and_mix_weighted(&sources, &out, usize::MAX).unwrap_err();
        assert!(
            err.to_string().contains("Refusing to mix binary dataset"),
            "expected .bin rejection, got: {err}"
        );
        assert!(!out.exists());

        let _ = std::fs::remove_file(&txt);
        let _ = std::fs::remove_file(&bin);
    }

    #[test]
    fn test_split_utf8_chunks_respects_boundaries() {
        let text = "aé漢".repeat(1000);
        let pieces = split_utf8_chunks(text.clone(), 7);
        assert!(pieces.len() > 1);
        for piece in &pieces {
            assert!(piece.len() <= 7);
        }
        assert_eq!(pieces.concat(), text);

        let short = "hello".to_string();
        assert_eq!(split_utf8_chunks(short.clone(), 16), vec![short]);
    }

    #[test]
    fn test_encode_file_streaming_deduped() {
        let dir = test_dir("dedupe");
        let tokenizer = test_tokenizer(&dir);

        let dup_path = dir.join("dups.txt");
        std::fs::write(
            &dup_path,
            "hello world\nhello world\nanother line\nhello world\n",
        )
        .unwrap();
        let unique_path = dir.join("unique.txt");
        std::fs::write(&unique_path, "hello world\nanother line\n").unwrap();

        let (dup_tokens, _bytes, duplicates) =
            tokenizer.encode_file_streaming_deduped(&dup_path, None).unwrap();
        let (unique_tokens, _) = tokenizer.encode_file_streaming(&unique_path, None).unwrap();

        assert_eq!(duplicates, 2, "three copies of a line minus the first");
        assert_eq!(dup_tokens, unique_tokens);
        assert!(!dup_tokens.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_encode_parquet_streaming_and_dedupe() {
        use arrow::array::StringArray;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;

        let dir = test_dir("parquet");
        let tokenizer = test_tokenizer(&dir);

        let schema =
            std::sync::Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let array = StringArray::from(vec!["hello world", "hello world", "another line"]);
        let batch =
            RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(array)]).unwrap();
        let parquet_path = dir.join("docs.parquet");
        let file = File::create(&parquet_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let (tokens, docs, duplicates) =
            encode_parquet_streaming(&parquet_path, &tokenizer, None, true).unwrap();
        assert_eq!((docs, duplicates), (2, 1));

        let unique_path = dir.join("unique.txt");
        std::fs::write(&unique_path, "hello world\nanother line\n").unwrap();
        let (unique_tokens, _) = tokenizer.encode_file_streaming(&unique_path, None).unwrap();
        assert_eq!(tokens, unique_tokens);

        let (all_tokens, docs_all, dups_all) =
            encode_parquet_streaming(&parquet_path, &tokenizer, None, false).unwrap();
        assert_eq!((docs_all, dups_all), (3, 0));
        assert!(all_tokens.len() > tokens.len());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_delimited_multiline_text_keeps_paragraphs_together() {
        let dir = test_dir("delimited_eot");
        let path = dir.join("stories.txt");
        std::fs::write(
            &path,
            "Paragraph 1 of story A.\nParagraph 2 of story A.\n<|endoftext|>\nParagraph 1 of story B.\nParagraph 2 of story B.\n<|endoftext|>\n",
        )
        .unwrap();

        assert_eq!(detect_format(&path).unwrap(), DocumentFormat::DelimitedText);

        let mut stream = DocumentStream::open(&path).unwrap();
        let doc1 = stream.next_document().unwrap().unwrap();
        let doc2 = stream.next_document().unwrap().unwrap();
        assert!(stream.next_document().unwrap().is_none());

        assert_eq!(doc1.text, "Paragraph 1 of story A.\nParagraph 2 of story A.");
        assert_eq!(doc2.text, "Paragraph 1 of story B.\nParagraph 2 of story B.");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_json_array_and_instruction_jsonl_readers() {
        let dir = test_dir("json_readers");

        // Top-level JSON array on a single line (like stories_00.json)
        let json_arr_path = dir.join("stories.json");
        std::fs::write(
            &json_arr_path,
            r#"[{"instruction": {"prompt": "ignored"}, "story": "First story\nSecond para"}, {"story": "Second story"}]"#,
        )
        .unwrap();

        assert_eq!(
            detect_format(&json_arr_path).unwrap(),
            DocumentFormat::JsonArray
        );
        let mut s = DocumentStream::open(&json_arr_path).unwrap();
        assert_eq!(
            s.next_document().unwrap().unwrap().text,
            "First story\nSecond para"
        );
        assert_eq!(s.next_document().unwrap().unwrap().text, "Second story");
        assert!(s.next_document().unwrap().is_none());

        // Instruction JSONL (like databricks-dolly-15k.jsonl)
        let jsonl_path = dir.join("dolly.jsonl");
        std::fs::write(
            &jsonl_path,
            "{\"instruction\": \"What is Rust?\", \"context\": \"Systems language\", \"response\": \"Fast and safe.\"}\n",
        )
        .unwrap();

        let mut s2 = DocumentStream::open(&jsonl_path).unwrap();
        assert_eq!(
            s2.next_document().unwrap().unwrap().text,
            "What is Rust?\n\nSystems language\n\nFast and safe."
        );
        assert!(s2.next_document().unwrap().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multi_source_mixing_and_binary_prep_without_temp_txt() {
        use arrow::array::StringArray;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;

        let dir = test_dir("multi_mix");
        let tokenizer = test_tokenizer(&dir);

        // Source 1: Parquet with multi-line documents
        let schema =
            std::sync::Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let array = StringArray::from(vec!["the quick brown fox\njumps over the lazy dog"]);
        let batch =
            RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(array)]).unwrap();
        let parquet_path = dir.join("wiki.parquet");
        let file = File::create(&parquet_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Source 2: Delimited text
        let txt_path = dir.join("stories.txt");
        std::fs::write(
            &txt_path,
            "the cat sat on the mat\nand looked at the sun\n<|endoftext|>\n",
        )
        .unwrap();

        let out_bin = dir.join("mixed.bin");
        let sources = vec![(parquet_path, 1.0), (txt_path, 1.0)];
        prepare_binary_dataset_from_sources(&sources, &out_bin, &tokenizer, None, None, false)
            .unwrap();

        assert!(is_binary_file(&out_bin).unwrap());
        let (tokens, _) =
            load_tokens_from_binary(&out_bin, None, Some(tokenizer_vocab_hash(&tokenizer)))
                .unwrap();
        let eos_id = tokenizer.end_of_text_id().unwrap();
        // Exactly 2 documents -> 2 EOS tokens in total (not 4 from broken lines!)
        let eos_count = tokens.iter().filter(|&&t| t == eos_id).count();
        assert_eq!(eos_count, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
