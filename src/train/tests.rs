    use super::*;
                use crate::model::Llm;
use crate::storage;
    use std::path::PathBuf;


    #[test]
    fn test_train_val_split() {
        let tokens: Vec<u32> = (0..1000).collect();
        let dataset = TextDataset { tokens, n_ctx: 32, step: 32, total_bytes: 1000 };

        let (train_ds, val_ds) = dataset.split(0.2);
        assert_eq!(train_ds.tokens.len(), 800);
        assert_eq!(val_ds.tokens.len(), 200);
        // Bytes are split proportionally to the token split.
        assert_eq!(train_ds.total_bytes, 800);
        assert_eq!(val_ds.total_bytes, 200);
    }

    #[test]
    fn test_perplexity_formula() {
        let loss = std::f32::consts::LN_10;
        let ppl = loss.exp();
        assert!((ppl - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_progress_event_json_keys() {
        let ev = TrainingEvent::Progress {
            step: 1,
            epoch: 1,
            epochs: 3,
            batch: 10,
            num_batches: 100,
            pct: 10.0,
            loss: 6.5,
            avg_loss: 7.0,
            lr: 0.00005,
            batches_per_sec: 12.3,
            elapsed_secs: 30.0,
            eta_secs: 270.0,
            current_n_ctx: 256,
        };
        let json = serde_json::to_string(&ev).unwrap();
        println!("PROGRESS_JSON={json}");
        assert!(json.contains("\"loss\""));
        assert!(json.contains("\"avg_loss\""));
        assert!(json.contains("\"num_batches\""));
        assert!(json.contains("\"batches_per_sec\""));
        assert!(json.contains("\"elapsed_secs\""));
        assert!(json.contains("\"eta_secs\""));
        assert!(json.contains("\"current_n_ctx\""));
    }

    #[test]
    fn test_sequence_length_warmup() {
        let target_ctx = 256;
        let warmup_steps = 1000;

        let ctx_start = compute_scheduled_n_ctx(0, warmup_steps, target_ctx);
        let ctx_mid = compute_scheduled_n_ctx(500, warmup_steps, target_ctx);
        let ctx_end = compute_scheduled_n_ctx(1000, warmup_steps, target_ctx);
        let ctx_after = compute_scheduled_n_ctx(1500, warmup_steps, target_ctx);

        assert_eq!(ctx_start, 64);
        assert!(ctx_mid > ctx_start && ctx_mid < ctx_end);
        assert_eq!(ctx_end, target_ctx);
        assert_eq!(ctx_after, target_ctx);
    }

    #[test]
    fn test_streaming_generation_matches_full_generation() {
        let device = Device::flex();
        let dir = test_dir("stream_gen");
        let (tokenizer, _corpus) = tiny_tokenizer_and_corpus(&dir);
        let config = ModelConfig::new(tokenizer.vocab_size(), 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let model: Llm = config.init_model(&device);

        let full = crate::generate::generate(
            &model.valid(),
            &tokenizer,
            "once",
            8,
            0.9,
            40,
            0.9,
            0.05,
            1.1,
            7,
            crate::model::KvCacheDtype::F32,
            None,
            &device,
        )
        .unwrap();

        let mut streamed = String::new();
        let also_full = crate::generate::generate_streaming(
            &model.valid(),
            &tokenizer,
            "once",
            8,
            0.9,
            40,
            0.9,
            0.05,
            1.1,
            7,
            crate::model::KvCacheDtype::F32,
            None,
            &device,
            |chunk| {
                streamed.push_str(chunk);
                true
            },
        )
        .unwrap();

        // Same seed => same trajectory as the non-streaming call.
        assert_eq!(also_full, full);
        // Streaming emits only the generated continuation (the prompt is
        // already known by the caller), which must appear in the full text.
        assert!(!streamed.is_empty(), "streaming emitted no chunks");
        assert!(
            also_full.contains(&streamed),
            "streamed chunks {streamed:?} not contained in {also_full:?}"
        );
    }

    #[test]
    fn test_should_stop_early() {
        assert!(!should_stop_early(0, 99)); // disabled
        assert!(!should_stop_early(3, 2));
        assert!(should_stop_early(3, 3));
        assert!(should_stop_early(1, 5));
    }

    #[test]
    fn test_eval_reports_bpb_and_batch_limit() {
        let device = Device::flex();
        let config = ModelConfig::new(64, 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let model: Llm = config.init_model(&device);

        let tokens: Vec<u32> = (0..160).map(|i| (i % 64) as u32).collect();
        let dataset = TextDataset { tokens, n_ctx: 8, step: 8, total_bytes: 1600 };

        let (loss, ppl, bpb) = evaluate_dataset(&model, &dataset, 4, 0, &device);
        assert!(loss.is_finite());
        assert!((ppl - loss.exp()).abs() < 1e-6);

        // BPB = loss / ln(2) * tokens per byte.
        let expected = loss / std::f32::consts::LN_2 * (dataset.tokens.len() as f32 / 1600.0);
        assert!((bpb.expect("bpb for text dataset") - expected).abs() < 1e-6);

        // max_batches > 0 limits the evaluation windows.
        let (limited, _, _) = evaluate_dataset(&model, &dataset, 4, 2, &device);
        assert!(limited.is_finite());

        // .bin inputs have no byte count: no BPB.
        let no_bytes = TextDataset {
            tokens: (0..160).map(|i| (i % 64) as u32).collect(),
            n_ctx: 8,
            step: 8,
            total_bytes: 0,
        };
        let (_, _, bpb_none) = evaluate_dataset(&model, &no_bytes, 4, 0, &device);
        assert!(bpb_none.is_none());
    }

    #[test]
    fn test_log_event_serialization() {
        let ev = TrainingEvent::LogEvent {
            line: "Dataset split: 100 training samples".into(),
            ts: 1700000000,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"LogEvent\""));
        assert!(json.contains("\"line\""));
        assert!(json.contains("\"ts\""));
        assert!(json.contains("Dataset split"));
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("noctambulo_neko_test_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_tokenizer_and_corpus(dir: &Path) -> (LlmTokenizer, String) {
        let corpus = "once upon a time there was a small black cat. the cat liked to sleep in the warm sun and chase little birds in the garden. one day the cat met a white dog and they became the best of friends. ".repeat(30);
        let corpus_path = dir.join("corpus.txt");
        std::fs::write(&corpus_path, &corpus).unwrap();
        let tokenizer_path = dir.join("tokenizer.json");
        crate::data::train_custom_tokenizer(
            corpus_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
            512,
        )
        .unwrap();
        let tokenizer = LlmTokenizer::from_file(&tokenizer_path).unwrap();
        (tokenizer, corpus)
    }

    /// Replaces every float parameter with a deterministic pattern so tests
    /// that compare numerical trajectories (gradient accumulation, resume)
    /// do not flake on random initialization.
    fn deterministic_fill(model: Llm) -> Llm {
        use burn::module::ModuleMapper;
        use burn::module::Param;
        struct Fill;
        impl ModuleMapper for Fill {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<D>>,
            ) -> Param<Tensor<D>> {
                param.map(|tensor| {
                    let dims = tensor.dims();
                    let n: usize = dims.iter().product();
                    let data: Vec<f32> = (0..n)
                        .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.001)
                        .collect();
                    Tensor::<D>::from_data(TensorData::new(data, dims), &tensor.device())
                })
            }
        }
        model.map(&mut Fill)
    }

    fn save_initial_weights(
        dir: &Path,
        config: &ModelConfig,
        device: &Device,
    ) -> PathBuf {
        let model: Llm = deterministic_fill(config.init_model(device));
        let path = storage::model_weights_path(dir);
        storage::save_module(&model, &path).unwrap();
        path
    }

    #[allow(clippy::too_many_arguments)]
    fn run_train_once(
        config: &ModelConfig,
        tokenizer: &LlmTokenizer,
        corpus: &str,
        epochs: usize,
        batch_size: usize,
        grad_accum: usize,
        max_tokens: usize,
        resume: &Path,
        save_dir: Option<&Path>,
        output_dir: &Path,
        device: &Device,
    ) -> Vec<(usize, f64)> {
        let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let train_device = device.clone().autodiff();
        train_with_events(
            None,
            Some(corpus),
            None,
            config.clone(),
            tokenizer,
            epochs,
            batch_size,
            0.1,
            None,
            Some(max_tokens),
            0.0,
            grad_accum,
            0,
            save_dir,
            Some(resume),
            &train_device,
            output_dir.to_str().unwrap(),
            0,
            LossOptions::default(),
            &tx,
            &cancel,
            None,
        )
        .expect("tiny training run failed");
        drop(tx);
        rx.into_iter()
            .filter_map(|event| match event {
                TrainingEvent::Progress { step, lr, .. } => Some((step, lr)),
                _ => None,
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_train_losses(
        config: &ModelConfig,
        tokenizer: &LlmTokenizer,
        corpus: &str,
        epochs: usize,
        batch_size: usize,
        max_tokens: usize,
        resume: &Path,
        output_dir: &Path,
        device: &Device,
    ) -> Vec<(usize, f32)> {
        let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let train_device = device.clone().autodiff();
        train_with_events(
            None,
            Some(corpus),
            None,
            config.clone(),
            tokenizer,
            epochs,
            batch_size,
            0.1,
            None,
            Some(max_tokens),
            0.0,
            1,
            0,
            None,
            Some(resume),
            &train_device,
            output_dir.to_str().unwrap(),
            0,
            LossOptions::default(),
            &tx,
            &cancel,
            None,
        )
        .expect("tiny training run failed");
        drop(tx);
        rx.into_iter()
            .filter_map(|event| match event {
                TrainingEvent::Progress { step, avg_loss, .. } => Some((step, avg_loss)),
                _ => None,
            })
            .collect()
    }

    fn load_flat_weights(dir: &Path, config: &ModelConfig, device: &Device) -> Vec<f32> {
        let model: Llm = storage::load_module(
            config.init_model(device),
            &storage::model_weights_path(dir),
        )
        .expect("failed to load trained weights");
        let mut out = Vec::new();
        out.extend(model.embed_tokens.weight.val().into_data().try_to_vec::<f32>().unwrap());
        out.extend(model.blocks[0].attn.q_proj.weight.val().into_data().try_to_vec::<f32>().unwrap());
        out.extend(model.blocks[0].mlp.down_proj.weight.val().into_data().try_to_vec::<f32>().unwrap());
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn flat_autodiff_weights(model: &Llm) -> Vec<f32> {
        let model = model.valid();
        let mut out = Vec::new();
        out.extend(model.embed_tokens.weight.val().into_data().try_to_vec::<f32>().unwrap());
        out.extend(model.blocks[0].attn.q_proj.weight.val().into_data().try_to_vec::<f32>().unwrap());
        out.extend(model.blocks[0].mlp.down_proj.weight.val().into_data().try_to_vec::<f32>().unwrap());
        out
    }

    fn fixed_grads(model: &Llm, input: &Tensor<2, Int>) -> GradientsParams {
        let logits = model.forward(input.clone());
        let loss = logits.mean();
        GradientsParams::from_grads(loss.backward(), model)
    }

    /// Finding #1: gradient accumulation must be mathematically equivalent to a
    /// single larger batch (linearity of the gradient). Today the training loop
    /// steps with only the last micro-batch gradients, so this test fails.
    #[test]
    fn test_grad_accum_equivalence() {
        let device = Device::flex();
        let dir = test_dir("grad_accum");
        let (tokenizer, corpus) = tiny_tokenizer_and_corpus(&dir);
        let config = ModelConfig::new(tokenizer.vocab_size(), 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let init = save_initial_weights(&dir, &config, &device);

        // 20 tokens at n_ctx=8 -> exactly 2 sequences.
        let out_ref = dir.join("out_ref");
        run_train_once(
            &config, &tokenizer, &corpus, 1, 2, 1, 20, &init, None, &out_ref, &device,
        );

        let out_accum = dir.join("out_accum");
        run_train_once(
            &config, &tokenizer, &corpus, 1, 1, 2, 20, &init, None, &out_accum, &device,
        );

        let w_ref = load_flat_weights(&out_ref, &config, &device);
        let w_accum = load_flat_weights(&out_accum, &config, &device);
        let diff = max_abs_diff(&w_ref, &w_accum);
        eprintln!("[GRAD_ACCUM] max_abs_diff(batch=2 vs batch=1+accum=2) = {diff:.6e}");
        // End-to-end check: AdamW normalizes each element, so float reduction
        // differences (batch-2 matmul vs two batch-1 matmuls) get amplified up
        // to ~1e-4. Dropping a micro-batch changes the update by ~2*min_lr
        // (~2e-2), i.e. two orders of magnitude above this threshold.
        assert!(
            diff < 1e-3,
            "gradient accumulation with 2 micro-batches must equal one batch of 2: \
             max_abs_diff={diff:.6e}"
        );
    }

    /// Optimizer-free proof of finding #1: accumulated micro-batch gradients
    /// must be identical to the full-batch gradient (linearity + 1/N scaling).
    #[test]
    fn test_accumulated_gradients_match_full_batch() {
        let device = Device::flex().autodiff();
        let config = ModelConfig::new(64, 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let model = config.init_model(&device).train();

        let seq_a = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1i32, 2, 3, 4, 5, 6, 7, 8], [1, 8]),
            &device,
        );
        let seq_b = Tensor::<2, Int>::from_data(
            TensorData::new(vec![8i32, 7, 6, 5, 4, 3, 2, 1], [1, 8]),
            &device,
        );

        // Reference: single full-batch forward.
        let full_batch = Tensor::cat(vec![seq_a.clone(), seq_b.clone()], 0);
        let full_loss = model.forward(full_batch).mean();
        let full_grads = GradientsParams::from_grads(full_loss.backward(), &model);

        // Accumulated: two micro-batches, each scaled by 1/N.
        let mut accumulator = GradientsAccumulator::<Llm>::new();
        for seq in [seq_a, seq_b] {
            let loss = model.forward(seq).mean() / 2.0;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            accumulator.accumulate(&model, grads);
        }
        let accum_grads = accumulator.grads();

        let param_id = model.embed_tokens.weight.id;
        let full = full_grads
            .get::<2>(param_id)
            .expect("full-batch embedding gradient")
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let accumulated = accum_grads
            .get::<2>(param_id)
            .expect("accumulated embedding gradient")
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        let diff = max_abs_diff(&full, &accumulated);
        let scale = full.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!("[GRAD_GRADS] max_abs_diff={diff:.3e} | gradient scale {scale:.3e}");
        assert!(
            diff <= 1e-5 * scale.max(1.0),
            "accumulated gradients must match the full-batch gradient: \
             max_abs_diff={diff:.3e} (scale {scale:.3e})"
        );
    }

    /// Finding #2 (scheduler): resuming from an epoch checkpoint must continue
    /// the LR schedule, not restart it. Compares the first LR after resume with
    /// the LR at the equivalent step of an uninterrupted 2-epoch run.
    #[test]
    fn test_resume_restores_lr_schedule() {
        let device = Device::flex();
        let dir = test_dir("resume_lr");
        let (tokenizer, corpus) = tiny_tokenizer_and_corpus(&dir);
        let config = ModelConfig::new(tokenizer.vocab_size(), 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let init = save_initial_weights(&dir, &config, &device);

        // 72 tokens -> 8 sequences -> 8 batches per epoch (batch_size=1).
        let continuous = run_train_once(
            &config, &tokenizer, &corpus, 2, 1, 1, 72, &init, None, &dir.join("continuous"), &device,
        );
        let lr_continuous_step_9 = continuous
            .iter()
            .find(|(step, _)| *step == 9)
            .map(|(_, lr)| *lr)
            .expect("continuous run must emit progress for step 9");

        let ck_dir = dir.join("ck");
        run_train_once(
            &config, &tokenizer, &corpus, 1, 1, 1, 72, &init, Some(&ck_dir), &dir.join("split_a"), &device,
        );
        let ckpt = ck_dir.join("checkpoint_epoch_1.bpk");
        assert!(ckpt.is_file(), "epoch checkpoint was not written to {ckpt:?}");

        let resumed = run_train_once(
            &config, &tokenizer, &corpus, 1, 1, 1, 72, &ckpt, None, &dir.join("split_b"), &device,
        );
        let (first_step, lr_resumed) = *resumed.first().expect("resumed run must emit progress");
        eprintln!(
            "[RESUME] first step after resume = {first_step}, lr = {lr_resumed:.8} | \
             continuous step 9 lr = {lr_continuous_step_9:.8}"
        );
        assert!(
            (lr_resumed - lr_continuous_step_9).abs() < 1e-9,
            "the LR schedule must continue after resume: resumed lr={lr_resumed:.8} \
             vs continuous step-9 lr={lr_continuous_step_9:.8}"
        );
    }

    /// A warm start may point at a model directory: the loader resolves
    /// `model_weights.bpk` and tolerates the absence of training sidecars.
    #[test]
    fn test_resume_from_model_directory() {
        let device = Device::flex();
        let dir = test_dir("resume_dir");
        let (tokenizer, corpus) = tiny_tokenizer_and_corpus(&dir);
        let config = ModelConfig::new(tokenizer.vocab_size(), 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let init = save_initial_weights(&dir, &config, &device);
        assert!(init.is_file(), "expected {init:?}");

        let out = dir.join("out");
        let events = run_train_once(
            &config, &tokenizer, &corpus, 1, 1, 1, 72, &dir, None, &out, &device,
        );
        assert!(!events.is_empty(), "warm start from a model directory must train");
        assert!(
            out.join("model_weights.bpk").is_file(),
            "training must save the final weights"
        );
    }

    /// Finding #2 (optimizer): dropping the AdamW moments on resume changes the
    /// next update, while `Optimizer::to_record`/`load_record` reproduces it.
    #[test]
    fn test_adamw_state_survives_record_roundtrip() {
        let device = Device::flex().autodiff();
        let config = ModelConfig::new(64, 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1i32; 8], [1, 8]),
            &device,
        );

        // Step 1: shared baseline weights + optimizer state.
        let mut model = config.init_model(&device).train();
        let mut optim = AdamWConfig::new()
            .with_weight_decay(0.1f32)
            .init();
        let grads = fixed_grads(&model, &input);
        model = optim.step(1e-2, model, grads);
        // Snapshot the optimizer state right after the first step (what a
        // proper checkpoint would persist alongside the weights).
        let record = optim.to_record();

        // Reference: same optimizer object continues (moments preserved).
        let mut model_ref = model.clone();
        let grads_ref = fixed_grads(&model_ref, &input);
        model_ref = optim.step(1e-2, model_ref, grads_ref);

        // Naive resume: only weights survive, optimizer moments are reset.
        let mut model_naive = model.clone();
        let mut optim_naive = AdamWConfig::new()
            .with_weight_decay(0.1f32)
            .init();
        let grads_naive = fixed_grads(&model_naive, &input);
        model_naive = optim_naive.step(1e-2, model_naive, grads_naive);

        // Proper resume: weights + optimizer record captured after step 1.
        let mut model_proper = model.clone();
        let mut optim_proper = AdamWConfig::new()
            .with_weight_decay(0.1f32)
            .init()
            .load_record(record);
        let grads_proper = fixed_grads(&model_proper, &input);
        model_proper = optim_proper.step(1e-2, model_proper, grads_proper);

        let w_ref = flat_autodiff_weights(&model_ref);
        let w_naive = flat_autodiff_weights(&model_naive);
        let w_proper = flat_autodiff_weights(&model_proper);

        let diff_naive = max_abs_diff(&w_ref, &w_naive);
        let diff_proper = max_abs_diff(&w_ref, &w_proper);
        eprintln!(
            "[ADAMW] naive resume diff = {diff_naive:.6e} | optimizer-record resume diff = {diff_proper:.6e}"
        );

        assert!(
            diff_naive > 1e-6,
            "a checkpoint without the optimizer record must change the next update \
             (diff={diff_naive:.6e}); if this fails the test setup is wrong"
        );
        assert!(
            diff_proper < 1e-6,
            "restoring Optimizer::to_record/load_record must reproduce the next update \
             exactly (diff={diff_proper:.6e})"
        );
    }

    /// The optimizer sidecar on disk must reproduce the in-memory record, and a
    /// corrupt sidecar must fail loudly instead of silently resetting Adam.
    #[test]
    fn test_optimizer_state_bpk_roundtrip() {
        let device = Device::flex().autodiff();
        let config = ModelConfig::new(64, 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1i32; 8], [1, 8]),
            &device,
        );
        let mut model = config.init_model(&device).train();
        let mut optim = AdamWConfig::new().with_weight_decay(0.1f32).init();
        let grads = fixed_grads(&model, &input);
        model = optim.step(1e-2, model, grads);

        let dir = test_dir("optim_bpk");
        std::fs::create_dir_all(&dir).unwrap();
        persist_train_state(&optim, &TrainState::default(), &dir).unwrap();

        let mut model_ref = model.clone();
        let grads_ref = fixed_grads(&model_ref, &input);
        model_ref = optim.step(1e-2, model_ref, grads_ref);

        let (mut restored, loaded) = load_optimizer_record(
            AdamWConfig::new().with_weight_decay(0.1f32).init(),
            &dir,
        )
        .expect("optimizer sidecar must load");
        assert!(loaded, "the sidecar must be reported as applied");
        let mut model_resumed = model.clone();
        let grads_resumed = fixed_grads(&model_resumed, &input);
        model_resumed = restored.step(1e-2, model_resumed, grads_resumed);

        let diff = max_abs_diff(
            &flat_autodiff_weights(&model_ref),
            &flat_autodiff_weights(&model_resumed),
        );
        assert!(
            diff < 1e-6,
            "on-disk optimizer state must reproduce the next update (diff={diff:.6e})"
        );

        std::fs::write(dir.join(OPTIMIZER_STATE_FILE), b"not a burnpack").unwrap();
        assert!(
            load_optimizer_record(AdamWConfig::new().init(), &dir).is_err(),
            "a corrupt optimizer sidecar must be an error, not a silent reset"
        );
    }

    /// Finding #3: stopping training (user stop / Ctrl+C) must persist an
    /// interrupted checkpoint plus the full training state.
    #[test]
    fn test_cancel_saves_interrupted_checkpoint() {
        let device = Device::flex();
        let dir = test_dir("cancel_save");
        let (tokenizer, corpus) = tiny_tokenizer_and_corpus(&dir);
        let config = ModelConfig::new(tokenizer.vocab_size(), 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let init = save_initial_weights(&dir, &config, &device);
        let out = dir.join("out");

        let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_thread = cancel.clone();
        let worker = std::thread::spawn(move || {
            let device = Device::flex().autodiff();
            train_with_events(
                None, Some(&corpus), None, config, &tokenizer, 4, 1, 0.1, None, Some(72), 0.0, 1, 0,
                None, Some(&init), &device, out.to_str().unwrap(), 0, LossOptions::default(),
                &tx, &cancel, None,
            )
            .expect("training run failed");
        });

        // Wait until the first micro-batch completes, then request a stop.
        loop {
            if let TrainingEvent::Progress { .. } = rx.recv().expect("training event") {
                break;
            }
        }
        cancel_thread.store(true, Ordering::SeqCst);
        worker.join().expect("worker thread panicked");

        assert!(
            dir.join("out/checkpoint_interrupted.bpk").is_file(),
            "interrupted checkpoint missing"
        );
        assert!(dir.join("out/train_state.json").is_file(), "train_state.json missing");
        assert!(
            dir.join("out/optimizer_state.bpk").is_file(),
            "optimizer_state.bin missing"
        );
    }

    /// Regression gate for the Stage D weight-decay bug: a `ModuleMapper`-based
    /// decoupled decay froze every mapped parameter, so training stayed at
    /// ln(vocab) no matter the learning rate. AdamW now applies weight decay
    /// natively; with the default `LossOptions` (weight_decay = 0.1) the loss
    /// must fall well below the random baseline on a corpus small enough to
    /// overfit.
    #[test]
    fn test_training_learns_with_default_weight_decay() {
        let device = Device::flex();
        let dir = test_dir("overfit");
        let (tokenizer, corpus) = tiny_tokenizer_and_corpus(&dir);
        let config = ModelConfig::new(tokenizer.vocab_size(), 32, 2, Some(2), 1, 8, 0.0, 0.0);
        let init = save_initial_weights(&dir, &config, &device);
        let out = dir.join("out");

        let losses = run_train_losses(
            &config, &tokenizer, &corpus, 8, 4, 0, &init, &out, &device,
        );
        assert!(!losses.is_empty(), "no progress events emitted");
        // Early progress events can carry a provisional avg_loss of 0 before
        // the first loss sync; ignore them when picking the baseline/final.
        let first = losses
            .iter()
            .map(|(_, l)| *l)
            .find(|l| *l > 0.0)
            .unwrap_or(losses.first().unwrap().1);
        let last = losses
            .iter()
            .rev()
            .map(|(_, l)| *l)
            .find(|l| *l > 0.0)
            .unwrap_or(losses.last().unwrap().1);
        eprintln!(
            "[OVERFIT] avg_loss {first:.4} -> {last:.4} over {} steps (chance ~ {:.2})",
            losses.len(),
            (tokenizer.vocab_size() as f32).ln()
        );
        assert!(
            last < first - 1.0,
            "training did not reduce the loss ({first:.4} -> {last:.4}); \
             default weight decay is breaking the optimizer again"
        );
    }

    #[test]
    fn test_training_learns_with_eviction_mask() {
        let device = Device::flex();
        let dir = test_dir("evict_overfit");
        let (tokenizer, corpus) = tiny_tokenizer_and_corpus(&dir);
        let mut config = ModelConfig::new(tokenizer.vocab_size(), 32, 2, Some(2), 1, 8, 0.0, 0.0);
        config.sink_tokens = 1;
        config.kv_eviction = Some("h2o".into());
        config.kv_budget = Some(4);
        config.kv_recent = 1;
        config
            .eviction_geometry()
            .unwrap()
            .expect("valid eviction geometry");
        let init = save_initial_weights(&dir, &config, &device);
        let out = dir.join("out");

        let losses = run_train_losses(
            &config, &tokenizer, &corpus, 8, 4, 0, &init, &out, &device,
        );
        assert!(!losses.is_empty(), "no progress events emitted");
        let first = losses
            .iter()
            .map(|(_, l)| *l)
            .find(|l| *l > 0.0)
            .unwrap_or(losses.first().unwrap().1);
        let last = losses
            .iter()
            .rev()
            .map(|(_, l)| *l)
            .find(|l| *l > 0.0)
            .unwrap_or(losses.last().unwrap().1);
        eprintln!(
            "[EVICT OVERFIT] avg_loss {first:.4} -> {last:.4} over {} steps (chance ~ {:.2})",
            losses.len(),
            (tokenizer.vocab_size() as f32).ln()
        );
        assert!(
            last < first - 1.0,
            "eviction-masked training did not reduce the loss ({first:.4} -> {last:.4})"
        );
    }

    #[test]
    fn test_compute_loss_label_smoothing_and_z_loss() {
                let device = Device::flex();

        let opts = |ls: f32, z: f32| LossOptions {
            weight_decay: 0.0,
            label_smoothing: ls,
            z_loss: z,
        };
        let scalar = |loss: Tensor<1>| loss.into_data().try_to_vec::<f32>().unwrap()[0];

        let logits = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 2.0, 0.0, 0.0], [1, 4]),
            &device,
        );
        let target = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1i32], [1, 1]),
            &device,
        );

        let lse = (1.0f32 + (2.0f32).exp() + 1.0 + 1.0).ln();
        let nll = lse - 2.0;
        let plain = scalar(compute_loss(logits.clone(), target.clone(), opts(0.0, 0.0)));
        assert!((plain - nll).abs() < 1e-5, "plain={plain} nll={nll}");

        // Uniform logits: NLL = ln(4) and smoothing keeps it unchanged.
        let uniform = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32; 4], [1, 4]),
            &device,
        );
        let t0 = Tensor::<2, Int>::from_data(
            TensorData::new(vec![0i32], [1, 1]),
            &device,
        );
        let ln4 = 4.0f32.ln();
        let smoothed = scalar(compute_loss(uniform.clone(), t0.clone(), opts(0.5, 0.0)));
        assert!((smoothed - ln4).abs() < 1e-5, "smoothed={smoothed} ln4={ln4}");

        // Z-loss adds mean(z^2) with z = logsumexp(logits).
        let with_z = scalar(compute_loss(uniform, t0, opts(0.0, 1.0)));
        let expected = ln4 + ln4 * ln4;
        assert!((with_z - expected).abs() < 1e-4, "with_z={with_z} expected={expected}");

        // Smoothing raises the loss when the model is confidently correct.
        let harder = scalar(compute_loss(logits, target, opts(0.5, 0.0)));
        assert!(harder > plain, "harder={harder} plain={plain}");
    }

    /// Multiple sources are mixed at the document level in memory (no
    /// `combined_corpus.txt`): both files contribute and the tokenizer sees one
    /// EOS per document.
    #[test]
    fn test_dataset_from_sources_streams_documents_in_memory() {
        let dir = test_dir("dataset_sources");
        let (tokenizer, _corpus) = tiny_tokenizer_and_corpus(&dir);
        let a = dir.join("a.txt");
        let b = dir.join("b.txt");
        std::fs::write(&a, "alpha beta gamma\n<|endoftext|>\n").unwrap();
        std::fs::write(&b, "delta epsilon zeta\n<|endoftext|>\n").unwrap();

        let sources = vec![(a, 1.0), (b, 1.0)];
        let dataset = TextDataset::from_sources_with_progress(
            &sources, &tokenizer, 4, None, None, None, None,
        )
        .unwrap();

        assert!(!dataset.tokens.is_empty());
        assert!(dataset.total_bytes > 0);
        let eos = tokenizer.end_of_text_id().expect("eos token");
        assert_eq!(
            dataset.tokens.iter().filter(|&&t| t == eos).count(),
            2,
            "one EOS per mixed document"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
