    use super::*;
        
    #[test]
    fn test_rope_spec_linear_scales_frequencies() {
        let base = rope_spec(256, 64, RopeScalingKind::None, 1.0, 10000.0);
        let linear = rope_spec(256, 64, RopeScalingKind::Linear, 4.0, 10000.0);
        assert_eq!(base.freqs.len(), linear.freqs.len());
        for (b, l) in base.freqs.iter().zip(linear.freqs.iter()) {
            assert!((b / 4.0 - l).abs() < 1e-7, "linear scaling mismatch: {b} vs {l}");
        }
        assert_eq!(linear.mscale, 1.0);
        // A factor <= 1 is a no-op.
        let noop = rope_spec(256, 64, RopeScalingKind::Linear, 1.0, 10000.0);
        assert_eq!(noop.freqs[1], base.freqs[1]);
    }

    #[test]
    fn test_rope_spec_ntk_lowers_frequencies() {
        let base = rope_spec(256, 64, RopeScalingKind::None, 1.0, 10000.0);
        let ntk = rope_spec(256, 64, RopeScalingKind::Ntk, 4.0, 10000.0);
        // The first frequency is 1.0 regardless of base; later ones shrink.
        assert!((base.freqs[0] - 1.0).abs() < 1e-6);
        assert!((ntk.freqs[0] - 1.0).abs() < 1e-6);
        assert!(ntk.freqs[1] < base.freqs[1]);
        assert!(ntk.freqs[ntk.freqs.len() - 2] < base.freqs[base.freqs.len() - 2]);
    }

    #[test]
    fn test_rope_spec_yarn_ramp_and_mscale() {
        let factor = 4.0;
        let half = 32; // head_dim = 64
        let original = rope_spec(256, 64, RopeScalingKind::None, 1.0, 10000.0);
        let yarn = rope_spec(256, 64, RopeScalingKind::Yarn, factor, 10000.0);

        // High frequencies (low dimensions) keep the original table; low
        // frequencies (high dimensions) are interpolated by `1/factor`.
        assert!((yarn.freqs[0] - original.freqs[0]).abs() < 1e-6);
        assert!((yarn.freqs[half - 1] - original.freqs[half - 1] / factor).abs() < 1e-6);
        assert!((yarn.mscale - (0.1 * factor.ln() + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_extended_ctx_len() {
        assert_eq!(extended_ctx_len(256, RopeScalingKind::None, 4.0), 256);
        assert_eq!(extended_ctx_len(256, RopeScalingKind::Linear, 1.0), 256);
        assert_eq!(extended_ctx_len(256, RopeScalingKind::Linear, 2.0), 512);
        assert_eq!(extended_ctx_len(256, RopeScalingKind::Yarn, 2.5), 640);
        // No table cap: angles are computed per position.
        assert_eq!(extended_ctx_len(4096, RopeScalingKind::Ntk, 100.0), 409_600);
    }

    #[test]
    fn test_rope_spec_wiring() {
                let device = Device::flex();

        let base = CausalSelfAttentionConfig {
            n_embd: 32,
            n_head: 4,
            n_kv_head: 4,
            n_ctx: 8,
            dropout: 0.0,
            attn_softcap: 0.0,
            rope_kind: RopeScalingKind::None,
            rope_factor: 1.0,
            rope_base: 10000.0,
            sliding_window: None,
            sink_tokens: 0,
            evict_budget: None,
            evict_recent: 0,
        };
        let attn = base.init(&device);
        assert_eq!(attn.rope_freqs.len(), 4); // head_dim / 2
        assert_eq!(attn.rope_mscale, 1.0);

        let yarn = CausalSelfAttentionConfig {
            rope_kind: RopeScalingKind::Yarn,
            rope_factor: 2.0,
            ..base
        };
        let attn = yarn.init(&device);
        assert!((attn.rope_mscale - (0.1 * 2.0f32.ln() + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cached_forward_matches_full_forward() {
                let device = Device::flex();
        let config = ModelConfig::new(50, 32, 4, Some(2), 2, 8, 0.0, 0.0);
        let model: Llm = config.init_model(&device);

        let tokens = vec![1i32, 2, 3, 4, 5, 6];
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(tokens.clone(), [1, 6]),
            &device,
        );

        // Prefill path: the cache must reproduce the uncached logits exactly.
        let full = model.forward(input.clone()).into_data().try_to_vec::<f32>().unwrap();
        let mut caches: Vec<KvCache> =
            model.new_kv_caches(model.max_seq_len(), KvCacheDtype::F32, &device);
        let cached = model
            .forward_with_cache(input, 0, &mut caches)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(full.len(), cached.len());
        let max_diff = full
            .iter()
            .zip(cached.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "prefill cache diverged from the full forward pass: max_diff={max_diff}"
        );

        // Decode path: feeding one new token at start_pos=6 must match the
        // corresponding row of a single 7-token uncached forward.
        let full7 = model
            .forward(Tensor::<2, Int>::from_data(
                TensorData::new(vec![1i32, 2, 3, 4, 5, 6, 7], [1, 7]),
                &device,
            ))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let step = Tensor::<2, Int>::from_data(
            TensorData::new(vec![7i32], [1, 1]),
            &device,
        );
        let step_logits = model
            .forward_with_cache(step, 6, &mut caches)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let vocab = 50;
        let expected = &full7[6 * vocab..7 * vocab];
        let step_diff = expected
            .iter()
            .zip(step_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            step_diff < 1e-4,
            "decode cache diverged from the full forward pass: max_diff={step_diff}"
        );
        assert_eq!(caches[0].len, 7);
    }

    #[test]
    fn test_forward_with_cache_past_n_ctx_with_scaling() {
                let device = Device::flex();
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.set_rope_scaling(RopeScalingKind::Linear, 2.0);
        let model: Llm = config.init_model(&device);
        assert_eq!(model.max_seq_len(), 16);
        let mut caches: Vec<KvCache> =
            model.new_kv_caches(model.max_seq_len(), KvCacheDtype::F32, &device);

        let prefill = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1i32, 2, 3, 4, 5, 6], [1, 6]),
            &device,
        );
        let out = model.forward_with_cache(prefill, 0, &mut caches);
        assert_eq!(out.dims(), [1, 6, 50]);

        // start_pos 6 + 6 tokens = 12, past the trained n_ctx (8) but within
        // the extended cache (16).
        let next = Tensor::<2, Int>::from_data(
            TensorData::new(vec![7i32, 8, 9, 10, 11, 12], [1, 6]),
            &device,
        );
        let out = model.forward_with_cache(next, 6, &mut caches);
        assert_eq!(out.dims(), [1, 6, 50]);
        assert_eq!(caches[0].len, 12);
        assert_eq!(caches[0].capacity, 16);
        assert!(12 <= model.max_seq_len());
    }

    #[test]
    fn test_model_initialization_and_forward() {
        let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        let config = ModelConfig::new(100, 64, 4, Some(2), 2, 32, 0.0, 0.0);
        let model: Llm = config.init_model(&device);

        let input_data = TensorData::new(vec![1i32, 2, 3, 4], [1, 4]);
        let input = Tensor::<2, Int>::from_data(input_data, &device);

        let logits = model.forward(input);
        assert_eq!(logits.dims(), [1, 4, 100]);
    }

    #[test]
    fn test_causal_mask_isolates_position_zero() {
        let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        let config = ModelConfig::new(100, 64, 4, Some(2), 2, 32, 0.0, 0.0);
        let model: Llm = config.init_model(&device);

        let input_a = TensorData::new(vec![7i32, 11, 13, 17], [1, 4]);
        let input_b = TensorData::new(vec![7i32, 99, 13, 17], [1, 4]);
        let ta = Tensor::<2, Int>::from_data(input_a, &device);
        let tb = Tensor::<2, Int>::from_data(input_b, &device);

        let la = model.forward(ta);
        let lb = model.forward(tb);

        let da = la.into_data().try_to_vec::<f32>().unwrap_or_default();
        let db = lb.into_data().try_to_vec::<f32>().unwrap_or_default();

        let v = 100;
        let pos0_a = &da[0..v];
        let pos0_b = &db[0..v];

        let max_diff = pos0_a
            .iter()
            .zip(pos0_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let changed_pos1 = pos0_a
            .iter()
            .zip(pos0_b.iter())
            .filter(|(a, b)| ((*a) - (*b)).abs() > 1e-5)
            .count();

        eprintln!("[CAUSALITY TEST] max_diff pos0 = {max_diff:.6}, changed_pos1 = {changed_pos1}/{v}");
        assert!(max_diff < 1e-4, "Position 0 logits changed when token at position 1 changed (max_diff={max_diff}). Causal mask broken!");
    }

    /// Runs the same token sequence through the cached decode loop and through
    /// a single full forward, comparing every row. Exercises the ring buffer,
    /// the per-position RoPE angles and the window/sink mask together.
    fn assert_cached_matches_full_with_window(
        window: Option<usize>,
        sink_tokens: usize,
        kv_share_group: usize,
        mla_dim: Option<usize>,
    ) -> (usize, usize) {
                let device = Device::flex();
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 4, 8, 0.0, 0.0);
        config.sliding_window = window;
        config.sink_tokens = sink_tokens;
        config.kv_share_group = kv_share_group;
        config.mla_latent_dim = mla_dim;
        let model: Llm = config.init_model(&device);
        assert_eq!(model.new_kv_caches(1, KvCacheDtype::F32, &device).len(), config.n_kv_groups());

        let tokens: Vec<i32> = vec![3, 5, 7, 11, 13, 17];
        let full = model
            .forward(Tensor::<2, Int>::from_data(
                TensorData::new(tokens.clone(), [1, tokens.len()]),
                &device,
            ))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        let capacity = model.kv_capacity();
        let expected_capacity = window.map(|w| w + sink_tokens).unwrap_or(model.max_seq_len());
        assert_eq!(capacity, expected_capacity);

        let mut caches = model.new_kv_caches(capacity, KvCacheDtype::F32, &device);
        for (pos, &tok) in tokens.iter().enumerate() {
            let step = Tensor::<2, Int>::from_data(
                TensorData::new(vec![tok], [1, 1]),
                &device,
            );
            let logits = model
                .forward_with_cache(step, pos, &mut caches)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let expected = &full[pos * 50..(pos + 1) * 50];
            let diff = expected
                .iter()
                .zip(logits.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                diff < 1e-4,
                "cached decode diverged at pos {pos} (window={window:?}, sinks={sink_tokens}): max_diff={diff}"
            );
        }
        (capacity, model.max_seq_len())
    }

    #[test]
    fn test_cached_decode_matches_full_without_window() {
        let (capacity, max_seq) = assert_cached_matches_full_with_window(None, 0, 1, None);
        assert_eq!(capacity, max_seq);
    }

    #[test]
    fn test_cached_decode_matches_full_with_ring_window() {
        let (capacity, _) = assert_cached_matches_full_with_window(Some(2), 0, 1, None);
        // Bounded cache: only `window` slots, all overwritten by the end.
        assert_eq!(capacity, 2);
    }

    #[test]
    fn test_cached_decode_matches_full_with_sinks() {
        let (capacity, _) = assert_cached_matches_full_with_window(Some(2), 1, 1, None);
        assert_eq!(capacity, 3);
    }

    #[test]
    fn test_cached_decode_matches_full_with_kv_sharing() {
        // 4 layers, 2 per group -> 2 caches; training and cached paths must
        // agree on the shared-K/V computation.
        let (capacity, _) = assert_cached_matches_full_with_window(None, 0, 2, None);
        assert_eq!(capacity, 8);
    }

    #[test]
    fn test_cached_decode_matches_full_with_mla() {
        let (capacity, _) = assert_cached_matches_full_with_window(None, 0, 1, Some(16));
        assert_eq!(capacity, 8);
    }

    #[test]
    fn test_evict_slots_protect_sinks_and_recent() {
                let device = Device::flex();
        let mut cache =
            KvCache::preallocated(1, 1, 4, 8, KvCacheDtype::F32, &device);
        cache.enable_eviction_with(2, 8, false);
        // Fill the cache: rows 0..8 are appended in order.
        let slots = alloc_evict_slots(&mut cache, 0, 8, 2);
        assert_eq!(slots, (0..8).collect::<Vec<_>>());
        assert_eq!(cache.len, 8);
        // pos 8: cutoff 6 -> victim is the lowest-position row among 2..6.
        let slots = alloc_evict_slots(&mut cache, 8, 1, 2);
        assert_eq!(slots, vec![2]);
        assert_eq!(cache.positions[2], 8);
        // pos 9: cutoff 7 -> victim among 3..7 (position 8 is recent).
        let slots = alloc_evict_slots(&mut cache, 9, 1, 2);
        assert_eq!(slots, vec![3]);
        // Sinks and recent rows survive; the live set is exactly the budget.
        assert_eq!(cache.positions[0], 0);
        assert_eq!(cache.positions[1], 1);
        assert!(cache.positions.contains(&8));
        assert!(cache.positions.contains(&9));
        assert!(cache.positions.iter().all(|p| *p != usize::MAX));
    }

    #[test]
    fn test_evict_picks_lowest_score_and_resets_it() {
                let device = Device::flex();
        let mut cache =
            KvCache::preallocated(1, 1, 4, 6, KvCacheDtype::F32, &device);
        cache.enable_eviction_with(1, 6, false);
        let _ = alloc_evict_slots(&mut cache, 0, 6, 1);
        // Position 3 is the least important; position 2 is hot.
        cache.scores = Some(Tensor::from_data(
            TensorData::new(vec![9.0, 9.0, 50.0, 0.5, 9.0, 9.0], [6]),
            &device,
        ));
        // pos 6: cutoff 5 -> candidates 1..5, lowest score is position 3.
        let slots = alloc_evict_slots(&mut cache, 6, 1, 1);
        assert_eq!(slots, vec![3]);
        let scores = cache
            .scores
            .clone()
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(scores[3], 0.0, "victim score must reset");
        assert_eq!(scores[2], 50.0, "hot rows keep their score");
    }

    #[test]
    fn test_sample_keep_mask_density_and_stochasticity() {
                let device = Device::flex();
        let n_ctx = 4096;
        let sinks = 4;
        let recent = 64;
        let budget = 516; // hitters = 448, p = 448 / 4028 ~ 0.111
        let mask = sample_keep_mask(
            n_ctx, n_ctx, budget, recent, sinks, &device,
        );
        assert_eq!(mask.dims(), [1, n_ctx]);
        let kept = mask
            .clone()
            .into_data()
            .try_to_vec::<bool>()
            .unwrap()
            .iter()
            .filter(|b| **b)
            .count();
        let expected = 448.0f64;
        // Binomial sd ~ 20; allow 4 sd of slack.
        assert!(
            (kept as f64 - expected).abs() < 80.0,
            "keep-mask density off: kept {kept}, expected ~{expected}"
        );
        let other = sample_keep_mask(
            n_ctx, n_ctx, budget, recent, sinks, &device,
        );
        assert_ne!(
            mask.into_data().try_to_vec::<bool>().unwrap(),
            other.into_data().try_to_vec::<bool>().unwrap(),
            "each forward must resample the keep set"
        );
    }

    #[test]
    fn test_evict_cached_decode_stays_bounded_and_finite() {
        
        fn run(dtype: KvCacheDtype) {
            let device = Device::flex();
            let mut config = ModelConfig::new(50, 32, 4, Some(2), 2, 32, 0.0, 0.0);
            config.sink_tokens = 2;
            config.kv_eviction = Some("h2o".into());
            config.kv_budget = Some(8);
            config.kv_recent = 2;
            let geo = config.eviction_geometry().unwrap().unwrap();
            let model: Llm = config.init_model(&device);
            let capacity = geo.budget.min(model.max_seq_len());
            let mut caches = model.new_kv_caches(capacity, dtype, &device);
            for c in caches.iter_mut() {
                c.enable_eviction_with(geo.recent, capacity, false);
            }
            for pos in 0..20 {
                let tok = (pos % config.vocab_size) as i32;
                let input = Tensor::<2, Int>::from_data(
                    TensorData::new(vec![tok], [1, 1]),
                    &device,
                );
                let logits = model.forward_with_cache(input, pos, &mut caches);
                let vals = logits.into_data().try_to_vec::<f32>().unwrap();
                assert!(
                    vals.iter().all(|v| v.is_finite()),
                    "{} eviction produced non-finite logits at pos {pos}",
                    dtype.as_str()
                );
            }
            for c in &caches {
                assert_eq!(c.len, capacity, "cache must stay at the budget");
                let mut alive: Vec<usize> = c
                    .positions
                    .iter()
                    .copied()
                    .filter(|p| *p != usize::MAX)
                    .collect();
                let before = alive.len();
                alive.sort_unstable();
                alive.dedup();
                assert_eq!(alive.len(), before, "slot positions must be unique");
            }
        }

        // The f16 cache path is covered by the WGPU tests; f32/int8/q4
        // exercise every eviction writer branch on the CPU backend.
        for dtype in [KvCacheDtype::F32, KvCacheDtype::Int8, KvCacheDtype::Q4] {
            run(dtype);
        }
    }

    #[test]
    fn test_spill_replay_restores_dense_context() {
                let device = Device::flex();
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 16, 0.0, 0.0);
        config.kv_eviction = Some("h2o".into());
        config.kv_budget = Some(4);
        config.kv_recent = 0;
        config.sink_tokens = 0;
        let model: Llm = config.init_model(&device);

        let tokens: Vec<i32> = (0..13).map(|i| (i * 7) % 50).collect();
        let step = |tok: i32| {
            Tensor::<2, Int>::from_data(
                TensorData::new(vec![tok], [1, 1]),
                &device,
            )
        };

        // Reference: a dense cache with room for the whole sequence.
        let mut dense = model.new_kv_caches(16, KvCacheDtype::F32, &device);
        let mut dense_logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            dense_logits = model
                .forward_with_cache(step(tok), pos, &mut dense)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
        }

        // Bounded cache with spill: the first 12 tokens evict 8 rows to RAM.
        let mut caches = model.new_kv_caches(16, KvCacheDtype::F32, &device);
        for c in caches.iter_mut() {
            c.enable_eviction_with(0, 4, true);
        }
        for (pos, &tok) in tokens[..12].iter().enumerate() {
            model.forward_with_cache(step(tok), pos, &mut caches);
        }
        assert_eq!(caches[0].len, 4, "the budget bounds the resident rows");
        assert_eq!(caches[0].spilled_total, 8, "every eviction is archived");

        // Grow the budget to the full context and replay the archive.
        let replayed = caches[0].grow_and_replay(12);
        assert_eq!(replayed, 8);
        assert_eq!(caches[0].len, 12);
        let mut positions: Vec<usize> = caches[0]
            .positions
            .iter()
            .copied()
            .filter(|p| *p != usize::MAX)
            .collect();
        positions.sort_unstable();
        assert_eq!(positions, (0..12).collect::<Vec<_>>());

        // The next token must attend the restored dense context.
        let cached = model
            .forward_with_cache(step(tokens[12]), 12, &mut caches)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let diff = dense_logits
            .iter()
            .zip(cached.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff < 1e-4,
            "replayed cache diverged from the dense context: max_diff={diff}"
        );
    }

    #[test]
    fn test_spill_replay_prefers_high_scores() {
                let device = Device::flex();
        for dtype in [KvCacheDtype::F32, KvCacheDtype::Int8, KvCacheDtype::Q4] {
            let mut cache =
                KvCache::preallocated(1, 2, 4, 6, dtype, &device);
            cache.enable_eviction_with(0, 4, true);
            let _ = alloc_evict_slots(&mut cache, 0, 4, 0);
            // Position 1 is the coldest row, then position 3.
            cache.scores = Some(Tensor::from_data(
                TensorData::new(vec![5.0, 1.0, 9.0, 2.0, 0.0, 0.0], [6]),
                &device,
            ));
            let slots = alloc_evict_slots(&mut cache, 4, 2, 0);
            assert_eq!(slots, vec![1, 3]);

            let spill = cache.spill.as_ref().expect("spill enabled");
            let mut archived: Vec<(usize, f32)> =
                spill.rows.iter().map(|r| (r.position, r.score)).collect();
            archived.sort_by_key(|r| r.0);
            assert_eq!(archived, vec![(1, 1.0), (3, 2.0)]);

            // Growth replays both rows and restores the full six-row window.
            let replayed = cache.grow_and_replay(2);
            assert_eq!(replayed, 2);
            assert_eq!(cache.len, 6);
            let mut positions: Vec<usize> = cache
                .positions
                .iter()
                .copied()
                .filter(|p| *p != usize::MAX)
                .collect();
            positions.sort_unstable();
            assert_eq!(positions, vec![0, 1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn test_evict_mixed_batch_reuses_unwritten_slots() {
                let device = Device::flex();
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 16, 0.0, 0.0);
        config.kv_eviction = Some("h2o".into());
        config.kv_budget = Some(4);
        config.kv_recent = 1;
        config.sink_tokens = 0;
        let model: Llm = config.init_model(&device);
        let mut caches = model.new_kv_caches(8, KvCacheDtype::F32, &device);
        for c in caches.iter_mut() {
            c.enable_eviction_with(1, 4, true);
        }
        // A single 6-token call appends 4 rows and evicts 2 in the same batch;
        // slots assigned earlier in the call must be reused without archiving
        // (their data does not exist yet) and `len` must stay at the budget.
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1, 2, 3, 4, 5, 6], [1, 6]),
            &device,
        );
        let logits = model.forward_with_cache(input, 0, &mut caches);
        let values = logits.into_data().try_to_vec::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        assert_eq!(caches[0].len, 4);
        assert_eq!(caches[0].spilled_total, 0, "unwritten slots are not archived");
        let mut positions: Vec<usize> = caches[0]
            .positions
            .iter()
            .copied()
            .filter(|p| *p != usize::MAX)
            .collect();
        positions.sort_unstable();
        assert_eq!(positions, vec![2, 3, 4, 5], "only the newest rows survive");
    }

    #[test]
    fn test_mla_cache_is_latent_only() {
                let device = Device::flex();
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 2, 8, 0.0, 0.0);
        config.mla_latent_dim = Some(16);
        let model: Llm = config.init_model(&device);
        let caches = model.new_kv_caches(4, KvCacheDtype::F32, &device);
        assert_eq!(caches.len(), 2);
        assert_eq!(caches[0].k.as_ref().unwrap().dims(), [1, 1, 4, 16]);
        assert!(caches[0].v.is_none(), "MLA caches only the latent");
    }

    #[test]
    fn test_mla_weight_folding_matches_expansion() {
                let device = Device::flex();
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.mla_latent_dim = Some(16);
        let model: Llm = config.init_model(&device);
        let mla = &model.mla[0];

        let x = Tensor::<2>::from_data(
            TensorData::new(
                (0..32).map(|i| (i as f32) * 0.01 - 0.16).collect::<Vec<f32>>(),
                [1, 32],
            ),
            &device,
        );
        let direct = mla
            .k_up
            .forward(mla.kv_down.forward(x.clone()))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        // Export folds W_eff = kv_down.weight @ k_up.weight (Burn layout) and
        // applies it as x @ W_eff.
        let folded_w = mla.kv_down.weight.val().matmul(mla.k_up.weight.val());
        let folded = x
            .matmul(folded_w)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let diff = direct
            .iter()
            .zip(folded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-5, "folded MLA weights differ from expansion: {diff}");
    }

    #[test]
    fn test_quantized_rows_roundtrip() {
                let device = Device::flex();
        let data: Vec<f32> = (0..8 * 64)
            .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.013)
            .collect();
        let x = Tensor::<2>::from_data(TensorData::new(data, [8, 64]), &device);

        for dtype in [KvCacheDtype::Int8, KvCacheDtype::Q4] {
            let (packed, scales) = quantize_rows(x.clone(), dtype);
            assert_eq!(packed.dims(), [8, 64 / dtype.pack()]);
            let back = dequantize_rows(packed, scales.clone(), dtype);
            let xv = x.clone().into_data().try_to_vec::<f32>().unwrap();
            let bv = back.into_data().try_to_vec::<f32>().unwrap();
            let sv = scales.into_data().try_to_vec::<f32>().unwrap();
            let max_scale = sv.iter().cloned().fold(0.0f32, f32::max);
            let max_err = xv
                .iter()
                .zip(bv.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err <= max_scale / 2.0 + 1e-6,
                "{} roundtrip error {max_err} exceeds half the max scale {max_scale}",
                dtype.as_str()
            );
            assert!(max_err > 0.0, "quantization should not be lossless");
        }
    }

    #[test]
    fn test_zero_packed_word_dequantizes_to_zero() {
                let device = Device::flex();
        for dtype in [KvCacheDtype::Int8, KvCacheDtype::Q4] {
            let row_len = 64;
            let words = row_len / dtype.pack();
            let packed =
                Tensor::<2, Int>::zeros([2, words], &device);
            let scales =
                Tensor::<2>::ones([2, row_len / quant_block(row_len)], &device);
            let back = dequantize_rows(packed, scales, dtype);
            let values = back.into_data().try_to_vec::<f32>().unwrap();
            assert!(
                values.iter().all(|v| *v == 0.0),
                "empty {} slots must dequantize to exactly zero",
                dtype.as_str()
            );
        }
    }

    #[test]
    fn test_quantized_cache_close_to_f32_cache() {
        // Same decode path, only the cache storage changes. Quantization error
        // is bounded per block; the tolerances cover the accumulated effect
        // over five decode steps.
        
        fn check(
            device: &Device,
            config: ModelConfig,
            dtype: KvCacheDtype,
            tol: f32,
        ) {
            let model: Llm = config.init_model(device);
            let tokens: Vec<i32> = vec![3, 5, 7, 11, 13];

            let run = |dt: KvCacheDtype| -> Vec<Vec<f32>> {
                let mut caches = model.new_kv_caches(8, dt, device);
                tokens
                    .iter()
                    .enumerate()
                    .map(|(pos, &tok)| {
                        let step = Tensor::<2, Int>::from_data(
                            TensorData::new(vec![tok], [1, 1]),
                            device,
                        );
                        model
                            .forward_with_cache(step, pos, &mut caches)
                            .into_data()
                            .try_to_vec::<f32>()
                            .unwrap()
                    })
                    .collect()
            };
            let reference = run(KvCacheDtype::F32);
            let quantized = run(dtype);
            for pos in 0..tokens.len() {
                let diff = reference[pos]
                    .iter()
                    .zip(quantized[pos].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    diff < tol,
                    "{} cache diverged from f32 at pos {pos}: max_diff={diff}",
                    dtype.as_str()
                );
            }
        }

        let mut mla = ModelConfig::new(50, 64, 4, Some(2), 2, 16, 0.0, 0.0);
        mla.mla_latent_dim = Some(32);
        let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        for (dtype, tol) in [(KvCacheDtype::Int8, 5e-2), (KvCacheDtype::Q4, 2e-1)] {
            check(
                &device,
                ModelConfig::new(50, 64, 4, Some(2), 2, 16, 0.0, 0.0),
                dtype,
                tol,
            );
            check(
                &device,
                ModelConfig::new(50, 64, 4, Some(2), 2, 16, 0.0, 0.0),
                dtype,
                tol,
            );
            check(&device, mla.clone(), dtype, tol);
        }
    }

    #[test]
    fn test_f16_cache_close_to_f32_cache() {
        // Same decode path, only the cache dtype changes: isolates f16 rounding
        // from any backend-level differences between batched and single-token
        // forwards.
        
        fn check(device: &Device) {
            let config = ModelConfig::new(50, 32, 4, Some(2), 2, 8, 0.0, 0.0);
            let model: Llm = config.init_model(device);
            let tokens: Vec<i32> = vec![3, 5, 7, 11, 13];

            let run = |f16: bool| -> Vec<Vec<f32>> {
                let mut caches = model.new_kv_caches(8, if f16 { KvCacheDtype::F16 } else { KvCacheDtype::F32 }, device);
                tokens
                    .iter()
                    .enumerate()
                    .map(|(pos, &tok)| {
                        let step = Tensor::<2, Int>::from_data(
                            TensorData::new(vec![tok], [1, 1]),
                            device,
                        );
                        model
                            .forward_with_cache(step, pos, &mut caches)
                            .into_data()
                            .try_to_vec::<f32>()
                            .unwrap()
                    })
                    .collect()
            };
            let f32_logits = run(false);
            let f16_logits = run(true);
            for pos in 0..tokens.len() {
                let diff = f32_logits[pos]
                    .iter()
                    .zip(f16_logits[pos].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    diff < 5e-2,
                    "f16 cache diverged from f32 cache at pos {pos}: max_diff={diff}"
                );
            }
        }

        let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        check(&device);
    }

    #[test]
    fn test_f16_cache_really_stores_f16() {
                let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        let config = ModelConfig::new(50, 32, 4, Some(2), 2, 8, 0.0, 0.0);
        let model: Llm = config.init_model(&device);
        let caches = model.new_kv_caches(8, KvCacheDtype::F16, &device);
        assert_eq!(caches[0].k.as_ref().unwrap().dtype(), DType::F16);
        assert_eq!(caches[0].v.as_ref().unwrap().dtype(), DType::F16);
    }

    #[test]
    fn test_kv_sharing_reduces_cache_count() {
                let device = Device::flex();
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 4, 8, 0.0, 0.0);
        config.kv_share_group = 4;
        let model: Llm = config.init_model(&device);
        assert_eq!(model.new_kv_caches(8, KvCacheDtype::F32, &device).len(), 1);
        // The grouped model is a different function than the ungrouped one.
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1i32, 2, 3, 4], [1, 4]),
            &device,
        );
        let grouped = model.forward(input.clone()).into_data().try_to_vec::<f32>().unwrap();
        config.kv_share_group = 1;
        let plain: Llm = config.init_model(&device);
        let plain_out = plain.forward(input).into_data().try_to_vec::<f32>().unwrap();
        let diff = grouped
            .iter()
            .zip(plain_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff > 1e-4, "grouped KV must change the computation (diff={diff})");
    }

    #[test]
    fn test_ring_cache_keeps_only_window_positions() {
                let device = Device::flex();
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.sliding_window = Some(2);
        let model: Llm = config.init_model(&device);
        let mut caches = model.new_kv_caches(model.kv_capacity(), KvCacheDtype::F32, &device);

        for pos in 0..6usize {
            let step = Tensor::<2, Int>::from_data(
                TensorData::new(vec![pos as i32 + 1], [1, 1]),
                &device,
            );
            model.forward_with_cache(step, pos, &mut caches);
        }

        let cache = &caches[0];
        assert_eq!(cache.len, 2);
        let mut stored: Vec<usize> = cache
            .positions
            .iter()
            .copied()
            .filter(|&p| p != usize::MAX)
            .collect();
        stored.sort_unstable();
        assert_eq!(stored, vec![4, 5]);
    }

    #[test]
    fn test_sliding_window_limits_attention_span() {
                let device = Device::flex();
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.sliding_window = Some(2);
        let model: Llm = config.init_model(&device);

        let run = |first: i32| -> Vec<f32> {
            model
                .forward(Tensor::<2, Int>::from_data(
                    TensorData::new(vec![first, 11, 13, 17], [1, 4]),
                    &device,
                ))
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
        };
        let a = run(7);
        let b = run(49);

        let row_diff = |row: usize| -> f32 {
            a[row * 50..(row + 1) * 50]
                .iter()
                .zip(b[row * 50..(row + 1) * 50].iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        // Row 1 sees position 0 (distance 1), row 3 does not (distance 3 > 2).
        assert!(row_diff(1) > 1e-4, "window should include the previous token");
        assert!(row_diff(3) < 1e-4, "window must drop positions older than 2");
    }

    #[test]
    fn test_sink_tokens_keep_first_position_visible() {
                let device = Device::flex();
        let mut config = ModelConfig::new(50, 32, 4, Some(2), 1, 8, 0.0, 0.0);
        config.sliding_window = Some(2);
        config.sink_tokens = 1;
        let model: Llm = config.init_model(&device);

        let run = |first: i32| -> Vec<f32> {
            model
                .forward(Tensor::<2, Int>::from_data(
                    TensorData::new(vec![first, 11, 13, 17], [1, 4]),
                    &device,
                ))
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
        };
        let a = run(7);
        let b = run(49);
        let diff = a[3 * 50..4 * 50]
            .iter()
            .zip(b[3 * 50..4 * 50].iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff > 1e-4,
            "sink token at position 0 must stay visible at row 3 (diff={diff})"
        );
    }
    /// Regression test for the burn-wgpu `autotune` GEMV bug: with autotune
    /// enabled, the M=1 matmul path (token-by-token generation) picked a GEMV
    /// kernel that returned wrong values for output widths >= 32, so cached
    /// decoding diverged from the batched forward pass. The non-autotuned
    /// fallback kernels are exact; this test fails if autotune is re-enabled
    /// or the fallback regresses.
    #[test]
    fn test_wgpu_cached_decode_matches_full_forward() {
        
        fn check(device: &Device) {
            let config = ModelConfig::new(50, 32, 4, Some(2), 2, 8, 0.0, 0.0);
            let model: Llm = config.init_model(device);
            let input = Tensor::<2, Int>::from_data(
                TensorData::new(vec![1i32, 2, 3, 4, 5, 6], [1, 6]),
                device,
            );
            let full = model
                .forward(input.clone())
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();

            let mut caches = model.new_kv_caches(model.max_seq_len(), KvCacheDtype::F32, device);
            let cached = model
                .forward_with_cache(input, 0, &mut caches)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let prefill_diff = full
                .iter()
                .zip(cached.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                prefill_diff < 1e-4,
                "prefill cache diverged from the full forward: max_diff={prefill_diff}"
            );

            // One decode step must match the last row of a 7-token forward.
            let full7 = model
                .forward(Tensor::<2, Int>::from_data(
                    TensorData::new(vec![1i32, 2, 3, 4, 5, 6, 7], [1, 7]),
                    device,
                ))
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let step =
                Tensor::<2, Int>::from_data(TensorData::new(vec![7i32], [1, 1]), device);
            let step_logits = model
                .forward_with_cache(step, 6, &mut caches)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let vocab = 50;
            let step_diff = full7[6 * vocab..7 * vocab]
                .iter()
                .zip(step_logits.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                step_diff < 1e-4,
                "single-token decode diverged from the full forward: max_diff={step_diff}"
            );
        }

        let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        check(&device);
        check(&device);
    }
