    use super::*;
    use crate::storage;
    use burn::tensor::DeviceKind;

    fn compare(config: ModelConfig) -> f32 {
        let cpu = Device::flex();
        let gpu = Device::wgpu(DeviceKind::DefaultDevice);
        let model = config.init_model(&cpu);
        let dir = std::env::temp_dir().join(format!(
            "noctambulo_parity_{}_{}",
            config.n_embd,
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = storage::model_weights_path(&dir);
        storage::save_module(&model, &path).unwrap();
        let gpu_model = storage::load_module(config.init_model(&gpu), &path).unwrap();

        let tokens: Vec<i32> = (0..8).map(|i| ((i * 7 + 3) % config.vocab_size) as i32).collect();
        let cpu_in = Tensor::<2, Int>::from_data(
            TensorData::new(tokens.clone(), [1, 8]),
            &cpu,
        );
        let gpu_in = Tensor::<2, Int>::from_data(
            TensorData::new(tokens, [1, 8]),
            &gpu,
        );
        let a = model
            .forward(cpu_in)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        let b = gpu_model
            .forward(gpu_in)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_wgpu_forward_matches_flex_tiny() {
        let mut config = ModelConfig::new(1000, 128, 4, Some(2), 2, 64, 0.0, 0.0);
        config.ffn_hidden = Some(512);
        let diff = compare(config);
        assert!(diff < 1e-3, "wgpu vs flex logits differ by {diff}");
    }

    #[test]
    fn test_wgpu_forward_matches_flex_medium() {
        let mut config = ModelConfig::new(1000, 256, 4, Some(2), 2, 64, 0.0, 0.0);
        config.ffn_hidden = Some(1024);
        let diff = compare(config);
        assert!(diff < 1e-3, "wgpu vs flex logits differ by {diff}");
    }

    #[test]
    fn test_wgpu_forward_matches_flex_qwen_like() {
        let mut config = ModelConfig::new(1000, 896, 14, Some(2), 2, 64, 0.0, 0.0);
        config.ffn_hidden = Some(4864);
        config.rope_base = 1_000_000.0;
        let diff = compare(config);
        assert!(diff < 1e-3, "wgpu vs flex logits differ by {diff}");
    }
