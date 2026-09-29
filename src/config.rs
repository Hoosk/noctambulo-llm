use serde::{Deserialize, Serialize};

/// Compute precision used for training. Records are always serialized with
/// `FullPrecisionSettings` (f32), so a model trained in f16 can be loaded and
/// exported by the f32 inference path without conversion steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    #[default]
    F32,
    F16,
}

impl DType {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => Ok(Self::F32),
            "f16" | "fp16" | "half" => Ok(Self::F16),
            // WGSL has no bf16 type; cubecl panics deep in the shader compiler.
            "bf16" | "bfloat16" => Err(
                "bf16 is not supported by the WGPU/WGSL backend; use f16 (or f32)".into(),
            ),
            other => Err(format!("Unknown dtype '{other}'. Available: f32, f16")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

/// RoPE context-extension strategy. `Linear` and `Yarn` map to the GGUF
/// `rope.scaling.type` values understood by llama.cpp; `Ntk` is our extension
/// (dynamic NTK base adjustment) and is exported as `ntk`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeScalingKind {
    #[default]
    None,
    Linear,
    Ntk,
    Yarn,
}

impl RopeScalingKind {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "none" | "off" => Ok(Self::None),
            "linear" => Ok(Self::Linear),
            "ntk" | "dynamic" | "dynamic-ntk" => Ok(Self::Ntk),
            "yarn" => Ok(Self::Yarn),
            other => Err(format!(
                "Unknown rope scaling '{other}'. Available: none, linear, ntk, yarn"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Ntk => "ntk",
            Self::Yarn => "yarn",
        }
    }
}

fn default_rope_factor() -> f32 {
    1.0
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    #[serde(default)]
    pub n_kv_head: usize,
    pub n_layer: usize,
    pub n_ctx: usize,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default)]
    pub attn_softcap: f32,
    /// RoPE context-extension strategy (`none`, `linear`, `ntk`, `yarn`).
    /// Absent in older configs, which is equivalent to `none`.
    #[serde(default)]
    pub rope_scaling: Option<String>,
    /// Context-extension factor (e.g. `2.0` doubles the usable context).
    #[serde(default = "default_rope_factor")]
    pub rope_factor: f32,
}

impl ModelConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        n_embd: usize,
        n_head: usize,
        n_kv_head: Option<usize>,
        n_layer: usize,
        n_ctx: usize,
        dropout: f64,
        attn_softcap: f32,
    ) -> Self {
        assert!(
            n_embd.is_multiple_of(n_head),
            "n_embd ({}) must be evenly divisible by n_head ({})",
            n_embd,
            n_head
        );

        let kv_head = n_kv_head.unwrap_or(n_head);
        let kv_head = if kv_head == 0 { n_head } else { kv_head };

        assert!(
            n_head.is_multiple_of(kv_head),
            "n_head ({}) must be divisible by n_kv_head ({}) for GQA",
            n_head,
            kv_head
        );

        assert!(vocab_size > 0, "vocab_size must be greater than 0");
        assert!(n_ctx > 0, "n_ctx must be greater than 0");
        assert!(n_layer > 0, "n_layer must be greater than 0");

        Self {
            vocab_size,
            n_embd,
            n_head,
            n_kv_head: kv_head,
            n_layer,
            n_ctx,
            dropout,
            attn_softcap,
            rope_scaling: None,
            rope_factor: 1.0,
        }
    }

    /// Parsed RoPE strategy; unknown strings fall back to `none`.
    pub fn rope_kind(&self) -> RopeScalingKind {
        self.rope_scaling
            .as_deref()
            .and_then(|s| RopeScalingKind::parse(s).ok())
            .unwrap_or_default()
    }

    pub fn set_rope_scaling(&mut self, kind: RopeScalingKind, factor: f32) {
        self.rope_scaling = match kind {
            RopeScalingKind::None => None,
            other => Some(other.as_str().to_string()),
        };
        self.rope_factor = if factor.is_finite() && factor > 0.0 { factor } else { 1.0 };
    }

    pub fn get_n_kv_head(&self) -> usize {
        if self.n_kv_head == 0 {
            self.n_head
        } else {
            self.n_kv_head
        }
    }

    /// Total trainable parameters for this architecture (attention biases and
    /// tied embeddings included; MLP hidden size aligned to 256 like the model).
    pub fn param_count(&self) -> usize {
        let n_embd = self.n_embd;
        let kv_dim = self.get_n_kv_head() * (n_embd / self.n_head);
        let hidden = ((8 * n_embd) / 3).div_ceil(256) * 256;

        let embed = self.vocab_size * n_embd;
        let q = n_embd * n_embd + n_embd;
        let k = n_embd * kv_dim + kv_dim;
        let v = n_embd * kv_dim + kv_dim;
        let out = n_embd * n_embd;
        let norms = 2 * n_embd;
        let mlp = 3 * n_embd * hidden;
        let block = q + k + v + out + norms + mlp;

        embed + self.n_layer * block + n_embd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_parse_and_default() {
        assert_eq!(DType::parse("f32").unwrap(), DType::F32);
        assert_eq!(DType::parse("FP16").unwrap(), DType::F16);
        assert_eq!(DType::parse("half").unwrap(), DType::F16);
        assert_eq!(DType::default(), DType::F32);
        assert_eq!(DType::F16.as_str(), "f16");
        assert!(DType::parse("int8").is_err());
        // bf16 has no WGSL support: reject with a helpful message.
        let err = DType::parse("bfloat16").unwrap_err();
        assert!(err.contains("bf16 is not supported"), "{err}");
    }

    #[test]
    fn test_param_count_matches_known_architecture() {
        let cfg = ModelConfig::new(8000, 256, 4, Some(2), 4, 256, 0.1, 0.0);
        let hidden = 768usize;
        let kv_dim = 2 * 64;
        let block = (256 * 256 + 256) + 2 * (256 * kv_dim + kv_dim) + 256 * 256 + 2 * 256 + 3 * 256 * hidden;
        let expected = 8000 * 256 + 4 * block + 256;
        assert_eq!(cfg.param_count(), expected);
    }

    #[test]
    fn test_rope_scaling_kind_parse() {
        assert_eq!(RopeScalingKind::parse("none").unwrap(), RopeScalingKind::None);
        assert_eq!(RopeScalingKind::parse("").unwrap(), RopeScalingKind::None);
        assert_eq!(RopeScalingKind::parse("LINEAR").unwrap(), RopeScalingKind::Linear);
        assert_eq!(RopeScalingKind::parse("dynamic").unwrap(), RopeScalingKind::Ntk);
        assert_eq!(RopeScalingKind::parse("yarn").unwrap(), RopeScalingKind::Yarn);
        assert!(RopeScalingKind::parse("magic").is_err());
        assert_eq!(RopeScalingKind::Linear.as_str(), "linear");
    }

    #[test]
    fn test_rope_fields_default_when_absent() {
        let json = r#"{"vocab_size":100,"n_embd":64,"n_head":4,"n_layer":2,"n_ctx":32}"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.rope_kind(), RopeScalingKind::None);
        assert_eq!(cfg.rope_factor, 1.0);
    }

    #[test]
    fn test_set_rope_scaling_roundtrip() {
        let mut cfg = ModelConfig::new(100, 64, 4, Some(2), 2, 32, 0.0, 0.0);
        cfg.set_rope_scaling(RopeScalingKind::Yarn, 4.0);
        assert_eq!(cfg.rope_kind(), RopeScalingKind::Yarn);
        assert_eq!(cfg.rope_factor, 4.0);

        let bytes = serde_json::to_vec(&cfg).unwrap();
        let back: ModelConfig = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.rope_kind(), RopeScalingKind::Yarn);
        assert_eq!(back.rope_factor, 4.0);

        // None removes the key entirely and invalid factors reset to 1.0.
        cfg.set_rope_scaling(RopeScalingKind::None, -2.0);
        assert_eq!(cfg.rope_scaling, None);
        assert_eq!(cfg.rope_factor, 1.0);
    }
}
