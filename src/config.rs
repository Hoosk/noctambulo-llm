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
    /// Sliding attention window in tokens. `None` (default) is full causal
    /// attention; setting it bounds the KV cache to `window + sink_tokens`.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Initial tokens kept attendable forever (attention sinks).
    #[serde(default)]
    pub sink_tokens: usize,
    /// Number of consecutive layers that share one K/V projection and cache
    /// (cross-layer KV reuse, DeepSeek CSA-style). `1` disables sharing.
    #[serde(default = "default_kv_share_group")]
    pub kv_share_group: usize,
    /// Multi-head Latent Attention: when set, K/V are compressed into a
    /// latent of this size and only the latent is cached (DeepSeek MLA-style).
    #[serde(default)]
    pub mla_latent_dim: Option<usize>,
    /// Heavy-hitter KV eviction policy (`"h2o"`). Absent/`none` keeps the
    /// dense (or sliding-window) cache.
    #[serde(default)]
    pub kv_eviction: Option<String>,
    /// Resident cache rows including sinks and recent tokens (required when
    /// `kv_eviction` is set).
    #[serde(default)]
    pub kv_budget: Option<usize>,
    /// Most recent tokens protected from eviction.
    #[serde(default = "default_kv_recent")]
    pub kv_recent: usize,
    /// FFN hidden size. `None` uses the engine default
    /// `ceil((8/3)·n_embd / 256)·256`; imported third-party models store their
    /// real `intermediate_size`.
    #[serde(default)]
    pub ffn_hidden: Option<usize>,
    /// RMSNorm epsilon (`rms_norm_eps` in HF configs). Older checkpoints omit
    /// it, which defaults to the engine's original `1e-6`.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base frequency (`rope_theta` in HF configs; `10000` for the
    /// original models, `1e6` for Qwen2.5, `5e5` for Llama-3).
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    /// Origin of an imported checkpoint (`gguf` / `hf`); native training
    /// leaves it `None`.
    #[serde(default)]
    pub import_source: Option<String>,
    /// Checkpoint architecture version. Bumped whenever the module layout
    /// changes; older versions cannot be loaded (records are positional).
    #[serde(default = "default_arch_version")]
    pub arch_version: u32,
}

/// Resolved heavy-hitter eviction geometry: `budget = sinks + recent + hitters`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictionGeometry {
    /// Resident rows in the bounded cache.
    pub budget: usize,
    /// Most recent tokens always kept.
    pub recent: usize,
    /// Attention sinks always kept (initial tokens).
    pub sinks: usize,
    /// Old rows kept by accumulated attention score.
    pub hitters: usize,
}

/// Current module layout. 1 = pre-KV-cache-refactor, 2 = dynamic RoPE +
/// sliding window, 3 = grouped cross-layer KV projections, 4 = MLA latent,
/// 5 = configurable FFN hidden size / RMSNorm eps / RoPE base (third-party
/// imports). Versions 4 and 5 share the layout when the new fields keep their
/// defaults, so both load.
pub const ARCH_VERSION: u32 = 5;

/// Oldest architecture version this build can load.
pub const MIN_ARCH_VERSION: u32 = 4;

fn default_kv_share_group() -> usize {
    1
}

fn default_kv_recent() -> usize {
    64
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_base() -> f32 {
    10000.0
}

fn default_arch_version() -> u32 {
    1
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
            sliding_window: None,
            sink_tokens: 0,
            kv_share_group: 1,
            mla_latent_dim: None,
            kv_eviction: None,
            kv_budget: None,
            kv_recent: default_kv_recent(),
            ffn_hidden: None,
            rms_norm_eps: default_rms_norm_eps(),
            rope_base: default_rope_base(),
            import_source: None,
            arch_version: ARCH_VERSION,
        }
    }

    /// Whether the model compresses K/V into an MLA latent.
    pub fn use_mla(&self) -> bool {
        self.mla_latent_dim.is_some_and(|d| d > 0)
    }

    /// Latent dimension when MLA is enabled.
    pub fn mla_dim(&self) -> usize {
        self.mla_latent_dim.filter(|d| *d > 0).unwrap_or(0)
    }

    /// Resolved heavy-hitter eviction geometry. `None` means the cache is
    /// dense (or ring-bounded by `sliding_window`). Errs on unknown policies,
    /// on `sliding_window` + eviction together, and on budgets that leave no
    /// room for at least one heavy hitter.
    pub fn eviction_geometry(&self) -> Result<Option<EvictionGeometry>, String> {
        match self
            .kv_eviction
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            None => return Ok(None),
            Some("") | Some("none") | Some("off") => return Ok(None),
            Some("h2o") => {}
            Some(other) => {
                return Err(format!(
                    "Unknown KV eviction policy '{other}'. Available: none, h2o"
                ))
            }
        }
        if self.sliding_window.is_some() {
            return Err(
                "sliding_window and kv_eviction are mutually exclusive: use one bounded-cache policy"
                    .into(),
            );
        }
        let budget = self
            .kv_budget
            .filter(|b| *b > 0)
            .ok_or_else(|| "kv_eviction=h2o requires kv_budget (> 0)".to_string())?;
        let sinks = self.sink_tokens;
        let recent = self.kv_recent;
        if budget <= sinks + recent {
            return Err(format!(
                "kv_budget ({budget}) must exceed sink_tokens ({sinks}) + kv_recent ({recent}) \
                 so at least one heavy hitter fits"
            ));
        }
        Ok(Some(EvictionGeometry {
            budget,
            recent,
            sinks,
            hitters: budget - sinks - recent,
        }))
    }

    /// Applies a CLI/API eviction override: `policy` replaces the configured
    /// policy (empty keeps it) and `budget`/`recent` (0 keeps the configured
    /// value).
    pub fn set_kv_eviction(&mut self, policy: &str, budget: usize, recent: usize) {
        let policy = policy.trim();
        if !policy.is_empty() {
            self.kv_eviction = match policy.to_ascii_lowercase().as_str() {
                "none" | "off" => None,
                other => Some(other.to_string()),
            };
        }
        if budget > 0 {
            self.kv_budget = Some(budget);
        }
        if recent > 0 {
            self.kv_recent = recent;
        }
    }

    /// Number of layer groups (each group owns one K/V projection and cache).
    pub fn n_kv_groups(&self) -> usize {
        let group = self.kv_share_group.max(1);
        self.n_layer.div_ceil(group)
    }

    /// Rejects checkpoints saved with an incompatible module layout.
    pub fn check_arch_version(&self, path: &str) -> Result<(), String> {
        if !self.is_compatible() {
            return Err(format!(
                "checkpoint at {path} uses architecture version {} but this build supports \
                 {}..={} (module layout changed); retrain from scratch or re-import",
                self.arch_version, MIN_ARCH_VERSION, ARCH_VERSION
            ));
        }
        Ok(())
    }

    /// Whether this build can load the checkpoint layout.
    pub fn is_compatible(&self) -> bool {
        self.arch_version >= MIN_ARCH_VERSION && self.arch_version <= ARCH_VERSION
    }

    /// FFN hidden size: the configured value or the engine's aligned default.
    pub fn ffn_hidden(&self) -> usize {
        self.ffn_hidden
            .filter(|h| *h > 0)
            .unwrap_or_else(|| ((8 * self.n_embd) / 3).div_ceil(256) * 256)
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
    /// K/V projections are counted once per sharing group.
    pub fn param_count(&self) -> usize {
        let n_embd = self.n_embd;
        let kv_dim = self.get_n_kv_head() * (n_embd / self.n_head);
        let hidden = self.ffn_hidden();

        let embed = self.vocab_size * n_embd;
        let q = n_embd * n_embd + n_embd;
        let out = n_embd * n_embd;
        let norms = 2 * n_embd;
        let mlp = 3 * n_embd * hidden;
        let per_layer = q + out + norms + mlp;
        let per_group = if self.use_mla() {
            let d = self.mla_dim();
            // kv_down (n_embd -> d) + k_up/v_up (d -> kv_dim), no biases.
            n_embd * d + 2 * kv_dim * d
        } else {
            let kv = n_embd * kv_dim + kv_dim;
            2 * kv
        };

        embed + self.n_layer * per_layer + self.n_kv_groups() * per_group + n_embd
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

    #[test]
    fn test_kv_share_groups_and_param_count() {
        let cfg = ModelConfig::new(8000, 256, 4, Some(2), 4, 256, 0.1, 0.0);
        assert_eq!(cfg.n_kv_groups(), 4);

        let mut grouped = cfg.clone();
        grouped.kv_share_group = 2;
        assert_eq!(grouped.n_kv_groups(), 2);

        // Grouping drops one K/V pair per removed group.
        let kv_dim = 2 * 64;
        let pair = 2 * (256 * kv_dim + kv_dim);
        assert_eq!(cfg.param_count() - grouped.param_count(), 2 * pair);
    }

    #[test]
    fn test_arch_version_gate() {
        let cfg = ModelConfig::new(100, 64, 4, Some(2), 2, 32, 0.0, 0.0);
        assert!(cfg.check_arch_version("fresh").is_ok());

        // Configs without the field default to version 1 and are rejected.
        let old = r#"{"vocab_size":100,"n_embd":64,"n_head":4,"n_layer":2,"n_ctx":32}"#;
        let cfg: ModelConfig = serde_json::from_str(old).unwrap();
        assert_eq!(cfg.arch_version, 1);
        assert_eq!(cfg.kv_share_group, 1);
        let err = cfg.check_arch_version("old.bin").unwrap_err();
        assert!(err.contains("architecture version"), "{err}");
    }

    #[test]
    fn test_eviction_geometry_and_validation() {
        let mut cfg = ModelConfig::new(100, 64, 4, Some(2), 2, 64, 0.0, 0.0);
        assert_eq!(cfg.eviction_geometry().unwrap(), None);
        assert_eq!(cfg.kv_recent, 64);

        cfg.sink_tokens = 4;
        cfg.kv_eviction = Some("h2o".into());
        // The budget is required.
        let err = cfg.eviction_geometry().unwrap_err();
        assert!(err.contains("requires kv_budget"), "{err}");
        cfg.kv_budget = Some(100);
        let geo = cfg.eviction_geometry().unwrap().unwrap();
        assert_eq!(
            geo,
            EvictionGeometry { budget: 100, recent: 64, sinks: 4, hitters: 32 }
        );

        // The budget must leave room for at least one heavy hitter.
        cfg.kv_budget = Some(68);
        let err = cfg.eviction_geometry().unwrap_err();
        assert!(err.contains("must exceed"), "{err}");

        // Sliding window conflicts with eviction.
        cfg.kv_budget = Some(100);
        cfg.sliding_window = Some(32);
        let err = cfg.eviction_geometry().unwrap_err();
        assert!(err.contains("mutually exclusive"), "{err}");

        // Unknown policy.
        cfg.sliding_window = None;
        cfg.kv_eviction = Some("magic".into());
        let err = cfg.eviction_geometry().unwrap_err();
        assert!(err.contains("Unknown KV eviction"), "{err}");

        // "none"/"off" disable eviction even with a budget.
        cfg.kv_eviction = Some("none".into());
        assert_eq!(cfg.eviction_geometry().unwrap(), None);
    }

    #[test]
    fn test_eviction_fields_default_when_absent() {        let json = r#"{"vocab_size":100,"n_embd":64,"n_head":4,"n_layer":2,"n_ctx":32}"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.kv_eviction, None);
        assert_eq!(cfg.kv_budget, None);
        assert_eq!(cfg.kv_recent, 64);
        assert_eq!(cfg.eviction_geometry().unwrap(), None);
    }

    #[test]
    fn test_set_kv_eviction_override() {
        let mut cfg = ModelConfig::new(100, 64, 4, Some(2), 2, 64, 0.0, 0.0);
        cfg.set_kv_eviction("h2o", 128, 16);
        assert_eq!(cfg.kv_eviction.as_deref(), Some("h2o"));
        assert_eq!(cfg.kv_budget, Some(128));
        assert_eq!(cfg.kv_recent, 16);
        // Empty policy / zero knobs keep the configured values.
        cfg.set_kv_eviction("", 0, 0);
        assert_eq!(cfg.kv_budget, Some(128));
        assert_eq!(cfg.kv_recent, 16);
        // "none" turns it off.
        cfg.set_kv_eviction("none", 0, 0);
        assert_eq!(cfg.kv_eviction, None);
    }

    #[test]
    fn test_third_party_config_fields() {
        let json = r#"{"vocab_size":100,"n_embd":64,"n_head":4,"n_layer":2,"n_ctx":32}"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.ffn_hidden(), ((8 * 64) / 3usize).div_ceil(256) * 256);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert_eq!(cfg.rope_base, 10000.0);
        assert_eq!(cfg.import_source, None);

        let mut custom = cfg.clone();
        custom.arch_version = ARCH_VERSION;
        custom.ffn_hidden = Some(192);
        custom.rms_norm_eps = 1e-5;
        custom.rope_base = 1e6;
        custom.import_source = Some("hf".into());
        assert_eq!(custom.ffn_hidden(), 192);
        let round: ModelConfig =
            serde_json::from_slice(&serde_json::to_vec(&custom).unwrap()).unwrap();
        assert_eq!(round.ffn_hidden, Some(192));
        assert_eq!(round.rms_norm_eps, 1e-5);
        assert_eq!(round.rope_base, 1e6);
        assert_eq!(round.import_source.as_deref(), Some("hf"));
        assert_eq!(round.arch_version, ARCH_VERSION);

        // v4 checkpoints keep loading (the new fields default to the old
        // behavior).
        let mut v4 = cfg.clone();
        v4.arch_version = 4;
        assert!(v4.is_compatible());
        assert!(v4.check_arch_version("v4").is_ok());
    }
}
