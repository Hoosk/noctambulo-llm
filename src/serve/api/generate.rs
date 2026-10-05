use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}},
};
use serde::Deserialize;

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;
use super::paths::resolve_model_dir;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GenerateRequest {
    model_dir: String,
    prompt: String,
    #[serde(default = "default_max_new_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temp")]
    temperature: f32,
    #[serde(default = "default_topk")]
    top_k: usize,
    #[serde(default = "default_topp")]
    top_p: f32,
    #[serde(default = "default_minp")]
    min_p: f32,
    #[serde(default = "default_rep")]
    repetition_penalty: f32,
    /// RoPE context-extension scaling: none, linear, ntk or yarn.
    #[serde(default)]
    rope_scaling: Option<String>,
    #[serde(default = "default_rope_factor")]
    rope_factor: f32,
    /// Compute precision: f32 (default) or f16.
    #[serde(default)]
    dtype: Option<String>,
    /// KV cache precision: f32 (default), f16, int8 or q4.
    #[serde(default)]
    kv_cache_dtype: Option<String>,
}

fn default_max_new_tokens() -> usize {
    60
}

fn default_temp() -> f32 {
    0.8
}

fn default_topk() -> usize {
    40
}

fn default_topp() -> f32 {
    0.9
}

fn default_minp() -> f32 {
    0.05
}

fn default_rep() -> f32 {
    1.2
}

fn default_rope_factor() -> f32 {
    1.0
}

/// Sampling knobs shared by the streaming and non-streaming endpoints.
struct Sampling {
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    rope_kind: crate::config::RopeScalingKind,
    rope_factor: f32,
    dtype: crate::config::DType,
    cache_dtype: crate::model::KvCacheDtype,
}

impl Sampling {
    fn from_request(req: GenerateRequest) -> Result<Self, ApiError> {
        let rope_kind = match req.rope_scaling.as_deref() {
            Some(name) => crate::config::RopeScalingKind::parse(name).map_err(ApiError::bad_request)?,
            None => crate::config::RopeScalingKind::None,
        };
        let rope_factor = if req.rope_factor.is_finite() && req.rope_factor > 0.0 {
            req.rope_factor
        } else {
            1.0
        };
        let dtype = match req.dtype.as_deref() {
            Some(name) => crate::config::DType::parse(name).map_err(ApiError::bad_request)?,
            None => crate::config::DType::F32,
        };
        let cache_dtype = match req.kv_cache_dtype.as_deref() {
            Some(name) => {
                crate::model::KvCacheDtype::parse(name).map_err(ApiError::bad_request)?
            }
            None => crate::model::KvCacheDtype::F32,
        };
        Ok(Self {
            prompt: req.prompt,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_k: req.top_k,
            top_p: req.top_p,
            min_p: req.min_p,
            repetition_penalty: req.repetition_penalty,
            rope_kind,
            rope_factor,
            dtype,
            cache_dtype,
        })
    }
}

pub(super) async fn generate_text(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;

    let model_dir = resolve_model_dir(&state.server.models_dir, &req.model_dir)?;
    let s = Sampling::from_request(req)?;

    let result = tokio::task::spawn_blocking(move || run_generate(model_dir, s, false, None)).await;

    match result {
        Ok(Ok(text)) => Ok(Json(serde_json::json!({ "text": text }))),
        Ok(Err(e)) => Err(ApiError::internal(e)),
        Err(e) => Err(ApiError::internal(format!("generate task failed: {e}"))),
    }
}

/// Token-by-token generation over SSE. Dropping the client stream drops the
/// receiver, which makes `blocking_send` fail and stops the generation loop.
pub(super) async fn generate_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, ApiError> {
    check_token(&state.server, &headers, None)?;

    let model_dir = resolve_model_dir(&state.server.models_dir, &req.model_dir)?;
    let s = Sampling::from_request(req)?;

    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
    tokio::task::spawn_blocking(move || {
        let streamed = run_generate(model_dir, s, true, Some(tx));
        if let Err(e) = streamed {
            eprintln!("generate stream error: {e}");
        }
    });

    let stream = async_stream::stream! {
        while let Some(chunk) = rx.recv().await {
            let data = serde_json::json!({ "token": chunk }).to_string();
            yield Ok::<Event, axum::Error>(Event::default().event("token").data(data));
        }
        yield Ok(Event::default().event("done").data("{\"done\":true}"));
    };

    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// Load a saved model and run generation on it. With `stream` enabled the
/// continuation is pushed chunk by chunk through `tx`; the returned string is
/// then empty (the caller only uses it to surface errors).
fn run_generate(
    model_dir: std::path::PathBuf,
    s: Sampling,
    stream: bool,
    tx: Option<tokio::sync::mpsc::Sender<String>>,
) -> Result<String, String> {
    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model_weights");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("config.json not found in {}: {e}", model_dir.display()))?;
    let mut config: crate::config::ModelConfig =
        serde_json::from_str(&config_str).map_err(|e| format!("bad config: {e}"))?;
    config
        .check_arch_version(&config_path.display().to_string())
        .map_err(|e| format!("incompatible checkpoint: {e}"))?;
    // Request-level RoPE override for context extension; `none` + factor 1
    // keeps whatever the checkpoint config declares.
    config.set_rope_scaling(s.rope_kind, s.rope_factor);

    let tokenizer = crate::data::resolve_tokenizer(Some(model_dir.as_path()))
        .map_err(|e| format!("tokenizer not found: {e}"))?;

    match (s.dtype, s.cache_dtype) {
        // An f16 cache needs a backend that really stores f16 (the fused
        // backend's f16 casts are no-ops).
        (crate::config::DType::F32, crate::model::KvCacheDtype::F16) => {
            run_generate_backend::<crate::backend::WgpuHalf<f32>>(
                config, weights_path, tokenizer, s, stream, tx,
            )
        }
        (crate::config::DType::F32, _) => run_generate_backend::<burn::backend::Wgpu>(
            config, weights_path, tokenizer, s, stream, tx,
        ),
        (crate::config::DType::F16, _) => {
            run_generate_backend::<crate::backend::WgpuHalf<burn::tensor::f16>>(
                config, weights_path, tokenizer, s, stream, tx,
            )
        }
    }
}

fn run_generate_backend<B: burn::tensor::backend::Backend>(
    config: crate::config::ModelConfig,
    weights_path: std::path::PathBuf,
    tokenizer: crate::data::LlmTokenizer,
    s: Sampling,
    stream: bool,
    tx: Option<tokio::sync::mpsc::Sender<String>>,
) -> Result<String, String> {
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

    let device = B::Device::default();
    let model: crate::model::Llm<B> = config.init_model(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(weights_path, &device)
        .map_err(|e| format!("failed to load weights: {e:?}"))?;
    let model = model.load_record(record).align_compute_dtype();

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);

    if stream {
        let tx = tx.ok_or("streaming requested without a channel")?;
        crate::generate::generate_streaming(
            &model,
            &tokenizer,
            &s.prompt,
            s.max_tokens,
            s.temperature,
            s.top_k,
            s.top_p,
            s.min_p,
            s.repetition_penalty,
            seed,
            s.cache_dtype,
            &device,
            |chunk| tx.blocking_send(chunk.to_string()).is_ok(),
        )
        .map_err(|e| format!("generate error: {e}"))?;
        return Ok(String::new());
    }

    crate::generate::generate(
        &model,
        &tokenizer,
        &s.prompt,
        s.max_tokens,
        s.temperature,
        s.top_k,
        s.top_p,
        s.min_p,
        s.repetition_penalty,
        seed,
        s.cache_dtype,
        &device,
    )
    .map_err(|e| format!("generate error: {e}"))
}
