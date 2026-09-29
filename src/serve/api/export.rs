use axum::{Json, extract::State, http::HeaderMap};
use serde::Deserialize;

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;
use super::paths::{is_safe_export_name, resolve_model_dir};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ExportRequest {
    model_dir: String,
    #[serde(default = "default_model_name")]
    model_name: String,
}

fn default_model_name() -> String {
    "kuro-neko".into()
}

pub(super) async fn export_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ExportRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_token(&state.server, &headers, None)?;

    let model_dir = resolve_model_dir(&state.server.models_dir, &req.model_dir)?;
    if !is_safe_export_name(&req.model_name) {
        return Err(ApiError::bad_request("invalid model_name"));
    }

    let name = req.model_name;
    let result = tokio::task::spawn_blocking(move || run_export(model_dir, &name)).await;

    match result {
        Ok(Ok(path)) => Ok(Json(serde_json::json!({ "gguf": path }))),
        Ok(Err(e)) => Err(ApiError::internal(e)),
        Err(e) => Err(ApiError::internal(format!("export task failed: {e}"))),
    }
}

fn run_export(model_dir: std::path::PathBuf, model_name: &str) -> Result<String, String> {
    use burn::backend::Wgpu;
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model_weights");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("config.json not found in {}: {e}", model_dir.display()))?;
    let config: crate::config::ModelConfig =
        serde_json::from_str(&config_str).map_err(|e| format!("bad config: {e}"))?;

    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    let model: crate::model::Llm<Wgpu> = config.init_model(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(weights_path.clone(), &device)
        .map_err(|e| format!("failed to load weights: {e:?}"))?;
    let model = model.load_record(record);

    crate::export::export_to_gguf(&model, &config, "custom_tokenizer.json", model_name, "exported")
        .map_err(|e| format!("export error: {e}"))
}
