//! HTTP server to control training from the browser, usable locally or on a
//! remote machine with a GPU (WGPU is local to the host running this binary).

pub mod api;
pub mod runner;
pub mod state;

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::http::{HeaderValue, header};
use runner::JobRegistry;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::set_header::SetResponseHeaderLayer;

pub use state::ServerState;

/// Spawn the Axum server and block forever.
pub async fn run(
    bind: std::net::SocketAddr,
    token: Option<String>,
    datasets_dir: std::path::PathBuf,
    models_dir: std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let state = Arc::new(ServerState::new(token, datasets_dir, models_dir));

    let (api_router, registry) = api::router(state.clone());
    let frontend_dir = frontend_dir();
    let app = Router::new()
        .merge(api_router)
        .fallback_service(
            ServeDir::new(&frontend_dir)
                .not_found_service(ServeFile::new(frontend_dir.join("index.html"))),
        )
        // Revalidate static assets/API responses on every request: without this
        // browsers heuristically cache the JS modules and keep running stale
        // frontend code after an update.
        .layer(SetResponseHeaderLayer::if_not_present(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        ));

    let listener = tokio::net::TcpListener::bind(bind).await?;
    println!("🐈‍⬛ Kuro-Neko serve listening on http://{bind}");
    if let Some(t) = &*state.token.read().unwrap() {
        println!("   API token: {}", t);
        println!("   Envía el header `X-API-Token: {t}` en cada petición.");
    } else {
        println!("   ⚠ Sin token configurado (solo para red local de confianza).");
    }
    println!("   Datasets: {:?}", state.datasets_dir);
    println!("   Modelos:  {:?}", state.models_dir);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state.clone(), registry))
        .await?;
    Ok(())
}

/// Wait for Ctrl+C, request a graceful training stop and give the worker time
/// to persist its interrupted checkpoint before the server exits.
async fn shutdown_signal(state: Arc<ServerState>, registry: Arc<JobRegistry>) {
    if tokio::signal::ctrl_c().await.is_ok() {
        println!("\n   ⏹ Ctrl+C: stopping training (saving checkpoint, up to 30s)...");
        registry.request_stop();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while state.job.read().unwrap().running && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
    }
}

/// Locate the frontend directory (checked-out repo or compiled asset).
fn frontend_dir() -> PathBuf {
    let candidate = PathBuf::from("frontend");
    if candidate.join("index.html").exists() {
        candidate
    } else {
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/frontend"))
    }
}
