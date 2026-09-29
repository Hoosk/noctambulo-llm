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
use tower::ServiceBuilder;
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
    let index_html = frontend_dir.join("index.html");
    if !index_html.is_file() {
        println!("   ⚠ Frontend not built: run `npm install && npm run build` inside web/.");
    }

    // Content-hashed Vite assets keep their name forever, so they are cached
    // hard; everything else (including API responses) revalidates on every
    // request so a rebuild or redeploy is picked up without manual cache
    // busting.
    let assets = ServiceBuilder::new()
        .layer(SetResponseHeaderLayer::overriding(
            header::CACHE_CONTROL,
            HeaderValue::from_static("public, max-age=31536000, immutable"),
        ))
        .service(ServeDir::new(frontend_dir.join("assets")));

    let spa = ServiceBuilder::new()
        .layer(SetResponseHeaderLayer::if_not_present(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        ))
        // `.fallback` keeps the fallback's 200 status; `not_found_service`
        // would force 404 for every SPA route.
        .service(ServeDir::new(&frontend_dir).fallback(ServeFile::new(&index_html)));

    let app = Router::new()
        .merge(api_router)
        .layer(SetResponseHeaderLayer::if_not_present(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        ))
        .nest_service("/assets", assets)
        .fallback_service(spa);

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
    println!("   Web UI:   {:?}", frontend_dir);
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

/// Locate the built frontend (repo checkout or a `web/dist` next to the CWD).
fn frontend_dir() -> PathBuf {
    let candidate = PathBuf::from("web/dist");
    if candidate.join("index.html").exists() {
        candidate
    } else {
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/web/dist"))
    }
}
