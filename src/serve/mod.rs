//! HTTP server to control training from the browser, usable locally or on a
//! remote machine with a GPU (WGPU is local to the host running this binary).

pub mod api;
pub mod runner;
pub mod state;

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};

pub use state::ServerState;

/// Spawn the Axum server and block forever.
pub async fn run(
    bind: std::net::SocketAddr,
    token: Option<String>,
    datasets_dir: std::path::PathBuf,
    models_dir: std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let state = Arc::new(ServerState::new(token, datasets_dir, models_dir));

    let cors = CorsLayer::new().allow_origin(Any);

    let router_state = state.clone();
    let frontend_dir = frontend_dir();
    let app = Router::new()
        .merge(api::router(router_state))
        .fallback_service(
            ServeDir::new(&frontend_dir)
                .not_found_service(ServeFile::new(frontend_dir.join("index.html"))),
        )
        .layer(cors);

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
    axum::serve(listener, app).await?;
    Ok(())
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
