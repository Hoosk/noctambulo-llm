//! HTTP server to control training from the browser, usable locally or on a
//! remote machine with a GPU (WGPU is local to the host running this binary).

pub mod api;
pub mod runner;
pub mod state;

use std::sync::Arc;

use axum::Router;
use tower_http::cors::{Any, CorsLayer};

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
    let app = Router::new()
        .merge(api::router(router_state))
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
