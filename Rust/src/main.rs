mod config;
mod models;
mod routes;
mod store;
mod ws;

use std::sync::Arc;

use axum::response::IntoResponse;
use tower_http::services::ServeDir;
use tracing_subscriber::EnvFilter;

use routes::AppState;

async fn serve_index() -> impl IntoResponse {
    match tokio::fs::read_to_string("static/index.html").await {
        Ok(content) => axum::response::Html(content).into_response(),
        Err(_) => axum::Json(serde_json::json!({
            "message": "NN Training Monitor Server",
            "docs": "/docs"
        }))
        .into_response(),
    }
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    println!("\nShutting down gracefully...");
}

#[tokio::main]
async fn main() {
    let config = config::ServerConfig::load().expect("Failed to load configuration");

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level)),
        )
        .init();

    let store = Arc::new(store::MetricsStore::new(
        config.max_runs,
        config.max_steps_per_run,
    ));
    let ws_manager = Arc::new(ws::WsManager::new());
    let state = AppState {
        store,
        ws_manager,
        config: config.clone(),
    };

    let app = axum::Router::new()
        .route(
            "/api/v1/metrics/layerwise",
            axum::routing::post(routes::metrics::post_metrics),
        )
        .route("/api/v1/runs", axum::routing::get(routes::runs::get_runs))
        .route(
            "/api/v1/runs/{run_id}",
            axum::routing::get(routes::runs::get_run),
        )
        .route(
            "/api/v1/runs/{run_id}/latest",
            axum::routing::get(routes::runs::get_latest_step),
        )
        .route("/health", axum::routing::get(routes::health::get_health))
        .route("/ws", axum::routing::get(routes::ws_route::ws_handler))
        .nest_service("/static", ServeDir::new("static"))
        .fallback(serve_index)
        .with_state(state);

    let host_display = if config.host == "0.0.0.0" {
        "localhost"
    } else {
        &config.host
    };

    println!("{}", "=".repeat(50));
    println!("NN Training Monitor Server (Rust)");
    println!("{}", "=".repeat(50));
    println!("Server started successfully!");
    println!("Host: {}", config.host);
    println!("Port: {}", config.port);
    println!("Access URL: http://{}:{}", host_display, config.port);
    println!(
        "WebSocket endpoint: ws://{}:{}/ws",
        host_display, config.port
    );
    println!("Max concurrent runs: {}", config.max_runs);
    println!("Max steps per run: {}", config.max_steps_per_run);
    println!("{}", "=".repeat(50));

    let listener = tokio::net::TcpListener::bind(format!("{}:{}", config.host, config.port))
        .await
        .expect("Failed to bind address");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("Server error");
}
