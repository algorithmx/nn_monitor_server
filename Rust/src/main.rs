mod config;
mod ingest;
mod models;
mod persist;
mod routes;
mod store;
mod ws;

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::http::HeaderValue;
use axum::response::IntoResponse;
use tower_http::cors::CorsLayer;
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

async fn shutdown_signal(jsonl_store: Arc<persist::JsonlStore>) {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    tracing::info!("\nShutting down gracefully...");
    tracing::info!("Flushing buffered data...");
    if let Err(e) = jsonl_store.flush_all().await {
        tracing::error!("Failed to flush all buffers: {}", e);
    }
    tracing::info!("Flush complete.");
}

#[tokio::main]
async fn main() {
    let config = config::ServerConfig::load().expect("Failed to load configuration");

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level)),
        )
        .init();

    let jsonl_store = Arc::new(
        persist::JsonlStore::new(
            std::path::PathBuf::from(&config.data_dir),
            std::time::Duration::from_secs(config.flush_timeout_secs),
            config.max_steps_per_run,
        )
        .await,
    );
    let _flush_task = jsonl_store.start_flush_task();
    let store = Arc::new(
        store::MetricsStore::new(config.max_runs, config.max_steps_per_run)
            .with_persist(Arc::clone(&jsonl_store)),
    );
    let ws_manager = Arc::new(ws::WsManager::new());
    let (ingest_tx, ingest_rx, ingest_stats) = ingest::channel(config.ingest_queue_size);
    let _ingest_worker = ingest::spawn_worker(
        ingest_rx,
        Arc::clone(&store),
        Arc::clone(&ws_manager),
        Arc::clone(&ingest_stats),
    );
    let state = AppState {
        store,
        ws_manager,
        ingest_tx,
        ingest_stats,
    };

    let cors = if config.cors_origins.iter().any(|o| o == "*") {
        CorsLayer::permissive()
    } else {
        let origins: Vec<HeaderValue> = config
            .cors_origins
            .iter()
            .map(|o| o.parse::<HeaderValue>().unwrap())
            .collect();
        CorsLayer::new().allow_origin(origins)
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
        .with_state(state)
        .layer(DefaultBodyLimit::max(config.max_request_size))
        .layer(cors);

    let host_display = if config.host == "0.0.0.0" {
        "localhost"
    } else {
        &config.host
    };

    tracing::info!("{}", "=".repeat(50));
    tracing::info!("NN Training Monitor Server (Rust)");
    tracing::info!("{}", "=".repeat(50));
    tracing::info!("Server started successfully!");
    tracing::info!("Host: {}", config.host);
    tracing::info!("Port: {}", config.port);
    tracing::info!("Access URL: http://{}:{}", host_display, config.port);
    tracing::info!(
        "WebSocket endpoint: ws://{}:{}/ws",
        host_display, config.port
    );
    tracing::info!("Max concurrent runs: {}", config.max_runs);
    tracing::info!("Max steps per run: {}", config.max_steps_per_run);
    tracing::info!("Data directory: {}", config.data_dir);
    tracing::info!("Flush timeout: {}s", config.flush_timeout_secs);
    tracing::info!("Ingest queue size: {}", config.ingest_queue_size);
    tracing::info!("{}", "=".repeat(50));

    let listener = tokio::net::TcpListener::bind(format!("{}:{}", config.host, config.port))
        .await
        .expect("Failed to bind address");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(jsonl_store))
        .await
        .expect("Server error");
}
