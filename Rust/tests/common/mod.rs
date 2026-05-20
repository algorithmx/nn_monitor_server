use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;

pub fn build_test_app() -> Router {
    let store = Arc::new(nn_monitor_server::store::MetricsStore::new(10, 1000));
    let ws_manager = Arc::new(nn_monitor_server::ws::WsManager::new());
    let (ingest_tx, ingest_rx, ingest_stats) = nn_monitor_server::ingest::channel(4096);
    let _ingest_worker = nn_monitor_server::ingest::spawn_worker(
        ingest_rx,
        Arc::clone(&store),
        Arc::clone(&ws_manager),
        Arc::clone(&ingest_stats),
    );
    let state = nn_monitor_server::routes::AppState {
        store,
        ws_manager,
        ingest_tx,
        ingest_stats,
        config: nn_monitor_server::config::ServerConfig {
            max_runs: 10,
            max_steps_per_run: 1000,
            max_request_size: 2_000_000,
            ingest_queue_size: 4096,
            host: "0.0.0.0".to_string(),
            port: 8000,
            log_level: "warning".to_string(),
            cors_origins: vec!["*".to_string()],
            data_dir: "./data".to_string(),
            flush_timeout_secs: 300,
        },
    };
    Router::new()
        .route(
            "/api/v1/metrics/layerwise",
            post(nn_monitor_server::routes::metrics::post_metrics),
        )
        .route(
            "/api/v1/runs",
            get(nn_monitor_server::routes::runs::get_runs),
        )
        .route(
            "/api/v1/runs/{run_id}",
            get(nn_monitor_server::routes::runs::get_run),
        )
        .route(
            "/api/v1/runs/{run_id}/latest",
            get(nn_monitor_server::routes::runs::get_latest_step),
        )
        .route(
            "/health",
            get(nn_monitor_server::routes::health::get_health),
        )
        .with_state(state)
}

pub fn valid_payload() -> serde_json::Value {
    serde_json::json!({
        "metadata": {
            "run_id": "test_run",
            "timestamp": 1.0,
            "global_step": 100,
            "batch_size": 32
        },
        "layer_statistics": [{
            "layer_id": "layer1",
            "layer_type": "Linear",
            "depth_index": 0,
            "intermediate_features": {
                "activation_std": 0.5,
                "activation_mean": 0.1,
                "activation_shape": [32, 64]
            },
            "gradient_flow": {
                "gradient_l2_norm": 0.1,
                "gradient_std": 0.05,
                "gradient_max_abs": 0.2
            },
            "parameter_statistics": {
                "weight": {
                    "std": 0.1,
                    "mean": 0.0,
                    "spectral_norm": 1.0,
                    "frobenius_norm": 0.5
                }
            }
        }],
        "cross_layer_analysis": {
            "feature_std_gradient": 0.01,
            "gradient_norm_ratio": {}
        }
    })
}

pub fn valid_payload_with(run_id: &str, global_step: u64) -> serde_json::Value {
    let mut p = valid_payload();
    p["metadata"]["run_id"] = serde_json::Value::String(run_id.to_string());
    p["metadata"]["global_step"] = serde_json::Value::from(global_step);
    p
}
