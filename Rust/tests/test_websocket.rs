mod common;

use std::sync::Arc;
use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

// ==================== Test Infrastructure ====================

fn build_ws_test_app() -> Router {
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
        .route("/ws", get(nn_monitor_server::routes::ws_route::ws_handler))
        .with_state(state)
}

async fn start_test_server() -> (String, tokio::task::JoinHandle<()>) {
    let app = build_ws_test_app();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("ws://{}:{}", addr.ip(), addr.port()), handle)
}

fn http_base_url(ws_url: &str) -> String {
    ws_url.replace("ws://", "http://")
}

async fn connect_ws(ws_url: &str) -> WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>> {
    let url = format!("{}/ws", ws_url);
    let (stream, _response) = connect_async(&url).await.unwrap();
    stream
}

async fn read_ws_message(
    ws: &mut WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
) -> Option<Value> {
    timeout(Duration::from_secs(5), async {
        while let Some(msg) = ws.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    return serde_json::from_str(&text).ok();
                }
                Ok(Message::Close(_)) => return None,
                _ => continue,
            }
        }
        None
    })
    .await
    .ok()
    .flatten()
}

async fn post_metrics(base_url: &str, payload: &Value) -> reqwest::Response {
    let client = Client::new();
    client
        .post(format!("{}/api/v1/metrics/layerwise", base_url))
        .json(payload)
        .send()
        .await
        .unwrap()
}

// ==================== Test Cases ====================

#[tokio::test]
async fn test_ws_connect_and_initial_runs() {
    let (ws_url, _handle) = start_test_server().await;
    let mut ws = connect_ws(&ws_url).await;

    let msg = read_ws_message(&mut ws).await;
    assert!(msg.is_some(), "Should receive initial_runs message");

    let msg = msg.unwrap();
    assert_eq!(
        msg["type"], "initial_runs",
        "Message type should be initial_runs"
    );
    assert!(msg["data"].is_object(), "data should be an object");
}

#[tokio::test]
async fn test_ws_ping_pong() {
    let (ws_url, _handle) = start_test_server().await;
    let mut ws = connect_ws(&ws_url).await;

    // Read and discard initial_runs
    let _initial = read_ws_message(&mut ws).await;

    // Send ping
    ws.send(Message::Text(r#"{"action":"ping"}"#.into()))
        .await
        .unwrap();

    let msg = read_ws_message(&mut ws).await;
    assert!(msg.is_some(), "Should receive pong message");

    let msg = msg.unwrap();
    assert_eq!(msg["type"], "pong", "Should receive pong");
}

#[tokio::test]
async fn test_ws_subscribe_run_existing() {
    let (ws_url, _handle) = start_test_server().await;
    let base_url = http_base_url(&ws_url);

    // POST metrics to create a run first
    let payload = common::valid_payload_with("run_sub_1", 100);
    let resp = post_metrics(&base_url, &payload).await;
    assert_eq!(resp.status(), 202);

    // Small delay to ensure broadcast is processed
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Connect WS
    let mut ws = connect_ws(&ws_url).await;

    // Read initial_runs (should contain our run)
    let initial = read_ws_message(&mut ws).await.unwrap();
    assert_eq!(initial["type"], "initial_runs");
    assert!(initial["data"]["run_sub_1"].is_object());

    // Subscribe to the run
    ws.send(Message::Text(
        r#"{"action":"subscribe_run","run_id":"run_sub_1"}"#.into(),
    ))
    .await
    .unwrap();

    let msg = read_ws_message(&mut ws).await;
    assert!(msg.is_some(), "Should receive run_history message");

    let msg = msg.unwrap();
    assert_eq!(msg["type"], "run_history");
    assert_eq!(msg["run_id"], "run_sub_1");
    assert!(msg["data"].is_object());
    assert!(msg["data"]["steps"].is_array());
}

#[tokio::test]
async fn test_ws_subscribe_run_not_found() {
    let (ws_url, _handle) = start_test_server().await;
    let mut ws = connect_ws(&ws_url).await;

    // Read and discard initial_runs
    let _initial = read_ws_message(&mut ws).await;

    // Subscribe to nonexistent run
    ws.send(Message::Text(
        r#"{"action":"subscribe_run","run_id":"nonexistent_run"}"#.into(),
    ))
    .await
    .unwrap();

    let msg = read_ws_message(&mut ws).await;
    assert!(msg.is_some(), "Should receive error message");

    let msg = msg.unwrap();
    assert_eq!(msg["type"], "error");
    assert!(msg["message"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn test_ws_subscribe_run_lite_mode() {
    let (ws_url, _handle) = start_test_server().await;
    let base_url = http_base_url(&ws_url);

    // POST metrics to create a run with rich layer data
    let payload = common::valid_payload_with("run_lite_1", 100);
    let resp = post_metrics(&base_url, &payload).await;
    assert_eq!(resp.status(), 202);

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut ws = connect_ws(&ws_url).await;
    let _initial = read_ws_message(&mut ws).await;

    // Subscribe with lite mode
    ws.send(Message::Text(
        r#"{"action":"subscribe_run","run_id":"run_lite_1","lite":true}"#.into(),
    ))
    .await
    .unwrap();

    let msg = read_ws_message(&mut ws).await;
    assert!(msg.is_some(), "Should receive run_history message");

    let msg = msg.unwrap();
    assert_eq!(msg["type"], "run_history");
    assert_eq!(msg["run_id"], "run_lite_1");

    let steps = msg["data"]["steps"].as_array().unwrap();
    assert!(!steps.is_empty(), "Should have at least one step");

    let layers = steps[0]["layers"].as_array().unwrap();
    assert!(!layers.is_empty(), "Should have at least one layer");

    let layer = &layers[0];
    let layer_obj = layer.as_object().unwrap();

    // Lite mode should NOT contain these fields
    assert!(
        !layer_obj.contains_key("parameter_statistics"),
        "Lite mode should strip parameter_statistics"
    );

    // Check intermediate_features does not contain activation_mean
    if let Some(ifeatures) = layer_obj
        .get("intermediate_features")
        .and_then(|v| v.as_object())
    {
        assert!(
            !ifeatures.contains_key("activation_mean"),
            "Lite mode should strip activation_mean"
        );
        assert!(
            !ifeatures.contains_key("activation_shape"),
            "Lite mode should strip activation_shape"
        );
    }

    // Check gradient_flow does not contain gradient_std
    if let Some(gf) = layer_obj.get("gradient_flow").and_then(|v| v.as_object()) {
        assert!(
            !gf.contains_key("gradient_std"),
            "Lite mode should strip gradient_std"
        );
    }
}

#[tokio::test]
async fn test_ws_invalid_json() {
    let (ws_url, _handle) = start_test_server().await;
    let mut ws = connect_ws(&ws_url).await;

    // Read and discard initial_runs
    let _initial = read_ws_message(&mut ws).await;

    // Send invalid JSON
    ws.send(Message::Text("not valid json".into()))
        .await
        .unwrap();

    let msg = read_ws_message(&mut ws).await;
    assert!(msg.is_some(), "Should receive error message");

    let msg = msg.unwrap();
    assert_eq!(msg["type"], "error");
    assert_eq!(msg["message"], "Invalid JSON format");
}

#[tokio::test]
async fn test_ws_unknown_action_ignored() {
    let (ws_url, _handle) = start_test_server().await;
    let mut ws = connect_ws(&ws_url).await;

    // Read and discard initial_runs
    let _initial = read_ws_message(&mut ws).await;

    // Send unknown action
    ws.send(Message::Text(r#"{"action":"unknown"}"#.into()))
        .await
        .unwrap();

    // Connection should stay open for at least 1 second (no crash, no close)
    let result = timeout(Duration::from_secs(2), async { ws.next().await }).await;

    // Either no message arrives (timeout) or it's not an error/close
    match result {
        Ok(Some(Ok(Message::Close(_)))) => {
            panic!("Connection should stay open for unknown action");
        }
        Ok(Some(Ok(Message::Text(_)))) => {}
        Ok(None) => {
            panic!("Stream ended unexpectedly");
        }
        Ok(Some(Err(e))) => {
            panic!("WebSocket error: {}", e);
        }
        Err(_) => {}
        _ => {}
    }
}

#[tokio::test]
async fn test_ws_broadcast_new_metrics() {
    let (ws_url, _handle) = start_test_server().await;
    let base_url = http_base_url(&ws_url);

    // Connect WS first
    let mut ws = connect_ws(&ws_url).await;

    // Read and discard initial_runs
    let _initial = read_ws_message(&mut ws).await;

    // POST metrics via HTTP
    let payload = common::valid_payload_with("run_broadcast", 42);
    let resp = post_metrics(&base_url, &payload).await;
    assert_eq!(resp.status(), 202);

    // WS client should receive new_metrics broadcast
    let msg = read_ws_message(&mut ws).await;
    assert!(msg.is_some(), "Should receive new_metrics broadcast");

    let msg = msg.unwrap();
    assert_eq!(msg["type"], "new_metrics");
    assert_eq!(msg["run_id"], "run_broadcast");
    assert!(msg["data"].is_object());
    assert_eq!(msg["data"]["step"], 42);
}

#[tokio::test]
async fn test_ws_multiple_clients_broadcast() {
    let (ws_url, _handle) = start_test_server().await;
    let base_url = http_base_url(&ws_url);

    // Connect 2 WS clients
    let mut ws1 = connect_ws(&ws_url).await;
    let mut ws2 = connect_ws(&ws_url).await;

    // Read initial_runs from both
    let _init1 = read_ws_message(&mut ws1).await;
    let _init2 = read_ws_message(&mut ws2).await;

    // POST metrics
    let payload = common::valid_payload_with("run_multi", 200);
    let resp = post_metrics(&base_url, &payload).await;
    assert_eq!(resp.status(), 202);

    // Both clients should receive the broadcast
    let msg1 = read_ws_message(&mut ws1).await;
    let msg2 = read_ws_message(&mut ws2).await;

    assert!(msg1.is_some(), "Client 1 should receive new_metrics");
    assert!(msg2.is_some(), "Client 2 should receive new_metrics");

    let msg1 = msg1.unwrap();
    let msg2 = msg2.unwrap();

    assert_eq!(msg1["type"], "new_metrics");
    assert_eq!(msg1["run_id"], "run_multi");
    assert_eq!(msg2["type"], "new_metrics");
    assert_eq!(msg2["run_id"], "run_multi");
}

#[tokio::test]
async fn test_ws_disconnect_decrements_count() {
    let (ws_url, _handle) = start_test_server().await;
    let base_url = http_base_url(&ws_url);

    let client = Client::new();

    // Check initial health — 0 connections
    let health = client
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();
    let health_json: Value = health.json().await.unwrap();
    assert_eq!(
        health_json["active_connections"], 0,
        "Should start with 0 connections"
    );

    // Connect WS
    let mut ws = connect_ws(&ws_url).await;

    // Read initial_runs to ensure connection is fully established
    let _initial = read_ws_message(&mut ws).await;

    // Small delay for connection count to propagate
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Check health — should be 1 connection
    let health = client
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();
    let health_json: Value = health.json().await.unwrap();
    assert_eq!(
        health_json["active_connections"], 1,
        "Should have 1 connection after WS connect"
    );

    // Disconnect by closing
    ws.close(None).await.unwrap();

    // Small delay for disconnect to propagate
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Check health — should be 0 connections
    let health = client
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();
    let health_json: Value = health.json().await.unwrap();
    assert_eq!(
        health_json["active_connections"], 0,
        "Should have 0 connections after WS disconnect"
    );
}

#[tokio::test]
async fn test_ws_initial_runs_contains_existing_runs() {
    let (ws_url, _handle) = start_test_server().await;
    let base_url = http_base_url(&ws_url);

    // Create two runs via HTTP before connecting WS
    let p1 = common::valid_payload_with("run_a", 10);
    let p2 = common::valid_payload_with("run_b", 20);
    let r1 = post_metrics(&base_url, &p1).await;
    let r2 = post_metrics(&base_url, &p2).await;
    assert_eq!(r1.status(), 202);
    assert_eq!(r2.status(), 202);

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Connect WS and check initial_runs contains both runs
    let mut ws = connect_ws(&ws_url).await;
    let msg = read_ws_message(&mut ws).await.unwrap();
    assert_eq!(msg["type"], "initial_runs");
    assert!(msg["data"]["run_a"].is_object(), "run_a should be present");
    assert!(msg["data"]["run_b"].is_object(), "run_b should be present");
}

#[tokio::test]
async fn test_ws_full_mode_includes_all_fields() {
    let (ws_url, _handle) = start_test_server().await;
    let base_url = http_base_url(&ws_url);

    // POST metrics
    let payload = common::valid_payload_with("run_full_mode", 50);
    let resp = post_metrics(&base_url, &payload).await;
    assert_eq!(resp.status(), 202);

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut ws = connect_ws(&ws_url).await;
    let _initial = read_ws_message(&mut ws).await;

    // Subscribe with full mode (lite: false, which is default)
    ws.send(Message::Text(
        r#"{"action":"subscribe_run","run_id":"run_full_mode","lite":false}"#.into(),
    ))
    .await
    .unwrap();

    let msg = read_ws_message(&mut ws).await.unwrap();
    assert_eq!(msg["type"], "run_history");
    assert_eq!(msg["run_id"], "run_full_mode");

    let steps = msg["data"]["steps"].as_array().unwrap();
    let layers = steps[0]["layers"].as_array().unwrap();
    let layer = &layers[0];
    let layer_obj = layer.as_object().unwrap();

    // Full mode should contain all fields
    assert!(
        layer_obj.contains_key("parameter_statistics"),
        "Full mode should include parameter_statistics"
    );
    if let Some(ifeatures) = layer_obj
        .get("intermediate_features")
        .and_then(|v| v.as_object())
    {
        assert!(
            ifeatures.contains_key("activation_mean"),
            "Full mode should include activation_mean"
        );
    }
}
