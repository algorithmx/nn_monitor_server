mod common;

use axum::http::StatusCode;
use axum_test::TestServer;
use common::{build_test_app, valid_payload, valid_payload_with};

// ==================== Batch Size Boundary ====================

#[tokio::test]
async fn test_batch_size_zero_rejected() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["metadata"]["batch_size"] = serde_json::json!(0);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().expect("detail should be array");
    assert!(
        detail[0]["msg"].as_str().unwrap().contains("batch_size"),
        "error should mention batch_size"
    );
}

#[tokio::test]
async fn test_batch_size_one_accepted() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["metadata"]["batch_size"] = serde_json::json!(1);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);
}

// ==================== Timestamp Boundary ====================

#[tokio::test]
async fn test_timestamp_zero_rejected() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["metadata"]["timestamp"] = serde_json::json!(0.0);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().expect("detail should be array");
    assert!(
        detail[0]["msg"].as_str().unwrap().contains("timestamp"),
        "error should mention timestamp"
    );
}

#[tokio::test]
async fn test_timestamp_negative_rejected() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["metadata"]["timestamp"] = serde_json::json!(-1.0);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().expect("detail should be array");
    assert!(
        detail[0]["msg"].as_str().unwrap().contains("timestamp"),
        "error should mention timestamp"
    );
}

// ==================== Global Step Boundary ====================

#[tokio::test]
async fn test_global_step_zero_accepted() {
    let server = TestServer::new(build_test_app()).unwrap();
    let payload = valid_payload_with("run_step_zero", 0);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn test_very_large_global_step() {
    let server = TestServer::new(build_test_app()).unwrap();
    let payload = valid_payload_with("run_large_step", u64::MAX);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);
    let body: serde_json::Value = response.json();
    assert_eq!(body["run_id"], "run_large_step");
}

// ==================== Layer ID Sanitization ====================

#[tokio::test]
async fn test_multiple_layers_all_sanitized() {
    let server = TestServer::new(build_test_app()).unwrap();

    let mut payload = valid_payload();
    let base_layer = payload["layer_statistics"][0].clone();

    payload["layer_statistics"][0]["layer_id"] = serde_json::json!("encoder.block.1.linear");
    payload["layer_statistics"][0]["depth_index"] = serde_json::json!(0);

    let mut layer1 = base_layer.clone();
    layer1["layer_id"] = serde_json::json!("decoder.attention.head.2");
    layer1["depth_index"] = serde_json::json!(1);

    let mut layer2 = base_layer.clone();
    layer2["layer_id"] = serde_json::json!("output.proj.final");
    layer2["depth_index"] = serde_json::json!(2);

    payload["layer_statistics"] =
        serde_json::json!([payload["layer_statistics"][0], layer1, layer2]);

    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);

    let response = server.get("/api/v1/runs/test_run/latest").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    let layers = body["layers"].as_array().expect("layers should be array");

    assert_eq!(layers.len(), 3);
    assert_eq!(layers[0]["layer_id"], "encoder/block/1/linear");
    assert_eq!(layers[1]["layer_id"], "decoder/attention/head/2");
    assert_eq!(layers[2]["layer_id"], "output/proj/final");
}

#[tokio::test]
async fn test_layer_groups_with_dots() {
    let server = TestServer::new(build_test_app()).unwrap();

    let mut payload = valid_payload();
    payload["metadata"]["layer_groups"] = serde_json::json!({
        "encoder": ["enc.block.1", "enc.block.2"],
        "decoder": ["dec.attn.head"]
    });

    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);

    let response = server.get("/api/v1/runs/test_run/latest").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();

    let groups = body["layer_groups"]
        .as_object()
        .expect("layer_groups should be object");

    let enc = groups["encoder"]
        .as_array()
        .expect("encoder should be array");
    assert_eq!(enc[0], "enc/block/1");
    assert_eq!(enc[1], "enc/block/2");

    let dec = groups["decoder"]
        .as_array()
        .expect("decoder should be array");
    assert_eq!(dec[0], "dec/attn/head");
}

#[tokio::test]
async fn test_cross_layer_analysis_preserved() {
    let server = TestServer::new(build_test_app()).unwrap();

    let mut payload = valid_payload();
    payload["cross_layer_analysis"] = serde_json::json!({
        "feature_std_gradient": -0.123,
        "gradient_norm_ratio": {
            "layer2_to_prev": 0.586,
            "layer3_to_prev": 0.528
        }
    });

    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);

    let response = server.get("/api/v1/runs/test_run/latest").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();

    let cla = &body["cross_layer"];
    assert_eq!(cla["feature_std_gradient"], -0.123);
    let ratio = cla["gradient_norm_ratio"]
        .as_object()
        .expect("gradient_norm_ratio should be object");
    assert_eq!(ratio["layer2_to_prev"], 0.586);
    assert_eq!(ratio["layer3_to_prev"], 0.528);
}

// ==================== Run Info Edge Cases ====================

#[tokio::test]
async fn test_run_info_latest_step_null_for_empty() {
    let run_info = nn_monitor_server::models::RunInfo {
        created_at: "2024-01-01T00:00:00.000000".to_string(),
        last_update: "2024-01-01T00:00:00.000000".to_string(),
        step_count: 0,
        latest_step: None,
    };
    let json = serde_json::to_value(&run_info).unwrap();
    assert!(
        json["latest_step"].is_null(),
        "latest_step should be null when no steps exist, got: {:?}",
        json["latest_step"]
    );
    assert_eq!(json["step_count"], 0);
}

#[tokio::test]
async fn test_concurrent_posts_same_run() {
    let server = TestServer::new(build_test_app()).unwrap();

    for step in 1..=5u64 {
        server
            .post("/api/v1/metrics/layerwise")
            .json(&valid_payload_with("concurrent_run", step))
            .await;
    }

    let response = server.get("/api/v1/runs").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert_eq!(body["concurrent_run"]["step_count"], 5);
    assert_eq!(body["concurrent_run"]["latest_step"], 5);
}

// ==================== Response Format Contracts ====================

#[tokio::test]
async fn test_runs_endpoint_returns_object_not_array() {
    let server = TestServer::new(build_test_app()).unwrap();

    let response = server.get("/api/v1/runs").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert!(
        body.is_object(),
        "GET /runs should return a JSON object, got: {:?}",
        body
    );

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("fmt_run", 1))
        .await;
    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    assert!(
        body.is_object(),
        "GET /runs should return a JSON object after adding data"
    );
}

#[tokio::test]
async fn test_run_endpoint_steps_is_array() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("arr_run", 42))
        .await;

    let response = server.get("/api/v1/runs/arr_run").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert!(
        body["steps"].is_array(),
        "steps field should be a JSON array, got: {:?}",
        body["steps"]
    );
    let steps = body["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["step"], 42);
}

#[tokio::test]
async fn test_health_response_format() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/health").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();

    assert!(
        body.get("status").is_some(),
        "health response should have 'status' field"
    );
    assert_eq!(body["status"], "healthy");
    assert!(
        body.get("active_connections").is_some(),
        "health response should have 'active_connections' field"
    );
    assert!(
        body["active_connections"].is_number(),
        "active_connections should be a number"
    );
}

#[tokio::test]
async fn test_metrics_accepted_response_format() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("fmt_test", 10))
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);
    let body: serde_json::Value = response.json();

    assert!(
        body.get("status").is_some(),
        "accepted response should have 'status' field"
    );
    assert_eq!(body["status"], "accepted");
    assert!(
        body.get("run_id").is_some(),
        "accepted response should have 'run_id' field"
    );
    assert_eq!(body["run_id"], "fmt_test");
}
