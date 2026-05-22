mod common;

use axum::http::StatusCode;
use axum_test::TestServer;
use common::{build_test_app, valid_payload_with};

#[tokio::test]
async fn test_empty_runs() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/api/v1/runs").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert!(body.as_object().unwrap().is_empty());
}

#[tokio::test]
async fn test_single_run() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run1", 100))
        .await;

    let response = server.get("/api/v1/runs").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert!(body.as_object().unwrap().contains_key("run1"));

    let run_info = &body["run1"];
    assert!(run_info["created_at"].is_string());
    assert!(run_info["last_update"].is_string());
    assert_eq!(run_info["step_count"], 1);
    assert_eq!(run_info["latest_step"], 100);
}

#[tokio::test]
async fn test_multiple_runs() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run1", 100))
        .await;
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run2", 200))
        .await;
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run3", 300))
        .await;

    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    assert_eq!(body.as_object().unwrap().len(), 3);
    assert!(body.as_object().unwrap().contains_key("run1"));
    assert!(body.as_object().unwrap().contains_key("run2"));
    assert!(body.as_object().unwrap().contains_key("run3"));
}

#[tokio::test]
async fn test_run_info_fields() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run1", 100))
        .await;

    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    let run_info = &body["run1"];

    assert!(run_info.get("created_at").is_some());
    assert!(run_info.get("last_update").is_some());
    assert!(run_info.get("step_count").is_some());
    assert!(run_info.get("latest_step").is_some());

    assert!(run_info["created_at"].is_string());
    assert!(run_info["last_update"].is_string());
    assert!(run_info["step_count"].is_number());
    assert!(run_info["latest_step"].is_number());
}

#[tokio::test]
async fn test_get_run_by_id() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("test_run", 100))
        .await;

    let response = server.get("/api/v1/runs/test_run").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();

    assert!(body["created_at"].is_string());
    assert!(body["last_update"].is_string());
    assert!(body["steps"].is_array());
}

#[tokio::test]
async fn test_get_run_not_found() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/api/v1/runs/nonexistent").await;
    assert_eq!(response.status_code(), StatusCode::NOT_FOUND);
    let body: serde_json::Value = response.json();
    assert!(body["detail"]["error"].is_string());
    assert!(body["detail"]["message"].is_string());
}

#[tokio::test]
async fn test_latest_step() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 100))
        .await;
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 200))
        .await;

    let response = server.get("/api/v1/runs/r1/latest").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert_eq!(body["step"], 200);
}

#[tokio::test]
async fn test_latest_step_not_found() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/api/v1/runs/nonexistent/latest").await;
    assert_eq!(response.status_code(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_step_count_increments() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 100))
        .await;
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 200))
        .await;
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 300))
        .await;

    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    assert_eq!(body["r1"]["step_count"], 3);
}

#[tokio::test]
async fn test_step_count_with_duplicate() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 100))
        .await;
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 100))
        .await;

    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    assert_eq!(body["r1"]["step_count"], 1);
}

#[tokio::test]
async fn test_runs_after_multiple_posts() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 300))
        .await;
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 100))
        .await;
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 200))
        .await;

    let response = server.get("/api/v1/runs/r1").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    let steps = body["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0]["step"], 100);
    assert_eq!(steps[1]["step"], 200);
    assert_eq!(steps[2]["step"], 300);
}

#[tokio::test]
async fn test_health_endpoint() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/health").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert_eq!(body["status"], "healthy");
    assert_eq!(body["active_connections"], 0);
}

#[tokio::test]
async fn test_health_active_connections_type() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/health").await;
    let body: serde_json::Value = response.json();
    assert!(
        body["active_connections"].is_number(),
        "active_connections should be a number"
    );
    assert_eq!(body["active_connections"].as_u64().unwrap(), 0);
}

#[tokio::test]
async fn test_run_data_has_steps() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 100))
        .await;

    let response = server.get("/api/v1/runs/r1").await;
    let body: serde_json::Value = response.json();
    let steps = body["steps"].as_array().expect("steps should be array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["step"], 100);
    assert!(steps[0]["timestamp"].is_number());
    assert!(steps[0]["batch_size"].is_number());
}

#[tokio::test]
async fn test_cors_headers_present_on_get_runs() {
    let server = TestServer::new(build_test_app()).unwrap();

    let response = server
        .get("/api/v1/runs")
        .add_header("Origin", "https://example.com")
        .await;

    assert_eq!(response.status_code(), StatusCode::OK);
    assert!(
        response
            .headers()
            .contains_key("access-control-allow-origin"),
        "CORS Access-Control-Allow-Origin header should be present"
    );
    assert_eq!(
        response.headers()["access-control-allow-origin"]
            .to_str()
            .unwrap(),
        "*",
        "CORS origin should allow all origins (test app is permissive)"
    );
}

#[tokio::test]
async fn test_latest_step_has_layers() {
    let server = TestServer::new(build_test_app()).unwrap();

    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("r1", 100))
        .await;

    let response = server.get("/api/v1/runs/r1/latest").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert!(body["layers"].is_array());
    assert!(!body["layers"].as_array().unwrap().is_empty());
}
