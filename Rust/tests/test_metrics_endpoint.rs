mod common;

use axum::http::StatusCode;
use axum_test::TestServer;
use common::{build_test_app, valid_payload, valid_payload_with};

#[tokio::test]
async fn test_valid_metrics_accepted() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload())
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);
    let body: serde_json::Value = response.json();
    assert_eq!(body["status"], "accepted");
    assert_eq!(body["run_id"], "test_run");
}

#[tokio::test]
async fn test_invalid_json() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server
        .post("/api/v1/metrics/layerwise")
        .text("{invalid json")
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().expect("detail should be array");
    assert!(detail[0]["msg"].as_str().unwrap().contains("Invalid JSON"));
    assert_eq!(detail[0]["type"], "json_invalid");
}

#[tokio::test]
async fn test_missing_field() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload
        .as_object_mut()
        .unwrap()
        .get_mut("metadata")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove("batch_size");
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().expect("detail should be array");
    assert!(detail[0]["msg"].as_str().unwrap().contains("batch_size"));
}

#[tokio::test]
async fn test_nan_rejected() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["layer_statistics"][0]["intermediate_features"]["activation_std"] =
        serde_json::json!("NaN");
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_infinity_rejected() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["layer_statistics"][0]["gradient_flow"]["gradient_l2_norm"] =
        serde_json::json!("Infinity");
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_negative_rejected() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["metadata"]["batch_size"] = serde_json::json!(-1);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_empty_run_id() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["metadata"]["run_id"] = serde_json::json!("");
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().expect("detail should be array");
    assert!(detail[0]["msg"].as_str().unwrap().contains("run_id"));
}

#[tokio::test]
async fn test_empty_layer_statistics() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["layer_statistics"] = serde_json::json!([]);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().expect("detail should be array");
    assert!(detail[0]["msg"]
        .as_str()
        .unwrap()
        .contains("layer_statistics"));
}

#[tokio::test]
async fn test_unsorted_depth_index() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    let layer0 = payload["layer_statistics"][0].clone();
    let mut layer1 = layer0.clone();
    layer1["layer_id"] = serde_json::json!("layer2");
    payload["layer_statistics"][0]["depth_index"] = serde_json::json!(1);
    layer1["depth_index"] = serde_json::json!(0);
    payload["layer_statistics"]
        .as_array_mut()
        .unwrap()
        .push(layer1);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().expect("detail should be array");
    assert!(detail[0]["msg"].as_str().unwrap().contains("depth_index"));
}

#[tokio::test]
async fn test_duplicate_step_replaces() {
    let server = TestServer::new(build_test_app()).unwrap();

    let payload = valid_payload_with("r1", 100);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);

    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);

    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    assert_eq!(body["r1"]["step_count"], 1);
}

#[tokio::test]
async fn test_multiple_runs() {
    let server = TestServer::new(build_test_app()).unwrap();

    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run_a", 100))
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);

    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run_b", 200))
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);

    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    assert!(body.as_object().unwrap().contains_key("run_a"));
    assert!(body.as_object().unwrap().contains_key("run_b"));
}

#[tokio::test]
async fn test_optional_bias_null() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["layer_statistics"][0]["parameter_statistics"]["bias"] = serde_json::Value::Null;
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn test_short_activation_shape() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["layer_statistics"][0]["intermediate_features"]["activation_shape"] =
        serde_json::json!([32]);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().expect("detail should be array");
    assert!(detail[0]["msg"]
        .as_str()
        .unwrap()
        .contains("activation_shape"));
}

#[tokio::test]
async fn test_valid_with_layer_groups() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["metadata"]["layer_groups"] = serde_json::json!({"encoder": ["layer1"]});
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);
    let body: serde_json::Value = response.json();
    assert_eq!(body["status"], "accepted");
}

#[tokio::test]
async fn test_zero_values_accepted() {
    let server = TestServer::new(build_test_app()).unwrap();
    let mut payload = valid_payload();
    payload["layer_statistics"][0]["intermediate_features"]["activation_std"] =
        serde_json::json!(0.0);
    payload["layer_statistics"][0]["gradient_flow"]["gradient_l2_norm"] = serde_json::json!(0.0);
    let response = server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);
}
