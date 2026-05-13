mod common;

use axum::http::StatusCode;
use axum_test::TestServer;
use common::{build_test_app, valid_payload};

#[tokio::test]
async fn test_422_is_list_format() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server
        .post("/api/v1/metrics/layerwise")
        .text("{invalid")
        .await;
    assert_eq!(response.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: serde_json::Value = response.json();
    assert!(
        body["detail"].is_array(),
        "422 detail should be an array"
    );
}

#[tokio::test]
async fn test_422_has_loc_field() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server
        .post("/api/v1/metrics/layerwise")
        .text("{invalid")
        .await;
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().unwrap();
    assert!(detail[0].get("loc").is_some(), "error should have loc field");
    assert_eq!(detail[0]["loc"], serde_json::json!(["body"]));
}

#[tokio::test]
async fn test_422_has_msg_field() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server
        .post("/api/v1/metrics/layerwise")
        .text("{invalid")
        .await;
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().unwrap();
    assert!(detail[0].get("msg").is_some(), "error should have msg field");
    assert!(detail[0]["msg"].is_string());
}

#[tokio::test]
async fn test_422_has_type_field() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server
        .post("/api/v1/metrics/layerwise")
        .text("{invalid")
        .await;
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_array().unwrap();
    assert!(
        detail[0].get("type").is_some(),
        "error should have type field"
    );
    assert_eq!(detail[0]["type"], "json_invalid");
}

#[tokio::test]
async fn test_404_is_dict_format() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/api/v1/runs/nonexistent").await;
    assert_eq!(response.status_code(), StatusCode::NOT_FOUND);
    let body: serde_json::Value = response.json();
    assert!(
        body["detail"].is_object(),
        "404 detail should be an object"
    );
}

#[tokio::test]
async fn test_404_has_error_key() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/api/v1/runs/nonexistent").await;
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_object().unwrap();
    assert!(
        detail.contains_key("error"),
        "404 detail should have error key"
    );
}

#[tokio::test]
async fn test_404_has_message_key() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/api/v1/runs/nonexistent").await;
    let body: serde_json::Value = response.json();
    let detail = body["detail"].as_object().unwrap();
    assert!(
        detail.contains_key("message"),
        "404 detail should have message key"
    );
}

#[tokio::test]
async fn test_404_message_includes_run_id() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/api/v1/runs/my_special_run").await;
    let body: serde_json::Value = response.json();
    let msg = body["detail"]["message"].as_str().unwrap();
    assert!(
        msg.contains("my_special_run"),
        "message should contain the run_id, got: {}",
        msg
    );
}

#[tokio::test]
async fn test_422_content_type_json() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server
        .post("/api/v1/metrics/layerwise")
        .text("{invalid")
        .await;
    let ct = response
        .headers()
        .get("content-type")
        .expect("should have content-type header")
        .to_str()
        .unwrap();
    assert!(
        ct.contains("application/json"),
        "content-type should be application/json, got: {}",
        ct
    );
}

#[tokio::test]
async fn test_404_content_type_json() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/api/v1/runs/nonexistent").await;
    let ct = response
        .headers()
        .get("content-type")
        .expect("should have content-type header")
        .to_str()
        .unwrap();
    assert!(
        ct.contains("application/json"),
        "content-type should be application/json, got: {}",
        ct
    );
}

#[tokio::test]
async fn test_404_error_value_not_found() {
    let server = TestServer::new(build_test_app()).unwrap();
    let response = server.get("/api/v1/runs/nonexistent").await;
    let body: serde_json::Value = response.json();
    assert_eq!(body["detail"]["error"], "not_found");
}

#[tokio::test]
async fn test_422_missing_required_field() {
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
    assert_eq!(detail[0]["loc"], serde_json::json!(["body"]));
    let msg = detail[0]["msg"].as_str().unwrap();
    assert!(
        msg.contains("batch_size"),
        "error message should mention batch_size, got: {}",
        msg
    );
    assert_eq!(detail[0]["type"], "value_error");
}
