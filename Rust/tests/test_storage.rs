mod common;

use axum::http::StatusCode;
use axum_test::TestServer;
use common::{build_test_app, valid_payload, valid_payload_with};
use nn_monitor_server::models::MetricsPayload;
use nn_monitor_server::store::MetricsStore;
use std::time::Duration;

// ==================== Dedup Tests ====================

#[tokio::test]
async fn test_step_dedup_replaces_data() {
    let server = TestServer::new(build_test_app()).unwrap();

    // POST step 100 with batch_size=32
    let first = valid_payload_with("run1", 100);
    server
        .post("/api/v1/metrics/layerwise")
        .json(&first)
        .await;

    // POST step 100 again with batch_size=64 (different value)
    let mut second = valid_payload_with("run1", 100);
    second["metadata"]["batch_size"] = serde_json::json!(64);
    server
        .post("/api/v1/metrics/layerwise")
        .json(&second)
        .await;

    let response = server.get("/api/v1/runs/run1").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    let steps = body["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 1, "step count should be 1 after dedup");
    assert_eq!(steps[0]["step"], 100);
    assert_eq!(
        steps[0]["batch_size"], 64,
        "stored data should be from the second POST"
    );
}

#[tokio::test]
async fn test_step_dedup_no_count_increment() {
    let server = TestServer::new(build_test_app()).unwrap();

    for _ in 0..3 {
        server
            .post("/api/v1/metrics/layerwise")
            .json(&valid_payload_with("run1", 100))
            .await;
    }

    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    assert_eq!(
        body["run1"]["step_count"], 1,
        "step_count should be 1 even after 3 identical submissions"
    );
}

#[tokio::test]
async fn test_step_dedup_preserves_ordering() {
    let server = TestServer::new(build_test_app()).unwrap();

    for &step in &[300u64, 100, 200] {
        server
            .post("/api/v1/metrics/layerwise")
            .json(&valid_payload_with("run1", step))
            .await;
    }

    let response = server.get("/api/v1/runs/run1").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    let steps = body["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0]["step"], 100, "steps should be sorted: [100, 200, 300]");
    assert_eq!(steps[1]["step"], 200);
    assert_eq!(steps[2]["step"], 300);
}

#[tokio::test]
async fn test_step_dedup_in_middle() {
    let server = TestServer::new(build_test_app()).unwrap();

    // POST steps [100, 200, 300]
    for &step in &[100u64, 200, 300] {
        server
            .post("/api/v1/metrics/layerwise")
            .json(&valid_payload_with("run1", step))
            .await;
    }

    // Re-POST step 200 (dedup the middle element)
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run1", 200))
        .await;

    let response = server.get("/api/v1/runs/run1").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    let steps = body["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 3, "dedup should not change step count");
    assert_eq!(steps[0]["step"], 100, "order should remain [100, 200, 300]");
    assert_eq!(steps[1]["step"], 200);
    assert_eq!(steps[2]["step"], 300);
}

#[tokio::test]
async fn test_multiple_runs_dedup_independent() {
    let server = TestServer::new(build_test_app()).unwrap();

    // POST step 100 for both run_a and run_b
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run_a", 100))
        .await;
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run_b", 100))
        .await;

    // Dedup run_a step 100
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run_a", 100))
        .await;

    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    assert_eq!(
        body["run_a"]["step_count"], 1,
        "run_a should still have 1 step"
    );
    assert_eq!(
        body["run_b"]["step_count"], 1,
        "run_b should be unaffected by run_a dedup"
    );
}

// ==================== Eviction Tests ====================

#[tokio::test]
async fn test_run_eviction_oldest_by_last_update() {
    let store = MetricsStore::new(3, 1000);

    // Add 3 runs with distinct timestamps
    for i in 0..3u64 {
        let json = valid_payload_with(&format!("run{}", i), i);
        let payload: MetricsPayload = serde_json::from_value(json.clone()).unwrap();
        store.add_metrics(payload, json).await;
        tokio::time::sleep(Duration::from_millis(15)).await;
    }

    // Update run0 so it becomes the newest by last_update
    let json = valid_payload_with("run0", 999);
    let payload: MetricsPayload = serde_json::from_value(json.clone()).unwrap();
    store.add_metrics(payload, json).await;

    // Add run3 — should evict run1 (oldest last_update), NOT run0 (just updated)
    let json = valid_payload_with("run3", 0);
    let payload: MetricsPayload = serde_json::from_value(json.clone()).unwrap();
    store.add_metrics(payload, json).await;

    assert!(
        store.get_run("run0").await.is_some(),
        "run0 (recently updated) should NOT be evicted"
    );
    assert!(
        store.get_run("run1").await.is_none(),
        "run1 (oldest last_update) should be evicted"
    );
    assert!(store.get_run("run2").await.is_some(), "run2 should exist");
    assert!(store.get_run("run3").await.is_some(), "run3 should exist");
}

#[tokio::test]
async fn test_run_eviction_after_max() {
    let server = TestServer::new(build_test_app()).unwrap();

    // POST to 11 different runs (store max is 10)
    for i in 0u64..11 {
        server
            .post("/api/v1/metrics/layerwise")
            .json(&valid_payload_with(&format!("run{}", i), 100))
            .await;
    }

    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    let runs = body.as_object().unwrap();
    assert_eq!(runs.len(), 10, "only 10 runs should exist after eviction");
    // The oldest run (run0) should have been evicted
    assert!(
        !runs.contains_key("run0"),
        "oldest run (run0) should be evicted"
    );
    for i in 1u64..11 {
        assert!(
            runs.contains_key(&format!("run{}", i)),
            "run{} should exist",
            i
        );
    }
}

#[tokio::test]
async fn test_step_eviction_after_max() {
    let server = TestServer::new(build_test_app()).unwrap();

    // POST 1001 steps for same run (store max is 1000)
    for step in 1u64..=1001 {
        server
            .post("/api/v1/metrics/layerwise")
            .json(&valid_payload_with("run1", step))
            .await;
    }

    let response = server.get("/api/v1/runs").await;
    let body: serde_json::Value = response.json();
    let step_count = body["run1"]["step_count"].as_u64().unwrap();
    assert!(
        step_count <= 1000,
        "step_count should be <= 1000, got {}",
        step_count
    );
}

#[tokio::test]
async fn test_step_eviction_keeps_latest() {
    let server = TestServer::new(build_test_app()).unwrap();

    // POST 1001 steps for same run
    for step in 1u64..=1001 {
        server
            .post("/api/v1/metrics/layerwise")
            .json(&valid_payload_with("run1", step))
            .await;
    }

    let response = server.get("/api/v1/runs/run1").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    let steps = body["steps"].as_array().unwrap();
    assert!(
        steps.len() <= 1000,
        "should have at most 1000 steps, got {}",
        steps.len()
    );
    assert_eq!(
        steps[0]["step"].as_u64().unwrap(),
        2,
        "step 1 should be evicted, step 2 should be first"
    );
    assert_eq!(
        steps[steps.len() - 1]["step"].as_u64().unwrap(),
        1001,
        "step 1001 should be the last step"
    );
}

// ==================== Sanitization Tests ====================

#[tokio::test]
async fn test_layer_id_sanitization_dots_to_slashes() {
    let server = TestServer::new(build_test_app()).unwrap();

    let mut payload = valid_payload();
    payload["layer_statistics"][0]["layer_id"] = serde_json::json!("encoder.linear1");

    server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;

    let response = server.get("/api/v1/runs/test_run/latest").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    let layer_id = body["layers"][0]["layer_id"]
        .as_str()
        .expect("layer_id should be a string");
    assert_eq!(
        layer_id, "encoder/linear1",
        "dots in layer_id should become slashes"
    );
}

#[tokio::test]
async fn test_layer_groups_values_sanitized() {
    let server = TestServer::new(build_test_app()).unwrap();

    let mut payload = valid_payload();
    payload["metadata"]["layer_groups"] = serde_json::json!({
        "encoder": ["encoder.linear1", "encoder.relu.1"]
    });

    server
        .post("/api/v1/metrics/layerwise")
        .json(&payload)
        .await;

    let response = server.get("/api/v1/runs/test_run/latest").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    let groups = body["layer_groups"].as_object().expect("layer_groups should be an object");

    // Keys should be unchanged
    assert!(
        groups.contains_key("encoder"),
        "layer_groups keys should be unchanged"
    );

    // Values should be sanitized (dots → slashes)
    let enc_values = groups["encoder"]
        .as_array()
        .expect("encoder values should be an array");
    assert_eq!(
        enc_values[0].as_str().unwrap(),
        "encoder/linear1",
        "layer_groups values should have dots converted to slashes"
    );
    assert_eq!(enc_values[1].as_str().unwrap(), "encoder/relu/1");
}

// ==================== Timestamp Tests ====================

#[tokio::test]
async fn test_duplicate_step_updates_last_update() {
    let server = TestServer::new(build_test_app()).unwrap();

    // First POST
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run1", 100))
        .await;

    let response = server.get("/api/v1/runs/run1").await;
    let body: serde_json::Value = response.json();
    let first_update = body["last_update"]
        .as_str()
        .unwrap()
        .to_string();

    tokio::time::sleep(Duration::from_millis(15)).await;

    // Same step again (dedup)
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run1", 100))
        .await;

    let response = server.get("/api/v1/runs/run1").await;
    let body: serde_json::Value = response.json();
    let second_update = body["last_update"]
        .as_str()
        .unwrap()
        .to_string();

    assert_ne!(
        first_update, second_update,
        "last_update must change even for duplicate step submission"
    );
}

#[tokio::test]
async fn test_created_at_preserved_on_dedup() {
    let server = TestServer::new(build_test_app()).unwrap();

    // First POST
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run1", 100))
        .await;

    let response = server.get("/api/v1/runs/run1").await;
    let body: serde_json::Value = response.json();
    let created_at = body["created_at"]
        .as_str()
        .unwrap()
        .to_string();

    // Same step again (dedup)
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run1", 100))
        .await;

    let response = server.get("/api/v1/runs/run1").await;
    let body: serde_json::Value = response.json();
    let created_at_after = body["created_at"]
        .as_str()
        .unwrap()
        .to_string();

    assert_eq!(
        created_at, created_at_after,
        "created_at should not change on dedup"
    );
}

#[tokio::test]
async fn test_latest_step_after_dedup() {
    let server = TestServer::new(build_test_app()).unwrap();

    // POST step 100
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run1", 100))
        .await;

    // POST step 100 again (dedup)
    server
        .post("/api/v1/metrics/layerwise")
        .json(&valid_payload_with("run1", 100))
        .await;

    let response = server.get("/api/v1/runs/run1/latest").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert_eq!(
        body["step"], 100,
        "latest_step should still be 100 after dedup"
    );
}

// ==================== Edge Cases ====================

#[tokio::test]
async fn test_empty_store_get_all_runs() {
    let server = TestServer::new(build_test_app()).unwrap();

    let response = server.get("/api/v1/runs").await;
    assert_eq!(response.status_code(), StatusCode::OK);
    let body: serde_json::Value = response.json();
    assert!(
        body.as_object().unwrap().is_empty(),
        "empty store should return empty JSON object"
    );
}
